"""Calcul du Profit & Loss (Section 18).

P&L_total = Cash_cumule (flux de tresorerie des transactions) +
            valorisation mark-to-market de la position ouverte.
"""
from __future__ import annotations

import pandas as pd

from core.backtest import point_in_time_pricing
from core.models import CalibrationParams, impact_signe, position_a_date


def resolve_execution_prices(
    transactions: pd.DataFrame,
    price_history: pd.DataFrame,
    calib: CalibrationParams,
    spread_marche_proxy: float,
) -> pd.DataFrame:
    """Complete `prix_execution` manquant par BID_final/ASK_final point-in-time (Section 18).

    BID_final si le client vend a CFG (sens='Vend'), ASK_final si le client
    achete a CFG (sens='Achete').

    Optimisation : le pricing point-in-time ne depend que de la DATE de la
    transaction, pas de la transaction elle-meme. Comme des centaines de
    transactions partagent la meme journee, on ne calcule le pricing qu'une seule
    fois par date distincte (memoisation), au lieu d'une fois par transaction.
    Sur un historique de plusieurs milliers d'operations, cela fait passer le
    temps de calcul de plusieurs dizaines de secondes a une fraction de seconde.
    """
    df = transactions.copy()
    df["prix_execution_effectif"] = df["prix_execution"]

    mid_defaut = float(price_history["mid_marche"].iloc[0])

    # Le champ date des transactions inclut l'heure : on memoise par JOUR (les prix
    # d'historique sont quotidiens, donc le pricing point-in-time est identique pour
    # toutes les transactions d'une meme journee). On passe ainsi de ~2500 calculs a
    # ~500, ce qui est l'essentiel du gain de performance.
    df["_jour"] = df["date"].str.slice(0, 10)
    cache: dict = {}
    for jour in df["_jour"].unique():
        cache[jour] = point_in_time_pricing(price_history, transactions, jour, calib, spread_marche_proxy)

    def _mid_moment(jour):
        pit = cache.get(jour)
        return pit["mid_marche_observe"] if pit else mid_defaut

    def _prix_effectif(row):
        if not pd.isna(row["prix_execution"]):
            return row["prix_execution"]
        pit = cache.get(row["_jour"])
        if pit is None:
            return _mid_moment(row["_jour"])
        return pit["bid_final"] if row["sens"] == "Vend" else pit["ask_final"]

    df["mid_marche_moment"] = df["_jour"].map(_mid_moment)
    df["prix_execution_effectif"] = df.apply(_prix_effectif, axis=1)
    df = df.drop(columns=["_jour"])

    # Flux de tresorerie : client vend -> CFG paie (sortie de cash) ; client achete -> CFG encaisse
    sens_vend = df["sens"] == "Vend"
    df["flux_tresorerie"] = df["montant"] * df["prix_execution_effectif"]
    df.loc[sens_vend, "flux_tresorerie"] *= -1

    # Ecart de spread capture par rapport au MID du moment (vectorise)
    ecart = df["prix_execution_effectif"] - df["mid_marche_moment"]
    df["ecart_spread_capture"] = ecart.where(~sens_vend, -ecart)
    df["gain_spread_mad"] = df["ecart_spread_capture"] * df["montant"]

    return df


def compute_pnl_total(
    transactions_priced: pd.DataFrame, position_actuelle: float, mid_marche_actuel: float
) -> tuple[float, float]:
    """P&L_total = Cash_cumule + Position_actuelle * MID_marche_actuel."""
    cash_cumule = float(transactions_priced["flux_tresorerie"].sum())
    pnl_total = cash_cumule + position_actuelle * mid_marche_actuel
    return pnl_total, cash_cumule


def compute_daily_pnl(
    transactions_priced: pd.DataFrame,
    price_history: pd.DataFrame,
    transactions_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Série de P&L quotidien : pour chaque date d'historique, valorisation point-in-time
    (transactions dont la date <= d, position au MID_marche de ce jour-la).

    Optimisation : plutot que de refiltrer toutes les transactions a chaque jour
    (O(jours x transactions)), on precalcule une seule fois le cumul du cash et de
    la position par date, puis on aligne ces cumuls sur l'historique par recherche
    ordonnee (merge_asof). Le calcul devient lineaire.
    """
    from core.models import impact_signe

    history = price_history.sort_values("date").reset_index(drop=True)
    if history.empty:
        return pd.DataFrame()

    # Cumul du cash par date (a partir des transactions valorisees)
    cash = (transactions_priced[["date", "flux_tresorerie"]]
            .groupby("date", as_index=False)["flux_tresorerie"].sum()
            .sort_values("date"))
    cash["cash_cumule"] = cash["flux_tresorerie"].cumsum()

    # Cumul de la position par date (a partir des transactions brutes)
    raw = transactions_raw.copy()
    raw["signe_montant"] = raw["sens"].map({"Vend": 1.0, "Achete": -1.0}).fillna(0.0) * raw["montant"]
    pos = (raw[["date", "signe_montant"]]
           .groupby("date", as_index=False)["signe_montant"].sum()
           .sort_values("date"))
    pos["position"] = pos["signe_montant"].cumsum()

    # Alignement sur l'historique par recherche ordonnee (derniere valeur <= date).
    # merge_asof exige des cles numeriques ou temporelles : on convertit les dates
    # (chaines ISO) en datetime pour le rapprochement, puis on repasse aux colonnes finales.
    # L'historique est quotidien (minuit) tandis que les transactions sont horodatees
    # dans la journee : on cale donc chaque jour d'historique sur sa FIN (23:59:59)
    # pour capturer toutes les operations de la journee, y compris le debouclage de 17:30.
    h = history[["date", "mid_marche"]].copy()
    h["_dt"] = pd.to_datetime(h["date"]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    cash["_dt"] = pd.to_datetime(cash["date"])
    pos["_dt"] = pd.to_datetime(pos["date"])
    h = h.sort_values("_dt")
    cash = cash.sort_values("_dt")
    pos = pos.sort_values("_dt")

    h = pd.merge_asof(h, cash[["_dt", "cash_cumule"]], on="_dt")
    h = pd.merge_asof(h, pos[["_dt", "position"]], on="_dt")
    h["cash_cumule"] = h["cash_cumule"].fillna(0.0)
    h["position"] = h["position"].fillna(0.0)

    h["pnl_total"] = h["cash_cumule"] + h["position"] * h["mid_marche"]
    h["pnl_du_jour"] = h["pnl_total"].diff().fillna(h["pnl_total"])
    return h[["date", "mid_marche", "position", "cash_cumule", "pnl_total", "pnl_du_jour"]]
