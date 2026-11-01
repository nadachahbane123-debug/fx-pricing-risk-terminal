"""Attribution du P&L et métriques quantifiées de performance du modèle.

Deux questions auxquelles le pricer ne répondait pas encore :

1. **D'où vient le résultat ?** Un P&L global ne dit pas si la banque a gagné
   parce qu'elle a bien cote, bien géré son inventaire, ou simplement parce
   que le marche est allé dans son sens. C'est la premiere chose qu'un
   responsable de desk cherche à savoir.

2. **Le modèle apporte-t-il quelque chose ?** Le backtest comparait
   visuellement les courbes avec et sans skew. Un graphique ne suffit pas :
   il faut des chiffres comparables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from core.backtest import point_in_time_pricing
from core.models import CalibrationParams


# --------------------------------------------------------------------------- #
# Attribution du P&L
# --------------------------------------------------------------------------- #

@dataclass
class AttributionPnL:
    marge_spread: float        # marge captee sur l'ecart bid/ask
    effet_skew: float          # effet du decalage de prix lie a l'inventaire
    effet_marche: float        # variation de valeur de la position ouverte
    total: float
    n_transactions: int


def attribuer_pnl(transactions: pd.DataFrame, history: pd.DataFrame, mid_actuel: float,
                  calib: CalibrationParams, spread_marche: float) -> AttributionPnL:
    """Decompose le resultat en trois sources, par une identite comptable exacte.

        P&L total = marge de spread + effet du skew + effet de marche

    Pour une transaction de montant M et de sens q = +1 (le client vend, la
    banque achete au BID) ou q = -1 (le client achete, la banque vend a l'ASK) :

        marge de spread = M x (ecart du prix de base au MID du jour)
        effet du skew   = M x (ecart entre le prix de base et le prix execute)
        effet de marche = q x M x (MID actuel - MID du jour de la transaction)

    Les deux premiers termes ont pour somme le gain de spread du module PNL
    (ecart entre le prix execute et le MID du moment) ; le troisieme reevalue
    la position au cours actuel. Le total est donc celui de PNL<GO>.

    Le prix d'execution manquant est reconstitue point-in-time (BID_final ou
    ASK_final du jour), comme dans le module PNL.
    """
    if transactions.empty:
        return AttributionPnL(0.0, 0.0, 0.0, 0.0, 0)

    # Le pricing point-in-time ne depend que du JOUR : on le calcule une fois par
    # date distincte (memoisation), comme dans le module PNL.
    cache: dict = {}

    marge = skew_eff = marche = 0.0
    for _, r in transactions.iterrows():
        jour = str(r["date"])[:10]
        if jour not in cache:
            cache[jour] = point_in_time_pricing(history, transactions, jour, calib, spread_marche)
        pit = cache[jour]
        if pit is None:
            continue
        montant = float(r["montant"])
        vend = str(r["sens"]) == "Vend"
        mid = pit["mid_marche_observe"]

        px = r.get("prix_execution")
        if px is None or (isinstance(px, float) and math.isnan(px)):
            px = pit["bid_final"] if vend else pit["ask_final"]
        px = float(px)

        if vend:
            marge += montant * (mid - pit["bid_base"])
            skew_eff += montant * (pit["bid_base"] - px)
            marche += montant * (mid_actuel - mid)
        else:
            marge += montant * (pit["ask_base"] - mid)
            skew_eff += montant * (px - pit["ask_base"])
            marche -= montant * (mid_actuel - mid)

    return AttributionPnL(
        marge_spread=marge,
        effet_skew=skew_eff,
        effet_marche=marche,
        total=marge + skew_eff + marche,
        n_transactions=len(transactions),
    )


# --------------------------------------------------------------------------- #
# Metriques quantifiees du backtest
# --------------------------------------------------------------------------- #

@dataclass
class MetriquesBacktest:
    n_jours: int
    pnl_marche_total: float        # P&L de la position portee sur la periode
    pnl_moyen_quotidien: float
    ecart_type_pnl: float
    sharpe_annualise: float
    pire_journee: float
    meilleure_journee: float
    position_moyenne: float
    position_max: float
    skew_moyen_pips: float         # decalage moyen consenti
    skew_max_pips: float
    cout_cumule_skew_mad: float    # marge sacrifiee sur la periode
    jours_position_longue: int
    jours_position_courte: int


def metriques_backtest(bt: pd.DataFrame) -> MetriquesBacktest | None:
    """Chiffre le comportement du modèle sur la période rejouée.

    Le backtest visuel montrait deux courbes. Ces metriques repondent aux
    questions qu'un responsable de desk poserait : combien la position a-t-elle
    rapporte ou coute, avec quelle regularite, et quel prix le skew a-t-il fait
    payer en marge.

    Le ratio de Sharpe est annualisé sur 252 jours ouvres, convention standard.

    **Limite methodologique importante, à énoncer clairement** : ce backtest
    suppose que le flux client aurait été strictement identique avec et sans
    skew. Or c'est exactement l'inverse que le skew cherche à provoquer — un
    prix décalé doit attirer un flux différent, donc modifier la trajectoire
    de la position. Les chiffres ci-dessous mesurent donc fidèlement le **cout**
    du skew (la marge sacrifiée, qui est observable), mais ne peuvent pas
    mesurer son **bénéfice** (le rééquilibrage induit, qui supposerait de
    modéliser la réaction des clients au prix — c'est precisement l'objet du
    module d'intensité des flux).
    """
    if bt is None or bt.empty or len(bt) < 3:
        return None

    df = bt.copy()
    mid_col = "mid_marche_observe" if "mid_marche_observe" in df.columns else "mid_marche"
    if mid_col not in df.columns or "position" not in df.columns:
        return None

    df["d_mid"] = df[mid_col].diff().fillna(0.0)
    # La position de la veille subit la variation de cours du jour
    df["pnl_jour"] = df["position"].shift(1).fillna(0.0) * df["d_mid"]

    pnl_total = float(df["pnl_jour"].sum())
    pnl_moyen = float(df["pnl_jour"].mean())
    sd = float(df["pnl_jour"].std(ddof=1)) if len(df) > 2 else 0.0
    sharpe = (pnl_moyen / sd * math.sqrt(252)) if sd > 0 else 0.0

    skew_abs = df["skew"].abs() if "skew" in df.columns else pd.Series([0.0] * len(df))
    # Cout du skew : ecart consenti x volume traite ce jour-la (variation de position)
    volume_jour = df["position"].diff().abs().fillna(0.0)
    cout_cumule = float((skew_abs * volume_jour).sum())

    return MetriquesBacktest(
        n_jours=len(df),
        pnl_marche_total=pnl_total,
        pnl_moyen_quotidien=pnl_moyen,
        ecart_type_pnl=sd,
        sharpe_annualise=float(sharpe),
        pire_journee=float(df["pnl_jour"].min()),
        meilleure_journee=float(df["pnl_jour"].max()),
        position_moyenne=float(df["position"].abs().mean()),
        position_max=float(df["position"].abs().max()),
        skew_moyen_pips=float(skew_abs.mean() * 10_000),
        skew_max_pips=float(skew_abs.max() * 10_000),
        cout_cumule_skew_mad=cout_cumule,
        jours_position_longue=int((df["position"] > 0).sum()),
        jours_position_courte=int((df["position"] < 0).sum()),
    )


# --------------------------------------------------------------------------- #
# Limites prudentielles de position de change
# --------------------------------------------------------------------------- #

@dataclass
class RatioPrudentiel:
    position_nette_mad: float
    position_longue_mad: float
    position_courte_mad: float
    fonds_propres_mad: float
    ratio_par_devise: float      # en % des fonds propres
    ratio_global: float          # en % des fonds propres
    limite_par_devise: float
    limite_globale: float
    conforme_devise: bool
    conforme_global: bool


def ratios_position_change(positions_mad: dict[str, float], fonds_propres_mad: float,
                           limite_par_devise: float = 10.0,
                           limite_globale: float = 20.0) -> RatioPrudentiel:
    """Ratios prudentiels de position de change, rapportés aux fonds propres.

    Au-delà de la bande de fluctuation de +/-5 %, une banque marocaine est
    soumise à des limites sur sa **position de change**, exprimées en
    pourcentage de ses fonds propres : une limite par devise, et une limite
    globale toutes devises confondues.

    Convention de calcul retenue : la position globale est mesurée comme le
    maximum entre la somme des positions longues et la valeur absolue de la
    somme des positions courtes — approche dite du "shorthand method", usuelle
    en réglementation prudentielle.

    Les seuils par defaut (10 % par devise, 20 % au global) correspondent aux
    ordres de grandeur usuels, mais doivent étre confirmés auprès de la
    Direction des Risques : ils relevent de la réglementation en vigueur et
    peuvent évoluer.
    """
    longues = sum(v for v in positions_mad.values() if v > 0)
    courtes = abs(sum(v for v in positions_mad.values() if v < 0))
    nette = sum(positions_mad.values())
    globale = max(longues, courtes)

    max_devise = max((abs(v) for v in positions_mad.values()), default=0.0)

    ratio_dev = (max_devise / fonds_propres_mad * 100) if fonds_propres_mad > 0 else 0.0
    ratio_glob = (globale / fonds_propres_mad * 100) if fonds_propres_mad > 0 else 0.0

    return RatioPrudentiel(
        position_nette_mad=nette,
        position_longue_mad=longues,
        position_courte_mad=courtes,
        fonds_propres_mad=fonds_propres_mad,
        ratio_par_devise=ratio_dev,
        ratio_global=ratio_glob,
        limite_par_devise=limite_par_devise,
        limite_globale=limite_globale,
        conforme_devise=ratio_dev <= limite_par_devise,
        conforme_global=ratio_glob <= limite_globale,
    )
