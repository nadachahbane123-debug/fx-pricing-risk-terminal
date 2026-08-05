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


def attribuer_pnl(transactions: pd.DataFrame, history: pd.DataFrame,
                  mid_actuel: float, gamma: float, sigma: float) -> AttributionPnL:
    """Décompose le résultat en trois sources distinctes.

    **Marge de spread** : ce que la banque encaisse mécaniquement en achetant
    sous le mid et en vendant au-dessus. C'est le revenu structurel du métier
    de teneur de marché, indépendant de toute vue directionnelle.

    **Effet du skew** : le décalage appliqué au prix selon la position
    détenue. Contrairement à la marge de spread, cet effet peut être négatif :
    coter moins cher pour attirer un flux compensateur à un coût immédiat. La
    question est de savoir si ce coût est compensé par la réduction du risque.

    **Effet de marche** : la réévaluation de la position ouverte suite au
    mouvement du cours. C'est la composante que le teneur de marché ne
    controle pas, et qu'il cherche précisément à minimiser.
    """
    if transactions.empty:
        return AttributionPnL(0.0, 0.0, 0.0, 0.0, 0)

    tx = transactions.copy()
    tx["date"] = tx["date"].astype(str)

    hist = history.copy()
    hist["date"] = hist["date"].astype(str)
    mid_par_date = dict(zip(hist["date"], hist["mid_marche"]))

    marge_spread = 0.0
    effet_skew = 0.0
    position_courante = 0.0

    for _, r in tx.iterrows():
        montant = float(r["montant"])
        sens = str(r["sens"])
        mid_jour = mid_par_date.get(r["date"], mid_actuel)

        # Skew qui s'appliquait compte tenu de la position AVANT ce deal
        k = gamma * (sigma ** 2)
        skew = -k * (position_courante / 1_000_000.0)

        prix_exec = r.get("prix_execution")
        if prix_exec is None or (isinstance(prix_exec, float) and math.isnan(prix_exec)):
            # Prix non renseigne : on reconstitue le prix qui aurait ete cote
            prix_exec = mid_jour + skew

        # Marge de spread : ecart entre le prix traite et le prix de reservation
        prix_reservation = mid_jour + skew
        if sens == "Vend":       # le client vend, CFG achete sous le mid
            marge_unitaire = prix_reservation - float(prix_exec)
            position_courante += montant
        else:                    # le client achete, CFG vend au-dessus du mid
            marge_unitaire = float(prix_exec) - prix_reservation
            position_courante -= montant

        marge_spread += marge_unitaire * montant
        effet_skew += skew * montant * (1 if sens == "Vend" else -1)

    # Effet de marche : reevaluation de la position finale ouverte
    mid_initial = hist["mid_marche"].iloc[0] if not hist.empty else mid_actuel
    effet_marche = position_courante * (mid_actuel - mid_initial)

    total = marge_spread + effet_skew + effet_marche
    return AttributionPnL(
        marge_spread=marge_spread,
        effet_skew=effet_skew,
        effet_marche=effet_marche,
        total=total,
        n_transactions=len(tx),
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
