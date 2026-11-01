"""Attribution du P&L (version alignee sur le module PNL).

Identite exacte, verifiee transaction par transaction :

    P&L total = marge de spread + effet du skew + effet de marche

avec, pour une transaction de montant M et de sens q = +1 (le client vend,
la banque achete) ou q = -1 (le client achete, la banque vend) :

    marge de spread  = M x (demi-ecart de base au MID)
    effet du skew    = M x (ecart entre le prix de base et le prix execute)
    effet de marche  = q x M x (MID actuel - MID du jour de la transaction)

La somme des deux premiers termes est exactement le gain de spread du module
PNL (ecart entre le prix execute et le MID du moment) ; le troisieme est la
reevaluation de la position au cours actuel. Le total est donc celui de PNL<GO>.
"""
from __future__ import annotations

import math

import pandas as pd

from core.attribution import AttributionPnL
from core.backtest import point_in_time_pricing
from core.models import CalibrationParams


def attribuer_pnl(transactions: pd.DataFrame, history: pd.DataFrame, mid_actuel: float,
                  calib: CalibrationParams, spread_marche: float) -> AttributionPnL:
    if transactions.empty:
        return AttributionPnL(0.0, 0.0, 0.0, 0.0, 0)

    marge = skew_eff = marche = 0.0
    for _, r in transactions.iterrows():
        pit = point_in_time_pricing(history, transactions, str(r["date"]), calib, spread_marche)
        if pit is None:
            continue
        montant = float(r["montant"])
        vend = str(r["sens"]) == "Vend"          # le client vend : la banque achete au BID
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

    return AttributionPnL(marge, skew_eff, marche, marge + skew_eff + marche, len(transactions))
