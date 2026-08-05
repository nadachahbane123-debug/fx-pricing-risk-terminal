"""Rejeu historique point-in-time du skew Ho & Stoll (Section 16).

Règle d'or : pour reproduire fidèlement ce que le pricer aurait affiché le
jour `d`, chaque quantité (sigma, gamma, position) est recalculée en
n'utilisant QUE les données disponibles a la date `d` (pas de biais
d'anticipation / look-ahead bias).

Limite assumée et documentée dans l'UI : seul un historique agrégé du MID de
marche quotidien est disponible (pas les cotations bid/ask completes de
chaque banque a chaque date passee). Le spread de marche courant est donc
utilisé comme proxy constant pour reconstituer les prix historiques.
"""
from __future__ import annotations

import pandas as pd

from core.models import (
    CalibrationParams,
    calibrer_gamma,
    ho_stoll,
    position_a_date,
    prix_base,
    volatilite,
)


def point_in_time_sigma(price_history: pd.DataFrame, as_of_date: str) -> float:
    """sigma_d : ecart-type (normalise par sqrt(jours ecoules)) des MID <= as_of_date (fenetre glissante)."""
    subset = price_history[price_history["date"] <= as_of_date].sort_values("date")
    return volatilite(subset)


def point_in_time_pricing(
    price_history: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of_date: str,
    calib: CalibrationParams,
    spread_marche_proxy: float,
) -> dict | None:
    """Calcule sigma_d, gamma_d, position_d, skew_d et les prix finaux a la date `d`.

    `as_of_date` n'a pas besoin d'exister exactement dans `price_history` : le
    dernier MID connu a cette date (ou avant) est utilise, ce qui permet de
    valoriser des transactions survenues entre deux points d'historique.
    """
    subset = price_history[price_history["date"] <= as_of_date].sort_values("date")
    if subset.empty:
        return None
    mid_marche_d = float(subset["mid_marche"].iloc[-1])

    sigma_d = volatilite(subset)
    gamma_d = calibrer_gamma(sigma_d, calib)
    position_d = position_a_date(transactions, as_of_date)

    bid_base_d, ask_base_d = prix_base(mid_marche_d, spread_marche_proxy)
    prix_final_d = ho_stoll(gamma_d, sigma_d, position_d, mid_marche_d, spread_marche_proxy)

    return {
        "date": as_of_date,
        "mid_marche_observe": mid_marche_d,
        "sigma": sigma_d,
        "gamma": gamma_d,
        "position": position_d,
        "bid_base": bid_base_d,
        "ask_base": ask_base_d,
        "skew": prix_final_d.skew,
        "prix_reservation": prix_final_d.prix_reservation,
        "ask_final": prix_final_d.ask_final,
        "bid_final": prix_final_d.bid_final,
    }


def run_backtest(
    price_history: pd.DataFrame,
    transactions: pd.DataFrame,
    calib: CalibrationParams,
    spread_marche_proxy: float,
) -> pd.DataFrame:
    """Rejoue le skew Ho-Stoll jour par jour, a partir du 3e jour d'historique disponible."""
    history = price_history.sort_values("date").reset_index(drop=True)
    if len(history) < 3:
        return pd.DataFrame()

    rows = []
    for d in history["date"].iloc[2:]:
        r = point_in_time_pricing(history, transactions, d, calib, spread_marche_proxy)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)
