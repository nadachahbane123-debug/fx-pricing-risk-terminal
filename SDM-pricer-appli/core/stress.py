"""Tests de resistance : scenarios de choc (Section 19)."""
from __future__ import annotations

from dataclasses import dataclass

from core.models import (
    CalibrationParams,
    bande_fluctuation,
    calibrer_gamma,
    conforme_bande,
    ho_stoll,
    mid_theorique,
)

SCENARIOS_PAR_DEFAUT = [
    {"nom": "EUR/USD +2%", "choc_pct": 0.02, "vol_multiplier": None},
    {"nom": "EUR/USD -2%", "choc_pct": -0.02, "vol_multiplier": None},
    {"nom": "EUR/USD +5% (choc extreme)", "choc_pct": 0.05, "vol_multiplier": None},
    {"nom": "USD/MAD +2% (choc de marche propre)", "choc_pct": None, "choc_marche_pct": 0.02, "vol_multiplier": None},
    {"nom": "USD/MAD -2% (choc de marche propre)", "choc_pct": None, "choc_marche_pct": -0.02, "vol_multiplier": None},
    {"nom": "Volatilite x2 (regime de crise)", "choc_pct": None, "vol_multiplier": 2.0},
]


@dataclass
class ResultatScenario:
    nom: str
    eur_usd_choque: float
    mid_theo_choque: float
    borne_basse_choquee: float
    borne_haute_choquee: float
    mid_marche_choque: float
    sigma_choque: float
    gamma_choque: float
    skew_choque: float
    ask_final_choque: float
    bid_final_choque: float
    conforme_ask: bool
    conforme_bid: bool
    impact_pnl: float


def run_scenario(
    nom: str,
    eur_usd_actuel: float,
    mid_marche_actuel: float,
    spread_marche: float,
    sigma_actuel: float,
    position_actuelle: float,
    calib: CalibrationParams,
    choc_pct: float | None,
    vol_multiplier: float | None,
    choc_marche_pct: float | None = None,
) -> ResultatScenario:
    """Applique un choc EUR/USD, un choc de marche propre et/ou un choc de
    volatilite (impacte sigma -> gamma -> skew), puis recalcule le pricing complet.

    - Choc EUR/USD : transmis au USD/MAD par la fonction du MID theorique
      (a*x^2 + b*x + c). Le MID de marche suit la meme variation relative que
      le MID theorique, ce qui respecte le signe economique : un euro qui
      s'apprecie face au dollar fait BAISSER le USD/MAD.
    - Choc de marche propre : variation relative directe du MID de marche
      (liquidite, flux), sans passer par l'EUR/USD ; le MID theorique est inchange.
    """
    eur_usd_c = eur_usd_actuel * (1 + choc_pct) if choc_pct is not None else eur_usd_actuel
    mid_theo_c = mid_theorique(eur_usd_c)
    borne_basse_c, borne_haute_c = bande_fluctuation(mid_theo_c)

    mid_theo_0 = mid_theorique(eur_usd_actuel)
    mid_marche_c = mid_marche_actuel * (mid_theo_c / mid_theo_0)  # transmission par le MID theorique
    if choc_marche_pct is not None:
        mid_marche_c *= (1 + choc_marche_pct)

    sigma_c = sigma_actuel * vol_multiplier if vol_multiplier is not None else sigma_actuel
    gamma_c = calibrer_gamma(sigma_c, calib)

    prix_final_c = ho_stoll(gamma_c, sigma_c, position_actuelle, mid_marche_c, spread_marche)

    impact_pnl = position_actuelle * (mid_marche_c - mid_marche_actuel)

    return ResultatScenario(
        nom=nom,
        eur_usd_choque=eur_usd_c,
        mid_theo_choque=mid_theo_c,
        borne_basse_choquee=borne_basse_c,
        borne_haute_choquee=borne_haute_c,
        mid_marche_choque=mid_marche_c,
        sigma_choque=sigma_c,
        gamma_choque=gamma_c,
        skew_choque=prix_final_c.skew,
        ask_final_choque=prix_final_c.ask_final,
        bid_final_choque=prix_final_c.bid_final,
        conforme_ask=conforme_bande(prix_final_c.ask_final, borne_basse_c, borne_haute_c),
        conforme_bid=conforme_bande(prix_final_c.bid_final, borne_basse_c, borne_haute_c),
        impact_pnl=impact_pnl,
    )


def run_all_scenarios(
    scenarios: list[dict],
    eur_usd_actuel: float,
    mid_marche_actuel: float,
    spread_marche: float,
    sigma_actuel: float,
    position_actuelle: float,
    calib: CalibrationParams,
) -> list[ResultatScenario]:
    return [
        run_scenario(
            s["nom"],
            eur_usd_actuel,
            mid_marche_actuel,
            spread_marche,
            sigma_actuel,
            position_actuelle,
            calib,
            s.get("choc_pct"),
            s.get("vol_multiplier"),
            s.get("choc_marche_pct"),
        )
        for s in scenarios
    ]
