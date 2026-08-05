"""Logique métier pure du pricer USD/MAD CFG Bank.

Aucune dépendance à Streamlit ici : ce module est testable isolement.
Chaque fonction référence la section du cahier des charges qu'elle implémente.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Coefficients de la regression MID theorique (Section 1)
COEF_A = 2.9621
COEF_B = -11.963
COEF_C = 19.416

BANDE_PCT = 0.05  # +/-5% (Section 2 et 12)


# --------------------------------------------------------------------------- #
# Section 1-2 : MID theorique et bande reglementaire
# --------------------------------------------------------------------------- #

def mid_theorique(eur_usd: float) -> float:
    """MID_theorique = a*(EUR/USD)^2 + b*(EUR/USD) + c"""
    return COEF_A * eur_usd**2 + COEF_B * eur_usd + COEF_C


def bande_fluctuation(mid_theo: float) -> tuple[float, float]:
    """Borne_basse = MID_theo * 0.95 ; Borne_haute = MID_theo * 1.05"""
    return mid_theo * (1 - BANDE_PCT), mid_theo * (1 + BANDE_PCT)


# --------------------------------------------------------------------------- #
# Section 3 : analyse du marche reel (cotations concurrentes)
# --------------------------------------------------------------------------- #

def mid_spread_banque(bid: float, ask: float) -> tuple[float, float]:
    """MID_banque = (bid+ask)/2 ; Spread_banque = (ask-bid)/MID_banque"""
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid
    return mid, spread


def marche_stats(quotes: pd.DataFrame) -> tuple[float, float]:
    """MID_marche et Spread_marche = moyennes des MID/Spread de chaque banque.

    `quotes` doit contenir les colonnes 'bid' et 'ask'.
    """
    mids, spreads = [], []
    for _, row in quotes.iterrows():
        m, s = mid_spread_banque(row["bid"], row["ask"])
        mids.append(m)
        spreads.append(s)
    return float(np.mean(mids)), float(np.mean(spreads))


def ecart_theorique_marche(mid_marche: float, mid_theo: float) -> float:
    """Ecart_theorique_marche = (MID_marche - MID_theo) / MID_theo"""
    return (mid_marche - mid_theo) / mid_theo


# --------------------------------------------------------------------------- #
# Section 4 : prix de base CFG (avant gestion de position)
# --------------------------------------------------------------------------- #

def prix_base(mid_marche: float, spread_marche: float) -> tuple[float, float]:
    """ASK_base = MID_marche*(1+Spread/2) ; BID_base = ASK_base - Spread*MID_marche"""
    ask_base = mid_marche * (1 + spread_marche / 2)
    bid_base = ask_base - spread_marche * mid_marche
    return bid_base, ask_base


# --------------------------------------------------------------------------- #
# Section 5 : position (journal de transactions)
# --------------------------------------------------------------------------- #

def impact_signe(sens: str, montant: float) -> float:
    """Vend (client vend a CFG) -> +montant ; Achete (client achete a CFG) -> -montant"""
    return montant if sens == "Vend" else -montant


def position_cumulee(transactions: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes 'impact' et 'position' (cumulee) au journal, trie par date."""
    df = transactions.sort_values(["date", "id"] if "id" in transactions.columns else "date").copy()
    df["impact"] = df.apply(lambda r: impact_signe(r["sens"], r["montant"]), axis=1)
    df["position"] = df["impact"].cumsum()
    return df


def position_a_date(transactions: pd.DataFrame, as_of_date: str) -> float:
    """Position cumulee de CFG en ne considerant que les transactions <= as_of_date."""
    if transactions.empty:
        return 0.0
    subset = transactions[transactions["date"] <= as_of_date]
    if subset.empty:
        return 0.0
    # Vectorise : client vend -> desk long (+), client achete -> desk short (-)
    signe = subset["sens"].map({"Vend": 1.0, "Achete": -1.0}).fillna(0.0)
    return float((signe * subset["montant"]).sum())


# --------------------------------------------------------------------------- #
# Section 6 : volatilite
# --------------------------------------------------------------------------- #

def volatilite(price_history: pd.DataFrame) -> float:
    """sigma = ecart-type des variations quotidiennes du MID de marché,
    normalisées par le nombre REEL de jours calendaires écoulés entre deux
    observations (et non "une ligne = un jour").

    Justification : sous l'hypothèse standard de marché aléatoire (mouvement
    brownien, la meme hypothese qui sous-tend Ho-Stoll et Almgren-Chriss), la
    VARIANCE d'une variation de prix est proportionnelle au temps écoulé :
        Var(delta_S sur n jours) = n * Var(delta_S sur 1 jour)
    donc l'ECART-TYPE (racine de la variance) croit en sqrt(n), pas en n.
    Pour ramener une variation observée sur n jours à une variation
    "equivalente 1 jour", on divise donc par sqrt(n) -- pas par n.

    `price_history` doit contenir les colonnes 'date' (str ou datetime) et
    'mid_marche', avec au moins 3 lignes pour produire un resultat non nul.
    """
    if len(price_history) < 3:
        return 0.0

    df = price_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    diffs = df["mid_marche"].diff()
    jours_ecoules = df["date"].diff().dt.days

    # Securite : si deux lignes partagent la meme date (jours_ecoules = 0),
    # on les ignore plutot que de diviser par zero.
    valid = jours_ecoules > 0
    variations_normalisees = (diffs[valid] / jours_ecoules[valid].pow(0.5)).dropna()

    if len(variations_normalisees) < 2:
        return 0.0
    return float(variations_normalisees.std(ddof=1))


# --------------------------------------------------------------------------- #
# Section 7 : calibration de gamma
# --------------------------------------------------------------------------- #

@dataclass
class CalibrationParams:
    position_limite: float = 5_000_000.0
    decalage_cible_pips: float = 10.0  # 10 pips = 0.0010 MAD

    @property
    def decalage_cible_mad(self) -> float:
        return self.decalage_cible_pips * 0.0001


def calibrer_gamma(sigma: float, params: CalibrationParams) -> float:
    """k_cible = decalage_cible / (position_limite/1M) ; gamma = k_cible / sigma^2

    Ne pas arrondir : valeur exacte utilisee en aval.
    """
    if sigma == 0:
        return 0.0
    k_cible = params.decalage_cible_mad / (params.position_limite / 1_000_000)
    return k_cible / sigma**2


# --------------------------------------------------------------------------- #
# Section 8 : modele de decalage de prix - Ho & Stoll (1981)
# --------------------------------------------------------------------------- #

@dataclass
class PrixFinal:
    k: float
    skew: float
    prix_reservation: float
    ask_final: float
    bid_final: float


def ho_stoll(gamma: float, sigma: float, position: float, mid_marche: float, spread_marche: float) -> PrixFinal:
    """k = gamma*sigma^2 ; Skew = -k*(Position/1M)
    Prix_reservation = MID_marche + Skew
    ASK_final = Prix_reservation + Spread*MID/2 ; BID_final = Prix_reservation - Spread*MID/2
    """
    k = gamma * sigma**2
    skew = -k * (position / 1_000_000)
    prix_reservation = mid_marche + skew
    demi_spread = (spread_marche * mid_marche) / 2
    return PrixFinal(
        k=k,
        skew=skew,
        prix_reservation=prix_reservation,
        ask_final=prix_reservation + demi_spread,
        bid_final=prix_reservation - demi_spread,
    )


# --------------------------------------------------------------------------- #
# Section 9-10 : calibration de eta et lambda
# --------------------------------------------------------------------------- #

def calibrer_eta(spread_marche: float, mid_marche: float) -> float:
    """eta = Spread_marche_absolu = Spread_marche * MID_marche (proxy simplifie)."""
    return spread_marche * mid_marche


def calibrer_lambda(gamma: float) -> float:
    """lambda = gamma (meme aversion au risque pour pricing et execution)."""
    return gamma


# --------------------------------------------------------------------------- #
# Section 11 : modele de liquidation - Almgren & Chriss (2000)
# --------------------------------------------------------------------------- #

@dataclass
class TrajectoireLiquidation:
    kappa: float
    demi_vie: float | None
    jours: list[int]
    position_restante: list[float]
    montant_vendu: list[float]


def kappa_ac(lambda_: float, sigma: float, eta: float) -> float:
    """kappa = sqrt(lambda * sigma^2 / eta)"""
    if eta <= 0 or lambda_ <= 0:
        return 0.0
    val = lambda_ * sigma**2 / eta
    return math.sqrt(val) if val > 0 else 0.0


def trajectoire_almgren_chriss(position_actuelle: float, kappa: float, horizon_t: int) -> TrajectoireLiquidation:
    """x_j = X * sinh(kappa*(T-j)) / sinh(kappa*T), montant_vendu_j = x_(j-1) - x_j.

    Gere le cas limite kappa*T proche de 0 (repli lineaire).
    """
    x0 = abs(position_actuelle)
    jours = list(range(0, horizon_t + 1))
    kt = kappa * horizon_t

    if abs(kt) < 1e-6:
        # Repli lineaire : liquidation uniforme sur l'horizon
        position_restante = [x0 * (1 - j / horizon_t) if horizon_t > 0 else 0.0 for j in jours]
    else:
        position_restante = [
            x0 * math.sinh(kappa * (horizon_t - j)) / math.sinh(kt) for j in jours
        ]

    montant_vendu = [0.0] + [
        position_restante[j - 1] - position_restante[j] for j in range(1, len(jours))
    ]

    demi_vie = (math.log(2) / kappa) if kappa > 0 else None

    return TrajectoireLiquidation(
        kappa=kappa,
        demi_vie=demi_vie,
        jours=jours,
        position_restante=position_restante,
        montant_vendu=montant_vendu,
    )


# --------------------------------------------------------------------------- #
# Section 12 : controles de conformite
# --------------------------------------------------------------------------- #

def conforme_ecart(ecart_pct: float) -> bool:
    return abs(ecart_pct) <= BANDE_PCT


def conforme_bande(prix: float, borne_basse: float, borne_haute: float) -> bool:
    return borne_basse <= prix <= borne_haute


# --------------------------------------------------------------------------- #
# Pipeline complet : calcule tout l'etat du pricer a partir des entrees brutes
# --------------------------------------------------------------------------- #

@dataclass
class EtatPricer:
    eur_usd: float
    mid_theo: float
    borne_basse: float
    borne_haute: float
    mid_marche: float
    spread_marche: float
    ecart_theo_marche: float
    bid_base: float
    ask_base: float
    position: float
    sigma: float
    gamma: float
    eta: float
    lambda_: float
    prix_final: PrixFinal
    conforme_ecart: bool
    conforme_ask: bool
    conforme_bid: bool


def calculer_etat_pricer(
    eur_usd: float,
    quotes: pd.DataFrame,
    position: float,
    sigma: float,
    calib: CalibrationParams,
) -> EtatPricer:
    """Assemble l'ensemble des sections 1 a 12 en un seul etat coherent."""
    mt = mid_theorique(eur_usd)
    bb, bh = bande_fluctuation(mt)
    mid_marche, spread_marche = marche_stats(quotes)
    ecart = ecart_theorique_marche(mid_marche, mt)
    bid_base, ask_base = prix_base(mid_marche, spread_marche)
    gamma = calibrer_gamma(sigma, calib)
    eta = calibrer_eta(spread_marche, mid_marche)
    lambda_ = calibrer_lambda(gamma)
    prix_final = ho_stoll(gamma, sigma, position, mid_marche, spread_marche)

    return EtatPricer(
        eur_usd=eur_usd,
        mid_theo=mt,
        borne_basse=bb,
        borne_haute=bh,
        mid_marche=mid_marche,
        spread_marche=spread_marche,
        ecart_theo_marche=ecart,
        bid_base=bid_base,
        ask_base=ask_base,
        position=position,
        sigma=sigma,
        gamma=gamma,
        eta=eta,
        lambda_=lambda_,
        prix_final=prix_final,
        conforme_ecart=conforme_ecart(ecart),
        conforme_ask=conforme_bande(prix_final.ask_final, bb, bh),
        conforme_bid=conforme_bande(prix_final.bid_final, bb, bh),
    )
