"""Mesures de risque quotidiennes d'un desk de change : VaR et cours croisés.

Deux indicateurs qu'un teneur de marche consulte quotidiennement, au-delà du
seul suivi de sa position brute :

* la **VaR (Value at Risk)**, qui traduit la position en perte potentielle
  chiffrée en dirhams, sur un horizon et un niveau de confiance donnes ;
* les **cours croisés**, ici l'EUR/MAD, déduit du USD/MAD coté par la banque et
  du cours EUR/USD international — une paire tout aussi suivie que le USD/MAD
  sur un desk marocain, l'euro étant la première devise du panier du dirham.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Quantiles de la loi normale centree reduite, pour les niveaux de confiance usuels.
# Valeurs tabulees : evite une dependance a scipy pour deux constantes.
Z_SCORES = {0.95: 1.6449, 0.99: 2.3263}


@dataclass
class VaRResult:
    horizon_jours: int
    position_usd: float
    sigma_quotidien: float      # ecart-type des variations quotidiennes du MID (en MAD)
    var_95: float               # perte potentielle en MAD, niveau 95%
    var_99: float               # perte potentielle en MAD, niveau 99%
    ecart_type_pnl: float       # ecart-type du P&L sur l'horizon, en MAD


def var_parametrique(position_usd: float, sigma_quotidien: float, horizon_jours: int = 1) -> VaRResult:
    """VaR paramétrique (dite \"variance-covariance\").

    Hypothèse : les variations du cours suivent une loi normale, hypothèse
    standard de la VaR paramétrique — et cohérente avec celle déjà retenue par
    les modèles de Ho & Stoll et d'Almgren & Chriss utilises dans ce pricer.

        ecart_type_PnL = |position| x sigma_quotidien x racine(horizon)
        VaR(c)         = z(c) x ecart_type_PnL

    L'écart-type du P&L croit en racine du temps, et non linéairement : c'est la
    meme convention que celle appliquée au calcul de sigma sur un historique a
    pas irrégulier.

    Limite à garder en tete : cette VaR ne dit rien de l'ampleur des pertes
    au-dela du seuil (elle ignore les evenements extremes), et suppose une
    volatilite stable — deux hypotheses mises en defaut en période de crise.
    """
    ecart_type_pnl = abs(position_usd) * sigma_quotidien * math.sqrt(max(horizon_jours, 0))
    return VaRResult(
        horizon_jours=horizon_jours,
        position_usd=position_usd,
        sigma_quotidien=sigma_quotidien,
        var_95=Z_SCORES[0.95] * ecart_type_pnl,
        var_99=Z_SCORES[0.99] * ecart_type_pnl,
        ecart_type_pnl=ecart_type_pnl,
    )


def simuler_pnl(position_usd: float, sigma_quotidien: float, horizon_jours: int = 1,
                n: int = 4000, seed: int = 42) -> list[float]:
    """Tire n scenarios de P&L a l'horizon donné, pour visualiser la distribution.

    Sert uniquement a l'illustration graphique de la VaR : les seuils affichés
    restent ceux de la formule parametrique ci-dessus.
    """
    import random
    rng = random.Random(seed)
    ecart_type = abs(position_usd) * sigma_quotidien * math.sqrt(max(horizon_jours, 0))
    if ecart_type == 0:
        return [0.0] * n
    return [rng.gauss(0.0, ecart_type) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Cours croises
# --------------------------------------------------------------------------- #

@dataclass
class CroisEURMAD:
    eur_usd: float
    usd_mad_bid: float
    usd_mad_ask: float
    eur_mad_bid: float
    eur_mad_ask: float
    eur_mad_mid: float
    spread_pips: float


def cross_eur_mad(eur_usd: float, usd_mad_bid: float, usd_mad_ask: float) -> CroisEURMAD:
    """Déduit l'EUR/MAD du USD/MAD coté par la banque et du cours EUR/USD.

        EUR/MAD = EUR/USD x USD/MAD

    Le cours EUR/USD étant traite ici comme une donnée de marché unique (sans
    fourchette propre), l'écart bid/ask de l'EUR/MAD reflète uniquement celui
    du USD/MAD, transposé par le facteur EUR/USD.
    """
    bid = eur_usd * usd_mad_bid
    ask = eur_usd * usd_mad_ask
    mid = (bid + ask) / 2.0
    return CroisEURMAD(
        eur_usd=eur_usd,
        usd_mad_bid=usd_mad_bid,
        usd_mad_ask=usd_mad_ask,
        eur_mad_bid=bid,
        eur_mad_ask=ask,
        eur_mad_mid=mid,
        spread_pips=(ask - bid) * 10_000,
    )


def valeur_du_pip(montant_usd: float) -> float:
    """Valeur en MAD d'un pip (0,0001) pour un montant donne en USD.

    Repère de calcul mental permanent sur un desk : de combien varie la
    contrevaleur d'une opération lorsque le cours bouge d'un pip.
    """
    return abs(montant_usd) * 0.0001
