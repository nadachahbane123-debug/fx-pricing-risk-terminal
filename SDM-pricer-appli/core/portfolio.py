"""Risque de portefeuille multi-devises : corrélation, VaR agrégée, couverture croisée.

Ajouter l'EUR/MAD en le déduisant du USD/MAD par un simple calcul croisé
n'apporte rien : c'est une multiplication, pas une modélisation. Ce qui a du
sens, en revanche, c'est de traiter les deux paires comme un **portefeuille**,
dont le risque n'est pas la somme des risques individuels.

Cette question prend une résonance particulière au Maroc. Le dirham étant
arrimé à un panier composé à 60 % d'euros et 40 % de dollars, l'EUR/MAD et le
USD/MAD ne sont pas deux paires indépendantes : elles sont liées **par
construction réglementaire**. Une position longue USD compensée par une
position courte EUR ne porte pas le meme risque que deux positions longues,
car le panier absorbe une partie du mouvement commun.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

Z_SCORES = {0.95: 1.6449, 0.99: 2.3263}

# Poids officiels du panier de reference du dirham (Bank Al-Maghrib)
POIDS_PANIER = {"EUR": 0.60, "USD": 0.40}


@dataclass
class MatriceRisque:
    devises: list[str]
    sigmas: list[float]                  # ecarts-types quotidiens, en MAD
    correlation: list[list[float]]       # matrice de correlation
    covariance: list[list[float]]        # matrice de covariance


@dataclass
class VaRPortefeuille:
    var_95: float
    var_99: float
    ecart_type: float
    horizon_jours: int
    var_non_diversifiee: float   # somme des VaR individuelles
    benefice_diversification: float
    contributions: dict          # contribution de chaque devise au risque total


def correlation_series(serie_a: pd.Series, serie_b: pd.Series) -> float:
    """Corrélation des variations quotidiennes de deux séries de cours.

    On corrèle les **variations**, pas les niveaux : deux cours peuvent
    évoluer autour de moyennes très différentes tout en bougeant de facon
    parfaitement synchrone d'un jour à l'autre. C'est ce co-mouvement
    quotidien qui détermine le risque du portefeuille.
    """
    a = serie_a.diff().dropna()
    b = serie_b.diff().dropna()
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a.iloc[-n:].reset_index(drop=True), b.iloc[-n:].reset_index(drop=True)
    if a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
        return 0.0
    return float(a.corr(b))


def construire_matrice(devises: list[str], sigmas: list[float],
                       correlations: dict[tuple[str, str], float]) -> MatriceRisque:
    """Assemble la matrice de covariance à partir des volatilités et corrélations.

        Cov(i,j) = rho(i,j) x sigma_i x sigma_j
    """
    n = len(devises)
    corr = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cle = (devises[i], devises[j])
            cle_inv = (devises[j], devises[i])
            corr[i][j] = correlations.get(cle, correlations.get(cle_inv, 0.0))

    cov = [[corr[i][j] * sigmas[i] * sigmas[j] for j in range(n)] for i in range(n)]
    return MatriceRisque(devises=devises, sigmas=sigmas, correlation=corr, covariance=cov)


def var_portefeuille(positions: list[float], matrice: MatriceRisque,
                     horizon_jours: int = 1) -> VaRPortefeuille:
    """VaR agrégée du portefeuille de devises.

        variance_portefeuille = w^T . Sigma . w
        VaR(c) = z(c) x racine(variance) x racine(horizon)

    ou w est le vecteur des positions (en unites de devise étrangère) et Sigma
    la matrice de covariance des variations de cours.

    Le point interessant est la comparaison avec la **VaR non diversifiée**
    (simple somme des VaR individuelles) : l'écart entre les deux mesure le
    bénéfice de diversification, c'est-à-dire ce que la corrélation imparfaite
    entre les deux paires fait économiser en capital de risque.
    """
    n = len(positions)
    variance = 0.0
    for i in range(n):
        for j in range(n):
            variance += positions[i] * positions[j] * matrice.covariance[i][j]

    variance = max(variance, 0.0)
    ecart_type = math.sqrt(variance) * math.sqrt(max(horizon_jours, 0))

    var95 = Z_SCORES[0.95] * ecart_type
    var99 = Z_SCORES[0.99] * ecart_type

    # VaR non diversifiée : on ignore les correlations
    var_indiv = [
        Z_SCORES[0.99] * abs(positions[i]) * matrice.sigmas[i] * math.sqrt(max(horizon_jours, 0))
        for i in range(n)
    ]
    var_non_div = sum(var_indiv)

    # Contribution marginale de chaque devise au risque total
    contributions = {}
    if ecart_type > 0:
        for i in range(n):
            cov_i_ptf = sum(positions[j] * matrice.covariance[i][j] for j in range(n))
            contributions[matrice.devises[i]] = (positions[i] * cov_i_ptf) / variance if variance > 0 else 0.0

    return VaRPortefeuille(
        var_95=var95,
        var_99=var99,
        ecart_type=ecart_type,
        horizon_jours=horizon_jours,
        var_non_diversifiee=var_non_div,
        benefice_diversification=var_non_div - var99,
        contributions=contributions,
    )


# --------------------------------------------------------------------------- #
# Couverture croisee
# --------------------------------------------------------------------------- #

@dataclass
class CouvertureCroisee:
    position_a_couvrir: float
    devise_couverture: str
    ratio_optimal: float
    montant_couverture: float
    variance_avant: float
    variance_apres: float
    reduction_risque_pct: float


def couverture_croisee(position_usd: float, sigma_usd: float, sigma_eur: float,
                       rho: float, devise_couverture: str = "EUR") -> CouvertureCroisee:
    """Ratio de couverture minimisant la variance du portefeuille.

    Si la banque porte une position en USD et souhaite en réduire le risque
    sans la déboucler, elle peut prendre une position de sens oppose sur une
    devise corrélée. Le ratio qui minimise la variance résiduelle est :

        h* = - rho x (sigma_usd / sigma_eur)

    Démonstration : la variance du portefeuille couvert vaut
        V(h) = P^2 sigma_u^2 + h^2 sigma_e^2 + 2 P h rho sigma_u sigma_e
    En annulant dV/dh, on obtient h* = -P x rho x sigma_u / sigma_e.

    Interet pratique au Maroc : lorsque les deux paires sont fortement
    correlees (ce qui est structurellement le cas via le panier du dirham),
    une couverture partielle en EUR/MAD peut reduire sensiblement le risque
    d'une position USD/MAD sans avoir a la liquider — donc sans subir le cout
    d'impact de marche modelise par Almgren & Chriss.
    """
    if sigma_eur <= 0:
        return CouvertureCroisee(position_usd, devise_couverture, 0.0, 0.0, 0.0, 0.0, 0.0)

    ratio = -rho * (sigma_usd / sigma_eur)
    montant = position_usd * ratio

    var_avant = (position_usd ** 2) * (sigma_usd ** 2)
    var_apres = (
        (position_usd ** 2) * (sigma_usd ** 2)
        + (montant ** 2) * (sigma_eur ** 2)
        + 2 * position_usd * montant * rho * sigma_usd * sigma_eur
    )
    var_apres = max(var_apres, 0.0)

    reduction = (1.0 - math.sqrt(var_apres) / math.sqrt(var_avant)) * 100 if var_avant > 0 else 0.0

    return CouvertureCroisee(
        position_a_couvrir=position_usd,
        devise_couverture=devise_couverture,
        ratio_optimal=ratio,
        montant_couverture=montant,
        variance_avant=var_avant,
        variance_apres=var_apres,
        reduction_risque_pct=reduction,
    )


def exposition_panier(position_usd: float, position_eur: float) -> dict:
    """Exposition de la position au panier de reference du dirham.

    Le cours central du dirham est calculé à partir d'un panier 60 % EUR /
    40 % USD. Une position dont la répartition reproduit ces poids est, en
    théorie, moins exposée aux mouvements du panier lui-meme : elle bouge
    avec la référence plutot que contre elle.

    Cet indicateur mesure l'écart entre la répartition effective de la
    position et celle du panier officiel.
    """
    total = abs(position_usd) + abs(position_eur)
    if total == 0:
        return {"poids_usd": 0.0, "poids_eur": 0.0, "ecart_panier": 0.0, "commentaire": "Position plate"}

    poids_usd = abs(position_usd) / total
    poids_eur = abs(position_eur) / total
    ecart = abs(poids_usd - POIDS_PANIER["USD"]) + abs(poids_eur - POIDS_PANIER["EUR"])

    if ecart < 0.10:
        commentaire = "Répartition proche du panier de référence : exposition au panier limitée"
    elif poids_usd > POIDS_PANIER["USD"]:
        commentaire = "Surponderation USD par rapport au panier de référence"
    else:
        commentaire = "Surponderation EUR par rapport au panier de référence"

    return {
        "poids_usd": poids_usd,
        "poids_eur": poids_eur,
        "poids_panier_usd": POIDS_PANIER["USD"],
        "poids_panier_eur": POIDS_PANIER["EUR"],
        "ecart_panier": ecart,
        "commentaire": commentaire,
    }
