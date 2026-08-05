"""Modélisation de l'intensité des flux clients et détermination du spread optimal.

Jusqu'ici, le pricer **reprenait** le spread observé sur le marché. Il ne le
**décidait** pas. Or c'est précisément la décision centrale du market making :

    Plus j'élargis mon spread, plus je gagne par transaction,
    mais moins de clients acceptent mon prix.

Ce module modélise cet arbitrage, l'optimise, et permet de calibrer le modèle
sur les demandes de cotation réellement observées.

Le modèle d'intensité exponentielle retenu est celui qui sous-tend toute la
littérature moderne du market making (Avellaneda-Stoikov 2008 ; Guéant,
Lehalle & Fernandez-Tapia 2013) :

    lambda(delta) = A x exp(-k x delta)

où `delta` est l'écart de prix proposé par rapport au mid, et `lambda`
l'intensité d'arrivée des transactions (nombre de deals par unité de temps).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ParametresIntensite:
    """Paramètres du modèle lambda(delta) = A x exp(-k x delta).

    A : intensité maximale, atteinte lorsque le prix propose est exactement au
        mid (delta = 0). C'est le flux qu'on capterait en renoncant à toute
        marge — la taille du gisement potentiel de clientèle.

    k : sensibilité du client au prix. Un k élevé décrit une clientèle très
        sensible, qui compare systématiquement les cotations et se détourne
        dès le moindre écart. Un k faible décrit une clientèle captive.
    """
    A: float
    k: float
    n_observations: int = 0
    r2: float = 0.0  # qualite de l'ajustement, si calibre sur donnees


def intensite(delta: float, params: ParametresIntensite) -> float:
    """Intensité d'arrivée des transactions pour un écart de prix donné."""
    return params.A * math.exp(-params.k * delta)


def pnl_espere(delta: float, params: ParametresIntensite) -> float:
    """Espérance de gain par unité de temps : marge unitaire x volume attendu.

        E[P&L] = delta x lambda(delta) = delta x A x exp(-k x delta)

    Cette fonction est le cœur du problème : elle croît d'abord (on gagne plus
    par deal) puis décroît (on perd trop de deals). Son maximum définit le
    spread optimal.
    """
    return delta * intensite(delta, params)


def spread_optimal(params: ParametresIntensite) -> float:
    """Écart de prix maximisant l'espérance de gain.

    Dérivation :
        d/d(delta) [ delta x A x exp(-k delta) ]
            = A x exp(-k delta) x (1 - k x delta) = 0
        => delta* = 1 / k

    Résultat remarquable : le spread optimal **ne dépend pas de A**. Le volume
    global de clientèle change le montant gagné, mais pas le prix auquel il
    faut coter. Seule compte la sensibilité au prix.
    """
    if params.k <= 0:
        return 0.0
    return 1.0 / params.k


def spread_optimal_avec_inventaire(params: ParametresIntensite, gamma: float,
                                   sigma: float, position: float) -> tuple[float, float]:
    """Écarts optimaux à l'achat et à la vente, tenant compte de la position.

    Le spread optimal pur (1/k) suppose un teneur de marché indifférent à son
    inventaire. En pratique, une position déjà longue doit conduire à coter
    plus agressivement à la vente qu'à l'achat, pour attirer le flux qui
    rééquilibrera le stock.

    On combine donc ici les deux mécanismes déja présents dans le pricer :
    le spread optimal issu de l'intensité des flux, et le skew de position du
    modèle de Ho & Stoll.

    Renvoie (delta_bid, delta_ask), les écarts à appliquer de part et d'autre
    du mid de marché.
    """
    base = spread_optimal(params)
    skew = gamma * (sigma ** 2) * (position / 1_000_000.0)

    delta_bid = max(base + skew, 0.0)   # position longue -> on paie moins cher a l'achat
    delta_ask = max(base - skew, 0.0)   # position longue -> on vend moins cher
    return delta_bid, delta_ask


# --------------------------------------------------------------------------- #
# Calibration sur les demandes de cotation observees
# --------------------------------------------------------------------------- #

@dataclass
class PointCalibration:
    delta: float          # ecart propose (en MAD)
    n_demandes: int       # nombre de RFQ a ce niveau
    n_acceptees: int      # nombre de deals conclus

    @property
    def taux_acceptation(self) -> float:
        return self.n_acceptees / self.n_demandes if self.n_demandes else 0.0


def calibrer_intensite(points: list[PointCalibration]) -> ParametresIntensite | None:
    """Estime A et k par régression log-linéaire sur les taux d'acceptation.

    Méthode : le modèle étant lambda(delta) = A x exp(-k delta), le passage au
    logarithme le rend linéaire :

        ln(lambda) = ln(A) - k x delta

    Une régression des moindres carrés de ln(taux d'acceptation) sur delta
    fournit donc directement -k comme pente et ln(A) comme ordonnée à
    l'origine. C'est l'estimation la plus simple et la plus transparente ; des
    approches par maximum de vraisemblance existent mais exigent un volume de
    données dont on ne dispose pas ici.

    Renvoie None si les données sont insuffisantes (moins de deux niveaux de
    spread exploitables) — il vaut mieux ne rien afficher qu'un paramètre
    estime sur un seul point.
    """
    valides = [p for p in points if p.n_demandes > 0 and p.n_acceptees > 0]
    if len(valides) < 2:
        return None

    xs = [p.delta for p in valides]
    ys = [math.log(p.taux_acceptation) for p in valides]
    n = len(xs)

    moy_x = sum(xs) / n
    moy_y = sum(ys) / n
    cov = sum((x - moy_x) * (y - moy_y) for x, y in zip(xs, ys))
    var_x = sum((x - moy_x) ** 2 for x in xs)

    if var_x == 0:
        return None

    pente = cov / var_x
    ordonnee = moy_y - pente * moy_x

    # Coefficient de determination
    ss_tot = sum((y - moy_y) ** 2 for y in ys)
    ss_res = sum((y - (ordonnee + pente * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return ParametresIntensite(
        A=math.exp(ordonnee),
        k=max(-pente, 1e-9),   # k doit etre positif : une pente positive serait aberrante
        n_observations=sum(p.n_demandes for p in valides),
        r2=r2,
    )


def courbe_pnl(params: ParametresIntensite, delta_max: float, n_points: int = 120) -> tuple[list[float], list[float], list[float]]:
    """Génère les courbes d'intensité et de P&L espéré, pour visualisation.

    Renvoie (deltas, intensites, pnl_esperes).
    """
    deltas = [delta_max * i / (n_points - 1) for i in range(n_points)]
    intensites = [intensite(d, params) for d in deltas]
    pnls = [pnl_espere(d, params) for d in deltas]
    return deltas, intensites, pnls


def comparer_au_marche(params: ParametresIntensite, spread_marche_absolu: float) -> dict:
    """Compare le spread optimal issu du modele au demi-spread observé sur le marché.

    C'est la question que pose ce module : la banque cote-t-elle trop cher
    (elle perd du flux inutilement) ou trop bon marché (elle laisse de la
    marge sur la table) ?
    """
    demi_spread_marche = spread_marche_absolu / 2.0
    delta_opt = spread_optimal(params)

    if delta_opt <= 0:
        return {"diagnostic": "Modele non calibre", "ecart_pips": 0.0}

    ecart = delta_opt - demi_spread_marche
    if abs(ecart) < 0.00002:      # moins de 0,2 pip d'ecart
        diagnostic = "Cotation alignée sur l'optimum du modèle"
    elif ecart > 0:
        diagnostic = "Le modèle suggère un spread plus large : marge laissée sur la table"
    else:
        diagnostic = "Le modèle suggère un spread plus serre : flux perdu inutilement"

    return {
        "delta_optimal": delta_opt,
        "demi_spread_marche": demi_spread_marche,
        "ecart_pips": ecart * 10_000.0,
        "pnl_optimal": pnl_espere(delta_opt, params),
        "pnl_au_spread_marche": pnl_espere(demi_spread_marche, params),
        "diagnostic": diagnostic,
    }


# =========================================================================== #
# Regression logistique sur les demandes de cotation individuelles
# =========================================================================== #
#
# La calibration exponentielle ci-dessus travaille sur des taux d'acceptation
# **agreges** par niveau de spread : elle resume plusieurs dizaines de RFQ en
# un seul point. Une regression logistique exploite au contraire chaque
# demande individuellement, et admet plusieurs variables explicatives.
#
#     P(acceptation) = 1 / (1 + exp(-(b0 + b1 delta + b2 ln(montant) + b3 sens)))
#
# Trois avantages sur le modele exponentiel :
#   - chaque observation contribue, au lieu de moyennes par palier ;
#   - la taille de l'operation et son sens entrent dans l'explication ;
#   - la probabilite est **bornee dans [0,1] par construction**, ce que
#     A x exp(-k delta) ne garantit pas : cette derniere depasse 1 pour de
#     petits ecarts, ce qui n'a pas de sens comme probabilite.
#
# Contrepartie a assumer : le modele exponentiel fournit la solution
# analytique delta* = 1/k. La logistique impose une optimisation numerique.
# C'est un arbitrage entre tractabilite theorique et rigueur statistique.

import numpy as np


@dataclass
class ModeleLogistique:
    coefficients: np.ndarray          # [constante, delta, ln(montant), sens_achete]
    noms: list[str]
    n_observations: int
    log_vraisemblance: float
    pseudo_r2: float                  # R2 de McFadden
    taux_bien_classes: float
    convergence: bool


def _sigmoide(z: np.ndarray) -> np.ndarray:
    # Formulation stable : evite l'overflow de exp() pour les grands |z|
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _matrice_design(deltas: np.ndarray, montants: np.ndarray, sens: np.ndarray) -> np.ndarray:
    """Construit la matrice des variables explicatives.

    Le montant entre en logarithme : l'effet d'un passage de 100 k a 200 k USD
    n'est pas du meme ordre que celui d'un passage de 5 M a 5,1 M. Une échelle
    logarithmique traduit cette décroissance de sensibilité.
    """
    return np.column_stack([
        np.ones(len(deltas)),
        deltas * 10_000.0,                        # en pips, pour un conditionnement correct
        np.log(np.maximum(montants, 1.0)),
        sens.astype(float),
    ])


def calibrer_logistique(rfq_log, ridge: float = 1e-4) -> ModeleLogistique | None:
    """Estimation par maximum de vraisemblance (algorithme de Newton-Raphson / IRLS).

    A chaque itération, le problème est réécrit comme une régression linéaire
    pondérée, les poids valant p(1-p) : les observations dont la probabilité
    prédite est proche de 0,5 -- donc les plus informatives -- pèsent le plus.

    Une pénalisation de crete (ridge) très faible est ajoutée pour éviter la
    divergence en cas de separation parfaite, situation ou les coefficients
    tendraient vers l'infini.
    """
    if rfq_log is None or len(rfq_log) < 30:
        return None
    if not {"delta_propose", "montant", "sens", "acceptee"}.issubset(rfq_log.columns):
        return None

    deltas = rfq_log["delta_propose"].to_numpy(dtype=float)
    montants = rfq_log["montant"].to_numpy(dtype=float)
    sens = (rfq_log["sens"].astype(str) == "Achete").to_numpy()
    y = rfq_log["acceptee"].to_numpy(dtype=float)

    if len(np.unique(y)) < 2:
        return None

    X = _matrice_design(deltas, montants, sens)
    n, k = X.shape
    beta = np.zeros(k)
    converge = False

    for _ in range(100):
        p = _sigmoide(X @ beta)
        W = np.clip(p * (1 - p), 1e-9, None)
        gradient = X.T @ (y - p) - ridge * beta
        H = -(X.T * W) @ X - ridge * np.eye(k)
        try:
            pas = np.linalg.solve(H, gradient)
        except np.linalg.LinAlgError:
            break
        beta_new = beta - pas
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            converge = True
            break
        beta = beta_new

    p = _sigmoide(X @ beta)
    eps = 1e-12
    ll = float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    p0 = float(np.mean(y))
    ll0 = float(n * (p0 * math.log(p0 + eps) + (1 - p0) * math.log(1 - p0 + eps)))
    pseudo_r2 = 1.0 - ll / ll0 if ll0 != 0 else 0.0
    bien_classes = float(np.mean((p >= 0.5) == (y >= 0.5)))

    return ModeleLogistique(
        coefficients=beta,
        noms=["Constante", "Ecart (pips)", "ln(Montant)", "Sens = Achete"],
        n_observations=n,
        log_vraisemblance=ll,
        pseudo_r2=pseudo_r2,
        taux_bien_classes=bien_classes,
        convergence=converge,
    )


def proba_acceptation(modele: ModeleLogistique, delta: float, montant: float,
                      sens_achete: bool) -> float:
    """Probabilite qu'un client accepte la cotation proposee."""
    x = np.array([1.0, delta * 10_000.0, math.log(max(montant, 1.0)), 1.0 if sens_achete else 0.0])
    return float(_sigmoide(np.array([float(x @ modele.coefficients)]))[0])


def spread_optimal_logistique(modele: ModeleLogistique, montant: float, sens_achete: bool,
                              delta_max: float = 0.012, n_points: int = 600) -> dict:
    """Ecart maximisant l'espérance de gain sous le modèle logistique.

    On maximise numeriquement delta x P(acceptation | delta, montant, sens).
    Contrairement au modèle exponentiel, l'optimum dépend ici du **montant**
    et du **sens** de l'opération : chaque profil de transaction admet son
    propre spread optimal, ce qui fournit une grille de cotation differenciée
    fondée sur les données plutot que sur des seuils choisis.
    """
    grille = np.linspace(1e-6, delta_max, n_points)
    probas = np.array([proba_acceptation(modele, d, montant, sens_achete) for d in grille])
    esperances = grille * probas
    i = int(np.argmax(esperances))
    return {
        "delta_optimal": float(grille[i]),
        "delta_optimal_pips": float(grille[i] * 10_000),
        "proba_a_optimum": float(probas[i]),
        "esperance_max": float(esperances[i]),
        "grille": grille,
        "probas": probas,
        "esperances": esperances,
    }


def grille_optimale_par_taille(modele: ModeleLogistique, montants: list[float],
                               sens_achete: bool = True) -> "pd.DataFrame":
    """Spread optimal par tranche de montant, dérivé du modèle estimé.

    Cette grille remplace avantageusement les multiplicateurs fixes du module
    RFQ : ceux-ci avaient été posés à dire d'expert, tandis que ceux-ci sont
    déduits du comportement observé de la clientèle.
    """
    import pandas as pd
    lignes = []
    ref = None
    for m in montants:
        opt = spread_optimal_logistique(modele, m, sens_achete)
        if ref is None:
            ref = opt["delta_optimal"]
        lignes.append({
            "MONTANT": f"{m:,.0f} USD".replace(",", " "),
            "SPREAD OPTIMAL": f"{opt['delta_optimal_pips']:.2f} pips",
            "MULTIPLICATEUR": f"x{opt['delta_optimal'] / ref:.2f}" if ref else "-",
            "PROBA ACCEPTATION": f"{opt['proba_a_optimum'] * 100:.1f} %",
        })
    return pd.DataFrame(lignes)
