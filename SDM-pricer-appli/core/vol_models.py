"""Modeles de volatilite conditionnelle : GARCH, GJR-GARCH (TGARCH) et EGARCH.

Le pricer estime aujourd'hui sigma comme un ecart-type glissant : une
**constante** sur toute la periode. Or cette constante irrigue l'ensemble des
decisions du desk :

    k     = gamma x sigma^2          -> intensite du skew (Ho & Stoll)
    kappa = racine(lambda sigma^2/eta) -> vitesse de liquidation (Almgren & Chriss)
    VaR   = z x |position| x sigma x racine(h)

Un ecart-type constant suppose implicitement que le risque de porter une
position est identique un jour calme et un jour de crise. Les series
financieres contredisent frontalement cette hypothese : elles presentent un
regroupement des extremes (\"volatility clustering\"), une forte variation
succedant a une forte variation.

Les modeles de la famille GARCH rendent la variance **conditionnelle** a
l'information passee. En conservant gamma fixe (calibre sur un objectif
metier) et en rendant seul sigma conditionnel, le pricer acquiert une
propriete qu'il n'avait pas : le skew s'elargit automatiquement quand le
marche devient volatil, et la liquidation s'accelere.

L'estimation est realisee par maximum de vraisemblance, implementee ici
directement (scipy) plutot que via une bibliotheque dediee : la structure du
modele reste ainsi entierement explicite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Esperance de |z| pour une loi normale centree reduite : racine(2/pi).
# Intervient dans la specification EGARCH.
ESP_ABS_Z = math.sqrt(2.0 / math.pi)


@dataclass
class ResultatGARCH:
    modele: str
    params: dict
    log_vraisemblance: float
    aic: float
    bic: float
    n_obs: int
    sigma_conditionnel: np.ndarray   # serie des sigma_t estimes
    residus_standardises: np.ndarray
    persistance: float               # alpha + beta (proche de 1 = chocs durables)
    variance_long_terme: float
    convergence: bool
    demi_vie_choc: float             # en jours


# --------------------------------------------------------------------------- #
# Preparation des donnees
# --------------------------------------------------------------------------- #

def variations_normalisees(price_history: pd.DataFrame) -> np.ndarray:
    """Variations quotidiennes du MID, normalisees par racine(jours ecoules).

    Meme convention que la fonction `volatilite` du module models : sous
    hypothese de marche aleatoire, la variance croit proportionnellement au
    temps, donc l'ecart-type en racine du temps. Une variation observee sur
    trois jours (week-end) est donc divisee par racine(3) pour etre ramenee a
    un equivalent quotidien.

    On travaille sur les variations **absolues** (en MAD) et non sur des
    rendements logarithmiques, pour rester coherent avec la definition de
    sigma deja utilisee dans tout le pricer.
    """
    if price_history is None or len(price_history) < 3:
        return np.array([])

    df = price_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    diffs = df["mid_marche"].diff()
    jours = df["date"].diff().dt.days
    valides = jours > 0
    return (diffs[valides] / np.sqrt(jours[valides])).dropna().to_numpy(dtype=float)



# --------------------------------------------------------------------------- #
# Mise a l'echelle
# --------------------------------------------------------------------------- #
# Les variations quotidiennes du USD/MAD sont de l'ordre de 1e-4 MAD, donc les
# variances manipulees valoisinent 1e-8. A cette echelle, les gradients de la
# log-vraisemblance sont numeriquement negligeables et l'optimiseur reste bloque
# sur son point de depart. La pratique standard en estimation GARCH consiste a
# normaliser la serie avant l'ajustement, puis a retablir l'echelle sur les
# parametres estimes.
#
# Si z = c x eps, alors sigma_z = c x sigma_eps, d'ou :
#   - alpha, beta, gamma sont invariants (ce sont des rapports de variances) ;
#   - omega se retablit par omega = omega_z / c^2 (GARCH, GJR) ;
#   - pour l'EGARCH, specifie en log-variance : omega = omega_z - 2 ln(c)(1-beta) ;
#   - la log-vraisemblance se corrige de +n ln(c) (changement de variable).

def _facteur_echelle(eps: np.ndarray) -> float:
    ec = float(np.std(eps))
    return 1.0 / ec if ec > 0 else 1.0


# --------------------------------------------------------------------------- #
# GARCH(1,1)
# --------------------------------------------------------------------------- #

def _filtre_garch11(params: np.ndarray, eps: np.ndarray) -> np.ndarray:
    omega, alpha, beta = params
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = max(np.var(eps), 1e-14)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
            sigma2[t] = 1e-14
    return sigma2


def _neg_loglik_garch11(params: np.ndarray, eps: np.ndarray) -> float:
    sigma2 = _filtre_garch11(params, eps)
    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
        return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    return -ll if np.isfinite(ll) else 1e10


def ajuster_garch11(eps: np.ndarray) -> ResultatGARCH | None:
    """GARCH(1,1) de Bollerslev (1986), estime par maximum de vraisemblance.

        sigma^2_t = omega + alpha x eps^2_{t-1} + beta x sigma^2_{t-1}

    Interpretation des parametres :
      - alpha : reaction immediate a un choc (poids de la surprise de la veille)
      - beta  : persistance de la volatilite (memoire du regime precedent)
      - alpha + beta : persistance globale. Une valeur proche de 1 signale des
        chocs a effet durable ; une valeur superieure ou egale a 1 rendrait le
        processus non stationnaire, d'ou la contrainte imposee a l'optimisation.

    La variance de long terme vaut omega / (1 - alpha - beta) : c'est le niveau
    vers lequel la volatilite conditionnelle revient en l'absence de nouveaux
    chocs.
    """
    eps = np.asarray(eps, dtype=float)
    eps = eps[np.isfinite(eps)]
    n = len(eps)
    if n < 20:
        return None

    if float(np.std(eps)) <= 0:
        return None

    c = _facteur_echelle(eps)
    z = eps * c                      # serie normalisee (variance unitaire)

    meilleur = None
    for a0, b0 in ((0.10, 0.85), (0.05, 0.90), (0.20, 0.70)):
        x0 = np.array([1.0 - a0 - b0, a0, b0])
        r = minimize(_neg_loglik_garch11, x0, args=(z,), method="SLSQP",
                     bounds=[(1e-10, 10.0), (1e-6, 0.999), (1e-6, 0.999)],
                     constraints=[{"type": "ineq", "fun": lambda p: 0.999 - (p[1] + p[2])}],
                     options={"maxiter": 2000, "ftol": 1e-14})
        if meilleur is None or r.fun < meilleur.fun:
            meilleur = r
    res = meilleur

    omega_z, alpha, beta = res.x
    omega = omega_z / c ** 2                       # retablissement de l'echelle
    sigma2 = _filtre_garch11(res.x, z) / c ** 2
    ll = -res.fun + len(z) * math.log(c)           # correction du changement de variable
    k = 3
    persistance = alpha + beta
    var_lt = omega / (1 - persistance) if persistance < 1 else float("nan")
    demi_vie = math.log(0.5) / math.log(persistance) if 0 < persistance < 1 else float("inf")

    return ResultatGARCH(
        modele="GARCH(1,1)",
        params={"omega": omega, "alpha": alpha, "beta": beta},
        log_vraisemblance=ll,
        aic=2 * k - 2 * ll,
        bic=k * math.log(n) - 2 * ll,
        n_obs=n,
        sigma_conditionnel=np.sqrt(sigma2),
        residus_standardises=eps / np.sqrt(sigma2),
        persistance=persistance,
        variance_long_terme=var_lt,
        convergence=bool(res.success),
        demi_vie_choc=demi_vie,
    )


# --------------------------------------------------------------------------- #
# GJR-GARCH (TGARCH) — asymetrie
# --------------------------------------------------------------------------- #

def _filtre_gjr(params: np.ndarray, eps: np.ndarray) -> np.ndarray:
    omega, alpha, gamma_a, beta = params
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = max(np.var(eps), 1e-14)
    for t in range(1, n):
        choc = eps[t - 1] ** 2
        indic = 1.0 if eps[t - 1] < 0 else 0.0
        sigma2[t] = omega + (alpha + gamma_a * indic) * choc + beta * sigma2[t - 1]
        if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
            sigma2[t] = 1e-14
    return sigma2


def _neg_loglik_gjr(params: np.ndarray, eps: np.ndarray) -> float:
    sigma2 = _filtre_gjr(params, eps)
    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
        return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    return -ll if np.isfinite(ll) else 1e10


def ajuster_gjr_garch(eps: np.ndarray) -> ResultatGARCH | None:
    """GJR-GARCH de Glosten, Jagannathan et Runkle (1993), dit aussi TGARCH.

        sigma^2_t = omega + (alpha + gamma x 1[eps_{t-1} < 0]) eps^2_{t-1}
                    + beta x sigma^2_{t-1}

    Le terme gamma capture l'**effet de levier** : un choc negatif accroit
    davantage la volatilite future qu'un choc positif de meme ampleur. Un
    gamma significativement positif valide cette asymetrie.

    Nuance importante pour le change : l'effet de levier est un fait stylise
    documente principalement sur les **actions**, ou une baisse degrade le
    bilan des entreprises. Sur une paire de devises, la notion de \"mauvaise
    nouvelle\" est ambigue -- une baisse du USD/MAD est defavorable a un
    exportateur mais favorable a un importateur. Un gamma faible ou non
    significatif sur cette paire n'est donc pas un echec du modele, mais un
    resultat coherent avec la nature de l'actif.
    """
    eps = np.asarray(eps, dtype=float)
    eps = eps[np.isfinite(eps)]
    n = len(eps)
    if n < 25:
        return None

    if float(np.std(eps)) <= 0:
        return None

    c = _facteur_echelle(eps)
    z = eps * c

    meilleur = None
    for a0, g0, b0 in ((0.05, 0.05, 0.85), (0.08, 0.02, 0.86), (0.03, 0.10, 0.85)):
        x0 = np.array([max(1.0 - a0 - g0 / 2 - b0, 1e-3), a0, g0, b0])
        r = minimize(_neg_loglik_gjr, x0, args=(z,), method="SLSQP",
                     bounds=[(1e-10, 10.0), (0.0, 0.999), (-0.5, 0.999), (1e-6, 0.999)],
                     constraints=[{"type": "ineq",
                                   "fun": lambda p: 0.999 - (p[1] + p[2] / 2 + p[3])}],
                     options={"maxiter": 2000, "ftol": 1e-14})
        if meilleur is None or r.fun < meilleur.fun:
            meilleur = r
    res = meilleur

    omega_z, alpha, gamma_a, beta = res.x
    omega = omega_z / c ** 2
    sigma2 = _filtre_gjr(res.x, z) / c ** 2
    ll = -res.fun + len(z) * math.log(c)
    k = 4
    persistance = alpha + gamma_a / 2 + beta
    var_lt = omega / (1 - persistance) if persistance < 1 else float("nan")
    demi_vie = math.log(0.5) / math.log(persistance) if 0 < persistance < 1 else float("inf")

    return ResultatGARCH(
        modele="GJR-GARCH(1,1)",
        params={"omega": omega, "alpha": alpha, "gamma": gamma_a, "beta": beta},
        log_vraisemblance=ll,
        aic=2 * k - 2 * ll,
        bic=k * math.log(n) - 2 * ll,
        n_obs=n,
        sigma_conditionnel=np.sqrt(sigma2),
        residus_standardises=eps / np.sqrt(sigma2),
        persistance=persistance,
        variance_long_terme=var_lt,
        convergence=bool(res.success),
        demi_vie_choc=demi_vie,
    )


# --------------------------------------------------------------------------- #
# EGARCH — asymetrie en log-variance
# --------------------------------------------------------------------------- #

def _filtre_egarch(params: np.ndarray, eps: np.ndarray) -> np.ndarray:
    omega, alpha, gamma_a, beta = params
    n = len(eps)
    log_s2 = np.empty(n)
    log_s2[0] = math.log(max(np.var(eps), 1e-14))
    for t in range(1, n):
        sig_prec = math.sqrt(math.exp(min(log_s2[t - 1], 50.0)))
        z = eps[t - 1] / sig_prec if sig_prec > 0 else 0.0
        log_s2[t] = (omega + alpha * (abs(z) - ESP_ABS_Z) + gamma_a * z
                     + beta * log_s2[t - 1])
        log_s2[t] = float(np.clip(log_s2[t], -60.0, 50.0))
    return np.exp(log_s2)


def _neg_loglik_egarch(params: np.ndarray, eps: np.ndarray) -> float:
    sigma2 = _filtre_egarch(params, eps)
    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
        return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    return -ll if np.isfinite(ll) else 1e10


def ajuster_egarch(eps: np.ndarray) -> ResultatGARCH | None:
    """EGARCH(1,1) de Nelson (1991), specifie sur le logarithme de la variance.

        ln(sigma^2_t) = omega + alpha (|z_{t-1}| - E|z|) + gamma z_{t-1}
                        + beta ln(sigma^2_{t-1})

    Deux avantages sur le GARCH standard :
      - la variance modelisee est un logarithme, donc **positive par
        construction** : aucune contrainte de positivite n'est necessaire sur
        les parametres ;
      - le terme gamma capture l'asymetrie de facon continue, sans recourir a
        une variable indicatrice.

    La persistance est ici portee par le seul coefficient beta.
    """
    eps = np.asarray(eps, dtype=float)
    eps = eps[np.isfinite(eps)]
    n = len(eps)
    if n < 25:
        return None

    if float(np.std(eps)) <= 0:
        return None

    c = _facteur_echelle(eps)
    z = eps * c

    meilleur = None
    for a0, g0, b0 in ((0.15, -0.03, 0.92), (0.20, 0.00, 0.85), (0.10, -0.08, 0.95)):
        x0 = np.array([0.0, a0, g0, b0])
        r = minimize(_neg_loglik_egarch, x0, args=(z,), method="L-BFGS-B",
                     bounds=[(-15.0, 5.0), (-1.0, 1.5), (-1.0, 1.0), (-0.999, 0.999)],
                     options={"maxiter": 2000, "ftol": 1e-15})
        if meilleur is None or r.fun < meilleur.fun:
            meilleur = r
    res = meilleur

    omega_z, alpha, gamma_a, beta = res.x
    omega = omega_z - 2 * math.log(c) * (1 - beta)   # specification en log-variance
    sigma2 = _filtre_egarch(res.x, z) / c ** 2
    ll = -res.fun + len(z) * math.log(c)
    k = 4
    var_lt = math.exp(omega / (1 - beta)) if abs(beta) < 1 else float("nan")
    demi_vie = math.log(0.5) / math.log(abs(beta)) if 0 < abs(beta) < 1 else float("inf")

    return ResultatGARCH(
        modele="EGARCH(1,1)",
        params={"omega": omega, "alpha": alpha, "gamma": gamma_a, "beta": beta},
        log_vraisemblance=ll,
        aic=2 * k - 2 * ll,
        bic=k * math.log(n) - 2 * ll,
        n_obs=n,
        sigma_conditionnel=np.sqrt(sigma2),
        residus_standardises=eps / np.sqrt(sigma2),
        persistance=abs(beta),
        variance_long_terme=var_lt,
        convergence=bool(res.success),
        demi_vie_choc=demi_vie,
    )


# --------------------------------------------------------------------------- #
# Prevision
# --------------------------------------------------------------------------- #

def prevoir_volatilite(res: ResultatGARCH, eps: np.ndarray, horizon: int = 10) -> np.ndarray:
    """Prevision de la volatilite conditionnelle sur `horizon` jours.

    Pour un GARCH(1,1), la prevision a h pas converge geometriquement vers la
    volatilite de long terme :

        sigma^2_{t+h} = var_lt + (alpha+beta)^{h-1} x (sigma^2_{t+1} - var_lt)

    C'est cette propriete de retour a la moyenne qui distingue une prevision
    GARCH d'une simple extrapolation : apres un choc, le modele anticipe une
    normalisation progressive, a une vitesse dictee par la persistance.
    """
    p = res.params
    sigma2_last = float(res.sigma_conditionnel[-1] ** 2)
    eps_last = float(eps[-1])

    if res.modele.startswith("GARCH"):
        omega, alpha, beta = p["omega"], p["alpha"], p["beta"]
        s2_next = omega + alpha * eps_last ** 2 + beta * sigma2_last
        pers = alpha + beta
    elif res.modele.startswith("GJR"):
        omega, alpha, gam, beta = p["omega"], p["alpha"], p["gamma"], p["beta"]
        indic = 1.0 if eps_last < 0 else 0.0
        s2_next = omega + (alpha + gam * indic) * eps_last ** 2 + beta * sigma2_last
        pers = alpha + gam / 2 + beta
    else:  # EGARCH
        omega, alpha, gam, beta = p["omega"], p["alpha"], p["gamma"], p["beta"]
        sig_prec = math.sqrt(sigma2_last)
        z = eps_last / sig_prec if sig_prec > 0 else 0.0
        log_s2 = omega + alpha * (abs(z) - ESP_ABS_Z) + gam * z + beta * math.log(sigma2_last)
        s2_next = math.exp(float(np.clip(log_s2, -60, 50)))
        pers = abs(beta)

    var_lt = res.variance_long_terme
    if not np.isfinite(var_lt):
        var_lt = s2_next

    out = []
    for h in range(1, horizon + 1):
        s2_h = var_lt + (pers ** (h - 1)) * (s2_next - var_lt)
        out.append(math.sqrt(max(s2_h, 0.0)))
    return np.array(out)


# --------------------------------------------------------------------------- #
# Impact sur les parametres du pricer
# --------------------------------------------------------------------------- #

@dataclass
class ImpactPricer:
    sigma_constant: float
    sigma_conditionnel: float
    ratio: float
    k_constant: float
    k_conditionnel: float
    skew_constant_pips: float
    skew_conditionnel_pips: float
    kappa_constant: float
    kappa_conditionnel: float
    var99_constant: float
    var99_conditionnel: float


def impact_sur_pricer(sigma_cst: float, sigma_cond: float, gamma: float,
                      position: float, eta: float, lambda_: float) -> ImpactPricer:
    """Compare les grandeurs du pricer sous volatilite constante et conditionnelle.

    Point central de la demarche : **gamma reste inchange**. Il a ete calibre
    sur un objectif de gestion (un decalage cible a la position limite), qui
    releve d'une politique de risque et n'a aucune raison de varier au jour le
    jour. Seule la volatilite devient conditionnelle.

    La consequence est automatique : k = gamma x sigma^2 augmente en regime
    volatil, donc le skew s'elargit et la banque incite plus fortement au
    reequilibrage, precisement lorsque porter l'inventaire coute le plus cher.
    """
    k_cst = gamma * sigma_cst ** 2
    k_cond = gamma * sigma_cond ** 2
    pos_m = position / 1_000_000.0

    kap_cst = math.sqrt(lambda_ * sigma_cst ** 2 / eta) if eta > 0 else 0.0
    kap_cond = math.sqrt(lambda_ * sigma_cond ** 2 / eta) if eta > 0 else 0.0

    z99 = 2.3263
    return ImpactPricer(
        sigma_constant=sigma_cst,
        sigma_conditionnel=sigma_cond,
        ratio=(sigma_cond / sigma_cst) if sigma_cst > 0 else float("nan"),
        k_constant=k_cst,
        k_conditionnel=k_cond,
        skew_constant_pips=-k_cst * pos_m * 10_000,
        skew_conditionnel_pips=-k_cond * pos_m * 10_000,
        kappa_constant=kap_cst,
        kappa_conditionnel=kap_cond,
        var99_constant=z99 * abs(position) * sigma_cst,
        var99_conditionnel=z99 * abs(position) * sigma_cond,
    )


# --------------------------------------------------------------------------- #
# Simulation d'une serie longue realiste
# --------------------------------------------------------------------------- #

def simuler_serie_garch(n_jours: int = 1200, mid_initial: float = 9.3534,
                        omega: float = 2.0e-8, alpha: float = 0.09,
                        beta: float = 0.88, seed: int = 42,
                        date_fin: str = "2026-07-23") -> pd.DataFrame:
    """Genere une serie de MID quotidiens presentant un vrai regroupement de volatilite.

    Justification : l'historique reel disponible ne compte que quelques
    dizaines d'observations, alors qu'une estimation GARCH stable en requiert
    couramment plusieurs centaines. Estimer un GARCH sur un echantillon aussi
    court produirait des coefficients sans signification statistique.

    Cette serie simulee permet de **demontrer que la chaine d'estimation
    fonctionne** -- les parametres retrouves doivent etre proches de ceux
    utilises pour la generation -- en attendant de disposer d'un historique
    reel suffisant. Elle est explicitement identifiee comme simulee dans
    l'interface, et ne se substitue jamais aux donnees observees dans le
    calcul du prix affiche.

    Seuls les jours ouvres (lundi-vendredi) sont generes.
    """
    rng = np.random.default_rng(seed)
    var_lt = omega / (1 - alpha - beta)

    sigma2 = var_lt
    eps_prec = 0.0
    variations = []
    for _ in range(n_jours):
        sigma2 = omega + alpha * eps_prec ** 2 + beta * sigma2
        eps = math.sqrt(sigma2) * rng.standard_normal()
        variations.append(eps)
        eps_prec = eps

    niveau = mid_initial
    serie = []
    for e in variations:
        niveau += e
        serie.append(niveau)

    fin = pd.Timestamp(date_fin)
    dates = pd.bdate_range(end=fin, periods=n_jours)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "mid_marche": np.round(serie, 6)})


def comparer_modeles(resultats: list[ResultatGARCH]) -> pd.DataFrame:
    """Tableau comparatif des modeles ajustes, classes par AIC croissant.

    Le critere AIC arbitre entre qualite d'ajustement et parcimonie : un
    modele plus riche n'est retenu que si le gain de vraisemblance justifie
    les parametres supplementaires.
    """
    lignes = []
    for r in resultats:
        if r is None:
            continue
        lignes.append({
            "MODELE": r.modele,
            "LOG-VRAIS.": f"{r.log_vraisemblance:,.1f}".replace(",", " "),
            "AIC": f"{r.aic:,.1f}".replace(",", " "),
            "BIC": f"{r.bic:,.1f}".replace(",", " "),
            "PERSISTANCE": f"{r.persistance:.4f}",
            "DEMI-VIE CHOC": f"{r.demi_vie_choc:.1f} j" if np.isfinite(r.demi_vie_choc) else "n/a",
            "CONVERGENCE": "OUI" if r.convergence else "NON",
            "_aic": r.aic,
        })
    df = pd.DataFrame(lignes)
    if not df.empty:
        df = df.sort_values("_aic").drop(columns=["_aic"]).reset_index(drop=True)
    return df
