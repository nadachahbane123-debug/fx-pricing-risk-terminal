"""Analyse de series chronologiques appliquee au USD/MAD.

Predire un cours de change est repute sans valeur : depuis Meese et Rogoff
(1983), la litterature etablit qu'aucun modele econometrique ne bat
durablement la marche aleatoire sur les grandes devises. Appliquer un ARIMA
au USD/MAD dans l'espoir de prevoir le cours de demain serait donc un
contresens.

Le dirham n'est cependant pas une devise en flottement libre. Il est arrime a
un panier (60 % EUR / 40 % USD) et evolue dans une bande de +/- 5 % geree par
Bank Al-Maghrib. Cette architecture cree un argument **structurel**, et non
statistique, en faveur d'un retour a la moyenne : plus le cours s'ecarte du
niveau central implique par le panier, plus la probabilite d'un rappel vers
ce niveau augmente.

La question posee ici n'est donc pas \"quel sera le cours demain\", mais :
**le USD/MAD presente-t-il une reversion vers son niveau central, et a quelle
vitesse ?** Cette formulation est testable, et sa reponse a une portee
operationnelle directe : une reversion rapide reduit le risque de porter une
position, une reversion lente ou inexistante l'accroit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Valeurs critiques asymptotiques du test de Dickey-Fuller avec constante
# (Fuller, 1976). Elles ne suivent pas une loi de Student : sous l'hypothese
# nulle de racine unitaire, la statistique a une distribution non standard.
VC_DICKEY_FULLER = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


# --------------------------------------------------------------------------- #
# Decomposition
# --------------------------------------------------------------------------- #

@dataclass
class Decomposition:
    dates: list
    serie: np.ndarray
    tendance: np.ndarray
    residu: np.ndarray
    fenetre: int
    part_tendance: float     # part de la variance expliquee par la tendance


def decomposer(price_history: pd.DataFrame, fenetre: int = 21) -> Decomposition | None:
    """Decomposition tendance / residu par moyenne mobile centree.

    La decomposition classique d'une serie chronologique distingue tendance,
    saisonnalite et residu. Sur une serie de cours de change quotidiens, la
    composante saisonniere n'a pas de sens economique -- il n'y a pas de
    \"mois du dirham fort\" -- et n'est donc pas estimee ici. La decomposition
    se limite a isoler la tendance de fond des fluctuations de court terme.

    La fenetre par defaut de 21 jours correspond approximativement a un mois
    de cotation ouvree.
    """
    if price_history is None or len(price_history) < fenetre + 5:
        return None

    df = price_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    serie = df["mid_marche"].to_numpy(dtype=float)
    tendance = pd.Series(serie).rolling(window=fenetre, center=True, min_periods=1).mean().to_numpy()
    residu = serie - tendance

    var_tot = float(np.var(serie))
    part = 1.0 - float(np.var(residu)) / var_tot if var_tot > 0 else 0.0

    return Decomposition(
        dates=df["date"].dt.strftime("%Y-%m-%d").tolist(),
        serie=serie, tendance=tendance, residu=residu,
        fenetre=fenetre, part_tendance=part,
    )


# --------------------------------------------------------------------------- #
# Autocorrelations
# --------------------------------------------------------------------------- #

def acf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    """Fonction d'autocorrelation empirique."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.array([])
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        return np.zeros(min(nlags, n - 1) + 1)
    return np.array([float(np.dot(xc[k:], xc[:n - k]) / denom) for k in range(min(nlags, n - 1) + 1)])


def pacf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    """Autocorrelation partielle, par l'algorithme de Durbin-Levinson.

    L'autocorrelation partielle mesure la correlation entre y_t et y_{t-k}
    **une fois retire** l'effet des retards intermediaires. C'est l'outil de
    lecture standard pour identifier l'ordre d'un processus autoregressif :
    un AR(p) presente une PACF qui s'annule au-dela du retard p.
    """
    r = acf(x, nlags)
    if len(r) == 0:
        return np.array([])
    p = np.zeros(len(r))
    p[0] = 1.0
    if len(r) > 1:
        phi = np.zeros((len(r), len(r)))
        phi[1, 1] = r[1]
        p[1] = r[1]
        for k in range(2, len(r)):
            num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
            den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
            phi[k, k] = num / den if den != 0 else 0.0
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
            p[k] = phi[k, k]
    return p


def bande_confiance_acf(n: int) -> float:
    """Bande de confiance a 95 % pour l'ACF sous hypothese de bruit blanc : +/- 1.96/racine(n)."""
    return 1.96 / math.sqrt(n) if n > 0 else 0.0


# --------------------------------------------------------------------------- #
# Test de racine unitaire
# --------------------------------------------------------------------------- #

@dataclass
class ResultatDickeyFuller:
    statistique: float
    n_obs: int
    n_retards: int
    rho: float
    valeurs_critiques: dict
    stationnaire_5pct: bool
    conclusion: str


def test_dickey_fuller(y: np.ndarray, n_retards: int = 1) -> ResultatDickeyFuller | None:
    """Test augmente de Dickey-Fuller, avec constante.

    On estime par moindres carres :

        dy_t = a + rho x y_{t-1} + somme(phi_i x dy_{t-i}) + e_t

    L'hypothese nulle est rho = 0, soit la presence d'une racine unitaire :
    la serie est alors une marche aleatoire, les chocs y sont permanents, et
    aucune reversion n'est a attendre. La rejeter en faveur de rho < 0 revient
    a conclure a la stationnarite, donc a l'existence d'un retour a la moyenne.

    La statistique est le t de rho, compare aux valeurs critiques de Fuller :
    sous l'hypothese nulle, cette statistique ne suit pas une loi de Student,
    d'ou l'usage d'une table specifique.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 20:
        return None

    dy = np.diff(y)
    depart = n_retards + 1
    if len(dy) - depart < 10:
        return None

    Y = dy[depart:]
    cols = [np.ones(len(Y)), y[depart:-1]]
    for i in range(1, n_retards + 1):
        cols.append(dy[depart - i:-i])
    X = np.column_stack(cols)

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    ddl = len(Y) - X.shape[1]
    if ddl <= 0:
        return None
    s2 = float(resid @ resid) / ddl
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None

    rho = float(beta[1])
    se_rho = math.sqrt(max(cov[1, 1], 1e-30))
    t_stat = rho / se_rho if se_rho > 0 else 0.0

    stat_5 = t_stat < VC_DICKEY_FULLER["5%"]
    if stat_5:
        concl = ("Hypothese de racine unitaire rejetee au seuil de 5 % : la serie "
                 "presente un retour a la moyenne.")
    else:
        concl = ("Hypothese de racine unitaire non rejetee : sur cet echantillon, la "
                 "serie se comporte comme une marche aleatoire.")

    return ResultatDickeyFuller(
        statistique=t_stat, n_obs=len(Y), n_retards=n_retards, rho=rho,
        valeurs_critiques=VC_DICKEY_FULLER, stationnaire_5pct=stat_5, conclusion=concl,
    )


# --------------------------------------------------------------------------- #
# AR(p) et selection d'ordre
# --------------------------------------------------------------------------- #

@dataclass
class ResultatAR:
    ordre: int
    constante: float
    coefficients: np.ndarray
    sigma_residu: float
    aic: float
    bic: float
    n_obs: int


def ajuster_ar(y: np.ndarray, p: int) -> ResultatAR | None:
    """Ajustement d'un AR(p) par moindres carres ordinaires."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < p + 10:
        return None

    Y = y[p:]
    X = np.column_stack([np.ones(len(Y))] + [y[p - i:-i if i else None] for i in range(1, p + 1)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    k = p + 1
    m = len(Y)
    sigma2 = float(resid @ resid) / m
    if sigma2 <= 0:
        return None
    ll = -0.5 * m * (math.log(2 * math.pi) + math.log(sigma2) + 1)
    return ResultatAR(
        ordre=p, constante=float(beta[0]), coefficients=beta[1:],
        sigma_residu=math.sqrt(sigma2),
        aic=2 * k - 2 * ll, bic=k * math.log(m) - 2 * ll, n_obs=m,
    )


def selectionner_ordre_ar(y: np.ndarray, p_max: int = 5) -> pd.DataFrame:
    """Compare les AR(p) pour p = 1..p_max et classe par AIC croissant."""
    lignes = []
    for p in range(1, p_max + 1):
        r = ajuster_ar(y, p)
        if r is None:
            continue
        lignes.append({"ORDRE": f"AR({p})", "AIC": r.aic, "BIC": r.bic,
                       "SIGMA RESIDU": r.sigma_residu, "_p": p})
    df = pd.DataFrame(lignes)
    return df.sort_values("AIC").reset_index(drop=True) if not df.empty else df


# --------------------------------------------------------------------------- #
# Processus d'Ornstein-Uhlenbeck
# --------------------------------------------------------------------------- #

@dataclass
class ResultatOU:
    phi: float               # coefficient autoregressif
    theta: float             # vitesse de retour a la moyenne
    mu: float                # niveau d'equilibre
    sigma: float             # volatilite instantanee
    demi_vie: float          # en jours
    reversion_detectee: bool


def calibrer_ornstein_uhlenbeck(y: np.ndarray, dt: float = 1.0) -> ResultatOU | None:
    """Calibration d'un processus de retour a la moyenne sur la serie de cours.

    Le processus d'Ornstein-Uhlenbeck s'ecrit en temps continu :

        dS_t = theta (mu - S_t) dt + sigma dW_t

    Sa discretisation exacte est un AR(1) : S_t = c + phi S_{t-1} + e_t, avec
    phi = exp(-theta dt). On estime donc l'AR(1) par moindres carres, puis on
    en deduit :

        theta = -ln(phi) / dt          (vitesse de rappel)
        mu    = c / (1 - phi)          (niveau d'equilibre)
        demi-vie = ln(2) / theta       (temps de resorption de la moitie d'un ecart)

    La demi-vie est la grandeur la plus parlante pour un desk : elle indique
    en combien de jours la moitie d'un ecart au niveau d'equilibre se resorbe
    en moyenne. Une demi-vie courte rend le portage d'une position moins
    risque ; une demi-vie tres longue signale, a l'inverse, l'absence de
    reversion exploitable.
    """
    r = ajuster_ar(y, 1)
    if r is None:
        return None

    phi = float(r.coefficients[0])
    if phi <= 0 or phi >= 1:
        # phi >= 1 : pas de reversion (racine unitaire ou explosif)
        return ResultatOU(phi=phi, theta=0.0, mu=float(np.mean(y)),
                          sigma=r.sigma_residu, demi_vie=float("inf"),
                          reversion_detectee=False)

    theta = -math.log(phi) / dt
    mu = r.constante / (1 - phi)
    sigma = r.sigma_residu * math.sqrt(2 * theta / (1 - phi ** 2)) if phi ** 2 < 1 else r.sigma_residu
    demi_vie = math.log(2) / theta if theta > 0 else float("inf")

    return ResultatOU(phi=phi, theta=theta, mu=mu, sigma=sigma,
                      demi_vie=demi_vie, reversion_detectee=True)


def ecart_au_panier(mid_marche: float, mid_theorique: float) -> dict:
    """Ecart relatif entre le cours de marche et le niveau central implique par le panier.

    C'est cet ecart, et non le cours lui-meme, qui constitue le candidat
    naturel a une reversion : le mecanisme de bande de Bank Al-Maghrib porte
    precisement sur la distance au cours central.
    """
    ecart = (mid_marche - mid_theorique) / mid_theorique if mid_theorique else 0.0
    return {
        "ecart_relatif": ecart,
        "ecart_pct": ecart * 100,
        "position_dans_bande": (ecart + 0.05) / 0.10,   # 0 = borne basse, 1 = borne haute
        "marge_avant_borne_pct": (0.05 - abs(ecart)) * 100,
    }
