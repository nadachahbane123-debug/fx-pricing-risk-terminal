"""Segmentation de la clientele par apprentissage non supervise (k-means).

Les tranches de cotation du module RFQ (Retail, Corporate, Institutionnel,
Bloc) reposent sur des seuils de montant fixes -- 500 k, 2 M, 10 M USD -- et
sur des multiplicateurs de spread posés à dire d'expert. Ces valeurs n'ont
été déduites d'aucune observation : elles ont été choisies parce qu'elles
paraissaient raisonnables.

C'est exactement le problème qu'on avait résolu pour le coefficient gamma, en
le dérivant d'un objectif de gestion au lieu de le fixer arbitrairement. La
meme démarche s'applique ici : plutot que de postuler des tranches, on laisse
la structure émerger du comportement observé des clients.

Le partitionnement est réalisé par k-means, implementé directement pour
garder la mécanique explicite : initialisation k-means++, itérations de Lloyd,
et sélection du nombre de groupes par le coefficient de silhouette.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ResultatKMeans:
    k: int
    labels: np.ndarray
    centres: np.ndarray
    inertie: float               # somme des carres intra-classes
    silhouette: float
    n_iterations: int


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def standardiser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre et réduit chaque variable.

    Etape indispensable avant tout k-means : l'algorithme repose sur des
    distances euclidiennes. Sans normalisation, une variable exprimée en
    millions de dollars écraserait complètement une variable exprimée en
    pourcentage, et le partitionnement ne refléterait que la première.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


# --------------------------------------------------------------------------- #
# k-means
# --------------------------------------------------------------------------- #

def _init_kmeanspp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Initialisation k-means++ : les centres initiaux sont tires en favorisant
    les points eloignes de ceux deja choisis.

    Une initialisation purement aleatoire conduit frequemment a des optima
    locaux mediocres ; k-means++ reduit fortement ce risque.
    """
    n = len(X)
    centres = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(axis=2), axis=1)
        total = d2.sum()
        if total <= 0:
            centres.append(X[rng.integers(n)])
            continue
        probas = d2 / total
        centres.append(X[rng.choice(n, p=probas)])
    return np.array(centres)


def kmeans(X: np.ndarray, k: int, seed: int = 42, max_iter: int = 300,
           n_essais: int = 10) -> ResultatKMeans | None:
    """Partitionnement k-means (algorithme de Lloyd), meilleur de plusieurs essais."""
    X = np.asarray(X, dtype=float)
    if len(X) < k or k < 2:
        return None

    rng = np.random.default_rng(seed)
    meilleur = None

    for _ in range(n_essais):
        centres = _init_kmeanspp(X, k, rng)
        labels = np.zeros(len(X), dtype=int)
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            nouveaux = d.argmin(axis=1)
            if n_iter > 1 and np.array_equal(nouveaux, labels):
                break
            labels = nouveaux
            for j in range(k):
                pts = X[labels == j]
                if len(pts) > 0:
                    centres[j] = pts.mean(axis=0)
        inertie = float(sum(((X[labels == j] - centres[j]) ** 2).sum() for j in range(k)))
        if meilleur is None or inertie < meilleur[0]:
            meilleur = (inertie, labels.copy(), centres.copy(), n_iter)

    inertie, labels, centres, n_iter = meilleur
    return ResultatKMeans(k=k, labels=labels, centres=centres, inertie=inertie,
                          silhouette=silhouette(X, labels), n_iterations=n_iter)


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Coefficient de silhouette moyen.

    Pour chaque point, on compare sa distance moyenne aux membres de son
    propre groupe (a) a sa distance moyenne au groupe voisin le plus proche
    (b). Le coefficient (b - a) / max(a, b) vaut 1 pour un point parfaitement
    classe, 0 pour un point a la frontiere, et devient negatif s'il aurait ete
    mieux place ailleurs.

    Lecture usuelle : au-dela de 0,5 la structure est nette ; entre 0,25 et
    0,5 elle est raisonnable ; en dessous, le partitionnement n'apporte pas
    grand-chose.
    """
    X = np.asarray(X, dtype=float)
    uniques = np.unique(labels)
    if len(uniques) < 2 or len(X) <= len(uniques):
        return 0.0

    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(len(X)):
        meme = labels == labels[i]
        meme[i] = False
        if meme.sum() == 0:
            scores.append(0.0)
            continue
        a = float(D[i, meme].mean())
        b = min(float(D[i, labels == c].mean()) for c in uniques if c != labels[i])
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def choisir_k(X: np.ndarray, ks: tuple[int, ...] = (2, 3, 4, 5), seed: int = 42) -> pd.DataFrame:
    """Compare plusieurs nombres de groupes et classe par silhouette decroissante."""
    lignes = []
    for k in ks:
        r = kmeans(X, k, seed=seed)
        if r is None:
            continue
        lignes.append({"NOMBRE DE GROUPES": k,
                       "SILHOUETTE": round(r.silhouette, 4),
                       "INERTIE INTRA": round(r.inertie, 3),
                       "_s": r.silhouette})
    df = pd.DataFrame(lignes)
    return df.sort_values("_s", ascending=False).drop(columns=["_s"]).reset_index(drop=True) if not df.empty else df


# --------------------------------------------------------------------------- #
# Construction des profils clients
# --------------------------------------------------------------------------- #

VARIABLES = ["Montant moyen", "Nombre de demandes", "Taux d'acceptation", "Biais directionnel"]


def construire_profils(rfq_log: pd.DataFrame) -> pd.DataFrame | None:
    """Agrege le journal des demandes de cotation en un profil par client.

    Quatre variables comportementales sont retenues :
      - le montant moyen traite, qui approche la taille du client ;
      - le nombre de demandes, qui mesure son activite ;
      - le taux d'acceptation, qui revele sa sensibilite au prix ;
      - le biais directionnel (part d'achats), qui distingue un importateur
        structurellement acheteur de devises d'un exportateur vendeur.

    Cette derniere variable a une portee operationnelle directe : un client
    systematiquement acheteur deplace la position du desk toujours dans le
    meme sens, ce qui pese davantage sur l'inventaire qu'un client au flux
    equilibre.
    """
    if rfq_log is None or rfq_log.empty or "client" not in rfq_log.columns:
        return None

    g = rfq_log.groupby("client").agg(
        montant_moyen=("montant", "mean"),
        n_demandes=("id", "count"),
        taux_acceptation=("acceptee", "mean"),
    ).reset_index()

    biais = rfq_log.assign(_a=(rfq_log["sens"].astype(str) == "Achete").astype(float)) \
                   .groupby("client")["_a"].mean().reset_index(name="biais_achat")
    profils = g.merge(biais, on="client")
    return profils[profils["n_demandes"] >= 3].reset_index(drop=True)


def segmenter(profils: pd.DataFrame, k: int, seed: int = 42):
    """Applique le k-means aux profils clients et renvoie (profils annotes, resultat)."""
    if profils is None or len(profils) < k:
        return None, None
    X = profils[["montant_moyen", "n_demandes", "taux_acceptation", "biais_achat"]].to_numpy(dtype=float)
    Xs, _, _ = standardiser(X)
    res = kmeans(Xs, k, seed=seed)
    if res is None:
        return None, None
    out = profils.copy()
    out["segment"] = res.labels
    return out, res


def decrire_segments(profils_annotes: pd.DataFrame) -> pd.DataFrame:
    """Statistiques descriptives par segment, avec une etiquette lisible.

    Les etiquettes sont attribuees a partir du montant moyen relatif : elles
    servent la lecture, mais restent une interpretation posee **apres** le
    partitionnement, lequel n'utilise aucune etiquette prealable.
    """
    if profils_annotes is None or profils_annotes.empty:
        return pd.DataFrame()

    agg = profils_annotes.groupby("segment").agg(
        n_clients=("client", "count"),
        montant_moyen=("montant_moyen", "mean"),
        n_demandes=("n_demandes", "mean"),
        taux_acceptation=("taux_acceptation", "mean"),
        biais_achat=("biais_achat", "mean"),
    ).reset_index().sort_values("montant_moyen").reset_index(drop=True)

    etiquettes = ["Petits montants", "Montants intermediaires", "Gros montants", "Tres gros montants"]
    agg["etiquette"] = [etiquettes[min(i, len(etiquettes) - 1)] for i in range(len(agg))]

    return pd.DataFrame({
        "SEGMENT": agg["etiquette"],
        "CLIENTS": agg["n_clients"],
        "MONTANT MOYEN": agg["montant_moyen"].map(lambda v: f"{v:,.0f} USD".replace(",", " ")),
        "DEMANDES / CLIENT": agg["n_demandes"].map(lambda v: f"{v:.1f}"),
        "TAUX D'ACCEPTATION": agg["taux_acceptation"].map(lambda v: f"{v * 100:.1f} %"),
        "BIAIS ACHETEUR": agg["biais_achat"].map(lambda v: f"{v * 100:.0f} %"),
    })
