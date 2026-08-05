"""Détection d'anomalies sur les cotations concurrentes et estimation robuste du MID.

Le MID de marché est aujourd'hui la **moyenne arithmétique** des cotations des
neuf banques concurrentes. Cette statistique a un défaut connu : elle n'est pas
robuste. Une seule cotation erronée -- saisie fautive, prix périmé, décimale
mal placée -- se propage directement au prix affiché au client, puis, par
l'intermédiaire de sigma et du skew, à l'ensemble des décisions du desk.

Ce module répond à ce risque de deux facons complémentaires :
  - en **signalant** les cotations aberrantes, pour permettre leur vérification ;
  - en proposant une estimation **robuste** du MID, insensible à quelques
    valeurs extremes.

La méthode retenue repose sur la médiane et l'écart absolu median (MAD)
plutot que sur la moyenne et l'eéart-type. Le motif est simple : l'écart-type
est lui-meme contamine par les valeurs aberrantes qu'il est censé servir à
détecter. Une seule cotation très éloignée gonfle l'écart-type au point de se
faire passer pour normale -- c'est l'effet de masquage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Facteur de coherence : pour une loi normale, MAD x 1.4826 estime l'ecart-type.
FACTEUR_MAD = 1.4826

# Seuil usuel du score z modifie (Iglewicz et Hoaglin, 1993).
SEUIL_DEFAUT = 3.5


@dataclass
class ResultatAnomalies:
    quotes: pd.DataFrame          # cotations enrichies (score, statut)
    n_anomalies: int
    mediane: float
    mad: float
    ecart_type_robuste: float
    methode_echelle: str
    mid_moyenne: float            # estimation actuelle, non robuste
    mid_median: float             # estimation robuste
    mid_moyenne_elaguee: float    # moyenne apres retrait des anomalies
    ecart_pips: float             # ecart entre estimation robuste et moyenne
    seuil: float


def detecter(quotes: pd.DataFrame, seuil: float = SEUIL_DEFAUT) -> ResultatAnomalies | None:
    """Détecte les cotations aberrantes par score z modifié.

        score_i = 0.6745 x (x_i - mediane) / MAD

    où MAD est l'écart absolu median. Le facteur 0,6745 (inverse de 1,4826)
    calibre le score pour qu'il s'interprète comme un écart-type sur des
    données normales, rendant le seuil de 3,5 comparable à un seuil classique
    de 3 écarts-types -- mais sans la vulnerabilité de ce dernier.

    Le score est calculé sur le MID de chaque banque, c'est-à-dire sur le
    niveau de prix coté, et non sur la largeur de fourchette : c'est un niveau
    aberrant qui contamine la moyenne, un spread inhabituel relevant d'une
    politique commerciale et non d'une erreur.
    """
    if quotes is None or len(quotes) < 4:
        return None
    if not {"bid", "ask"}.issubset(quotes.columns):
        return None

    df = quotes.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    x = df["mid"].to_numpy(dtype=float)

    mediane = float(np.median(x))
    mad = float(np.median(np.abs(x - mediane)))

    # Cas degenere du MAD : lorsque plus de la moitie des cotations sont
    # rigoureusement identiques -- situation frequente sur un marche etroit ou
    # les banques s'alignent au centime -- l'ecart absolu median vaut zero, et
    # tous les scores s'annuleraient, y compris pour une cotation manifestement
    # aberrante. Iglewicz et Hoaglin recommandent dans ce cas de se rabattre sur
    # l'ecart absolu **moyen** a la mediane, dont le facteur de coherence est
    # 1,253314 pour une loi normale.
    methode_echelle = "MAD"
    if mad > 0:
        echelle = mad / 0.6745
    else:
        mean_ad = float(np.mean(np.abs(x - mediane)))
        methode_echelle = "Ecart absolu moyen (repli, MAD nul)"
        echelle = mean_ad * 1.253314 if mean_ad > 0 else 0.0

    if echelle > 0:
        scores = (x - mediane) / echelle
    else:
        # Toutes les cotations sont rigoureusement identiques : aucune dispersion,
        # donc aucune anomalie detectable.
        scores = np.zeros_like(x)

    df["score"] = scores
    df["anomalie"] = np.abs(scores) > seuil

    n_anom = int(df["anomalie"].sum())
    sains = df.loc[~df["anomalie"], "mid"]

    mid_moy = float(np.mean(x))
    mid_med = mediane
    mid_elag = float(sains.mean()) if len(sains) > 0 else mid_moy

    return ResultatAnomalies(
        quotes=df,
        n_anomalies=n_anom,
        mediane=mediane,
        mad=mad,
        ecart_type_robuste=echelle,
        methode_echelle=methode_echelle,
        mid_moyenne=mid_moy,
        mid_median=mid_med,
        mid_moyenne_elaguee=mid_elag,
        ecart_pips=(mid_elag - mid_moy) * 10_000,
        seuil=seuil,
    )


def comparer_estimateurs(res: ResultatAnomalies) -> pd.DataFrame:
    """Met en regard les trois estimations possibles du MID de marché."""
    return pd.DataFrame([
        {"ESTIMATEUR": "Moyenne (méthode actuelle)",
         "VALEUR": f"{res.mid_moyenne:.4f}",
         "ROBUSTE": "Non",
         "COMMENTAIRE": "Sensible à une cotation erronée"},
        {"ESTIMATEUR": "Médiane",
         "VALEUR": f"{res.mid_median:.4f}",
         "ROBUSTE": "Oui",
         "COMMENTAIRE": "Insensible jusqu'a 50 % de valeurs aberrantes"},
        {"ESTIMATEUR": "Moyenne élaguée",
         "VALEUR": f"{res.mid_moyenne_elaguee:.4f}",
         "ROBUSTE": "Oui",
         "COMMENTAIRE": "Moyenne après retrait des anomalies detectées"},
    ])


def injecter_anomalie(quotes: pd.DataFrame, indice: int = 0,
                      decalage_pips: float = 120.0) -> pd.DataFrame:
    """Introduit volontairement une cotation erronée, pour démonstration.

    Sans anomalie réelle dans les donnees, il est impossible de montrer que le
    dispositif fonctionne. Cette fonction sert uniquement à l'illustration
    dans l'interface : elle simule une erreur de saisie sur une banque, du
    type decimale mal placée ou prix reste figé sur une cotation ancienne.
    """
    df = quotes.copy()
    if 0 <= indice < len(df):
        d = decalage_pips / 10_000.0
        df.loc[df.index[indice], "bid"] = float(df.loc[df.index[indice], "bid"]) + d
        df.loc[df.index[indice], "ask"] = float(df.loc[df.index[indice], "ask"]) + d
    return df


def impact_sur_prix(res: ResultatAnomalies, spread_marche: float) -> dict:
    """Traduit l'écart d'estimation du MID en écart sur le prix affiché au client.

    L'intéret du calcul est de ramener une question statistique -- quel
    estimateur de tendance centrale retenir -- à sa conséquence concrete :
    de combien de pips le prix propose au client diffère selon la methode.
    """
    demi_spread = spread_marche * res.mid_moyenne / 2.0
    return {
        "ask_moyenne": res.mid_moyenne + demi_spread,
        "ask_robuste": res.mid_moyenne_elaguee + demi_spread,
        "bid_moyenne": res.mid_moyenne - demi_spread,
        "bid_robuste": res.mid_moyenne_elaguee - demi_spread,
        "ecart_pips": res.ecart_pips,
    }
