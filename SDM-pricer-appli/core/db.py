"""Persistance SQLite : journal de transactions et historique de prix (Section 13).

Crée la base au premier lancement, pre-remplit les données de démo, et expose
des fonctions simples utilisées par l'UI (aucune dependance Streamlit ici).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "pricer.db"

# Historique de MID de marche pour la demo (donnees de depart)
# Serie continue sur les jours ouvres (lun-ven) du 25/06 au 16/07/2026 --
# generee une fois de facon reproductible (seed=42) pour combler le trou
# 07/07 -> 14/07 -> 16/07 constate lors des premiers lancements de l'app.
# Valeurs illustratives, a remplacer par de vraies observations quotidiennes.
SEED_PRICE_HISTORY = [
    ("2026-06-25", 9.3489),
    ("2026-06-26", 9.3502),
    ("2026-06-29", 9.3495),
    ("2026-06-30", 9.3511),
    ("2026-07-01", 9.3520),
    ("2026-07-02", 9.3508),
    ("2026-07-03", 9.3517),
    ("2026-07-04", 9.3529),
    ("2026-07-06", 9.3524),
    ("2026-07-07", 9.3534),
    ("2026-07-08", 9.3533),
    ("2026-07-09", 9.3531),
    ("2026-07-10", 9.3530),
    ("2026-07-13", 9.3536),
    ("2026-07-14", 9.3535),
    ("2026-07-15", 9.3522),
    ("2026-07-16", 9.3525),
]

# Transactions d'exemple pour la demo
SEED_TRANSACTIONS = [
    ("2026-07-07", "Client Alpha SARL", "Vend", 2_000_000, None),
    ("2026-07-07", "Client Beta Corp", "Achete", 500_000, None),
    ("2026-07-08", "Client Gamma SA", "Vend", 1_000_000, None),
    ("2026-07-08", "Client Delta Trading", "Achete", 300_000, None),
    ("2026-07-09", "Client Epsilon Import", "Vend", 700_000, None),
    ("2026-07-09", "Client Zeta Export", "Achete", 900_000, None),
]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Cree les tables si absentes et pre-remplit avec les donnees de demo."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            client TEXT NOT NULL,
            sens TEXT NOT NULL CHECK (sens IN ('Vend', 'Achete')),
            montant REAL NOT NULL,
            prix_execution REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            mid_marche REAL NOT NULL,
            eur_usd REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_breaches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            horodatage TEXT NOT NULL,
            position REAL NOT NULL,
            limite REAL NOT NULL,
            pct REAL NOT NULL
        )
        """
    )
    # Journal des demandes de cotation : c'est cette table qui permet de
    # calibrer le modele d'intensite des flux sur des donnees observees plutot
    # que sur des parametres supposes (voir core/flow.py).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rfq_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            horodatage TEXT NOT NULL,
            client TEXT,
            sens TEXT NOT NULL,
            montant REAL NOT NULL,
            delta_propose REAL NOT NULL,
            prix_propose REAL NOT NULL,
            acceptee INTEGER NOT NULL
        )
        """
    )
    # Historique EUR/MAD, necessaire au calcul de la correlation entre paires
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eurmad_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            mid REAL NOT NULL
        )
        """
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) AS n FROM price_history")
    besoin_seed = cur.fetchone()["n"] == 0
    if besoin_seed:
        # Jeu de donnees complet et coherent : prix (avec impact de marche des gros
        # flux), transactions, depassements de limite. Genere en une fois pour que
        # les trois series soient mutuellement coherentes.
        from core import datagen
        usdmad, eurusd_list, transactions, breaches = datagen.generer_jeu_complet(500)
        eurusd = dict(eurusd_list)
        cur.executemany(
            "INSERT INTO price_history (date, mid_marche, eur_usd) VALUES (?, ?, ?)",
            [(d, mid, eurusd.get(d)) for d, mid in usdmad],
        )
        cur.executemany(
            "INSERT INTO transactions (date, client, sens, montant, prix_execution) VALUES (?, ?, ?, ?, ?)",
            transactions,
        )
        if breaches:
            cur.executemany(
                "INSERT INTO risk_breaches (horodatage, position, limite, pct) VALUES (?, ?, ?, ?)",
                breaches,
            )
        conn.commit()

    conn.close()


# --------------------------------------------------------------------------- #
# Transactions
# --------------------------------------------------------------------------- #

def insert_transaction(date: str, client: str, sens: str, montant: float, prix_execution: float | None) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO transactions (date, client, sens, montant, prix_execution) VALUES (?, ?, ?, ?, ?)",
        (date, client, sens, montant, prix_execution),
    )
    conn.commit()
    conn.close()


def get_transactions() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date, id", conn)
    conn.close()
    return df


# --------------------------------------------------------------------------- #
# Historique de prix
# --------------------------------------------------------------------------- #

def get_price_history() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM price_history ORDER BY date", conn)
    conn.close()
    return df


def get_last_price_history_date() -> str | None:
    conn = get_connection()
    row = conn.execute("SELECT MAX(date) AS d FROM price_history").fetchone()
    conn.close()
    return row["d"] if row and row["d"] else None


def insert_price_history_if_new(date: str, mid_marche: float, eur_usd: float | None = None) -> bool:
    """Insère un enregistrement pour `date` s'il n'existe pas deja. Renvoie True si inséré.

    Le cours EUR/USD du jour est conservé aux cotes du MID : c'est le couple
    (EUR/USD, MID) qui permet de re-estimer la relation de panier par régression
    (voir core/ml.py), plutot que de la prendre comme donnée.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO price_history (date, mid_marche, eur_usd) VALUES (?, ?, ?)",
            (date, mid_marche, eur_usd),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Depassements de seuil de risque
# --------------------------------------------------------------------------- #

def log_risk_breach(horodatage: str, position: float, limite: float, pct: float) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO risk_breaches (horodatage, position, limite, pct) VALUES (?, ?, ?, ?)",
        (horodatage, position, limite, pct),
    )
    conn.commit()
    conn.close()


def get_risk_breaches() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM risk_breaches ORDER BY horodatage DESC", conn)
    conn.close()
    return df


# --------------------------------------------------------------------------- #
# Journal des demandes de cotation (RFQ)
# --------------------------------------------------------------------------- #

def insert_rfq(horodatage: str, client: str, sens: str, montant: float,
               delta_propose: float, prix_propose: float, acceptee: bool) -> None:
    """Enregistre une demande de cotation, acceptée ou non.

    Les refus sont aussi importants que les acceptations : sans eux, il est
    impossible d'estimer la sensibilité au prix de la clientèle.
    """
    conn = get_connection()
    conn.execute(
        "INSERT INTO rfq_log (horodatage, client, sens, montant, delta_propose, prix_propose, acceptee) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (horodatage, client, sens, montant, delta_propose, prix_propose, 1 if acceptee else 0),
    )
    conn.commit()
    conn.close()


def get_rfq_log() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM rfq_log ORDER BY horodatage DESC", conn)
    conn.close()
    return df


def seed_rfq_log_if_empty() -> None:
    """Pré-remplit le journal des RFQ avec un échantillon illustratif.

    Ces données sont **simulées** : elles servent uniquement à rendre le module
    de calibration démontrable en l'absence d'historique réel. Le taux
    d'acceptation y décroit avec l'écart de prix propose, conformément au
    modèle exponentiel — ce qui permet de vérifier que la calibration
    retrouve bien les paramètres attendus.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM rfq_log")
    if cur.fetchone()["n"] > 0:
        conn.close()
        return

    import random
    import math as _math
    rng = random.Random(42)

    # --- Profil des contreparties -------------------------------------- #
    # Les demandes sont generees a partir de types de clients distincts, avec
    # chacun sa taille d'operation, sa frequence et son orientation. Sans cette
    # heterogeneite, tous les clients seraient statistiquement identiques et
    # aucune segmentation ne pourrait etre mise en evidence : le clustering ne
    # cree pas de structure, il ne fait que la reveler.
    #
    # (nom, montant moyen, dispersion, part d'achats, nombre de demandes)
    PROFILS = [
        ("Alpha Import SARL",      180_000,  60_000, 0.88, 95),
        ("Beta Distribution",      240_000,  80_000, 0.85, 88),
        ("Gamma Negoce",           150_000,  50_000, 0.90, 78),
        ("Delta Textile Export", 1_600_000, 500_000, 0.14, 62),
        ("Epsilon Agro Export",  2_100_000, 700_000, 0.10, 55),
        ("Zeta Phosphates",      2_800_000, 900_000, 0.18, 48),
        ("Eta Assurances",       7_500_000, 2_500_000, 0.52, 26),
        ("Theta Caisse",         9_200_000, 3_000_000, 0.48, 22),
        ("Iota Industries",      3_400_000, 1_100_000, 0.60, 44),
        ("Kappa Energie",        4_100_000, 1_300_000, 0.55, 38),
    ]

    # Modele logistique d'acceptation, commun a toutes les contreparties.
    B0, B_DELTA, B_LNM, B_SENS = 3.05, -0.085, -0.10, 0.25

    # Niveaux d'ecart discrets : necessaires pour que la calibration
    # exponentielle, qui agrege par palier, dispose d'effectifs suffisants.
    niveaux = [0.0008, 0.0012, 0.0016, 0.0021, 0.0026, 0.0032, 0.0040, 0.0050, 0.0062]

    rows = []
    compteur = 0
    # Effet de persistance : l'issue de la demande precedente du meme client influe
    # legerement sur la suivante (un client satisfait d'un prix accepte tend a
    # reaccepter). Cet effet, module par B_PERSISTANCE, rend l'etude de persistance
    # client (module events) pertinente. Il reste modere, pour ne pas dominer les
    # autres facteurs.
    B_PERSISTANCE = 0.55
    derniere_issue: dict[str, int] = {}
    # Date de base : chaque demande d'un meme client recoit un horodatage
    # strictement croissant, afin que l'ordre chronologique reconstitue plus tard
    # coincide avec l'ordre dans lequel la persistance a ete injectee.
    from datetime import datetime as _dt, timedelta as _td
    base_date = _dt(2026, 1, 1, 9, 0, 0)
    horodatage_client: dict[str, _dt] = {}
    for nom, m_moyen, m_ecart, part_achat, n_demandes in PROFILS:
        for _ in range(n_demandes):
            delta = rng.choice(niveaux)
            montant = max(50_000.0, rng.gauss(m_moyen, m_ecart))
            montant = round(montant / 10_000) * 10_000
            achete = rng.random() < part_achat

            z = (B0 + B_DELTA * (delta * 10_000) + B_LNM * _math.log(montant)
                 + B_SENS * (1.0 if achete else 0.0))
            if nom in derniere_issue:
                z += B_PERSISTANCE * (1.0 if derniere_issue[nom] == 1 else -1.0)
            accepte = rng.random() < 1.0 / (1.0 + _math.exp(-z))
            derniere_issue[nom] = 1 if accepte else 0

            # Horodatage croissant par client (une demande toutes les ~3 heures)
            h = horodatage_client.get(nom, base_date) + _td(hours=3)
            horodatage_client[nom] = h
            rows.append((
                h.strftime("%Y-%m-%dT%H:%M:%S"),
                nom,
                "Achete" if achete else "Vend",
                montant,
                delta,
                9.3534 + (delta if achete else -delta),
                1 if accepte else 0,
            ))
            compteur += 1

    cur.executemany(
        "INSERT INTO rfq_log (horodatage, client, sens, montant, delta_propose, prix_propose, acceptee) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Historique EUR/MAD
# --------------------------------------------------------------------------- #

def get_eurmad_history() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM eurmad_history ORDER BY date", conn)
    conn.close()
    return df


def insert_eurmad_if_new(date: str, mid: float) -> bool:
    conn = get_connection()
    try:
        conn.execute("INSERT INTO eurmad_history (date, mid) VALUES (?, ?)", (date, mid))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def seed_eurmad_if_empty(usdmad_history: pd.DataFrame, eur_usd_ref: float = 1.1430) -> None:
    """Reconstitue un historique EUR/MAD a partir de l'historique USD/MAD.

    Faute de serie EUR/MAD observée, l'historique est reconstruit par le
    croisement EUR/MAD = EUR/USD x USD/MAD, en faisant varier légèrement
    l'EUR/USD autour de sa valeur de référence. Les deux séries sont donc
    corrélées par construction — ce qui est **cohérent avec la réalité du
    panier du dirham**, mais reste une reconstitution et non une observation.
    Cette limite est signalée dans l'interface.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM eurmad_history")
    if cur.fetchone()["n"] > 0:
        conn.close()
        return
    conn.close()

    if usdmad_history is None or usdmad_history.empty:
        return

    import random
    rng = random.Random(7)
    eur_usd = eur_usd_ref
    rows = []
    for _, r in usdmad_history.iterrows():
        eur_usd = eur_usd * (1.0 + rng.gauss(0.0, 0.0018))
        rows.append((str(r["date"]), round(eur_usd * float(r["mid_marche"]), 4)))

    conn = get_connection()
    conn.executemany("INSERT OR IGNORE INTO eurmad_history (date, mid) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
