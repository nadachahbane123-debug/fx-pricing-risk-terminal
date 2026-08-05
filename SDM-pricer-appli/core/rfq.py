"""RFQ (Request for Quote) — cotation d'un prix ferme pour un montant client donné.

C'est l'acte quotidien élémentaire d'un desk de change : un client demande un prix
pour un montant précis, le teneur de marche répond par un prix ferme, et la
transaction est enregistrée si le client l'accepte.

Deux mécanismes s'ajoutent ici au prix affiché (BID/ASK "ecran") du pricer :

1. Une **tarification par tranche de montant** (pricing ladder). Le prix ecran vaut
   pour un montant standard ; au-dela, le spread s'élargit, car un montant plus
   important est plus couteux et plus risque à couvrir. Ce principe de segmentation
   par taille est celui décrit par Barzykin, Bergault et Gueant (2021-2023) pour le
   market making de change de gré à gré.

2. Le **skew de position** déjà calculé par le modèle de Ho & Stoll (1981), qui
   décale l'ensemble de la cotation selon l'inventaire détenu par la banque.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Tranches de montant (pricing ladder)
# --------------------------------------------------------------------------- #
# (libelle, seuil superieur en USD, multiplicateur applique au demi-spread)
# Le multiplicateur vaut 1.0 pour la tranche de reference : le prix ecran
# correspond donc a une operation de taille standard.

TIERS: list[tuple[str, float, float]] = [
    ("TIER 1 — Retail", 500_000.0, 1.00),
    ("TIER 2 — Corporate", 2_000_000.0, 1.35),
    ("TIER 3 — Institutionnel", 10_000_000.0, 1.80),
    ("TIER 4 — Bloc", float("inf"), 2.50),
]


def tier_for(montant: float) -> tuple[str, float]:
    """Renvoie (libelle du tier, multiplicateur de spread) pour un montant donne."""
    m = abs(montant)
    for label, seuil, mult in TIERS:
        if m <= seuil:
            return label, mult
    return TIERS[-1][0], TIERS[-1][2]


@dataclass
class QuoteRFQ:
    sens_client: str          # "Achete" (CFG vend USD) ou "Vend" (CFG achete USD)
    montant_usd: float
    tier_label: str
    tier_mult: float
    prix_reservation: float   # MID ajuste du skew de position (Ho & Stoll)
    demi_spread_base: float   # demi-spread ecran, avant elargissement par tranche
    demi_spread_applique: float
    prix_client: float        # le prix ferme propose au client
    contrevaleur_mad: float
    marge_estimee_mad: float  # marge theorique captee par rapport au prix de reservation
    prix_ecran: float         # prix qui aurait ete affiche sans effet de taille


def coter(
    sens_client: str,
    montant_usd: float,
    prix_reservation: float,
    spread_marche: float,
    mid_marche: float,
) -> QuoteRFQ:
    """Construit une cotation ferme pour un montant client.

    `prix_reservation` est le MID déjà décalé du skew de position (sortie du
    modèle de Ho & Stoll) : la cotation hérite donc automatiquement de la
    politique de gestion d'inventaire de la banque.
    """
    tier_label, tier_mult = tier_for(montant_usd)

    demi_spread_base = (spread_marche * mid_marche) / 2.0
    demi_spread_applique = demi_spread_base * tier_mult

    if sens_client == "Achete":
        # Le client achete des USD : CFG vend, donc cote son ASK (au-dessus du MID)
        prix_client = prix_reservation + demi_spread_applique
        prix_ecran = prix_reservation + demi_spread_base
    else:
        # Le client vend des USD : CFG achete, donc cote son BID (en dessous du MID)
        prix_client = prix_reservation - demi_spread_applique
        prix_ecran = prix_reservation - demi_spread_base

    contrevaleur = abs(montant_usd) * prix_client
    marge = abs(montant_usd) * demi_spread_applique

    return QuoteRFQ(
        sens_client=sens_client,
        montant_usd=abs(montant_usd),
        tier_label=tier_label,
        tier_mult=tier_mult,
        prix_reservation=prix_reservation,
        demi_spread_base=demi_spread_base,
        demi_spread_applique=demi_spread_applique,
        prix_client=prix_client,
        contrevaleur_mad=contrevaleur,
        marge_estimee_mad=marge,
        prix_ecran=prix_ecran,
    )


def position_apres_deal(position_actuelle: float, sens_client: str, montant_usd: float) -> float:
    """Position de CFG apres execution du deal.

    Convention du pricer : le client VEND a CFG -> CFG achete -> position +.
    """
    impact = abs(montant_usd) if sens_client == "Vend" else -abs(montant_usd)
    return position_actuelle + impact
