"""Change a terme (forward) et swaps de change.

Le pricer spot répond à la question "quel prix aujourd'hui ?". Or une part
majeure de l'activité d'un desk de change corporate porte sur des opérations
à terme : un importateur qui doit payer en dollars dans trois mois, un
exportateur qui recevra des devises dans six mois.

Contrairement au spot, le prix à terme ne résulte pas d'une modélisation du
risque d'inventaire : il est **impose par un arbitrage**. Si le cours à terme
s'écartait de la parite des taux, il serait possible de construire un profit
sans risque en combinant un emprunt, un change au comptant et un placement.
C'est ce qui rend le pricing du terme structurellement différent de celui du
spot — et c'est aussi ce qui le rend vérifiable.

Convention retenue : base 360 jours (Money Market), usuelle sur le MAD et
l'USD pour les operations de tresorerie.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BASE_JOURS = 360.0

# Echeances standard cotees par un desk de change
ECHEANCES_STANDARD: list[tuple[str, int]] = [
    ("1S", 7),
    ("1M", 30),
    ("2M", 60),
    ("3M", 90),
    ("6M", 180),
    ("9M", 270),
    ("12M", 360),
]


@dataclass
class CotationTerme:
    tenor: str
    jours: int
    spot: float
    taux_domestique: float      # taux MAD, en decimal (0.0275 = 2,75%)
    taux_etranger: float        # taux USD, en decimal
    forward_outright: float     # le cours a terme complet
    points_terme: float         # (forward - spot), exprime en pips
    report_deport: str          # "REPORT" si forward > spot, "DEPORT" sinon
    cout_couverture_annualise: float  # en % annualise, vu du client


def forward_outright(spot: float, taux_domestique: float, taux_etranger: float, jours: int) -> float:
    """Cours à terme par la parité des taux d'intérêt couverte (Covered Interest Parity).

        F = S x (1 + r_dom x t/360) / (1 + r_etr x t/360)

    Intuition : acheter des dollars dans 3 mois doit couter exactement le meme
    prix que d'emprunter des dirhams aujourd'hui, les changer au comptant, et
    placer les dollars obtenus pendant 3 mois. Si les deux chemins ne
    coutaient pas la meme chose, un arbitragiste exploiterait l'ecart jusqu'a
    le refermer.
    """
    t = jours / BASE_JOURS
    return spot * (1.0 + taux_domestique * t) / (1.0 + taux_etranger * t)


def points_terme(spot: float, forward: float) -> float:
    """Ecart entre le terme et le comptant, exprimé en pips (1 pip = 0,0001).

    C'est ce que le desk cote réellement à ses clients : on n'annonce pas
    "9,3210", on annonce "-320 points" sur le spot du moment. Cette convention
    isole la composante taux d'interet de la composante cours de change.
    """
    return (forward - spot) * 10_000.0


def coter_terme(spot: float, taux_domestique: float, taux_etranger: float,
                tenor: str, jours: int) -> CotationTerme:
    fwd = forward_outright(spot, taux_domestique, taux_etranger, jours)
    pts = points_terme(spot, fwd)
    t = jours / BASE_JOURS
    cout_annualise = ((fwd - spot) / spot) / t if (spot > 0 and t > 0) else 0.0

    return CotationTerme(
        tenor=tenor,
        jours=jours,
        spot=spot,
        taux_domestique=taux_domestique,
        taux_etranger=taux_etranger,
        forward_outright=fwd,
        points_terme=pts,
        report_deport="REPORT" if fwd > spot else "DEPORT",
        cout_couverture_annualise=cout_annualise,
    )


def grille_terme(spot: float, taux_domestique: float, taux_etranger: float,
                 echeances: list[tuple[str, int]] | None = None) -> list[CotationTerme]:
    """Grille complete des echeances standard — l'ecran de cotation d'un desk FX."""
    echeances = echeances or ECHEANCES_STANDARD
    return [coter_terme(spot, taux_domestique, taux_etranger, t, j) for t, j in echeances]


# --------------------------------------------------------------------------- #
# Swap de change
# --------------------------------------------------------------------------- #

@dataclass
class SwapChange:
    montant: float
    jours: int
    tenor: str
    spot: float
    forward: float
    points_swap: float
    jambe_spot_mad: float
    jambe_terme_mad: float
    resultat_net_mad: float


def swap_de_change(montant_usd: float, spot: float, taux_domestique: float,
                   taux_etranger: float, tenor: str, jours: int) -> SwapChange:
    """Swap de change : un achat au comptant et une vente à terme (ou l'inverse),
    conclus simultanément avec la meme contrepartie.

    C'est l'instrument le plus traité au monde en volume sur le marché des
    changes, et pourtant le moins visible : il ne crée **aucune position de
    change nette** (les deux jambes s'annulent en montant), mais il déplace
    de la trésorerie d'une devise à l'autre sur une periode donnee. Un desk
    l'utilise pour financer un décalage de dates plutot que pour prendre une
    vue sur le cours.

    Le résultat du swap est donc purement un différentiel de taux : c'est le
    "cout de portage" de la devise sur la période.
    """
    fwd = forward_outright(spot, taux_domestique, taux_etranger, jours)
    jambe_spot = montant_usd * spot
    jambe_terme = montant_usd * fwd
    return SwapChange(
        montant=montant_usd,
        jours=jours,
        tenor=tenor,
        spot=spot,
        forward=fwd,
        points_swap=points_terme(spot, fwd),
        jambe_spot_mad=jambe_spot,
        jambe_terme_mad=jambe_terme,
        resultat_net_mad=jambe_terme - jambe_spot,
    )


# --------------------------------------------------------------------------- #
# Risque de decalage de dates (gap risk)
# --------------------------------------------------------------------------- #

@dataclass
class EcheancierPosition:
    """Position ventilée par échéance : un desk peut etre plat en montant total
    tout en portant un risque réel de décalage de dates."""
    lignes: list[dict] = field(default_factory=list)
    position_nette_totale: float = 0.0
    position_brute_totale: float = 0.0
    gap_max: float = 0.0
    tenor_gap_max: str = ""
    est_plat_en_montant: bool = False


def echeancier(positions_par_tenor: dict[str, float], seuil_plat: float = 1_000.0) -> EcheancierPosition:
    """Ventile la position par échéance et identifie le risque de décalage.

    Un desk peut afficher une position nette nulle — donc apparemment sans
    risque — tout en étant acheteur à 1 mois et vendeur à 6 mois. Il n'a alors
    aucune exposition au **niveau** du cours de change, mais une exposition
    réelle a la **déformation de la courbe de taux** entre les deux échéances.
    C'est précisément ce que le seul suivi de la position nette ne montre pas.
    """
    lignes = []
    net = 0.0
    brut = 0.0
    gap_max = 0.0
    tenor_max = ""

    for tenor, montant in positions_par_tenor.items():
        lignes.append({"tenor": tenor, "position": montant})
        net += montant
        brut += abs(montant)
        if abs(montant) > gap_max:
            gap_max = abs(montant)
            tenor_max = tenor

    return EcheancierPosition(
        lignes=lignes,
        position_nette_totale=net,
        position_brute_totale=brut,
        gap_max=gap_max,
        tenor_gap_max=tenor_max,
        est_plat_en_montant=abs(net) < seuil_plat,
    )


# --------------------------------------------------------------------------- #
# Courbe de taux par interpolation
# --------------------------------------------------------------------------- #

def interpoler_taux(points_courbe: list[tuple[int, float]], jours_cible: int) -> float:
    """Interpolation linéaire d'un taux pour une échéance non cotee.

    Le marché ne cote qu'un nombre limite d'échéances (1M, 3M, 6M, 12M...).
    Pour valoriser une opération à 47 jours, il faut construire un taux
    intermédiaire. L'interpolation linéaire est la convention la plus simple ;
    des méthodes plus fines (splines, Nelson-Siegel) existent mais supposent
    un nombre de points de marché supérieur à ce dont on dispose ici.

    `points_courbe` : liste de (jours, taux), supposée non vide.
    """
    if not points_courbe:
        return 0.0
    pts = sorted(points_courbe)

    if jours_cible <= pts[0][0]:
        return pts[0][1]
    if jours_cible >= pts[-1][0]:
        return pts[-1][1]

    for i in range(len(pts) - 1):
        j0, r0 = pts[i]
        j1, r1 = pts[i + 1]
        if j0 <= jours_cible <= j1:
            if j1 == j0:
                return r0
            poids = (jours_cible - j0) / (j1 - j0)
            return r0 + poids * (r1 - r0)
    return pts[-1][1]
