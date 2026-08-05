"""Génération de données historiques réalistes pour la base du pricer.

L'application etait livree avec des historiques très courts (moins de 20 points),
insuffisants pour les modules qui reposent sur une profondeur temporelle : VaR,
volatilité conditionnelle (GARCH), séries chronologiques, études d'événement.

Ce module génère des historiques longs et cohérents entre eux :
  - une série USD/MAD sur environ deux ans de jours ouvres, avec regroupement de
    volatilité (le processus generateur est un GARCH, ce qui donne des periodes
    calmes et des periodes agitees, comme un vrai marché) ;
  - une série EUR/USD cohérente, dont on deduit l'EUR/MAD par croisement ;
  - un flux de transactions clients aligne sur ces prix et sur les profils de
    contreparties déjà utilises pour les RFQ ;
  - un historique de position, dont on derive les dépassements de limite.

Toutes les séries sont deterministes a graine fixee : deux generations donnent le
même résultat, ce qui garantit la reproductibilite.
"""
from __future__ import annotations 

import math 
from datetime import datetime ,timedelta 

import numpy as np 


# Profils de contreparties, repris a l'identique du seed RFQ pour la coherence.
# (nom, montant moyen, dispersion, part d'achats de USD)
PROFILS_CLIENTS =[
("Alpha Import SARL",180_000 ,60_000 ,0.88 ),
("Beta Distribution",240_000 ,80_000 ,0.85 ),
("Gamma Negoce",150_000 ,50_000 ,0.90 ),
("Delta Textile Export",1_600_000 ,500_000 ,0.14 ),
("Epsilon Agro Export",2_100_000 ,700_000 ,0.10 ),
("Zeta Phosphates",2_800_000 ,900_000 ,0.18 ),
("Eta Assurances",7_500_000 ,2_500_000 ,0.52 ),
("Theta Caisse",9_200_000 ,3_000_000 ,0.48 ),
("Iota Industries",3_400_000 ,1_100_000 ,0.60 ),
("Kappa Energie",4_100_000 ,1_300_000 ,0.55 ),
]


def _jours_ouvres (date_fin :datetime ,n :int )->list [datetime ]:
    """Renvoie les n derniers jours ouvres (lundi-vendredi) se terminant a date_fin."""
    jours =[]
    d =date_fin 
    while len (jours )<n :
        if d .weekday ()<5 :# 0-4 = lundi-vendredi
            jours .append (d )
        d -=timedelta (days =1 )
    return list (reversed (jours ))


    # --------------------------------------------------------------------------- #
    # Series de prix
    # --------------------------------------------------------------------------- #

def generer_usdmad (n_jours :int =500 ,mid_final :float =9.3525 ,
date_fin :str ="2026-07-16",seed :int =1002 ,
transactions :list [tuple ]|None =None )->list [tuple [str ,float ]]:
    """Série USD/MAD quotidienne avec regroupement de volatilité.

    Le cours est génère par un GARCH(1,1) sur les variations, ce qui reproduit
    l'alternance de periodes calmes et agitees observée sur les marchés. Le niveau
    est ancre de sorte que la derniere valeur soit proche du mid de référence du
    pricer, pour la continuite avec le reste de l'application.

    Le dirham etant encadre par une bande, on ajoute un leger rappel vers un niveau
    central : le cours ne derive pas indefiniment, il oscille autour de sa moyenne.

    Si un flux de `transactions` est fourni, un leger impact de marché est ajoute :
    un jour a fort déséquilibre acheteur ou vendeur pousse temporairement le cours,
    impact qui se résorbe ensuite. Cela rend l'étude de résilience du prix (module
    events) pertinente : sans cet effet, les gros flux n'auraient aucune trace dans
    les cours. Il s'agit d'un phenomene de microstructure bien réel (impact de
    marché), ici reproduit de facon simplifiee pour la demonstration.
    """
    rng =np .random .default_rng (seed )

    # Parametres GARCH : volatilite quotidienne moyenne de l'ordre de 8e-4 (coherent
    # avec le sigma utilise dans le pricer), persistance elevee.
    omega ,alpha ,beta =1.8e-8 ,0.12 ,0.86 
    theta_rappel =0.004 # rappel tres leger : preserve le clustering
    mid_central =mid_final 

    var_lt =omega /(1 -alpha -beta )
    sigma2 =var_lt 
    eps_prec =0.0 

    variations =[]
    for _ in range (n_jours ):
        sigma2 =omega +alpha *eps_prec **2 +beta *sigma2 
        eps =math .sqrt (sigma2 )*rng .standard_normal ()
        variations .append (eps )
        eps_prec =eps 

    dates =_jours_ouvres (datetime .fromisoformat (date_fin ),n_jours )

    # Impact de marche : flux net par jour -> pression sur le cours
    impact_par_date ={}
    if transactions is not None :
        flux_jour :dict [str ,float ]={}
        for horodatage ,nom ,sens ,montant ,prix in transactions :
            if nom =="Interbancaire":
                continue # le debouclage n'a pas d'impact client
            d =horodatage [:10 ]
            # Client achete USD -> pression haussiere ; client vend -> baissiere
            signe =1.0 if sens =="Achete"else -1.0 
            flux_jour [d ]=flux_jour .get (d ,0.0 )+signe *montant 
            # Coefficient d'impact : ~1 pip par tranche de 10 M de flux net
        for d ,flux in flux_jour .items ():
            impact_par_date [d ]=(flux /10_000_000.0 )*0.0001 

            # Construction du niveau, avec rappel vers le central et impact transitoire
    niveaux =[]
    niveau =mid_central 
    impact_prec =0.0 
    for eps ,d in zip (variations ,dates ):
        d_str =d .strftime ("%Y-%m-%d")
        impact_jour =impact_par_date .get (d_str ,0.0 )
        # L'impact du jour s'ajoute, celui de la veille se resorbe a moitie
        niveau +=(eps +theta_rappel *(mid_central -niveau )
        +impact_jour -0.5 *impact_prec )
        impact_prec =impact_jour 
        niveaux .append (niveau )

        # Recalage pour que la derniere valeur soit exactement mid_final
    decalage =mid_final -niveaux [-1 ]
    niveaux =[n +decalage for n in niveaux ]

    return [(d .strftime ("%Y-%m-%d"),round (v ,4 ))for d ,v in zip (dates ,niveaux )]


def generer_eurusd (n_jours :int =500 ,eurusd_final :float =1.1430 ,
date_fin :str ="2026-07-16",seed :int =11430 )->list [tuple [str ,float ]]:
    """Série EUR/USD quotidienne, marché aléatoire avec faible rappel.

    L'EUR/USD est un taux flottant : pas de bande, mais une volatilité réaliste
    (de l'ordre de 0,4 % par jour). Sert a calculer l'EUR/MAD par croisement.
    """
    rng =np .random .default_rng (seed )
    sigma =0.0045 
    theta =0.01 
    central =eurusd_final 

    niveau =central 
    niveaux =[]
    for _ in range (n_jours ):
        niveau +=sigma *niveau *rng .standard_normal ()+theta *(central -niveau )
        niveaux .append (niveau )
    decalage =eurusd_final -niveaux [-1 ]
    niveaux =[n +decalage for n in niveaux ]

    dates =_jours_ouvres (datetime .fromisoformat (date_fin ),n_jours )
    return [(d .strftime ("%Y-%m-%d"),round (v ,4 ))for d ,v in zip (dates ,niveaux )]


    # --------------------------------------------------------------------------- #
    # Transactions
    # --------------------------------------------------------------------------- #

def generer_transactions (usdmad :list [tuple [str ,float ]],seed :int =77 ,
trades_par_jour_moyen :float =4.0 )->list [tuple ]:
    """Flux de transactions clients aligne sur l'historique de prix.

    Chaque jour ouvre, un nombre aléatoire de clients passent des opérations. Le
    prix d'exécution est le mid du jour, legerement decale selon le sens (le desk
    encaisse un demi-spread). Les contreparties et leurs tailles typiques
    reprennent les profils utilises pour les RFQ, garantissant la cohérence entre
    les modules.

    Le flux n'est pas symetrique jour après jour : certains jours sont nettement
    acheteurs, d'autres vendeurs, ce qui fait evoluer la position du desk de facon
    réaliste et cree des periodes de tension (utiles pour l'étude des dépassements
    de limite).
    """
    rng =np .random .default_rng (seed )
    demi_spread =0.0027 # ~2,7 pips, coherent avec le pricer
    rows =[]
    position_courante =0.0 # suivi pour le debouclage de fin de journee

    for date ,mid in usdmad :
    # Biais directionnel du jour : rend certains jours nettement desequilibres
        biais_jour =rng .normal (0 ,0.35 )
        n_trades =max (0 ,int (rng .poisson (trades_par_jour_moyen )))

        for i in range (n_trades ):
        # Ponderation : les petits importateurs/exportateurs sont frequents,
        # les gros institutionnels rares. Sans cela, chaque journee comporterait
        # plusieurs blocs de 8-9M et la limite serait franchie en permanence, ce
        # qui n'a rien de realiste pour un desk de la taille de CFG.
            poids =np .array ([5 ,5 ,4 ,3 ,3 ,2 ,1 ,1 ,2 ,2 ],dtype =float )
            poids /=poids .sum ()
            idx =rng .choice (len (PROFILS_CLIENTS ),p =poids )
            profil =PROFILS_CLIENTS [idx ]
            nom ,m_moyen ,m_ecart ,part_achat =profil 

            montant =max (50_000.0 ,rng .normal (m_moyen ,m_ecart ))
            montant =round (montant /10_000 )*10_000 

            # Probabilite d'achat = tendance du client + biais du jour
            p_achat =1.0 /(1.0 +math .exp (-(
            math .log (part_achat /(1 -part_achat ))+biais_jour )))
            achete =rng .random ()<p_achat 

            # Le client achete du USD -> il paie l'ASK ; il vend -> il touche le BID
            prix =mid +(demi_spread if achete else -demi_spread )
            heure =9 +(i %8 )
            minute =(i *13 )%60 

            rows .append ((
            f"{date }T{heure :02d}:{minute :02d}:00",
            nom ,
            "Achete"if achete else "Vend",
            float (montant ),
            round (prix ,4 ),
            ))
            # Client achete -> desk vend -> position desk baisse, et inversement
            position_courante +=(-montant if achete else montant )

            # Debouclage partiel de fin de journee : le desk ne porte pas une grosse
            # position ouverte pendant la nuit. Il ramene son exposition proche de zero
            # via une operation interbancaire (contrepartie « Interbancaire »), en
            # laissant un residu aleatoire modeste. C'est ce qui empeche la position de
            # deriver indefiniment, tout en laissant des pics intrajournaliers realistes.
        residu_cible =rng .normal (0 ,600_000 )
        a_deboucler =position_courante -residu_cible 
        if abs (a_deboucler )>=100_000 :
            montant_deb =round (abs (a_deboucler )/10_000 )*10_000 
            # Si la position du desk est positive (long USD), le desk vend -> cote
            # « Achete » du point de vue de la contrepartie interbancaire.
            sens_cp ="Achete"if a_deboucler >0 else "Vend"
            prix =mid +(demi_spread if sens_cp =="Achete"else -demi_spread )
            rows .append ((
            f"{date }T17:30:00",
            "Interbancaire",
            sens_cp ,
            float (montant_deb ),
            round (prix ,4 ),
            ))
            position_courante +=(-montant_deb if sens_cp =="Achete"else montant_deb )

    return rows 


    # --------------------------------------------------------------------------- #
    # Position et depassements de limite
    # --------------------------------------------------------------------------- #

def calculer_position_et_breaches (transactions :list [tuple ],limite :float =20_000_000.0 ,
seuil_alerte :float =0.80 ,seuil_sortie :float =0.60 ):
    """Reconstruit la position cumulée du desk et detecte les episodes de dépassement.

    Convention du pricer : un client qui VEND du USD fait ACHETER le desk, donc la
    position du desk augmente (+) ; un client qui ACHETE fait vendre le desk (-).

    Note sur la limite : elle vaut ici 20 M USD, et ne doit pas etre confondue avec
    le paramètre de 5 M utilise ailleurs pour calibrer le décalage cible du skew
    (gamma). Ce sont deux notions distinctes : la premiere est une limite prudentielle
    sur l'exposition totale du desk, la seconde un objectif de gestion servant au
    pricing. La limite de 20 M est calibrée pour que les dépassements restent des
    événements exceptionnels compte tenu du flux, condition nécessaire pour en faire
    une étude d'événement exploitable.

    Un episode de dépassement (breach) demarre lorsque la position absolue franchit
    le seuil d'alerte (80 % de la limite). Pour eviter de comptabiliser chaque
    oscillation autour du seuil comme un nouvel événement, on n'en ouvre un nouveau
    qu'après etre repasse sous un seuil de sortie plus bas (60 %) : c'est une
    hystérésis, technique standard pour isoler des episodes distincts. On enregistre
    l'horodatage d'entree, la position et le pic atteint pendant l'episode.
    """
    position =0.0 
    historique_position =[]
    breaches =[]
    en_depassement =False 
    pic_courant =0.0 
    horodatage_debut =None 

    trans_triees =sorted (transactions ,key =lambda t :t [0 ])

    for horodatage ,nom ,sens ,montant ,prix in trans_triees :
        delta =montant if sens =="Vend"else -montant 
        position +=delta 
        historique_position .append ((horodatage ,position ))

        pct =abs (position )/limite 
        if not en_depassement and pct >=seuil_alerte :
        # Debut d'un nouvel episode
            en_depassement =True 
            pic_courant =abs (position )
            horodatage_debut =horodatage 
        elif en_depassement :
            pic_courant =max (pic_courant ,abs (position ))
            if pct <seuil_sortie :
            # Fin de l'episode : on l'enregistre
                breaches .append ((horodatage_debut ,round (pic_courant ,2 ),limite ,
                round (pic_courant /limite *100 ,1 )))
                en_depassement =False 

                # Episode encore ouvert en fin de periode
    if en_depassement :
        breaches .append ((horodatage_debut ,round (pic_courant ,2 ),limite ,
        round (pic_courant /limite *100 ,1 )))

    return historique_position ,breaches 


    # --------------------------------------------------------------------------- #
    # Orchestration
    # --------------------------------------------------------------------------- #

def generer_jeu_complet (n_jours :int =500 ,seed :int =1002 ):
    """Génère un jeu de données cohérent : prix, transactions, position, breaches.

    La dependance est circulaire (les transactions dépendent des prix, et les prix
    subissent l'impact des transactions), donc on procède en deux temps :
      1. une série de prix « brute » sans impact, pour caler les transactions ;
      2. les transactions générées sur cette base ;
      3. la série de prix finale, qui intègre l'impact de marché de ces transactions.

    Renvoie (usdmad, eurusd, transactions, breaches).
    """
    usdmad_brut =generer_usdmad (n_jours ,seed =seed )
    transactions =generer_transactions (usdmad_brut ,seed =seed +7 )
    usdmad =generer_usdmad (n_jours ,seed =seed ,transactions =transactions )
    eurusd =generer_eurusd (n_jours ,seed =seed +100 )
    _ ,breaches =calculer_position_et_breaches (transactions )
    return usdmad ,eurusd ,transactions ,breaches 
