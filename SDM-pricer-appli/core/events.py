"""Études d'événement sur l'activite du desk, inspirees de l'étude dividendes.

L'étude sur les dividendes reposait sur une idee simple et puissante : aligner un
ensemble d'événements sur un instant t = 0, puis observer le comportement moyen du
cours autour de cet instant (« temps événementiel »). Ce module transpose cette
mecanique a trois questions propres au desk de change :

  A. Résilience du prix après un gros flux
     Quand une opération de gros montant passe, le MID de marché subit-il un choc,
     et en combien de temps revient-il a son niveau ? C'est la mesure empirique de
     l'impact de marché et de la résilience, deux notions que le modèle
     d'Almgren-Chriss (module de liquidation) suppose sans les mesurer.

  B. Étude des dépassements de limite
     Après un franchissement de la limite de position, combien de temps le desk
     met-il a revenir dans les clous, et quelle ampleur la position atteint-elle ?

  C. Persistance du comportement client
     Un client qui vient de refuser plusieurs cotations est-il plus enclin a
     refuser la suivante ? Existe-t-il un « momentum » d'acceptation exploitable
     pour affiner le pricing ?

Chaque étude renvoie a la fois des indicateurs agreges et les séries nécessaires
aux graphiques evenementiels.
"""
from __future__ import annotations 

from dataclasses import dataclass 

import numpy as np 
import pandas as pd 


# =========================================================================== #
# A. Resilience du prix apres un gros flux
# =========================================================================== #

@dataclass 
class ResilienceResult :
    n_evenements :int 
    seuil_montant :float 
    impact_immediat_pips :float # decalage moyen du MID le jour de l'evenement
    impact_max_pips :float # decalage maximal moyen atteint
    demi_vie_jours :float # jours pour resorber la moitie de l'impact
    taux_resorption_5j :float # % de l'impact resorbe apres 5 jours
    courbe :pd .DataFrame # trajectoire moyenne (t, ecart_pips)


def etude_resilience (transactions :pd .DataFrame ,price_history :pd .DataFrame ,
quantile_seuil :float =0.90 ,
jours_avant :int =3 ,jours_apres :int =15 )->ResilienceResult |None :
    """Étudie la réaction du MID de marché autour des journées a fort flux net.

    Plutot que de raisonner transaction par transaction (les gros trades acheteurs et
    vendeurs d'une même journée se compensent), on agrege le flux net par jour : la
    somme algébrique des montants, comptee positivement pour les achats clients de USD
    (pression haussiere) et negativement pour les ventes. On retient les journées dont
    le flux net absolu depasse un seuil eleve (par defaut le 9e décile).

    Pour chaque telle journée, on aligne la trajectoire du MID sur le jour de
    l'événement (t = 0), exprimee en écart de pips par rapport au MID de la veille, et
    orientée dans le sens du flux (un flux acheteur pousse le prix a la hausse). On
    moyenne toutes ces trajectoires : la courbe revele l'impact immédiat du déséquilibre
    puis la vitesse de retour a la normale — c'est la mesure empirique de la résilience
    du marché.
    """
    if transactions .empty or price_history .empty :
        return None 

    px =price_history .copy ()
    px ["date"]=pd .to_datetime (px ["date"])
    px =px .sort_values ("date").reset_index (drop =True )
    px ["idx"]=range (len (px ))

    # Flux net journalier (achats clients - ventes clients), hors debouclage interbancaire
    tx =transactions .copy ()
    tx =tx [tx ["client"]!="Interbancaire"]
    tx ["date_j"]=pd .to_datetime (tx ["date"]).dt .normalize ()
    tx ["signe"]=np .where (tx ["sens"]=="Achete",1.0 ,-1.0 )# achat client -> +
    tx ["flux"]=tx ["signe"]*tx ["montant"]
    flux_jour =tx .groupby ("date_j")["flux"].sum ().reset_index ()

    seuil =flux_jour ["flux"].abs ().quantile (quantile_seuil )
    gros_jours =flux_jour [flux_jour ["flux"].abs ()>=seuil ]
    if gros_jours .empty :
        return None 

    date_to_idx =dict (zip (px ["date"],px ["idx"]))
    pip =10_000.0 
    trajectoires =[]
    for _ ,j in gros_jours .iterrows ():
        d =j ["date_j"]
        if d not in date_to_idx :
            cand =px [px ["date"]>=d ]
            if cand .empty :
                continue 
            i0 =int (cand .iloc [0 ]["idx"])
        else :
            i0 =date_to_idx [d ]
        if i0 -jours_avant <0 or i0 +jours_apres >=len (px ):
            continue 

        ref =px .loc [i0 -1 ,"mid_marche"]
        signe =1.0 if j ["flux"]>0 else -1.0 # oriente dans le sens du flux
        for t in range (-jours_avant ,jours_apres +1 ):
            ecart =(px .loc [i0 +t ,"mid_marche"]-ref )*pip *signe 
            trajectoires .append ({"t":t ,"ecart_pips":ecart })

    if not trajectoires :
        return None 

    df =pd .DataFrame (trajectoires )
    courbe =df .groupby ("t").agg (ecart_pips =("ecart_pips","mean"),
    n =("ecart_pips","size")).reset_index ()

    apres =courbe [courbe ["t"]>=0 ].reset_index (drop =True )
    impact_immediat =float (apres .loc [apres ["t"]==0 ,"ecart_pips"].iloc [0 ])
    impact_max =float (apres ["ecart_pips"].max ())

    demi_vie =float ("nan")
    if impact_immediat >0 :
        seuil_moitie =impact_immediat /2 
        sous =apres [(apres ["t"]>0 )&(apres ["ecart_pips"]<=seuil_moitie )]
        if not sous .empty :
            demi_vie =float (sous .iloc [0 ]["t"])

    ecart_5j =apres .loc [apres ["t"]==min (5 ,int (apres ["t"].max ())),"ecart_pips"]
    taux_resorption_5j =(float (1 -ecart_5j .iloc [0 ]/impact_immediat )*100 
    if impact_immediat >0 and not ecart_5j .empty else float ("nan"))

    return ResilienceResult (
    n_evenements =len (gros_jours ),
    seuil_montant =float (seuil ),
    impact_immediat_pips =impact_immediat ,
    impact_max_pips =impact_max ,
    demi_vie_jours =demi_vie ,
    taux_resorption_5j =taux_resorption_5j ,
    courbe =courbe ,
    )


    # =========================================================================== #
    # B. Etude des depassements de limite
    # =========================================================================== #

@dataclass 
class BreachResult :
    n_episodes :int 
    duree_mediane_jours :float 
    pic_median_pct :float 
    pic_max_pct :float 
    courbe :pd .DataFrame # trajectoire moyenne de la position autour du breach


def _position_journaliere (transactions :pd .DataFrame )->pd .DataFrame :
    """Position de pic intraday du desk par jour de bourse.

    On suit la position cumulée opération par opération, et on retient pour chaque
    jour la position dont la valeur absolue est la plus elevee (le pic intraday).
    C'est cette position de pointe, et non celle de clôture, qui declenche les
    dépassements de limite : le desk peut passer la journée très expose puis
    déboucler en fin de séance, la clôture ne refletant alors pas le risque porte.
    """
    tx =transactions .copy ()
    tx ["horodatage"]=pd .to_datetime (tx ["date"])
    tx ["date_j"]=tx ["horodatage"].dt .normalize ()
    tx ["signe"]=np .where (tx ["sens"]=="Vend",1.0 ,-1.0 )# Vend client -> desk long
    tx ["flux"]=tx ["signe"]*tx ["montant"]
    tx =tx .sort_values ("horodatage")
    tx ["position"]=tx ["flux"].cumsum ()
    # Pic intraday = position de valeur absolue maximale dans la journee
    idx_pic =tx .groupby ("date_j")["position"].apply (lambda s :s .abs ().idxmax ())
    pics =tx .loc [idx_pic .values ,["date_j","position"]].rename (columns ={"date_j":"date"})
    return pics .sort_values ("date").reset_index (drop =True )


def etude_depassements (transactions :pd .DataFrame ,breaches :pd .DataFrame ,
limite :float =20_000_000.0 ,
jours_avant :int =5 ,jours_apres :int =15 )->BreachResult |None :
    """Analyse la position du desk autour des episodes de dépassement de limite.

    On reconstitue la position de clôture quotidienne, puis on aligne sa trajectoire
    sur chaque debut d'episode de dépassement (t = 0). La courbe moyenne, exprimee en
    pourcentage de la limite, montre la montee en charge avant le franchissement et
    la vitesse de retour dans les limites après.
    """
    if transactions .empty or breaches is None or breaches .empty :
        return None 

    pos =_position_journaliere (transactions )
    pos =pos .sort_values ("date").reset_index (drop =True )
    pos ["idx"]=range (len (pos ))
    pos ["pct"]=pos ["position"].abs ()/limite *100 
    date_to_idx =dict (zip (pos ["date"],pos ["idx"]))

    br =breaches .copy ()
    br ["date"]=pd .to_datetime (br ["horodatage"]).dt .normalize ()

    trajectoires =[]
    durees =[]
    pics =[]
    for _ ,b in br .iterrows ():
        d =b ["date"]
        # jour de position le plus proche
        cand =pos [pos ["date"]>=d ]
        if cand .empty :
            continue 
        i0 =int (cand .iloc [0 ]["idx"])
        if i0 -jours_avant <0 or i0 +jours_apres >=len (pos ):
            continue 

        for t in range (-jours_avant ,jours_apres +1 ):
            trajectoires .append ({"t":t ,"pct":pos .loc [i0 +t ,"pct"]})

            # Duree de l'episode : nombre de jours consecutifs au-dessus de 80 %
        duree =0 
        for t in range (0 ,jours_apres +1 ):
            if pos .loc [i0 +t ,"pct"]>=80 :
                duree +=1 
            else :
                break 
        durees .append (duree )
        pics .append (float (b ["pct"]))

    if not trajectoires :
        return None 

    df =pd .DataFrame (trajectoires )
    courbe =df .groupby ("t").agg (pct =("pct","mean"),n =("pct","size")).reset_index ()

    return BreachResult (
    n_episodes =len (pics ),
    duree_mediane_jours =float (np .median (durees ))if durees else float ("nan"),
    pic_median_pct =float (np .median (pics ))if pics else float ("nan"),
    pic_max_pct =float (np .max (pics ))if pics else float ("nan"),
    courbe =courbe ,
    )


    # =========================================================================== #
    # C. Persistance du comportement client
    # =========================================================================== #

@dataclass 
class PersistanceResult :
    n_paires :int 
    taux_apres_acceptation :float # P(accepte | precedente acceptee)
    taux_apres_refus :float # P(accepte | precedente refusee)
    momentum :float # difference des deux taux (points de %)
    p_value :float # test du khi-deux d'independance
    significatif :bool 
    matrice :pd .DataFrame # matrice de transition 2x2


def etude_persistance_client (rfq_log :pd .DataFrame )->PersistanceResult |None :
    """Teste si l'acceptation d'une cotation dépend de l'issue de la précédente.

    Pour chaque client, on ordonne ses demandes dans le temps et on forme les paires
    (issue précédente, issue courante). On estimé alors deux probabilites : celle
    d'accepter après une acceptation, et celle d'accepter après un refus. Un écart
    marque entre les deux revele un « momentum » comportemental : le passe recent du
    client informe sur sa decision presente, ce qui a une portee directe pour le
    pricing (un client en série d'acceptations peut supporter un spread plus large).

    L'independance est testee par un khi-deux sur la matrice de transition.
    """
    if rfq_log is None or rfq_log .empty :
        return None 
    if not {"client","horodatage","acceptee"}.issubset (rfq_log .columns ):
        return None 

    df =rfq_log .copy ()
    df ["horodatage"]=pd .to_datetime (df ["horodatage"])
    df =df .sort_values (["client","horodatage"])

    # Issue precedente au sein de chaque client
    df ["prec"]=df .groupby ("client")["acceptee"].shift (1 )
    paires =df .dropna (subset =["prec"])
    if len (paires )<30 :
        return None 

    prec =paires ["prec"].astype (int ).to_numpy ()
    cur =paires ["acceptee"].astype (int ).to_numpy ()

    # Matrice de transition (lignes = issue precedente, colonnes = issue courante)
    n11 =int (((prec ==1 )&(cur ==1 )).sum ())
    n10 =int (((prec ==1 )&(cur ==0 )).sum ())
    n01 =int (((prec ==0 )&(cur ==1 )).sum ())
    n00 =int (((prec ==0 )&(cur ==0 )).sum ())

    apres_acc =n11 /(n11 +n10 )*100 if (n11 +n10 )else float ("nan")
    apres_ref =n01 /(n01 +n00 )*100 if (n01 +n00 )else float ("nan")

    # Test du khi-deux d'independance
    try :
        from scipy .stats import chi2_contingency 
        table =np .array ([[n11 ,n10 ],[n01 ,n00 ]])
        _ ,p_val ,_ ,_ =chi2_contingency (table )
    except Exception :
        p_val =float ("nan")

    matrice =pd .DataFrame (
    [[n11 ,n10 ],[n01 ,n00 ]],
    index =["Precedente acceptee","Precedente refusee"],
    columns =["Accepte","Refuse"],
    )

    return PersistanceResult (
    n_paires =len (paires ),
    taux_apres_acceptation =apres_acc ,
    taux_apres_refus =apres_ref ,
    momentum =apres_acc -apres_ref ,
    p_value =float (p_val ),
    significatif =bool (p_val <0.05 )if not np .isnan (p_val )else False ,
    matrice =matrice ,
    )
