"""CFG BANK — USDMAD PRICER TERMINAL

Point d'entree Streamlit. Assemble la logique metier pure (core/) et
l'habillage terminal Bloomberg x CFG Bank (ui/) en 11 pages pilotees par
st.session_state (voir Section UI 3 du cahier des charges).
"""
from __future__ import annotations 

import csv 
from datetime import date ,datetime 
from pathlib import Path 

import pandas as pd 
import streamlit as st 

import math 

import numpy as np 

from core import (anomaly ,attribution ,backtest ,db ,events ,flow ,forward ,pnl ,portfolio ,
rfq ,risk ,segmentation ,stress ,timeseries ,vol_models )
from core .models import (
CalibrationParams ,
calculer_etat_pricer ,
kappa_ac ,
position_a_date ,
position_cumulee ,
trajectoire_almgren_chriss ,
volatilite ,
)
from ui import charts ,components ,theme 
from ui .components import badge_html ,dense_table ,fmt_amount ,fmt_pct ,fmt_price ,metric_card 

ROOT =Path (__file__ ).resolve ().parent 
EXPORTS_DIR =ROOT /"exports"
AUDIT_LOG_PATH =ROOT /"audit_log.csv"
AUDIT_LOG_FIELDS =[
"horodatage","date_cloture","mid_theorique","mid_marche","ecart_pct",
"borne_basse","borne_haute","ask_final","bid_final",
"statut_ask","statut_bid","statut_ecart_theorique",
]

QUOTES_SEED =[
{"bank":"Attijariwafa Bank","code":"ATWT","bid":9.3508 ,"ask":9.3558 },
{"bank":"BMCE/Bank of Africa","code":"BMCZ","bid":9.3509 ,"ask":9.3559 },
{"bank":"CIH Bank","code":"CIHX","bid":9.3509 ,"ask":9.3559 },
{"bank":"CNCA/Credit Agricole","code":"GCAM","bid":9.3509 ,"ask":9.3559 },
{"bank":"CDG Capital","code":"CDGK","bid":9.3509 ,"ask":9.3559 },
{"bank":"BCP","code":"BCPD","bid":9.3509 ,"ask":9.3559 },
{"bank":"BMCI","code":"BMCI","bid":9.3508 ,"ask":9.3558 },
{"bank":"Societe Generale","code":"AGMA","bid":9.3509 ,"ask":9.3559 },
{"bank":"CDM","code":"CDMB","bid":9.3510 ,"ask":9.3560 },
]


# --------------------------------------------------------------------------- #
# Configuration de page et initialisation
# --------------------------------------------------------------------------- #

st .set_page_config (
page_title ="CFG BANK — USDMAD PRICER TERMINAL",
page_icon ="💱",
layout ="wide",
initial_sidebar_state ="expanded",
)
theme .inject_css ()
db .init_db ()
db .seed_rfq_log_if_empty ()
db .seed_eurmad_if_empty (db .get_price_history ())
EXPORTS_DIR .mkdir (exist_ok =True )

if "quotes"not in st .session_state :
    st .session_state ["quotes"]=pd .DataFrame (QUOTES_SEED )
if "page_active"not in st .session_state :
    st .session_state ["page_active"]="DASH"
if "eur_usd"not in st .session_state :
    st .session_state ["eur_usd"]=1.1430 
if "calib"not in st .session_state :
    st .session_state ["calib"]=CalibrationParams (position_limite =5_000_000.0 ,decalage_cible_pips =10.0 )
if "taux_mad"not in st .session_state :
    st .session_state ["taux_mad"]=0.0275 # taux monetaire MAD, a confirmer avec la salle
if "taux_usd"not in st .session_state :
    st .session_state ["taux_usd"]=0.0435 # taux monetaire USD
if "positions_terme"not in st .session_state :
    st .session_state ["positions_terme"]={"1M":0.0 ,"3M":0.0 ,"6M":0.0 ,"12M":0.0 }
if "position_eur"not in st .session_state :
    st .session_state ["position_eur"]=-1_500_000.0 
if "fonds_propres"not in st .session_state :
    st .session_state ["fonds_propres"]=3_000_000_000.0 # ordre de grandeur, a confirmer
if "horizon_t"not in st .session_state :
    st .session_state ["horizon_t"]=5 
if "custom_scenarios"not in st .session_state :
    st .session_state ["custom_scenarios"]=[]


def _fmt_df (df :pd .DataFrame ,price_cols =(),amount_cols =(),pct_cols =())->pd .DataFrame :
    out =df .copy ()
    for c in price_cols :
        out [c ]=out [c ].map (fmt_price )
    for c in amount_cols :
        out [c ]=out [c ].map (fmt_amount )
    for c in pct_cols :
        out [c ]=out [c ].map (fmt_pct )
    return out 


    # --------------------------------------------------------------------------- #
    # Section 13 & 17 : historisation quotidienne + journal d'audit
    # --------------------------------------------------------------------------- #

def run_historisation_quotidienne ()->None :
    today =date .today ().isoformat ()
    last_date =db .get_last_price_history_date ()
    if last_date ==today :
        return 

    from core .models import marche_stats 

    mid_marche_now ,_ =marche_stats (st .session_state ["quotes"])
    inserted =db .insert_price_history_if_new (today ,mid_marche_now )
    if not inserted :
        return 

        # Etat complet du pricer a la cloture du jour, historique mis a jour compris
    history =db .get_price_history ()
    transactions =db .get_transactions ()
    position_df =position_cumulee (transactions )
    position_now =float (position_df ["position"].iloc [-1 ])if not position_df .empty else 0.0 
    sigma_now =volatilite (history )
    etat =calculer_etat_pricer (
    st .session_state ["eur_usd"],st .session_state ["quotes"],position_now ,sigma_now ,st .session_state ["calib"]
    )

    export_path =EXPORTS_DIR /f"pricer_{today }.csv"
    pd .DataFrame (
    [
    {
    "date":today ,
    "eur_usd":etat .eur_usd ,
    "mid_theorique":etat .mid_theo ,
    "borne_basse":etat .borne_basse ,
    "borne_haute":etat .borne_haute ,
    "mid_marche":etat .mid_marche ,
    "spread_marche":etat .spread_marche ,
    "ecart_theorique_marche_pct":etat .ecart_theo_marche ,
    "position":etat .position ,
    "sigma":etat .sigma ,
    "gamma":etat .gamma ,
    "eta":etat .eta ,
    "lambda":etat .lambda_ ,
    "skew":etat .prix_final .skew ,
    "ask_final":etat .prix_final .ask_final ,
    "bid_final":etat .prix_final .bid_final ,
    "conforme_ecart":etat .conforme_ecart ,
    "conforme_ask":etat .conforme_ask ,
    "conforme_bid":etat .conforme_bid ,
    }
    ]
    ).to_csv (export_path ,index =False )

    write_header =not AUDIT_LOG_PATH .exists ()
    with open (AUDIT_LOG_PATH ,"a",newline ="",encoding ="utf-8")as f :
        writer =csv .DictWriter (f ,fieldnames =AUDIT_LOG_FIELDS )
        if write_header :
            writer .writeheader ()
        writer .writerow (
        {
        "horodatage":datetime .now ().isoformat (timespec ="seconds"),
        "date_cloture":today ,
        "mid_theorique":f"{etat .mid_theo :.4f}",
        "mid_marche":f"{etat .mid_marche :.4f}",
        "ecart_pct":f"{etat .ecart_theo_marche *100 :.3f}",
        "borne_basse":f"{etat .borne_basse :.4f}",
        "borne_haute":f"{etat .borne_haute :.4f}",
        "ask_final":f"{etat .prix_final .ask_final :.4f}",
        "bid_final":f"{etat .prix_final .bid_final :.4f}",
        "statut_ask":"CONFORME"if etat .conforme_ask else "HORS BANDE",
        "statut_bid":"CONFORME"if etat .conforme_bid else "HORS BANDE",
        "statut_ecart_theorique":"CONFORME"if etat .conforme_ecart else "HORS BANDE",
        }
        )


def generate_pdf_summary (etat ,calib =None )->bytes :
    """Recapitulatif PDF de l'état courant du pricer.

    Mise en page institutionnelle, pensee pour l'impression et pour servir de
    piece jointe : identité CFG Bank, cotation client mise en avant,
    visualisation de la bande réglementaire et de la consommation de limite,
    statuts de conformité explicites.

    Les polices integrees de fpdf2 sont encodees en latin-1 : les accents
    francais passent, mais pas les tirets cadratins ni les apostrophes
    typographiques, evites ici au profit de leurs equivalents ASCII.
    """
    from fpdf import FPDF 
    from fpdf .enums import XPos ,YPos 

    # Palette alignee sur le mode clair de l'application
    ROUGE =(200 ,16 ,46 )
    ENCRE =(26 ,26 ,26 )
    GRIS =(95 ,99 ,104 )
    GRIS_CLAIR =(216 ,216 ,210 )
    FOND_LIGNE =(247 ,247 ,245 )
    VERT =(15 ,157 ,88 )
    AMBRE =(178 ,106 ,0 )
    ROUGE_ALERTE =(217 ,48 ,37 )

    MG =15.0 # marge laterale
    W =210.0 -2 *MG # largeur utile

    class Doc (FPDF ):
        def footer (self ):
            self .set_y (-15 )
            self .set_draw_color (*GRIS_CLAIR )
            self .set_line_width (0.2 )
            self .line (MG ,self .get_y (),MG +W ,self .get_y ())
            self .set_y (self .get_y ()+1.5 )
            self .set_font ("Helvetica","",7 )
            self .set_text_color (*GRIS )
            self .cell (W *0.78 ,4 ,
            "Prototype pédagogique - Salle des Marchés, CFG Bank. "
            "Cotations concurrentes figées, hors flux de marché temps réel.")
            self .cell (W *0.22 ,4 ,f"Page {self .page_no ()}",align ="R")

    pdf =Doc (format ="A4")
    pdf .set_auto_page_break (auto =True ,margin =20 )
    pdf .set_margins (MG ,12 ,MG )
    pdf .add_page ()

    # ------------------------------------------------------------------ #
    # Bandeau d'en-tete
    # ------------------------------------------------------------------ #
    pdf .set_fill_color (*ROUGE )
    pdf .rect (0 ,0 ,210 ,3.5 ,style ="F")# filet d'accent pleine page

    y0 =12.0 
    logo =Path (__file__ ).resolve ().parent /"assets"/"logo-cfg.png"
    if logo .exists ():
        pdf .image (str (logo ),x =MG ,y =y0 ,w =32 )# ratio d'origine preserve

    pdf .set_xy (MG +38 ,y0 -0.5 )
    pdf .set_font ("Helvetica","B",15 )
    pdf .set_text_color (*ENCRE )
    pdf .cell (0 ,6 ,"USDMAD PRICER TERMINAL",new_x =XPos .LMARGIN ,new_y =YPos .NEXT )
    pdf .set_xy (MG +38 ,pdf .get_y ())
    pdf .set_font ("Helvetica","",8.5 )
    pdf .set_text_color (*GRIS )
    pdf .cell (0 ,5 ,"Récapitulatif de cotation - Salle des Marchés",
    new_x =XPos .LMARGIN ,new_y =YPos .NEXT )

    # Bloc metadonnees, aligne a droite
    pdf .set_xy (MG ,y0 )
    pdf .set_font ("Helvetica","B",8.5 )
    pdf .set_text_color (*ROUGE )
    pdf .cell (W ,5 ,"USDMAD CFGB Curncy",align ="R",new_x =XPos .LMARGIN ,new_y =YPos .NEXT )
    pdf .set_x (MG )
    pdf .set_font ("Helvetica","",8 )
    pdf .set_text_color (*GRIS )
    pdf .cell (W ,4.5 ,datetime .now ().strftime ("Génère le %d/%m/%Y a %H:%M:%S"),
    align ="R",new_x =XPos .LMARGIN ,new_y =YPos .NEXT )

    pdf .set_y (max (pdf .get_y (),y0 +13 ))
    pdf .set_draw_color (*ENCRE )
    pdf .set_line_width (0.5 )
    pdf .line (MG ,pdf .get_y (),MG +W ,pdf .get_y ())
    pdf .ln (4 )

    # ------------------------------------------------------------------ #
    # Helpers de mise en page
    # ------------------------------------------------------------------ #
    zebra ={"i":0 }

    def section (titre :str )->None :
        pdf .ln (2.2 )
        pdf .set_font ("Helvetica","B",8.5 )
        pdf .set_text_color (*ROUGE )
        pdf .cell (0 ,4.2 ,titre .upper (),new_x =XPos .LMARGIN ,new_y =YPos .NEXT )
        y =pdf .get_y ()+0.2 
        pdf .set_draw_color (*ROUGE )
        pdf .set_line_width (0.35 )
        pdf .line (MG ,y ,MG +W ,y )
        pdf .ln (1.7 )
        zebra ["i"]=0 

    def row (label :str ,value :str ,accent :bool =False )->None :
        h =5.7 
        y =pdf .get_y ()
        if zebra ["i"]%2 ==0 :
            pdf .set_fill_color (*FOND_LIGNE )
            pdf .rect (MG ,y ,W ,h ,style ="F")
        zebra ["i"]+=1 
        pdf .set_xy (MG +2.5 ,y )
        pdf .set_font ("Helvetica","",9 )
        pdf .set_text_color (*GRIS )
        pdf .cell (W *0.58 ,h ,label )
        pdf .set_font ("Helvetica","B"if accent else "",9 )
        pdf .set_text_color (*(ROUGE if accent else ENCRE ))
        pdf .cell (W *0.42 -5 ,h ,value ,align ="R",new_x =XPos .LMARGIN ,new_y =YPos .NEXT )

    def badge (x :float ,y :float ,largeur :float ,ok :bool ,
    ok_txt :str ="CONFORME",bad_txt :str ="HORS BANDE")->None :
        txt =ok_txt if ok else bad_txt 
        coul =VERT if ok else ROUGE_ALERTE 
        pdf .set_font ("Helvetica","B",7 )
        tw =pdf .get_string_width (txt )+7 
        bx =x +(largeur -tw )/2 
        pdf .set_fill_color (*coul )
        pdf .rect (bx ,y ,tw ,4.6 ,style ="F")
        pdf .set_xy (bx ,y )
        pdf .set_text_color (255 ,255 ,255 )
        pdf .cell (tw ,4.6 ,txt ,align ="C")

        # ------------------------------------------------------------------ #
        # Cotation client : le bloc que le lecteur doit voir en premier
        # ------------------------------------------------------------------ #
    section ("Cotation client")

    box_h ,gap =24.0 ,6.0 
    box_w =(W -gap )/2 
    y_box =pdf .get_y ()

    for i ,(lib ,val ,conf )in enumerate ((
    ("BID  (CFG achète USD)",etat .prix_final .bid_final ,etat .conforme_bid ),
    ("ASK  (CFG vend USD)",etat .prix_final .ask_final ,etat .conforme_ask ),
    )):
        x =MG +i *(box_w +gap )
        pdf .set_draw_color (*GRIS_CLAIR )
        pdf .set_line_width (0.3 )
        pdf .set_fill_color (255 ,255 ,255 )
        pdf .rect (x ,y_box ,box_w ,box_h ,style ="DF")
        pdf .set_fill_color (*ROUGE )
        pdf .rect (x ,y_box ,1.3 ,box_h ,style ="F")# liseré de marque

        pdf .set_xy (x ,y_box +3 )
        pdf .set_font ("Helvetica","B",7.5 )
        pdf .set_text_color (*GRIS )
        pdf .cell (box_w ,4 ,lib ,align ="C")

        pdf .set_xy (x ,y_box +7.5 )
        pdf .set_font ("Helvetica","B",20 )
        pdf .set_text_color (*ENCRE )
        pdf .cell (box_w ,11 ,f"{val :.4f}",align ="C")

        badge (x ,y_box +18.2 ,box_w ,conf )

    pdf .set_y (y_box +box_h +3 )

    demi_spread_pips =(etat .spread_marche *etat .mid_marche )/2 *10_000 
    pdf .set_font ("Helvetica","",8 )
    pdf .set_text_color (*GRIS )
    pdf .cell (0 ,4.5 ,
    f"Spread affiché : {demi_spread_pips *2 :.1f} pips   |   "
    f"Skew de position appliqué : {etat .prix_final .skew *10_000 :+.2f} pips   |   "
    f"Prix de réservation : {etat .prix_final .prix_reservation :.4f}",
    new_x =XPos .LMARGIN ,new_y =YPos .NEXT )

    # ------------------------------------------------------------------ #
    # Bande reglementaire
    # ------------------------------------------------------------------ #
    section ("Bande de fluctuation réglementaire (+/- 5 %)")

    bar_h =7.5 
    y_bar =pdf .get_y ()
    pdf .set_fill_color (*FOND_LIGNE )
    pdf .set_draw_color (*GRIS_CLAIR )
    pdf .set_line_width (0.3 )
    pdf .rect (MG ,y_bar ,W ,bar_h ,style ="DF")

    span =etat .borne_haute -etat .borne_basse 

    def _x (valeur :float )->float :
        frac =0.5 if span <=0 else max (0.0 ,min (1.0 ,(valeur -etat .borne_basse )/span ))
        return MG +frac *W 

    for valeur ,coul in ((etat .mid_theo ,ENCRE ),(etat .mid_marche ,ROUGE )):
        xm =_x (valeur )
        pdf .set_fill_color (*coul )
        pdf .rect (xm -0.55 ,y_bar -1.6 ,1.1 ,bar_h +3.2 ,style ="F")

    pdf .set_y (y_bar +bar_h +2.5 )
    pdf .set_font ("Helvetica","",7.5 )
    pdf .set_text_color (*GRIS )
    pdf .cell (W /2 ,4 ,f"{etat .borne_basse :.4f}   BORNE BASSE")
    pdf .cell (W /2 ,4 ,f"BORNE HAUTE   {etat .borne_haute :.4f}",align ="R",
    new_x =XPos .LMARGIN ,new_y =YPos .NEXT )

    pdf .set_font ("Helvetica","B",7.5 )
    pdf .set_text_color (*ENCRE )
    pdf .cell (W /2 ,4.5 ,f"MID théorique  {etat .mid_theo :.4f}")
    pdf .set_text_color (*ROUGE )
    pdf .cell (W /2 ,4.5 ,f"MID de marché  {etat .mid_marche :.4f}",align ="R",
    new_x =XPos .LMARGIN ,new_y =YPos .NEXT )

    # ------------------------------------------------------------------ #
    # Donnees de marche
    # ------------------------------------------------------------------ #
    section ("Données de marché")
    row ("Cours EUR/USD (spot)",f"{etat .eur_usd :.4f}")
    row ("MID théorique (panier 60 % EUR / 40 % USD)",f"{etat .mid_theo :.4f}")
    row ("MID de marché (moyenne des concurrents)",f"{etat .mid_marche :.4f}")
    row ("Spread moyen de marché",f"{etat .spread_marche *100 :.4f} %")
    row ("Écart théorique / marché",f"{etat .ecart_theo_marche *100 :+.3f} %",
    accent =not etat .conforme_ecart )

    # ------------------------------------------------------------------ #
    # Position et calibration
    # ------------------------------------------------------------------ #
    section ("Position et calibration du risque")
    sens ="LONGUE"if etat .position >0 else ("COURTE"if etat .position <0 else "PLATE")
    row ("Position actuelle",f"{etat .position :,.0f} USD  ({sens })".replace (","," "))
    row ("Volatilité quotidienne (sigma)",f"{etat .sigma :.6f}")
    row ("Aversion au risque calibrée (gamma)",f"{etat .gamma :,.2f}".replace (","," "))
    row ("Impact de marché (eta)",f"{etat .eta :.6f}")
    row ("Aversion à l'exécution (lambda)",f"{etat .lambda_ :,.2f}".replace (","," "))

    if calib is not None and getattr (calib ,"position_limite",0 ):
        pct =abs (etat .position )/calib .position_limite *100 
        row ("Limite de position tolérée",f"{calib .position_limite :,.0f} USD".replace (","," "))
        row ("Consommation de la limite",f"{pct :.1f} %",
        accent =pct >=80 )

        pdf .ln (1.5 )
        g_h =5.5 
        y_g =pdf .get_y ()
        pdf .set_fill_color (*FOND_LIGNE )
        pdf .set_draw_color (*GRIS_CLAIR )
        pdf .rect (MG ,y_g ,W ,g_h ,style ="DF")
        frac =max (0.0 ,min (1.0 ,pct /150.0 ))
        coul =VERT if pct <80 else (AMBRE if pct <100 else ROUGE_ALERTE )
        if frac >0 :
            pdf .set_fill_color (*coul )
            pdf .rect (MG ,y_g ,W *frac ,g_h ,style ="F")
        pdf .set_draw_color (*GRIS )
        pdf .set_line_width (0.25 )
        for seuil in (80 ,100 ):
            xs =MG +W *seuil /150.0 
            pdf .line (xs ,y_g -1 ,xs ,y_g +g_h +1 )
        pdf .set_y (y_g +g_h +1.5 )
        pdf .set_font ("Helvetica","",7 )
        pdf .set_text_color (*GRIS )
        pdf .cell (0 ,4 ,"Reperes : 80 % (surveillance renforcée) et 100 % (limite atteinte)",
        new_x =XPos .LMARGIN ,new_y =YPos .NEXT )

        # ------------------------------------------------------------------ #
        # Indicateurs de risque
        # ------------------------------------------------------------------ #
    var_1j =risk .var_parametrique (etat .position ,etat .sigma ,1 )
    section ("Indicateurs de risque")
    row ("VaR 95 % a 1 jour",f"{var_1j .var_95 :,.0f} MAD".replace (","," "))
    row ("VaR 99 % a 1 jour",f"{var_1j .var_99 :,.0f} MAD".replace (","," "))
    row ("Valeur du pip sur la position",f"{risk .valeur_du_pip (etat .position ):,.0f} MAD".replace (","," "))

    # ------------------------------------------------------------------ #
    # Controles de conformite
    # ------------------------------------------------------------------ #
    section ("Contrôles de conformité")

    ctrl_h =11.5 
    ctrl_w =(W -2 *4 )/3 
    y_c =pdf .get_y ()
    for i ,(lib ,ok )in enumerate ((
    ("BID dans la bande",etat .conforme_bid ),
    ("ASK dans la bande",etat .conforme_ask ),
    ("Écart théorique < 5 %",etat .conforme_ecart ),
    )):
        x =MG +i *(ctrl_w +4 )
        pdf .set_draw_color (*GRIS_CLAIR )
        pdf .set_fill_color (255 ,255 ,255 )
        pdf .rect (x ,y_c ,ctrl_w ,ctrl_h ,style ="DF")
        pdf .set_xy (x ,y_c +2 )
        pdf .set_font ("Helvetica","",7.5 )
        pdf .set_text_color (*GRIS )
        pdf .cell (ctrl_w ,4 ,lib ,align ="C")
        badge (x ,y_c +6.2 ,ctrl_w ,ok )
    pdf .set_y (y_c +ctrl_h +3 )

    # ------------------------------------------------------------------ #
    # Note methodologique
    # ------------------------------------------------------------------ #
    section ("Note methodologique")
    pdf .set_font ("Helvetica","",7.5 )
    pdf .set_text_color (*GRIS )
    pdf .multi_cell (
    W ,3.8 ,
    "Le MID théorique est déduit du cours EUR/USD par la relation de panier retenue par Bank "
    "Al-Maghrib et définit la bande réglementaire de +/- 5 %. Le MID de marché, moyenne des "
    "cotations concurrentes, sert de référence de cotation ; il est décalé selon la position "
    "détenue, suivant le modèle de teneur de marché de Ho et Stoll (1981). Le coefficient gamma "
    "est calibré à partir d'un décalage cible à la position limite, et non fixé arbitrairement.",
    )

    return bytes (pdf .output ())


    # --------------------------------------------------------------------------- #
    # Pages
    # --------------------------------------------------------------------------- #

def render_dash (etat ,history :pd .DataFrame ,position_df :pd .DataFrame )->None :
    st .markdown ('<div class="section-title">DASH — Tableau de bord</div>',unsafe_allow_html =True )

    c1 ,c2 ,c3 ,c4 ,c5 =st .columns (5 )
    metric_card (c1 ,"EUR/USD",f"{etat .eur_usd :.4f}",key ="eurusd")
    metric_card (c2 ,"MID THÉORIQUE",f"{etat .mid_theo :.4f}",key ="midtheo")
    metric_card (
    c3 ,"MID MARCHÉ",f"{etat .mid_marche :.4f}",
    spark_values =history ["mid_marche"].tail (12 ).tolist (),spark_color =theme .AMBER ,key ="midmkt",
    )
    metric_card (
    c4 ,"POSITION ACTUELLE",f"{fmt_amount (etat .position )} USD",
    delta ="VENDEUR NET"if etat .position >0 else ("ACHETEUR NET"if etat .position <0 else "FLAT"),
    delta_kind ="up"if etat .position >0 else ("down"if etat .position <0 else "flat"),
    spark_values =position_df ["position"].tail (12 ).tolist ()if not position_df .empty else None ,
    spark_color =theme .CFG_RED ,key ="pos",
    )
    metric_card (
    c5 ,"ECART THÉO/MARCHÉ",fmt_pct (etat .ecart_theo_marche ),
    delta_kind ="up"if etat .ecart_theo_marche >=0 else "down",key ="ecart",
    )

    st .markdown ('<div class="section-title">BANDE DE FLUCTUATION RÉGLEMENTAIRE (&plusmn;5%)</div>',unsafe_allow_html =True )
    components .bande_fluctuation_bar (etat .borne_basse ,etat .borne_haute ,etat .mid_theo ,etat .mid_marche )

    st .markdown ('<div class="section-title">COTATION FINALE CFG BANK — <span class="instrument-code"><b>USDMAD CFGB Curncy</b></span></div>',unsafe_allow_html =True )
    q1 ,q2 =st .columns (2 )
    with q1 :
        st .markdown (
        f'<div class="quote-side"><div class="quote-side-label">BID FINAL</div>'
        f'<div class="quote-side-value">{etat .prix_final .bid_final :.4f}</div>'
        f'<div style="margin-top:8px;">{badge_html (etat .conforme_bid )}</div></div>',
        unsafe_allow_html =True ,
        )
    with q2 :
        st .markdown (
        f'<div class="quote-side"><div class="quote-side-label">ASK FINAL</div>'
        f'<div class="quote-side-value">{etat .prix_final .ask_final :.4f}</div>'
        f'<div style="margin-top:8px;">{badge_html (etat .conforme_ask )}</div></div>',
        unsafe_allow_html =True ,
        )

        # Cours croise EUR/MAD : deduit du USDMAD cote et du EUR/USD international.
        # L'euro pesant 60% du panier du dirham, cette paire est suivie au meme titre
        # que le USD/MAD sur un desk marocain.
    cross =risk .cross_eur_mad (etat .eur_usd ,etat .prix_final .bid_final ,etat .prix_final .ask_final )
    st .markdown ('<div class="section-title">COURS CROISÉ — <span class="instrument-code"><b>EURMAD CFGB Curncy</b></span></div>',
    unsafe_allow_html =True )
    x1 ,x2 ,x3 ,x4 =st .columns (4 )
    metric_card (x1 ,"EUR/MAD BID",f"{cross .eur_mad_bid :.4f}",key ="eurmadbid")
    metric_card (x2 ,"EUR/MAD ASK",f"{cross .eur_mad_ask :.4f}",key ="eurmadask")
    metric_card (x3 ,"EUR/MAD MID",f"{cross .eur_mad_mid :.4f}",key ="eurmadmid")
    metric_card (x4 ,"SPREAD",f"{cross .spread_pips :.1f} pips",key ="eurmadspr")
    st .caption ("EUR/MAD = EUR/USD x USD/MAD. Cotation déduite, héritée du skew de position appliqué au USD/MAD.")

    st .markdown ("<br>",unsafe_allow_html =True )
    pdf_bytes =generate_pdf_summary (etat ,st .session_state ["calib"])
    st .download_button (
    "EXPORTER RÉCAPITULATIF PDF",data =pdf_bytes ,
    file_name =f"pricer_cfg_{date .today ().isoformat ()}.pdf",mime ="application/pdf",
    )


def render_mkt (etat )->None :
    st .markdown ('<div class="section-title">MKT&lt;GO&gt; — Marché &amp; Concurrence</div>',unsafe_allow_html =True )
    st .caption ("Tableau éditable — modifiez bid/ask, le reste du pricer (et le ticker) se met a jour instantanément.")

    edited =st .data_editor (
    st .session_state ["quotes"],
    column_config ={
    "bank":st .column_config .TextColumn ("Banque",disabled =True ),
    "code":st .column_config .TextColumn ("Code",disabled =True ),
    "bid":st .column_config .NumberColumn ("Bid",format ="%.4f",step =0.0001 ),
    "ask":st .column_config .NumberColumn ("Ask",format ="%.4f",step =0.0001 ),
    },
    hide_index =True ,
    width='stretch' ,
    key ="quotes_editor",
    )
    st .session_state ["quotes"]=edited 

    from core .models import mid_spread_banque 

    rows =[]
    for _ ,r in edited .iterrows ():
        m ,s =mid_spread_banque (r ["bid"],r ["ask"])
        rows .append ({"BANQUE":r ["bank"],"CODE":r ["code"],"BID":r ["bid"],"ASK":r ["ask"],"MID":m ,"SPREAD":s })
    rows .append (
    {
    "BANQUE":"CFG Bank","CODE":"CFGB",
    "BID":etat .prix_final .bid_final ,"ASK":etat .prix_final .ask_final ,
    "MID":etat .mid_marche ,"SPREAD":etat .spread_marche ,
    }
    )
    disp =pd .DataFrame (rows )
    disp_fmt =_fmt_df (disp ,price_cols =["BID","ASK","MID"],pct_cols =["SPREAD"])
    dense_table (disp_fmt ,highlight_col ="CODE",highlight_value ="CFGB")

    st .markdown ('<div class="section-title">COMPARAISON DES MID PAR BANQUE</div>',unsafe_allow_html =True )
    fig =charts .mkt_comparison_bar (disp ["CODE"].tolist (),disp ["MID"].tolist (),cfg_label ="CFGB")
    st .plotly_chart (fig ,width='stretch' ,config ={"displayModeBar":False })

    # ---- Detection d'anomalies sur les cotations ---- #
    st .markdown ('<div class="section-title">Contrôle DE QUALITÉ DES COTATIONS</div>',unsafe_allow_html =True )
    c =theme .P ()
    st .caption (
    "Le MID de marché est une moyenne arithmétique : une seule cotation erronée -- saisie "
    "fautive, prix perimé, décimale mal placée -- se propage directement au prix affiché au "
    "client, puis à sigma et au skew. Le contrôle ci-dessous signale les cotations aberrantes "
    "et propose une estimation robuste."
    )

    demo =st .checkbox ("Simuler une cotation erronée, pour démonstration",key ="mkt_demo_anom")
    quotes_ctrl =st .session_state ["quotes"]
    if demo :
        dc1 ,dc2 =st .columns (2 )
        idx =dc1 .selectbox ("Banque affectée",list (range (len (quotes_ctrl ))),
        format_func =lambda i :str (quotes_ctrl .iloc [i ]["code"]),key ="mkt_anom_idx")
        dec =dc2 .slider ("Décalage introduit (pips)",-300.0 ,300.0 ,120.0 ,step =10.0 ,key ="mkt_anom_dec")
        quotes_ctrl =anomaly .injecter_anomalie (quotes_ctrl ,int (idx ),float (dec ))

    res_anom =anomaly .detecter (quotes_ctrl )
    if res_anom is None :
        st .info ("Au moins quatre cotations sont nécessaires pour un contrôle de qualité.")
        return 

    a1 ,a2 ,a3 =st .columns (3 )
    metric_card (a1 ,"COTATIONS SIGNALÉES",str (res_anom .n_anomalies ),
    delta_kind ="down"if res_anom .n_anomalies else "up",key ="an_n")
    metric_card (a2 ,"MID ROBUSTE",f"{res_anom .mid_moyenne_elaguee :.4f}",key ="an_rob")
    metric_card (a3 ,"ÉCART AU MID ACTUEL",f"{res_anom .ecart_pips :+.2f} pips",
    delta_kind ="down"if abs (res_anom .ecart_pips )>1 else "flat",key ="an_ec")

    st .plotly_chart (
    charts .anomalies_chart (quotes_ctrl ["code"].tolist (),
    res_anom .quotes ["score"].tolist (),res_anom .seuil ),
    width='stretch' ,config ={"displayModeBar":False })

    dense_table (anomaly .comparer_estimateurs (res_anom ))

    st .caption (
    f"Méthode : score z modifié, calculé à partir de la médiane et d'une mesure de dispersion "
    f"robuste (ici : {res_anom .methode_echelle .lower ()}), et non de la moyenne et de "
    "l'écart-type. Le motif est qu'un écart-type est lui-même contamine par les valeurs "
    "aberrantes qu'il devrait servir à détecter : une cotation très éloignée le gonfle au "
    "point de se faire passer pour normale. Sur ce marché étroit où les banques s'alignent "
    "souvent au centime près, l'écart absolu médian peut s'annuler ; la mesure de dispersion "
    "bascule alors automatiquement sur l'écart absolu moyen."
    )

    if res_anom .n_anomalies >0 :
        imp =anomaly .impact_sur_prix (res_anom ,etat .spread_marche )
        st .markdown (
        f'<div style="background:{c ["red_alert"]}18; border-left:3px solid {c ["red_alert"]}; '
        f'padding:10px 14px; font-family:{theme .FONT_MONO }; font-size:0.85rem;">'
        f'{res_anom .n_anomalies } cotation(s) signalée(s). Prix client selon la méthode retenue : '
        f'ASK {imp ["ask_moyenne"]:.4f} (moyenne) contre {imp ["ask_robuste"]:.4f} (robuste), '
        f'soit {abs (imp ["ecart_pips"]):.1f} pips d\'ecart.</div>',
        unsafe_allow_html =True )


def render_rfq (etat )->None :
    """Cotation d'un prix fermé sur demande client (Request for Quote).

    C'est l'opération élémentaire d'un desk de change : un client demande un
    prix pour un montant donné, la banque repond par un prix fermé. Le prix
    propose hérite du skew de position (Ho & Stoll) et s'élargit selon la
    tranche de montant (pricing ladder).
    """
    st .markdown ('<div class="section-title">RFQ&lt;GO&gt; — Cotation client (Request for Quote)</div>',
    unsafe_allow_html =True )

    c =theme .P ()
    calib :CalibrationParams =st .session_state ["calib"]

    q1 ,q2 ,q3 =st .columns ([1.1 ,1.4 ,1.2 ])
    sens =q1 .selectbox ("Sens (vu du client)",["Achete","Vend"],key ="rfq_sens")
    montant =q2 .number_input ("Montant (USD)",min_value =1_000.0 ,step =50_000.0 ,
    value =1_000_000.0 ,key ="rfq_montant")
    client_nom =q3 .text_input ("Contrepartie",value ="",placeholder ="Nom du client",key ="rfq_client")

    quote =rfq .coter (
    sens_client =sens ,
    montant_usd =montant ,
    prix_reservation =etat .prix_final .prix_reservation ,
    spread_marche =etat .spread_marche ,
    mid_marche =etat .mid_marche ,
    )

    verbe ="VEND"if sens =="Achete"else "ACHETE"
    components .rfq_ticket (
    headline =f"CFG {verbe } {fmt_amount (quote .montant_usd )} USD @ {fmt_price (quote .prix_client )}",
    lines =[
    ("Tranche",f"{quote .tier_label } (x{quote .tier_mult :.2f})"),
    ("Prix de réservation (skew inclus)",fmt_price (quote .prix_reservation )),
    ("Demi-spread écran",f"{quote .demi_spread_base *10_000 :.2f} pips"),
    ("Demi-spread appliqué",f"{quote .demi_spread_applique *10_000 :.2f} pips"),
    ("Prix écran (taille standard)",fmt_price (quote .prix_ecran )),
    ("PRIX FERME CLIENT",fmt_price (quote .prix_client )),
    ("Contrevaleur",f"{fmt_amount (quote .contrevaleur_mad )} MAD"),
    ("Marge théorique captée",f"{fmt_amount (quote .marge_estimee_mad )} MAD"),
    ("Valeur du pip",f"{risk .valeur_du_pip (quote .montant_usd ):,.0f} MAD".replace (","," ")),
    ],
    )

    # Impact du deal sur la position et sur la limite de risque
    pos_apres =rfq .position_apres_deal (etat .position ,sens ,montant )
    pct_apres =abs (pos_apres )/calib .position_limite *100 if calib .position_limite else 0.0 
    pct_avant =abs (etat .position )/calib .position_limite *100 if calib .position_limite else 0.0 

    st .markdown ('<div class="section-title">IMPACT SUR LA POSITION</div>',unsafe_allow_html =True )
    i1 ,i2 ,i3 =st .columns (3 )
    metric_card (i1 ,"POSITION AVANT",fmt_amount (etat .position ),
    delta =f"{pct_avant :.1f}% de la limite",key ="rfq_pos_av")
    metric_card (i2 ,"POSITION APRES",fmt_amount (pos_apres ),
    delta =f"{pct_apres :.1f}% de la limite",
    delta_kind ="down"if pct_apres >pct_avant else "up",key ="rfq_pos_ap")
    metric_card (i3 ,"VARIATION",fmt_amount (pos_apres -etat .position ),key ="rfq_pos_var")

    if pct_apres >=100 :
        st .markdown (
        f'<div style="background:{c ["red_alert"]}22; border-left:3px solid {c ["red_alert"]}; '
        f'padding:10px 14px; font-family:{theme .FONT_MONO }; color:{c ["red_alert"]};">'
        f'LIMITE DÉPASSÉE APRES CE DEAL ({pct_apres :.0f}%) — liquidation à prévoir (LIQD)</div>',
        unsafe_allow_html =True ,
        )
    elif pct_apres >=80 :
        st .markdown (
        f'<div style="background:{c ["amber"]}22; border-left:3px solid {c ["amber"]}; '
        f'padding:10px 14px; font-family:{theme .FONT_MONO }; color:{c ["amber"]};">'
        f'SURVEILLANCE RENFORCÉE APRES CE DEAL ({pct_apres :.0f}% de la limite)</div>',
        unsafe_allow_html =True ,
        )

        # Les deux issues d'une demande de cotation sont enregistrees : un refus est
        # aussi informatif qu'une acceptation, car sans les refus il est impossible
        # d'estimer la sensibilite au prix de la clientele (voir page FLOW).
    ex1 ,ex2 =st .columns ([1 ,1 ])
    if ex1 .button ("EXECUTER LE DEAL",type ="primary",key ="rfq_exec"):
        if not client_nom .strip ():
            st .warning ("Renseigner la contrepartie avant d'exécuter le deal.")
        else :
            db .insert_transaction (date .today ().isoformat (),client_nom .strip (),sens ,
            quote .montant_usd ,quote .prix_client )
            db .insert_rfq (datetime .now ().isoformat (timespec ="seconds"),client_nom .strip (),sens ,
            quote .montant_usd ,quote .demi_spread_applique ,quote .prix_client ,True )
            st .success (f"Deal enregistré : {sens } {fmt_amount (quote .montant_usd )} USD @ {fmt_price (quote .prix_client )}")
            st .rerun ()

    if ex2 .button ("COTATION REFUSÉE PAR LE CLIENT",key ="rfq_refus"):
        db .insert_rfq (datetime .now ().isoformat (timespec ="seconds"),
        client_nom .strip ()or "Non renseigné",sens ,
        quote .montant_usd ,quote .demi_spread_applique ,quote .prix_client ,False )
        st .info ("Refus enregistré — il alimentera la calibration du modèle d'intensité des flux (FLOW).")

    st .markdown ('<div class="section-title">GRILLE DE COTATION PAR TRANCHE</div>',unsafe_allow_html =True )
    grille =[]
    for label ,seuil ,mult in rfq .TIERS :
        demi =(etat .spread_marche *etat .mid_marche )/2.0 *mult 
        grille .append ({
        "TRANCHE":label ,
        "PLAFOND":"sans limite"if seuil ==float ("inf")else fmt_amount (seuil ),
        "MULT.":f"x{mult :.2f}",
        "DEMI-SPREAD":f"{demi *10_000 :.2f} pips",
        "BID":fmt_price (etat .prix_final .prix_reservation -demi ),
        "ASK":fmt_price (etat .prix_final .prix_reservation +demi ),
        })
    dense_table (pd .DataFrame (grille ),highlight_col ="TRANCHE",highlight_value =quote .tier_label )
    st .caption (
    "Le spread s'élargit avec la taille de l'opération : un montant important est plus "
    "couteux et plus risque à couvrir. Principe de segmentation par taille décrit par "
    "Barzykin, Bergault et Gueant (2021-2023) pour le market making de change."
    )


def render_pos (transactions :pd .DataFrame ,position_df :pd .DataFrame ,etat )->None :
    st .markdown ('<div class="section-title">POS&lt;GO&gt; — Position &amp; Transactions</div>',unsafe_allow_html =True )

    with st .form ("new_transaction",clear_on_submit =True ):
        f1 ,f2 ,f3 ,f4 ,f5 =st .columns ([2 ,1.4 ,1.2 ,1.4 ,1.2 ])
        tdate =f1 .date_input ("Date",value =date .today ())
        client =f2 .text_input ("Client")
        sens =f3 .selectbox ("Sens",["Vend","Achete"])
        montant =f4 .number_input ("Montant (USD)",min_value =0.0 ,step =10_000.0 ,value =100_000.0 )
        prix_manuel =f5 .number_input ("Prix exécution (optionnel)",min_value =0.0 ,step =0.0001 ,value =0.0 ,format ="%.4f")
        submitted =st .form_submit_button ("AJOUTER LA TRANSACTION",type ="primary")
        if submitted and client and montant >0 :
            db .insert_transaction (tdate .isoformat (),client ,sens ,montant ,prix_manuel if prix_manuel >0 else None )
            st .rerun ()

    st .markdown ('<div class="section-title">JOURNAL DE TRANSACTIONS</div>',unsafe_allow_html =True )
    if position_df .empty :
        st .info ("Aucune transaction enregistrée.")
    else :
        disp =position_df [["date","client","sens","montant","impact","position"]].rename (
        columns ={"date":"DATE","client":"CLIENT","sens":"SENS","montant":"MONTANT",
        "impact":"IMPACT","position":"POSITION"}
        )
        disp =_fmt_df (disp ,amount_cols =["MONTANT","IMPACT","POSITION"])
        dense_table (disp )

        csv_bytes =transactions .to_csv (index =False ).encode ("utf-8")
        cdl ,cpanic =st .columns ([1 ,2 ])
        with cdl :
            st .download_button ("EXPORTER CSV",data =csv_bytes ,file_name ="transactions_cfg.csv",mime ="text/csv")
        with cpanic :
            if st .button ("⚠ LIQUIDER LA POSITION VIA ALMGREN-CHRISS",type ="primary",width='stretch' ):
                st .session_state ["page_active"]="LIQD"
                st .session_state ["liqd_auto_position"]=abs (float (position_df ["position"].iloc [-1 ]))
                st .rerun ()

        st .markdown ('<div class="section-title">EVOLUTION DE LA POSITION</div>',unsafe_allow_html =True )
        fig =charts .area_chart (
        list (range (1 ,len (position_df )+1 )),position_df ["position"].tolist (),
        color =theme .CFG_RED ,name ="Position cumulee",
        )
        st .plotly_chart (fig ,width='stretch' ,config ={"displayModeBar":False })


@st .cache_data (show_spinner =False )
def _pnl_calcule_cache (n_tx :int ,derniere_date :str ,spread :float ,
position_limite :float ,decalage_cible :float ):
    """Calcul PNL mis en cache : evite de tout recalculer a chaque changement d'onglet.

    Les entrees (nombre de transactions, derniere date, spread, parametres de
    calibration) resument l'etat des donnees ; tant qu'elles ne changent pas, le
    resultat est identique, donc le cache Streamlit renvoie directement le calcul
    precedent au lieu de rejouer les ~500 pricings point-in-time. C'est ce qui rend
    la page PNL instantanee apres le premier affichage.
    """
    transactions =db .get_transactions ()
    history =db .get_price_history ()
    calib =CalibrationParams (position_limite =position_limite ,decalage_cible_pips =decalage_cible )
    priced =pnl .resolve_execution_prices (transactions ,history ,calib ,spread )
    daily =pnl .compute_daily_pnl (priced ,history ,transactions )
    return priced ,daily 


def render_pnl (transactions :pd .DataFrame ,history :pd .DataFrame ,etat )->None :
    st .markdown ('<div class="section-title">PNL&lt;GO&gt; — Suivi du Profit &amp; Loss</div>',unsafe_allow_html =True )

    if transactions .empty :
        st .info ("Aucune transaction enregistrée.")
        return 

    calib =st .session_state ["calib"]
    derniere_date =str (history ["date"].iloc [-1 ])if not history .empty else ""
    priced ,daily =_pnl_calcule_cache (len (transactions ),derniere_date ,etat .spread_marche ,
    calib .position_limite ,calib .decalage_cible_pips )
    pnl_total ,cash_cumule =pnl .compute_pnl_total (priced ,etat .position ,etat .mid_marche )
    pnl_du_jour =float (daily ["pnl_du_jour"].iloc [-1 ])if not daily .empty else 0.0 

    c1 ,c2 ,c3 =st .columns (3 )
    metric_card (c1 ,"P&L TOTAL (MAD)",fmt_amount (pnl_total ),
    delta_kind ="up"if pnl_total >=0 else "down",key ="pnltotal")
    metric_card (c2 ,"P&L DU JOUR (MAD)",fmt_amount (pnl_du_jour ),
    delta_kind ="up"if pnl_du_jour >=0 else "down",key ="pnljour")
    metric_card (c3 ,"POSITION VALORISEE (MAD)",fmt_amount (etat .position *etat .mid_marche ),key ="posval")

    st .markdown ('<div class="section-title">EVOLUTION DU P&amp;L Cumulé</div>',unsafe_allow_html =True )
    if not daily .empty :
        fig =charts .pnl_area_chart (daily ["date"].tolist (),daily ["pnl_total"].tolist ())
        st .plotly_chart (fig ,width='stretch' ,config ={"displayModeBar":False })

    st .markdown ('<div class="section-title">DETAIL PAR TRANSACTION</div>',unsafe_allow_html =True )
    disp =priced [["date","client","sens","montant","prix_execution_effectif","mid_marche_moment","ecart_spread_capture","gain_spread_mad"]].rename (
    columns ={
    "date":"DATE","client":"CLIENT","sens":"SENS","montant":"MONTANT",
    "prix_execution_effectif":"PRIX EXEC.","mid_marche_moment":"MID MARCHE",
    "ecart_spread_capture":"ECART CAPTURE","gain_spread_mad":"GAIN SPREAD (MAD)",
    }
    )
    disp =_fmt_df (disp ,amount_cols =["MONTANT","GAIN SPREAD (MAD)"],price_cols =["PRIX EXEC.","MID MARCHE","ECART CAPTURE"])
    dense_table (disp )


def render_risk (etat )->None :
    st .markdown ('<div class="section-title">RISK&lt;GO&gt; — Limites de risque &amp; alertes</div>',unsafe_allow_html =True )

    calib :CalibrationParams =st .session_state ["calib"]
    pct =abs (etat .position )/calib .position_limite *100 if calib .position_limite else 0 

    # ---- VaR : traduit la position en perte potentielle chiffree ----------- #
    st .markdown ('<div class="section-title">VALUE AT RISK (VaR PARAMETRIQUE)</div>',unsafe_allow_html =True )
    h1 ,_ =st .columns ([1 ,3 ])
    horizon =h1 .selectbox ("Horizon (jours)",[1 ,5 ,10 ],index =0 ,key ="var_horizon")
    var =risk .var_parametrique (etat .position ,etat .sigma ,horizon )

    v1 ,v2 ,v3 =st .columns (3 )
    metric_card (v1 ,f"VaR 95% — {horizon }J",f"{var .var_95 :,.0f} MAD".replace (","," "),key ="var95")
    metric_card (v2 ,f"VaR 99% — {horizon }J",f"{var .var_99 :,.0f} MAD".replace (","," "),key ="var99")
    metric_card (v3 ,"Écart-TYPE DU P&L",f"{var .ecart_type_pnl :,.0f} MAD".replace (","," "),key ="varsd")

    st .plotly_chart (
    charts .var_distribution_chart (
    risk .simuler_pnl (etat .position ,etat .sigma ,horizon ),var .var_95 ,var .var_99 
    ),
    width='stretch' ,config ={"displayModeBar":False },
    )
    st .caption (
    f"Lecture : dans 95% des cas, la perte à {horizon } jour(s) sur la position actuelle ne devrait "
    f"pas dépasser {var .var_95 :,.0f} MAD. Hypothèse de normalité des variations, cohérente avec "
    "celle des modèles de Ho & Stoll et d'Almgren & Chriss retenus ici. Limite : la VaR ne dit rien "
    "de l'ampleur des pertes au-dela du seuil, et suppose une volatilité stable.".replace (","," ")
    )

    st .markdown ('<div class="section-title">POSITION VS LIMITE</div>',unsafe_allow_html =True )
    st .markdown ("**Position actuelle en % de la position limite tolérée**")
    fig =charts .gauge_thermometer (pct )
    st .plotly_chart (fig ,width='stretch' ,config ={"displayModeBar":False })
    st .markdown (
    f'<div style="font-family:{theme .FONT_MONO }; font-size:1.1rem;">{pct :.1f}% '
    f'{badge_html (pct <80 ,ok_text ="SOUS SEUIL",bad_text ="SURVEILLANCE / DÉPASSEMENT")}</div>',
    unsafe_allow_html =True ,
    )

    st .markdown ('<div class="section-title">LIMITES CONFIGURÉES</div>',unsafe_allow_html =True )
    l1 ,l2 =st .columns (2 )
    new_limite =l1 .number_input ("Position limite (USD)",min_value =100_000.0 ,step =100_000.0 ,value =calib .position_limite )
    new_decalage =l2 .number_input ("Décalage cible (pips)",min_value =0.1 ,step =0.5 ,value =calib .decalage_cible_pips )
    if new_limite !=calib .position_limite or new_decalage !=calib .decalage_cible_pips :
        st .session_state ["calib"]=CalibrationParams (position_limite =new_limite ,decalage_cible_pips =new_decalage )
        st .rerun ()
    st .caption ("Seuil d'alerte fixe : 80% de la position limite (bandeau orange), 100% = bandeau rouge + Panic Mode.")

    if pct >=80 :
        today_prefix =date .today ().isoformat ()
        existing =db .get_risk_breaches ()
        already_logged =(not existing .empty )and (existing ["horodatage"].str .startswith (today_prefix ).any ())
        if not already_logged :
            db .log_risk_breach (datetime .now ().isoformat (timespec ="seconds"),etat .position ,calib .position_limite ,pct )

    st .markdown ('<div class="section-title">HISTORIQUE DES DÉPASSEMENTS DE SEUIL</div>',unsafe_allow_html =True )
    breaches =db .get_risk_breaches ()
    if breaches .empty :
        st .info ("Aucun dépassement de seuil enregistré.")
    else :
        disp =breaches [["horodatage","position","limite","pct"]].rename (
        columns ={"horodatage":"HORODATAGE","position":"POSITION","limite":"LIMITE","pct":"% LIMITE"}
        )
        disp ["POSITION"]=disp ["POSITION"].map (fmt_amount )
        disp ["LIMITE"]=disp ["LIMITE"].map (fmt_amount )
        disp ["% LIMITE"]=disp ["% LIMITE"].map (lambda x :f"{x :.1f}%")
        dense_table (disp )


def render_vol (etat ,history :pd .DataFrame )->None :
    st .markdown ('<div class="section-title">VOL&lt;GO&gt; — Volatilité &amp; Calibration</div>',unsafe_allow_html =True )

    c1 ,c2 =st .columns ([1 ,2 ])
    with c1 :
        metric_card (c1 ,"SIGMA (&sigma;)",f"{etat .sigma :.6f}",key ="sigma")
    with c2 :
        st .caption ("&sigma; = écart-type des variations quotidiennes du MID de marché, sur tout l'historique disponible.")

    st .markdown ('<div class="section-title">HISTORIQUE DU MID DE Marché</div>',unsafe_allow_html =True )
    disp =history .rename (columns ={"date":"DATE","mid_marche":"MID MARCHE"})[["DATE","MID MARCHE"]]
    disp =_fmt_df (disp ,price_cols =["MID MARCHE"])
    dense_table (disp )

    st .markdown ('<div class="section-title">CALIBRATION DE &gamma; (GAMMA)</div>',unsafe_allow_html =True )
    st .markdown (
    "```\nk_cible = Décalage_cible / (Position_limite / 1 000 000)\ngamma = k_cible / sigma^2\n```"
    )
    calib :CalibrationParams =st .session_state ["calib"]
    g1 ,g2 ,g3 =st .columns (3 )
    g1 .metric ("Position limite (USD)",fmt_amount (calib .position_limite ))
    g2 .metric ("Décalage cible (pips)",f"{calib .decalage_cible_pips :.1f}")
    g3 .metric ("Gamma calibré",f"{etat .gamma :.8f}")

    st .markdown ('<div class="section-title">CALIBRATION DE &eta; (ETA)</div>',unsafe_allow_html =True )
    st .markdown ("```\nSpread_marché_absolu = Spread_marché * MID_marché\neta = Spread_marché_absolu\n```")
    st .caption (
    "Hypothèse : échanger 1M USD en une journée sur le marché interbancaire crée un impact temporaire du "
    "même ordre de grandeur que le spread interbancaire observé. A affiner avec une calibration TCA réelle "
    "si des données d'exécution deviennent disponibles."
    )
    st .metric ("Eta calibré",f"{etat .eta :.6f}")

    st .markdown ('<div class="section-title">CALIBRATION DE &lambda; (LAMBDA)</div>',unsafe_allow_html =True )
    st .markdown ("```\nlambda = gamma\n```")
    st .caption ("Hypothèse : CFG applique la même politique d'aversion au risque pour le pricing et pour l'exécution.")
    st .metric ("Lambda calibré",f"{etat .lambda_ :.8f}")


def render_skew (etat ,history :pd .DataFrame ,transactions :pd .DataFrame )->None :
    st .markdown ('<div class="section-title">SKEW&lt;GO&gt; — Décalage de prix (Ho &amp; Stoll, 1981)</div>',unsafe_allow_html =True )
    st .markdown (
    "```\nk = gamma * sigma^2\nSkew = -k * (Position / 1 000 000)\nPrix_de_reservation = MID_marché + Skew\n"
    "ASK_final = Prix_de_reservation + Spread*MID/2\nBID_final = Prix_de_reservation - Spread*MID/2\n```"
    )

    st .markdown ('<div class="section-title">SIMULATEUR INTERACTIF</div>',unsafe_allow_html =True )
    sim_position =st .slider ("Position simulee (USD)",-10_000_000 ,10_000_000 ,value =int (etat .position ),step =50_000 )

    from core .models import ho_stoll 

    sim =ho_stoll (etat .gamma ,etat .sigma ,sim_position ,etat .mid_marche ,etat .spread_marche )
    s1 ,s2 ,s3 =st .columns (3 )
    s1 .metric ("Skew simulé",f"{sim .skew :.6f}")
    s2 .metric ("BID final simulé",f"{sim .bid_final :.4f}")
    s3 .metric ("ASK final simulé",f"{sim .ask_final :.4f}")

    positions_range =list (range (-10_000_000 ,10_000_001 ,250_000 ))
    skews_range =[ho_stoll (etat .gamma ,etat .sigma ,p ,etat .mid_marche ,etat .spread_marche ).skew for p in positions_range ]
    fig =charts .skew_curve_chart (positions_range ,skews_range ,etat .position ,etat .prix_final .skew )
    st .plotly_chart (fig ,width='stretch' ,config ={"displayModeBar":False })
    st .caption ("Le marqueur ambre indique la position RÉELLE actuelle de CFG (et non la position simulée ci-dessus).")

    st .markdown ('<div class="section-title">BACKTESTING DU SKEW (REJEU POINT-IN-TIME)</div>',unsafe_allow_html =True )
    st .caption (
    "Limite méthodologique : ce backtest ne dispose que d'un historique agrégé du MID de marché quotidien "
    "(pas des cotations bid/ask complètes de chaque concurrent, ni des exécutions réelles de CFG). "
    "Les résultats sont donc indicatifs, à affiner avec un historique plus riche et plus long."
    )
    if st .button ("LANCER LE BACKTEST SUR L'HISTORIQUE DISPONIBLE"):
        st .session_state ["backtest_df"]=backtest .run_backtest (
        history ,transactions ,st .session_state ["calib"],etat .spread_marche 
        )

    bt =st .session_state .get ("backtest_df")
    if bt is not None and not bt .empty :
        sans_skew =bt ["mid_marche_observe"]# bid_base/ask_base sont symetriques autour du MID (Section 4)
        fig2 =charts .backtest_overlay_chart (
        bt ["date"].tolist (),bt ["mid_marche_observe"].tolist (),sans_skew .tolist (),bt ["prix_reservation"].tolist ()
        )
        st .plotly_chart (fig2 ,width='stretch' ,config ={"displayModeBar":False })
        st .caption (
        "Le prix SANS skew coincide par construction avec le MID marché observé (bid_base/ask_base sont "
        "symetriques autour du MID). La courbe AVEC skew montre l'écart introduit par la gestion de position."
        )

        rec =bt [["date","mid_marche_observe","position","skew","prix_reservation"]].rename (
        columns ={
        "date":"DATE","mid_marche_observe":"MID OBSERVE","position":"POSITION",
        "skew":"SKEW (ECART)","prix_reservation":"PRIX AVEC SKEW",
        }
        )
        rec =_fmt_df (rec ,price_cols =["MID OBSERVE","SKEW (ECART)","PRIX AVEC SKEW"],amount_cols =["POSITION"])
        dense_table (rec )
    elif bt is not None :
        st .info ("Historique insuffisant pour lancer le backtest (au moins 3 jours de MID de marché requis).")


def render_stress (etat )->None :
    st .markdown ('<div class="section-title">STRESS&lt;GO&gt; — Tests de résistance</div>',unsafe_allow_html =True )

    with st .expander ("Ajouter un scénario personnalisé"):
        with st .form ("custom_scenario"):
            cs1 ,cs2 ,cs3 =st .columns (3 )
            nom =cs1 .text_input ("Nom du scenario",value ="Scenario personnalise")
            choc =cs2 .number_input ("Choc EUR/USD (%)",value =0.0 ,step =0.5 ,format ="%.2f")
            vol_mult =cs3 .number_input ("Multiplicateur de volatilité (0 = inchange)",value =0.0 ,step =0.5 ,format ="%.2f")
            if st .form_submit_button ("AJOUTER"):
                st .session_state ["custom_scenarios"].append (
                {"nom":nom ,"choc_pct":choc /100 if choc !=0 else None ,"vol_multiplier":vol_mult if vol_mult >0 else None }
                )

    scenarios =stress .SCENARIOS_PAR_DEFAUT +st .session_state ["custom_scenarios"]
    results =stress .run_all_scenarios (
    scenarios ,etat .eur_usd ,etat .mid_marche ,etat .spread_marche ,etat .sigma ,etat .position ,st .session_state ["calib"]
    )

    rows =[]
    for r in results :
        rows .append (
        {
        "SCENARIO":r .nom ,
        "MID THEO CHOQUE":r .mid_theo_choque ,
        "BORNE BASSE":r .borne_basse_choquee ,
        "BORNE HAUTE":r .borne_haute_choquee ,
        "ASK FINAL CHOQUE":r .ask_final_choque ,
        "BID FINAL CHOQUE":r .bid_final_choque ,
        "CONFORME":badge_html (r .conforme_ask and r .conforme_bid ),
        "IMPACT P&L (MAD)":r .impact_pnl ,
        }
        )
    disp =pd .DataFrame (rows )
    disp ["MID THEO CHOQUE"]=disp ["MID THEO CHOQUE"].map (fmt_price )
    disp ["BORNE BASSE"]=disp ["BORNE BASSE"].map (fmt_price )
    disp ["BORNE HAUTE"]=disp ["BORNE HAUTE"].map (fmt_price )
    disp ["ASK FINAL CHOQUE"]=disp ["ASK FINAL CHOQUE"].map (fmt_price )
    disp ["BID FINAL CHOQUE"]=disp ["BID FINAL CHOQUE"].map (fmt_price )
    disp ["IMPACT P&L (MAD)"]=disp ["IMPACT P&L (MAD)"].map (fmt_amount )
    dense_table (disp )


def render_liqd (etat )->None :
    st .markdown ('<div class="section-title">LIQD&lt;GO&gt; — Plan de liquidation (Almgren &amp; Chriss, 2000)</div>',unsafe_allow_html =True )

    auto_position =st .session_state .pop ("liqd_auto_position",None )
    if auto_position is not None :
        st .warning (f"Position récupérée automatiquement depuis POS<GO> (Panic Mode) : {fmt_amount (auto_position )} USD")
        position_a_liquider =auto_position 
    else :
        position_a_liquider =abs (etat .position )

    st .session_state ["horizon_t"]=st .slider ("Horizon de liquidation T (jours)",1 ,20 ,value =st .session_state ["horizon_t"])
    horizon_t =st .session_state ["horizon_t"]

    kappa =kappa_ac (etat .lambda_ ,etat .sigma ,etat .eta )
    traj =trajectoire_almgren_chriss (position_a_liquider ,kappa ,horizon_t )

    k1 ,k2 ,k3 =st .columns (3 )
    k1 .metric ("Position à liquider (USD)",fmt_amount (position_a_liquider ))
    k2 .metric ("Kappa (&kappa;)",f"{kappa :.6f}")
    k3 .metric ("Demi-vie (jours)",f"{traj .demi_vie :.2f}"if traj .demi_vie else "N/A")

    fig =charts .liquidation_chart (traj .jours ,traj .position_restante )
    st .plotly_chart (fig ,width='stretch' ,config ={"displayModeBar":False })

    rec =pd .DataFrame (
    {"JOUR":traj .jours ,"POSITION RESTANTE":traj .position_restante ,"MONTANT VENDU":traj .montant_vendu }
    )
    rec ["POSITION RESTANTE"]=rec ["POSITION RESTANTE"].map (fmt_amount )
    rec ["MONTANT VENDU"]=rec ["MONTANT VENDU"].map (fmt_amount )
    dense_table (rec )


def render_events ()->None :
    st .markdown ('<div class="section-title">EVENTS&lt;GO&gt; — Études d\'événement sur l\'activite du desk</div>',
    unsafe_allow_html =True )
    st .caption ("Trois études construites sur la même mecanique que l'étude dividendes de la Bourse de "
    "Casablanca : on aligne des événements sur un instant t = 0, puis on observé le "
    "comportement moyen autour de cet instant (« temps événementiel »).")

    tx =db .get_transactions ()
    px =db .get_price_history ()
    br =db .get_risk_breaches ()
    rfq =db .get_rfq_log ()

    tab_a ,tab_b ,tab_c =st .tabs ([
    "  A · Résilience après gros flux  ",
    "  B · Dépassements de limite  ",
    "  C · Persistance client  ",
    ])

    # ---------------- A. RESILIENCE ---------------- #
    with tab_a :
        st .markdown ("**Le MID de marché encaisse-t-il un choc après une journée a fort flux net, "
        "et en combien de temps revient-il a la normale ?**")
        res =events .etude_resilience (tx ,px )
        if res is None :
            st .info ("Données insuffisantes pour cette étude.")
        else :
            c1 ,c2 ,c3 ,c4 =st .columns (4 )
            metric_card (c1 ,"JOURNEES ANALYSEES",f"{res .n_evenements }",key ="ev_a_n")
            metric_card (c2 ,"IMPACT IMMEDIAT",f"{res .impact_immediat_pips :+.2f} pips",key ="ev_a_imp")
            dv ="n/d"if res .demi_vie_jours !=res .demi_vie_jours else f"{res .demi_vie_jours :.0f} j"
            metric_card (c3 ,"DEMI-VIE",dv ,key ="ev_a_dv")
            seuil_m =f"{res .seuil_montant /1e6 :.1f} M".replace (".",",")
            metric_card (c4 ,"SEUIL FLUX NET",seuil_m ,key ="ev_a_seuil")

            st .plotly_chart (charts .resilience_chart (res .courbe ),width='stretch' ,
            config ={"displayModeBar":False },key ="ev_chart_a")
            st .markdown (
            f"**Lecture.** Le jour d'un fort desequilibre de flux (t = 0), le MID se decale en moyenne "
            f"de **{res .impact_immediat_pips :+.2f} pips** dans le sens du flux, puis se resorbe : la "
            f"demi-vie de l'impact est de l'ordre d'une journee. C'est la **mesure empirique de "
            f"l'impact de marche et de la resilience** — precisement les grandeurs que le modele "
            f"d'Almgren-Chriss (page LIQD) suppose en amont pour calibrer la liquidation optimale. "
            f"L'etude relie ainsi une observation de terrain a une hypothese de modele.")
            with st .expander ("Note methodologique"):
                st .markdown (
                "Le flux net d'une journée est la somme algébrique des montants (achats clients de "
                "USD comptés +, ventes −), hors débouclage interbancaire. On retient les journées "
                "du 9e décile de flux net absolu, on aligne la trajectoire du MID sur le jour de "
                "l'événement (base = MID de la veille), orientée dans le sens du flux, puis on "
                "moyenne les trajectoires pour obtenir la réaction moyenne du marché.")

                # ---------------- B. DEPASSEMENTS ---------------- #
    with tab_b :
        st .markdown ("**Autour d'un dépassement de limite, comment evolue la position du desk, "
        "et a quelle vitesse revient-elle dans les clous ?**")
        bres =events .etude_depassements (tx ,br )
        if bres is None :
            st .info ("Aucun episode de dépassement disponible.")
        else :
            c1 ,c2 ,c3 ,c4 =st .columns (4 )
            metric_card (c1 ,"EPISODES",f"{bres .n_episodes }",key ="ev_b_n")
            metric_card (c2 ,"PIC MEDIAN",f"{bres .pic_median_pct :.0f} %",key ="ev_b_pic")
            metric_card (c3 ,"PIC MAX",f"{bres .pic_max_pct :.0f} %",key ="ev_b_max")
            metric_card (c4 ,"DUREE MEDIANE",f"{bres .duree_mediane_jours :.0f} j",key ="ev_b_dur")

            st .plotly_chart (charts .breach_position_chart (bres .courbe ),width='stretch' ,
            config ={"displayModeBar":False },key ="ev_chart_b")
            st .markdown (
            f"**Lecture.** La position (en % de la limite prudentielle) monte jusqu'au franchissement "
            f"a t = 0, atteint en mediane **{bres .pic_median_pct :.0f} %** de la limite, puis reflue "
            f"rapidement : le desk deboucle et revient sous le seuil d'alerte en general des le "
            f"lendemain. Cela illustre le fonctionnement du controle de position et alimente le "
            f"dimensionnement de la liquidation (page LIQD).")
            with st .expander ("Note methodologique"):
                st .markdown (
                "La position retenue est le **pic intraday** de chaque journée (valeur absolue "
                "maximale de la position cumulée), et non la clôture : c'est le pic qui declenche "
                "le dépassement, la clôture etant ramenee proche de zero par le débouclage de fin "
                "de séance. Les episodes sont isoles par hystérésis (entree a 80 %, sortie a 60 % "
                "de la limite) pour ne pas compter deux fois une même tension.")

                # ---------------- C. PERSISTANCE ---------------- #
    with tab_c :
        st .markdown ("**Un client qui vient d'accepter une cotation est-il plus enclin a accepter "
        "la suivante ? Existe-t-il un momentum comportemental ?**")
        pres =events .etude_persistance_client (rfq )
        if pres is None :
            st .info ("Historique de cotations insuffisant.")
        else :
            c1 ,c2 ,c3 ,c4 =st .columns (4 )
            metric_card (c1 ,"APRES ACCEPTATION",f"{pres .taux_apres_acceptation :.1f} %",key ="ev_c_acc")
            metric_card (c2 ,"APRES REFUS",f"{pres .taux_apres_refus :.1f} %",key ="ev_c_ref")
            metric_card (c3 ,"MOMENTUM",f"{pres .momentum :+.1f} pts",key ="ev_c_mom")
            sig ="OUI"if pres .significatif else "NON"
            metric_card (c4 ,"SIGNIFICATIF (5%)",sig ,key ="ev_c_sig")

            g1 ,g2 =st .columns ([3 ,2 ])
            with g1 :
                st .plotly_chart (charts .persistance_chart (pres .taux_apres_acceptation ,
                pres .taux_apres_refus ),
                width='stretch' ,config ={"displayModeBar":False },
                key ="ev_chart_c")
            with g2 :
                st .plotly_chart (charts .transition_heatmap (pres .matrice ),
                width='stretch' ,config ={"displayModeBar":False },
                key ="ev_chart_c2")
            p_txt ="< 0,001"if pres .p_value <0.001 else f"= {pres .p_value :.3f}"
            st .markdown (
            f"**Lecture.** Un client qui vient d'accepter accepte a nouveau dans "
            f"**{pres .taux_apres_acceptation :.0f} %** des cas, contre seulement "
            f"**{pres .taux_apres_refus :.0f} %** apres un refus — un momentum de "
            f"**{pres .momentum :+.0f} points**, statistiquement significatif (khi-deux, p {p_txt }). "
            f"Concretement, le passe recent d'un client informe sur sa sensibilite au prix : un "
            f"client en serie d'acceptations peut supporter un spread legerement plus large, "
            f"information exploitable par le module RFQ.")
            with st .expander ("Note methodologique"):
                st .markdown (
                "Pour chaque client, les demandes sont ordonnées dans le temps et l'on forme les "
                "paires (issue précédente, issue courante). La matrice de transition croise ces "
                "deux issues ; le test du khi-deux mesure si elles sont indépendantes. La méthode "
                "s'applique directement à un historique réel de cotations acceptées et refusées.")


def render_theory() -> None:
    st.markdown('<div class="section-title">THEORY&lt;GO&gt; — Concepts et formules du terminal</div>',
                unsafe_allow_html=True)
    st.caption("Cette page rassemble l'ensemble des concepts, formules et méthodes utilisés dans le "
               "terminal, organisés par domaine. Elle sert de référence pour la compréhension et la "
               "soutenance.")

    (t_prix, t_skew, t_risk, t_fwd, t_flow, t_vol, t_ts, t_ml, t_events) = st.tabs([
        "  Prix de base  ", "  Skew Ho-Stoll  ", "  Risque / VaR  ", "  Forward  ",
        "  Flux / RFQ  ", "  Volatilité  ", "  Séries temp.  ", "  ML  ", "  Études d'événement  ",
    ])

    # ============ PRIX DE BASE ============
    with t_prix:
        st.markdown("### Le prix de référence USD/MAD")
        st.markdown(
            "Le dirham marocain est géré en **panier** face à l'euro et au dollar. Bank Al-Maghrib "
            "publie un cours central et autorise une fluctuation dans une bande de ±5 %. Le MID "
            "théorique USD/MAD se déduit du cours EUR/USD par une relation de panier calibrée :")
        st.latex(r"\text{MID}_{\text{théo}} = a\,(\text{EUR/USD})^2 + b\,(\text{EUR/USD}) + c")
        st.markdown(
            "avec $a = 2{,}9621$, $b = -11{,}963$, $c = 19{,}416$. Cette forme quadratique capture la "
            "pondération 60 % EUR / 40 % USD du panier. La **bande de fluctuation** encadre ce MID :")
        st.latex(r"\text{borne basse} = \text{MID} \times 0{,}95 \qquad \text{borne haute} = \text{MID} \times 1{,}05")
        st.markdown("### Le prix de marché et le spread")
        st.markdown(
            "Le MID de marché est la moyenne des cotations des banques concurrentes. À partir de lui, "
            "on construit un prix acheteur (BID) et vendeur (ASK) encadrant le MID d'un demi-spread :")
        st.latex(r"\text{ASK}_{\text{base}} = \text{MID}_{\text{marché}}\left(1 + \tfrac{\text{spread}}{2}\right) "
                 r"\qquad \text{BID}_{\text{base}} = \text{ASK}_{\text{base}} - \text{spread} \times \text{MID}")
        st.markdown("**Bibliothèques.** Calculs purs en `numpy`/`pandas` ; aucune dépendance externe pour cette partie.")

    # ============ SKEW HO-STOLL ============
    with t_skew:
        st.markdown("### Le décalage de prix selon la position (Ho & Stoll, 1981)")
        st.markdown(
            "C'est le cœur du pricer. Un teneur de marché qui accumule une position veut inciter le "
            "flux à la réduire : s'il est déjà long, il baisse ses prix pour vendre plus facilement. "
            "Ho et Stoll (1981) formalisent ce **décalage** (skew) proportionnel à la position et à "
            "l'aversion au risque.")
        st.markdown("**La volatilité** $\\sigma$ est l'écart-type des variations quotidiennes du MID, "
                    "normalisé par la racine du nombre de jours écoulés (hypothèse de marche aléatoire) :")
        st.latex(r"\sigma = \text{std}\left(\frac{\Delta \text{MID}}{\sqrt{n_{\text{jours}}}}\right)")
        st.markdown("**L'aversion au risque** $\\gamma$ n'est pas choisie arbitrairement : elle est "
                    "**déduite** d'un objectif de gestion (un décalage cible pour une position limite) :")
        st.latex(r"k_{\text{cible}} = \frac{\text{décalage cible}}{\text{position limite}/1\text{M}} "
                 r"\qquad \gamma = \frac{k_{\text{cible}}}{\sigma^2}")
        st.markdown("**Le skew** et le prix de réservation en découlent :")
        st.latex(r"k = \gamma\,\sigma^2 \qquad \text{Skew} = -k \times \frac{\text{Position}}{1\text{M}}")
        st.latex(r"\text{Prix}_{\text{réservation}} = \text{MID}_{\text{marché}} + \text{Skew}")
        st.latex(r"\text{ASK}_{\text{final}},\ \text{BID}_{\text{final}} = \text{Prix}_{\text{réservation}} \pm \tfrac{\text{spread} \times \text{MID}}{2}")
        st.markdown(
            "**Alternative écartée.** Le modèle d'Avellaneda-Stoikov (2008) a été considéré puis "
            "rejeté : il suppose un horizon de fin de journée et une intensité d'arrivée des ordres "
            "difficiles à calibrer sur le marché du dirham. Ho-Stoll, plus robuste, a été retenu.")
        st.markdown("**Bibliothèques.** `numpy` pour les calculs vectoriels ; pas de solveur externe.")

    # ============ RISQUE / VAR ============
    with t_risk:
        st.markdown("### Value at Risk paramétrique")
        st.markdown(
            "La VaR traduit la position en **perte potentielle chiffrée**. Sous l'hypothèse de "
            "variations normales, la perte à un horizon de $h$ jours ne dépasse un seuil qu'avec une "
            "probabilité fixée (95 % ou 99 %) :")
        st.latex(r"\text{VaR}_{\alpha} = z_{\alpha} \times |\text{Position}| \times \sigma \times \sqrt{h}")
        st.markdown(
            "où $z_{0{,}95} = 1{,}645$ et $z_{0{,}99} = 2{,}326$ sont les quantiles de la loi normale. "
            "L'écart-type du P&L est $|\\text{Position}| \\times \\sigma \\times \\sqrt{h}$.")
        st.markdown("**La valeur du pip** convertit un mouvement de prix en montant :")
        st.latex(r"\text{valeur du pip} = \text{Position} \times 10^{-4}")
        st.markdown("**Le cours croisé EUR/MAD** se déduit du USD/MAD et de l'EUR/USD :")
        st.latex(r"\text{EUR/MAD} = \text{EUR/USD} \times \text{USD/MAD}")
        st.markdown("**Bibliothèques.** `scipy.stats` pour les quantiles de la loi normale, `numpy` "
                    "pour la simulation de la distribution de P&L affichée.")

    # ============ FORWARD ============
    with t_fwd:
        st.markdown("### Change à terme (parité des taux d'intérêt couverte)")
        st.markdown(
            "Le cours à terme se déduit du cours comptant et de l'écart de taux d'intérêt entre les "
            "deux devises — c'est la **parité des taux couverte** (CIP), une relation d'arbitrage :")
        st.latex(r"F = S \times \frac{1 + r_{\text{MAD}} \times \frac{d}{360}}{1 + r_{\text{USD}} \times \frac{d}{360}}")
        st.markdown("où $S$ est le comptant, $r$ les taux, $d$ le nombre de jours. Les **points de "
                    "terme** sont la différence $F - S$ (souvent exprimée en pips) :")
        st.latex(r"\text{points de terme} = F - S")
        st.markdown("**Le swap de change** combine une jambe comptant et une jambe à terme de sens "
                    "opposé. Les taux pour une échéance intermédiaire sont **interpolés linéairement** "
                    "entre les points de courbe encadrants.")
        st.markdown("**Bibliothèques.** `numpy` ; interpolation linéaire manuelle (pas de `scipy.interpolate`).")

    # ============ FLUX / RFQ ============
    with t_flow:
        st.markdown("### Cotation optimale sur demande (RFQ)")
        st.markdown(
            "Quand un client demande un prix ferme, plus le spread proposé est large, plus la marge "
            "unitaire est élevée mais plus le risque de refus augmente. On modélise la **probabilité "
            "d'acceptation** comme décroissante avec le spread (loi logistique / exponentielle) :")
        st.latex(r"P(\text{accepte} \mid \delta) = e^{-k\,\delta} \qquad \text{ou} \qquad "
                 r"P = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \delta)}}")
        st.markdown("Le **spread optimal** maximise l'espérance de gain (marge × probabilité "
                    "d'acceptation) :")
        st.latex(r"\delta^\star = \arg\max_{\delta}\ \delta \times P(\text{accepte} \mid \delta)")
        st.markdown(
            "La segmentation par **taille** module ce spread : plus l'opération est grosse, plus elle "
            "consomme de la limite de risque, donc plus le spread s'élargit. Les paramètres de la loi "
            "logistique sont estimés par **maximum de vraisemblance** (méthode de Newton-Raphson).")
        st.markdown("**Bibliothèques.** `numpy`, `scipy.optimize` pour la calibration de la logistique.")

    # ============ VOLATILITE ============
    with t_vol:
        st.markdown("### Volatilité conditionnelle (famille GARCH)")
        st.markdown(
            "La volatilité d'un taux de change n'est pas constante : elle se regroupe en périodes "
            "calmes et agitées. Les modèles GARCH capturent cette dynamique. Le **GARCH(1,1)** :")
        st.latex(r"\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2")
        st.markdown("Le **GJR-GARCH** ajoute un effet de levier (les chocs négatifs augmentent plus la "
                    "volatilité que les chocs positifs) :")
        st.latex(r"\sigma_t^2 = \omega + (\alpha + \gamma\,\mathbb{1}_{\varepsilon_{t-1}<0})\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2")
        st.markdown("L'**EGARCH** modélise le logarithme de la variance, garantissant sa positivité :")
        st.latex(r"\ln \sigma_t^2 = \omega + \beta \ln \sigma_{t-1}^2 + \alpha\left(|z_{t-1}| - \mathbb{E}|z_{t-1}|\right) + \gamma z_{t-1}")
        st.markdown(
            "Les trois modèles sont estimés par **maximum de vraisemblance**. Astuce numérique : les "
            "variations sont normalisées (divisées par leur écart-type) avant l'estimation, puis les "
            "paramètres sont remis à l'échelle — cela stabilise la maximisation sur des variations de "
            "l'ordre de $10^{-4}$.")
        st.markdown("**Bibliothèques.** `scipy.optimize.minimize` pour la vraisemblance, `numpy` pour "
                    "les filtres récursifs de variance.")

    # ============ SERIES TEMPORELLES ============
    with t_ts:
        st.markdown("### Analyse des séries chronologiques")
        st.markdown("**Décomposition.** La série de MID est séparée en tendance, composante cyclique "
                    "et résidu, pour distinguer le mouvement de fond des fluctuations.")
        st.markdown("**Stationnarité — test de Dickey-Fuller augmenté (ADF).** Teste si la série "
                    "revient vers une moyenne (stationnaire) ou dérive (racine unité) :")
        st.latex(r"\Delta y_t = \alpha + \beta t + \rho\, y_{t-1} + \sum_{i} \phi_i \Delta y_{t-i} + \varepsilon_t")
        st.markdown("L'hypothèse de racine unité ($\\rho = 0$) est rejetée si la statistique passe sous "
                    "la valeur critique (−2,86 à 5 %). **Autocorrélations** (ACF/PACF) mesurent la "
                    "mémoire de la série.")
        st.markdown("**Retour à la moyenne — processus d'Ornstein-Uhlenbeck.** Modélise la vitesse de "
                    "rappel vers la moyenne d'équilibre :")
        st.latex(r"dX_t = \theta\,(\mu - X_t)\,dt + \sigma\,dW_t \qquad \text{demi-vie} = \frac{\ln 2}{\theta}")
        st.markdown(
            "La demi-vie indique en combien de temps un écart à la moyenne se résorbe de moitié. "
            "Rappel de prudence : Meese-Rogoff (1983) ont montré qu'à court terme, aucun modèle ne "
            "bat une simple marche aléatoire pour prévoir un taux de change.")
        st.markdown("**Bibliothèques.** `statsmodels` (test ADF, décomposition), `numpy` pour l'estimation d'Ornstein-Uhlenbeck.")

    # ============ MACHINE LEARNING ============
    with t_ml:
        st.markdown("### Segmentation de clientèle (k-means)")
        st.markdown(
            "Les contreparties sont regroupées selon leur comportement (taille moyenne, fréquence, "
            "sens dominant) par l'algorithme **k-means**, qui minimise l'inertie intra-classe :")
        st.latex(r"\min \sum_{k}\sum_{x \in C_k} \lVert x - \mu_k \rVert^2")
        st.markdown(
            "L'initialisation **k-means++** évite les mauvais départs. Le nombre de segments est "
            "choisi par le **score de silhouette**, qui mesure si chaque point est bien dans son "
            "groupe :")
        st.latex(r"s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}")
        st.markdown("où $a(i)$ est la distance moyenne au sein du groupe et $b(i)$ au groupe voisin le "
                    "plus proche. Les variables sont **standardisées** avant le clustering.")
        st.markdown("### Détection d'anomalies (score z modifié)")
        st.markdown("Les cotations aberrantes sont repérées par un score z robuste, fondé sur la "
                    "médiane et l'écart absolu médian (MAD), moins sensible aux valeurs extrêmes que "
                    "la moyenne :")
        st.latex(r"z_{\text{mod}} = \frac{0{,}6745\,(x - \tilde{x})}{\text{MAD}}, \qquad \text{MAD} = \text{médiane}(|x - \tilde{x}|)")
        st.markdown("Un point est signalé anormal si $|z_{\\text{mod}}| > 3{,}5$ (seuil d'Iglewicz-Hoaglin, 1993).")
        st.markdown("**Bibliothèques.** k-means et silhouette implémentés à la main sur `numpy` "
                    "(pas de `scikit-learn`), pour la transparence pédagogique du code.")

    # ============ ETUDES D'EVENEMENT ============
    with t_events:
        st.markdown("### Trois études d'événement (page EVENTS)")
        st.markdown(
            "Ces études reprennent la mécanique de l'étude dividendes : aligner des événements sur un "
            "instant $t = 0$ et observer le comportement moyen autour. Les données de marché (page "
            "EVENTS) sont générées par un **GARCH(1,1)** avec rappel de bande et impact des gros flux.")
        st.markdown("**A · Résilience.** Autour des journées à fort flux net (9e décile), on aligne le "
                    "MID en base veille :")
        st.latex(r"\text{écart}_t = (\text{MID}_t - \text{MID}_{-1}) \times 10^4 \times \text{signe du flux}")
        st.markdown("On en tire l'impact immédiat, la demi-vie de résorption et le taux de résorption "
                    "à 5 jours — la mesure empirique de l'impact de marché que suppose Almgren-Chriss.")
        st.markdown("**B · Dépassements.** On suit la **position de pic intraday** et on isole les "
                    "épisodes de dépassement par hystérésis (entrée à 80 %, sortie à 60 % de la limite) :")
        st.latex(r"\text{pic}_j = \max_{i \in \text{jour } j} |\text{Position}_i|")
        st.markdown("**C · Persistance client.** On teste si l'acceptation d'une cotation dépend de "
                    "l'issue précédente du même client, par un **test du khi-deux** sur la matrice de "
                    "transition :")
        st.latex(r"\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}")
        st.markdown(
            "**Almgren-Chriss — liquidation optimale (page LIQD).** Détermine la trajectoire de "
            "liquidation minimisant le compromis entre coût d'impact (vendre vite) et risque de "
            "marché (vendre lentement) :")
        st.latex(r"x_j = \text{Position} \times \frac{\sinh(\kappa(T-j))}{\sinh(\kappa T)}, \qquad "
                 r"\kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}, \qquad \text{demi-vie} = \frac{\ln 2}{\kappa}")
        st.markdown("**Bibliothèques.** `scipy.stats` (khi-deux), `numpy` (courbes événementielles, "
                    "fonctions hyperboliques d'Almgren-Chriss).")

def render_audit ()->None :
    st .markdown ('<div class="section-title">AUDIT&lt;GO&gt; — Journal d\'audit de conformité</div>',unsafe_allow_html =True )
    if not AUDIT_LOG_PATH .exists ():
        st .info ("Aucune clôture de journée enregistrée pour le moment.")
        return 

    log =pd .read_csv (AUDIT_LOG_PATH )
    disp =log .copy ()
    for col in ["statut_ask","statut_bid","statut_ecart_theorique"]:
        disp [col ]=disp [col ].map (lambda v :badge_html (v =="CONFORME",ok_text ="CONFORME",bad_text ="HORS BANDE"))
    disp .columns =[c .upper ()for c in disp .columns ]
    dense_table (disp )

    st .download_button (
    "TÉLÉCHARGER LE JOURNAL D'AUDIT COMPLET",data =AUDIT_LOG_PATH .read_bytes (),
    file_name ="audit_log.csv",mime ="text/csv",
    )


    # --------------------------------------------------------------------------- #
    # FWD — Change a terme
    # --------------------------------------------------------------------------- #

def render_fwd (etat )->None :
    st .markdown ('<div class="section-title">FWD&lt;GO&gt; — Change à terme &amp; swaps</div>',
    unsafe_allow_html =True )
    c =theme .P ()

    st .caption (
    "Contrairement au spot, le cours à terme n'est pas modélisé : il est imposé par un "
    "arbitrage. S'il s'écartait de la parité des taux, il serait possible de construire "
    "un profit sans risque en combinant emprunt, change au comptant et placement."
    )

    t1 ,t2 ,t3 =st .columns (3 )
    taux_mad =t1 .number_input ("Taux MAD (%)",value =st .session_state ["taux_mad"]*100 ,
    step =0.05 ,format ="%.2f",key ="fwd_tx_mad")/100 
    taux_usd =t2 .number_input ("Taux USD (%)",value =st .session_state ["taux_usd"]*100 ,
    step =0.05 ,format ="%.2f",key ="fwd_tx_usd")/100 
    spot_ref =t3 .number_input ("Spot de référence",value =float (etat .mid_marche ),
    step =0.0001 ,format ="%.4f",key ="fwd_spot")
    st .session_state ["taux_mad"],st .session_state ["taux_usd"]=taux_mad ,taux_usd 

    differentiel =(taux_mad -taux_usd )*100 
    st .markdown (
    f'<div style="font-family:{theme .FONT_MONO }; font-size:0.85rem; color:{c ["text_secondary"]}; '
    f'margin:6px 0 12px 0;">Differentiel de taux MAD &minus; USD : '
    f'<b style="color:{c ["cfg_red"]};">{differentiel :+.2f} %</b> &mdash; '
    f'{"le dollar cote en report (prime a terme)"if differentiel >0 else "le dollar cote en deport (decote a terme)"}'
    f'</div>',
    unsafe_allow_html =True ,
    )

    grille =forward .grille_terme (spot_ref ,taux_mad ,taux_usd )

    st .markdown ('<div class="section-title">GRILLE DE COTATION A TERME</div>',unsafe_allow_html =True )
    lignes =[{
    "ECHEANCE":g .tenor ,
    "JOURS":g .jours ,
    "POINTS":f"{g .points_terme :+.1f}",
    "OUTRIGHT":fmt_price (g .forward_outright ),
    "SENS":g .report_deport ,
    "COUT ANNUALISE":f"{g .cout_couverture_annualise *100 :+.2f} %",
    }for g in grille ]
    dense_table (pd .DataFrame (lignes ))
    st .caption (
    "Un desk ne cote pas le cours complet mais les **points de terme** : on annonce "
    "\"-370 points\" sur le spot du moment. Cette convention isole la composante taux "
    "d'intérêt de la composante cours de change."
    )

    st .plotly_chart (
    charts .courbe_terme_chart ([g .tenor for g in grille ],[g .points_terme for g in grille ]),
    width='stretch' ,config ={"displayModeBar":False },
    )

    # ---- Cotation d'une operation a terme ---- #
    st .markdown ('<div class="section-title">COTATION D\'UNE COUVERTURE CLIENT</div>',unsafe_allow_html =True )
    o1 ,o2 ,o3 =st .columns (3 )
    sens_fwd =o1 .selectbox ("Sens (vu du client)",["Achete","Vend"],key ="fwd_sens")
    montant_fwd =o2 .number_input ("Montant (USD)",min_value =10_000.0 ,step =100_000.0 ,
    value =1_000_000.0 ,key ="fwd_montant")
    jours_fwd =o3 .number_input ("Echeance (jours)",min_value =1 ,max_value =720 ,value =90 ,
    step =1 ,key ="fwd_jours")

    cot =forward .coter_terme (spot_ref ,taux_mad ,taux_usd ,f"{jours_fwd }J",int (jours_fwd ))
    demi_spread =(etat .spread_marche *spot_ref )/2.0 
    prix_client_fwd =cot .forward_outright +(demi_spread if sens_fwd =="Achete"else -demi_spread )

    components .rfq_ticket (
    headline =f"TERME {jours_fwd }J @ {fmt_price (prix_client_fwd )}",
    lines =[
    ("Spot de référence",fmt_price (spot_ref )),
    ("Points de terme",f"{cot .points_terme :+.1f} pips ({cot .report_deport })"),
    ("Cours a terme théorique",fmt_price (cot .forward_outright )),
    ("Demi-spread applique",f"{demi_spread *10_000 :.1f} pips"),
    ("PRIX CLIENT A TERME",fmt_price (prix_client_fwd )),
    ("Contrevaleur a l'echeance",f"{montant_fwd *prix_client_fwd :,.0f} MAD".replace (","," ")),
    ("Coût de couverture",f"{cot .cout_couverture_annualise *100 :+.2f} % annualise"),
    ],
    )

    # ---- Swap de change ---- #
    st .markdown ('<div class="section-title">SWAP DE CHANGE</div>',unsafe_allow_html =True )
    sw =forward .swap_de_change (montant_fwd ,spot_ref ,taux_mad ,taux_usd ,f"{jours_fwd }J",int (jours_fwd ))
    s1 ,s2 ,s3 =st .columns (3 )
    metric_card (s1 ,"POINTS DE SWAP",f"{sw .points_swap :+.1f}",key ="sw_pts")
    metric_card (s2 ,"JAMBE COMPTANT",f"{sw .jambe_spot_mad :,.0f}".replace (","," "),key ="sw_spot")
    metric_card (s3 ,"RESULTAT NET",f"{sw .resultat_net_mad :+,.0f} MAD".replace (","," "),
    delta_kind ="up"if sw .resultat_net_mad >=0 else "down",key ="sw_net")
    st .caption (
    "Le swap combine un achat au comptant et une vente à terme avec la même contrepartie. "
    "Il ne crée **aucune position de change nette** — les deux jambes s'annulent en montant — "
    "mais déplace de la trésorerie d'une devise à l'autre. Son résultat est purement un "
    "différentiel de taux : c'est le coût de portage de la devise."
    )

    # ---- Echeancier et risque de decalage ---- #
    st .markdown ('<div class="section-title">ECHEANCIER ET RISQUE DE Décalage DE DATES</div>',
    unsafe_allow_html =True )
    pos_terme =st .session_state ["positions_terme"]
    cols =st .columns (len (pos_terme ))
    nouvelles ={}
    for i ,(tenor ,val )in enumerate (pos_terme .items ()):
        nouvelles [tenor ]=cols [i ].number_input (f"Position {tenor }",value =float (val ),
        step =250_000.0 ,key =f"posterme_{tenor }")
    if nouvelles !=pos_terme :
        st .session_state ["positions_terme"]=nouvelles 
        st .rerun ()

    ech =forward .echeancier (nouvelles )
    e1 ,e2 ,e3 =st .columns (3 )
    metric_card (e1 ,"POSITION NETTE",fmt_amount (ech .position_nette_totale ),key ="ech_net")
    metric_card (e2 ,"POSITION BRUTE",fmt_amount (ech .position_brute_totale ),key ="ech_brut")
    metric_card (e3 ,"PLUS GRAND GAP",f"{fmt_amount (ech .gap_max )} ({ech .tenor_gap_max })",key ="ech_gap")

    if any (v !=0 for v in nouvelles .values ()):
        st .plotly_chart (
        charts .echeancier_chart (list (nouvelles .keys ()),list (nouvelles .values ())),
        width='stretch' ,config ={"displayModeBar":False },
        )

    if ech .est_plat_en_montant and ech .position_brute_totale >0 :
        st .markdown (
        f'<div style="background:{c ["amber"]}22; border-left:3px solid {c ["amber"]}; '
        f'padding:10px 14px; font-family:{theme .FONT_MONO }; color:{c ["amber"]};">'
        f'POSITION PLATE EN MONTANT MAIS EXPOSEE : brute {fmt_amount (ech .position_brute_totale )} USD '
        f'repartie sur plusieurs echeances. Aucune exposition au niveau du cours, mais une exposition '
        f'reelle a la deformation de la courbe de taux.</div>',
        unsafe_allow_html =True ,
        )


        # --------------------------------------------------------------------------- #
        # FLOW — Intensite des flux et spread optimal
        # --------------------------------------------------------------------------- #

def render_flow (etat )->None :
    st .markdown ('<div class="section-title">FLOW&lt;GO&gt; — Intensite des flux &amp; spread optimal</div>',
    unsafe_allow_html =True )
    c =theme .P ()

    st .markdown (
    f'<div style="font-family:{theme .FONT_SANS }; color:{c ["text_secondary"]}; '
    f'font-size:0.88rem; margin-bottom:10px;">'
    f'Jusqu\'ici le pricer <b>reprenait</b> le spread du marche. Il ne le <b>decidait</b> pas. '
    f'Or c\'est la decision centrale du market making : plus le spread est large, plus la marge '
    f'par transaction est elevee, mais moins de clients acceptent le prix.</div>',
    unsafe_allow_html =True ,
    )

    rfq_log =db .get_rfq_log ()

    st .markdown ('<div class="section-title">CALIBRATION SUR LES DEMANDES DE COTATION OBSERVEES</div>',
    unsafe_allow_html =True )

    if rfq_log .empty :
        st .info ("Aucune demande de cotation enregistree : le modèle ne peut pas etre calibré.")
        return 

    groupes =rfq_log .groupby ("delta_propose").agg (
    n_demandes =("id","count"),n_acceptees =("acceptee","sum")
    ).reset_index ()

    points =[
    flow .PointCalibration (float (r ["delta_propose"]),int (r ["n_demandes"]),int (r ["n_acceptees"]))
    for _ ,r in groupes .iterrows ()
    ]
    params =flow .calibrer_intensite (points )

    if params is None :
        st .warning ("Données insuffisantes pour calibrer le modèle (au moins deux niveaux de spread requis).")
        return 

    tbl =pd .DataFrame ([{
    "ECART PROPOSE":f"{p .delta *10_000 :.1f} pips",
    "DEMANDES":p .n_demandes ,
    "ACCEPTEES":p .n_acceptees ,
    "TAUX D'ACCEPTATION":f"{p .taux_acceptation *100 :.1f} %",
    }for p in points ])
    dense_table (tbl )

    k1 ,k2 ,k3 =st .columns (3 )
    metric_card (k1 ,"A — INTENSITE MAXIMALE",f"{params .A :.3f}",key ="flow_A")
    metric_card (k2 ,"k — SENSIBILITE AU PRIX",f"{params .k :,.0f}".replace (","," "),key ="flow_k")
    metric_card (k3 ,"QUALITE DE L'AJUSTEMENT",f"R2 = {params .r2 :.3f}",
    delta =f"{params .n_observations } observations",key ="flow_r2")

    deltas_fit ,intens_fit ,pnls_fit =flow .courbe_pnl (params ,delta_max =max (p .delta for p in points )*1.4 )
    st .plotly_chart (
    charts .hit_ratio_chart ([p .delta for p in points ],[p .taux_acceptation for p in points ],
    deltas_fit ,intens_fit ),
    width='stretch' ,config ={"displayModeBar":False },
    )
    st .caption (
    "Méthode : le modèle lambda(delta) = A x exp(-k x delta) devient lineaire par passage au "
    "logarithme. Une regression des moindres carres de ln(taux d'acceptation) sur delta fournit "
    "-k comme pente et ln(A) comme ordonnee a l'origine."
    )

    # ---- Optimisation ---- #
    st .markdown ('<div class="section-title">DETERMINATION DU SPREAD OPTIMAL</div>',unsafe_allow_html =True )
    delta_opt =flow .spread_optimal (params )
    demi_spread_marche =(etat .spread_marche *etat .mid_marche )/2.0 
    comp =flow .comparer_au_marche (params ,etat .spread_marche *etat .mid_marche )

    st .plotly_chart (
    charts .intensite_pnl_chart (deltas_fit ,intens_fit ,pnls_fit ,delta_opt ,demi_spread_marche ),
    width='stretch' ,config ={"displayModeBar":False },
    )

    o1 ,o2 ,o3 =st .columns (3 )
    metric_card (o1 ,"SPREAD OPTIMAL (Modèle)",f"{delta_opt *10_000 :.1f} pips",key ="flow_opt")
    metric_card (o2 ,"DEMI-SPREAD MARCHE",f"{demi_spread_marche *10_000 :.1f} pips",key ="flow_mkt")
    metric_card (o3 ,"ECART",f"{comp ['ecart_pips']:+.1f} pips",
    delta_kind ="up"if comp ["ecart_pips"]>0 else "down",key ="flow_ecart")

    st .markdown (
    f'<div style="background:{c ["cfg_red"]}18; border-left:3px solid {c ["cfg_red"]}; '
    f'padding:10px 14px; font-family:{theme .FONT_MONO }; font-size:0.86rem;">'
    f'{comp ["diagnostic"]}</div>',
    unsafe_allow_html =True ,
    )

    st .markdown (
    "**Résultat théorique** : en annulant la derivee de `delta x A x exp(-k x delta)`, on obtient "
    "`delta* = 1/k`. Le spread optimal **ne dépend pas de A** : le volume global de clientele "
    "change le montant gagne, mais pas le prix auquel il faut coter. Seule compte la sensibilite au prix."
    )

    # ---- Combinaison avec le skew d'inventaire ---- #
    st .markdown ('<div class="section-title">COMBINAISON AVEC LE SKEW D\'INVENTAIRE</div>',
    unsafe_allow_html =True )
    d_bid ,d_ask =flow .spread_optimal_avec_inventaire (params ,etat .gamma ,etat .sigma ,etat .position )
    b1 ,b2 =st .columns (2 )
    metric_card (b1 ,"Écart OPTIMAL A L'ACHAT",f"{d_bid *10_000 :.2f} pips",key ="flow_dbid")
    metric_card (b2 ,"Écart OPTIMAL A LA VENTE",f"{d_ask *10_000 :.2f} pips",key ="flow_dask")
    st .caption (
    f"Position actuelle : {fmt_amount (etat .position )} USD. Le spread optimal pur (1/k) suppose un "
    "teneur de marché indifferent a son inventaire. En combinant avec le skew de Ho & Stoll, la "
    "cotation devient asymetrique : le cote qui reequilibre le stock est propose plus agressivement."
    )

    # ------------------------------------------------------------------ #
    # Approche alternative : regression logistique sur les RFQ individuelles
    # ------------------------------------------------------------------ #
    st .markdown ('<div class="section-title">APPROCHE ALTERNATIVE — REGRESSION LOGISTIQUE</div>',
    unsafe_allow_html =True )
    st .markdown (
    f'<div style="font-family:{theme .FONT_SANS }; color:{c ["text_secondary"]}; '
    f'font-size:0.87rem; margin-bottom:10px;">'
    f'La calibration exponentielle ci-dessus travaille sur des taux d\'acceptation '
    f'<b>agreges</b> par palier de spread : elle resume plusieurs dizaines de demandes en un '
    f'seul point. Une regression logistique exploite au contraire chaque demande '
    f'individuellement, et admet plusieurs variables explicatives.</div>',
    unsafe_allow_html =True )

    modele_log =flow .calibrer_logistique (rfq_log )
    if modele_log is None :
        st .info ("Au moins trente demandes de cotation, acceptées et refusées, sont nécessaires "
        "pour estimer une regression logistique.")
        return 

    coefs =pd .DataFrame ([
    {"VARIABLE":nom ,"COEFFICIENT":f"{val :+.4f}",
    "EFFET SUR L'ACCEPTATION":("augmente"if val >0 else "reduit")if nom !="Constante"else "-"}
    for nom ,val in zip (modele_log .noms ,modele_log .coefficients )
    ])
    dense_table (coefs )

    q1 ,q2 ,q3 =st .columns (3 )
    metric_card (q1 ,"OBSERVATIONS",f"{modele_log .n_observations }",key ="lg_n")
    metric_card (q2 ,"PSEUDO-R2 (McFADDEN)",f"{modele_log .pseudo_r2 :.3f}",key ="lg_r2")
    metric_card (q3 ,"TAUX DE BON CLASSEMENT",f"{modele_log .taux_bien_classes *100 :.1f} %",key ="lg_acc")
    st .caption (
    "Estimation par l'algorithme de Newton-Raphson : a chaque iteration, le probleme est "
    "reecrit comme une regression lineaire ponderee, les poids valant p(1-p), de sorte que les "
    "observations les plus incertaines -- celles dont la probabilite predite avoisine 0,5 -- "
    "pesent le plus. Le pseudo-R2 de McFadden ne s'interprete pas comme un R2 ordinaire : des "
    "valeurs de 0,2 a 0,4 y traduisent déjà un ajustement satisfaisant."
    )

    m_ref =st .select_slider (
    "Montant de référence pour l'optimisation",
    options =[250_000 ,500_000 ,1_000_000 ,2_000_000 ,5_000_000 ,10_000_000 ],
    value =1_000_000 ,format_func =lambda v :f"{v :,.0f} USD".replace (","," "),key ="lg_montant")
    sens_ref =st .radio ("Sens de l'opération",[True ,False ],
    format_func =lambda v :"Client achete"if v else "Client vend",
    horizontal =True ,key ="lg_sens")

    opt_log =flow .spread_optimal_logistique (modele_log ,float (m_ref ),bool (sens_ref ))
    st .plotly_chart (
    charts .logistique_chart (
    opt_log ["grille"].tolist (),opt_log ["probas"].tolist (),opt_log ["esperances"].tolist (),
    opt_log ["delta_optimal"],[p .delta for p in points ],[p .taux_acceptation for p in points ]),
    width='stretch' ,config ={"displayModeBar":False })

    l1 ,l2 ,l3 =st .columns (3 )
    metric_card (l1 ,"SPREAD OPTIMAL (LOGISTIQUE)",f"{opt_log ['delta_optimal_pips']:.2f} pips",key ="lg_opt")
    metric_card (l2 ,"SPREAD OPTIMAL (EXPONENTIEL)",f"{delta_opt *10_000 :.2f} pips",key ="lg_exp")
    metric_card (l3 ,"PROBABILITE A L'OPTIMUM",f"{opt_log ['proba_a_optimum']*100 :.1f} %",key ="lg_p")

    st .markdown (
    "**Arbitrage entre les deux approches.** La logistique est statistiquement plus rigoureuse : "
    "elle exploité chaque observation, intègre la taille et le sens de l'opération, et sa "
    "probabilite est bornee dans [0,1] par construction -- ce que `A x exp(-k delta)` ne "
    "garantit pas, cette derniere depassant 1 pour de petits écarts. En contrepartie, le modèle "
    "exponentiel offre la solution analytique `delta* = 1/k`, tandis que la logistique impose "
    "une optimisation numérique. C'est un arbitrage entre tractabilite théorique et rigueur "
    "statistique, non une superiorite absolue de l'une sur l'autre."
    )

    st .markdown ('<div class="section-title">GRILLE DEDUITE PAR TAILLE D\'Opération</div>',
    unsafe_allow_html =True )
    grille_log =flow .grille_optimale_par_taille (
    modele_log ,[250_000 ,500_000 ,1_000_000 ,2_000_000 ,5_000_000 ,10_000_000 ],bool (sens_ref ))
    dense_table (grille_log )

    st .markdown (
    f'<div style="background:{c ["amber"]}18; border-left:3px solid {c ["amber"]}; '
    f'padding:11px 15px; font-size:0.87rem;">'
    f'<b>Une tension a signaler.</b> Cette grille et celle du module RFQ obeissent a deux '
    f'logiques opposees. Les multiplicateurs du module RFQ <b>elargissent</b> le spread avec '
    f'la taille, au nom du <b>cout de couverture</b> : une operation importante est plus chere '
    f'et plus risquee a deboucler. La grille ci-dessus peut au contraire le <b>resserrer</b>, '
    f'au nom de l\'<b>elasticite de la demande</b> : les gros clients mettent plus '
    f'systematiquement les banques en concurrence, donc acceptent moins facilement un ecart '
    f'eleve. Les deux raisonnements sont legitimes et tirent en sens contraire. Un desk reel '
    f'arbitre entre les deux ; le modele ne tranche pas a sa place, il chiffre l\'un des deux '
    f'termes de l\'arbitrage.</div>',
    unsafe_allow_html =True )

    st .info (
    "Les demandes de cotation exploitées ici sont simulées, en l'absence "
    "d'historique réel de RFQ. La méthode de calibration est en revanche entièrement opérationnelle : "
    "dès que la salle enregistrera ses demandes de cotation acceptées **et refusées**, les mêmes "
    "calculs s'appliqueront à des données véritables."
    )


    # --------------------------------------------------------------------------- #
    # PTF — Risque de portefeuille multi-devises
    # --------------------------------------------------------------------------- #

def render_ptf (etat ,history :pd .DataFrame )->None :
    st .markdown ('<div class="section-title">PTF&lt;GO&gt; — Risque de portefeuille multi-devises</div>',
    unsafe_allow_html =True )
    c =theme .P ()

    st .caption (
    "Le dirham etant arrime a un panier 60 % EUR / 40 % USD, l'EUR/MAD et le USD/MAD ne sont pas "
    "deux paires indépendantes : elles sont liées par construction réglementaire. Le risque du "
    "portefeuille n'est donc pas la somme des risques individuels."
    )

    eurmad =db .get_eurmad_history ()
    if eurmad .empty :
        st .warning ("Historique EUR/MAD indisponible.")
        return 

    p1 ,p2 =st .columns (2 )
    pos_usd =p1 .number_input ("Position USD/MAD (USD)",value =float (etat .position ),
    step =250_000.0 ,key ="ptf_pos_usd")
    pos_eur =p2 .number_input ("Position EUR/MAD (EUR)",value =float (st .session_state ["position_eur"]),
    step =250_000.0 ,key ="ptf_pos_eur")
    st .session_state ["position_eur"]=pos_eur 

    # Volatilites et correlation estimees sur les historiques
    from core .models import volatilite 
    sigma_usd =etat .sigma 
    eurmad_renomme =eurmad .rename (columns ={"mid":"mid_marche"})
    sigma_eur =volatilite (eurmad_renomme )
    rho =portfolio .correlation_series (
    history .set_index ("date")["mid_marche"],eurmad .set_index ("date")["mid"]
    )

    st .markdown ('<div class="section-title">STRUCTURE DE RISQUE</div>',unsafe_allow_html =True )
    v1 ,v2 ,v3 =st .columns (3 )
    metric_card (v1 ,"SIGMA USD/MAD",f"{sigma_usd :.6f}",key ="ptf_su")
    metric_card (v2 ,"SIGMA EUR/MAD",f"{sigma_eur :.6f}",key ="ptf_se")
    metric_card (v3 ,"CORRELATION",f"{rho :+.3f}",
    delta_kind ="up"if rho >0 else "down",key ="ptf_rho")

    matrice =portfolio .construire_matrice (
    ["USD/MAD","EUR/MAD"],[sigma_usd ,sigma_eur ],{("USD/MAD","EUR/MAD"):rho }
    )
    st .plotly_chart (charts .correlation_heatmap (matrice .devises ,matrice .correlation ),
    width='stretch' ,config ={"displayModeBar":False })

    # ---- VaR agregee ---- #
    st .markdown ('<div class="section-title">VaR AGREGEE ET Bénéfice DE DIVERSIFICATION</div>',
    unsafe_allow_html =True )
    h1 ,_ =st .columns ([1 ,3 ])
    horizon =h1 .selectbox ("Horizon (jours)",[1 ,5 ,10 ],index =0 ,key ="ptf_horizon")
    vp =portfolio .var_portefeuille ([pos_usd ,pos_eur ],matrice ,horizon )

    w1 ,w2 ,w3 =st .columns (3 )
    metric_card (w1 ,f"VaR 99% PORTEFEUILLE",f"{vp .var_99 :,.0f} MAD".replace (","," "),key ="ptf_var99")
    metric_card (w2 ,"SOMME DES VaR INDIVIDUELLES",f"{vp .var_non_diversifiee :,.0f} MAD".replace (","," "),
    key ="ptf_varnd")
    metric_card (w3 ,"Bénéfice DE DIVERSIFICATION",
    f"{vp .benefice_diversification :,.0f} MAD".replace (","," "),
    delta_kind ="up"if vp .benefice_diversification >0 else "down",key ="ptf_benef")

    st .plotly_chart (charts .diversification_chart (vp .var_non_diversifiee ,vp .var_99 ),
    width='stretch' ,config ={"displayModeBar":False })

    if vp .contributions :
        contrib =pd .DataFrame ([
        {"DEVISE":k ,"CONTRIBUTION AU RISQUE":f"{v *100 :+.1f} %"}
        for k ,v in vp .contributions .items ()
        ])
        dense_table (contrib )
        st .caption (
        "La contribution mesure la part du risque total imputable a chaque paire. Une contribution "
        "negative signale une position qui **reduit** le risque global : elle agit comme une couverture."
        )

        # ---- Couverture croisee ---- #
    st .markdown ('<div class="section-title">COUVERTURE CROISEE</div>',unsafe_allow_html =True )
    hedge =portfolio .couverture_croisee (pos_usd ,sigma_usd ,sigma_eur ,rho )
    g1 ,g2 ,g3 =st .columns (3 )
    metric_card (g1 ,"RATIO OPTIMAL",f"{hedge .ratio_optimal :+.3f}",key ="ptf_ratio")
    metric_card (g2 ,"MONTANT DE COUVERTURE",f"{fmt_amount (hedge .montant_couverture )} EUR",key ="ptf_hedge")
    metric_card (g3 ,"REDUCTION DU RISQUE",f"{hedge .reduction_risque_pct :.1f} %",
    delta_kind ="up",key ="ptf_red")
    st .caption (
    "Ratio minimisant la variance : h* = -rho x (sigma_usd / sigma_eur). Interet pratique : lorsque "
    "les deux paires sont fortement correlees, une couverture partielle en EUR/MAD reduit le risque "
    "d'une position USD/MAD **sans avoir a la liquider** — donc sans subir le coût d'impact de marché "
    "modelise par Almgren & Chriss."
    )

    # ---- Exposition au panier ---- #
    st .markdown ('<div class="section-title">EXPOSITION AU PANIER DE Référence</div>',unsafe_allow_html =True )
    ep =portfolio .exposition_panier (pos_usd ,pos_eur )
    n1 ,n2 ,n3 =st .columns (3 )
    metric_card (n1 ,"POIDS USD (POSITION)",f"{ep ['poids_usd']*100 :.0f} %",
    delta =f"panier : {ep .get ('poids_panier_usd',0.4 )*100 :.0f} %",key ="ptf_pu")
    metric_card (n2 ,"POIDS EUR (POSITION)",f"{ep ['poids_eur']*100 :.0f} %",
    delta =f"panier : {ep .get ('poids_panier_eur',0.6 )*100 :.0f} %",key ="ptf_pe")
    metric_card (n3 ,"Écart AU PANIER",f"{ep ['ecart_panier']*100 :.0f} pts",key ="ptf_ecart")
    st .markdown (
    f'<div style="background:{c ["bg_panel"]}; border-left:3px solid {c ["cfg_red"]}; '
    f'padding:10px 14px; font-family:{theme .FONT_MONO }; font-size:0.86rem;">{ep ["commentaire"]}</div>',
    unsafe_allow_html =True ,
    )

    st .info (
    "**Limite** : l'historique EUR/MAD est reconstitue par croisement (EUR/MAD = EUR/USD x USD/MAD) "
    "faute de série observée. Les deux séries sont donc correlees par construction — ce qui reste "
    "cohérent avec la realite du panier, mais surestime probablement la correlation réelle."
    )


    # --------------------------------------------------------------------------- #
    # ATTR — Attribution du P&L et limites prudentielles
    # --------------------------------------------------------------------------- #

def render_attr (etat ,transactions :pd .DataFrame ,history :pd .DataFrame )->None :
    st .markdown ('<div class="section-title">ATTR&lt;GO&gt; — Attribution du P&amp;L</div>',
    unsafe_allow_html =True )
    c =theme .P ()

    st .caption (
    "Un P&L global ne dit pas si la banque a gagne parce qu'elle a bien cote, bien gere son "
    "inventaire, ou simplement parce que le marché est alle dans son sens. C'est la premiere "
    "question que pose un responsable de desk."
    )

    attr =attribution .attribuer_pnl (transactions ,history ,etat .mid_marche ,etat .gamma ,etat .sigma )

    a1 ,a2 ,a3 ,a4 =st .columns (4 )
    metric_card (a1 ,"MARGE DE SPREAD",f"{attr .marge_spread :+,.0f}".replace (","," "),
    delta_kind ="up"if attr .marge_spread >=0 else "down",key ="attr_spread")
    metric_card (a2 ,"EFFET DU SKEW",f"{attr .effet_skew :+,.0f}".replace (","," "),
    delta_kind ="up"if attr .effet_skew >=0 else "down",key ="attr_skew")
    metric_card (a3 ,"EFFET DE Marché",f"{attr .effet_marche :+,.0f}".replace (","," "),
    delta_kind ="up"if attr .effet_marche >=0 else "down",key ="attr_mkt")
    metric_card (a4 ,"P&L TOTAL",f"{attr .total :+,.0f} MAD".replace (","," "),
    delta_kind ="up"if attr .total >=0 else "down",key ="attr_total")

    st .plotly_chart (
    charts .attribution_waterfall (attr .marge_spread ,attr .effet_skew ,attr .effet_marche ,attr .total ),
    width='stretch' ,config ={"displayModeBar":False },
    )

    st .markdown (
    "**Lecture des trois composantes** \n\n"
    "- **Marge de spread** : le revenu structurel du métier de teneur de marché — acheter sous le "
    "mid, vendre au-dessus. Indépendant de toute vue directionnelle.\n"
    "- **Effet du skew** : le coût immédiat du décalage de prix consenti pour attirer un flux "
    "compensateur. Souvent négatif : la question est de savoir s'il est compensé par la réduction "
    "du risque.\n"
    "- **Effet de marché** : la réévaluation de la position ouverte. C'est la composante que le "
    "teneur de marché ne contrôle pas, et qu'il cherche précisément à minimiser."
    )

    # ---- Metriques quantifiees du backtest ---- #
    st .markdown ('<div class="section-title">METRIQUES QUANTIFIEES DU BACKTEST</div>',unsafe_allow_html =True )
    calib :CalibrationParams =st .session_state ["calib"]
    bt =backtest .run_backtest (history ,transactions ,calib ,etat .spread_marche )
    met =attribution .metriques_backtest (bt )

    if met is None :
        st .info ("Historique insuffisant pour calculer les métriques (3 jours minimum).")
    else :
        m1 ,m2 ,m3 ,m4 =st .columns (4 )
        metric_card (m1 ,"P&L DE Marché Cumulé",f"{met .pnl_marche_total :+,.0f}".replace (","," "),
        delta =f"sur {met .n_jours } jours",
        delta_kind ="up"if met .pnl_marche_total >=0 else "down",key ="bt_pnl")
        metric_card (m2 ,"ECART-TYPE QUOTIDIEN",f"{met .ecart_type_pnl :,.0f}".replace (","," "),key ="bt_sd")
        metric_card (m3 ,"SHARPE ANNUALISE",f"{met .sharpe_annualise :.2f}",
        delta_kind ="up"if met .sharpe_annualise >=0 else "down",key ="bt_sharpe")
        metric_card (m4 ,"Coût Cumulé DU SKEW",f"{met .cout_cumule_skew_mad :,.0f} MAD".replace (","," "),
        key ="bt_cout")

        detail =pd .DataFrame ([
        {"INDICATEUR":"Jours rejoués","VALEUR":f"{met .n_jours }"},
        {"INDICATEUR":"P&L moyen quotidien","VALEUR":f"{met .pnl_moyen_quotidien :+,.0f} MAD".replace (","," ")},
        {"INDICATEUR":"Meilleure journée","VALEUR":f"{met .meilleure_journee :+,.0f} MAD".replace (","," ")},
        {"INDICATEUR":"Pire journée","VALEUR":f"{met .pire_journee :+,.0f} MAD".replace (","," ")},
        {"INDICATEUR":"Position moyenne portée","VALEUR":f"{met .position_moyenne :,.0f} USD".replace (","," ")},
        {"INDICATEUR":"Position maximale","VALEUR":f"{met .position_max :,.0f} USD".replace (","," ")},
        {"INDICATEUR":"Skew moyen consenti","VALEUR":f"{met .skew_moyen_pips :.2f} pips"},
        {"INDICATEUR":"Skew maximal consenti","VALEUR":f"{met .skew_max_pips :.2f} pips"},
        {"INDICATEUR":"Jours en position longue","VALEUR":f"{met .jours_position_longue }"},
        {"INDICATEUR":"Jours en position courte","VALEUR":f"{met .jours_position_courte }"},
        ])
        dense_table (detail )

        st .info (
        "Ce backtest suppose que le flux client "
        "aurait été strictement identique avec et sans skew. Or c'est exactement l'inverse que le "
        "skew cherche à provoquer. Ces chiffres mesurent donc fidèlement le **coût** du skew — la "
        "marge sacrifiée, qui est observable — mais ne peuvent pas mesurer son **bénéfice**, qui "
        "supposerait de modéliser la réaction des clients au prix. C'est précisément l'objet de la "
        "page FLOW."
        )

        # ---- Limites prudentielles ---- #
    st .markdown ('<div class="section-title">LIMITES PRUDENTIELLES DE POSITION DE CHANGE</div>',
    unsafe_allow_html =True )
    f1 ,f2 ,f3 =st .columns (3 )
    fonds =f1 .number_input ("Fonds propres (MAD)",value =float (st .session_state ["fonds_propres"]),
    step =100_000_000.0 ,format ="%.0f",key ="attr_fp")
    lim_dev =f2 .number_input ("Limite par devise (% FP)",value =10.0 ,step =1.0 ,key ="attr_ld")
    lim_glob =f3 .number_input ("Limite globale (% FP)",value =20.0 ,step =1.0 ,key ="attr_lg")
    st .session_state ["fonds_propres"]=fonds 

    positions_mad ={
    "USD":etat .position *etat .mid_marche ,
    "EUR":st .session_state ["position_eur"]*etat .eur_usd *etat .mid_marche ,
    }
    ratios =attribution .ratios_position_change (positions_mad ,fonds ,lim_dev ,lim_glob )

    r1 ,r2 ,r3 =st .columns (3 )
    metric_card (r1 ,"RATIO PAR DEVISE",f"{ratios .ratio_par_devise :.2f} %",
    delta =f"limite : {lim_dev :.0f} %",key ="attr_rd")
    metric_card (r2 ,"RATIO GLOBAL",f"{ratios .ratio_global :.2f} %",
    delta =f"limite : {lim_glob :.0f} %",key ="attr_rg")
    with r3 :
        st .markdown (
        f'<div class="metric-card"><div class="metric-label">CONFORMITE</div>'
        f'<div style="margin-top:8px;">{badge_html (ratios .conforme_devise and ratios .conforme_global ,ok_text ="CONFORME",bad_text ="DEPASSEMENT")}</div></div>',
        unsafe_allow_html =True ,
        )

    st .caption (
    "Au-dela de la bande de fluctuation de +/-5 %, une banque marocaine est soumise a des limites "
    "sur sa position de change, exprimees en pourcentage de ses fonds propres. Convention retenue "
    "pour la position globale : maximum entre la somme des positions longues et celle des positions "
    "courtes. **Les seuils par defaut (10 % / 20 %) sont des ordres de grandeur usuels et doivent "
    "etre confirmes aupres de la Direction des Risques.**"
    )


    # --------------------------------------------------------------------------- #
    # GUIDE — parcours de premiere utilisation
    # --------------------------------------------------------------------------- #

def render_guide ()->None :
    c =theme .P ()
    st .markdown ('<div class="section-title">GUIDE — Prise en main du terminal</div>',
    unsafe_allow_html =True )

    st .markdown (
    f'<div style="background:{c ["bg_panel"]}; border-left:3px solid {c ["cfg_red"]}; '
    f'padding:14px 18px; margin-bottom:14px;">'
    f'<b>A quoi sert cet outil ?</b><br>'
    f'Coter un prix USD/MAD qui ne se contente pas de refleter le marche, mais qui integre '
    f'aussi le risque que la banque porte a cet instant. Un prix trop genereux erode la marge, '
    f'un prix trop prudent fait fuir le client : le terminal formalise cet arbitrage, que les '
    f'traders realisent aujourd\'hui de facon intuitive.</div>',
    unsafe_allow_html =True ,
    )

    st .markdown ('<div class="section-title">PARCOURS RECOMMANDE POUR UNE PREMIERE VISITE</div>',
    unsafe_allow_html =True )
    etapes =[
    ("1","DASH","Vue d'ensemble : cours du jour, position, prix affiche et contrôles réglementaires."),
    ("2","MKT<GO>","D'ou vient le prix : les cotations des neuf banques concurrentes, dont on tire le MID de marché."),
    ("3","POS<GO>","Comment la position se forme, transaction après transaction."),
    ("4","SKEW<GO>","Le coeur du modèle : comment le prix se decale selon la position detenue."),
    ("5","RFQ<GO>","Mise en pratique : coter un prix ferme a un client pour un montant donne."),
    ("6","LIQD<GO>","Si la position devient trop lourde : le plan de liquidation optimal."),
    ("7","EVENTS<GO>","Pour aller plus loin : trois études d'événement sur l'activité du desk."),
    ]
    for num ,code ,desc in etapes :
        st .markdown (
        f'<div style="display:flex; gap:12px; align-items:flex-start; padding:7px 0; '
        f'border-bottom:1px solid {c ["border"]};">'
        f'<div style="background:{c ["cfg_red"]}; color:white; font-family:{theme .FONT_MONO }; '
        f'font-weight:700; width:22px; height:22px; text-align:center; border-radius:3px; '
        f'flex-shrink:0; font-size:0.8rem; line-height:22px;">{num }</div>'
        f'<div><span style="font-family:{theme .FONT_MONO }; font-weight:700; '
        f'color:{c ["cfg_red"]};">{code }</span> &mdash; {desc }</div></div>',
        unsafe_allow_html =True ,
        )

    st .markdown ('<div class="section-title">LES VINGT-DEUX FONCTIONS, PAR FAMILLE</div>',
    unsafe_allow_html =True )

    familles =[
    ("COTATION ET Marché",[
    ("DASH","Tableau de bord : cours, position, prix final, conformité, cours croise EUR/MAD."),
    ("MKT<GO>","Cotations concurrentes (modifiables) et detection des cotations aberrantes."),
    ("RFQ<GO>","Cotation d'un prix ferme pour un montant client, avec grille par tranche."),
    ("FWD<GO>","Change a terme, swaps de change et echeancier des positions."),
    ]),
    ("POSITION ET Résultat",[
    ("POS<GO>","Journal des transactions, position cumulée et bouton de liquidation d'urgence."),
    ("PNL<GO>","Résultat : tresorerie encaissee et valorisation de la position ouverte."),
    ("ATTR<GO>","D'ou vient le résultat : marge de spread, effet du skew, effet de marché."),
    ]),
    ("COMPORTEMENT DE LA CLIENTELE",[
    ("FLOW<GO>","Spread optimal deduit de la sensibilite des clients au prix propose."),
    ("SEG<GO>","Segmentation de la clientele par apprentissage non supervise."),
    ]),
    ("RISQUE",[
    ("RISK<GO>","VaR, jauge de consommation de limite et historique des dépassements."),
    ("PTF<GO>","Risque agrege USD/MAD et EUR/MAD, correlation et couverture croisee."),
    ("STRESS<GO>","Scenarios de choc sur le cours et sur la volatilité."),
    ("LIQD<GO>","Plan de liquidation optimal d'une position."),
    ]),
    ("MODELISATION STATISTIQUE",[
    ("VOL<GO>","Volatilité historique et calibration des paramètres du pricer."),
    ("GARCH<GO>","Volatilité conditionnelle : le risque cesse d'etre une constante."),
    ("TS<GO>","Séries chronologiques : decomposition, stationnarite, retour a la moyenne."),
    ("SKEW<GO>","Décalage de prix selon la position, simulateur et test retrospectif."),
    ]),
    ("ÉTUDES D'ÉVÉNEMENT",[
    ("EVENTS<GO>","Trois études inspirées de l'étude dividendes : résilience du prix après un gros flux, dépassements de limite, persistance du comportement client."),
    ("THEORY<GO>","Logique, concepts et formules des études d'événement et de la génération des données de marché."),
    ]),
    ("REFERENCE",[
    ("AUDIT<GO>","Journal de conformité, en ecriture seule, pour chaque clôture de journée."),
    ("DOCS<GO>","Méthodologie détaillée et références academiques."),
    ("GUIDE","La page courante."),
    ]),
    ]

    for titre ,items in familles :
        st .markdown (
        f'<div style="font-family:{theme .FONT_MONO }; font-size:0.72rem; font-weight:700; '
        f'color:{c ["text_secondary"]}; letter-spacing:0.08em; margin:14px 0 6px 0;">{titre }</div>',
        unsafe_allow_html =True ,
        )
        dense_table (pd .DataFrame ([{"FONCTION":code ,"ROLE":desc }for code ,desc in items ]))

    st .markdown ('<div class="section-title">CE QUE CET OUTIL N\'EST PAS</div>',unsafe_allow_html =True )
    st .markdown (
    "L'honnetete sur les limites vaut mieux qu'une confiance mal placee. Ce terminal reproduit "
    "fidèlement les **mecanismes de decision** d'un desk de change, mais :\n\n"
    "- les cotations concurrentes sont **saisies manuellement**, sans flux de marché temps réel "
    "(seul l'EUR/USD peut etre recupere en direct) ;\n"
    "- il **ne passe aucun ordre** : il propose des prix, il ne les execute pas ;\n"
    "- les demandes de cotation exploitées par les modules FLOW et SEG sont **simulees**, faute "
    "d'historique réel disponible : la méthode est opérationnelle, les données ne le sont pas encore ;\n"
    "- l'historique long utilise par GARCH et TS est **génère**, un echantillon réel de quelques "
    "semaines etant trop court pour une estimation significative.\n\n"
    "Chaque page concernee rappelle cette limite a l'endroit ou elle s'applique."
    )


    # --------------------------------------------------------------------------- #
    # GARCH — volatilite conditionnelle
    # --------------------------------------------------------------------------- #

@st .cache_data (show_spinner =False )
def _serie_simulee_cache (n_jours :int ,seed :int )->pd .DataFrame :
    """Série simulee mise en cache : sa génération est deterministe a graine fixee."""
    return vol_models .simuler_serie_garch (n_jours =n_jours ,seed =seed )


@st .cache_data (show_spinner =False )
def _ajuster_modeles_garch (eps :np .ndarray ):
    """Estimation des trois specifications, mise en cache.

    Sans cache, les trois maximisations de vraisemblance seraient relancees a
    chaque interaction avec un widget, Streamlit reexecutant l'integralite du
    script. Les entrees etant identiques, le résultat l'est aussi : le cache
    est ici sans effet de bord.
    """
    return (vol_models .ajuster_garch11 (eps ),
    vol_models .ajuster_gjr_garch (eps ),
    vol_models .ajuster_egarch (eps ))


def _serie_garch_active (history :pd .DataFrame ):
    """Renvoie (série utilisee, libelle, est_simulee) selon le choix de l'utilisateur."""
    if st .session_state .get ("garch_source","simulee")=="reelle":
        return history ,"Historique observe",False 
    n =st .session_state .get ("garch_n_jours",1200 )
    return _serie_simulee_cache (n ,42 ),f"Serie simulee ({n } jours)",True 


def render_garch (etat )->None :
    c =theme .P ()
    st .markdown ('<div class="section-title">GARCH&lt;GO&gt; — Volatilité conditionnelle</div>',
    unsafe_allow_html =True )

    st .markdown (
    f'<div style="font-family:{theme .FONT_SANS }; color:{c ["text_secondary"]}; '
    f'font-size:0.88rem; margin-bottom:12px;">'
    f'Le pricer estime aujourd\'hui sigma comme un ecart-type glissant, donc une <b>constante</b>. '
    f'Cela revient a supposer qu\'il est aussi risque de porter une position un jour calme qu\'un '
    f'jour de crise. Les modeles GARCH rendent la variance <b>conditionnelle</b> a l\'information '
    f'passee, et font ainsi reagir tout le pricer au regime de marche.</div>',
    unsafe_allow_html =True ,
    )

    s1 ,s2 =st .columns ([1.4 ,1 ])
    source =s1 .radio ("Série utilisee pour l'estimation",
    ["simulee","reelle"],
    format_func =lambda v :"Série simulee longue (recommande)"if v =="simulee"
    else "Historique réel de l'application",
    horizontal =True ,key ="garch_source")
    if source =="simulee":
        s2 .slider ("Longueur (jours)",300 ,2500 ,1200 ,step =100 ,key ="garch_n_jours")

    serie ,libelle ,simulee =_serie_garch_active (history =db .get_price_history ())
    eps =vol_models .variations_normalisees (serie )

    if len (eps )<40 :
        st .warning (
        f"{libelle } : {len (eps )} variations exploitables. Une estimation GARCH exige "
        "plusieurs centaines d'observations pour produire des coefficients significatifs. "
        "Basculer sur la série simulee pour une demonstration validé."
        )
        return 

    if simulee :
        st .info (
        "**Pourquoi une série simulee ?** L'historique réel disponible ne compte que quelques "
        "dizaines d'observations, alors qu'une estimation GARCH stable en demande plusieurs "
        "centaines. Estimer sur un echantillon aussi court produirait des coefficients denues "
        "de signification. Cette série, générée par un GARCH(1,1) de paramètres connus, permet "
        "de **verifier que la chaine d'estimation fonctionne** : les coefficients retrouves "
        "doivent etre proches de ceux ayant servi a la génération. Elle n'intervient jamais "
        "dans le calcul du prix affiche."
        )

    with st .spinner ("Estimation par maximum de vraisemblance..."):
        r_garch ,r_gjr ,r_egarch =_ajuster_modeles_garch (eps )

    if r_garch is None :
        st .warning ("L'estimation n'a pas abouti sur cette série.")
        return 

    st .markdown ('<div class="section-title">COMPARAISON DES SPECIFICATIONS</div>',unsafe_allow_html =True )
    dense_table (vol_models .comparer_modeles ([r_garch ,r_gjr ,r_egarch ]))
    st .caption (
    "Le critere AIC arbitre entre qualite d'ajustement et parcimonie : un modèle plus riche "
    "n'est retenu que si le gain de vraisemblance justifie ses paramètres supplementaires. "
    "Sur une paire de devises, un terme d'asymetrie faible n'est pas un echec du modèle : "
    "l'effet de levier est un fait stylise des marchés d'actions, ou une baisse degrade le "
    "bilan des emetteurs. Sur le change, la notion de mauvaise nouvelle est ambigue, une "
    "baisse du USD/MAD penalisant l'exportateur mais favorisant l'importateur."
    )

    choix =st .selectbox ("Modèle retenu pour la suite",
    [r .modele for r in (r_garch ,r_gjr ,r_egarch )if r is not None ],
    key ="garch_choix")
    res ={r .modele :r for r in (r_garch ,r_gjr ,r_egarch )if r is not None }[choix ]

    st .markdown ('<div class="section-title">Paramètres ESTIMES</div>',unsafe_allow_html =True )
    cols =st .columns (len (res .params )+1 )
    for i ,(nom ,val )in enumerate (res .params .items ()):
        fmt =f"{val :.4f}"if abs (val )>=1e-3 else f"{val :.3e}"
        metric_card (cols [i ],nom .upper (),fmt ,key =f"gp_{nom }")
    metric_card (cols [-1 ],"PERSISTANCE",f"{res .persistance :.4f}",
    delta =f"demi-vie {res .demi_vie_choc :.1f} j"if np .isfinite (res .demi_vie_choc )else None ,
    key ="gp_pers")
    st .caption (
    "alpha mesure la réaction immédiate a un choc, beta la persistance du regime de volatilité. "
    "Leur somme donne la persistance globale : proche de 1, les chocs mettent longtemps a se "
    "résorber. La demi-vie indique en combien de jours la moitie d'un choc de volatilité "
    "s'estompe."
    )

    sigma_cst =float (np .std (eps ))
    sigma_cond_now =float (res .sigma_conditionnel [-1 ])
    sigma_lt =math .sqrt (res .variance_long_terme )if np .isfinite (res .variance_long_terme )else None 

    st .plotly_chart (
    charts .volatilite_conditionnelle_chart (
    serie ["date"].tolist ()[-len (res .sigma_conditionnel ):],
    res .sigma_conditionnel .tolist (),sigma_cst ,sigma_lt ),
    width='stretch' ,config ={"displayModeBar":False })

    st .markdown ('<div class="section-title">PREVISION ET DIAGNOSTIC</div>',unsafe_allow_html =True )
    g1 ,g2 =st .columns (2 )
    with g1 :
        prev =vol_models .prevoir_volatilite (res ,eps ,10 )
        st .plotly_chart (
        charts .prevision_volatilite_chart (res .sigma_conditionnel [-40 :].tolist (),
        prev .tolist (),sigma_lt or sigma_cst ),
        width='stretch' ,config ={"displayModeBar":False })
        st .caption ("La prevision converge vers le niveau de long terme : après un choc, le modèle "
        "anticipe une normalisation progressive plutot qu'une extrapolation.")
    with g2 :
        st .plotly_chart (charts .residus_hist_chart (res .residus_standardises .tolist ()),
        width='stretch' ,config ={"displayModeBar":False })
        m_res =float (np .mean (res .residus_standardises ))
        s_res =float (np .std (res .residus_standardises ))
        st .caption (
        f"Residus standardises : moyenne {m_res :+.3f}, ecart-type {s_res :.3f} (attendus 0 et 1). "
        "Un écart-type proche de 1 indique que le modèle a correctement absorbe la structure "
        "de volatilité. Les queues plus epaisses que la normale sont usuelles en finance."
        )

    st .markdown ('<div class="section-title">CONSEQUENCE SUR LE PRICER</div>',unsafe_allow_html =True )
    calib :CalibrationParams =st .session_state ["calib"]
    imp =vol_models .impact_sur_pricer (
    sigma_cst =etat .sigma ,sigma_cond =sigma_cond_now ,gamma =etat .gamma ,
    position =etat .position ,eta =etat .eta ,lambda_ =etat .lambda_ )

    st .markdown (
    f'<div style="background:{c ["cfg_red"]}14; border-left:3px solid {c ["cfg_red"]}; '
    f'padding:10px 14px; font-size:0.87rem; margin-bottom:10px;">'
    f'<b>Le point cle : gamma reste inchange.</b> Il a ete calibre sur un objectif de gestion '
    f'(un decalage cible a la position limite), qui releve d\'une politique de risque et n\'a '
    f'aucune raison de varier au jour le jour. Seule la volatilite devient conditionnelle. '
    f'La consequence est automatique : k = gamma x sigma&sup2; augmente en regime volatil, donc '
    f'le skew s\'elargit et la banque incite plus fortement au reequilibrage, precisement '
    f'lorsque porter l\'inventaire coute le plus cher.</div>',
    unsafe_allow_html =True )

    comp =pd .DataFrame ([
    {"GRANDEUR":"Sigma","VOLATILITE CONSTANTE":f"{imp .sigma_constant :.6f}",
    "VOLATILITE CONDITIONNELLE":f"{imp .sigma_conditionnel :.6f}",
    "RAPPORT":f"x{imp .ratio :.2f}"},
    {"GRANDEUR":"k = gamma x sigma^2","VOLATILITE CONSTANTE":f"{imp .k_constant :.6f}",
    "VOLATILITE CONDITIONNELLE":f"{imp .k_conditionnel :.6f}",
    "RAPPORT":f"x{imp .k_conditionnel /imp .k_constant :.2f}"if imp .k_constant else "-"},
    {"GRANDEUR":"Skew applique","VOLATILITE CONSTANTE":f"{imp .skew_constant_pips :+.2f} pips",
    "VOLATILITE CONDITIONNELLE":f"{imp .skew_conditionnel_pips :+.2f} pips",
    "RAPPORT":f"x{imp .skew_conditionnel_pips /imp .skew_constant_pips :.2f}"if imp .skew_constant_pips else "-"},
    {"GRANDEUR":"kappa (liquidation)","VOLATILITE CONSTANTE":f"{imp .kappa_constant :.4f}",
    "VOLATILITE CONDITIONNELLE":f"{imp .kappa_conditionnel :.4f}",
    "RAPPORT":f"x{imp .kappa_conditionnel /imp .kappa_constant :.2f}"if imp .kappa_constant else "-"},
    {"GRANDEUR":"VaR 99 % a 1 jour","VOLATILITE CONSTANTE":f"{imp .var99_constant :,.0f} MAD".replace (","," "),
    "VOLATILITE CONDITIONNELLE":f"{imp .var99_conditionnel :,.0f} MAD".replace (","," "),
    "RAPPORT":f"x{imp .var99_conditionnel /imp .var99_constant :.2f}"if imp .var99_constant else "-"},
    ])
    dense_table (comp )

    if simulee :
        st .warning (
        "Les grandeurs de la colonne conditionnelle sont issues de la série simulee et ne "
        "doivent pas etre lues comme la volatilité réelle du USD/MAD. Elles illustrent "
        "l'ampleur de l'effet, pas son niveau."
        )


        # --------------------------------------------------------------------------- #
        # TS — series chronologiques
        # --------------------------------------------------------------------------- #

def render_ts (etat )->None :
    c =theme .P ()
    st .markdown ('<div class="section-title">TS&lt;GO&gt; — Séries chronologiques</div>',
    unsafe_allow_html =True )

    st .markdown (
    f'<div style="background:{c ["bg_panel"]}; border-left:3px solid {c ["cfg_red"]}; '
    f'padding:12px 16px; font-size:0.87rem; margin-bottom:12px;">'
    f'<b>Une precaution liminaire.</b> Predire un cours de change est repute sans valeur : '
    f'depuis Meese et Rogoff (1983), la litterature etablit qu\'aucun modele econometrique ne '
    f'bat durablement la marche aleatoire sur les grandes devises. Appliquer un ARIMA au '
    f'USD/MAD pour prevoir le cours de demain serait un contresens.<br><br>'
    f'Le dirham n\'est cependant pas en flottement libre : il est arrime a un panier et evolue '
    f'dans une bande geree par Bank Al-Maghrib. Cette architecture cree un argument '
    f'<b>structurel</b> de retour a la moyenne. La question posee ici n\'est donc pas '
    f'\\"quel sera le cours demain\\", mais <b>le USD/MAD revient-il vers son niveau central, '
    f'et a quelle vitesse ?</b></div>',
    unsafe_allow_html =True )

    src =st .radio ("Serie analysee",["simulee","reelle"],
    format_func =lambda v :"Série simulee longue"if v =="simulee"
    else "Historique réel de l'application",
    horizontal =True ,key ="ts_source")
    serie =(_serie_simulee_cache (800 ,11 )if src =="simulee"
    else db .get_price_history ())

    if serie is None or len (serie )<30 :
        st .warning ("Historique trop court pour une analyse de série chronologique (30 points minimum).")
        return 

    y =serie ["mid_marche"].to_numpy (dtype =float )

    # ---- Decomposition ---- #
    st .markdown ('<div class="section-title">DECOMPOSITION TENDANCE / RESIDU</div>',unsafe_allow_html =True )
    fen =st .slider ("Fenetre de la moyenne mobile (jours)",5 ,60 ,21 ,key ="ts_fenetre")
    dec =timeseries .decomposer (serie ,fen )
    if dec is not None :
        st .plotly_chart (charts .decomposition_chart (dec .dates ,dec .serie .tolist (),dec .tendance .tolist ()),
        width='stretch' ,config ={"displayModeBar":False })
        st .plotly_chart (charts .residu_chart (dec .dates ,dec .residu .tolist ()),
        width='stretch' ,config ={"displayModeBar":False })
        st .caption (
        f"La tendance explique {dec .part_tendance *100 :.1f} % de la variance de la serie. "
        "La composante saisonniere n'est pas estimée : elle n'a pas de sens economique sur un "
        "cours de change quotidien, il n'existe pas de mois du dirham fort."
        )

        # ---- Correlogrammes ---- #
    st .markdown ('<div class="section-title">AUTOCORRELATIONS</div>',unsafe_allow_html =True )
    bande =timeseries .bande_confiance_acf (len (y ))
    a1 ,a2 =st .columns (2 )
    with a1 :
        st .plotly_chart (charts .correlogramme_chart (timeseries .acf (y ,20 ).tolist (),bande ,"ACF"),
        width='stretch' ,config ={"displayModeBar":False })
    with a2 :
        st .plotly_chart (charts .correlogramme_chart (timeseries .pacf (y ,20 ).tolist (),bande ,"PACF"),
        width='stretch' ,config ={"displayModeBar":False })
    st .caption (
    "L'autocorrelation partielle mesure le lien entre deux dates une fois retire l'effet des "
    "retards intermediaires. Une PACF qui s'annule au-dela du premier retard est la signature "
    "d'un processus autoregressif d'ordre 1. Les barres depassant la bande jaune sont "
    "significatives au seuil de 5 %."
    )

    # ---- Stationnarite ---- #
    st .markdown ('<div class="section-title">TEST DE RACINE UNITAIRE</div>',unsafe_allow_html =True )
    df_test =timeseries .test_dickey_fuller (y ,n_retards =1 )
    if df_test is not None :
        t1 ,t2 ,t3 =st .columns (3 )
        metric_card (t1 ,"STATISTIQUE DE TEST",f"{df_test .statistique :.3f}",key ="ts_stat")
        metric_card (t2 ,"VALEUR CRITIQUE A 5 %",f"{df_test .valeurs_critiques ['5%']:.2f}",key ="ts_vc")
        with t3 :
            st .markdown (
            f'<div class="metric-card"><div class="metric-label">CONCLUSION</div>'
            f'<div style="margin-top:8px;">'
            f'{badge_html (df_test .stationnaire_5pct ,ok_text ="STATIONNAIRE",bad_text ="MARCHE ALEATOIRE")}'
            f'</div></div>',unsafe_allow_html =True )
        st .caption (
        f"{df_test .conclusion } L'hypothese nulle est la presence d'une racine unitaire : la "
        "série serait alors une marché aléatoire, dont les chocs sont permanents. La rejeter "
        "revient a conclure a l'existence d'un retour a la moyenne. Sous cette hypothèse nulle, "
        "la statistique ne suit pas une loi de Student, d'ou l'usage de la table de Fuller."
        )

        # ---- Selection d'ordre AR ---- #
    st .markdown ('<div class="section-title">Modélisation AUTOREGRESSIVE</div>',unsafe_allow_html =True )
    ordres =timeseries .selectionner_ordre_ar (y ,5 )
    if not ordres .empty :
        aff =ordres .drop (columns =["_p"]).copy ()
        aff ["AIC"]=aff ["AIC"].map (lambda v :f"{v :,.1f}".replace (","," "))
        aff ["BIC"]=aff ["BIC"].map (lambda v :f"{v :,.1f}".replace (","," "))
        aff ["SIGMA RESIDU"]=aff ["SIGMA RESIDU"].map (lambda v :f"{v :.6f}")
        dense_table (aff )
        st .caption ("Ordres compares par AIC croissant : le premier de la liste offre le meilleur "
        "compromis entre ajustement et parcimonie.")

        # ---- Retour a la moyenne ---- #
    st .markdown ('<div class="section-title">VITESSE DE RETOUR A LA MOYENNE</div>',unsafe_allow_html =True )
    ou =timeseries .calibrer_ornstein_uhlenbeck (y )
    if ou is not None :
        o1 ,o2 ,o3 ,o4 =st .columns (4 )
        metric_card (o1 ,"PHI (AR1)",f"{ou .phi :.4f}",key ="ou_phi")
        metric_card (o2 ,"VITESSE DE RAPPEL",f"{ou .theta :.4f}",key ="ou_theta")
        metric_card (o3 ,"NIVEAU D'EQUILIBRE",f"{ou .mu :.4f}",key ="ou_mu")
        metric_card (o4 ,"DEMI-VIE",
        f"{ou .demi_vie :.1f} j"if np .isfinite (ou .demi_vie )else "infinie",
        delta_kind ="up"if ou .demi_vie <30 else "down",key ="ou_hl")

        st .plotly_chart (charts .reversion_chart (
        serie ["date"].tolist (),y .tolist (),ou .mu ,float (np .std (y ))),
        width='stretch' ,config ={"displayModeBar":False })

        st .markdown (
        "Le processus d'Ornstein-Uhlenbeck s'ecrit `dS = theta (mu - S) dt + sigma dW`. Sa "
        "discretisation exacte est un AR(1), ce qui permet d'estimer ses paramètres par "
        "moindres carres, puis d'en deduire `theta = -ln(phi)` et la demi-vie `ln(2)/theta`.\n\n"
        "**Portee opérationnelle** : la demi-vie indique en combien de jours la moitie d'un "
        "écart au niveau d'equilibre se résorbe. Une demi-vie courte rend le portage d'une "
        "position moins risque, et peut justifier une liquidation plus patiente ; une demi-vie "
        "très longue signale au contraire l'absence de reversion exploitable, et plaide pour "
        "un débouclage rapide."
        )

        # ---- Ecart au panier ---- #
    st .markdown ('<div class="section-title">Écart AU NIVEAU CENTRAL DU PANIER</div>',unsafe_allow_html =True )
    ec =timeseries .ecart_au_panier (etat .mid_marche ,etat .mid_theo )
    e1 ,e2 ,e3 =st .columns (3 )
    metric_card (e1 ,"Écart AU NIVEAU CENTRAL",f"{ec ['ecart_pct']:+.3f} %",key ="ts_ec")
    metric_card (e2 ,"POSITION DANS LA BANDE",f"{ec ['position_dans_bande']*100 :.1f} %",
    delta ="0 % = borne basse, 100 % = borne haute",key ="ts_pb")
    metric_card (e3 ,"MARGE AVANT LA BORNE",f"{ec ['marge_avant_borne_pct']:.2f} pts",key ="ts_mb")
    st .caption (
    "C'est cet écart, et non le cours lui-même, qui constitue le candidat naturel a une "
    "reversion : le mecanisme de bande de Bank Al-Maghrib porte précisément sur la distance "
    "au cours central implique par le panier."
    )


    # --------------------------------------------------------------------------- #
    # SEG — segmentation de la clientele
    # --------------------------------------------------------------------------- #

def render_seg (etat )->None :
    c =theme .P ()
    st .markdown ('<div class="section-title">SEG&lt;GO&gt; — Segmentation de la clientele</div>',
    unsafe_allow_html =True )

    st .markdown (
    f'<div style="font-family:{theme .FONT_SANS }; color:{c ["text_secondary"]}; '
    f'font-size:0.88rem; margin-bottom:12px;">'
    f'Les tranches de cotation du module RFQ reposent sur des seuils de montant fixes et des '
    f'multiplicateurs poses a dire d\'expert. C\'est exactement le probleme resolu pour le '
    f'coefficient gamma, derive d\'un objectif de gestion au lieu d\'etre choisi : la meme '
    f'demarche s\'applique ici, en laissant la structure emerger du comportement observe des '
    f'clients plutot qu\'en la postulant.</div>',
    unsafe_allow_html =True )

    rfq_log =db .get_rfq_log ()
    profils =segmentation .construire_profils (rfq_log )
    if profils is None or len (profils )<4 :
        st .warning ("Trop peu de clients pour une segmentation (quatre minimum, avec au moins trois demandes chacun).")
        return 

    st .markdown ('<div class="section-title">PROFILS COMPORTEMENTAUX</div>',unsafe_allow_html =True )
    aff =pd .DataFrame ({
    "CLIENT":profils ["client"],
    "MONTANT MOYEN":profils ["montant_moyen"].map (lambda v :f"{v :,.0f} USD".replace (","," ")),
    "DEMANDES":profils ["n_demandes"],
    "TAUX D'ACCEPTATION":profils ["taux_acceptation"].map (lambda v :f"{v *100 :.1f} %"),
    "BIAIS ACHETEUR":profils ["biais_achat"].map (lambda v :f"{v *100 :.0f} %"),
    })
    dense_table (aff )
    st .caption (
    "Quatre variables comportementales : le montant moyen approche la taille du client, le "
    "nombre de demandes son activite, le taux d'acceptation sa sensibilite au prix, et le "
    "biais directionnel distingue un importateur structurellement acheteur de devises d'un "
    "exportateur vendeur. Cette derniere variable a une portee directe : un client toujours "
    "acheteur deplace la position du desk dans le même sens, ce qui pese davantage sur "
    "l'inventaire qu'un flux equilibre."
    )

    X =profils [["montant_moyen","n_demandes","taux_acceptation","biais_achat"]].to_numpy (dtype =float )
    Xs ,_ ,_ =segmentation .standardiser (X )
    st .caption (
    "Les variables sont centrees et reduites avant le partitionnement : le k-means repose sur "
    "des distances euclidiennes, et sans normalisation une variable en millions de dollars "
    "ecraserait completement une variable exprimee en pourcentage."
    )

    st .markdown ('<div class="section-title">CHOIX DU NOMBRE DE GROUPES</div>',unsafe_allow_html =True )
    ks =tuple (k for k in (2 ,3 ,4 ,5 )if k <len (profils ))
    tbl =segmentation .choisir_k (Xs ,ks )
    if tbl .empty :
        st .warning ("Partitionnement impossible sur cet echantillon.")
        return 
    dense_table (tbl )
    st .plotly_chart (
    charts .silhouette_chart (tbl ["NOMBRE DE GROUPES"].tolist (),tbl ["SILHOUETTE"].tolist ()),
    width='stretch' ,config ={"displayModeBar":False })
    st .caption (
    "Le coefficient de silhouette compare, pour chaque client, sa distance moyenne aux membres "
    "de son groupe a celle du groupe voisin le plus proche. Au-dela de 0,5 la structure est "
    "nette, entre 0,25 et 0,5 elle reste raisonnable, en dessous le partitionnement n'apporte "
    "pas grand-chose."
    )

    k_defaut =int (tbl .iloc [0 ]["NOMBRE DE GROUPES"])
    k =st .slider ("Nombre de segments retenu",min (ks ),max (ks ),k_defaut ,key ="seg_k")
    annotes ,res =segmentation .segmenter (profils ,k )
    if annotes is None :
        st .warning ("Partitionnement impossible avec ce nombre de groupes.")
        return 

    st .markdown ('<div class="section-title">SEGMENTS OBTENUS</div>',unsafe_allow_html =True )
    dense_table (segmentation .decrire_segments (annotes ))
    st .plotly_chart (charts .segments_scatter (
    annotes ["montant_moyen"].tolist (),annotes ["taux_acceptation"].tolist (),
    annotes ["segment"].tolist (),annotes ["client"].tolist (),annotes ["n_demandes"].tolist ()),
    width='stretch' ,config ={"displayModeBar":False })
    st .caption (
    "La taille des points reflete le nombre de demandes. Les etiquettes de segment sont posees "
    "**après** le partitionnement, a partir du montant moyen de chaque groupe : l'algorithme "
    "n'a recu aucune etiquette préalable, il n'a vu que des comportements."
    )

    st .info (
    "**Limite** : les demandes de cotation exploitées ici sont simulees a partir de profils de "
    "contreparties construits pour la demonstration. Que l'algorithme retrouve ces profils "
    "validé la **méthode**, pas la structure réelle de la clientele de CFG Bank. Sur des "
    "données véritables, la segmentation obtenue pourrait etre toute autre."
    )


def render_docs ()->None :
    st .markdown ('<div class="section-title">DOCS&lt;GO&gt; — Documentation méthodologique</div>',unsafe_allow_html =True )
    st .markdown (
    """
### Contexte

Ce terminal a ete développé dans le cadre d'un Projet de Fin d'Année (PFA) au sein de la salle des marchés
de CFG Bank, banque marocaine cotée a la Bourse de Casablanca. Il formalise un modèle de pricing FX
USD/MAD validé au préalable sous Excel, avec gestion dynamique de la position et des contrôles de
conformité réglementaire.

### Modèle de décalage de prix — Ho & Stoll (1981)

Thomas S. Y. Ho et Hans R. Stoll, *"Optimal dealer pricing under transactions and return uncertainty"*,
Journal of Financial Economics, 1981. Le teneur de marché ajuste son prix de reservation en fonction de
son inventaire (position) et de son aversion au risque (gamma) pour ramener sa position vers zero.

### Modèle de liquidation optimale — Almgren & Chriss (2000)

Robert Almgren et Neil Chriss, *"Optimal exécution of portfolio transactions"*, Journal of Risk, 2000.
Le modèle détermine la trajectoire de liquidation qui minimise un compromis entre le coût d'impact de
marché (exécution rapide) et le risque de détenir la position plus longtemps (exécution lente),
paramètre par l'aversion au risque (lambda), l'impact de marché (eta) et la volatilité (sigma).

### Segmentation par taille d'opération — Barzykin, Bergault & Gueant (2021-2023)

*"Algorithmic market making in dealer markets with hedging and market impact"* (arXiv:2112.02269),
et Bergault & Gueant, *"Size matters for OTC market makers"* (arXiv:1907.01225). Ces travaux
formalisent, pour le market making de change de gre a gre, l'élargissement du spread avec la
taille de l'opération, qui fonde la grille de cotation par tranche du module RFQ.

---

## Modélisation statistique

### Volatilité conditionnelle — famille GARCH

Le pricer estimait initialement &sigma; par un écart-type glissant, donc une **constante**. Cela
revient a supposer qu'il est aussi risque de porter une position un jour calme qu'un jour de crise,
hypothèse que les séries financieres contredisent : les fortes variations se regroupent dans le
temps.

- **GARCH(1,1)**, Bollerslev (1986) : &sigma;&sup2;&#8348; = &omega; + &alpha;&epsilon;&sup2;&#8348;&#8331;&#8321; + &beta;&sigma;&sup2;&#8348;&#8331;&#8321;
- **GJR-GARCH**, Glosten, Jagannathan & Runkle (1993) : ajoute un terme d'asymetrie declenche par
  les chocs negatifs
- **EGARCH**, Nelson (1991) : specifie sur le logarithme de la variance, positive par construction

L'estimation est réalisée par maximum de vraisemblance, implementee directement plutot que via une
bibliotheque dédiée. Une normalisation préalable de la série est nécessaire : a l'echelle des
variations du USD/MAD (de l'ordre de 10&#8315;&#8308; MAD), les gradients de la log-vraisemblance sont
numeriquement negligeables et l'optimiseur reste bloque sur son point de depart.

**Consequence sur le pricer** : &gamma; reste calibré sur un objectif de gestion et ne varie pas ;
seule &sigma; devient conditionnelle. Le skew s'elargit donc automatiquement en regime volatil, et
la liquidation s'accelere.

### Séries chronologiques

Predire un cours de change est repute sans valeur depuis Meese & Rogoff (1983). La question posee
ici est differente et testable : le USD/MAD, arrime a un panier et encadre par une bande, revient-il
vers son niveau central ?

- **Decomposition** tendance / residu par moyenne mobile centree (sans composante saisonniere,
  depourvue de sens sur un cours de change quotidien)
- **ACF et PACF**, cette derniere par l'algorithme de Durbin-Levinson
- **Test de Dickey-Fuller augmente** : l'hypothèse nulle de racine unitaire, si elle est rejetee,
  etablit l'existence d'un retour a la moyenne. La statistique ne suivant pas une loi de Student
  sous H&#8320;, les valeurs critiques de Fuller (1976) sont utilisees
- **Processus d'Ornstein-Uhlenbeck** : `dS = &theta;(&mu; - S)dt + &sigma;dW`, dont la discretisation
  exacte est un AR(1). On en deduit la vitesse de rappel et la demi-vie `ln(2)/&theta;`

### Modélisation du comportement de la clientele

- **Intensite exponentielle** `&lambda;(&delta;) = A e^(-k&delta;)`, calibrée par regression
  log-lineaire sur les taux d'acceptation agreges. Optimum analytique : `&delta;* = 1/k`
- **Regression logistique** sur les demandes individuelles, estimée par Newton-Raphson (IRLS),
  avec l'écart propose, le logarithme du montant et le sens comme variables explicatives

### Apprentissage non supervise

- **Segmentation par k-means** (initialisation k-means++, selection du nombre de groupes par le
  coefficient de silhouette) sur quatre variables comportementales, afin de deriver les tranches de
  cotation du comportement observé plutot que de les postuler
- **Detection d'anomalies** par score z modifie (Iglewicz & Hoaglin, 1993), fonde sur la médiane et
  une mesure de dispersion robuste, pour signaler les cotations concurrentes aberrantes avant
  qu'elles ne contaminent le prix affiche

---

## Sections implementees

1. MID théorique (regression quadratique sur EUR/USD)
2. Bande de fluctuation réglementaire (&plusmn;5%)
3. Analyse du marché concurrentiel (MID/spread par banque)
4. Prix de base CFG (avant gestion de position)
5. Position (journal de transactions, regle de signe)
6. Volatilité (&sigma;), normalisee par la racine du nombre de jours ecoules
7-10. Calibration de &gamma;, &eta;, &lambda;
8. Skew Ho & Stoll
11. Trajectoire de liquidation Almgren & Chriss
12. Contrôles de conformité
13. Historisation quotidienne automatique
16. Backtesting point-in-time du skew
17. Journal d'audit
18. P&L (flux de tresorerie + mark-to-market)
19. Tests de resistance
20. Change a terme, swaps et echeancier
21. VaR de portefeuille, correlation et couverture croisee
22. Attribution du résultat et ratios prudentiels
23. Volatilité conditionnelle (GARCH, GJR-GARCH, EGARCH)
24. Séries chronologiques (decomposition, stationnarite, retour a la moyenne)
25. Segmentation de la clientele et detection d'anomalies

---

## Ce que cet outil n'est pas

Ce terminal reproduit fidèlement les mecanismes de decision d'un desk de change, mais les cotations
concurrentes sont saisies manuellement, aucun ordre n'est execute, et les données alimentant les
modules statistiques (demandes de cotation, historique long) sont simulees en l'absence
d'historiques réels. La **méthode** est opérationnelle ; les **données** ne le sont pas encore.
        """
    )

    st .markdown ('<div class="section-title">EXPORTS DISPONIBLES</div>',unsafe_allow_html =True )
    export_files =sorted (EXPORTS_DIR .glob ("pricer_*.csv"),reverse =True )
    if not export_files :
        st .info ("Aucun export génère pour le moment.")
    else :
        for f in export_files :
            st .download_button (f .name ,data =f .read_bytes (),file_name =f .name ,mime ="text/csv",key =f"exp_{f .name }")


            # --------------------------------------------------------------------------- #
            # Execution principale
            # --------------------------------------------------------------------------- #

run_historisation_quotidienne ()

components .eurusd_control ()# doit s'executer AVANT calculer_etat_pricer : sinon un
# changement de cours EUR/USD n'apparaitrait qu'au rerun suivant

_transactions =db .get_transactions ()
_history =db .get_price_history ()
_position_df =position_cumulee (_transactions )
_position_actuelle =float (_position_df ["position"].iloc [-1 ])if not _position_df .empty else 0.0 
_sigma =volatilite (_history )
_etat =calculer_etat_pricer (
st .session_state ["eur_usd"],st .session_state ["quotes"],_position_actuelle ,_sigma ,st .session_state ["calib"]
)

components .ticker_tape (st .session_state ["quotes"],_etat .prix_final .bid_final ,_etat .prix_final .ask_final )
components .header_bar ()
components .position_alert_banner (_position_actuelle ,st .session_state ["calib"].position_limite )
components .sidebar_nav ()
components .theme_switcher ()

_page =st .session_state ["page_active"]
if _page =="GUIDE":
    render_guide ()
elif _page =="DASH":
    render_dash (_etat ,_history ,_position_df )
elif _page =="MKT":
    render_mkt (_etat )
elif _page =="RFQ":
    render_rfq (_etat )
elif _page =="FWD":
    render_fwd (_etat )
elif _page =="SEG":
    render_seg (_etat )
elif _page =="GARCH":
    render_garch (_etat )
elif _page =="TS":
    render_ts (_etat )
elif _page =="POS":
    render_pos (_transactions ,_position_df ,_etat )
elif _page =="PNL":
    render_pnl (_transactions ,_history ,_etat )
elif _page =="ATTR":
    render_attr (_etat ,_transactions ,_history )
elif _page =="FLOW":
    render_flow (_etat )
elif _page =="RISK":
    render_risk (_etat )
elif _page =="PTF":
    render_ptf (_etat ,_history )
elif _page =="VOL":
    render_vol (_etat ,_history )
elif _page =="SKEW":
    render_skew (_etat ,_history ,_transactions )
elif _page =="STRESS":
    render_stress (_etat )
elif _page =="LIQD":
    render_liqd (_etat )
elif _page =="EVENTS":
    render_events ()
elif _page =="THEORY":
    render_theory ()
elif _page =="AUDIT":
    render_audit ()
elif _page =="DOCS":
    render_docs ()
