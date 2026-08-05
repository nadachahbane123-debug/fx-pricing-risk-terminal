"""Composants d'interface reutilisables : cartes, badges, ticker, sparklines, bandeaux, jauge, nav.

Les couleurs sont résolues a chaque appel via theme.P(), afin que l'ensemble de
l'interface suive le mode d'affichage (sombre / clair) choisi par l'utilisateur.
"""
from __future__ import annotations 

import base64 
from datetime import datetime 
from pathlib import Path 

import pandas as pd 
import streamlit as st 

from ui import charts ,theme 
from ui .theme import FONT_MONO 

LOGO_PATH =Path (__file__ ).resolve ().parent .parent /"assets"/"logo-cfg.png"


def _logo_b64 ()->str |None :
    if not LOGO_PATH .exists ():
        return None 
    return base64 .b64encode (LOGO_PATH .read_bytes ()).decode ("ascii")


    # --------------------------------------------------------------------------- #
    # Formatage
    # --------------------------------------------------------------------------- #

def fmt_price (x :float )->str :
    return f"{x :,.4f}"


def fmt_amount (x :float )->str :
    return f"{x :,.0f}"


def fmt_pct (x :float )->str :
    return f"{x *100 :,.2f}%"


    # --------------------------------------------------------------------------- #
    # Ticker tape — genere a partir de st.session_state["quotes"]
    # --------------------------------------------------------------------------- #

def ticker_tape (quotes :pd .DataFrame ,cfg_bid :float |None =None ,cfg_ask :float |None =None )->None :
    c =theme .P ()
    items =[]
    for _ ,r in quotes .iterrows ():
        items .append (
        f'<span class="tk-bank">{r ["code"]}</span> <span class="tk-pair">USDMAD</span> '
        f'<span class="tk-bid">{r ["bid"]:.4f}</span>/<span class="tk-ask">{r ["ask"]:.4f}</span>'
        )
    if cfg_bid is not None and cfg_ask is not None :
        items .append (
        f'<span class="tk-bank" style="color:{c ["cfg_red"]} !important">CFGB</span> '
        f'<span class="tk-pair">USDMAD</span> '
        f'<span class="tk-bid">{cfg_bid :.4f}</span>/<span class="tk-ask">{cfg_ask :.4f}</span>'
        )
    sep ='<span class="tk-sep">&bull;</span>'
    content =sep .join (items )
    full =content +sep +content # duplique pour un defilement continu sans coupure
    st .markdown (f'<div class="ticker-wrap"><div class="ticker-move">{full }</div></div>',unsafe_allow_html =True )


    # --------------------------------------------------------------------------- #
    # En-tete avec statut live
    # --------------------------------------------------------------------------- #

def header_bar ()->None :
    now =datetime .now ()
    logo_b64 =_logo_b64 ()
    logo_html =(
    f'<img src="data:image/png;base64,{logo_b64 }" style="height:28px; margin-right:14px; vertical-align:middle;">'
    if logo_b64 
    else ""
    )
    st .markdown (
    f"""
        <div class="term-header">
            <div class="term-title" style="display:flex; align-items:center;">{logo_html }<span>&mdash;</span>&nbsp;USDMAD PRICER TERMINAL</div>
            <div class="term-status">
                <span><span class="live-dot"></span><span class="live-label">LIVE</span></span>
                <span>DERNIERE MAJ&nbsp;{now .strftime ('%H:%M:%S')}</span>
                <span>{now .strftime ('%d/%m/%Y')}</span>
            </div>
        </div>
        """,
    unsafe_allow_html =True ,
    )


    # --------------------------------------------------------------------------- #
    # Bandeau d'alerte de position persistant
    # --------------------------------------------------------------------------- #

def position_alert_banner (position :float ,limite :float )->float :
    """Affiche le bandeau orange (>=80%) ou rouge (>=100%). Renvoie le % courant."""
    pct =(abs (position )/limite *100 )if limite >0 else 0.0 

    if pct >=100 :
        st .markdown (
        f'<div class="risk-banner risk-banner-crit">'
        f"&#128680; LIMITE DE POSITION D&Eacute;PASS&Eacute;E &mdash; LIQUIDATION RECOMMAND&Eacute;E "
        f"({pct :.0f}% DE LA LIMITE)</div>",
        unsafe_allow_html =True ,
        )
        if st .button ("→ ACCÉDER AU PANIC MODE (LIQD<GO>)",key ="banner_panic_link"):
            st .session_state ["page_active"]="LIQD"
            st .session_state ["liqd_auto_position"]=abs (position )
            st .rerun ()
    elif pct >=80 :
        st .markdown (
        f'<div class="risk-banner risk-banner-warn">'
        f"&#9888; POSITION &Agrave; {pct :.0f}% DE LA LIMITE &mdash; SURVEILLANCE RENFORC&Eacute;E</div>",
        unsafe_allow_html =True ,
        )
    return pct 


    # --------------------------------------------------------------------------- #
    # Cartes de metriques façon terminal
    # --------------------------------------------------------------------------- #

def metric_card (
col ,
label :str ,
value :str ,
delta :str |None =None ,
delta_kind :str ="flat",
spark_values :list [float ]|None =None ,
spark_color :str |None =None ,
key :str ="",
)->None :
    c =theme .P ()
    spark_color =spark_color or c ["green"]
    with col :
        arrow ={"up":"▲","down":"▼"}.get (delta_kind ,"")
        delta_html =f'<div class="metric-delta delta-{delta_kind }">{arrow } {delta }</div>'if delta else ""
        st .markdown (
        f'<div class="metric-card"><div class="metric-label">{label }</div>'
        f'<div class="metric-value">{value }</div>{delta_html }</div>',
        unsafe_allow_html =True ,
        )
        if spark_values and len (spark_values )>1 :
            fig =charts .sparkline (spark_values ,spark_color )
            fig .update_layout (paper_bgcolor =c ["bg_panel"],plot_bgcolor =c ["bg_panel"])
            st .plotly_chart (fig ,width='stretch' ,config ={"displayModeBar":False },key =f"spark_{key }")


            # --------------------------------------------------------------------------- #
            # Badges de conformite
            # --------------------------------------------------------------------------- #

def badge_html (is_ok :bool ,ok_text :str ="CONFORME",bad_text :str ="HORS BANDE")->str :
    cls ="badge-ok"if is_ok else "badge-bad"
    text =ok_text if is_ok else bad_text 
    return f'<span class="badge {cls }"><span class="badge-dot"></span>{text }</span>'


def badge (is_ok :bool ,ok_text :str ="CONFORME",bad_text :str ="HORS BANDE")->None :
    st .markdown (badge_html (is_ok ,ok_text ,bad_text ),unsafe_allow_html =True )


    # --------------------------------------------------------------------------- #
    # Bande de fluctuation reglementaire — barre avec curseurs
    # --------------------------------------------------------------------------- #

def bande_fluctuation_bar (borne_basse :float ,borne_haute :float ,mid_theo :float ,mid_marche :float )->None :
    c =theme .P ()
    span =borne_haute -borne_basse 
    pct_theo =max (0.0 ,min (100.0 ,(mid_theo -borne_basse )/span *100 ))if span else 50 
    pct_marche =max (0.0 ,min (100.0 ,(mid_marche -borne_basse )/span *100 ))if span else 50 
    st .markdown (
    f"""
        <div style="margin: 4px 0 12px 0;">
          <div class="band-track">
            <div class="band-cursor" style="left:{pct_theo :.2f}%; background-color:{c ['text_primary']};" title="MID theorique"></div>
            <div class="band-cursor" style="left:{pct_marche :.2f}%; background-color:{c ['amber']};" title="MID marche"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-family:{FONT_MONO };
                      font-size:0.7rem; color:{c ['text_secondary']}; margin-top:4px;">
            <span style="color:{c ['text_secondary']};">{borne_basse :.4f} (BORNE BASSE)</span>
            <span style="color:{c ['text_primary']};">&#9670; TH&Eacute;O {mid_theo :.4f}</span>
            <span style="color:{c ['amber']};">&#9670; MARCH&Eacute; {mid_marche :.4f}</span>
            <span style="color:{c ['text_secondary']};">{borne_haute :.4f} (BORNE HAUTE)</span>
          </div>
        </div>
        """,
    unsafe_allow_html =True ,
    )


    # --------------------------------------------------------------------------- #
    # Tableaux denses
    # --------------------------------------------------------------------------- #

def dense_table (df :pd .DataFrame ,highlight_col :str |None =None ,highlight_value :str |None =None )->None :
    headers ="".join (f"<th>{c }</th>"for c in df .columns )
    rows_html =[]
    for _ ,r in df .iterrows ():
        row_class =' class="cfg-row"'if highlight_col and str (r [highlight_col ])==highlight_value else ""
        cells ="".join (f"<td>{r [c ]}</td>"for c in df .columns )
        rows_html .append (f"<tr{row_class }>{cells }</tr>")
    html =(
    '<div style="overflow-x:auto;">'
    f'<table class="dense-table"><thead><tr>{headers }</tr></thead><tbody>{"".join (rows_html )}</tbody></table>'
    "</div>"
    )
    st .markdown (html ,unsafe_allow_html =True )


    # --------------------------------------------------------------------------- #
    # Navigation par codes de fonction, pilotee par st.session_state
    # --------------------------------------------------------------------------- #

PAGES :list [tuple [str ,str ]]=[
("GUIDE","[ GUIDE ]"),
("DASH","[ DASH ]"),
("MKT","[ MKT<GO> ]"),
("RFQ","[ RFQ<GO> ]"),
("FWD","[ FWD<GO> ]"),
("POS","[ POS<GO> ]"),
("PNL","[ PNL<GO> ]"),
("ATTR","[ ATTR<GO> ]"),
("FLOW","[ FLOW<GO> ]"),
("SEG","[ SEG<GO> ]"),
("RISK","[ RISK<GO> ]"),
("PTF","[ PTF<GO> ]"),
("VOL","[ VOL<GO> ]"),
("GARCH","[ GARCH<GO> ]"),
("TS","[ TS<GO> ]"),
("SKEW","[ SKEW<GO> ]"),
("STRESS","[ STRESS<GO> ]"),
("LIQD","[ LIQD<GO> ]"),
("EVENTS","[ EVENTS<GO> ]"),
("THEORY","[ THEORY<GO> ]"),
("AUDIT","[ AUDIT<GO> ]"),
("DOCS","[ DOCS<GO> ]"),
]


def theme_switcher ()->None :
    """Selecteur de mode d'affichage (sombre / clair) dans la barre laterale.

    Le changement est applique au rerun suivant : inject_css() etant appele en
    tout debut de script, un st.rerun() explicite est nécessaire pour que le
    nouveau CSS soit génère avec la bonne palette.
    """
    c =theme .P ()
    st .sidebar .markdown (
    f'<div style="font-family:{FONT_MONO }; font-weight:700; color:{c ["cfg_red"]}; '
    f'letter-spacing:0.05em; margin:14px 0 6px 0;">AFFICHAGE</div>',
    unsafe_allow_html =True ,
    )
    current =theme .mode ()
    choix =st .sidebar .radio (
    "Mode d'affichage",
    options =["dark","light"],
    format_func =lambda m :"Sombre (terminal)"if m =="dark"else "Clair",
    index =0 if current =="dark"else 1 ,
    horizontal =True ,
    key ="theme_mode_radio",
    label_visibility ="collapsed",
    )
    if choix !=current :
        theme .set_mode (choix )
        st .rerun ()


def eurusd_control ()->None :
    """Section 'Données de marché' de la barre laterale : saisie manuelle du
    cours EUR/USD + recuperation live via l'API gratuite Frankfurter (aucune
    cle requise). Repli silencieux sur la saisie manuelle si l'API est
    injoignable — l'application ne doit jamais planter faute de reseau.
    """
    c =theme .P ()
    st .sidebar .markdown (
    f'<div style="font-family:{FONT_MONO }; font-weight:700; color:{c ["cfg_red"]}; '
    f'letter-spacing:0.05em; margin-bottom:6px; margin-top:4px;">DONNEES DE MARCHE</div>',
    unsafe_allow_html =True ,
    )

    col_input ,col_btn =st .sidebar .columns ([2 ,1 ])
    with col_input :
        new_val =st .number_input (
        "EUR/USD",value =float (st .session_state ["eur_usd"]),
        format ="%.4f",step =0.0001 ,key ="eurusd_manual_input",label_visibility ="collapsed",
        )
        if new_val !=st .session_state ["eur_usd"]:
            st .session_state ["eur_usd"]=new_val 

    with col_btn :
        if st .button ("LIVE",key ="btn_live_eurusd",width='stretch' ):
            live_val =fetch_eurusd_live ()
            if live_val is not None :
                st .session_state ["eur_usd"]=live_val 
                st .session_state ["eurusd_last_live_fetch"]=datetime .now ().strftime ("%H:%M:%S")
                st .rerun ()
            else :
                st .sidebar .warning ("API indisponible — cours manuel conserve.")

    last_fetch =st .session_state .get ("eurusd_last_live_fetch")
    if last_fetch :
        st .sidebar .caption (f"Dernier cours live recupere a {last_fetch }")
    else :
        st .sidebar .caption ("Cours manuel (aucune recuperation live effectuee)")


def fetch_eurusd_live ()->float |None :
    """Recupere le cours EUR/USD en direct via l'API Frankfurter (gratuite, sans cle).
    Renvoie None si indisponible — l'appelant conserve alors la valeur manuelle.
    """
    try :
        import requests 
        r =requests .get ("https://api.frankfurter.app/latest?from=EUR&to=USD",timeout =5 )
        r .raise_for_status ()
        return float (r .json ()["rates"]["USD"])
    except Exception :
        return None 


def sidebar_nav ()->None :
    c =theme .P ()
    st .sidebar .markdown (
    f'<div style="font-family:{FONT_MONO }; font-weight:700; color:{c ["cfg_red"]}; '
    f'letter-spacing:0.05em; margin-bottom:10px;">CFG BANK TERMINAL</div>',
    unsafe_allow_html =True ,
    )
    active =st .session_state .get ("page_active","DASH")
    for code ,label in PAGES :
        kind ="primary"if code ==active else "secondary"
        if st .sidebar .button (label ,key =f"nav_{code }",type =kind ,width='stretch' ):
            st .session_state ["page_active"]=code 
            st .rerun ()


def rfq_ticket (lines :list [tuple [str ,str ]],headline :str )->None :
    """Ticket de cotation client (RFQ), presente comme un ticket de deal."""
    rows ="".join (f'<div class="rfq-line"><span>{k }</span><span>{v }</span></div>'for k ,v in lines )
    st .markdown (
    f'<div class="rfq-ticket"><div class="rfq-headline">{headline }</div>{rows }</div>',
    unsafe_allow_html =True ,
    )
