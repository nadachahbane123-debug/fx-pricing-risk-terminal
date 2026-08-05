"""Palette de couleurs et injection CSS globale — identite "terminal Bloomberg x CFG Bank".

Deux modes : "dark" (terminal, par defaut) et "light" (fond clair, plus lisible en
salle tres eclairee et pour les captures d'ecran d'un rapport). Le mode courant est
stocke dans st.session_state["theme_mode"].

IMPORTANT : les couleurs ne doivent plus etre importees une fois pour toutes
(`from ui.theme import CFG_RED`), car elles changent selon le mode. Il faut passer
par `theme.P()["cfg_red"]`, evalue a chaque rendu. L'acces par attribut de module
(`theme.CFG_RED`) reste supporte pour compatibilite et renvoie la valeur du mode
courant.
"""
from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #

DARK = {
    "bg_main": "#0A0E14",
    "bg_panel": "#12161F",
    "bg_panel_alt": "#0F131A",
    "border": "#2A2F3A",
    "cfg_red": "#C8102E",
    "green": "#00D964",
    "red_alert": "#FF3B30",
    "amber": "#F5A623",
    "text_primary": "#E8E8E8",
    "text_secondary": "#8A8F98",
    "neutral_series": "#9BA1AC",
    "ticker_bg": "#000000",
}

LIGHT = {
    "bg_main": "#F7F7F5",
    "bg_panel": "#FFFFFF",
    "bg_panel_alt": "#F1F1EE",
    "border": "#D8D8D2",
    "cfg_red": "#C8102E",
    "green": "#0F9D58",       # vert assombri : lisible sur fond clair
    "red_alert": "#D93025",
    "amber": "#B26A00",       # ambre assombri, sinon illisible sur blanc
    "text_primary": "#1A1A1A",
    "text_secondary": "#5F6368",
    "neutral_series": "#8A8F98",
    "ticker_bg": "#1A1A1A",   # le ticker reste sombre : c'est sa signature visuelle
}

PALETTES = {"dark": DARK, "light": LIGHT}

FONT_MONO = "'JetBrains Mono', 'IBM Plex Mono', monospace"
FONT_SANS = "'Inter', -apple-system, sans-serif"

_LEGACY_NAMES = {
    "BG_MAIN": "bg_main",
    "BG_PANEL": "bg_panel",
    "BG_PANEL_ALT": "bg_panel_alt",
    "BORDER": "border",
    "CFG_RED": "cfg_red",
    "GREEN": "green",
    "RED_ALERT": "red_alert",
    "AMBER": "amber",
    "TEXT_PRIMARY": "text_primary",
    "TEXT_SECONDARY": "text_secondary",
}


def mode() -> str:
    """Mode courant : 'dark' (defaut) ou 'light'."""
    return st.session_state.get("theme_mode", "dark")


def set_mode(new_mode: str) -> None:
    st.session_state["theme_mode"] = new_mode if new_mode in PALETTES else "dark"


def P() -> dict:
    """Palette du mode courant."""
    return PALETTES[mode()]


def is_dark() -> bool:
    return mode() == "dark"


def __getattr__(name: str):
    """Compatibilite : theme.CFG_RED, theme.AMBER... renvoient la valeur du mode courant."""
    key = _LEGACY_NAMES.get(name)
    if key is not None:
        return P()[key]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def inject_css() -> None:
    c = P()
    dark = is_dark()

    hover_row = "rgba(200, 16, 46, 0.08)" if dark else "rgba(200, 16, 46, 0.06)"
    card_shadow = "none" if dark else "0 1px 2px rgba(0,0,0,0.06)"
    glow = "0 0 6px" if dark else "0 0 3px"
    warn_bg = 0.15 if dark else 0.18
    crit_bg = 0.15 if dark else 0.12
    ok_bg = 0.12 if dark else 0.15

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: {FONT_SANS};
        }}

        /* ---------- Masquage des elements Streamlit par defaut ---------- */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{
            background: transparent;
            height: 0;
        }}
        div[data-testid="stToolbar"] {{visibility: hidden; height: 0;}}
        div[data-testid="stDecoration"] {{display: none;}}
        #stDecoration {{display: none;}}
        .stDeployButton {{display: none;}}

        .block-container {{
            padding-top: 0.5rem;
            padding-bottom: 1rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            max-width: 100%;
        }}

        body, .stApp {{
            background-color: {c['bg_main']};
            color: {c['text_primary']};
        }}

        /* Textes natifs Streamlit */
        .stApp p, .stApp span, .stApp label, .stApp li, .stMarkdown {{
            color: {c['text_primary']};
        }}
        [data-testid="stCaptionContainer"] p {{
            color: {c['text_secondary']};
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
            color: {c['text_primary']};
        }}

        /* Champs de saisie */
        .stTextInput input, .stNumberInput input, .stDateInput input,
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {c['bg_panel']};
            color: {c['text_primary']};
            border-color: {c['border']};
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background-color: {c['bg_panel']};
            border-right: 1px solid {c['border']};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
        }}

        section[data-testid="stSidebar"] div.stButton > button {{
            width: 100%;
            font-family: {FONT_MONO};
            font-weight: 600;
            letter-spacing: 0.03em;
            background-color: {c['bg_main']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
            border-radius: 3px;
            text-align: left;
            padding: 0.45rem 0.8rem;
            margin-bottom: 2px;
            transition: all 0.12s ease;
        }}
        section[data-testid="stSidebar"] div.stButton > button:hover {{
            border-color: {c['cfg_red']};
            color: {c['cfg_red']};
        }}
        section[data-testid="stSidebar"] div.stButton > button:focus {{
            box-shadow: none;
        }}
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
            background-color: {c['cfg_red']};
            border-color: {c['cfg_red']};
            color: white;
        }}

        /* ---------- Boutons generaux ---------- */
        div.stButton > button {{
            font-family: {FONT_MONO};
            border-radius: 3px;
            background-color: {c['bg_panel']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
        }}
        div.stButton > button[kind="primary"] {{
            background-color: {c['cfg_red']};
            border-color: {c['cfg_red']};
            color: white;
        }}

        /* ---------- Ticker tape (toujours sombre) ---------- */
        .ticker-wrap {{
            width: 100%;
            overflow: hidden;
            background-color: {c['ticker_bg']};
            border-bottom: 1px solid {c['border']};
            padding: 6px 0;
            box-sizing: border-box;
        }}
        .ticker-move {{
            display: inline-block;
            white-space: nowrap;
            animation: ticker-scroll 45s linear infinite;
            font-family: {FONT_MONO};
            font-size: 0.82rem;
        }}
        .ticker-wrap:hover .ticker-move {{
            animation-play-state: paused;
        }}
        @keyframes ticker-scroll {{
            0%   {{ transform: translateX(0%); }}
            100% {{ transform: translateX(-50%); }}
        }}
        .tk-bank {{ color: #E8E8E8 !important; font-weight: 600; }}
        .tk-pair {{ color: #8A8F98 !important; }}
        .tk-bid {{ color: #00D964 !important; }}
        .tk-ask {{ color: #FF6B60 !important; }}
        .tk-sep {{ color: #444 !important; padding: 0 14px; }}

        /* ---------- Header avec statut live ---------- */
        .term-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 4px;
            border-bottom: 1px solid {c['border']};
            margin-bottom: 6px;
        }}
        .term-title {{
            font-family: {FONT_MONO};
            font-weight: 700;
            font-size: 1.05rem;
            color: {c['text_primary']};
            letter-spacing: 0.02em;
        }}
        .term-title span {{ color: {c['cfg_red']}; }}
        .term-status {{
            display: flex;
            align-items: center;
            gap: 18px;
            font-family: {FONT_MONO};
            font-size: 0.8rem;
            color: {c['text_secondary']};
        }}
        .term-status span {{ color: {c['text_secondary']}; }}
        .live-dot {{
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background-color: {c['green']};
            margin-right: 6px;
            animation: pulse-dot 1.4s ease-in-out infinite;
            box-shadow: {glow} {c['green']};
        }}
        @keyframes pulse-dot {{
            0%   {{ opacity: 1; }}
            50%  {{ opacity: 0.35; }}
            100% {{ opacity: 1; }}
        }}
        .live-label {{ color: {c['green']} !important; font-weight: 700; }}

        /* ---------- Bandeau d'alerte de position ---------- */
        .risk-banner {{
            font-family: {FONT_MONO};
            font-weight: 600;
            font-size: 0.85rem;
            padding: 8px 16px;
            border-radius: 3px;
            margin-bottom: 10px;
            letter-spacing: 0.01em;
        }}
        .risk-banner-warn {{
            background-color: rgba(245, 166, 35, {warn_bg});
            border: 1px solid {c['amber']};
            color: {c['amber']} !important;
        }}
        .risk-banner-crit {{
            background-color: rgba(255, 59, 48, {crit_bg});
            border: 1px solid {c['red_alert']};
            color: {c['red_alert']} !important;
        }}

        /* ---------- Cartes de metriques ---------- */
        .metric-card {{
            background-color: {c['bg_panel']};
            border: 1px solid {c['border']};
            border-radius: 4px;
            padding: 10px 14px;
            height: 100%;
            box-shadow: {card_shadow};
        }}
        .metric-label {{
            font-family: {FONT_SANS};
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {c['text_secondary']} !important;
            margin-bottom: 4px;
        }}
        .metric-value {{
            font-family: {FONT_MONO};
            font-size: 1.5rem;
            font-weight: 700;
            color: {c['text_primary']} !important;
            line-height: 1.15;
        }}
        .metric-delta {{
            font-family: {FONT_MONO};
            font-size: 0.78rem;
            font-weight: 600;
            margin-top: 3px;
        }}
        .delta-up {{ color: {c['green']} !important; }}
        .delta-down {{ color: {c['red_alert']} !important; }}
        .delta-flat {{ color: {c['text_secondary']} !important; }}

        /* ---------- Badges de conformite ---------- */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            padding: 3px 10px;
            border-radius: 20px;
        }}
        .badge-ok {{
            background-color: rgba(0, 217, 100, {ok_bg});
            color: {c['green']} !important;
            border: 1px solid {c['green']}66;
        }}
        .badge-bad {{
            background-color: rgba(255, 59, 48, {crit_bg});
            color: {c['red_alert']} !important;
            border: 1px solid {c['red_alert']}66;
        }}
        .badge-dot {{
            width: 6px; height: 6px; border-radius: 50%;
            background-color: currentColor;
        }}

        /* ---------- Section titles ---------- */
        .section-title {{
            font-family: {FONT_MONO};
            font-weight: 700;
            font-size: 0.95rem;
            color: {c['cfg_red']} !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid {c['border']};
            padding-bottom: 5px;
            margin: 14px 0 10px 0;
        }}

        /* ---------- Tableaux denses ---------- */
        table.dense-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: {FONT_MONO};
            font-size: 0.8rem;
        }}
        table.dense-table th {{
            text-align: right;
            color: {c['text_secondary']} !important;
            text-transform: uppercase;
            font-size: 0.68rem;
            letter-spacing: 0.04em;
            font-weight: 600;
            border-bottom: 1px solid {c['border']};
            padding: 6px 10px;
        }}
        table.dense-table th:first-child, table.dense-table td:first-child {{
            text-align: left;
        }}
        table.dense-table td {{
            text-align: right;
            padding: 5px 10px;
            border-bottom: 1px solid {c['bg_main']};
            color: {c['text_primary']};
        }}
        table.dense-table tr:nth-child(odd) td {{
            background-color: {c['bg_panel_alt']};
        }}
        table.dense-table tr:hover td {{
            background-color: {hover_row};
        }}
        .cfg-row td {{
            color: {c['cfg_red']} !important;
            font-weight: 700;
        }}

        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
            font-family: {FONT_MONO};
        }}

        /* ---------- Jauge de risque ---------- */
        .gauge-track {{
            position: relative;
            width: 100%;
            height: 22px;
            background: linear-gradient(90deg, {c['green']} 0%, {c['green']} 60%, {c['amber']} 80%, {c['red_alert']} 100%);
            border-radius: 3px;
            border: 1px solid {c['border']};
            overflow: visible;
        }}
        .gauge-cursor {{
            position: absolute;
            top: -4px;
            width: 3px;
            height: 30px;
            background-color: {c['text_primary']};
            box-shadow: 0 0 4px rgba(0,0,0,0.4);
        }}

        /* ---------- Bande de fluctuation ---------- */
        .band-track {{
            position: relative;
            width: 100%;
            height: 22px;
            background: linear-gradient(90deg, {c['bg_panel_alt']} 0%, {c['bg_panel']} 50%, {c['bg_panel_alt']} 100%);
            border-radius: 3px;
            border: 1px solid {c['border']};
            overflow: visible;
        }}
        .band-cursor {{
            position: absolute;
            top: -4px;
            width: 3px;
            height: 30px;
            box-shadow: 0 0 4px rgba(0,0,0,0.35);
        }}

        /* ---------- Bloc BID/ASK final ---------- */
        .quote-block {{ display: flex; gap: 14px; }}
        .quote-side {{
            flex: 1;
            background-color: {c['bg_panel']};
            border: 1px solid {c['border']};
            border-radius: 4px;
            padding: 16px;
            text-align: center;
            box-shadow: {card_shadow};
        }}
        .quote-side-label {{
            font-family: {FONT_SANS};
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {c['text_secondary']} !important;
            margin-bottom: 6px;
        }}
        .quote-side-value {{
            font-family: {FONT_MONO};
            font-size: 2.4rem;
            font-weight: 700;
            color: {c['text_primary']} !important;
            line-height: 1.1;
        }}

        /* ---------- Ticket RFQ ---------- */
        .rfq-ticket {{
            background-color: {c['bg_panel']};
            border: 1px solid {c['cfg_red']};
            border-left: 4px solid {c['cfg_red']};
            border-radius: 4px;
            padding: 14px 18px;
            font-family: {FONT_MONO};
            box-shadow: {card_shadow};
        }}
        .rfq-line {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 0.86rem;
            color: {c['text_primary']};
            border-bottom: 1px dashed {c['border']};
        }}
        .rfq-line:last-child {{ border-bottom: none; }}
        .rfq-line span:first-child {{ color: {c['text_secondary']} !important; }}
        .rfq-headline {{
            font-family: {FONT_MONO};
            font-size: 1.9rem;
            font-weight: 700;
            color: {c['cfg_red']} !important;
            text-align: center;
            padding: 6px 0 10px 0;
        }}

        /* ---------- Instrument code ---------- */
        .instrument-code {{
            font-family: {FONT_MONO};
            font-size: 0.78rem;
            color: {c['text_secondary']} !important;
            letter-spacing: 0.03em;
        }}
        .instrument-code b {{ color: {c['cfg_red']} !important; }}

        hr {{ border-color: {c['border']}; }}

        ::-webkit-scrollbar {{ height: 8px; width: 8px; }}
        ::-webkit-scrollbar-track {{ background: {c['bg_main']}; }}
        ::-webkit-scrollbar-thumb {{ background: {c['border']}; border-radius: 4px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
