"""Graphiques Plotly, theme dynamique (sombre / clair) — remplace les graphiques natifs Streamlit.

Les couleurs sont résolues a CHAQUE appel via theme.P(), et non importees une fois
pour toutes, afin que les graphiques suivent le mode d'affichage choisi.
"""
from __future__ import annotations 

import plotly .graph_objects as go 

from ui import theme 

FONT_MONO ="JetBrains Mono, IBM Plex Mono, monospace"


def _layout (**overrides )->dict :
    c =theme .P ()
    base =dict (
    paper_bgcolor =c ["bg_panel"],
    plot_bgcolor =c ["bg_panel"],
    font =dict (family =FONT_MONO ,color =c ["text_secondary"],size =11 ),
    margin =dict (l =40 ,r =20 ,t =30 ,b =30 ),
    xaxis =dict (gridcolor =c ["border"],zerolinecolor =c ["border"],showgrid =True ,color =c ["text_secondary"]),
    yaxis =dict (gridcolor =c ["border"],zerolinecolor =c ["border"],showgrid =True ,color =c ["text_secondary"]),
    legend =dict (orientation ="h",yanchor ="bottom",y =1.02 ,xanchor ="left",x =0 ,font =dict (size =10 )),
    hovermode ="x unified",
    hoverlabel =dict (bgcolor =c ["bg_main"],font =dict (family =FONT_MONO ,color =c ["text_primary"])),
    )
    base .update (overrides )
    return base 


def sparkline (values :list [float ],color :str |None =None )->go .Figure :
    """Mini-graphique sans axes ni titre, pour accompagner une carte de metrique."""
    color =color or theme .P ()["green"]
    fig =go .Figure (
    go .Scatter (
    y =values ,
    mode ="lines",
    line =dict (color =color ,width =1.6 ),
    fill ="tozeroy",
    fillcolor =_alpha (color ,0.15 ),
    hoverinfo ="skip",
    )
    )
    fig .update_layout (
    paper_bgcolor ="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    margin =dict (l =0 ,r =0 ,t =0 ,b =0 ),
    height =42 ,
    xaxis =dict (visible =False ),
    yaxis =dict (visible =False ),
    showlegend =False ,
    )
    return fig 


def mkt_comparison_bar (banks :list [str ],mids :list [float ],cfg_label :str )->go .Figure :
    """Comparaison des MID par banque, CFG Bank mise en evidence en rouge institutionnel."""
    c =theme .P ()
    colors =[c ["cfg_red"]if b ==cfg_label else c ["neutral_series"]for b in banks ]
    fig =go .Figure (
    go .Bar (x =banks ,y =mids ,marker_color =colors ,text =[f"{m :.4f}"for m in mids ],textposition ="outside")
    )
    fig .update_layout (**_layout (height =340 ))
    fig .update_yaxes (title =None )
    return fig 


def area_chart (x :list ,y :list ,color :str |None =None ,name :str ="",height :int =320 )->go .Figure :
    """Aire sous la courbe style terminal (position, liquidation, etc.)."""
    color =color or theme .P ()["green"]
    fig =go .Figure (
    go .Scatter (
    x =x ,y =y ,mode ="lines",
    line =dict (color =color ,width =2 ),
    fill ="tozeroy",fillcolor =_alpha (color ,0.15 ),
    name =name ,
    )
    )
    fig .update_layout (**_layout (height =height ,showlegend =bool (name )))
    return fig 


def pnl_area_chart (dates :list ,pnl :list )->go .Figure :
    """Aire du P&L cumulé : verte si positif en fin de periode, rouge sinon."""
    c =theme .P ()
    color =c ["green"]if (pnl and pnl [-1 ]>=0 )else c ["red_alert"]
    fig =go .Figure (
    go .Scatter (
    x =dates ,y =pnl ,mode ="lines+markers",
    line =dict (color =color ,width =2 ),
    marker =dict (size =4 ,color =color ),
    fill ="tozeroy",fillcolor =_alpha (color ,0.15 ),
    name ="P&L cumule",
    )
    )
    fig .add_hline (y =0 ,line_color =c ["border"],line_width =1 )
    fig .update_layout (**_layout (height =340 ))
    return fig 


def skew_curve_chart (positions :list [float ],skews :list [float ],position_actuelle :float ,skew_actuel :float )->go .Figure :
    c =theme .P ()
    fig =go .Figure ()
    fig .add_trace (
    go .Scatter (x =positions ,y =skews ,mode ="lines",line =dict (color =c ["cfg_red"],width =2 ),name ="Skew(Position)")
    )
    fig .add_trace (
    go .Scatter (
    x =[position_actuelle ],y =[skew_actuel ],mode ="markers",
    marker =dict (size =11 ,color =c ["amber"],symbol ="diamond",line =dict (color =c ["text_primary"],width =1 )),
    name ="Position actuelle",
    )
    )
    fig .add_hline (y =0 ,line_color =c ["border"],line_width =1 )
    fig .add_vline (x =0 ,line_color =c ["border"],line_width =1 )
    fig .update_layout (**_layout (height =380 ))
    fig .update_xaxes (title ="Position (USD)")
    fig .update_yaxes (title ="Skew (MAD)")
    return fig 


def liquidation_chart (jours :list [int ],position_restante :list [float ])->go .Figure :
    c =theme .P ()
    fig =go .Figure (
    go .Scatter (
    x =jours ,y =position_restante ,mode ="lines+markers",
    line =dict (color =c ["cfg_red"],width =2 ,shape ="spline"),
    marker =dict (size =6 ,color =c ["cfg_red"]),
    fill ="tozeroy",fillcolor =_alpha (c ["cfg_red"],0.15 ),
    name ="Position restante",
    )
    )
    fig .update_layout (**_layout (height =360 ))
    fig .update_xaxes (title ="Jour",dtick =1 )
    fig .update_yaxes (title ="Position restante (USD)")
    return fig 


def backtest_overlay_chart (dates :list ,mid_observe :list ,sans_skew :list ,avec_skew :list )->go .Figure :
    c =theme .P ()
    fig =go .Figure ()
    fig .add_trace (go .Scatter (x =dates ,y =mid_observe ,mode ="lines+markers",name ="MID marché observé",
    line =dict (color =c ["text_primary"],width =2 ),marker =dict (size =5 )))
    fig .add_trace (go .Scatter (x =dates ,y =sans_skew ,mode ="lines",name ="Prix final SANS skew",
    line =dict (color =c ["neutral_series"],width =2 ,dash ="dot")))
    fig .add_trace (go .Scatter (x =dates ,y =avec_skew ,mode ="lines+markers",name ="Prix final AVEC skew (Ho-Stoll)",
    line =dict (color =c ["cfg_red"],width =2 ),marker =dict (size =5 )))
    fig .update_layout (**_layout (height =400 ))
    return fig 


def gauge_thermometer (pct :float )->go .Figure :
    """Jauge horizontale type thermometre (0-150% de la limite)."""
    c =theme .P ()
    color =c ["green"]if pct <80 else (c ["amber"]if pct <100 else c ["red_alert"])
    fig =go .Figure (
    go .Bar (x =[min (pct ,150 )],y =["Position"],orientation ="h",marker_color =color ,width =0.5 )
    )
    fig .add_vline (x =80 ,line_color =c ["amber"],line_dash ="dash",line_width =1 )
    fig .add_vline (x =100 ,line_color =c ["red_alert"],line_dash ="dash",line_width =1 )
    fig .update_layout (**_layout (height =110 ))
    fig .update_xaxes (range =[0 ,150 ],title ="% de la position limite")
    fig .update_yaxes (visible =False )
    return fig 


def var_distribution_chart (pnl_sim :list [float ],var_95 :float ,var_99 :float )->go .Figure :
    """Distribution simulee du P&L a 1 jour, avec les seuils de VaR 95% et 99%."""
    c =theme .P ()
    fig =go .Figure (
    go .Histogram (x =pnl_sim ,nbinsx =60 ,marker_color =c ["neutral_series"],opacity =0.75 ,name ="P&L simule")
    )
    fig .add_vline (x =-var_95 ,line_color =c ["amber"],line_dash ="dash",line_width =2 ,
    annotation_text ="VaR 95%",annotation_position ="top")
    fig .add_vline (x =-var_99 ,line_color =c ["red_alert"],line_dash ="dash",line_width =2 ,
    annotation_text ="VaR 99%",annotation_position ="bottom")
    fig .update_layout (**_layout (height =320 ,hovermode ="closest",showlegend =False ))
    fig .update_xaxes (title ="P&L a 1 jour (MAD)")
    fig .update_yaxes (title ="Frequence")
    return fig 


def intraday_flow_chart (heures :list ,montants_achat :list ,montants_vente :list )->go .Figure :
    """Flux clients par tranche horaire : achats vs ventes (vu du client)."""
    c =theme .P ()
    fig =go .Figure ()
    fig .add_trace (go .Bar (x =heures ,y =montants_achat ,name ="Client achete (CFG vend)",marker_color =c ["red_alert"]))
    fig .add_trace (go .Bar (x =heures ,y =montants_vente ,name ="Client vend (CFG achete)",marker_color =c ["green"]))
    fig .update_layout (**_layout (height =300 ,barmode ="relative",hovermode ="x"))
    fig .update_xaxes (title ="Date")
    fig .update_yaxes (title ="Montant (USD)")
    return fig 


def _alpha (hex_color :str ,alpha :float )->str :
    hex_color =hex_color .lstrip ("#")
    r ,g ,b =int (hex_color [0 :2 ],16 ),int (hex_color [2 :4 ],16 ),int (hex_color [4 :6 ],16 )
    return f"rgba({r },{g },{b },{alpha })"


    # --------------------------------------------------------------------------- #
    # Change a terme
    # --------------------------------------------------------------------------- #

def courbe_terme_chart (tenors :list [str ],points :list [float ])->go .Figure :
    """Courbe des points de terme par echeance — l'ecran de cotation forward."""
    c =theme .P ()
    couleur =c ["green"]if (points and points [-1 ]>=0 )else c ["red_alert"]
    fig =go .Figure (
    go .Scatter (
    x =tenors ,y =points ,mode ="lines+markers",
    line =dict (color =couleur ,width =2.5 ),
    marker =dict (size =8 ,color =couleur ),
    fill ="tozeroy",fillcolor =_alpha (couleur ,0.12 ),
    name ="Points de terme",
    )
    )
    fig .add_hline (y =0 ,line_color =c ["border"],line_width =1 )
    fig .update_layout (**_layout (height =330 ))
    fig .update_xaxes (title ="Echeance")
    fig .update_yaxes (title ="Points de terme (pips)")
    return fig 


def echeancier_chart (tenors :list [str ],positions :list [float ])->go .Figure :
    """Position ventilee par echeance : rend visible le risque de décalage de dates."""
    c =theme .P ()
    couleurs =[c ["green"]if p >=0 else c ["red_alert"]for p in positions ]
    fig =go .Figure (
    go .Bar (x =tenors ,y =positions ,marker_color =couleurs ,
    text =[f"{p :,.0f}".replace (","," ")for p in positions ],textposition ="outside")
    )
    fig .add_hline (y =0 ,line_color =c ["text_secondary"],line_width =1.5 )
    fig .update_layout (**_layout (height =320 ,hovermode ="x"))
    fig .update_xaxes (title ="Echeance")
    fig .update_yaxes (title ="Position (USD)")
    return fig 


    # --------------------------------------------------------------------------- #
    # Intensite des flux et spread optimal
    # --------------------------------------------------------------------------- #

def intensite_pnl_chart (deltas :list [float ],intensites :list [float ],pnls :list [float ],
delta_opt :float ,delta_marche :float )->go .Figure :
    """Superpose l'intensite des flux et le P&L espere, avec l'optimum marque.

    Deux axes : l'intensite decroit continument avec le spread, tandis que le
    P&L espere passe par un maximum. C'est ce maximum qui definit le prix a coter.
    """
    c =theme .P ()
    deltas_pips =[d *10_000 for d in deltas ]

    fig =go .Figure ()
    fig .add_trace (go .Scatter (
    x =deltas_pips ,y =intensites ,mode ="lines",name ="Intensite des flux (taux d'acceptation)",
    line =dict (color =c ["neutral_series"],width =2 ,dash ="dot"),yaxis ="y2",
    ))
    fig .add_trace (go .Scatter (
    x =deltas_pips ,y =pnls ,mode ="lines",name ="P&L espere",
    line =dict (color =c ["cfg_red"],width =2.5 ),
    fill ="tozeroy",fillcolor =_alpha (c ["cfg_red"],0.12 ),
    ))
    fig .add_vline (x =delta_opt *10_000 ,line_color =c ["green"],line_dash ="dash",line_width =2 ,
    annotation_text ="Optimum du modèle",annotation_position ="top")
    fig .add_vline (x =delta_marche *10_000 ,line_color =c ["amber"],line_dash ="dot",line_width =2 ,
    annotation_text ="Demi-spread marche",annotation_position ="bottom")

    lay =_layout (height =380 ,hovermode ="x unified")
    lay ["yaxis2"]=dict (overlaying ="y",side ="right",showgrid =False ,
    color =c ["text_secondary"],title ="Taux d'acceptation")
    fig .update_layout (**lay )
    fig .update_xaxes (title ="Écart au mid (pips)")
    fig .update_yaxes (title ="P&L espere (unite arbitraire)")
    return fig 


def hit_ratio_chart (deltas :list [float ],taux :list [float ],
deltas_fit :list [float ],taux_fit :list [float ])->go .Figure :
    """Taux d'acceptation observes et courbe ajustee — la calibration rendue visible."""
    c =theme .P ()
    fig =go .Figure ()
    fig .add_trace (go .Scatter (
    x =[d *10_000 for d in deltas ],y =taux ,mode ="markers",name ="Taux observe",
    marker =dict (size =11 ,color =c ["amber"],symbol ="circle",line =dict (color =c ["text_primary"],width =1 )),
    ))
    fig .add_trace (go .Scatter (
    x =[d *10_000 for d in deltas_fit ],y =taux_fit ,mode ="lines",
    name ="Modèle ajuste  A x exp(-k x delta)",
    line =dict (color =c ["cfg_red"],width =2.5 ),
    ))
    fig .update_layout (**_layout (height =340 ,hovermode ="closest"))
    fig .update_xaxes (title ="Écart propose (pips)")
    fig .update_yaxes (title ="Taux d'acceptation",range =[0 ,1 ])
    return fig 


    # --------------------------------------------------------------------------- #
    # Portefeuille multi-devises
    # --------------------------------------------------------------------------- #

def correlation_heatmap (devises :list [str ],matrice :list [list [float ]])->go .Figure :
    """Matrice de correlation entre paires de devises."""
    c =theme .P ()
    fig =go .Figure (go .Heatmap (
    z =matrice ,x =devises ,y =devises ,
    colorscale =[[0 ,c ["bg_panel"]],[0.5 ,c ["amber"]],[1 ,c ["cfg_red"]]],
    zmin =-1 ,zmax =1 ,
    text =[[f"{v :.3f}"for v in ligne ]for ligne in matrice ],
    texttemplate ="%{text}",textfont =dict (size =14 ,family =FONT_MONO ),
    showscale =True ,
    ))
    fig .update_layout (**_layout (height =300 ,hovermode ="closest"))
    return fig 


def diversification_chart (var_non_div :float ,var_div :float )->go .Figure :
    """Compare la VaR agregee a la somme des VaR individuelles."""
    c =theme .P ()
    fig =go .Figure (go .Bar (
    x =["Somme des VaR individuelles","VaR du portefeuille"],
    y =[var_non_div ,var_div ],
    marker_color =[c ["neutral_series"],c ["cfg_red"]],
    text =[f"{var_non_div :,.0f}".replace (","," "),f"{var_div :,.0f}".replace (","," ")],
    textposition ="outside",
    ))
    fig .update_layout (**_layout (height =300 ,hovermode ="x"))
    fig .update_yaxes (title ="VaR 99% (MAD)")
    return fig 


    # --------------------------------------------------------------------------- #
    # Attribution du P&L
    # --------------------------------------------------------------------------- #

def attribution_waterfall (marge :float ,skew :float ,marche :float ,total :float )->go .Figure :
    """Cascade d'attribution : d'ou vient le résultat."""
    c =theme .P ()
    fig =go .Figure (go .Waterfall (
    orientation ="v",
    measure =["relative","relative","relative","total"],
    x =["Marge de spread","Effet du skew","Effet de marché","P&L total"],
    y =[marge ,skew ,marche ,total ],
    connector =dict (line =dict (color =c ["border"])),
    increasing =dict (marker =dict (color =c ["green"])),
    decreasing =dict (marker =dict (color =c ["red_alert"])),
    totals =dict (marker =dict (color =c ["cfg_red"])),
    text =[f"{v :,.0f}".replace (","," ")for v in [marge ,skew ,marche ,total ]],
    textposition ="outside",
    ))
    fig .update_layout (**_layout (height =360 ,hovermode ="x"))
    fig .update_yaxes (title ="MAD")
    return fig 


    # --------------------------------------------------------------------------- #
    # Volatilite conditionnelle (GARCH)
    # --------------------------------------------------------------------------- #

def volatilite_conditionnelle_chart (dates :list ,sigma_cond :list [float ],
sigma_constant :float ,sigma_lt :float |None =None )->go .Figure :
    """Volatilité conditionnelle estimée, comparee a l'écart-type constant actuel."""
    c =theme .P ()
    fig =go .Figure ()
    fig .add_trace (go .Scatter (x =dates ,y =sigma_cond ,mode ="lines",name ="Sigma conditionnel (GARCH)",
    line =dict (color =c ["cfg_red"],width =1.8 ),
    fill ="tozeroy",fillcolor =_alpha (c ["cfg_red"],0.10 )))
    fig .add_hline (y =sigma_constant ,line_color =c ["text_primary"],line_dash ="dash",line_width =1.6 ,
    annotation_text ="Sigma constant (méthode actuelle)",annotation_position ="top left")
    if sigma_lt :
        fig .add_hline (y =sigma_lt ,line_color =c ["amber"],line_dash ="dot",line_width =1.4 ,
        annotation_text ="Niveau de long terme",annotation_position ="bottom left")
    fig .update_layout (**_layout (height =360 ))
    fig .update_yaxes (title ="Volatilité quotidienne (MAD)")
    return fig 


def prevision_volatilite_chart (sigma_passe :list [float ],prevision :list [float ],
sigma_lt :float )->go .Figure :
    """Historique recent et prevision de volatilité, avec le niveau de long terme."""
    c =theme .P ()
    n_p =len (sigma_passe )
    fig =go .Figure ()
    fig .add_trace (go .Scatter (x =list (range (-n_p +1 ,1 )),y =sigma_passe ,mode ="lines",
    name ="Estime",line =dict (color =c ["neutral_series"],width =1.8 )))
    fig .add_trace (go .Scatter (x =list (range (1 ,len (prevision )+1 )),y =prevision ,mode ="lines+markers",
    name ="Prevision",line =dict (color =c ["cfg_red"],width =2.2 ),
    marker =dict (size =6 )))
    fig .add_hline (y =sigma_lt ,line_color =c ["amber"],line_dash ="dot",line_width =1.4 ,
    annotation_text ="Long terme",annotation_position ="bottom right")
    fig .add_vline (x =0 ,line_color =c ["border"],line_width =1 )
    fig .update_layout (**_layout (height =320 ))
    fig .update_xaxes (title ="Jours (0 = aujourd'hui)")
    fig .update_yaxes (title ="Sigma")
    return fig 


def residus_hist_chart (residus :list [float ])->go .Figure :
    """Distribution des residus standardises, comparee a la loi normale de référence."""
    import numpy as _np 
    c =theme .P ()
    r =_np .asarray (residus ,dtype =float )
    fig =go .Figure ()
    fig .add_trace (go .Histogram (x =r ,nbinsx =45 ,histnorm ="probability density",
    marker_color =c ["neutral_series"],opacity =0.75 ,name ="Residus"))
    xs =_np .linspace (float (r .min ()),float (r .max ()),200 )
    ys =_np .exp (-0.5 *xs **2 )/_np .sqrt (2 *_np .pi )
    fig .add_trace (go .Scatter (x =xs ,y =ys ,mode ="lines",name ="Loi normale N(0,1)",
    line =dict (color =c ["cfg_red"],width =2 )))
    fig .update_layout (**_layout (height =300 ,hovermode ="closest"))
    fig .update_xaxes (title ="Residu standardise")
    return fig 


    # --------------------------------------------------------------------------- #
    # Series chronologiques
    # --------------------------------------------------------------------------- #

def decomposition_chart (dates :list ,serie :list [float ],tendance :list [float ])->go .Figure :
    c =theme .P ()
    fig =go .Figure ()
    fig .add_trace (go .Scatter (x =dates ,y =serie ,mode ="lines",name ="MID observe",
    line =dict (color =c ["neutral_series"],width =1.4 )))
    fig .add_trace (go .Scatter (x =dates ,y =tendance ,mode ="lines",name ="Tendance (moyenne mobile)",
    line =dict (color =c ["cfg_red"],width =2.4 )))
    fig .update_layout (**_layout (height =330 ))
    fig .update_yaxes (title ="USD/MAD")
    return fig 


def residu_chart (dates :list ,residu :list [float ])->go .Figure :
    c =theme .P ()
    fig =go .Figure (go .Scatter (x =dates ,y =residu ,mode ="lines",
    line =dict (color =c ["amber"],width =1.2 )))
    fig .add_hline (y =0 ,line_color =c ["border"],line_width =1 )
    fig .update_layout (**_layout (height =220 ,showlegend =False ))
    fig .update_yaxes (title ="Residu")
    return fig 


def correlogramme_chart (valeurs :list [float ],bande :float ,titre :str )->go .Figure :
    """Correlogramme (ACF ou PACF) avec bande de confiance a 95 %."""
    c =theme .P ()
    lags =list (range (len (valeurs )))
    couleurs =[c ["cfg_red"]if abs (v )>bande else c ["neutral_series"]for v in valeurs ]
    fig =go .Figure (go .Bar (x =lags ,y =valeurs ,marker_color =couleurs ,width =0.55 ))
    for signe in (1 ,-1 ):
        fig .add_hline (y =signe *bande ,line_color =c ["amber"],line_dash ="dash",line_width =1.2 )
    fig .add_hline (y =0 ,line_color =c ["border"],line_width =1 )
    fig .update_layout (**_layout (height =280 ,hovermode ="x"))
    fig .update_xaxes (title =f"Retard - {titre }")
    fig .update_yaxes (title ="Correlation",range =[-1.05 ,1.05 ])
    return fig 


def reversion_chart (dates :list ,serie :list [float ],mu :float ,sigma_eq :float )->go .Figure :
    """Série et niveau d'equilibre estimé, avec une bande de plus ou moins un écart-type."""
    c =theme .P ()
    fig =go .Figure ()
    fig .add_trace (go .Scatter (x =dates ,y =serie ,mode ="lines",name ="MID observe",
    line =dict (color =c ["neutral_series"],width =1.5 )))
    fig .add_hline (y =mu ,line_color =c ["cfg_red"],line_width =2 ,
    annotation_text ="Niveau d'equilibre estimé",annotation_position ="top left")
    if sigma_eq and sigma_eq >0 :
        fig .add_hline (y =mu +sigma_eq ,line_color =c ["amber"],line_dash ="dot",line_width =1.2 )
        fig .add_hline (y =mu -sigma_eq ,line_color =c ["amber"],line_dash ="dot",line_width =1.2 )
    fig .update_layout (**_layout (height =340 ))
    fig .update_yaxes (title ="USD/MAD")
    return fig 


    # --------------------------------------------------------------------------- #
    # Regression logistique
    # --------------------------------------------------------------------------- #

def logistique_chart (grille :list [float ],probas :list [float ],esperances :list [float ],
delta_opt :float ,obs_deltas :list [float ],obs_taux :list [float ])->go .Figure :
    """Courbe de probabilite ajustee, taux observes, et esperance de gain."""
    c =theme .P ()
    pips =[d *10_000 for d in grille ]
    fig =go .Figure ()
    fig .add_trace (go .Scatter (x =[d *10_000 for d in obs_deltas ],y =obs_taux ,mode ="markers",
    name ="Taux observé par palier",
    marker =dict (size =10 ,color =c ["amber"],
    line =dict (color =c ["text_primary"],width =1 ))))
    fig .add_trace (go .Scatter (x =pips ,y =probas ,mode ="lines",name ="P(acceptation) - logistique",
    line =dict (color =c ["neutral_series"],width =2 ,dash ="dot")))
    m =max (esperances )if esperances else 1.0 
    fig .add_trace (go .Scatter (x =pips ,y =[e /m for e in esperances ],mode ="lines",
    name ="Esperance de gain (normalisee)",
    line =dict (color =c ["cfg_red"],width =2.4 ),
    fill ="tozeroy",fillcolor =_alpha (c ["cfg_red"],0.10 )))
    fig .add_vline (x =delta_opt *10_000 ,line_color =c ["green"],line_dash ="dash",line_width =2 ,
    annotation_text ="Optimum",annotation_position ="top")
    fig .update_layout (**_layout (height =380 ))
    fig .update_xaxes (title ="Écart au mid (pips)")
    fig .update_yaxes (title ="Probabilite / esperance normalisee",range =[0 ,1.05 ])
    return fig 


    # --------------------------------------------------------------------------- #
    # Segmentation
    # --------------------------------------------------------------------------- #

def segments_scatter (montants :list [float ],taux :list [float ],segments :list [int ],
noms :list [str ],tailles :list [float ])->go .Figure :
    """Positionnement des clients selon leur montant moyen et leur taux d'acceptation."""
    c =theme .P ()
    palette =[c ["cfg_red"],c ["green"],c ["amber"],c ["neutral_series"],c ["red_alert"]]
    fig =go .Figure ()
    for s in sorted (set (segments )):
        idx =[i for i ,v in enumerate (segments )if v ==s ]
        fig .add_trace (go .Scatter (
        x =[montants [i ]for i in idx ],y =[taux [i ]for i in idx ],
        mode ="markers+text",name =f"Segment {s +1 }",
        text =[noms [i ]for i in idx ],textposition ="top center",
        textfont =dict (size =9 ,color =c ["text_secondary"]),
        marker =dict (size =[max (9.0 ,min (30.0 ,tailles [i ]/4 ))for i in idx ],
        color =palette [s %len (palette )],
        line =dict (color =c ["bg_panel"],width =1 )),
        ))
    fig .update_layout (**_layout (height =420 ,hovermode ="closest"))
    fig .update_xaxes (title ="Montant moyen par opération (USD)",type ="log")
    fig .update_yaxes (title ="Taux d'acceptation")
    return fig 


def silhouette_chart (ks :list [int ],scores :list [float ])->go .Figure :
    """Coefficient de silhouette selon le nombre de groupes retenu."""
    c =theme .P ()
    best =scores .index (max (scores ))if scores else 0 
    couleurs =[c ["cfg_red"]if i ==best else c ["neutral_series"]for i in range (len (ks ))]
    fig =go .Figure (go .Bar (x =[str (k )for k in ks ],y =scores ,marker_color =couleurs ,
    text =[f"{s :.3f}"for s in scores ],textposition ="outside"))
    fig .update_layout (**_layout (height =280 ,hovermode ="x"))
    fig .update_xaxes (title ="Nombre de groupes")
    fig .update_yaxes (title ="Silhouette")
    return fig 


    # --------------------------------------------------------------------------- #
    # Detection d'anomalies
    # --------------------------------------------------------------------------- #

def anomalies_chart (codes :list [str ],scores :list [float ],seuil :float )->go .Figure :
    """Score d'anomalie par banque, avec le seuil de detection."""
    c =theme .P ()
    couleurs =[c ["red_alert"]if abs (s )>seuil else c ["neutral_series"]for s in scores ]
    fig =go .Figure (go .Bar (x =codes ,y =scores ,marker_color =couleurs ,
    text =[f"{s :.2f}"for s in scores ],textposition ="outside"))
    for signe in (1 ,-1 ):
        fig .add_hline (y =signe *seuil ,line_color =c ["amber"],line_dash ="dash",line_width =1.4 )
    fig .add_hline (y =0 ,line_color =c ["border"],line_width =1 )
    fig .update_layout (**_layout (height =320 ,hovermode ="x"))
    fig .update_yaxes (title ="Score z modifie")
    return fig 


    # --------------------------------------------------------------------------- #
    # Etudes d'evenement (inspirees de l'etude dividendes)
    # --------------------------------------------------------------------------- #

def resilience_chart (courbe )->go .Figure :
    """Trajectoire événementielle du MID autour des jours a fort flux net."""
    c =theme .P ()
    fig =go .Figure ()
    fig .add_trace (go .Scatter (
    x =courbe ["t"],y =courbe ["ecart_pips"],mode ="lines+markers",
    line =dict (color =c ["cfg_red"],width =2.4 ),marker =dict (size =5 ),
    name ="Écart du MID (pips)",fill ="tozeroy",fillcolor =_alpha (c ["cfg_red"],0.10 )))
    fig .add_hline (y =0 ,line_color =c ["border"],line_width =1 )
    fig .add_shape (type ="line",x0 =0 ,x1 =0 ,yref ="paper",y0 =0 ,y1 =1 ,
    line =dict (color =c ["amber"],width =1.6 ,dash ="dash"))
    fig .add_annotation (x =0 ,yref ="paper",y =1.04 ,text ="Jour du flux (t=0)",
    showarrow =False ,font =dict (color =c ["amber"],size =10 ,family =FONT_MONO ))
    fig .update_layout (**_layout (height =380 ,hovermode ="x"))
    fig .update_xaxes (title ="Jours de bourse autour de l'événement")
    fig .update_yaxes (title ="Écart du MID vs veille (pips)")
    return fig 


def breach_position_chart (courbe )->go .Figure :
    """Trajectoire moyenne de la position (% de la limite) autour d'un dépassement."""
    c =theme .P ()
    fig =go .Figure ()
    fig .add_trace (go .Scatter (
    x =courbe ["t"],y =courbe ["pct"],mode ="lines+markers",
    line =dict (color =c ["cfg_red"],width =2.4 ),marker =dict (size =5 ),
    name ="Position (% limite)"))
    fig .add_hline (y =80 ,line_color =c ["amber"],line_dash ="dash",line_width =1.4 ,
    annotation_text ="Seuil d'alerte 80 %",annotation_position ="right")
    fig .add_hline (y =100 ,line_color =c ["red_alert"],line_dash ="dash",line_width =1.4 ,
    annotation_text ="Limite 100 %",annotation_position ="right")
    fig .add_shape (type ="line",x0 =0 ,x1 =0 ,yref ="paper",y0 =0 ,y1 =1 ,
    line =dict (color =c ["text_secondary"],width =1.4 ,dash ="dot"))
    fig .add_annotation (x =0 ,yref ="paper",y =1.04 ,text ="Depassement (t=0)",
    showarrow =False ,font =dict (color =c ["text_secondary"],size =10 ,family =FONT_MONO ))
    fig .update_layout (**_layout (height =380 ,hovermode ="x"))
    fig .update_xaxes (title ="Jours de bourse autour du dépassement")
    fig .update_yaxes (title ="Position en % de la limite")
    return fig 


def persistance_chart (taux_apres_acc :float ,taux_apres_ref :float )->go .Figure :
    """Barres comparant le taux d'acceptation selon l'issue précédente."""
    c =theme .P ()
    fig =go .Figure (go .Bar (
    x =["Après une acceptation","Après un refus"],
    y =[taux_apres_acc ,taux_apres_ref ],
    marker_color =[c ["green"],c ["red_alert"]],
    text =[f"{taux_apres_acc :.1f} %",f"{taux_apres_ref :.1f} %"],
    textposition ="outside",width =0.5 ))
    fig .update_layout (**_layout (height =340 ,hovermode ="x",showlegend =False ))
    fig .update_xaxes (title ="Issue de la demande précédente du même client")
    fig .update_yaxes (title ="Taux d'acceptation de la demande suivante (%)")
    return fig 


def transition_heatmap (matrice )->go .Figure :
    """Matrice de transition des issues de cotation (heatmap)."""
    c =theme .P ()
    z =matrice .values .tolist ()
    fig =go .Figure (go .Heatmap (
    z =z ,x =["Accepte","Refuse"],y =["Prec. acceptee","Prec. refusee"],
    colorscale =[[0 ,c ["bg_panel"]],[1 ,c ["cfg_red"]]],
    text =z ,texttemplate ="%{text}",textfont =dict (family =FONT_MONO ,size =14 ),
    showscale =False ))
    fig .update_layout (**_layout (height =300 ,hovermode =False ))
    return fig 
