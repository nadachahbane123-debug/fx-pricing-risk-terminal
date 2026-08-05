# FX Pricing & Risk Terminal — USD/MAD

> Terminal de *market making* sur le change USD/MAD développé pour la salle des marchés de **CFG Bank** (Projet de Fin d'Année — École Mohammadia d'Ingénieurs).

L'application price le dollar américain face au dirham marocain en intégrant, à chaque instant, le risque de la position déjà détenue par le desk — c'est un pricer dynamique qui raisonne en teneur de marché. Elle réunit **22 modules** couvrant le pricing, la gestion de position, le risque, la volatilité, les séries temporelles, le machine learning et trois études d'événement.

**Stack technique :** `Python` · `Streamlit` · `Plotly` · `SQLite` · `NumPy` · `pandas` · `SciPy` *(sans dépendance externe payante)*.

---

## Aperçu des fonctionnalités

### Pricing & Cotation
* **MID théorique USD/MAD :** Calcul basé sur la relation de panier officiel (60 % EUR / 40 % USD) et respect de la bande réglementaire de ±5 % fixée par Bank Al-Maghrib.
* **Cotation client sur demande (RFQ) :** Génération automatique avec grille de spread adaptative selon la taille d'opération.
* **Marché à terme & Swaps :** Calcul des cours Forward et Swaps via la Parité des Taux d'Intérêt Couverte (CIP).

### Cœur du modèle — Décalage de prix selon la position
* **Modèle de Ho & Stoll (1981) :** Décalage (*skew*) du prix bid/ask en fonction de l'inventaire détenu et de l'aversion au risque du trader.
* **Aversion au risque calibrée :** Valeur déduite d'un objectif de gestion (décalage cible) plutôt que choisie arbitrairement.
* **Backtest point-in-time :** Rejeu historique rigoureux sans biais d'anticipation.

### Risque & P&L
* **Value at Risk (VaR) :** Calcul paramétrique (95 % et 99 % de confiance) sur différents horizons temporels.
* **Suivi & Attribution du P&L :** Séparation claire entre trésorerie engagée et valorisation Mark-to-Market (MtM).
* **Risque de portefeuille multi-devises :** Suivi des limites prudentielles avec journal de conformité en temps réel.
* **Liquidation optimale :** Plan d'exécution basé sur le modèle d'**Almgren & Chriss (2000)**.

### Analyse quantitative & ML
* **Volatilité conditionnelle :** Modélisation avancée via GARCH(1,1), GJR-GARCH et EGARCH.
* **Séries chronologiques :** Décomposition de tendance, tests de stationnarité (ADF) et analyse de retour à la moyenne.
* **Segmentation & Anomalies :** Clustering des clients par K-Means et détection d'anomalies via score Z modifié.

### Études d'événement (*Event Studies*)
* **Résilience du prix :** Analyse de l'impact de marché empirique suite à un choc de flux.
* **Dépassements de limites :** Étude du comportement de l'inventaire lors des franchissements de seuils critiques.
* **Acceptation client :** Test d'indépendance du Khi-deux sur la persistance d'acceptation des cotations.

---

## Architecture du projet

```text
fx-pricing-terminal/
├── app.py                     # Point d'entrée Streamlit (22 pages pilotées par st.session_state)
│
├── core/                      # Logique métier pure (indépendante de l'interface)
│   ├── models.py              # MID théorique, bande BAM, Ho-Stoll & calibration
│   ├── db.py                  # Couche de persistance SQLite (transactions, historique)
│   ├── datagen.py             # Générateur de données de marché (GARCH + impact de flux)
│   ├── events.py              # Modules d'études d'événement
│   ├── pnl.py                 # Calculs du P&L (Trésorerie + Mark-to-Market)
│   ├── risk.py                # Moteur de VaR et métriques de risque
│   ├── portfolio.py           # Portefeuille multi-devises et règles de conformité
│   ├── flow.py                # Traitement et analyse des flux d'ordres
│   ├── segmentation.py        # Algorithme K-Means pour le clustering client
│   ├── anomaly.py             # Détection d'anomalies par score Z modifié
│   ├── vol_models.py          # Modélisation GARCH / GJR-GARCH / EGARCH
│   ├── timeseries.py          # Décomposition et tests de stationnarité (ADF)
│   ├── forward.py             # Calculs Forward et Swaps (CIP)
│   ├── backtest.py            # Moteur de rejeu historique
│   └── stress.py              # Scénarios de stress testing
│
└── ui/                        # Composants d'interface utilisateur
    ├── theme.py               # Palette graphique & CSS personnalisé (style Bloomberg)
    ├── components.py          # KPI cards, badges, ticker défilant & navigation
    └── charts.py              # Graphiques Plotly configurés en mode sombre
```
---

## Modèles de référence

| Domaine | Modèle | Référence |
|---|---|---|
| Décalage de prix | Ho & Stoll | *Optimal dealer pricing under transactions and return uncertainty* (1981) |
| Liquidation | Almgren & Chriss | *Optimal execution of portfolio transactions* (2000) |
| Volatilité | GARCH / GJR / EGARCH | Bollerslev (1986), Glosten-Jagannathan-Runkle (1993), Nelson (1991) |
| Stationnarité | Dickey-Fuller augmenté | Fuller (1976) |
| Détection d'anomalies | Score z modifié | Iglewicz & Hoaglin (1993) |

Le détail méthodologique complet est disponible dans les pages **THEORY** et **DOCS** de l'application.

---

## Interface

Terminal sombre, police monospace pour les chiffres, accent rouge institutionnel CFG (`#C8102E`), code couleur vert/rouge pour les variations, ticker défilant, cartes de métriques avec *sparklines*, tableaux denses et jauge de risque.

---

##  Réalisé par

**Nada Chahbane** 
*Élève ingénieure en Modélisation Informatique Scientifique* — **École Mohammadia d'Ingénieurs (EMI)**  
Projet réalisé dans le cadre d'un stage de fin d'année au sein de la Salle des Marchés de **CFG Bank**.
