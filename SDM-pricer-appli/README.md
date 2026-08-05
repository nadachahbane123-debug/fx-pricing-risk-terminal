# CFG BANK — USDMAD PRICER TERMINAL

Pricer FX USD/MAD style "terminal Bloomberg", developpe pour la salle des marches de
CFG Bank (Projet de Fin d'Annee). Streamlit + Plotly + SQLite, sans dependance externe
payante.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`. Au premier lancement, `pricer.db`
est cree et pre-rempli avec des donnees de demonstration (cotations concurrentes,
historique de MID, transactions d'exemple).

## Structure du projet

```
app.py                     Point d'entree Streamlit — 11 pages pilotees par st.session_state
core/
  models.py                 Logique metier pure (sections 1-12 : MID theorique, bande,
                             marche, position, sigma, gamma/eta/lambda, Ho-Stoll, Almgren-Chriss,
                             conformite) — testable sans Streamlit
  db.py                      Persistance SQLite (transactions, historique de prix, depassements)
  pnl.py                     Calcul du P&L (flux de tresorerie + mark-to-market, section 18)
  backtest.py                Rejeu historique point-in-time du skew (section 16)
  stress.py                  Scenarios de choc / tests de resistance (section 19)
ui/
  theme.py                   Palette de couleurs + injection CSS globale (habillage terminal)
  components.py               Cartes de metriques, badges, ticker tape, sparklines,
                               bandeau d'alerte de position, navigation par codes de fonction
  charts.py                    Graphiques Plotly avec theme sombre reutilisable
.streamlit/config.toml       Theme sombre Streamlit natif
assets/logo-cfg.png          Logo CFG Bank (fond transparent)
exports/                     Exports CSV quotidiens generes automatiquement
audit_log.csv                Journal d'audit de conformite (ecriture en mode ajout uniquement)
pricer.db                    Base SQLite (creee au premier lancement)
```

## Fonctionnement

- **Navigation** : la barre laterale utilise des boutons libelles comme des codes de
  fonction Bloomberg (`DASH`, `MKT<GO>`, `POS<GO>`, ...) pilotes par
  `st.session_state["page_active"]` — pas de `st.tabs()` natif, ce qui permet a n'importe
  quel bouton de rediriger l'utilisateur vers une autre page (voir le bouton "Panic Mode"
  de la page `POS<GO>`, qui bascule automatiquement vers `LIQD<GO>`).
- **Ticker** : le bandeau defilant en haut de l'ecran est genere dynamiquement a partir
  du tableau de cotations concurrentes (`st.session_state["quotes"]`, editable dans
  `MKT<GO>`) — toute modification s'y reflete immediatement.
- **Historisation quotidienne** : a chaque lancement/rechargement, si la date du jour
  n'a pas encore d'enregistrement dans `price_history`, le MID de marche courant est
  calcule et enregistre, un export CSV est genere dans `exports/`, et une ligne est
  ajoutee a `audit_log.csv`.
- **Backtest du skew** : rejoue le modele Ho & Stoll jour par jour en n'utilisant que
  les donnees disponibles a chaque date (pas de biais d'anticipation).

## Modeles implementes

- **Ho & Stoll (1981)** — *Optimal dealer pricing under transactions and return
  uncertainty*, Journal of Financial Economics. Decalage du prix de reservation en
  fonction de la position et de l'aversion au risque.
- **Almgren & Chriss (2000)** — *Optimal execution of portfolio transactions*, Journal
  of Risk. Trajectoire de liquidation optimale.

Voir la page `DOCS<GO>` dans l'application pour le detail methodologique complet.

## Apercu

Terminal sombre (`#0A0E14`/`#12161F`), police monospace (JetBrains Mono) pour tous les
chiffres, accent rouge institutionnel CFG (`#C8102E`), vert/rouge pour les variations et
badges de conformite, bandeau ticker defilant, cartes de metriques avec sparklines,
tableaux denses a lignes alternees, jauge de risque type thermometre.
