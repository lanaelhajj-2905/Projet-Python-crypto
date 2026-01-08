# Projet Python


## Présentation générale

Ce projet vise à développer une **stratégie d’allocation quantitative sur cryptomonnaies**, fondée sur la **prévision de la volatilité conditionnelle**. 

Le projet est volontairement structuré en **trois grandes phases distinctes** :

1. **Construction de l'univers d'investissement**
2. **Stratégies de trading et allocation**
3. **Phase expérimentale complémentaire**

L'approche repose sur une **séparation stricte entre la donnée, l'analyse statistique et la logique de décision**, afin de garantir robustesse, lisibilité et reproductibilité.

## Quick Start
1. **Phase 1 (Univers)**: `python main_universe.py` -> Définit les 6 actifs.
2. **Phase 2 (GARCH)**: `python main_strategy.py` -> GARCH et inverse volatilité
3. **Phase 3 (Stratégie complémentaire)**: `python main_lowvol_trend.py` -> Exécute la stratégie Low-Vol + Trend.

---

## Phase 1 : Univers d'investissement

Cette phase constitue le **socle quantitatif du projet**. Elle est volontairement **indépendante de toute logique de trading ou d'optimisation**, afin d'éviter tout biais méthodologique.

L'objectif est de définir un **univers crypto liquide et représentatif**
--- 

## Univers retenu

L'univers final est composé de **6 cryptomonnaies**, sélectionnées parmi 18 actifs majeurs analysés.  L'étude détaillée se trouve dans `notebooks/universe-research.md`.

### Processus de sélection

**Étape 1 : Univers initial (18 cryptos)**
```
BTC, ETH, XRP, SOL, TRX, ADA, DOGE, AVAX, DOT,
LTC, SHIB, ICP, LINK, BCH, NEAR, UNI, ATOM, ETC
```

**Étape 2 : Analyse de corrélation**
- Calcul de la matrice de corrélation complète
- Identification des corrélations extrêmes (>0.9 ou <0.1)
- Analyse des structures de dépendance

**Étape 3 : Sélection optimale (6 cryptos)**

### Critères de sélection

1. **Corrélations modérées (0.2 - 0.7)**
   - Évite la redondance des actifs trop corrélés
   - Prévient l'instabilité d'estimation des corrélations trop faibles
   - Garantit des gains de diversification effectifs

2. **Diversité des chocs de volatilité**
   - Actifs réagissant à des facteurs distincts
   - Structure conditionnelle non redondante pour modèles GARCH
   - Transmission de volatilité différenciée

3. **Liquidité et capitalisation**
   - Top 10 par capitalisation (hors stablecoins)
   - Volume quotidien >$500M
   - Spreads bid-ask serrés

4. **Qualité des données historiques**
   - Historique >2 ans de données fiables
   - Absence de delisting sur la période
   - Données haute fréquence disponibles

5. **Diversité fonctionnelle**
   - Couverture de différents cas d'usage (paiement, DeFi, smart contracts)
   - Différentes architectures blockchain
   - Facteurs de risque complémentaires

### Actifs retenus

| Crypto | Justification | Corrélation moyenne | Cap. ($B) |
|--------|---------------|---------------------|-----------|
| **BTC** | Référence du marché, transmetteur principal de volatilité | - | ~1,200 |
| **ETH** | Infrastructure DeFi/smart contracts, 2ème émetteur de volatilité | 0.79 avec BTC | ~450 |
| **XRP** | Décorrélation structurelle (paiements interbancaires), corrélations moyennes | 0.3-0.5 | ~30 |
| **ADA** | 2ème transmetteur après ETH, dynamique altcoins | 0.7 avec BTC | ~10 |
| **SOL** | Chocs idiosyncratiques (FTX), infrastructure haute performance | 0.7 avec BTC | ~8 |
| **DOGE** | Volatilité extrême, comportement spéculatif découplé | 0.7 avec BTC | ~10 |

**Couverture** : ~70% de la capitalisation totale crypto (hors stablecoins)

### Justification détaillée

La sélection finale privilégie un **équilibre optimal entre diversification et stabilité économétrique** :

- **BTC + ETH** : Duo indispensable jouant des rôles économiques distincts
- **XRP** : Seul actif à corrélation véritablement modérée, enrichit la matrice de covariance
- **ADA + SOL** : Plateformes smart contracts complémentaires avec chocs propres
- **DOGE** : Capture la dimension spéculative/irrationnelle du marché

Cette configuration évite les écueils suivants :
- Corrélations >0.9 : redondance informationnelle (ex: LTC/BCH trop proches de BTC)
- Corrélations <0.1 : instabilité d'estimation GARCH
- Actifs illiquides : bruit dans les données (SHIB, NEAR)

**Référence complète** : Voir `notebooks/universe-research.md` pour l'analyse bibliographique détaillée (Chen 2024, Kyriazis 2021, Adams & Füss 2017, etc.)

---

## Données

* **Exchange** : Binance
* **Paires** : USDT (prioritaire), puis BUSD / USD
* **Fréquence** : Daily (1d)
* **Accès API** : `ccxt`

---

## Méthodologie

### 1. Construction d'un panel équilibré

* Suppression des dates avec observations manquantes
* Alignement temporel strict entre tous les actifs
* Arbitrage assumé entre profondeur historique et qualité des données

### 2. Rendements logarithmiques

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Justification** (Markowitz, 1952) :
- Additivité temporelle
- Symétrie des gains/pertes
- Propriétés statistiques supérieures

### 3. Annualisation

* **Facteur** : 365 jours (marchés crypto en continu)
* Volatilité annuelle : $\sigma_{ann} = \sigma_{daily} \times \sqrt{365}$
* Covariance annuelle : $\Sigma_{ann} = \Sigma_{daily} \times 365$

---

## Métriques calculées

| Métrique     | Usage                         |
| ------------ | ----------------------------- |
| Log returns  | Analyse des rendements        |
| Volatilité   | Mesure du risque              |
| Covariance   | Construction de portefeuilles |
| Corrélation  | Analyse de diversification    |
| Sharpe ratio | Performance ajustée du risque |
| Skewness     | Asymétrie des distributions   |
| Kurtosis     | Risque de queues épaisses     |

---

## Architecture de la phase 1 : 

```
PROJET-PYTHON-CRYPTO/
│
├─ src/
│   └─ universe/
│       ├─ fetcher.py         # Téléchargement des données
│       ├─ analyzer.py        # Calculs financiers
│       └─ exporter.py        # Exports CSV
│
├─ notebooks/
│   └─ universe-research.md   # Analyse et justification de la sélection
│
├─ data/
│   └─ processed/
│       └─ universe/          # Outputs (18 cryptos analysées)
│
├─ out/
│   └─ universe.logs/
│
├─ main_universe.py
├─ README.md

```

---

## Pipeline de données

```
Binance API → OHLCV (18 cryptos) → Cleaning → Panel alignment
           → Log returns → Statistics → CSV exports (18×18)
           → Analyse manuelle de corrélation → Sélection de 6 cryptos (Phase 2)
```

---

## Outputs générés

Tous les fichiers sont exportés dans `data/processed/universe/` :

| Fichier                | Description                     
| ---------------------- | -------------------------------
| coverage_report.csv    | Audit de couverture des données
| prices_close.csv       | Prix de clôture historiques 
| log_returns.csv        | Rendements logarithmiques
| covariance_daily.csv   | Covariance journalière
| covariance_annual.csv  | Covariance annualisée
| correlation.csv        | **Matrice de corrélation complète**
| summary_statistics.csv | Statistiques descriptives

**Note :** L'analyse de la matrice de corrélation (18×18) a permis d'identifier les 6 cryptomonnaies optimales pour la Phase 2. Cette sélection est documentée dans `notebooks/universe-research.md`.

---

## Exécution

### Installation

```bash
git clone https://github.com/lanaelhajj-2905/Projet-Python-crypto
cd projet-python-crypto

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Lancer l'analyse de l'univers d'investissement

```bash
# Analyse de l'univers complet (18 cryptos)
python main_universe.py

# Durée estimée : Analyse univers terminée en 16.43s
```

Les résultats sont automatiquement sauvegardés :
- Tous les outputs → `data/processed/universe/`
- Logs → `out/logs/universe.log`

**Étape suivante :** Analyser manuellement `correlation.csv` pour valider la sélection des 6 cryptomonnaies documentée dans `notebooks/universe-research.md`.


---

## Références académiques

### Fondements théoriques

- **Markowitz, H. (1952)**. *Portfolio Selection*. Journal of Finance, 7(1), 77-91.
- **Sharpe, W. F. (1966)**. *Mutual Fund Performance*. Journal of Business, 39(1), 119-138.

### Sélection d'actifs et corrélations

- **Chen, H. (2024)**. *Can optimal diversification beat the naive 1⁄N strategy in a highly correlated market? Empirical evidence from cryptocurrencies*. Okayama University.
- **Kyriazis, N. (2021)**. *A Survey on Volatility Fluctuations in the Decentralized Cryptocurrency Financial Assets*. University of Thessaly.
- **Adams, Z & Füss, R. (2017)**. *Are correlations constant? Empirical and theoretical results on popular correlation models in finance*. Journal of Banking & Finance, 84, 9-24.

### Transmission de volatilité

- **Korkusuz, B. (2025)**. *Volatility Transmission in Digital Assets: Ethereum's Rising Influence*. J. Risk Financial Manag.
- **Sahiner, M & Korkusuz, B. (2025)**. *Coin impact on cross-crypto realized volatility and dynamic cryptocurrency volatility connectedness*. Volume 11, article 129.


---

## Phase 2 : Stratégies de trading et allocation

Cette phase exploite l'univers d'investissement construit en Phase 1 pour développer et évaluer une **stratégie de trading quantitative** basée sur la prévision de volatilité conditionnelle.

### Objectifs opérationnels

* Sélectionner le **meilleur modèle GARCH** via une approche walk-forward
* Générer des **prévisions de volatilité conditionnelle** pour les 6 actifs retenus
* Construire une **stratégie d'allocation inverse-volatilité**
* Évaluer les performances en conditions réelles avec coûts de transaction

---

**Partie 1 : la selection du modèle de volatilité**

1. Données et période d’étude : 

La sélection du modèle de volatilité est réalisée exclusivement sur le Bitcoin (BTC-USD), considéré comme l’actif de référence du marché crypto.

• Source des données : Yahoo Finance

• Fréquence : quotidienne

• Période étudiée : janvier 2018 – décembre 2025

Les rendements sont calculés sous forme de rendements logarithmiques en pourcentage :
r_t = 100 × ln(P_t / P_{t-1})

Contrairement aux marchés actions, les marchés de crypto-actifs fonctionnent en continu (24h/24, 7j/7) et ne disposent pas d’une heure de clôture officielle. Les prix de clôture fournis par Yahoo Finance correspondent à une coupure calendaire quotidienne à 00:00 UTC. 
Cette convention, bien qu’arbitraire, est appliquée de manière cohérente dans le temps et entre actifs, ce qui permet la construction de séries de rendements journaliers comparables.



2. Modèles candidats testés :

| Modèle | Spécification | Distribution | Description |
|--------|---------------|--------------|--------------|
| GARCH(1,1)-Normal | p=1, o=0, q=1 | Normale | Modèle de référence simple, avec volatilité symétrique et innovations gaussiennes. |
| **GARCH(1,1)-t** | p=1, o=0, q=1 | **Student-t** | Permet de capturer les queues épaisses, caractéristique empirique majeure des rendements crypto. |
| EGARCH(1,1)-t | p=1, o=0, q=1 | Student-t | Modélise la variance sous forme logarithmique et autorise des effets asymétriques sans contrainte de positivité. |
| GJR-GARCH(1,1)-t | p=1, o=1, q=1 | Student-t |Introduit explicitement un effet de levier, où les chocs négatifs peuvent avoir un impact plus important sur la volatilité.|

Ces modèles couvrent les principales extensions utilisées dans la littérature académique sur la volatilité conditionnelle.

3. Justification du choix du Bitcoin pour la sélection :

 La sélection du modèle de volatilité est réalisée uniquement sur le **Bitcoin**. 
 Nous avons pris cette décision car :
-Il s’agit du crypto-actif le plus liquide et le plus ancien.
-Il présente une profondeur de marché plus élevée et moins de bruit.
-Sa série de rendements est longue et relativement stable ;
-Il constitue une référence naturelle pour calibrer un modèle générique de volatilité crypto.

L’hypothèse sous-jacente est que la dynamique de volatilité du Bitcoin est représentative, au moins qualitativement, de celle du marché crypto dans son ensemble.

4. Split temporel : Découpage Train / Validation / Test** :

Les données sont découpées en trois périodes : 
```
Training     : 2020-04-11 → 2021-12-31  (apprentissage initial) Utilisée pour constituer l’historique initial nécessaire à l’estimation des paramètres.
Validation   : 2022-01-01 → 2023-12-31  (sélection du modèle GARCH) Utilisée pour comparer les performances de prévision des différents modèles.
Test         : 2024-01-01 → 2025-12-31  (évaluation finale) Utilisée exclusivement pour l’évaluation finale hors échantillon.
```

**Note importante** : Le modèle est sélectionné **uniquement sur la période de validation** et n’est pas réajusté sur la période de test, garantissant une évaluation « honnête ».


5. Critère de sélection : QLIKE (Quasi-Likelihood) : 

La comparaison des modèles repose sur la fonction de perte QLIKE :

$$\text{QLIKE} = \log(\sigma^2) + \frac{r^2}{\sigma^2}$$

La QLIKE est une métrique cohérente pour l’évaluation des prévisions de variance et est robuste aux erreurs de spécification du modèle.
Un score QLIKE plus faible indique une meilleure capacité à prévoir la volatilité conditionnelle.

Interprétation du Q-like : Un QLIKE plus faible indique une meilleure capacité à prévoir la volatilité conditionnelle.

6. Modèle retenu : GARCH(1,1)-t :

Le modèle présentant le score QLIKE le plus faible sur la période de validation est : 
GARCH(1,1) avec innovations Student-t.
Ce choix est cohérent avec les caractéristiques du marché crypto, notamment la persistance de la volatilité et la présence de queues épaisses.

Le modèle retenu est défini par :

$$\sigma^2_{t+1} = \omega + \alpha \cdot r^2_t + \beta \cdot \sigma^2_t$$

La moyenne conditionnelle des rendements est supposée nulle, ce qui est une hypothèse standard à l’horizon journalier.
Les paramètres sont estimés par maximum de vraisemblance.

**Implémentation** :
- Les paramètres du modèle (ω, α, β) sont ré-estimés chaque lundi
- Entre deux recalibrages, les paramètres sont maintenus constants, tandis que la variance conditionnelle est mise à jour quotidiennement via la récursion GARCH.
- Un nombre minimal de 250 observations est requis avant toute estimation afin de garantir la stabilité des paramètres.
- **Update quotidien** : Entre deux refits, la variance conditionnelle est mise à jour quotidiennement via la récursion GARCH
- **Forecast next-day** : La volatilité prévue σ_{t+1} est stockée à la date t

7. Alignement temporel critique :
```python
# Les forecasts sont stockés comme σ_{t+1} à date t
vols_raw = forecast_volatility(returns, start, end)

# Pour construire les poids à date t basés sur σ_t :
vols_aligned = vols_raw.shift(1)  # Shift de 1 jour
```
La récursion GARCH produit une prévision de volatilité à un jour (σ̂_{t+1}).

Afin d’éviter tout biais d’anticipation, les séries de volatilité sont décalées d’un jour avant d’être utilisées pour la construction des signaux.
Ainsi, les décisions prises au jour t reposent exclusivement sur l’information disponible au jour t−1.


**Partie 2. Stratégie d'allocation**

1. Univers d’investissement et données :

La stratégie est appliquée à l'univers composé de six crypto-actifs de la phase 1 :
BTC, ETH, XRP, ADA, SOL et DOGE.

• Source des données : Yahoo Finance

• Fréquence : quotidienne

Contrairement à la sélection du modèle (Partie 1), l’univers multi-actifs ne dispose pas de données suffisamment longues avant 2020 pour l’ensemble des crypto-actifs. L’analyse de portefeuille débute donc plus tard.

2. Périodes Train / Validation / Test (stratégie multi-actifs) : 

Pour la stratégie d’allocation, les périodes sont définies comme suit :

1 - Train : 2020 – 2021  : Historique initial nécessaire pour produire les premières prévisions de volatilité.
2 - Validation : 2022 – 2023 : Utilisée pour analyser le comportement et la cohérence de la stratégie.
3 - Test : 2024 – 2025 : Période d’évaluation finale hors échantillon.

Cette différence de découpage par rapport à la Partie 1 s’explique par la disponibilité des données sur l’ensemble de l’univers.

3. Prévisions de volatilité multi-actifs : 

Pour chaque crypto-actif, la volatilité conditionnelle est estimée à l’aide du même modèle GARCH(1,1)-t sélectionné précédemment.

Les règles de recalibrage, de mise à jour et d’alignement temporel sont strictement identiques pour tous les actifs.


4. Stratégie : Inverse Volatility (Risk Parity) :

Les poids du portefeuille sont définis comme proportionnels à l’inverse de la volatilité estimée :

$$w_i(t) = \frac{1/\sigma_i(t)}{\sum_{j=1}^{N} 1/\sigma_j(t)}$$

**Cette approche vise à** :
-Allouer davantage de capital aux actifs moins volatils.
-Equilibrer les contributions au risque.
-Limiter l’exposition aux actifs extrêmement volatils.

5. Paramètres :
- **Plafonnement** : Poids maximum de 35% par actif (évite concentration excessive)
- **Rebalancement** : Hebdomadaire lundi (W-MON), poids constants entre refits et aucune position n’est prise tant qu’aucun signal de volatilité valide n’est disponible.
- **Coûts de transaction** : 10 bps (0.1%) par trade. Les coûts de transaction sont modélisés de manière linéaire comme proportionnels au turnover du portefeuille : coût_t = (bps / 10000) × Σ_i |w_{t,i} − w_{t−1,i}. Cette modélisation constitue une approximation standard, sans prise en compte explicite de l’impact de marché.


6. Benchmarks :

Deux benchmarks sont utilisés pour évaluer la performance relative de la stratégie :

| Benchmark | Description | Justification |
|-----------|-------------|---------------|
| **Equal-Weight (1/N)** | Allocation équipondérée (16.67% par actif) | Stratégie naïve robuste, fréquemment utilisée comme référence académique (DeMiguel et al., 2009) |
| **BTC Buy-and-Hold** | 100% Bitcoin | Référence du marché crypto |

---

7. Métriques d'évaluation :

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Total Return** | $(V_f / V_i) - 1$ | Rendement cumulé |
| **Annualized Return** | $(V_f / V_i)^{365/n} - 1$ | CAGR |
| **Annualized Vol** | $\sigma_{daily} \times \sqrt{365}$ | Risque annualisé |
| **Sharpe Ratio** | $(R - R_f) / \sigma$ | Rendement ajusté au risque |
| **Sortino Ratio** | $R / \text{Downside Dev}$ | Pénalise uniquement volatilité baissière |
| **Maximum Drawdown** | $\min(\text{Equity} / \text{Peak} - 1)$ | Perte maximale pic-à-creux |
| **Calmar Ratio** | $R_{ann}/ \text{MaxDD}$ | Rendement par unité de drawdown |


## Architecture de la phase 2

```
PROJET-PYTHON-CRYPTO/
│
├─ src/
│   └─ strategy/
│       ├─ data/
│       │   ├─ loaders.py         # Téléchargement Yahoo/Binance
│       │   └─ transforms.py      # Log returns, splits
│       ├─ models/
│       │   ├─ garch.py           # Forecaster GARCH
│       │   └─ selection.py       # Sélection via QLIKE
│       ├─ strategies/
│       │   └─ inverse_vol.py     # Stratégie Inverse-Vol
│       ├─ evaluation/
│       │   ├─ losses.py          # QLIKE, MSE, MAE
│       │   ├─ metrics.py         # Sharpe, MaxDD, Calmar
│       │   └─ backtest.py        # Backtester
│       └─ pipelines/
│           └─ strategy_pipeline.py  # Orchestrateur
│
├─ notebooks/
│   └─ strategy-analysis.ipynb    # Analyses complémentaires
│
├─ data/
│   └─ processed/
│       └─ strategy/              # Outputs (6 cryptos)
│
├─ out/
│   └─ logs/
│       └─ strategy.log
│
├─ main_strategy.py
└─ README.md
```

---

## Pipeline de traitement

```
Yahoo Finance → Prices (6 cryptos) → Log returns
                  ↓
         Model Selection (QLIKE)
                  ↓
    GARCH(1,1)-t Volatility Forecasts
                  ↓
         Inverse-Vol Weights
                  ↓
    Backtest (avec coûts) → Métriques de performance
```

---

## Outputs générés

Tous les fichiers sont exportés dans `data/processed/strategy/` :

| Fichier | Description |
|---------|-------------|
| model_selection.csv | Scores QLIKE des 4 modèles candidats |
| metrics_val.csv | Métriques de performance sur validation |
| metrics_test.csv | Métriques de performance sur test (honnête) |
| weights_val.csv | Poids du portefeuille (validation) |
| weights_test.csv | Poids du portefeuille (test) |

**Note** : Les fichiers `metrics_test.csv` contiennent l'évaluation finale de la stratégie (modèle sélectionné uniquement sur validation).

---

## Exécution

### Installation

```bash
cd projet-python-crypto

# Installer dépendances supplémentaires (si nécessaire)
pip install arch
```

### Lancer la stratégie

```bash
python main_strategy.py

# Durée estimée : 0,112 secondes
```

**Étapes exécutées** :
1. Chargement des données (6 cryptos)
2. Sélection du modèle GARCH (4 modèles testés)
3. Prévisions de volatilité (validation + test)
4. Calcul des poids (inverse-vol avec cap 35%)
5. Backtest avec coûts (10 bps)
6. Génération des métriques de performance

Les résultats sont automatiquement sauvegardés :
- Tous les outputs → `data/processed/strategy/`
- - Logs détaillés (sélection du modèle, rebalancements, métriques) → `out/logs/strategy.log`

---

## Résultats obtenus*

### Validation (2022-2023) - Bear Market

| Strategy | Total Return | Ann Return | Ann Vol | Sharpe | MaxDD | Obs |
|----------|--------------|------------|---------|--------|-------|-----|
| InverseVol_GARCH | -58.98% | -35.95% | 66.01% | -0.545 | -80.39% | 730 |
| EqualWeight | -61.80% | -38.20% | 69.36% | -0.551 | -83.44% | 730 |
| BTC_Only | -32.74% | -17.99% | 54.96% | -0.327 | -73.07% | 730 |

Nous pouvons constater que la stratégie bitcoin only s'en sort mieux que les deux autres approches. 
L'Inverse Volatility et l'equal weight ne parviennent ni à améliorer le rendement ni à réduire le risque : la volatilité annuelle et le drawdown maximal sont en réalité plus élevés que pour BTC Only. Dans un marché baissier, la diversification des cryptoactifs ne permet ni d’atténuer le risque ni de protéger le capital de manière significative.

### Test (2024-2025) - Bull Market

| Strategy | Total Return | Ann Return | Ann Vol | Sharpe | MaxDD | Obs |
|----------|--------------|------------|---------|--------|-------|-----|
| InverseVol_GARCH | -1.67% | -0.84% | 64.57% | -0.013 | -57.58% | 730 |
| EqualWeight | -11.77% | -6.07% | 69.06% | -0.088 | -62.36% | 730 |
| BTC_Only | +66.50% | +29.03% | 47.90% | +0.606 | -33.12% | 730 |

En marché haussier, la surperformance du Bitcoin only est encore plus marquée. Le Bitcoin affiche un total return de 66,50% alors que les deux autres stratégies ont toujours une performance négative. Une fois encore, la volatilité et le drawdown maximal des stratégies diversifiées sont supérieurs à ceux de BTC Only, ce qui montre que la diversification ne réduit pas réellement le risque et conduit à un rendement moindre. Même en contexte de marché haussier, la stratégie simple BTC Buy-and-Hold demeure la plus efficace.

*Les valeurs présentées sont indicatives et dépendent de la période exacte, des paramètres de rebalancement et des coûts de transaction retenus.*

## Phase 3 : phase expérimentale expérimentale complémentaire, tests de différentes stratégies

Afin d'optimiser le couple rendement/risque, nous avons exploré plusieurs approches de gestion de portefeuille, allant de l'allocation classique au Machine Learning.

*Stratégies Testées* :

- Stratégies de base (Baselines) :

  Equal Weight : Allocation uniforme.

  Inverse Volatility (Inv-Vol) : Pondération inverse au risque.

  Low Volatility (Low-Vol) : Sélection des actifs les moins risqués.

- Stratégies avec filtres systématiques :

  Low-Vol + Trend Filter : Sélection Low-Vol couplée à une moyenne mobile sur le Bitcoin.

  Inverse Volatility + Volatility Targeting : Ajustement de l'exposition globale au risque.

  Inverse Volatility + Trend Filter : Superposition d'une tendance macro.

  Inverse Volatility + Stress Filter : Coupure des positions basée sur des indicateurs statistiques de stress.

-  Approches avec Machine Learning (ML) :

   Low-Vol + Régression Logistique : Classification Risk-On / Risk-Off.

   Inverse Volatility + ML Risk Gate : Filtre d'entrée/sortie par apprentissage supervisé.

   Inverse Volatility + ML Gate avec hystérésis : Réduction du turnover (frais) via une zone tampon.

   Inverse Volatility + XGBoost Meta-Model : Prédiction de la performance relative.

Ces différents tests et résultats sont présents dans : `notebook/Machine Learning/`

**Stratégie Retenue : Low Volatility + Trend Filter**

Après analyse, la stratégie Low Volatility s'est avérée la plus efficace pour réduire la variance du portefeuille. Nous l'avons combinée à un filtre de tendance sur le Bitcoin pour protéger le capital lors des marchés baissiers (Bear Markets) prolongés.

Cette approche offre le meilleur compromis entre performance ajustée du risque (Sharpe) et simplicité d'implémentation.

*Principes de fonctionnement* :

- Univers d'investissement : BTC, ETH, DOGE, SOL, XRP, ADA (les actifs selectionnés dans la phase 1).
- Sélection par Volatilité : Calcul de la volatilité glissante sur 20 jours. Seuls les actifs dont la volatilité est inférieure à la médiane de l'univers sont conservés.
- Filtre de Tendance (Trend Gate) : 
  Si $Prix_{BTC} < MA200$ : Exposition nulle (100% Cash).Si $Prix_{BTC} > MA200$ : Allocation active.
- Allocation & Frais : Pondération égale (Equal-Weight) sur les actifs sélectionnés, rebalancement quotidien et intégration des frais de transaction (10 bps).

*Prérequis*:

pip install -r requirements.txt
```

Python 3.8+

## Structure du projet
```
.
├── data/
│   └── raw/              # Données OHLCV Binance (CSV)
├── src/
│   └── lowvol_trend/
│       ├── loader.py     # Chargement des données
│       ├── strategy.py   # Logique stratégie
│       ├── backtest.py   # Moteur de backtest
├── main_lowvol_trend.py
└── README.md

### Lancer la stratégie

```bash
python main_lowvol_trend.py
```
## Outputs générés :

Tous les fichiers sont exportés dans `data/processed/lowvol_trend/` :

| Fichier | Description |
|---------|-------------|
| equity.png | Equity curve - Low Vol + BTC Trend |
| stats.csv | Métriques de performance (Annual Return, Sharpe, Max DD, Volatilité)|
| trend_gate.csv | BTC trend gate |
| trend_gate.png | Visualisation des phases d'activation (MA200 BTC) |
| volatility.csv | Historique de la volatilité calculée par actif  |
| weight.csv | Historique des poids du portefeuille |






## Auteur

**Projet académique – M2 Gestion d'actifs**  
Année : 2025–2026



