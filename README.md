# Projet Python

## Présentation générale

Ce projet vise à développer une **stratégie d’allocation quantitative sur cryptomonnaies**, fondée sur la **prévision de la volatilité conditionnelle** et une logique explicite de **gestion du risque**.

Le projet est volontairement structuré en **deux grandes phases distinctes** :

1. **Construction de l'univers d'investissement**
2. **Stratégies de trading et allocation**

L'approche repose sur une **séparation stricte entre la donnée, l'analyse statistique et la logique de décision**, afin de garantir robustesse, lisibilité et reproductibilité.

> *Projet réalisé dans le cadre du M2 Gestion d'actifs.*

---

## Phase 1 : Univers d'investissement

Cette phase constitue le **socle quantitatif du projet**. Elle est volontairement **indépendante de toute logique de trading ou d'optimisation**, afin d'éviter tout biais méthodologique.

### Objectifs opérationnels

* Définir un **univers crypto liquide et représentatif**
* Télécharger, nettoyer et aligner les données de prix historiques
* Calculer les principales **statistiques de risque et de dépendance**
* Produire des **outputs exploitables** pour des travaux ultérieurs

--- 

## Univers retenu

L'univers final est composé de **6 cryptomonnaies**, sélectionnées parmi 18 actifs majeurs analysés, selon une étude détaillée dans `notebooks/universe-research.md`.

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

- **BTC + ETH** : Duo indispensable capturant les dynamiques fondamentales (réserve de valeur vs. DeFi)
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

### Lancer l'analyse

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

## Univers de trading

L'univers de trading est constitué des **6 cryptomonnaies** sélectionnées lors de la Phase 1 :

```
BTC, ETH, XRP, ADA, SOL, DOGE
```

Ces actifs ont été retenus pour leurs **corrélations modérées** (0.2-0.7), leur **diversité fonctionnelle** et leur **liquidité élevée**, garantissant un équilibre optimal entre diversification et stabilité économétrique.

---

## Méthodologie

### 1. Sélection du modèle GARCH

**Modèles candidats testés** :

| Modèle | Spécification | Distribution |
|--------|---------------|--------------|
| GARCH(1,1)-Normal | p=1, o=0, q=1 | Normale |
| **GARCH(1,1)-t** | p=1, o=0, q=1 | **Student-t** |
| EGARCH(1,1)-t | p=1, o=0, q=1 | Student-t |
| GJR-GARCH(1,1)-t | p=1, o=1, q=1 | Student-t |

**Actif utilisé** :

 La sélection du modèle de volatilité est réalisée uniquement sur le **Bitcoin**. 
 Nous avons pris cette décision car :
• Le BTC est l’actif le plus liquide et le plus ancien du marché crypto 
• Moins de bruit microstructurel est présent
• La série de rendements est longue

Nous avons donc fait l'hypothèse que la volatilité du bitcoin est représentatif du marché des cryptoactifs. 

**Critère de sélection** : QLIKE (Quasi-Likelihood)

$$\text{QLIKE} = \log(\sigma^2) + \frac{r^2}{\sigma^2}$$

**Justification** (Patton, 2011) :
- Métrique asymétrique robuste aux valeurs extrêmes
- Standard en finance pour évaluer les prévisions de volatilité
- Pénalise moins les sous-estimations que les sur-estimations

Interprétation du Q-like : Un QLIKE plus faible indique une meilleure capacité à prévoir la volatilité conditionnelle.

**Split temporel**

Les données sont découpées en trois périodes : 
```
Training     : 2020-04-11 → 2021-12-31  (apprentissage initial)
Validation   : 2022-01-01 → 2023-12-31  (sélection du modèle GARCH)
Test         : 2024-01-01 → 2025-12-31  (évaluation finale)
```

**Note importante** : Le modèle est sélectionné **uniquement sur la période de validation**. L'évaluation sur le test est **honnête** (pas de re-sélection, pas de data snooping).

**Procédure** :
1. Estimation walk-forward sur période de validation (2022-2023)
2. Refit hebdomadaire (W-MON) des paramètres
3. Forecast 1-step-ahead de la variance conditionnelle
4. Calcul du QLIKE moyen sur la validation
5. Sélection du modèle avec le **QLIKE le plus faible**

**Résultat** : GARCH(1,1) avec distribution Student-t est sélectionné. Il s'agit du modèle avec le Q-like le plus faible sur la période de validation.

### 2. Prévisions de volatilité conditionnelle

**Modèle retenu** : GARCH(1,1)-t

$$\sigma^2_{t+1} = \omega + \alpha \cdot r^2_t + \beta \cdot \sigma^2_t$$

**Implémentation** :
- **Refit hebdomadaire** : Les paramètres (ω, α, β) sont ré-estimés chaque lundi
- **Update quotidien** : Entre deux refits, la variance conditionnelle est mise à jour quotidiennement via la récursion GARCH
- **Forecast next-day** : La volatilité prévue σ_{t+1} est stockée à la date t

**Alignement temporel critique** :
```python
# Les forecasts sont stockés comme σ_{t+1} à date t
vols_raw = forecast_volatility(returns, start, end)

# Pour construire les poids à date t basés sur σ_t :
vols_aligned = vols_raw.shift(1)  # Shift de 1 jour
```
Ainsi, pour éviter tout biais d'anticipation : les volatilités sont décalés d'un jour et les poids au jour t utilisent uniquement l’information disponible au jour t−1. 

### 3. Stratégie d'allocation

**Stratégie** : Inverse Volatility (Risk Parity)

$$w_i(t) = \frac{1/\sigma_i(t)}{\sum_{j=1}^{N} 1/\sigma_j(t)}$$

**Paramètres** :
- **Cap** : Poids maximum de 35% par actif (évite concentration excessive)
- **Rebalancement** : Hebdomadaire lundi (W-MON), poids constants entre refits
- **Coûts de transaction** : 10 bps (0.1%) par trade, proportionnels au turnover

**Justification** :
- Stratégie simple et robuste, largement utilisée en gestion d'actifs
- Égalise la contribution au risque de chaque actif
- Pas de prévision de rendements nécessaire (uniquement volatilité)
- Performance empiriquement supérieure à l'allocation équipondérée sur marchés volatils


## Benchmarks

Deux benchmarks sont utilisés pour évaluer la performance relative de la stratégie :

| Benchmark | Description | Justification |
|-----------|-------------|---------------|
| **Equal-Weight (1/N)** | Allocation équipondérée (16.67% par actif) | Stratégie naïve robuste, fréquemment utilisée comme référence académique (DeMiguel et al., 2009) |
| **BTC Buy-and-Hold** | 100% Bitcoin | Référence du marché crypto |

---

## Métriques d'évaluation

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

**Note** : Les fichiers `metrics_test.csv` contiennent l'évaluation finale honnête de la stratégie (modèle sélectionné uniquement sur validation).

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

## Résultats obtenus

### Validation (2022-2023) - Bear Market

| Strategy | Total Return | Ann Return | Ann Vol | Sharpe | MaxDD | Obs |
|----------|--------------|------------|---------|--------|-------|-----|
| InverseVol_GARCH | -58.98% | -35.95% | 66.01% | -0.545 | -80.39% | 730 |
| EqualWeight | -61.80% | -38.20% | 69.36% | -0.551 | -83.44% | 730 |
| BTC_Only | -32.74% | -17.99% | 54.96% | -0.327 | -73.07% | 730 |

**Interprétation** :
- **BTC domine** : Sharpe supérieur (-0.327 vs. -0.545) et drawdown plus faible
- **Diversification altcoins inefficace** en bear market (corrélations élevées)
- Inverse-Vol et Equal-Weight : performances similaires, faible valeur ajoutée

### Test (2024-2025) - Bull Market

| Strategy | Total Return | Ann Return | Ann Vol | Sharpe | MaxDD | Obs |
|----------|--------------|------------|---------|--------|-------|-----|
| InverseVol_GARCH | -1.67% | -0.84% | 64.57% | -0.013 | -57.58% | 730 |
| EqualWeight | -11.77% | -6.07% | 69.06% | -0.088 | -62.36% | 730 |
| BTC_Only | +66.50% | +29.03% | 47.90% | +0.606 | -33.12% | 730 |

**Interprétation** :
- **BTC continue de dominer** : +29% annualisé, Sharpe 0.606
- **Stratégies diversifiées sous-performent** : Altcoins traînent vs. BTC
- Inverse-Vol limite les pertes (-0.84%) vs. Equal-Weight (-6.07%)
- La diversification **réduit le risque** mais **réduit aussi le rendement** en bull BTC

**Conclusion** :
Sur les deux périodes (bear + bull), la stratégie simple **BTC Buy-and-Hold surperforme** les stratégies diversifiées. L'allocation Inverse-Vol apporte une **gestion du risque modérée** (MaxDD réduit vs. Equal-Weight) mais sacrifie le rendement. En contexte crypto dominé par BTC, la diversification altcoins **réduit la performance** plus qu'elle ne réduit le risque.

*Les valeurs présentées sont indicatives et dépendent de la période exacte, des paramètres de rebalancement et des coûts de transaction retenus.*


## Auteur

**Projet académique – M2 Gestion d'actifs**  
Année : 2025–2026



