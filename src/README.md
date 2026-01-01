# 📊 Projet Python – Analyse Quantitative Crypto

## Présentation générale

à modifier !

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

L'univers final est composé de **6 cryptomonnaies**, sélectionnées parmi 18 actifs majeurs analysés, selon une méthodologie quantitative rigoureuse détaillée dans `notebooks/universe-research.md`.

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
- ❌ Corrélations >0.9 : redondance informationnelle (ex: LTC/BCH trop proches de BTC)
- ❌ Corrélations <0.1 : instabilité d'estimation GARCH
- ❌ Actifs illiquides : bruit dans les données (SHIB, NEAR)

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

## Architecture du projet

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
│   └─ logs/
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
git clone https://github.com/votre-username/projet-python-crypto.git
cd projet-python-crypto

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Lancer l'analyse

```bash
# Analyse de l'univers complet (18 cryptos)
python main_universe.py

# Durée estimée : Analyse univers termin�e en 16.43s
```

Les résultats sont automatiquement sauvegardés :
- Tous les outputs → `data/processed/universe/`
- Logs → `out/logs/universe.log`

**Étape suivante :** Analyser `correlation.csv` pour valider la sélection des 6 cryptomonnaies documentée dans `notebooks/universe-research.md`.


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

## Phase 2 : Stratégies de trading 

La prochaine phase utilisera l'univers de **6 cryptomonnaies retenues** pour :

à modifier !

## Auteur

**Projet académique – M2 Gestion d'actifs**  
Année : 2025-2026
---


