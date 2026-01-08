Dans ce notebook, l’idée n’était pas de “trouver la meilleure stratégie possible”, mais plutôt de montrer comment on peut construire, tester et améliorer progressivement des stratégies quantitatives à partir de données de marché, en comprenant à chaque étape ce qui marche, ce qui ne marche pas, et pourquoi.

Je suis parti d’un univers simple de cryptomonnaies (BTC, ETH, BNB, SOL, XRP, DOGE) avec des données journalières. Le premier objectif était simplement de mettre en place un cadre propre : chargement des données, calcul des rendements, alignement des dates, et mise en place d’un backtest cohérent avec des coûts de transaction et sans biais de regard vers le futur.

Stratégies de base : comprendre les ordres de grandeur

La première étape a été de tester des stratégies très simples :

Equal-weight (tout à poids égal),

Inverse-volatility,

Low-volatility.

Ces stratégies servent de références. On voit rapidement que l’equal-weight a souvent de bonnes performances brutes, mais avec une volatilité et des drawdowns très élevés. À l’inverse, les stratégies low-vol ou inverse-vol sont plus stables, mais sacrifient une partie du rendement. Cette étape permet surtout de comprendre le compromis rendement / risque, qui est central en gestion de portefeuille.

Ajouter des filtres simples de risque

Ensuite, j’ai cherché à améliorer le profil de risque sans complexifier inutilement :

filtre de tendance basé sur Bitcoin (exposition réduite quand BTC est sous sa moyenne mobile),

ciblage de volatilité du portefeuille,

mise à l’écart temporaire de certaines positions en période jugée défavorable.

Ces filtres améliorent souvent la volatilité et le drawdown, mais pas forcément la performance finale. C’est un point important : réduire le risque ne garantit pas une meilleure performance, mais peut rendre une stratégie plus “vivable” dans le temps.

Pourquoi aller vers le machine learning

À ce stade, plutôt que d’essayer de prédire directement les rendements (ce qui est très difficile et instable), j’ai choisi d’utiliser le machine learning pour une tâche plus réaliste :
détecter des périodes de stress de marché.

L’idée est simple : quand le marché devient très volatil ou désorganisé, on réduit l’exposition, sans chercher à timer précisément les points hauts ou bas.

J’ai donc construit des labels de “stress” basés sur la volatilité future du portefeuille, puis entraîné des modèles simples (logistic regression, random forest, XGBoost) à partir de features intuitives :

volatilité moyenne,

dispersion entre actifs,

momentum du Bitcoin,

indicateurs de volume et de range.

Les résultats montrent que les modèles captent un signal réel (AUC significativement au-dessus de 0.5), mais imparfait. C’est normal : le but n’est pas de prédire parfaitement, mais d’avoir un filtre probabiliste.

Intégration dans une stratégie réelle

Plutôt que d’utiliser le modèle pour décider “acheter ou vendre”, il est intégré comme un gate de risque :

quand la probabilité de stress est élevée → réduction de l’exposition,

sinon → stratégie classique (inverse-vol ou low-vol).

Cette approche est volontairement prudente. Les résultats montrent que le modèle réduit parfois la performance par rapport à une stratégie simple, mais diminue aussi certains risques. Cela illustre bien une réalité importante : ajouter de la complexité n’améliore pas forcément les résultats, et chaque brique doit être justifiée.

Pourquoi certaines pistes ont été abandonnées

Certaines stratégies plus complexes (multi-modèles, signaux agressifs, switching fréquent) ont été testées mais pas conservées, car :

elles augmentaient fortement le turnover,

elles étaient très sensibles aux paramètres,

ou leurs gains n’étaient pas robustes.

À l’inverse, les stratégies simples (inverse-vol, low-vol + filtres) sont plus stables, plus compréhensibles, et plus crédibles dans un cadre réel.