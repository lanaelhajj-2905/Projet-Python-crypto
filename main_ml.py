# main_ml.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from experiments.data_downloader import DataDownloader
from experiments.volatility import VolatilityCalculator
from experiments.ml_strategies import run_portfolio_strategy

# Dossier pour les exports
output_dir = Path("data/processed/ML")
output_dir.mkdir(parents=True, exist_ok=True)

# Liste des symboles
symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT","ADAUSDT"]

# Période
start = "2021-01-01"
end   = "2025-12-31"

# Téléchargement des données si nécessaire
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

downloader = DataDownloader(output_dir=data_dir)
for sym in symbols:
    file_path = data_dir / f"{sym}_1d.csv"
    if not file_path.exists():
        print(f"Téléchargement pour {sym}...")
        downloader.download_binance_public([sym])

# Calculer les retours log et volatilité
dfs = {}
vc = VolatilityCalculator()

for sym in symbols:
    file_path = data_dir / f"{sym}_1d.csv"
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    
    # Assurer que toutes les colonnes sont présentes
    for col in ["open","high","low","close","volume"]:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante pour {sym}: {col}")
    
    # Ajouter retours log et volatilité
    df = vc.add_returns(df)
    df = vc.add_volatility_features(df)
    
    # Filtrer période
    df = df.loc[start:end]
    dfs[sym] = df

# Dictionnaire pour tous les résultats
results = {}
all_equity_curves = {}

# Stratégies à tester
strategies = ["equal_weight", "inverse_vol", "low_vol", "ml_stress", "ml_combined"]

for strat in strategies:
    print("\n" + "="*80)
    print(f"=== STRATEGY: {strat} ===")
    print("="*80)
    try:
        res = run_portfolio_strategy(
            symbols=symbols,
            data_dir=data_dir,
            start=start,
            end=end,
            strategy=strat,
            cost_bps=10
        )
        results[strat] = res
        all_equity_curves[strat] = res["equity"]
    except Exception as e:
        print(f"❌ Erreur pour {strat}: {e}")

# --- EXPORT ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if results:
    # Export 1: Tableau comparatif
    comparison = pd.DataFrame({s: results[s]["stats"] for s in results if results[s] is not None}).T
    stats_file = output_dir / f"strategy_comparison_{timestamp}.csv"
    comparison.to_csv(stats_file)
    print(f"\n✓ Statistiques sauvegardées: {stats_file}")

    # Export 2: Courbes d'equity combinées
    if all_equity_curves:
        equity_df = pd.DataFrame(all_equity_curves)
        equity_file = output_dir / f"equity_curves_{timestamp}.csv"
        equity_df.to_csv(equity_file)
        print(f"✓ Courbes d'equity sauvegardées: {equity_file}")

    # Export 3: Détails par stratégie
    for strat_name, res in results.items():
        if "weights" in res and res["weights"] is not None:
            detail_file = output_dir / f"{strat_name}_details_{timestamp}.csv"
            res["weights"].to_csv(detail_file)
            print(f"✓ Détails {strat_name} sauvegardés: {detail_file}")

    # Export 4: Fichiers "latest" stables
    latest_stats = output_dir / "latest_strategy_comparison.csv"
    comparison.to_csv(latest_stats)
    print(f"\n✓ Derniers résultats (fichier stable): {latest_stats}")

    if all_equity_curves:
        latest_equity = output_dir / "latest_equity_curves.csv"
        equity_df.to_csv(latest_equity)
        print(f"✓ Dernières courbes (fichier stable): {latest_equity}")

else:
    print("\n❌ Aucune stratégie n'a fonctionné.")

if __name__ == "__main__":
    pass
