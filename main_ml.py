"""
main_ml.py

Script principal pour lancer le pipeline complet de backtesting
des stratégies de portefeuille crypto basées sur ML.
"""

from pathlib import Path
import pandas as pd
from experiments.data_downloader import DataDownloader
from experiments.ml_models import StressPredictor, DirectionalPredictor
from experiments.ml_strategies import PortfolioStrategy, run_portfolio_strategy
from experiments.datasets import build_stress_dataset, build_direction_dataset

def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    data_dir = "binance_public_data"
    strategies = [
        "equal_weight",
        "inverse_vol",
        "low_vol",
        "ml_stress",
        "ml_combined"
    ]
    cost_bps = 10

    # -----------------------------
    # Vérification / téléchargement des données
    # -----------------------------
    Path(data_dir).mkdir(exist_ok=True)
    data_missing = any(not Path(f"{data_dir}/{sym}_1d_2021_2025.csv").exists() for sym in symbols)

    if data_missing:
        print("\n⚠️ Données manquantes, téléchargement depuis Binance...")
        downloader = DataDownloader(output_dir=data_dir)
        downloader.download_binance_public(symbols=symbols, interval="1d", years=[2021,2022,2023,2024,2025])
        print("✓ Téléchargement terminé !\n")

    # -----------------------------
    # Lancement des stratégies
    # -----------------------------
    results = {}

    for strat in strategies:
        print("\n" + "="*80)
        print(f"RUNNING STRATEGY: {strat}")
        print("="*80)

        try:
            res = run_portfolio_strategy(
                symbols=symbols,
                data_dir=data_dir,
                strategy=strat,
                cost_bps=cost_bps
            )
            results[strat] = res

        except Exception as e:
            print(f"❌ Erreur pour {strat}: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------
    # Résumé comparatif
    # -----------------------------
    if results:
        print("\n" + "="*80)
        print("RÉSUMÉ COMPARATIF")
        print("="*80)

        comparison = pd.DataFrame({
            name: res["stats"] 
            for name, res in results.items()
        }).T
        comparison = comparison.sort_values("sharpe", ascending=False)
        print("\n", comparison.round(4))
    else:
        print("\n❌ Aucune stratégie n'a fonctionné.")

if __name__ == "__main__":
    main()
