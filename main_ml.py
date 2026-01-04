"""
main_ml.py

Script principal pour lancer le pipeline complet de backtesting
des stratégies de portefeuille crypto basées sur ML.
"""

from pathlib import Path
import pandas as pd
from datetime import datetime
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
    output_dir = "data/processed/ML"
    strategies = [
        "equal_weight",
        "inverse_vol",
        "low_vol",
        "ml_stress",
        "ml_combined"
    ]
    cost_bps = 10

    # Créer le dossier de sortie
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
    all_equity_curves = {}

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
            
            # Sauvegarder la courbe d'equity pour cette stratégie
            if "portfolio" in res and res["portfolio"] is not None:
                all_equity_curves[strat] = res["portfolio"]["equity"]

        except Exception as e:
            print(f"❌ Erreur pour {strat}: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------
    # Résumé comparatif et export CSV
    # -----------------------------
    if results:
        print("\n" + "="*80)
        print("RÉSUMÉ COMPARATIF")
        print("="*80)

        # Créer le tableau comparatif
        comparison = pd.DataFrame({
            name: res["stats"] 
            for name, res in results.items()
        }).T
        comparison = comparison.sort_values("sharpe", ascending=False)
        print("\n", comparison.round(4))

        # Timestamp pour les noms de fichiers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export 1: Tableau comparatif des statistiques
        stats_file = f"{output_dir}/strategy_comparison_{timestamp}.csv"
        comparison.to_csv(stats_file)
        print(f"\n✓ Statistiques sauvegardées: {stats_file}")

        # Export 2: Courbes d'equity combinées
        if all_equity_curves:
            equity_df = pd.DataFrame(all_equity_curves)
            equity_file = f"{output_dir}/equity_curves_{timestamp}.csv"
            equity_df.to_csv(equity_file)
            print(f"✓ Courbes d'equity sauvegardées: {equity_file}")

        # Export 3: Détails par stratégie (optionnel)
        for strat_name, res in results.items():
            if "portfolio" in res and res["portfolio"] is not None:
                detail_file = f"{output_dir}/{strat_name}_details_{timestamp}.csv"
                res["portfolio"].to_csv(detail_file)
                print(f"✓ Détails {strat_name} sauvegardés: {detail_file}")

        # Export 4: Résumé simple sans timestamp (pour Git)
        latest_stats = f"{output_dir}/latest_strategy_comparison.csv"
        comparison.to_csv(latest_stats)
        print(f"\n✓ Derniers résultats (fichier stable): {latest_stats}")

        if all_equity_curves:
            latest_equity = f"{output_dir}/latest_equity_curves.csv"
            equity_df.to_csv(latest_equity)
            print(f"✓ Dernières courbes (fichier stable): {latest_equity}")

    else:
        print("\n❌ Aucune stratégie n'a fonctionné.")

if __name__ == "__main__":
    main()
