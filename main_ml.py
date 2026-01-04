# main_ml.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from experiments.ml_strategies import run_portfolio_strategy
from experiments.volatility import VolatilityCalculator

def main():
    # --- Paramètres ---
    symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
    data_dir = Path("data/raw")
    output_dir = Path("data/processed/ML")
    output_dir.mkdir(parents=True, exist_ok=True)
    cost_bps = 10
    start = "2021-01-01"
    end   = "2025-12-31"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {}
    all_equity_curves = {}

    for strat in ["equal_weight", "inverse_vol", "low_vol", "ml_stress", "ml_combined"]:
        print("\n" + "="*90)
        print(f"=== STRATEGY: {strat} ===")
        print("="*90)

        try:
            res = run_portfolio_strategy(
                symbols=symbols,
                data_dir=data_dir,
                start=start,
                end=end,
                strategy=strat,
                cost_bps=cost_bps
            )
            results[strat] = res
            all_equity_curves[strat] = res["equity"]

            # Affichage rapide
            stats = res["stats"]
            print("\nRÉSULTATS")
            print(f"Return annualisé: {stats['ann_return']:.2%}")
            print(f"Vol annualisée:   {stats['ann_vol']:.2%}")
            print(f"Sharpe:           {stats['sharpe']:.3f}")
            print(f"Max Drawdown:     {stats['max_dd']:.2%}")
            print(f"Turnover moyen:   {stats['turnover_mean']:.2%}")

        except Exception as e:
            print(f"❌ Erreur pour {strat}: {e}")

    # --- Export ---
    if results:
        # 1. Tableau comparatif des stats
        comparison = pd.DataFrame({s: results[s]["stats"] for s in results}).T
        stats_file = output_dir / f"strategy_comparison_{timestamp}.csv"
        comparison.to_csv(stats_file)
        print(f"\n✓ Statistiques sauvegardées: {stats_file}")

        # 2. Courbes d’equity combinées
        if all_equity_curves:
            equity_df = pd.DataFrame(all_equity_curves)
            equity_file = output_dir / f"equity_curves_{timestamp}.csv"
            equity_df.to_csv(equity_file)
            print(f"✓ Courbes d'equity sauvegardées: {equity_file}")

        # 3. Détails par stratégie
        for strat_name, res in results.items():
            detail_file = output_dir / f"{strat_name}_details_{timestamp}.csv"
            res["weights"].to_csv(detail_file)
            print(f"✓ Détails {strat_name} sauvegardés: {detail_file}")

        # 4. Résumé stable (sans timestamp)
        comparison.to_csv(output_dir / "latest_strategy_comparison.csv")
        if all_equity_curves:
            equity_df.to_csv(output_dir / "latest_equity_curves.csv")
        print("\n✓ Export final terminé.")

    else:
        print("\n❌ Aucune stratégie n'a fonctionné.")


if __name__ == "__main__":
    main()
