# main_ml.py
import pandas as pd
from pathlib import Path
from datetime import datetime
from experiments.ml_strategies import run_portfolio_strategy

def main():
    strategies = [
        "equal_weight",
        "inverse_vol",
        "low_vol",
        "ml_stress",
        "ml_combined",
    ]

    output_dir = Path("data/processed/ML")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}
    all_equity_curves = {}

    for strat in strategies:
        print("\n" + "=" * 90)
        print(f"RUNNING STRATEGY: {strat}")
        print("=" * 90)

        try:
            res = run_portfolio_strategy(
                strategy=strat,
                output_dir=output_dir
            )
            results[strat] = res
            all_equity_curves[strat] = res["equity"]

        except Exception as e:
            print(f"❌ Erreur pour {strat}: {e}")

    # --- Export comparaison globale ---
    if results:
        # 1. Tableau comparatif des stats
        comparison = pd.DataFrame({s: results[s]["stats"] for s in results}).T
        stats_file = output_dir / f"strategy_comparison_{timestamp}.csv"
        comparison.to_csv(stats_file)
        print(f"\n✓ Statistiques sauvegardées: {stats_file}")

        # 2. Equity combinée
        if all_equity_curves:
            equity_df = pd.DataFrame(all_equity_curves)
            equity_file = output_dir / f"equity_curves_{timestamp}.csv"
            equity_df.to_csv(equity_file)
            print(f"✓ Courbes d'equity sauvegardées: {equity_file}")

        # 3. Détails par stratégie
        for strat_name, res in results.items():
            weights_file = output_dir / f"{strat_name}_weights_{timestamp}.csv"
            res["weights"].to_csv(weights_file)
            print(f"✓ Détails {strat_name} sauvegardés: {weights_file}")

        # 4. Résumés stables sans timestamp
        comparison_latest = output_dir / "latest_strategy_comparison.csv"
        comparison.to_csv(comparison_latest)
        if all_equity_curves:
            equity_latest = output_dir / "latest_equity_curves.csv"
            equity_df.to_csv(equity_latest)
        print("\n✓ Export final terminé.")

    else:
        print("\n❌ Aucune stratégie n'a fonctionné.")

if __name__ == "__main__":
    main()

