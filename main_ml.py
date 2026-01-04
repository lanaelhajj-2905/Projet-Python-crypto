from pathlib import Path
import pandas as pd
from experiments.data_downloader import DataDownloader
from experiments.ml_strategies import run_portfolio_strategy

def main():
    data_dir = Path("data")
    processed_dir = data_dir / "processed" / "ML"
    processed_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT","ADAUSDT"]

    # 1️⃣ Vérifier si les fichiers existent, sinon télécharger
    files_exist = all((data_dir / f"{s}_1d.csv").exists() for s in symbols)
    if not files_exist:
        print("📥 Téléchargement des données Binance...")
        dl = DataDownloader(output_dir=data_dir)
        dl.download_binance_public(symbols)

    # 2️⃣ Définir les stratégies
    strategies = ["equal_weight", "inverse_vol", "low_vol", "ml_stress", "ml_combined"]
    results = {}
    all_equity_curves = {}

    # 3️⃣ Lancer chaque stratégie
    for strat in strategies:
        print("\n" + "="*90)
        print(f"RUNNING STRATEGY: {strat}")
        print("="*90)

        try:
            res = run_portfolio_strategy(
                symbols=symbols,
                data_dir=str(data_dir),
                strategy=strat,
                cost_bps=10
            )
            results[strat] = res
            all_equity_curves[strat] = res["equity"]

        except Exception as e:
            print(f"❌ Erreur pour {strat}: {e}")

    # 4️⃣ Exporter les résultats
    if results:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Export 1: Tableau comparatif des statistiques
        comparison = pd.DataFrame({s: results[s]["stats"] for s in results}).T
        stats_file = processed_dir / f"strategy_comparison_{timestamp}.csv"
        comparison.to_csv(stats_file)
        print(f"\n✓ Statistiques sauvegardées: {stats_file}")

        # Export 2: Courbes d'equity combinées
        if all_equity_curves:
            equity_df = pd.DataFrame(all_equity_curves)
            equity_file = processed_dir / f"equity_curves_{timestamp}.csv"
            equity_df.to_csv(equity_file)
            print(f"✓ Courbes d'equity sauvegardées: {equity_file}")

        # Export 3: Détails par stratégie (poids / portfolio)
        for strat_name, res in results.items():
            if "weights" in res:
                detail_file = processed_dir / f"{strat_name}_details_{timestamp}.csv"
                res["weights"].to_csv(detail_file)
                print(f"✓ Détails {strat_name} sauvegardés: {detail_file}")

        # Export 4: Fichiers "latest" stables
        comparison.to_csv(processed_dir / "latest_strategy_comparison.csv")
        if all_equity_curves:
            equity_df.to_csv(processed_dir / "latest_equity_curves.csv")
        print("\n✓ Export final terminé.")

    else:
        print("\n❌ Aucune stratégie n'a fonctionné.")

if __name__ == "__main__":
    main()
