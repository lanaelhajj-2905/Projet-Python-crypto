# main_ml.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from experiments.data_downloader import DataDownloader
from experiments.volatility import VolatilityCalculator
from experiments.ml_strategies import run_portfolio_strategy

def main():
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
    data_dir = Path("data/raw")
    processed_dir = Path("data/processed/ML")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 1️⃣ Vérification / téléchargement
    missing_files = [s for s in symbols if not (data_dir / f"{s}_1d.csv").exists()]
    if missing_files:
        print(f"Fichiers manquants: {missing_files}, téléchargement...")
        dd = DataDownloader(output_dir=data_dir)
        dd.download_binance_public(missing_files, interval="1d")
    
    strategies = ["equal_weight", "inverse_vol", "low_vol", "ml_stress", "ml_combined"]
    results = {}
    all_equity_curves = {}
    all_vol_features = {}
    
    vc = VolatilityCalculator()
    
    # 2️⃣ Calcul volatilités et retours
    dfs = {}
    for sym in symbols:
        file_path = data_dir / f"{sym}_1d.csv"
        df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
        df = vc.add_returns(df)
        df = vc.add_volatility_features(df)
        dfs[sym] = df.loc["2021-01-01":"2025-12-31"]
        
        # Export de toutes les volatilités
        vol_file = processed_dir / f"{sym}_vol_features.csv"
        df.to_csv(vol_file)
        all_vol_features[sym] = df
        print(f"✓ Volatilités et retours pour {sym} sauvegardés: {vol_file}")
    
    # 3️⃣ Exécution des stratégies
    for strat in strategies:
        print("\n" + "="*90)
        print(f"=== STRATEGY: {strat} ===")
        print("="*90)
        try:
            res = run_portfolio_strategy(
                symbols=symbols,
                data_dir=data_dir,
                strategy=strat,
                cost_bps=10
            )
            results[strat] = res
            all_equity_curves[strat] = res["equity"]
            
            stats = res["stats"]
            print("\nRÉSULTATS")
            print(f"Return annualisé: {stats['ann_return']:.2%}")
            print(f"Vol annualisée:   {stats['ann_vol']:.2%}")
            print(f"Sharpe:           {stats['sharpe']:.3f}")
            print(f"Max Drawdown:     {stats['max_dd']:.2%}")
            print(f"Turnover moyen:   {stats['turnover_mean']:.2%}")
        except Exception as e:
            print(f"❌ Erreur pour {strat}: {e}")
    
    # 4️⃣ Export final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results:
        comparison = pd.DataFrame({s: results[s]["stats"] for s in results}).T
        stats_file = processed_dir / f"strategy_comparison_{timestamp}.csv"
        comparison.to_csv(stats_file)
        print(f"\n✓ Statistiques sauvegardées: {stats_file}")
        
        if all_equity_curves:
            equity_df = pd.DataFrame(all_equity_curves)
            equity_file = processed_dir / f"equity_curves_{timestamp}.csv"
            equity_df.to_csv(equity_file)
            print(f"✓ Courbes d'equity sauvegardées: {equity_file}")
        
        # Détails poids par stratégie
        for strat_name, res in results.items():
            detail_file = processed_dir / f"{strat_name}_weights_{timestamp}.csv"
            res["weights"].to_csv(detail_file)
            print(f"✓ Détails {strat_name} sauvegardés: {detail_file}")
        
        # Fichiers "latest" pour Git
        comparison.to_csv(processed_dir / "latest_strategy_comparison.csv")
        equity_df.to_csv(processed_dir / "latest_equity_curves.csv")
    else:
        print("\n❌ Aucune stratégie n'a fonctionné.")

if __name__ == "__main__":
    main()

