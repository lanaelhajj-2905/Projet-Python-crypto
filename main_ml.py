# main_ml.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from experiments.data_downloader import DataDownloader
from experiments.ml_strategies import run_portfolio_strategy


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]

    raw_dir = Path("data/raw")
    out_dir = Path("data/processed/ML")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==========================================================
    # 1. DOWNLOAD DATA (if missing)
    # ==========================================================
    downloader = DataDownloader(output_dir=raw_dir)
    missing = [s for s in symbols if not (raw_dir / f"{s}_1d.csv").exists()]

    if missing:
        print(f"Téléchargement données manquantes: {missing}")
        downloader.download_binance_public(missing)
    else:
        print("Données déjà présentes")

    # ==========================================================
    # 2. RUN STRATEGIES
    # ==========================================================
    strategies = [
        "equal_weight",
        "inverse_vol",
        "low_vol",
        "ml_stress",
        "ml_combined"
    ]

    results = {}
    equity_curves = {}

    for strat in strategies:
        print("\n" + "=" * 90)
        print(f"RUNNING STRATEGY: {strat}")
        print("=" * 90)

        try:
            res = run_portfolio_strategy(
                symbols=symbols,
                data_dir=raw_dir,
                strategy=strat,
            )
            results[strat] = res
            equity_curves[strat] = res["equity"]

        except Exception as e:
            print(f"Erreur pour {strat}: {e}")

    if not results:
        print("Aucune stratégie n'a fonctionné.")
        return

    # ==========================================================
    # 3. EXPORT CSV
    # ==========================================================
    comparison = pd.DataFrame({k: v["stats"] for k, v in results.items()}).T

    comparison.to_csv(out_dir / f"strategy_comparison_{timestamp}.csv")
    comparison.to_csv(out_dir / "latest_strategy_comparison.csv")

    equity_df = pd.DataFrame(equity_curves)
    equity_df.to_csv(out_dir / f"equity_curves_{timestamp}.csv")
    equity_df.to_csv(out_dir / "latest_equity_curves.csv")

    for strat, res in results.items():
        res["weights"].to_csv(out_dir / f"{strat}_weights_{timestamp}.csv")

    print("CSV exportés")

    # ==========================================================
    # 4. IMAGES
    # ==========================================================

    # ---- Equity curves
    plt.figure(figsize=(12, 6))
    for strat, eq in equity_curves.items():
        plt.plot(eq.index, eq, label=strat)
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curves.png")
    plt.close()

    # ---- Stats heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        comparison[["ann_return", "ann_vol", "sharpe", "max_dd"]],
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )
    plt.title("Strategy Statistics Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "stats_heatmap.png")
    plt.close()

    # ---- Weights heatmaps
    for strat, res in results.items():
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            res["weights"].T,
            cmap="viridis",
            cbar_kws={"label": "Weight"}
        )
        plt.title(f"Portfolio Weights – {strat}")
        plt.xlabel("Date")
        plt.ylabel("Asset")
        plt.tight_layout()
        plt.savefig(out_dir / f"weights_{strat}.png")
        plt.close()

    print("IMAGES générées")
    print(f"Résultats dans: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
