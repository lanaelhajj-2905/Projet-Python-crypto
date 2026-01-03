"""
Pipeline complet : sélection GARCH → forecast vol → poids → backtest.
"""

import pandas as pd
import logging

# Imports absolus pour éviter les ImportError
from src.strategy.data.loaders import load_multiple
from src.strategy.data.transforms import log_returns_df, split_data
from src.strategy.models.selection import select_best_model
from src.strategy.models.garch import GARCHForecaster
from src.strategy.strategies.inverse_vol import compute_weights, equal_weight, single_asset
from src.strategy.evaluation.backtest import backtest_multiple

logger = logging.getLogger(__name__)


class StrategyPipeline:
    """Pipeline orchestrateur."""

    def __init__(self, config: dict):
        """
        Args:
            config: Dict avec paramètres (tickers, dates, rebal_freq, cap, costs)
        """
        self.config = config
        self.results = {}

    def run(self):
        """Exécute le pipeline complet."""
        logger.info("=" * 60)
        logger.info("STARTING STRATEGY PIPELINE")
        logger.info("=" * 60)

        # 1. Data
        logger.info("\n[1/5] Loading data")
        prices = load_multiple(
            self.config["tickers"],
            self.config["start"],
            self.config["end"]
        )
        prices = prices.dropna(how="any")  # Panel équilibré

        returns = log_returns_df(prices)
        returns = returns.loc[self.config["train_start"]:]

        # 2. Model selection (sur BTC uniquement pour simplicité)
        logger.info("\n[2/5] Model selection")
        selection = select_best_model(returns["BTC"])
        best_model = selection["best_model"]
        logger.info(f"Selected: {best_model['name']}")

        # 3. Volatility forecasts
        logger.info("\n[3/5] Volatility forecasts")
        forecaster = GARCHForecaster(best_model, self.config["rebal_freq"])

        vols_val = forecaster.forecast_portfolio(
            returns, self.config["val_start"], self.config["val_end"]
        )
        vols_test = forecaster.forecast_portfolio(
            returns, self.config["test_start"], self.config["test_end"]
        )

        # CRITICAL: Align vols (shift by 1)
        vols_val_aligned = vols_val.shift(1)
        vols_test_aligned = vols_test.shift(1)

        # 4. Weights
        logger.info("\n[4/5] Computing weights")
        w_val = compute_weights(vols_val_aligned, self.config["rebal_freq"], self.config["cap"])
        w_test = compute_weights(vols_test_aligned, self.config["rebal_freq"], self.config["cap"])

        # Benchmarks
        assets = list(self.config["tickers"].keys())
        ew_val = equal_weight(w_val.index, assets)
        ew_test = equal_weight(w_test.index, assets)
        btc_val = single_asset(w_val.index, assets, "BTC")
        btc_test = single_asset(w_test.index, assets, "BTC")

        # 5. Backtest
        logger.info("\n[5/5] Backtesting")

        val_returns = returns.loc[self.config["val_start"]:self.config["val_end"]]
        test_returns = returns.loc[self.config["test_start"]:self.config["test_end"]]

        strats_val = {
            "InverseVol_GARCH": w_val,
            "EqualWeight": ew_val,
            "BTC_Only": btc_val
        }
        strats_test = {
            "InverseVol_GARCH": w_test,
            "EqualWeight": ew_test,
            "BTC_Only": btc_test
        }

        metrics_val = backtest_multiple(val_returns, strats_val, self.config["cost_bps"])
        metrics_test = backtest_multiple(test_returns, strats_test, self.config["cost_bps"])

        # Store results
        self.results = {
            "selection": selection,
            "vols_val": vols_val,
            "vols_test": vols_test,
            "weights_val": w_val,
            "weights_test": w_test,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test
        }

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 60)

        return self.results

    def print_summary(self):
        """Affiche un résumé des résultats."""
        print("\n" + "=" * 60)
        print("VALIDATION METRICS")
        print("=" * 60)
        print(self.results["metrics_val"].round(4).to_string(index=False))

        print("\n" + "=" * 60)
        print("TEST METRICS (HONEST)")
        print("=" * 60)
        print(self.results["metrics_test"].round(4).to_string(index=False))


# Mini test rapide pour vérifier que la classe est visible
if __name__ == "__main__":
    print("StrategyPipeline est défini correctement !")

