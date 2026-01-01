import logging
from datetime import datetime
from pathlib import Path

from src.universe.fetcher import CryptoDataFetcher, InsufficientDataError
from src.universe.analyzer import FinancialAnalyzer
from src.universe.exporter import DataExporter

# ===================== CONFIGURATION =====================

class Config:
    EXCHANGE_NAME = "binance"
    TIMEFRAME = "1d"
    RATE_LIMIT_SLEEP = 0.5
    ANNUAL_FACTOR = 365

    ASSETS = [
        "BTC", "ETH", "XRP", "SOL", "TRX", "ADA", "DOGE", "AVAX", "DOT",
        "LTC", "SHIB", "ICP", "LINK", "BCH", "NEAR", "UNI", "ATOM", "ETC"
    ]

    QUOTE_PRIORITY = ["USDT", "BUSD", "USD"]
    MIN_OBSERVATIONS = 100
    OHLCV_LIMIT = 1000

    OUTPUT_DIR = Path("data/processed/universe")


# ===================== LOGGING =====================

LOG_DIR = Path("out/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "universe.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ===================== PIPELINE =====================

def run_universe_analysis():
    start = datetime.now()
    logger.info("=" * 60)
    logger.info("DÉBUT ANALYSE UNIVERS CRYPTO")
    logger.info("=" * 60)

    try:
        fetcher = CryptoDataFetcher(Config)
        prices, coverage = fetcher.fetch_all_assets()

        prices = prices.dropna(how="any")
        logger.info(
            f"Panel équilibré : {prices.shape[0]} jours | {prices.shape[1]} actifs"
        )

        analyzer = FinancialAnalyzer(Config)
        returns = analyzer.calculate_log_returns(prices)

        exporter = DataExporter(Config.OUTPUT_DIR)
        exporter.export_all_results(
            prices=prices,
            returns=returns,
            cov_daily=analyzer.calculate_covariance_matrix(returns),
            cov_annual=analyzer.calculate_covariance_matrix(returns, annualize=True),
            corr=analyzer.calculate_correlation_matrix(returns),
            coverage=coverage,
            stats=analyzer.calculate_summary_statistics(returns)
        )

        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Analyse univers terminée en {duration:.2f}s")

    except InsufficientDataError as e:
        logger.error(f"Données insuffisantes : {e}")
        raise
    except Exception as e:
        logger.error(f"Erreur fatale : {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_universe_analysis()
