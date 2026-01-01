import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        logger.info(f"Log returns calculés: {len(log_returns)} observations")
        return log_returns

    def calculate_covariance_matrix(self, returns: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
        cov_matrix = returns.cov()
        if annualize:
            cov_matrix *= self.config.ANNUAL_FACTOR
            logger.info("Covariance annualisée")
        return cov_matrix

    @staticmethod
    def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        corr_matrix = returns.corr()
        logger.info("Matrice de corrélation calculée")
        return corr_matrix

    def calculate_summary_statistics(self, returns: pd.DataFrame) -> pd.DataFrame:
        stats = pd.DataFrame({
            'mean_daily': returns.mean(),
            'std_daily': returns.std(),
            'mean_annual': returns.mean() * self.config.ANNUAL_FACTOR,
            'std_annual': returns.std() * np.sqrt(self.config.ANNUAL_FACTOR),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(self.config.ANNUAL_FACTOR),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min(),
            'max': returns.max()
        })
        logger.info("Statistiques descriptives calculées")
        return stats
