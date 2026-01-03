"""
Backtester pour stratégies de portefeuille.
"""

import pandas as pd
from .metrics import metrics, compare
import logging

logger = logging.getLogger(__name__)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Turnover quotidien: Σ|w_i(t) - w_i(t-1)|"""
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def backtest(
    returns_pct: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 0.0
) -> pd.Series:
    """
    Backtest d'un portefeuille.
    
    Args:
        returns_pct: Rendements en % (ex: 2.5 pour 2.5%)
        weights: Poids du portefeuille
        cost_bps: Coûts de transaction en bps (ex: 10 pour 0.1%)
    
    Returns:
        Series de rendements quotidiens du portefeuille (fraction)
    """
    # Align returns et weights
    r = returns_pct.loc[weights.index, weights.columns] / 100.0
    
    # Rendement brut
    gross_rets = (weights * r).sum(axis=1)
    
    # Coûts
    if cost_bps > 0:
        to = turnover(weights)
        costs = (cost_bps / 10000.0) * to
        return gross_rets - costs
    
    return gross_rets


def backtest_multiple(
    returns_pct: pd.DataFrame,
    strategies: dict,
    cost_bps: float = 0.0,
    trading_days: float = 365.0
) -> pd.DataFrame:
    """
    Backtest plusieurs stratégies.
    
    Args:
        returns_pct: Rendements en %
        strategies: Dict {nom: weights_df}
        cost_bps: Coûts de transaction
        trading_days: Jours de trading par an
    
    Returns:
        DataFrame de comparaison des performances
    """
    results = {}
    
    for name, weights in strategies.items():
        logger.info(f"Backtesting {name}")
        rets = backtest(returns_pct, weights, cost_bps)
        results[name] = rets
    
    return compare(results, trading_days)


if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    
    # Test
    dates = pd.date_range("2022-01-01", "2023-12-31")
    
    # Rendements
    returns = pd.DataFrame({
        "BTC": np.random.randn(len(dates)) * 3 + 0.1,
        "ETH": np.random.randn(len(dates)) * 4 + 0.15
    }, index=dates)
    
    # Poids
    w1 = pd.DataFrame(0.5, index=dates, columns=["BTC", "ETH"])
    w2 = pd.DataFrame({"BTC": 1.0, "ETH": 0.0}, index=dates)
    
    # Backtest
    strats = {"50/50": w1, "BTC Only": w2}
    results = backtest_multiple(returns, strats, cost_bps=10)
    print(results.round(4))