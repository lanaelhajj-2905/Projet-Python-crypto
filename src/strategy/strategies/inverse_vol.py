"""
Stratégie Inverse Volatility avec rebalancement hebdomadaire.
"""

import numpy as np
import pandas as pd
from src.strategy.data import rebalance_dates
import logging

logger = logging.getLogger(__name__)


def inverse_vol_weights(vols: pd.DataFrame, cap: float = None) -> pd.DataFrame:
    """
    Poids inverse-volatilité: w_i = (1/σ_i) / Σ(1/σ_j)
    
    Args:
        vols: DataFrame de volatilités
        cap: Poids max par actif (ex: 0.35 pour 35%)
    
    Returns:
        DataFrame de poids (somme lignes = 1)
    """
    inv = 1.0 / vols
    weights = inv.div(inv.sum(axis=1), axis=0)
    
    if cap:
        weights = weights.clip(upper=cap)
        weights = weights.div(weights.sum(axis=1), axis=0)
    
    return weights


def apply_rebalancing(weights_daily: pd.DataFrame, freq: str = "W-MON") -> pd.DataFrame:
    """
    Applique rebalancement périodique (poids constants entre refits).
    """
    idx = weights_daily.index
    rebal = rebalance_dates(idx.min(), idx.max(), freq)
    
    weights_out = pd.DataFrame(index=idx, columns=weights_daily.columns, dtype=float)
    
    for i, d0 in enumerate(rebal):
        d1 = rebal[i + 1] if i + 1 < len(rebal) else idx.max() + pd.Timedelta(days=1)
        chunk = idx[(idx >= d0) & (idx < d1)]
        
        if len(chunk) > 0:
            w0 = weights_daily.loc[d0]
            weights_out.loc[chunk] = np.tile(w0.values, (len(chunk), 1))
    
    return weights_out


def compute_weights(
    vols: pd.DataFrame,
    rebal_freq: str = "W-MON",
    cap: float = None
) -> pd.DataFrame:
    """
    Pipeline complet : inverse-vol + rebalancement.
    
    Args:
        vols: Volatilités (ATTENTION: doivent être alignées avec shift(1))
        rebal_freq: Fréquence de rebalancement
        cap: Cap sur poids max
    
    Returns:
        Poids finaux
    """
    logger.info("Computing inverse-vol weights")
    w_daily = inverse_vol_weights(vols, cap)
    
    logger.info(f"Applying {rebal_freq} rebalancing")
    w_final = apply_rebalancing(w_daily, rebal_freq)
    
    # Validation
    max_err = (w_final.sum(axis=1) - 1).abs().max()
    logger.info(f"Weights sum error: {max_err:.2e}")
    
    return w_final


def equal_weight(index: pd.DatetimeIndex, assets: list) -> pd.DataFrame:
    """Benchmark équipondéré 1/N."""
    n = len(assets)
    return pd.DataFrame(1.0 / n, index=index, columns=assets)


def single_asset(index: pd.DatetimeIndex, assets: list, target: str) -> pd.DataFrame:
    """Benchmark 100% sur un actif."""
    w = pd.DataFrame(0.0, index=index, columns=assets)
    w[target] = 1.0
    return w


if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    
    # Test
    dates = pd.date_range("2022-01-01", "2023-12-31")
    vols = pd.DataFrame({
        "BTC": np.random.uniform(2, 4, len(dates)),
        "ETH": np.random.uniform(3, 5, len(dates)),
        "XRP": np.random.uniform(4, 6, len(dates))
    }, index=dates)
    
    weights = compute_weights(vols, cap=0.35)
    print(weights.head())
    print(f"\nMean weights:\n{weights.mean()}")