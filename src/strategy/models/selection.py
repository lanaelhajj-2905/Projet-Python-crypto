"""
Sélection du meilleur modèle GARCH via QLIKE.
"""

import pandas as pd
import logging
from src.strategy.models.garch import GARCHForecaster
from src.strategy.evaluation.losses import qlike_mean
from src.strategy.data.transforms import split_data

logger = logging.getLogger(__name__)


MODELS = [
    {"name": "GARCH(1,1)-Normal", "vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "normal"},
    {"name": "GARCH(1,1)-t", "vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
    {"name": "EGARCH(1,1)-t", "vol": "EGARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
    {"name": "GJR-GARCH(1,1)-t", "vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
]


def evaluate_model(
    returns: pd.Series,
    val_returns: pd.Series,
    model_spec: dict
) -> float:
    """
    Évalue un modèle sur la validation avec QLIKE.
    Walk-forward avec refit hebdomadaire.
    """
    forecaster = GARCHForecaster(model_spec, refit_freq="W-MON")
    
    # Forecast volatilité
    vols = forecaster.forecast_single(
        returns=returns,
        start=val_returns.index.min().strftime("%Y-%m-%d"),
        end=val_returns.index.max().strftime("%Y-%m-%d")
    )
    
    # Variance forecast
    var_forecast = (vols ** 2).dropna()
    
    # Align avec rendements réalisés
    common_idx = var_forecast.index.intersection(val_returns.index)
    r_real = val_returns.loc[common_idx].values
    v_fc = var_forecast.loc[common_idx].values
    
    if len(r_real) == 0:
        return float('inf')
    
    return qlike_mean(r_real, v_fc)


def select_best_model(returns: pd.Series) -> dict:
    """
    Sélectionne le meilleur modèle sur validation.
    
    Args:
        returns: Série complète de rendements
    
    Returns:
        {"best_model": spec, "results": DataFrame}
    """
    logger.info("=== MODEL SELECTION (QLIKE) ===")
    
    # Split
    df = returns.to_frame()
    train, val, test = split_data(df)
    train_val = pd.concat([train, val])
    
    # Évaluation
    results = []
    for spec in MODELS:
        logger.info(f"Testing {spec['name']}")
        
        try:
            score = evaluate_model(
                returns=train_val.iloc[:, 0],
                val_returns=val.iloc[:, 0],
                model_spec=spec
            )
            results.append({"Model": spec["name"], "QLIKE": score, **spec})
            logger.info(f"  QLIKE = {score:.6f}")
        
        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue
    
    df_results = pd.DataFrame(results).sort_values("QLIKE")
    best = df_results.iloc[0].to_dict()
    
    logger.info(f"\n✅ Best model: {best['Model']}")
    
    return {
        "best_model": {k: best[k] for k in ["name", "vol", "p", "o", "q", "dist"]},
        "results": df_results
    }


if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    
    # Test avec données synthétiques
    dates = pd.date_range("2018-01-01", "2025-12-31")
    rets = pd.Series(np.random.randn(len(dates)) * 2, index=dates)
    
    result = select_best_model(rets)
    print("\n", result["results"])