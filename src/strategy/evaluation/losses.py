"""
Fonctions de loss pour l'évaluation de prévisions de volatilité.

Ce module implémente différentes métriques de loss pour comparer
les prévisions de variance/volatilité avec les rendements réalisés :
    - QLIKE (Quasi-Likelihood) : métrique standard en finance
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
"""

import numpy as np
import pandas as pd
from typing import Union, Callable
import logging

logger = logging.getLogger(__name__)


# ============================================================
# QLIKE (QUASI-LIKELIHOOD)
# ============================================================

def qlike(
    r_realized: Union[float, np.ndarray],
    var_forecast: Union[float, np.ndarray],
    epsilon: float = 1e-12
) -> Union[float, np.ndarray]:
    """
    Calcule le score QLIKE (Quasi-Likelihood).
    
    QLIKE = log(σ²) + r² / σ²
    
    Args:
        r_realized: Rendement(s) réalisé(s) (%)
        var_forecast: Variance(s) prévue(s)
        epsilon: Seuil minimum pour éviter division par zéro
    
    Returns:
        Score(s) QLIKE (lower is better)
    
    Note:
        - Métrique asymétrique favorisant les sous-estimations
        - Robuste aux valeurs extrêmes
        - Standard en finance pour évaluer prévisions de volatilité
    
    References:
        Patton, A. (2011). Volatility forecast comparison using 
        imperfect volatility proxies. Journal of Econometrics.
    """
    var = np.maximum(var_forecast, epsilon)
    r_sq = r_realized ** 2
    
    return np.log(var) + r_sq / var


def qlike_mean(
    r_realized: np.ndarray,
    var_forecast: np.ndarray,
    epsilon: float = 1e-12
) -> float:
    """
    QLIKE moyen sur une série de prévisions.
    
    Args:
        r_realized: Array de rendements réalisés
        var_forecast: Array de variances prévues
        epsilon: Seuil minimum
    
    Returns:
        QLIKE moyen (scalar)
    """
    scores = qlike(r_realized, var_forecast, epsilon)
    return float(np.mean(scores))


# ============================================================
# MSE (MEAN SQUARED ERROR)
# ============================================================

def mse_variance(
    r_realized: np.ndarray,
    var_forecast: np.ndarray
) -> float:
    """
    MSE entre variance prévue et rendement² réalisé.
    
    MSE = mean((r² - σ²)²)
    
    Args:
        r_realized: Rendements réalisés
        var_forecast: Variances prévues
    
    Returns:
        MSE (lower is better)
    
    Note:
        Sensible aux outliers (penalise fortement grandes erreurs).
    """
    r_sq = r_realized ** 2
    return float(np.mean((r_sq - var_forecast) ** 2))


def rmse_variance(
    r_realized: np.ndarray,
    var_forecast: np.ndarray
) -> float:
    """
    RMSE (Root MSE) entre variance prévue et rendement² réalisé.
    
    Args:
        r_realized: Rendements réalisés
        var_forecast: Variances prévues
    
    Returns:
        RMSE (même unité que variance)
    """
    return float(np.sqrt(mse_variance(r_realized, var_forecast)))


# ============================================================
# MAE (MEAN ABSOLUTE ERROR)
# ============================================================

def mae_variance(
    r_realized: np.ndarray,
    var_forecast: np.ndarray
) -> float:
    """
    MAE entre variance prévue et rendement² réalisé.
    
    MAE = mean(|r² - σ²|)
    
    Args:
        r_realized: Rendements réalisés
        var_forecast: Variances prévues
    
    Returns:
        MAE (lower is better)
    
    Note:
        Moins sensible aux outliers que MSE.
    """
    r_sq = r_realized ** 2
    return float(np.mean(np.abs(r_sq - var_forecast)))


# ============================================================
# UTILITAIRES
# ============================================================

def compute_all_losses(
    r_realized: np.ndarray,
    var_forecast: np.ndarray
) -> dict:
    """
    Calcule toutes les métriques de loss.
    
    Args:
        r_realized: Rendements réalisés
        var_forecast: Variances prévues
    
    Returns:
        Dict avec toutes les losses
    """
    losses = {
        "QLIKE": qlike_mean(r_realized, var_forecast),
        "MSE": mse_variance(r_realized, var_forecast),
        "RMSE": rmse_variance(r_realized, var_forecast),
        "MAE": mae_variance(r_realized, var_forecast)
    }
    
    logger.info("Loss metrics computed:")
    for name, value in losses.items():
        logger.info(f"  {name}: {value:.6f}")
    
    return losses


def compare_forecasters(
    r_realized: np.ndarray,
    forecasts: dict,
    loss_fn: Callable = qlike_mean
) -> pd.DataFrame:
    """
    Compare plusieurs forecasters selon une loss function.
    
    Args:
        r_realized: Rendements réalisés (array)
        forecasts: Dict {nom_forecaster: var_forecast_array}
        loss_fn: Fonction de loss à utiliser
    
    Returns:
        DataFrame trié par score (meilleur en premier)
    """
    results = []
    
    for name, var_fc in forecasts.items():
        score = loss_fn(r_realized, var_fc)
        results.append({"Forecaster": name, "Score": score})
    
    df = pd.DataFrame(results).sort_values("Score").reset_index(drop=True)
    
    logger.info(f"Forecaster comparison ({loss_fn.__name__}):")
    logger.info(f"\n{df.to_string(index=False)}")
    
    return df


# ============================================================
# EXEMPLE D'USAGE
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Données synthétiques
    np.random.seed(42)
    n = 500
    
    # Rendements réalisés (%)
    r = np.random.randn(n) * 2
    
    # Variances prévues (3 forecasters)
    var_perfect = r ** 2  # Oracle (impossible en pratique)
    var_good = r ** 2 + np.random.randn(n) * 0.5  # Bon forecaster
    var_bad = np.ones(n) * 4  # Naive (variance constante)
    
    print("=== TEST 1: Single forecast ===")
    q = qlike_mean(r, var_good)
    print(f"QLIKE (good forecaster): {q:.6f}")
    
    print("\n=== TEST 2: All metrics ===")
    losses = compute_all_losses(r, var_good)
    
    print("\n=== TEST 3: Comparison ===")
    forecasts = {
        "Oracle": var_perfect,
        "Good": var_good,
        "Naive": var_bad
    }
    comparison = compare_forecasters(r, forecasts, qlike_mean)
    print(comparison)