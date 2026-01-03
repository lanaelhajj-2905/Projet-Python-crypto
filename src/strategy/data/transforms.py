"""
Transformations de données et préparation pour l'analyse.

Ce module fournit des fonctions pour :
    - Calcul de rendements logarithmiques
    - Split train/validation/test
    - Alignement de DataFrames
    - Nettoyage de données
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


# ============================================================
# CALCUL DE RENDEMENTS
# ============================================================

def compute_log_returns(
    prices: pd.Series,
    scale: float = 100.0
) -> pd.Series:
    """
    Calcule les rendements logarithmiques.
    
    r_t = scale * ln(P_t / P_{t-1})
    
    Args:
        prices: Series ou DataFrame de prix
        scale: Facteur d'échelle (100 pour compatibilité arch)
    
    Returns:
        Series de log returns
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    
    prices = pd.Series(prices).astype(float).dropna().sort_index()
    rets = scale * np.log(prices / prices.shift(1))
    rets = rets.dropna()
    rets.name = "log_return"
    
    logger.info(f"Log returns computed: {len(rets)} observations")
    return rets


def compute_log_returns_df(
    prices: pd.DataFrame,
    scale: float = 100.0
) -> pd.DataFrame:
    """
    Calcule les log returns pour un DataFrame multi-actifs.
    
    Args:
        prices: DataFrame de prix (colonnes = actifs)
        scale: Facteur d'échelle
    
    Returns:
        DataFrame de log returns
    """
    prices = prices.astype(float)
    rets = scale * np.log(prices / prices.shift(1))
    rets = rets.dropna(how="all")
    
    logger.info(f"Log returns computed: {rets.shape}")
    return rets


# ============================================================
# SPLITS TEMPORELS
# ============================================================

def split_train_val_test(
    data: pd.DataFrame,
    train_end: str = "2021-12-31",
    val_end: str = "2023-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise les données en train/validation/test.
    
    Args:
        data: DataFrame à diviser
        train_end: Dernière date du train
        val_end: Dernière date de la validation
    
    Returns:
        Tuple (train, val, test)
    
    Raises:
        ValueError: Si un split est trop petit
    """
    train = data.loc[:train_end]
    val = data.loc[
        pd.Timestamp(train_end) + pd.Timedelta(days=1):val_end
    ]
    test = data.loc[pd.Timestamp(val_end) + pd.Timedelta(days=1):]
    
    # Validation des tailles
    min_sizes = {"train": 250, "val": 100, "test": 100}
    
    for name, df, min_size in [
        ("train", train, min_sizes["train"]),
        ("val", val, min_sizes["val"]),
        ("test", test, min_sizes["test"])
    ]:
        if len(df) < min_size:
            raise ValueError(
                f"{name.capitalize()} too small: "
                f"{len(df)} < {min_size} observations"
            )
    
    logger.info(
        f"Split completed: train={len(train)}, "
        f"val={len(val)}, test={len(test)}"
    )
    
    return train, val, test


def create_date_splits(
    start: str = "2020-01-01",
    train_end: str = "2021-12-31",
    val_end: str = "2023-12-31",
    test_end: str = "2025-12-31"
) -> dict:
    """
    Crée un dictionnaire de dates de split.
    
    Args:
        start: Date de début des données
        train_end: Fin du training
        val_end: Fin de la validation
        test_end: Fin du test
    
    Returns:
        Dict avec toutes les dates de split
    """
    splits = {
        "TRAIN_START": start,
        "TRAIN_END": train_end,
        "VAL_START": pd.Timestamp(train_end) + pd.Timedelta(days=1),
        "VAL_END": val_end,
        "TEST_START": pd.Timestamp(val_end) + pd.Timedelta(days=1),
        "TEST_END": test_end
    }
    
    # Conversion en strings
    for key in ["VAL_START", "TEST_START"]:
        splits[key] = splits[key].strftime("%Y-%m-%d")
    
    logger.info(f"Date splits created: {splits}")
    return splits


# ============================================================
# ALIGNEMENT DE DONNÉES
# ============================================================

def align_dataframes(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Aligne plusieurs DataFrames sur l'intersection de leurs dates.
    
    Args:
        *dfs: DataFrames à aligner
    
    Returns:
        Tuple de DataFrames alignés
    """
    if len(dfs) == 0:
        return tuple()
    
    # Intersection des index
    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)
    
    if len(common_idx) == 0:
        raise ValueError("No common dates between DataFrames")
    
    aligned = tuple(df.loc[common_idx] for df in dfs)
    
    logger.info(
        f"Aligned {len(dfs)} DataFrames: {len(common_idx)} common dates"
    )
    return aligned


def create_balanced_panel(
    prices: pd.DataFrame,
    how: str = "any"
) -> pd.DataFrame:
    """
    Crée un panel équilibré (supprime les dates avec NA).
    
    Args:
        prices: DataFrame de prix
        how: "any" (supprime si au moins 1 NA) ou "all" (si tous NA)
    
    Returns:
        DataFrame sans valeurs manquantes
    """
    before = len(prices)
    clean = prices.dropna(how=how)
    after = len(clean)
    
    pct_removed = 100 * (before - after) / before if before > 0 else 0
    
    logger.info(
        f"Balanced panel: removed {before - after} dates "
        f"({pct_removed:.1f}%), {after} remaining"
    )
    
    return clean


# ============================================================
# VALIDATION DE DONNÉES
# ============================================================

def validate_returns(returns: pd.Series) -> None:
    """
    Valide qu'une série de rendements est correcte.
    
    Args:
        returns: Series de rendements
    
    Raises:
        ValueError: Si des problèmes critiques sont détectés
    """
    # NaN
    n_na = returns.isna().sum()
    if n_na > 0:
        pct_na = 100 * n_na / len(returns)
        logger.warning(f"Returns contain {n_na} NaN ({pct_na:.2f}%)")
    
    # Valeurs extrêmes (>100% daily improbable)
    extreme = returns[np.abs(returns) > 100]
    if len(extreme) > 0:
        logger.warning(f"Found {len(extreme)} extreme returns (>100%)")
        logger.warning(f"Max: {returns.max():.2f}%, Min: {returns.min():.2f}%")
    
    # Ordre chronologique
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Returns index not sorted chronologically")
    
    logger.info("Returns validation passed")


def check_data_coverage(
    data: pd.DataFrame,
    threshold: float = 0.95
) -> None:
    """
    Vérifie la couverture des données (% non-NaN).
    
    Args:
        data: DataFrame à vérifier
        threshold: Seuil minimum de couverture (0-1)
    
    Raises:
        ValueError: Si couverture < threshold
    """
    coverage = data.notna().mean()
    
    logger.info("Data coverage by column:")
    for col, cov in coverage.items():
        logger.info(f"  {col}: {cov*100:.1f}%")
    
    low_coverage = coverage[coverage < threshold]
    if len(low_coverage) > 0:
        logger.warning(
            f"Low coverage columns (<{threshold*100}%): "
            f"{low_coverage.to_dict()}"
        )


# ============================================================
# UTILITAIRES DATES
# ============================================================

def get_rebalance_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "W-MON"
) -> pd.DatetimeIndex:
    """
    Génère les dates de rebalancement.
    
    Args:
        start: Date de début
        end: Date de fin
        freq: Fréquence (ex: "W-MON" pour lundi hebdo)
    
    Returns:
        DatetimeIndex des dates de rebalancement
    """
    rebal_dates = pd.date_range(start, end, freq=freq)
    
    # Assure que start est inclus
    if len(rebal_dates) == 0 or rebal_dates[0] != start:
        rebal_dates = rebal_dates.insert(0, start)
    
    logger.info(f"Generated {len(rebal_dates)} rebalance dates ({freq})")
    return rebal_dates


# ============================================================
# EXEMPLE D'USAGE
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test 1: Log returns
    print("=== TEST 1: Log returns ===")
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    prices = pd.Series(
        np.exp(np.random.randn(len(dates)).cumsum() * 0.02 + np.log(30000)),
        index=dates,
        name="BTC"
    )
    rets = compute_log_returns(prices)
    print(rets.head())
    validate_returns(rets)
    
    # Test 2: Splits
    print("\n=== TEST 2: Splits ===")
    df = pd.DataFrame({"price": prices})
    train, val, test = split_train_val_test(df)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Test 3: Rebalance dates
    print("\n=== TEST 3: Rebalance dates ===")
    rebal = get_rebalance_dates(
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2022-03-31"),
        "W-MON"
    )
    print(rebal)