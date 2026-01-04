from pathlib import Path
import numpy as np
import pandas as pd

def build_stress_dataset(rets, close, high, low, openp, volu, test_days=365, h_stress=5, q_stress=0.85):
    """
    Construit le dataset pour prédire le stress (forte volatilité future)

    Args:
        rets: DataFrame des log-returns
        close, high, low, openp: DataFrames OHLC
        volu: DataFrame des volumes
        test_days: int, nombre de jours pour le test
        h_stress: int, horizon de volatilité future
        q_stress: float, quantile pour définir le stress

    Returns:
        ds: DataFrame features + label 'stress'
        cutoff: date de séparation train/test
    """

    # Volatilité rolling et EWMA
    vol_ewma = np.sqrt((rets**2).ewm(alpha=0.94).mean())
    vol_roll = rets.rolling(20).std()

    # Momentum multi-horizon
    mom5 = close.pct_change(5)
    mom20 = close.pct_change(20)
    mom60 = close.pct_change(60)

    # OHLC features
    range_ = (high - low).replace(0, np.nan)
    body = (close - openp).abs()
    body_ratio = body / range_

    # Volume features
    vol_norm = volu / volu.rolling(20).mean()

    # Label: stress = volatilité future du portefeuille
    w_eq = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    w_eq = w_eq.div(w_eq.sum(axis=1), axis=0)
    port_ret = (w_eq.shift(1) * rets).sum(axis=1)
    realised_future_vol = port_ret.rolling(h_stress).std().shift(-1)

    cutoff = rets.index.max() - pd.Timedelta(days=test_days)
    train_mask = rets.index < cutoff
    thr = realised_future_vol[train_mask].quantile(q_stress)
    y = (realised_future_vol >= thr).astype(int)

    # Features daily
    X = pd.DataFrame(index=rets.index)
    X["avg_vol20"] = vol_roll.mean(axis=1)
    X["avg_ewma"] = vol_ewma.mean(axis=1)
    X["cs_disp"] = vol_roll.std(axis=1)
    X["btc_mom20"] = mom20[rets.columns[0]]
    X["btc_mom60"] = mom60[rets.columns[0]]
    X["avg_vol_norm"] = vol_norm.mean(axis=1)
    X["avg_body_ratio"] = body_ratio.mean(axis=1)

    ds = pd.concat([X, y.rename("stress")], axis=1).dropna()
    return ds, cutoff


def build_direction_dataset(rets, close, high, low, openp, volu, test_days=365):
    """
    Construit le dataset pour prédire la direction du marché (up/down)

    Args:
        rets: DataFrame des log-returns
        close, high, low, openp: DataFrames OHLC
        volu: DataFrame des volumes
        test_days: int, nombre de jours pour le test

    Returns:
        ds: DataFrame features + label 'y_up'
        cutoff: date de séparation train/test
    """

    # Label: marché monte demain?
    w_eq = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    w_eq = w_eq.div(w_eq.sum(axis=1), axis=0)
    mkt_ret = (w_eq.shift(1) * rets).sum(axis=1)
    y_up = (mkt_ret.shift(-1) > 0).astype(int)

    # Features
    vol_ewma = np.sqrt((rets**2).ewm(alpha=0.94).mean())
    vol_roll = rets.rolling(20).std()
    mom5 = close.pct_change(5)
    mom20 = close.pct_change(20)
    mom60 = close.pct_change(60)

    range_ = (high - low).replace(0, np.nan)
    body = (close - openp).abs()
    body_ratio = body / range_
    vol_norm = volu / volu.rolling(20).mean()

    X = pd.DataFrame(index=rets.index)
    X["avg_vol20"] = vol_roll.mean(axis=1)
    X["avg_ewma"] = vol_ewma.mean(axis=1)
    X["cs_disp"] = vol_roll.std(axis=1)
    X["btc_mom5"] = mom5[rets.columns[0]]
    X["btc_mom20"] = mom20[rets.columns[0]]
    X["btc_mom60"] = mom60[rets.columns[0]]
    X["avg_vol_norm"] = vol_norm.mean(axis=1)
    X["avg_body_ratio"] = body_ratio.mean(axis=1)
    X["mkt_ret_1"] = mkt_ret

    ds = pd.concat([X, y_up.rename("y_up")], axis=1).dropna()
    cutoff = ds.index.max() - pd.Timedelta(days=test_days)

    return ds, cutoff
