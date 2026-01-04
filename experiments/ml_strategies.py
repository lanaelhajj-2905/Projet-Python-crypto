# experiments/ml_strategies.py

import pandas as pd
import numpy as np
from pathlib import Path

from experiments.volatility import VolatilityCalculator
from experiments.backtester import Backtester
from experiments.ml_models import StressPredictor, DirectionalPredictor
from experiments.datasets import build_stress_dataset, build_direction_dataset
from sklearn.metrics import roc_auc_score


class PortfolioStrategy:

    @staticmethod
    def equal_weight(rets):
        return pd.DataFrame(1.0, index=rets.index, columns=rets.columns)

    @staticmethod
    def inverse_vol(vol):
        w = 1.0 / vol.replace(0, np.nan)
        return w.fillna(0.0)

    @staticmethod
    def low_vol(vol, q=0.5):
        thresh = vol.quantile(q, axis=1)
        return (vol.le(thresh, axis=0)).astype(float)

    @staticmethod
    def risk_gate(p_stress, p_dir=None, mode="stress"):
        gate = pd.Series(1.0, index=p_stress.index)

        if mode == "stress":
            gate[p_stress > 0.7] = 0.2

        elif mode == "combined":
            gate[(p_stress > 0.7) & (p_dir < 0.4)] = 0.2
            gate[(p_stress > 0.7) & (p_dir > 0.6)] = 0.8

        return gate


def run_portfolio_strategy(
    symbols,
    data_dir,
    strategy="ml_combined"
):

    dfs = {}
    vc = VolatilityCalculator()

    # -------------------------------------------------
    # LOAD DATA + VOL FEATURES
    # -------------------------------------------------
    for s in symbols:
        df = pd.read_csv(
            Path(data_dir) / f"{s}_1d.csv",
            parse_dates=["timestamp"],
            index_col="timestamp"
        )

        df = vc.add_returns(df)
        df = vc.add_volatility_features(df)

        dfs[s] = df.dropna()

    rets = pd.DataFrame({s: dfs[s]["ret"] for s in symbols})
    vol  = pd.DataFrame({s: dfs[s]["vol_rolling"] for s in symbols})
    close = pd.DataFrame({s: dfs[s]["close"] for s in symbols})
    high  = pd.DataFrame({s: dfs[s]["high"] for s in symbols})
    low   = pd.DataFrame({s: dfs[s]["low"] for s in symbols})
    openp = pd.DataFrame({s: dfs[s]["open"] for s in symbols})
    volu  = pd.DataFrame({s: dfs[s]["volume"] for s in symbols})


    ps = PortfolioStrategy()

    # -------------------------------------------------
    # CLASSIC STRATEGIES
    # -------------------------------------------------
    if strategy == "equal_weight":
        weights = ps.equal_weight(rets)

    elif strategy == "inverse_vol":
        weights = ps.inverse_vol(vol)

    elif strategy == "low_vol":
        weights = ps.low_vol(vol)

    # -------------------------------------------------
    # ML STRATEGIES
    # -------------------------------------------------
    else:
        close = pd.DataFrame({s: dfs[s]["close"] for s in symbols})
        high  = pd.DataFrame({s: dfs[s]["high"] for s in symbols})
        low   = pd.DataFrame({s: dfs[s]["low"] for s in symbols})
        openp = pd.DataFrame({s: dfs[s]["open"] for s in symbols})
        volu  = pd.DataFrame({s: dfs[s]["volume"] for s in symbols})

        ds, split = build_stress_dataset(
            rets, close, high, low, openp, volu
        )

        Xtr = ds.loc[:split].drop(columns="stress")
        ytr = ds.loc[:split]["stress"]
        Xte = ds.loc[split:].drop(columns="stress")
        yte = ds.loc[split:]["stress"]

        stress = StressPredictor()
        stress.fit(Xtr, ytr)

        p_stress = pd.Series(
            stress.predict_proba(ds.drop(columns="stress")),
            index=ds.index
        )

        base = ps.low_vol(vol)

        if strategy == "ml_stress":
            gate = ps.risk_gate(p_stress, mode="stress")

        else:
            ds_dir, _ = build_direction_dataset(rets,close,high,low,openp,volu)


            dir_model = DirectionalPredictor()
            dir_model.fit(
                ds_dir.loc[:split].drop(columns="y_up"),
                ds_dir.loc[:split]["y_up"]
            )

            p_dir = pd.Series(
                dir_model.predict_proba(ds_dir.drop(columns="y_up")),
                index=ds_dir.index
            )

            gate = ps.risk_gate(p_stress, p_dir, mode="combined")

        weights = base.mul(gate, axis=0)

    # -------------------------------------------------
    # NORMALISATION + BACKTEST
    # -------------------------------------------------
    weights = weights.clip(lower=0)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)

    bt = Backtester()
    port_ret, turnover, _ = bt.run(weights, rets)

    stats = bt.performance_stats(port_ret)
    stats["turnover"] = turnover.mean()

    equity = (1 + port_ret).cumprod()

    return {
        "weights": weights,
        "returns": port_ret,
        "stats": stats,
        "equity": equity
    }
