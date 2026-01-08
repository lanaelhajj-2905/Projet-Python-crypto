import numpy as np
import pandas as pd

def run_bt(weights, rets, cost_bps=10):
    w = weights.shift(1).reindex(rets.index).fillna(0.0)

    gross = (w * rets).sum(axis=1)

    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 10000.0)

    net = gross - cost
    return net, turnover, w

def max_drawdown(eq):
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def stats(r, freq=365):
    r = r.dropna()
    eq = (1 + r).cumprod()

    ann = float(eq.iloc[-1] ** (freq / len(r)) - 1)
    vol = float(r.std() * np.sqrt(freq))
    sharpe = float((r.mean() * freq) / (r.std() * np.sqrt(freq)))
    dd = max_drawdown(eq)

    return {
        "ann_return": ann,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_dd": dd
    }
