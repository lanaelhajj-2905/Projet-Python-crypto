import pandas as pd
import numpy as np

def lowvol_trend_strategy(rets, close, win_vol=20, ma_trend=200):
    # rolling vol
    vol = rets.rolling(win_vol).std()

    # low-vol cross-section
    q = vol.quantile(0.5, axis=1)
    w_low = vol.le(q, axis=0).astype(float)

    # BTC trend gate
    btc = close["BTCUSDT"]
    btc_ma = btc.rolling(ma_trend).mean()
    trend_on = (btc > btc_ma).astype(float).fillna(0.0)

    # apply gate
    w = w_low.mul(trend_on, axis=0)

    # normalize
    w = w.clip(lower=0.0)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    return w, vol, trend_on
