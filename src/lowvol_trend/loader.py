import numpy as np
import pandas as pd
from pathlib import Path

def load_symbol(sym, data_dir, start, end):
    fn = Path(data_dir) / f"{sym}_1d.csv"
    df = pd.read_csv(fn, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index().loc[start:end].copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna(subset=["ret"])
