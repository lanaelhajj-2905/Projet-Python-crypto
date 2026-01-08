# main_lowvol_trend.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------- imports modules locaux ----------
from src.lowvol_trend.loader import load_symbol
from src.lowvol_trend.strategies import lowvol_trend_strategy
from src.lowvol_trend.backtest import run_bt, stats

# =====================
# PARAMS
# =====================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
DATA_DIR = "data/raw"
OUT_DIR = Path("data/processed/lowvol_trend")
START = "2021-01-01"
END = "2025-12-31"
COST_BPS = 10

OUT_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# =====================
# LOAD DATA
# =====================
dfs = {s: load_symbol(s, DATA_DIR, START, END) for s in SYMBOLS}

idx = None
for s in SYMBOLS:
    idx = dfs[s].index if idx is None else idx.intersection(dfs[s].index)

rets = pd.DataFrame({s: dfs[s].loc[idx, "ret"] for s in SYMBOLS})
close = pd.DataFrame({s: dfs[s].loc[idx, "close"] for s in SYMBOLS})

# =====================
# STRATEGY
# =====================
weights, vol, trend_on = lowvol_trend_strategy(rets, close)

# =====================
# BACKTEST
# =====================
r, to, w_used = run_bt(weights, rets, COST_BPS)
st = stats(r)
st["turnover_mean"] = float(to.mean())

summary = pd.DataFrame([st], index=["lowvol_trend"])
summary.to_csv(OUT_DIR / f"stats_{ts}.csv")

# =====================
# EXPORT CSV
# =====================
weights.to_csv(OUT_DIR / f"weights_{ts}.csv")
vol.to_csv(OUT_DIR / f"volatility_{ts}.csv")
trend_on.to_frame("trend_on").to_csv(OUT_DIR / f"trend_gate_{ts}.csv")

# =====================
# PLOTS
# =====================
eq = (1 + r.fillna(0)).cumprod()

plt.figure(figsize=(12,5))
plt.plot(eq.index, eq, label="lowvol_trend")
plt.legend()
plt.title("Equity curve – Low Vol + BTC Trend")
plt.tight_layout()
plt.savefig(OUT_DIR / "equity.png")
plt.close()

plt.figure(figsize=(10,4))
plt.plot(trend_on.index, trend_on, lw=1)
plt.title("BTC Trend Gate (MA200)")
plt.tight_layout()
plt.savefig(OUT_DIR / "trend_gate.png")
plt.close()

print(summary)
print(f"\nRésultats dans : {OUT_DIR.resolve()}")
