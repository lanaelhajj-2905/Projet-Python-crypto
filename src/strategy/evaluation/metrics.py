"""
Métriques de performance pour stratégies de trading.
"""

import numpy as np
import pandas as pd


def equity(rets: pd.Series, start: float = 1.0) -> pd.Series:
    """Courbe d'équité."""
    return start * (1 + rets).cumprod()


def total_return(rets: pd.Series) -> float:
    """Rendement total."""
    return float(equity(rets).iloc[-1] - 1.0)


def ann_return(rets: pd.Series, days: float = 365) -> float:
    """Rendement annualisé (CAGR)."""
    eq = equity(rets)
    return float(eq.iloc[-1] ** (days / len(rets)) - 1.0)


def ann_vol(rets: pd.Series, days: float = 365) -> float:
    """Volatilité annualisée."""
    return float(rets.std() * np.sqrt(days))


def sharpe(rets: pd.Series, rf: float = 0, days: float = 365) -> float:
    """Sharpe ratio."""
    ret = ann_return(rets, days)
    vol = ann_vol(rets, days)
    return (ret - rf) / vol if vol > 0 else np.nan


def max_dd(rets: pd.Series) -> float:
    """Maximum Drawdown."""
    eq = equity(rets)
    peak = eq.cummax()
    dd = (eq / peak) - 1
    return float(dd.min())


def calmar(rets: pd.Series, days: float = 365) -> float:
    """Calmar ratio."""
    ret = ann_return(rets, days)
    mdd = max_dd(rets)
    return ret / abs(mdd) if mdd < 0 else np.nan


def sortino(rets: pd.Series, days: float = 365) -> float:
    """Sortino ratio."""
    ret = ann_return(rets, days)
    downside = rets[rets < 0].std() * np.sqrt(days)
    return ret / downside if downside > 0 else np.nan


def metrics(rets: pd.Series, days: float = 365) -> dict:
    """Calcule toutes les métriques."""
    return {
        "TotalReturn": total_return(rets),
        "AnnReturn": ann_return(rets, days),
        "AnnVol": ann_vol(rets, days),
        "Sharpe": sharpe(rets, 0, days),
        "Sortino": sortino(rets, days),
        "MaxDD": max_dd(rets),
        "Calmar": calmar(rets, days),
        "Obs": len(rets)
    }


def compare(strategies: dict, days: float = 365) -> pd.DataFrame:
    """Compare plusieurs stratégies."""
    results = []
    for name, rets in strategies.items():
        m = metrics(rets, days)
        m["Strategy"] = name
        results.append(m)
    
    df = pd.DataFrame(results)
    return df[["Strategy"] + [c for c in df.columns if c != "Strategy"]]


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31")
    
    strat_a = pd.Series(np.random.randn(len(dates)) * 0.02 + 0.001, index=dates)
    strat_b = pd.Series(np.random.randn(len(dates)) * 0.03 + 0.0005, index=dates)
    
    print(compare({"A": strat_a, "B": strat_b}).round(4))