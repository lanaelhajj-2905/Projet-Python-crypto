# experiments/backtester.py

import pandas as pd
import numpy as np

class Backtester:
    """Engine de backtesting avec coûts de transaction."""

    def __init__(self, cost_bps=10):
        """
        cost_bps : frais de transaction en points de base (bps)
        """
        self.cost_bps = cost_bps

    def run(self, weights: pd.DataFrame, rets: pd.DataFrame):
        """
        Exécute un backtest.
        
        weights : DataFrame des poids (décision à t pour rendement t->t+1)
        rets    : DataFrame des returns des actifs
        """
        # Décaler les poids (t appliqué à t+1)
        w = weights.shift(1).reindex(rets.index).fillna(0.0)
        w = w.clip(lower=0.0)

        # Normalisation
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        # Returns bruts
        port_gross = (w * rets).sum(axis=1)

        # Turnover et coûts
        turnover = w.diff().abs().sum(axis=1).fillna(0.0)
        costs = turnover * (self.cost_bps / 10000.0)

        # Returns nets
        port_net = port_gross - costs

        return port_net, turnover, w

    @staticmethod
    def performance_stats(returns: pd.Series, freq: int = 365):
        """
        Calcule les statistiques de performance classiques.

        returns : Series des rendements quotidiens du portefeuille
        freq    : nombre de périodes annuelles (365 pour daily)
        """
        returns = returns.dropna()
        equity = (1 + returns).cumprod()

        ann_return = float(equity.iloc[-1] ** (freq / len(returns)) - 1) if len(returns) > 0 else np.nan
        ann_vol = float(returns.std() * np.sqrt(freq))
        sharpe = float((returns.mean() * freq) / (returns.std() * np.sqrt(freq))) if returns.std() > 0 else np.nan
        max_dd = float((equity / equity.cummax() - 1.0).min())

        return {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_dd": max_dd
        }

