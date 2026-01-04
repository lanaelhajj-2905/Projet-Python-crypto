from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from experiments.volatility import VolatilityCalculator
from experiments.ml_models import StressPredictor, DirectionalPredictor
from experiments.datasets import build_stress_dataset, build_direction_dataset
from experiments.backtester import Backtester
from sklearn.metrics import roc_auc_score
import numpy as np

class PortfolioStrategy:
    @staticmethod
    def equal_weight(rets):
        return pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    
    @staticmethod
    def inverse_volatility(vol):
        w = 1.0 / vol.replace(0, np.nan)
        return w.fillna(0.0)
    
    @staticmethod
    def low_volatility_filter(vol, quantile=0.5):
        q = vol.quantile(quantile, axis=1)
        return (vol.le(q, axis=0)).astype(float)
    
    @staticmethod
    def ml_risk_gate(p_stress, p_direction=None, mode="hysteresis"):
        P_ON, P_OFF, G_OFF = 0.7, 0.55, 0.2
        gate = pd.Series(1.0, index=p_stress.index)
        if mode == "hysteresis":
            state = 0
            for t in p_stress.index:
                p = float(p_stress.loc[t])
                if state == 0 and p >= P_ON:
                    state = 1
                elif state == 1 and p <= P_OFF:
                    state = 0
                gate.loc[t] = (G_OFF if state == 1 else 1.0)
        elif mode == "smooth":
            alpha = 2.0
            gate = 1.0 / (1.0 + alpha * p_stress.clip(0.0, 0.8))
        elif mode == "combined":
            if p_direction is None:
                raise ValueError("p_direction requis pour mode 'combined'")
            thr_stress, thr_up_hi, thr_up_lo = 0.85, 0.6, 0.4
            gate.loc[(p_stress >= p_stress.quantile(thr_stress)) & 
                     (p_direction <= p_direction.quantile(thr_up_lo))] = 0.2
            gate.loc[(p_stress >= p_stress.quantile(thr_stress)) & 
                     (p_direction >= p_direction.quantile(thr_up_hi))] = 0.8
        return gate

def run_portfolio_strategy(symbols=None, data_dir="data", start="2021-01-01", end="2025-12-31", strategy="ml_combined", cost_bps=10):
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]

    print("="*80)
    print(f"PORTFOLIO STRATEGY: {strategy}")
    print("="*80)

    # Charger les données
    dfs = {}
    for sym in symbols:
        fn = Path(data_dir) / f"{sym}_1d.csv"
        if not fn.exists():
            raise FileNotFoundError(f"{fn} manquant !")
        dfs[sym] = pd.read_csv(fn, parse_dates=["timestamp"], index_col="timestamp").sort_index().loc[start:end]

    rets = pd.DataFrame({s: np.log(dfs[s]["close"] / dfs[s]["close"].shift(1)) for s in symbols}).dropna()
    
    # Calcul volatilité
    vc = VolatilityCalculator()
    vol_df = pd.DataFrame(index=rets.index)
    for s in symbols:
        df_s = vc.add_returns(dfs[s])
        df_s = vc.add_volatility_features(df_s)
        vol_df[s] = df_s["vol_ewma"]

    ps = PortfolioStrategy()

    if strategy == "equal_weight":
        weights = ps.equal_weight(rets)
    elif strategy == "inverse_vol":
        weights = ps.inverse_volatility(vol_df)
    elif strategy == "low_vol":
        weights = ps.low_volatility_filter(vol_df)
    else:
        ds_stress, cutoff = build_stress_dataset(rets, vol_df)
        X_train, y_train = ds_stress[ds_stress.index < cutoff].drop(columns=["stress"]), ds_stress[ds_stress.index < cutoff]["stress"]
        X_test, y_test = ds_stress[ds_stress.index >= cutoff].drop(columns=["stress"]), ds_stress[ds_stress.index >= cutoff]["stress"]

        stress_model = StressPredictor()
        stress_model.fit(X_train, y_train)
        p_stress = pd.Series(index=ds_stress.index)
        p_stress.loc[X_train.index] = stress_model.predict_proba(X_train)
        p_stress.loc[X_test.index] = stress_model.predict_proba(X_test)
        print(f"Stress AUC test: {roc_auc_score(y_test, p_stress.loc[X_test.index]):.4f}")

        base_weights = ps.low_volatility_filter(vol_df)

        if strategy == "ml_stress":
            gate = ps.ml_risk_gate(p_stress, mode="hysteresis")
        else:
            ds_dir, _ = build_direction_dataset(rets, vol_df)
            X_train_dir, y_train_dir = ds_dir[ds_dir.index < cutoff].drop(columns=["y_up"]), ds_dir[ds_dir.index < cutoff]["y_up"]
            X_test_dir, y_test_dir = ds_dir[ds_dir.index >= cutoff].drop(columns=["y_up"]), ds_dir[ds_dir.index >= cutoff]["y_up"]

            dir_model = DirectionalPredictor(model_type="logit")
            dir_model.fit(X_train_dir, y_train_dir)
            p_direction = pd.Series(index=ds_dir.index)
            p_direction.loc[X_train_dir.index] = dir_model.predict_proba(X_train_dir)
            p_direction.loc[X_test_dir.index] = dir_model.predict_proba(X_test_dir)
            print(f"Direction AUC test: {roc_auc_score(y_test_dir, p_direction.loc[X_test_dir.index]):.4f}")

            gate = ps.ml_risk_gate(p_stress, p_direction, mode="combined")

        gate_aligned = gate.reindex(rets.index).ffill().fillna(1.0)
        weights = base_weights.mul(gate_aligned, axis=0)

    # Normalisation
    weights = weights.clip(lower=0.0)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Backtest
    bt = Backtester()
    port_ret, turnover, _ = bt.run(weights, rets)
    stats = bt.performance_stats(port_ret)
    stats["turnover_mean"] = float(turnover.mean())

    # Affichage
    print("\n" + "="*80)
    print("RÉSULTATS")
    print("="*80)
    print(f"Return annualisé: {stats['ann_return']:.2%}")
    print(f"Vol annualisée:   {stats['ann_vol']:.2%}")
    print(f"Sharpe:           {stats['sharpe']:.3f}")
    print(f"Max Drawdown:     {stats['max_dd']:.2%}")
    print(f"Turnover moyen:   {stats['turnover_mean']:.2%}")

    equity = (1 + port_ret.fillna(0)).cumprod()
    plt.figure(figsize=(12,6))
    plt.plot(equity.index, equity)
    plt.title(f"Equity Curve - {strategy}")
    plt.grid(alpha=0.3)
    plt.show()

    return {"returns": port_ret, "weights": weights, "stats": stats, "equity": equity}
