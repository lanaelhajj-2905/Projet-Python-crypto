# experiments/ml_strategies.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.backtester import Backtester
from experiments.ml_models import StressPredictor, DirectionalPredictor
from experiments.datasets import build_stress_dataset, build_direction_dataset
from experiments.volatility import VolatilityCalculator
from sklearn.metrics import roc_auc_score

class PortfolioStrategy:
    """Stratégies d'allocation de portefeuille"""
    
    @staticmethod
    def equal_weight(rets):
        return pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    
    @staticmethod
    def inverse_volatility(vol):
        w = 1.0 / vol.replace(0, np.nan)
        w = w.fillna(0.0)
        return w
    
    @staticmethod
    def low_volatility_filter(vol, quantile=0.5):
        q = vol.quantile(quantile, axis=1)
        w = (vol.le(q, axis=0)).astype(float)
        return w
    
    @staticmethod
    def ml_risk_gate(p_stress, p_direction=None, mode="hysteresis"):
        """Gate ML pour réduire exposition en période de stress"""
        if mode == "hysteresis":
            P_ON, P_OFF, G_OFF = 0.7, 0.55, 0.2
            gate = pd.Series(1.0, index=p_stress.index)
            state = 0
            for t in p_stress.index:
                p = float(p_stress.loc[t])
                if state == 0 and p >= P_ON:
                    state = 1
                elif state == 1 and p <= P_OFF:
                    state = 0
                gate.loc[t] = (G_OFF if state == 1 else 1.0)
            return gate
        elif mode == "smooth":
            ALPHA = 2.0
            p_capped = p_stress.clip(0.0, 0.8)
            gate = 1.0 / (1.0 + ALPHA * p_capped)
            return gate
        elif mode == "combined":
            if p_direction is None:
                raise ValueError("p_direction requis pour mode 'combined'")
            thr_stress = 0.85
            thr_up_hi = 0.6
            thr_up_lo = 0.4
            gate = pd.Series(1.0, index=p_stress.index)
            stress_hi = (p_stress >= p_stress.quantile(thr_stress))
            dir_hi = (p_direction >= p_direction.quantile(thr_up_hi))
            dir_lo = (p_direction <= p_direction.quantile(thr_up_lo))
            gate.loc[stress_hi & dir_lo] = 0.2
            gate.loc[stress_hi & dir_hi] = 0.8
            return gate

def run_portfolio_strategy(
    symbols=None,
    data_dir="data/raw",
    start="2021-01-01",
    end="2025-12-31",
    strategy="ml_combined",
    cost_bps=10
):
    """Pipeline complet de stratégie de portefeuille"""
    
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    
    print("="*80)
    print(f"PORTFOLIO STRATEGY: {strategy}")
    print("="*80)
    
    # Charger les données
    dfs = {}
    vc = VolatilityCalculator()
    for sym in symbols:
        file_path = Path(data_dir) / f"{sym}_1d.csv"
        df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
        df = vc.add_returns(df)
        df = vc.add_volatility_features(df)
        df = df.loc[start:end]
        dfs[sym] = df
    
    # save volatility table for inspection
    vol_inspect = pd.DataFrame(index=dfs[symbols[0]].loc[start:end].index)  # base index
    for s in symbols:
        tmp = dfs[s].loc[start:end]
        vol_inspect[f"{s}_vol_ewma"] = tmp["vol_ewma"]
        vol_inspect[f"{s}_vol_rolling"] = tmp["vol_rolling"]
        vol_inspect[f"{s}_vol_parkinson"] = tmp["vol_parkinson"]
        vol_inspect[f"{s}_vol_gk"] = tmp["vol_gk"]
        vol_inspect[f"{s}_vol_z"] = tmp["vol_z"]

    out = Path("data/processed/ML")
    out.mkdir(parents=True, exist_ok=True)
    vol_inspect.head(200).to_csv(out / "volatility_preview_head.csv")
    print("✓ volatility_preview_head.csv saved to data/processed/ML (contains vol_ewma, vol_parkinson, vol_gk, vol_z)")

    # Alignement des retours et volatilités
    rets = pd.DataFrame({s: dfs[s]["ret"] for s in symbols})
    vol  = pd.DataFrame({s: dfs[s]["vol_rolling"] for s in symbols})
    
    # Stratégie
    ps = PortfolioStrategy()
    
    if strategy == "equal_weight":
        weights = ps.equal_weight(rets)
    elif strategy == "inverse_vol":
        weights = ps.inverse_volatility(vol)
    elif strategy == "low_vol":
        weights = ps.low_volatility_filter(vol)
    elif strategy in ["ml_stress", "ml_combined"]:
        # Dataset stress
        ds_stress, cutoff = build_stress_dataset(rets)
        X_train = ds_stress[ds_stress.index < cutoff].drop(columns=["stress"])
        y_train = ds_stress[ds_stress.index < cutoff]["stress"]
        X_test = ds_stress[ds_stress.index >= cutoff].drop(columns=["stress"])
        y_test = ds_stress[ds_stress.index >= cutoff]["stress"]
        
        stress_model = StressPredictor()
        stress_model.fit(X_train, y_train)
        
        p_stress = pd.Series(index=ds_stress.index, dtype=float)
        p_stress.loc[X_train.index] = stress_model.predict_proba(X_train)
        p_stress.loc[X_test.index] = stress_model.predict_proba(X_test)
        print(f"Stress AUC test: {roc_auc_score(y_test, p_stress.loc[X_test.index]):.4f}")
        
        # Poids de base
        base_weights = ps.low_volatility_filter(vol)
        
        if strategy == "ml_stress":
            gate = ps.ml_risk_gate(p_stress, mode="hysteresis")
        else:
            ds_dir, _ = build_direction_dataset(rets)
            X_train_dir = ds_dir[ds_dir.index < cutoff].drop(columns=["y_up"])
            y_train_dir = ds_dir[ds_dir.index < cutoff]["y_up"]
            X_test_dir = ds_dir[ds_dir.index >= cutoff].drop(columns=["y_up"])
            y_test_dir = ds_dir[ds_dir.index >= cutoff]["y_up"]
            
            dir_model = DirectionalPredictor(model_type="logit")
            dir_model.fit(X_train_dir, y_train_dir)
            
            p_direction = pd.Series(index=ds_dir.index, dtype=float)
            p_direction.loc[X_train_dir.index] = dir_model.predict_proba(X_train_dir)
            p_direction.loc[X_test_dir.index] = dir_model.predict_proba(X_test_dir)
            print(f"Direction AUC test: {roc_auc_score(y_test_dir, p_direction.loc[X_test_dir.index]):.4f}")
            
            gate = ps.ml_risk_gate(p_stress, p_direction, mode="combined")
        
        gate_aligned = gate.reindex(rets.index).ffill().fillna(1.0)
        weights = base_weights.mul(gate_aligned, axis=0)
    
    # Normalisation finale
    weights = weights.clip(lower=0.0)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    
    # Backtest
    bt = Backtester()  # <-- doit être compatible sans arguments
    port_ret, turnover, _ = bt.run(weights, rets)
    stats = bt.performance_stats(port_ret)
    stats["turnover_mean"] = float(turnover.mean())
    
    # Equity
    equity = (1 + port_ret.fillna(0)).cumprod()
    
    return {
        "weights": weights,
        "returns": port_ret,
        "stats": stats,
        "equity": equity
    }
