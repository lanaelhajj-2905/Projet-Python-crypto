from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments.backtester import Backtester
from experiments.ml_models import StressPredictor, DirectionalPredictor
from experiments.datasets import build_stress_dataset, build_direction_dataset
from sklearn.metrics import (roc_auc_score)

class PortfolioStrategy:
    """Stratégies d'allocation de portefeuille"""
    
    @staticmethod
    def equal_weight(rets):
        """Buy & Hold Equal Weight"""
        return pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    
    @staticmethod
    def inverse_volatility(vol):
        """Inverse Volatility Weighting"""
        w = 1.0 / vol.replace(0, np.nan)
        w = w.fillna(0.0)
        return w
    
    @staticmethod
    def low_volatility_filter(vol, quantile=0.5):
        """Low Volatility Filter (garde les actifs sous médiane de vol)"""
        q = vol.quantile(quantile, axis=1)
        w = (vol.le(q, axis=0)).astype(float)
        return w
    
    @staticmethod
    def volatility_targeting(weights, rets, target_vol=0.60, window=60, max_lev=1.0):
        """Vol Targeting: scale weights pour atteindre vol cible"""
        w_lag = weights.shift(1).fillna(0.0)
        w_norm = w_lag.div(w_lag.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        
        port_ret = (w_norm * rets).sum(axis=1)
        port_vol = port_ret.rolling(window).std() * np.sqrt(365)
        
        scale = (target_vol / port_vol).clip(lower=0.0, upper=max_lev).fillna(0.0)
        return scale
    
    @staticmethod
    def ml_risk_gate(p_stress, p_direction=None, mode="hysteresis"):
        """
        Gate ML pour réduire l'exposition en période de stress
        
        Modes:
        - "hysteresis": seuils avec hystérésis
        - "smooth": réduction progressive
        - "combined": combine stress + direction
        """
        if mode == "hysteresis":
            # Paramètres
            P_ON = 0.70
            P_OFF = 0.55
            G_OFF = 0.20
            
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
            # Réduction progressive (alpha = agressivité)
            ALPHA = 2.0
            p_capped = p_stress.clip(0.0, 0.80)
            gate = 1.0 / (1.0 + ALPHA * p_capped)
            return gate
        
        elif mode == "combined":
            # Combine stress et direction
            if p_direction is None:
                raise ValueError("p_direction requis pour mode 'combined'")
            
            thr_stress = 0.85
            thr_up_hi = 0.60
            thr_up_lo = 0.40
            
            gate = pd.Series(1.0, index=p_stress.index)
            
            stress_hi = (p_stress >= p_stress.quantile(thr_stress))
            dir_hi = (p_direction >= p_direction.quantile(thr_up_hi))
            dir_lo = (p_direction <= p_direction.quantile(thr_up_lo))
            
            gate.loc[stress_hi & dir_lo] = 0.2  # très défensif
            gate.loc[stress_hi & dir_hi] = 0.8  # modérément défensif
            
            return gate

def run_portfolio_strategy(
    symbols=None,
    data_dir="binance_public_data",
    start="2021-01-01",
    end="2025-12-31",
    strategy="ml_combined",
    cost_bps=10
):
    """
    Pipeline complet de stratégie de portefeuille
    
    Stratégies disponibles:
    - "equal_weight": Buy & Hold égal
    - "inverse_vol": Inverse volatility
    - "low_vol": Low volatility filter
    - "ml_stress": ML stress gate (hystérésis)
    - "ml_combined": ML stress + direction
    """
    
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    
    print("="*80)
    print(f"PORTFOLIO STRATEGY: {strategy}")
    print("="*80)
    
    # 1. Charger les données
    print("\n1. Chargement des données...")
    dfs = {}
    for sym in symbols:
        # Chercher le fichier (plusieurs formats possibles)
        possible_files = [
            f"{data_dir}/{sym}_1d_2021_2025.csv",
            f"{data_dir}/{sym}_1d.csv",
            f"{data_dir}/{sym.replace('USDT', '_USDT')}_1d.csv"
        ]
        
        fn = None
        for f in possible_files:
            if Path(f).exists():
                fn = f
                break
        
        if fn is None:
            raise FileNotFoundError(f"Impossible de trouver le fichier pour {sym}. Cherché: {possible_files}")
        
        print(f"  Chargement {fn}...")
        df = pd.read_csv(fn, parse_dates=["timestamp"], index_col="timestamp")
        df = df.sort_index().loc[start:end]
        df["ret"] = np.log(df["close"] / df["close"].shift(1))
        dfs[sym] = df.dropna(subset=["ret"])
    
    # Aligner les dates
    common_idx = None
    for df in dfs.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    
    rets = pd.DataFrame({s: dfs[s].loc[common_idx, "ret"] for s in symbols}).dropna()
    close = pd.DataFrame({s: dfs[s].loc[rets.index, "close"] for s in symbols})
    high = pd.DataFrame({s: dfs[s].loc[rets.index, "high"] for s in symbols})
    low = pd.DataFrame({s: dfs[s].loc[rets.index, "low"] for s in symbols})
    openp = pd.DataFrame({s: dfs[s].loc[rets.index, "open"] for s in symbols})
    volu = pd.DataFrame({s: dfs[s].loc[rets.index, "volume"] for s in symbols})
    
    print(f"Période: {rets.index.min()} → {rets.index.max()} ({len(rets)} jours)")
    
    # 2. Calcul volatilité
    print("\n2. Calcul volatilité...")
    vol = rets.rolling(20).std()
    
    # 3. Stratégie de base
    ps = PortfolioStrategy()
    
    if strategy == "equal_weight":
        weights = ps.equal_weight(rets)
    
    elif strategy == "inverse_vol":
        weights = ps.inverse_volatility(vol)
    
    elif strategy == "low_vol":
        weights = ps.low_volatility_filter(vol, quantile=0.5)
    
    elif strategy in ["ml_stress", "ml_combined"]:
        print("\n3. Entraînement modèles ML...")
        
        # Dataset stress
        ds_stress, cutoff = build_stress_dataset(rets, close, high, low, openp, volu)
        
        X_train = ds_stress[ds_stress.index < cutoff].drop(columns=["stress"])
        y_train = ds_stress[ds_stress.index < cutoff]["stress"]
        X_test = ds_stress[ds_stress.index >= cutoff].drop(columns=["stress"])
        y_test = ds_stress[ds_stress.index >= cutoff]["stress"]
        
        # Entraîner stress predictor
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
        else:  # ml_combined
            # Dataset direction
            ds_dir, _ = build_direction_dataset(rets, close, high, low, openp, volu)
            
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
        
        # Apply gate
        gate_aligned = gate.reindex(rets.index).ffill().fillna(1.0)
        weights = base_weights.mul(gate_aligned, axis=0)
        
        print(f"Avg gate: {gate_aligned.mean():.3f}, Min: {gate_aligned.min():.3f}")
    
    # 4. Normalisation finale
    weights = weights.clip(lower=0.0)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    
    # 5. Backtest
    print("\n4. Backtest...")
    bt = Backtester(cost_bps=cost_bps)
    port_ret, turnover, _ = bt.run(weights, rets)
    
    stats = bt.performance_stats(port_ret)
    stats["turnover_mean"] = float(turnover.mean())
    
    # 6. Résultats
    print("\n" + "="*80)
    print("RÉSULTATS")
    print("="*80)
    print(f"Return annualisé: {stats['ann_return']:.2%}")
    print(f"Vol annualisée:   {stats['ann_vol']:.2%}")
    print(f"Sharpe:           {stats['sharpe']:.3f}")
    print(f"Max Drawdown:     {stats['max_dd']:.2%}")
    print(f"Turnover moyen:   {stats['turnover_mean']:.2%}")
    
    # 7. Plot
    equity = (1 + port_ret.fillna(0)).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity, linewidth=2)
    plt.title(f"Equity Curve - {strategy}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        "returns": port_ret,
        "weights": weights,
        "stats": stats,
        "equity": equity
    }
