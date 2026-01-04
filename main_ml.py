# main_ml.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from experiments.data_downloader import DataDownloader
from experiments.volatility import VolatilityCalculator
from experiments.ml_models import StressPredictor, DirectionalPredictor
from experiments.datasets import build_stress_dataset, build_direction_dataset

OUTPUT_DIR = Path("data/processed/ML")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PortfolioStrategy:
    """Stratégies d'allocation classiques"""
    
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
        w = (vol.le(q, axis=0)).astype(float)
        return w

    @staticmethod
    def ml_risk_gate(p_stress, p_direction=None, mode="hysteresis"):
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
                gate.loc[t] = G_OFF if state == 1 else 1.0
            return gate
        elif mode == "smooth":
            ALPHA = 2.0
            p_capped = p_stress.clip(0.0, 0.80)
            return 1.0 / (1.0 + ALPHA * p_capped)
        elif mode == "combined":
            if p_direction is None:
                raise ValueError("p_direction requis pour mode 'combined'")
            thr_stress, thr_up_hi, thr_up_lo = 0.85, 0.60, 0.40
            gate = pd.Series(1.0, index=p_stress.index)
            stress_hi = (p_stress >= p_stress.quantile(thr_stress))
            dir_hi = (p_direction >= p_direction.quantile(thr_up_hi))
            dir_lo = (p_direction <= p_direction.quantile(thr_up_lo))
            gate.loc[stress_hi & dir_lo] = 0.2
            gate.loc[stress_hi & dir_hi] = 0.8
            return gate


def run_portfolio_strategy(symbols=None, start="2021-01-01", end="2025-12-31",
                           strategy="ml_combined"):
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

    # 1️⃣ Charger / télécharger les données
    data_dir = Path("data/raw")
    downloader = DataDownloader(output_dir=data_dir)
    for sym in symbols:
        csv_file = data_dir / f"{sym}_1d.csv"
        if not csv_file.exists():
            downloader.download_binance_public([sym])

    dfs = {}
    for sym in symbols:
        df = pd.read_csv(data_dir / f"{sym}_1d.csv", parse_dates=["timestamp"], index_col="timestamp")
        df = df.sort_index().loc[start:end]
        dfs[sym] = df

    # 2️⃣ Calcul volatilité
    vc = VolatilityCalculator()
    rets = pd.DataFrame({s: vc.add_returns(dfs[s])["ret"] for s in symbols})
    vol = pd.DataFrame({s: vc.add_volatility_features(dfs[s])["vol_rolling"] for s in symbols})

    # 3️⃣ Stratégie
    ps = PortfolioStrategy()
    if strategy == "equal_weight":
        weights = ps.equal_weight(rets)
    elif strategy == "inverse_vol":
        weights = ps.inverse_volatility(vol)
    elif strategy == "low_vol":
        weights = ps.low_volatility_filter(vol)
    elif strategy in ["ml_stress", "ml_combined"]:
        # Dataset ML
        ds_stress, cutoff = build_stress_dataset(rets, vol)
        X_train = ds_stress[ds_stress.index < cutoff].drop(columns=["stress"])
        y_train = ds_stress[ds_stress.index < cutoff]["stress"]
        X_test = ds_stress[ds_stress.index >= cutoff].drop(columns=["stress"])
        y_test = ds_stress[ds_stress.index >= cutoff]["stress"]

        stress_model = StressPredictor()
        stress_model.fit(X_train, y_train)
        p_stress = pd.Series(index=ds_stress.index, dtype=float)
        p_stress.loc[X_train.index] = stress_model.predict_proba(X_train)
        p_stress.loc[X_test.index] = stress_model.predict_proba(X_test)

        base_weights = ps.low_volatility_filter(vol)
        if strategy == "ml_stress":
            gate = ps.ml_risk_gate(p_stress, mode="hysteresis")
        else:
            ds_dir, _ = build_direction_dataset(rets, vol)
            X_train_dir = ds_dir[ds_dir.index < cutoff].drop(columns=["y_up"])
            y_train_dir = ds_dir[ds_dir.index < cutoff]["y_up"]
            X_test_dir = ds_dir[ds_dir.index >= cutoff].drop(columns=["y_up"])
            y_test_dir = ds_dir[ds_dir.index >= cutoff]["y_up"]

            dir_model = DirectionalPredictor(model_type="logit")
            dir_model.fit(X_train_dir, y_train_dir)
            p_direction = pd.Series(index=ds_dir.index, dtype=float)
            p_direction.loc[X_train_dir.index] = dir_model.predict_proba(X_train_dir)
            p_direction.loc[X_test_dir.index] = dir_model.predict_proba(X_test_dir)

            gate = ps.ml_risk_gate(p_stress, p_direction, mode="combined")

        gate_aligned = gate.reindex(rets.index).ffill().fillna(1.0)
        weights = base_weights.mul(gate_aligned, axis=0)
    else:
        raise ValueError(f"Stratégie inconnue: {strategy}")

    # 4️⃣ Normalisation finale
    weights = weights.clip(lower=0.0)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 5️⃣ Backtest simple
    port_ret = (weights * rets).sum(axis=1)
    equity = (1 + port_ret.fillna(0)).cumprod()
    stats = {
        "ann_return": port_ret.mean() * 252,
        "ann_vol": port_ret.std() * np.sqrt(252),
        "sharpe": port_ret.mean() / port_ret.std() * np.sqrt(252),
        "max_dd": (equity.cummax() - equity).max()
    }

    # 6️⃣ Plot equity
    plt.figure(figsize=(12,6))
    plt.plot(equity.index, equity, linewidth=2)
    plt.title(f"Equity Curve - {strategy}")
    plt.grid(alpha=0.3)
    plt.show()

    return {"returns": port_ret, "weights": weights, "equity": equity, "stats": stats}


def main():
    strategies = ["equal_weight", "inverse_vol", "low_vol", "ml_stress", "ml_combined"]
    all_equity_curves = {}
    results = {}

    for strat in strategies:
        print(f"\n=== STRATEGY: {strat} ===")
        try:
            res = run_portfolio_strategy(strategy=strat)
            results[strat] = res
            all_equity_curves[strat] = res["equity"]
        except Exception as e:
            print(f"❌ Erreur pour {strat}: {e}")

    # Export CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results:
        # Stats comparatives
        comparison = pd.DataFrame({s: r["stats"] for s,r in results.items()}).T
        comparison_file = OUTPUT_DIR / f"strategy_comparison_{timestamp}.csv"
        comparison.to_csv(comparison_file)
        print(f"\n✓ Statistiques sauvegardées: {comparison_file}")

        # Equity
        if all_equity_curves:
            equity_df = pd.DataFrame(all_equity_curves)
            equity_file = OUTPUT_DIR / f"equity_curves_{timestamp}.csv"
            equity_df.to_csv(equity_file)
            print(f"✓ Courbes d'equity sauvegardées: {equity_file}")

        # Détails par stratégie
        for strat_name, res in results.items():
            detail_file = OUTPUT_DIR / f"{strat_name}_details_{timestamp}.csv"
            res["weights"].to_csv(detail_file)
            print(f"✓ Détails {strat_name} sauvegardés: {detail_file}")

    else:
        print("❌ Aucune stratégie n'a fonctionné.")


if __name__ == "__main__":
    main()
