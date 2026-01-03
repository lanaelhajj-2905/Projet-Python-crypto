"""
Prévisions de volatilité GARCH avec refit hebdomadaire.
"""

import numpy as np
import pandas as pd
from arch import arch_model
import logging

logger = logging.getLogger(__name__)


class GARCHForecaster:
    """Forecaster de volatilité GARCH."""
    
    def __init__(self, model_spec: dict, refit_freq: str = "W-MON"):
        """
        Args:
            model_spec: {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"}
            refit_freq: Fréquence de refit (ex: "W-MON")
        """
        self.spec = model_spec
        self.refit_freq = refit_freq
    
    def fit_garch(self, returns: pd.Series):
        """Fit un modèle GARCH."""
        am = arch_model(
            returns.dropna(),
            mean="Zero",
            vol=self.spec["vol"],
            p=self.spec["p"],
            o=self.spec["o"],
            q=self.spec["q"],
            dist=self.spec["dist"],
            rescale=False
        )
        return am.fit(disp="off")
    
    def forecast_single(
        self,
        returns: pd.Series,
        start: str,
        end: str
    ) -> pd.Series:
        """
        Forecast volatilité pour un actif.
        Refit hebdomadaire + update quotidien σ².
        """
        r = returns.dropna()
        forecast_dates = r.loc[start:end].index
        vols = pd.Series(index=forecast_dates, dtype=float)
        
        # Dates de refit
        refit_dates = pd.date_range(
            forecast_dates.min(),
            forecast_dates.max(),
            freq=self.refit_freq
        )
        if refit_dates[0] != forecast_dates.min():
            refit_dates = refit_dates.insert(0, forecast_dates.min())
        
        for i, refit_date in enumerate(refit_dates):
            # Chunk jusqu'au prochain refit
            next_refit = (
                refit_dates[i + 1] if i + 1 < len(refit_dates)
                else forecast_dates.max() + pd.Timedelta(days=1)
            )
            chunk = forecast_dates[
                (forecast_dates >= refit_date) & (forecast_dates < next_refit)
            ]
            chunk = pd.DatetimeIndex(chunk).intersection(r.index)
            
            if len(chunk) == 0:
                continue
            
            # Fit sur données jusqu'à veille du refit
            fit_end = refit_date - pd.Timedelta(days=1)
            train = r.loc[:fit_end]
            
            if len(train) < 250:
                continue
            
            try:
                res = self.fit_garch(train)
                p = res.params
                omega = float(p["omega"])
                alpha = float(p["alpha[1]"])
                beta = float(p["beta[1]"])
                
                # État initial
                sigma2_t = float(res.conditional_volatility.iloc[-1] ** 2)
                r_t = float(train.iloc[-1])
                
                # Récursion quotidienne
                for d in chunk:
                    sigma2_next = omega + alpha * (r_t ** 2) + beta * sigma2_t
                    vols.loc[d] = np.sqrt(max(sigma2_next, 1e-12))
                    
                    sigma2_t = sigma2_next
                    if d in r.index and pd.notna(r.loc[d]):
                        r_t = float(r.loc[d])
            
            except Exception as e:
                logger.error(f"Fit failed at {refit_date}: {e}")
                continue
        
        return vols
    
    def forecast_portfolio(
        self,
        returns_df: pd.DataFrame,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """Forecast pour plusieurs actifs."""
        logger.info(f"Forecasting: {start} to {end}")
        
        vols = pd.DataFrame(
            index=returns_df.loc[start:end].index,
            columns=returns_df.columns,
            dtype=float
        )
        
        for asset in returns_df.columns:
            logger.info(f"Processing {asset}")
            vols[asset] = self.forecast_single(returns_df[asset], start, end)
        
        return vols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    dates = pd.date_range("2020-01-01", "2023-12-31")
    rets = pd.DataFrame({
        "BTC": np.random.randn(len(dates)) * 3,
        "ETH": np.random.randn(len(dates)) * 4
    }, index=dates)
    
    model = {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"}
    forecaster = GARCHForecaster(model)
    vols = forecaster.forecast_portfolio(rets, "2022-01-01", "2023-12-31")
    print(vols.head())