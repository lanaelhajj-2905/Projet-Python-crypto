import numpy as np

class VolatilityCalculator:
    """Calcule différentes mesures de volatilité"""
    
    @staticmethod
    def add_returns(df):
        """Log-returns"""
        df = df.copy()
        df["ret"] = np.log(df["close"] / df["close"].shift(1))
        return df
    
    @staticmethod
    def add_volatility_features(df, window=20, lambda_ewma=0.94):
        """Volatilités simples: rolling, EWMA, Parkinson, Garman-Klass"""
        
        # Rolling
        df["vol_rolling"] = df["ret"].rolling(window).std()
        
        # EWMA
        df["vol_ewma"] = np.sqrt((df["ret"] ** 2).ewm(alpha=1 - lambda_ewma).mean())
        
        # Parkinson
        hl = np.log(df["high"] / df["low"])
        df["vol_parkinson"] = np.sqrt((hl ** 2).rolling(window).mean() / (4 * np.log(2)))
        
        # Garman-Klass
        ho = np.log(df["high"] / df["open"])
        lo = np.log(df["low"] / df["open"])
        co = np.log(df["close"] / df["open"])
        gk_var = 0.5 * (ho - lo) ** 2 - (2 * np.log(2) - 1) * (co ** 2)
        df["vol_gk"] = np.sqrt(gk_var.rolling(window).mean().clip(lower=0))
        
        # Vol regime (z-score)
        df["vol_z"] = (df["vol_ewma"] - df["vol_ewma"].rolling(252).mean()) / df["vol_ewma"].rolling(252).std()
        
        return df
