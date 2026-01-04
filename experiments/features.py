import numpy as np

class FeatureEngineer:

    @staticmethod
    def momentum(df, windows=[5, 20]):
        df = df.copy()
        for w in windows:
            df[f"mom_{w}"] = df["close"].pct_change(w)
        return df

    @staticmethod
    def volume_features(df, windows=[5, 20]):
        df = df.copy()
        for w in windows:
            df[f"vol_mean_{w}"] = df["volume"].rolling(w).mean()
        return df
