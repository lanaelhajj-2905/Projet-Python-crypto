import pandas as pd
import requests, zipfile, io
from pathlib import Path

class DataDownloader:
    """Télécharge les données OHLCV depuis Binance"""

    def __init__(self, exchange_name="binance", output_dir="data"):
        self.exchange_name = exchange_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def download_binance_public(self, symbols, interval="1d", years=None):
        """Télécharge depuis les archives publiques Binance"""
        if years is None:
            years = [2021, 2022, 2023, 2024, 2025]

        base_url = "https://data.binance.vision/data/spot/monthly/klines"
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]

        all_data = {}

        for symbol in symbols:
            print(f"Téléchargement {symbol}...")
            parts = []

            for year in years:
                for month in range(1, 13):
                    url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
                    try:
                        r = requests.get(url, timeout=30)
                        if r.status_code != 200:
                            continue

                        z = zipfile.ZipFile(io.BytesIO(r.content))
                        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
                        raw = z.read(csv_name)

                        df_month = pd.read_csv(io.BytesIO(raw), header=None)
                        df_month.columns = cols

                        for c in ["open", "high", "low", "close", "volume"]:
                            df_month[c] = pd.to_numeric(df_month[c], errors="coerce")

                        df_month["timestamp"] = pd.to_datetime(df_month["open_time"], unit="ms")
                        df_month = df_month.set_index("timestamp").sort_index()
                        parts.append(df_month[["open", "high", "low", "close", "volume"]])

                    except Exception:
                        continue

            if parts:
                df = pd.concat(parts).sort_index()
                df = df[~df.index.duplicated(keep="first")]
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    print(f"   Note: {nan_count} valeurs manquantes remplacées (ffill)")
                    df = df.ffill()
                all_data[symbol] = df
                
                output_file = self.output_dir / f"{symbol}_{interval}.csv"
                df.to_csv(output_file)
                print(f"Sauvegardé: {output_file} ({len(df)} points)")

        return all_data
