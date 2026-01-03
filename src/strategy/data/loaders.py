"""
Téléchargement de données de prix.
Binance (priorité) → Yahoo Finance (fallback).
"""

import pandas as pd
import yfinance as yf
from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def load_from_binance(symbol: str, start: str, end: str) -> pd.Series:
    """Télécharge depuis Binance."""
    client = Client()
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str=start,
        end_str=end
    )
    
    df = pd.DataFrame(klines, columns=[
        "time", "open", "high", "low", "close", "vol",
        "close_time", "qav", "trades", "tbbv", "tbqv", "ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    prices = df.set_index("time")["close"].astype(float)
    logger.info(f"✅ Binance: {len(prices)} days")
    return prices


def load_from_yahoo(ticker: str, start: str, end: str) -> pd.Series:
    """Télécharge depuis Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    prices = df["Close"].squeeze()
    logger.info(f"✅ Yahoo: {len(prices)} days")
    return prices


def load_single(
    ticker: str,
    start: str,
    end: str,
    binance_symbol: str = None,
    prefer: str = "binance"
) -> pd.Series:
    """Télécharge avec fallback automatique."""
    if prefer == "binance" and binance_symbol:
        try:
            return load_from_binance(binance_symbol, start, end)
        except (BinanceAPIException, Exception) as e:
            logger.warning(f"⚠️ Binance failed: {e}")
    
    return load_from_yahoo(ticker, start, end)


def load_multiple(
    tickers: Dict[str, str],
    start: str,
    end: str
) -> pd.DataFrame:
    """Télécharge plusieurs actifs depuis Yahoo."""
    df = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        progress=False
    )["Close"]
    
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    # Renomme avec noms courts
    inv_map = {v: k for k, v in tickers.items()}
    return df.rename(columns=inv_map)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    btc = load_single("BTC-USD", "2020-01-01", "2023-12-31", "BTCUSDT")
    print(btc.head())
    