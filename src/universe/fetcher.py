import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

class ExchangeConnectionError(Exception):
    pass

class InsufficientDataError(Exception):
    pass

class MarketNotFoundError(Exception):
    pass

#Exceptions :

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Échec après {max_retries} tentatives: {e}")
                        raise
                    logger.warning(f"Tentative {attempt + 1}/{max_retries} échouée: {e}")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

class CryptoDataFetcher:
    def __init__(self, config):
        self.config = config
        self.exchange = self._initialize_exchange()
        logger.info(f"Exchange initialisé : {config.EXCHANGE_NAME}")
    
    def _initialize_exchange(self):
        try:
            exchange_class = getattr(ccxt, self.config.EXCHANGE_NAME)
            exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
            exchange.load_markets()
            return exchange
        except AttributeError:
            raise ExchangeConnectionError(f"Exchange '{self.config.EXCHANGE_NAME}' non supporté")
        except Exception as e:
            raise ExchangeConnectionError(f"Erreur connexion exchange: {e}")

    def find_trading_pair(self, base: str) -> Optional[str]:
        for quote in self.config.QUOTE_PRIORITY:
            symbol = f"{base}/{quote}"
            if symbol in self.exchange.markets:
                return symbol
        logger.warning(f"Aucune paire trouvée pour {base}")
        return None

#réésai :
    @retry_on_failure()
    def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
        return df.set_index('timestamp').sort_index().drop_duplicates()


    def fetch_all_assets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        price_series_list, coverage_info, missing_assets = [], [], []
        for asset in self.config.ASSETS:
            symbol = self.find_trading_pair(asset)
            if symbol is None:
                missing_assets.append(asset)
                continue
            try:
                df = self.fetch_ohlcv_data(symbol, self.config.TIMEFRAME, self.config.OHLCV_LIMIT)
                if len(df) < self.config.MIN_OBSERVATIONS:
                    logger.warning(f"{asset}: seulement {len(df)} observations")
                    continue
                price_series_list.append(df['close'].rename(asset))
                coverage_info.append({
                    'asset': asset,
                    'symbol': symbol,
                    'first_date': df.index.min(),
                    'last_date': df.index.max(),
                    'n_observations': len(df),
                    'missing_pct': df['close'].isna().mean() * 100
                })
                time.sleep(self.config.RATE_LIMIT_SLEEP)
            except Exception as e:
                logger.error(f"Erreur pour {asset}: {e}")
                continue
        if not price_series_list:
            raise InsufficientDataError("Aucune donnée téléchargée")
        prices_df = pd.concat(price_series_list, axis=1).sort_index()
        logger.info(f"Données brutes : {len(prices_df)} lignes, {len(price_series_list)} cryptos")
        prices_df = prices_df.dropna(how="any")
        logger.info(f"Après alignement : {len(prices_df)} lignes conservées")
        prices_df = prices_df.dropna(how="any")
        coverage_df = pd.DataFrame(coverage_info).sort_values('first_date')
        return prices_df, coverage_df
