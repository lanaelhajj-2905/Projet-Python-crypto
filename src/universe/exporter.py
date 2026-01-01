import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def setup_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Répertoire de sortie : {output_dir.absolute()}")

class DataExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        setup_output_directory(output_dir)

    def export_dataframe(self, df: pd.DataFrame, filename: str, index: bool = True):
        filepath = self.output_dir / filename
        try:
            df.to_csv(filepath, index=index)
            logger.info(f"Exporté: {filepath}")
        except Exception as e:
            logger.error(f"Erreur export {filename}: {e}")
            raise

    def export_all_results(self, prices, returns, cov_daily, cov_annual, corr, coverage, stats):
        exports = [
            (coverage, 'coverage_report.csv', False),
            (prices, 'prices_close.csv', True),
            (returns, 'log_returns.csv', True),
            (cov_daily, 'covariance_daily.csv', True),
            (cov_annual, 'covariance_annual.csv', True),
            (corr, 'correlation.csv', True),
            (stats, 'summary_statistics.csv', True),
        ]
        for df, filename, use_index in exports:
            self.export_dataframe(df, filename, index=use_index)
