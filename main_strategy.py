"""
Point d'entrée principal pour la Phase 2 : Stratégies de trading.

Usage:
    python main_strategy.py
"""

import logging
from pathlib import Path
from src.strategy.pipelines import StrategyPipeline


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('out/logs/strategy.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Actifs (les 6 sélectionnés en Phase 1)
    "tickers": {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "XRP": "XRP-USD",
        "ADA": "ADA-USD",
        "SOL": "SOL-USD",
        "DOGE": "DOGE-USD"
    },
    
    # Dates
    "start": "2020-01-01",
    "end": "2025-12-31",
    "train_start": "2020-04-11",  # SOL listage
    "val_start": "2022-01-01",
    "val_end": "2023-12-31",
    "test_start": "2024-01-01",
    "test_end": "2025-12-31",
    
    # Paramètres stratégie
    "rebal_freq": "W-MON",  # Rebalancement hebdomadaire
    "cap": 0.35,            # Poids max 35% par actif
    "cost_bps": 10,         # 0.1% de coûts de transaction
}


# ============================================================
# MAIN
# ============================================================

def main():
    """Exécute le pipeline complet."""
    
    logger.info("Starting strategy analysis")
    logger.info(f"Config: {CONFIG}")
    
    # Crée répertoires de sortie
    Path("out/logs").mkdir(parents=True, exist_ok=True)
    Path("data/processed/strategy").mkdir(parents=True, exist_ok=True)
    
    # Pipeline
    pipeline = StrategyPipeline(CONFIG)
    results = pipeline.run()
    
    # Affiche résumé
    pipeline.print_summary()
    
    # Sauvegarde résultats
    logger.info("\nSaving results to data/processed/strategy/")
    results["metrics_val"].to_csv("data/processed/strategy/metrics_val.csv", index=False)
    results["metrics_test"].to_csv("data/processed/strategy/metrics_test.csv", index=False)
    results["weights_val"].to_csv("data/processed/strategy/weights_val.csv")
    results["weights_test"].to_csv("data/processed/strategy/weights_test.csv")
    results["selection"]["results"].to_csv("data/processed/strategy/model_selection.csv", index=False)
    
    logger.info("✅ Strategy analysis completed")


if __name__ == "__main__":
    main()