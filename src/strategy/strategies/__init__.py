"""Stratégies de trading."""

from .inverse_vol import compute_weights, equal_weight, single_asset

__all__ = [
    "compute_weights",
    "equal_weight",
    "single_asset"
]