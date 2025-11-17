"""Utility functions for PyTorch-based optimal transport."""

from .distances_torch import (
    euclidean_cost,
    squared_euclidean_cost,
    cost_matrix,
    gaussian_kernel,
)

__all__ = [
    "euclidean_cost",
    "squared_euclidean_cost",
    "cost_matrix",
    "gaussian_kernel",
]
