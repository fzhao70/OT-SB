"""Utility functions for optimal transport."""

from .distances import euclidean_cost, squared_euclidean_cost, cost_matrix
from .visualization import plot_transport_plan, plot_samples

__all__ = [
    "euclidean_cost",
    "squared_euclidean_cost",
    "cost_matrix",
    "plot_transport_plan",
    "plot_samples",
]
