"""
PyTorch-based Optimal Transport and Schrödinger Bridge library.

This module provides GPU-accelerated implementations of optimal transport algorithms
using PyTorch, enabling automatic differentiation and integration with deep learning.
"""

__version__ = "0.1.0"

from .core.sinkhorn_torch import sinkhorn, sinkhorn_log, sinkhorn_cost
from .core.exact_ot_torch import emd, wasserstein_distance
from .solvers.schrodinger_bridge_torch import SchrodingerBridgeSolver
from .utils.distances_torch import euclidean_cost, squared_euclidean_cost

__all__ = [
    "sinkhorn",
    "sinkhorn_log",
    "sinkhorn_cost",
    "emd",
    "wasserstein_distance",
    "SchrodingerBridgeSolver",
    "euclidean_cost",
    "squared_euclidean_cost",
]
