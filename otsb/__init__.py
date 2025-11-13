"""
OT-SB: A comprehensive library for Schrödinger Bridge and Optimal Transport.

This library provides implementations of various optimal transport algorithms
and Schrödinger bridge solvers for applications in machine learning, statistics,
and scientific computing.
"""

__version__ = "0.1.0"

from .core.sinkhorn import sinkhorn, sinkhorn_log
from .core.exact_ot import emd, emd2
from .solvers.schrodinger_bridge import SchrodingerBridgeSolver
from .utils.distances import euclidean_cost, squared_euclidean_cost

__all__ = [
    "sinkhorn",
    "sinkhorn_log",
    "emd",
    "emd2",
    "SchrodingerBridgeSolver",
    "euclidean_cost",
    "squared_euclidean_cost",
]
