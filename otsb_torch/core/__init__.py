"""Core optimal transport algorithms using PyTorch."""

from .sinkhorn_torch import sinkhorn, sinkhorn_log, sinkhorn_cost
from .exact_ot_torch import emd, wasserstein_distance

__all__ = ["sinkhorn", "sinkhorn_log", "sinkhorn_cost", "emd", "wasserstein_distance"]
