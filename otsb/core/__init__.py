"""Core optimal transport algorithms."""

from .sinkhorn import sinkhorn, sinkhorn_log
from .exact_ot import emd, emd2

__all__ = ["sinkhorn", "sinkhorn_log", "emd", "emd2"]
