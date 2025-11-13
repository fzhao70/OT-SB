"""
Exact optimal transport solvers using linear programming.

These solvers compute the exact optimal transport without entropic regularization
using the network simplex algorithm via scipy.optimize.linprog.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional


def emd(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    method: str = "highs",
) -> Tuple[np.ndarray, dict]:
    """
    Solve exact optimal transport problem (Earth Mover's Distance).

    Solves:
        min_{P ∈ Π(a,b)} <P, C>
    where Π(a,b) is the set of joint distributions with marginals a and b.

    Parameters
    ----------
    a : array-like, shape (n,)
        Source distribution (must sum to 1)
    b : array-like, shape (m,)
        Target distribution (must sum to 1)
    C : array-like, shape (n, m)
        Cost matrix
    method : str, default="highs"
        Linear programming solver method

    Returns
    -------
    P : array-like, shape (n, m)
        Optimal transport plan
    result_dict : dict
        Dictionary containing solver information
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    # Check inputs
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("a and b must be 1-dimensional arrays")
    if C.shape != (len(a), len(b)):
        raise ValueError(f"C must have shape ({len(a)}, {len(b)})")
    if not np.allclose(a.sum(), 1.0, atol=1e-7):
        raise ValueError(f"a must sum to 1, got {a.sum()}")
    if not np.allclose(b.sum(), 1.0, atol=1e-7):
        raise ValueError(f"b must sum to 1, got {b.sum()}")

    n, m = len(a), len(b)

    # Flatten cost matrix for linprog
    c = C.flatten()

    # Equality constraints: marginals
    # P @ 1_m = a (row sums)
    A_eq_rows = np.zeros((n, n * m))
    for i in range(n):
        A_eq_rows[i, i * m : (i + 1) * m] = 1

    # P.T @ 1_n = b (column sums)
    A_eq_cols = np.zeros((m, n * m))
    for j in range(m):
        A_eq_cols[j, j::m] = 1

    A_eq = np.vstack([A_eq_rows, A_eq_cols])
    b_eq = np.concatenate([a, b])

    # Bounds: 0 <= P_{ij} <= 1
    bounds = [(0, None) for _ in range(n * m)]

    # Solve
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)

    if not result.success:
        raise RuntimeError(f"Linear programming failed: {result.message}")

    P = result.x.reshape(n, m)

    result_dict = {
        "success": result.success,
        "cost": result.fun,
        "num_iter": result.nit if hasattr(result, "nit") else None,
    }

    return P, result_dict


def emd2(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    method: str = "highs",
) -> float:
    """
    Compute exact optimal transport cost (squared EMD).

    Parameters
    ----------
    a : array-like, shape (n,)
        Source distribution
    b : array-like, shape (m,)
        Target distribution
    C : array-like, shape (n, m)
        Cost matrix
    method : str, default="highs"
        Linear programming solver method

    Returns
    -------
    cost : float
        Optimal transport cost
    """
    _, result_dict = emd(a, b, C, method)
    return result_dict["cost"]


def wasserstein_distance(
    a: np.ndarray,
    b: np.ndarray,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    p: int = 2,
    method: str = "highs",
) -> float:
    """
    Compute p-Wasserstein distance between two empirical distributions.

    Parameters
    ----------
    a : array-like, shape (n,)
        Weights for source samples (must sum to 1)
    b : array-like, shape (m,)
        Weights for target samples (must sum to 1)
    X : array-like, shape (n, d), optional
        Source sample positions. If None, assumes 1D positions [0, 1, ..., n-1]
    Y : array-like, shape (m, d), optional
        Target sample positions. If None, assumes 1D positions [0, 1, ..., m-1]
    p : int, default=2
        Order of the Wasserstein distance
    method : str, default="highs"
        Linear programming solver method

    Returns
    -------
    distance : float
        p-Wasserstein distance
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Default positions if not provided
    if X is None:
        X = np.arange(len(a))[:, None]
    if Y is None:
        Y = np.arange(len(b))[:, None]

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Ensure 2D arrays
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    # Compute cost matrix
    C = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    if p != 2:
        C = C ** (p / 2)

    # Compute OT cost
    cost = emd2(a, b, C, method)

    # Return distance
    if p == 2:
        return np.sqrt(cost)
    else:
        return cost ** (1 / p)
