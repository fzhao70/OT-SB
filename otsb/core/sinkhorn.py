"""
Sinkhorn algorithm for entropic regularized optimal transport.

The Sinkhorn algorithm solves the entropic regularized optimal transport problem:
    min_{P ∈ Π(a,b)} <P, C> + ε H(P)
where C is the cost matrix, a and b are source and target distributions,
Π(a,b) is the set of joint distributions with marginals a and b,
and H(P) is the entropy of P.
"""

import numpy as np
from typing import Optional, Tuple


def sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    reg: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
    log: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Solve entropic regularized optimal transport using Sinkhorn algorithm.

    Parameters
    ----------
    a : array-like, shape (n,)
        Source distribution (must sum to 1)
    b : array-like, shape (m,)
        Target distribution (must sum to 1)
    C : array-like, shape (n, m)
        Cost matrix
    reg : float, default=1.0
        Regularization parameter (epsilon)
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-9
        Convergence tolerance
    log : bool, default=False
        If True, return additional log information

    Returns
    -------
    P : array-like, shape (n, m)
        Optimal transport plan
    log_dict : dict (if log=True)
        Dictionary containing convergence information
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    # Check inputs
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("a and b must be 1-dimensional arrays")
    if C.shape != (len(a), len(b)):
        raise ValueError(f"C must have shape ({len(a)}, {len(b)})")
    if not np.allclose(a.sum(), 1.0) or not np.allclose(b.sum(), 1.0):
        raise ValueError("a and b must sum to 1")
    if reg <= 0:
        raise ValueError("reg must be positive")

    # Initialize
    K = np.exp(-C / reg)
    u = np.ones(len(a)) / len(a)
    v = np.ones(len(b)) / len(b)

    errors = []
    for i in range(max_iter):
        u_prev = u.copy()

        # Sinkhorn iterations
        u = a / (K @ v)
        v = b / (K.T @ u)

        # Check convergence
        if i % 10 == 0:
            error = np.linalg.norm(u - u_prev)
            errors.append(error)
            if error < tol:
                break

    # Compute transport plan
    P = u[:, None] * K * v[None, :]

    if log:
        log_dict = {
            "num_iter": i + 1,
            "errors": errors,
            "u": u,
            "v": v,
        }
        return P, log_dict

    return P, {}


def sinkhorn_log(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    reg: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
    log: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Solve entropic regularized OT using log-domain stabilized Sinkhorn.

    This version is numerically more stable than standard Sinkhorn for small
    regularization parameters.

    Parameters
    ----------
    a : array-like, shape (n,)
        Source distribution (must sum to 1)
    b : array-like, shape (m,)
        Target distribution (must sum to 1)
    C : array-like, shape (n, m)
        Cost matrix
    reg : float, default=1.0
        Regularization parameter (epsilon)
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-9
        Convergence tolerance
    log : bool, default=False
        If True, return additional log information

    Returns
    -------
    P : array-like, shape (n, m)
        Optimal transport plan
    log_dict : dict (if log=True)
        Dictionary containing convergence information
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    # Check inputs
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("a and b must be 1-dimensional arrays")
    if C.shape != (len(a), len(b)):
        raise ValueError(f"C must have shape ({len(a)}, {len(b)})")
    if not np.allclose(a.sum(), 1.0) or not np.allclose(b.sum(), 1.0):
        raise ValueError("a and b must sum to 1")
    if reg <= 0:
        raise ValueError("reg must be positive")

    # Initialize in log domain
    log_a = np.log(a + 1e-16)
    log_b = np.log(b + 1e-16)
    log_K = -C / reg

    f = np.zeros(len(a))
    g = np.zeros(len(b))

    errors = []
    for i in range(max_iter):
        f_prev = f.copy()

        # Log-domain Sinkhorn iterations
        f = log_a - np.log(np.exp(log_K + g[None, :]).sum(axis=1) + 1e-16)
        g = log_b - np.log(np.exp(log_K.T + f[None, :]).sum(axis=1) + 1e-16)

        # Check convergence
        if i % 10 == 0:
            error = np.linalg.norm(f - f_prev)
            errors.append(error)
            if error < tol:
                break

    # Compute transport plan
    P = np.exp(f[:, None] + log_K + g[None, :])

    if log:
        log_dict = {
            "num_iter": i + 1,
            "errors": errors,
            "f": f,
            "g": g,
        }
        return P, log_dict

    return P, {}


def sinkhorn_cost(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    reg: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> float:
    """
    Compute the regularized optimal transport cost.

    Parameters
    ----------
    a : array-like, shape (n,)
        Source distribution
    b : array-like, shape (m,)
        Target distribution
    C : array-like, shape (n, m)
        Cost matrix
    reg : float, default=1.0
        Regularization parameter
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-9
        Convergence tolerance

    Returns
    -------
    cost : float
        Regularized optimal transport cost
    """
    P, _ = sinkhorn_log(a, b, C, reg, max_iter, tol)
    return np.sum(P * C)
