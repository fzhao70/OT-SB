"""
Distance metrics and cost functions for optimal transport.
"""

import numpy as np
from typing import Optional


def euclidean_cost(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.

    Parameters
    ----------
    X : array-like, shape (n, d)
        First set of points
    Y : array-like, shape (m, d)
        Second set of points

    Returns
    -------
    C : array-like, shape (n, m)
        Euclidean distance matrix
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def squared_euclidean_cost(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances.

    This is the most common cost function for optimal transport.

    Parameters
    ----------
    X : array-like, shape (n, d)
        First set of points
    Y : array-like, shape (m, d)
        Second set of points

    Returns
    -------
    C : array-like, shape (n, m)
        Squared Euclidean distance matrix
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    diff = X[:, None, :] - Y[None, :, :]
    return np.sum(diff**2, axis=2)


def manhattan_cost(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Manhattan (L1) distances.

    Parameters
    ----------
    X : array-like, shape (n, d)
        First set of points
    Y : array-like, shape (m, d)
        Second set of points

    Returns
    -------
    C : array-like, shape (n, m)
        Manhattan distance matrix
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    diff = X[:, None, :] - Y[None, :, :]
    return np.sum(np.abs(diff), axis=2)


def cosine_cost(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distances (1 - cosine similarity).

    Parameters
    ----------
    X : array-like, shape (n, d)
        First set of points
    Y : array-like, shape (m, d)
        Second set of points

    Returns
    -------
    C : array-like, shape (n, m)
        Cosine distance matrix
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    # Normalize
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-16)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16)

    # Cosine similarity
    similarity = X_norm @ Y_norm.T

    # Cosine distance
    return 1 - similarity


def cost_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "sqeuclidean",
    p: Optional[float] = None,
) -> np.ndarray:
    """
    Compute cost matrix between two sets of points.

    Parameters
    ----------
    X : array-like, shape (n, d)
        First set of points
    Y : array-like, shape (m, d)
        Second set of points
    metric : str, default="sqeuclidean"
        Distance metric to use. Options:
        - "sqeuclidean": squared Euclidean distance
        - "euclidean": Euclidean distance
        - "manhattan" or "cityblock": L1 distance
        - "cosine": cosine distance
        - "minkowski": Minkowski distance (requires p parameter)
    p : float, optional
        Parameter for Minkowski distance

    Returns
    -------
    C : array-like, shape (n, m)
        Cost matrix
    """
    if metric == "sqeuclidean":
        return squared_euclidean_cost(X, Y)
    elif metric == "euclidean":
        return euclidean_cost(X, Y)
    elif metric in ["manhattan", "cityblock"]:
        return manhattan_cost(X, Y)
    elif metric == "cosine":
        return cosine_cost(X, Y)
    elif metric == "minkowski":
        if p is None:
            raise ValueError("p parameter required for Minkowski distance")
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]
        diff = X[:, None, :] - Y[None, :, :]
        return np.sum(np.abs(diff) ** p, axis=2) ** (1 / p)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def gaussian_kernel(
    X: np.ndarray, Y: np.ndarray, sigma: float = 1.0
) -> np.ndarray:
    """
    Compute Gaussian (RBF) kernel matrix.

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Parameters
    ----------
    X : array-like, shape (n, d)
        First set of points
    Y : array-like, shape (m, d)
        Second set of points
    sigma : float, default=1.0
        Kernel bandwidth

    Returns
    -------
    K : array-like, shape (n, m)
        Kernel matrix
    """
    C = squared_euclidean_cost(X, Y)
    return np.exp(-C / (2 * sigma**2))
