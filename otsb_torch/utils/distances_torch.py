"""
Distance metrics and cost functions for optimal transport using PyTorch.
"""

import torch
from typing import Optional


def euclidean_cost(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances using PyTorch.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        First set of points
    Y : torch.Tensor, shape (m, d)
        Second set of points

    Returns
    -------
    C : torch.Tensor, shape (n, m)
        Euclidean distance matrix
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)

    if X.ndim == 1:
        X = X.unsqueeze(1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    return torch.sqrt(torch.sum(diff ** 2, dim=2))


def squared_euclidean_cost(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances using PyTorch.

    This is the most common cost function for optimal transport.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        First set of points
    Y : torch.Tensor, shape (m, d)
        Second set of points

    Returns
    -------
    C : torch.Tensor, shape (n, m)
        Squared Euclidean distance matrix
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)

    if X.ndim == 1:
        X = X.unsqueeze(1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    return torch.sum(diff ** 2, dim=2)


def manhattan_cost(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Manhattan (L1) distances using PyTorch.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        First set of points
    Y : torch.Tensor, shape (m, d)
        Second set of points

    Returns
    -------
    C : torch.Tensor, shape (n, m)
        Manhattan distance matrix
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)

    if X.ndim == 1:
        X = X.unsqueeze(1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    return torch.sum(torch.abs(diff), dim=2)


def cosine_cost(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine distances (1 - cosine similarity) using PyTorch.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        First set of points
    Y : torch.Tensor, shape (m, d)
        Second set of points

    Returns
    -------
    C : torch.Tensor, shape (n, m)
        Cosine distance matrix
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)

    if X.ndim == 1:
        X = X.unsqueeze(1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    # Normalize
    X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-16)
    Y_norm = Y / (torch.norm(Y, dim=1, keepdim=True) + 1e-16)

    # Cosine similarity
    similarity = X_norm @ Y_norm.t()

    # Cosine distance
    return 1 - similarity


def cost_matrix(
    X: torch.Tensor,
    Y: torch.Tensor,
    metric: str = "sqeuclidean",
    p: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute cost matrix between two sets of points using PyTorch.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        First set of points
    Y : torch.Tensor, shape (m, d)
        Second set of points
    metric : str, default="sqeuclidean"
        Distance metric to use
    p : float, optional
        Parameter for Minkowski distance

    Returns
    -------
    C : torch.Tensor, shape (n, m)
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
        X = torch.as_tensor(X, dtype=torch.float32)
        Y = torch.as_tensor(Y, dtype=torch.float32)
        if X.ndim == 1:
            X = X.unsqueeze(1)
        if Y.ndim == 1:
            Y = Y.unsqueeze(1)
        diff = X.unsqueeze(1) - Y.unsqueeze(0)
        return torch.sum(torch.abs(diff) ** p, dim=2) ** (1 / p)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def gaussian_kernel(
    X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0
) -> torch.Tensor:
    """
    Compute Gaussian (RBF) kernel matrix using PyTorch.

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        First set of points
    Y : torch.Tensor, shape (m, d)
        Second set of points
    sigma : float, default=1.0
        Kernel bandwidth

    Returns
    -------
    K : torch.Tensor, shape (n, m)
        Kernel matrix
    """
    C = squared_euclidean_cost(X, Y)
    return torch.exp(-C / (2 * sigma ** 2))
