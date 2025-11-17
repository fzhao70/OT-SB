"""
Sinkhorn algorithm for entropic regularized optimal transport using PyTorch.

Provides GPU-accelerated implementations with automatic differentiation support.
"""

import torch
from typing import Optional, Tuple, Dict


def sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    reg: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
    log: bool = False,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Solve entropic regularized optimal transport using Sinkhorn algorithm (PyTorch).

    Parameters
    ----------
    a : torch.Tensor, shape (n,)
        Source distribution (must sum to 1)
    b : torch.Tensor, shape (m,)
        Target distribution (must sum to 1)
    C : torch.Tensor, shape (n, m)
        Cost matrix
    reg : float, default=1.0
        Regularization parameter (epsilon)
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-9
        Convergence tolerance
    log : bool, default=False
        If True, return additional log information
    device : str, optional
        Device to run on ('cpu', 'cuda', 'cuda:0', etc.)

    Returns
    -------
    P : torch.Tensor, shape (n, m)
        Optimal transport plan
    log_dict : dict
        Dictionary containing convergence information (if log=True)
    """
    if device is None:
        device = a.device if isinstance(a, torch.Tensor) else 'cpu'

    # Convert to tensors if needed
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    b = torch.as_tensor(b, dtype=torch.float32, device=device)
    C = torch.as_tensor(C, dtype=torch.float32, device=device)

    # Check inputs
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("a and b must be 1-dimensional tensors")
    if C.shape != (len(a), len(b)):
        raise ValueError(f"C must have shape ({len(a)}, {len(b)})")
    if not torch.allclose(a.sum(), torch.tensor(1.0, device=device), atol=1e-5):
        raise ValueError("a must sum to 1")
    if not torch.allclose(b.sum(), torch.tensor(1.0, device=device), atol=1e-5):
        raise ValueError("b must sum to 1")
    if reg <= 0:
        raise ValueError("reg must be positive")

    # Initialize
    K = torch.exp(-C / reg)
    u = torch.ones(len(a), device=device) / len(a)
    v = torch.ones(len(b), device=device) / len(b)

    errors = []
    for i in range(max_iter):
        u_prev = u.clone()

        # Sinkhorn iterations
        u = a / (K @ v)
        v = b / (K.t() @ u)

        # Check convergence
        if i % 10 == 0:
            error = torch.norm(u - u_prev).item()
            errors.append(error)
            if error < tol:
                break

    # Compute transport plan
    P = u.unsqueeze(1) * K * v.unsqueeze(0)

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
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    reg: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
    log: bool = False,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Solve entropic regularized OT using log-domain stabilized Sinkhorn (PyTorch).

    This version is numerically more stable than standard Sinkhorn for small
    regularization parameters.

    Parameters
    ----------
    a : torch.Tensor, shape (n,)
        Source distribution (must sum to 1)
    b : torch.Tensor, shape (m,)
        Target distribution (must sum to 1)
    C : torch.Tensor, shape (n, m)
        Cost matrix
    reg : float, default=1.0
        Regularization parameter (epsilon)
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-9
        Convergence tolerance
    log : bool, default=False
        If True, return additional log information
    device : str, optional
        Device to run on

    Returns
    -------
    P : torch.Tensor, shape (n, m)
        Optimal transport plan
    log_dict : dict
        Dictionary containing convergence information (if log=True)
    """
    if device is None:
        device = a.device if isinstance(a, torch.Tensor) else 'cpu'

    # Convert to tensors
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    b = torch.as_tensor(b, dtype=torch.float32, device=device)
    C = torch.as_tensor(C, dtype=torch.float32, device=device)

    # Check inputs
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("a and b must be 1-dimensional tensors")
    if C.shape != (len(a), len(b)):
        raise ValueError(f"C must have shape ({len(a)}, {len(b)})")
    if not torch.allclose(a.sum(), torch.tensor(1.0, device=device), atol=1e-5):
        raise ValueError("a must sum to 1")
    if not torch.allclose(b.sum(), torch.tensor(1.0, device=device), atol=1e-5):
        raise ValueError("b must sum to 1")
    if reg <= 0:
        raise ValueError("reg must be positive")

    # Initialize in log domain
    log_a = torch.log(a + 1e-16)
    log_b = torch.log(b + 1e-16)
    log_K = -C / reg

    f = torch.zeros(len(a), device=device)
    g = torch.zeros(len(b), device=device)

    errors = []
    for i in range(max_iter):
        f_prev = f.clone()

        # Log-domain Sinkhorn iterations
        f = log_a - torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(log_K.t() + f.unsqueeze(0), dim=1)

        # Check convergence
        if i % 10 == 0:
            error = torch.norm(f - f_prev).item()
            errors.append(error)
            if error < tol:
                break

    # Compute transport plan
    P = torch.exp(f.unsqueeze(1) + log_K + g.unsqueeze(0))

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
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    reg: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
    device: Optional[str] = None,
) -> float:
    """
    Compute the regularized optimal transport cost using PyTorch.

    Parameters
    ----------
    a : torch.Tensor, shape (n,)
        Source distribution
    b : torch.Tensor, shape (m,)
        Target distribution
    C : torch.Tensor, shape (n, m)
        Cost matrix
    reg : float, default=1.0
        Regularization parameter
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-9
        Convergence tolerance
    device : str, optional
        Device to run on

    Returns
    -------
    cost : float
        Regularized optimal transport cost
    """
    P, _ = sinkhorn_log(a, b, C, reg, max_iter, tol, device=device)
    return (P * C).sum().item()
