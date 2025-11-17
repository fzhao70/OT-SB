"""
Exact optimal transport solvers using PyTorch.

Note: For exact OT, we use a network simplex approach via conversion to numpy
or use the POT library if available. For differentiable OT, use Sinkhorn instead.
"""

import torch
from typing import Tuple, Optional
import warnings


def emd(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Solve exact optimal transport problem using Earth Mover's Distance (PyTorch).

    Note: This function uses numpy backend for LP solving, then converts back to PyTorch.
    Gradients are not supported. For differentiable OT, use Sinkhorn algorithm.

    Parameters
    ----------
    a : torch.Tensor, shape (n,)
        Source distribution (must sum to 1)
    b : torch.Tensor, shape (m,)
        Target distribution (must sum to 1)
    C : torch.Tensor, shape (n, m)
        Cost matrix
    device : str, optional
        Device to return result on

    Returns
    -------
    P : torch.Tensor, shape (n, m)
        Optimal transport plan
    result_dict : dict
        Dictionary containing solver information
    """
    if device is None:
        device = a.device if isinstance(a, torch.Tensor) else 'cpu'

    # Convert to tensors
    a = torch.as_tensor(a, dtype=torch.float32, device='cpu')
    b = torch.as_tensor(b, dtype=torch.float32, device='cpu')
    C = torch.as_tensor(C, dtype=torch.float32, device='cpu')

    # Try to use POT library if available
    try:
        import ot as pot
        P_np = pot.emd(a.numpy(), b.numpy(), C.numpy())
        P = torch.from_numpy(P_np).float().to(device)
        cost = (P * C.to(device)).sum().item()

        result_dict = {
            "success": True,
            "cost": cost,
            "method": "POT",
        }
        return P, result_dict

    except ImportError:
        warnings.warn(
            "POT library not available. Falling back to scipy-based solver. "
            "Install POT for better performance: pip install POT"
        )

    # Fallback to scipy
    from scipy.optimize import linprog
    import numpy as np

    a_np = a.numpy()
    b_np = b.numpy()
    C_np = C.numpy()

    n, m = C_np.shape

    # Flatten cost matrix for linprog
    c = C_np.flatten()

    # Equality constraints: marginals
    A_eq_rows = np.zeros((n, n * m))
    for i in range(n):
        A_eq_rows[i, i * m : (i + 1) * m] = 1

    A_eq_cols = np.zeros((m, n * m))
    for j in range(m):
        A_eq_cols[j, j::m] = 1

    A_eq = np.vstack([A_eq_rows, A_eq_cols])
    b_eq = np.concatenate([a_np, b_np])

    # Bounds
    bounds = [(0, None) for _ in range(n * m)]

    # Solve
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not result.success:
        raise RuntimeError(f"Linear programming failed: {result.message}")

    P_np = result.x.reshape(n, m)
    P = torch.from_numpy(P_np).float().to(device)

    result_dict = {
        "success": result.success,
        "cost": result.fun,
        "method": "scipy",
    }

    return P, result_dict


def wasserstein_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    X: Optional[torch.Tensor] = None,
    Y: Optional[torch.Tensor] = None,
    p: int = 2,
    device: Optional[str] = None,
) -> float:
    """
    Compute p-Wasserstein distance between two empirical distributions (PyTorch).

    Parameters
    ----------
    a : torch.Tensor, shape (n,)
        Weights for source samples (must sum to 1)
    b : torch.Tensor, shape (m,)
        Weights for target samples (must sum to 1)
    X : torch.Tensor, shape (n, d), optional
        Source sample positions
    Y : torch.Tensor, shape (m, d), optional
        Target sample positions
    p : int, default=2
        Order of the Wasserstein distance
    device : str, optional
        Device to run on

    Returns
    -------
    distance : float
        p-Wasserstein distance
    """
    if device is None:
        device = a.device if isinstance(a, torch.Tensor) else 'cpu'

    # Convert to tensors
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    b = torch.as_tensor(b, dtype=torch.float32, device=device)

    # Default positions if not provided
    if X is None:
        X = torch.arange(len(a), dtype=torch.float32, device=device).unsqueeze(1)
    if Y is None:
        Y = torch.arange(len(b), dtype=torch.float32, device=device).unsqueeze(1)

    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)

    # Ensure 2D arrays
    if X.ndim == 1:
        X = X.unsqueeze(1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    # Compute cost matrix
    C = torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=2)
    if p != 2:
        C = C ** (p / 2)

    # Try POT first
    try:
        import ot as pot
        cost = pot.emd2(a.cpu().numpy(), b.cpu().numpy(), C.cpu().numpy())
        if p == 2:
            return float(cost ** 0.5)
        else:
            return float(cost ** (1 / p))
    except ImportError:
        pass

    # Fallback to our emd implementation
    _, result = emd(a, b, C, device=device)
    cost = result["cost"]

    # Return distance
    if p == 2:
        return float(cost ** 0.5)
    else:
        return float(cost ** (1 / p))
