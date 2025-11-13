"""
Schrödinger Bridge Problem (SBP) solver using PyTorch.

GPU-accelerated implementation with automatic differentiation support.
"""

import torch
from typing import Optional, Dict


class SchrodingerBridgeSolver:
    """
    Solver for the Schrödinger Bridge Problem using PyTorch.

    The Schrödinger Bridge finds the most likely stochastic bridge between
    two probability distributions, given a reference diffusion process.

    Parameters
    ----------
    n_steps : int, default=100
        Number of time discretization steps
    sigma : float, default=1.0
        Diffusion coefficient
    max_iter : int, default=100
        Maximum number of IPF iterations
    tol : float, default=1e-6
        Convergence tolerance
    device : str, default='cpu'
        Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
    """

    def __init__(
        self,
        n_steps: int = 100,
        sigma: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        device: str = 'cpu',
    ):
        self.n_steps = n_steps
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.history_ = []

    def fit(
        self,
        X0: torch.Tensor,
        X1: torch.Tensor,
        weights0: Optional[torch.Tensor] = None,
        weights1: Optional[torch.Tensor] = None,
    ) -> "SchrodingerBridgeSolver":
        """
        Solve the Schrödinger Bridge between source and target samples.

        Parameters
        ----------
        X0 : torch.Tensor, shape (n0, d)
            Source samples
        X1 : torch.Tensor, shape (n1, d)
            Target samples
        weights0 : torch.Tensor, shape (n0,), optional
            Weights for source samples (uniform if None)
        weights1 : torch.Tensor, shape (n1,), optional
            Weights for target samples (uniform if None)

        Returns
        -------
        self : SchrodingerBridgeSolver
            Fitted solver
        """
        X0 = torch.as_tensor(X0, dtype=torch.float32, device=self.device)
        X1 = torch.as_tensor(X1, dtype=torch.float32, device=self.device)

        if X0.ndim == 1:
            X0 = X0.unsqueeze(1)
        if X1.ndim == 1:
            X1 = X1.unsqueeze(1)

        n0, d = X0.shape
        n1, _ = X1.shape

        if weights0 is None:
            weights0 = torch.ones(n0, device=self.device) / n0
        else:
            weights0 = torch.as_tensor(weights0, dtype=torch.float32, device=self.device)
            weights0 = weights0 / weights0.sum()

        if weights1 is None:
            weights1 = torch.ones(n1, device=self.device) / n1
        else:
            weights1 = torch.as_tensor(weights1, dtype=torch.float32, device=self.device)
            weights1 = weights1 / weights1.sum()

        self.X0_ = X0
        self.X1_ = X1
        self.weights0_ = weights0
        self.weights1_ = weights1
        self.d_ = d

        # Time discretization
        self.t_ = torch.linspace(0, 1, self.n_steps, device=self.device)
        self.dt_ = 1.0 / (self.n_steps - 1)

        # Initialize potentials
        self.f_ = torch.zeros((self.n_steps, n0), device=self.device)
        self.g_ = torch.zeros((self.n_steps, n1), device=self.device)

        # Run IPF
        self._iterative_proportional_fitting()

        return self

    def _gaussian_kernel(
        self, X: torch.Tensor, Y: torch.Tensor, t: float
    ) -> torch.Tensor:
        """
        Compute Gaussian transition kernel.

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Source points
        Y : torch.Tensor, shape (m, d)
            Target points
        t : float
            Time interval

        Returns
        -------
        K : torch.Tensor, shape (n, m)
            Kernel matrix
        """
        variance = 2 * self.sigma ** 2 * t
        if variance < 1e-10:
            # Handle t=0 case
            if len(X) == len(Y):
                return torch.eye(len(X), device=self.device)
            else:
                return torch.zeros(len(X), len(Y), device=self.device)

        diff = X.unsqueeze(1) - Y.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=2)
        K = torch.exp(-dist_sq / variance)
        # Normalize
        K = K / ((2 * torch.pi * variance) ** (self.d_ / 2))
        return K

    def _iterative_proportional_fitting(self):
        """
        Run the Iterative Proportional Fitting algorithm.
        """
        errors = []

        for iteration in range(self.max_iter):
            f_old = self.f_.clone()

            # Forward pass: update f
            for i in range(self.n_steps - 1):
                t_step = self.t_[i + 1] - self.t_[i]
                K = self._gaussian_kernel(self.X0_, self.X1_, t_step.item())

                # Compute forward potential
                exp_g = torch.exp(self.g_[i + 1])
                denominator = K @ (self.weights1_ * exp_g)
                self.f_[i] = -torch.log(denominator + 1e-16)

            # Backward pass: update g
            for i in range(self.n_steps - 1, 0, -1):
                t_step = self.t_[i] - self.t_[i - 1]
                K = self._gaussian_kernel(self.X1_, self.X0_, t_step.item())

                # Compute backward potential
                exp_f = torch.exp(self.f_[i - 1])
                denominator = K @ (self.weights0_ * exp_f)
                self.g_[i] = -torch.log(denominator + 1e-16)

            # Check convergence
            error = torch.norm(self.f_ - f_old) / (torch.norm(f_old) + 1e-16)
            errors.append(error.item())

            if error < self.tol:
                break

        self.history_ = {
            "num_iter": iteration + 1,
            "errors": errors,
        }

    def sample_trajectory(
        self, n_samples: int = 100, random_state: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample trajectories from the Schrödinger Bridge.

        Parameters
        ----------
        n_samples : int, default=100
            Number of trajectories to sample
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        trajectories : torch.Tensor, shape (n_samples, n_steps, d)
            Sampled trajectories
        """
        if not hasattr(self, "X0_"):
            raise ValueError("Model must be fitted before sampling")

        if random_state is not None:
            torch.manual_seed(random_state)

        trajectories = torch.zeros(
            (n_samples, self.n_steps, self.d_), device=self.device
        )

        # Sample initial points from X0
        idx0 = torch.multinomial(self.weights0_, n_samples, replacement=True)
        trajectories[:, 0, :] = self.X0_[idx0]

        # Forward sampling
        for i in range(self.n_steps - 1):
            t_step = self.t_[i + 1] - self.t_[i]

            # Compute transition probabilities
            current_pos = trajectories[:, i, :]
            K = self._gaussian_kernel(current_pos, self.X1_, t_step.item())

            # Apply potentials
            exp_g = torch.exp(self.g_[i + 1])
            weights = K * (self.weights1_ * exp_g).unsqueeze(0)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-16)

            # Sample next positions
            for j in range(n_samples):
                idx = torch.multinomial(weights[j], 1).item()
                mean = self.X1_[idx]
                cov = self.sigma ** 2 * t_step.item() * torch.eye(
                    self.d_, device=self.device
                )
                trajectories[j, i + 1, :] = (
                    torch.randn(self.d_, device=self.device) @ cov.sqrt() + mean
                )

        return trajectories

    def get_transport_plan(self) -> torch.Tensor:
        """
        Get the transport plan between source and target.

        Returns
        -------
        P : torch.Tensor, shape (n0, n1)
            Transport plan matrix
        """
        if not hasattr(self, "X0_"):
            raise ValueError("Model must be fitted before getting transport plan")

        K = self._gaussian_kernel(self.X0_, self.X1_, 1.0)
        exp_f = torch.exp(self.f_[0])
        exp_g = torch.exp(self.g_[-1])

        P = (
            self.weights0_.unsqueeze(1)
            * exp_f.unsqueeze(1)
            * K
            * exp_g.unsqueeze(0)
            * self.weights1_.unsqueeze(0)
        )
        P = P / P.sum()

        return P
