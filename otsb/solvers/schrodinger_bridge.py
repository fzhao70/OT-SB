"""
Schrödinger Bridge Problem (SBP) solver.

The Schrödinger Bridge is a dynamic generalization of optimal transport that
finds the most likely stochastic process connecting two distributions, given
a prior reference process (typically a Brownian motion).

The SBP can be solved iteratively using the Iterative Proportional Fitting (IPF)
procedure, which alternates between forward and backward passes.
"""

import numpy as np
from typing import Optional, Callable, Tuple, Dict
from scipy.stats import multivariate_normal


class SchrodingerBridgeSolver:
    """
    Solver for the Schrödinger Bridge Problem.

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
    """

    def __init__(
        self,
        n_steps: int = 100,
        sigma: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.n_steps = n_steps
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.history_ = []

    def fit(
        self,
        X0: np.ndarray,
        X1: np.ndarray,
        weights0: Optional[np.ndarray] = None,
        weights1: Optional[np.ndarray] = None,
    ) -> "SchrodingerBridgeSolver":
        """
        Solve the Schrödinger Bridge between source and target samples.

        Parameters
        ----------
        X0 : array-like, shape (n0, d)
            Source samples
        X1 : array-like, shape (n1, d)
            Target samples
        weights0 : array-like, shape (n0,), optional
            Weights for source samples (uniform if None)
        weights1 : array-like, shape (n1,), optional
            Weights for target samples (uniform if None)

        Returns
        -------
        self : SchrodingerBridgeSolver
            Fitted solver
        """
        X0 = np.asarray(X0, dtype=np.float64)
        X1 = np.asarray(X1, dtype=np.float64)

        if X0.ndim == 1:
            X0 = X0[:, None]
        if X1.ndim == 1:
            X1 = X1[:, None]

        n0, d = X0.shape
        n1, _ = X1.shape

        if weights0 is None:
            weights0 = np.ones(n0) / n0
        else:
            weights0 = np.asarray(weights0, dtype=np.float64)
            weights0 = weights0 / weights0.sum()

        if weights1 is None:
            weights1 = np.ones(n1) / n1
        else:
            weights1 = np.asarray(weights1, dtype=np.float64)
            weights1 = weights1 / weights1.sum()

        self.X0_ = X0
        self.X1_ = X1
        self.weights0_ = weights0
        self.weights1_ = weights1
        self.d_ = d

        # Time discretization
        self.t_ = np.linspace(0, 1, self.n_steps)
        self.dt_ = 1.0 / (self.n_steps - 1)

        # Initialize potentials
        self.f_ = np.zeros((self.n_steps, n0))
        self.g_ = np.zeros((self.n_steps, n1))

        # Run IPF
        self._iterative_proportional_fitting()

        return self

    def _gaussian_kernel(
        self, X: np.ndarray, Y: np.ndarray, t: float
    ) -> np.ndarray:
        """
        Compute Gaussian transition kernel.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Source points
        Y : array-like, shape (m, d)
            Target points
        t : float
            Time interval

        Returns
        -------
        K : array-like, shape (n, m)
            Kernel matrix
        """
        # K(x, y, t) = exp(-||x - y||^2 / (2 * sigma^2 * t))
        variance = 2 * self.sigma**2 * t
        if variance < 1e-10:
            # Handle t=0 case
            return np.eye(len(X)) if len(X) == len(Y) else np.zeros((len(X), len(Y)))

        diff = X[:, None, :] - Y[None, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        K = np.exp(-dist_sq / variance)
        # Normalize
        K = K / (2 * np.pi * variance) ** (self.d_ / 2)
        return K

    def _iterative_proportional_fitting(self):
        """
        Run the Iterative Proportional Fitting algorithm.
        """
        errors = []

        for iteration in range(self.max_iter):
            f_old = self.f_.copy()

            # Forward pass: update f
            for i in range(self.n_steps - 1):
                t_step = self.t_[i + 1] - self.t_[i]
                K = self._gaussian_kernel(self.X0_, self.X1_, t_step)

                # Compute forward potential
                exp_g = np.exp(self.g_[i + 1])
                denominator = K @ (self.weights1_ * exp_g)
                self.f_[i] = -np.log(denominator + 1e-16)

            # Backward pass: update g
            for i in range(self.n_steps - 1, 0, -1):
                t_step = self.t_[i] - self.t_[i - 1]
                K = self._gaussian_kernel(self.X1_, self.X0_, t_step)

                # Compute backward potential
                exp_f = np.exp(self.f_[i - 1])
                denominator = K @ (self.weights0_ * exp_f)
                self.g_[i] = -np.log(denominator + 1e-16)

            # Check convergence
            error = np.linalg.norm(self.f_ - f_old) / (np.linalg.norm(f_old) + 1e-16)
            errors.append(error)

            if error < self.tol:
                break

        self.history_ = {
            "num_iter": iteration + 1,
            "errors": errors,
        }

    def sample_trajectory(
        self, n_samples: int = 100, random_state: Optional[int] = None
    ) -> np.ndarray:
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
        trajectories : array-like, shape (n_samples, n_steps, d)
            Sampled trajectories
        """
        if not hasattr(self, "X0_"):
            raise ValueError("Model must be fitted before sampling")

        if random_state is not None:
            np.random.seed(random_state)

        trajectories = np.zeros((n_samples, self.n_steps, self.d_))

        # Sample initial points from X0
        idx0 = np.random.choice(len(self.X0_), size=n_samples, p=self.weights0_)
        trajectories[:, 0, :] = self.X0_[idx0]

        # Forward sampling
        for i in range(self.n_steps - 1):
            t_step = self.t_[i + 1] - self.t_[i]

            # Compute transition probabilities
            current_pos = trajectories[:, i, :]
            K = self._gaussian_kernel(current_pos, self.X1_, t_step)

            # Apply potentials
            exp_g = np.exp(self.g_[i + 1])
            weights = K * (self.weights1_ * exp_g)[None, :]
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-16)

            # Sample next positions
            for j in range(n_samples):
                idx = np.random.choice(len(self.X1_), p=weights[j])
                mean = self.X1_[idx]
                cov = self.sigma**2 * t_step * np.eye(self.d_)
                trajectories[j, i + 1, :] = multivariate_normal.rvs(mean=mean, cov=cov)

        return trajectories

    def get_transport_plan(self) -> np.ndarray:
        """
        Get the transport plan between source and target.

        Returns
        -------
        P : array-like, shape (n0, n1)
            Transport plan matrix
        """
        if not hasattr(self, "X0_"):
            raise ValueError("Model must be fitted before getting transport plan")

        K = self._gaussian_kernel(self.X0_, self.X1_, 1.0)
        exp_f = np.exp(self.f_[0])
        exp_g = np.exp(self.g_[-1])

        P = (
            self.weights0_[:, None]
            * exp_f[:, None]
            * K
            * exp_g[None, :]
            * self.weights1_[None, :]
        )
        P = P / P.sum()

        return P
