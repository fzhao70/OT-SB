"""
Visualization utilities for optimal transport and Schrödinger Bridge.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_transport_plan(
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    threshold: float = 1e-4,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs,
) -> plt.Axes:
    """
    Visualize a transport plan between two point clouds.

    Draws lines between source and target points weighted by the transport plan.
    Only works for 1D or 2D point clouds.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Source points (d must be 1 or 2)
    Y : array-like, shape (m, d)
        Target points (d must be 1 or 2)
    P : array-like, shape (n, m)
        Transport plan matrix
    threshold : float, default=1e-4
        Only plot connections with weight > threshold
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    figsize : tuple, default=(10, 8)
        Figure size if creating new figure
    **kwargs
        Additional arguments passed to plot

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the plot
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    d = X.shape[1]
    if d > 2:
        raise ValueError("Can only visualize 1D or 2D point clouds")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Normalize transport plan for visualization
    P_viz = P / P.max()

    # Plot connections
    for i in range(len(X)):
        for j in range(len(Y)):
            if P_viz[i, j] > threshold:
                alpha = P_viz[i, j]
                if d == 1:
                    ax.plot(
                        [0, 1],
                        [X[i, 0], Y[j, 0]],
                        "k-",
                        alpha=alpha * 0.5,
                        linewidth=alpha * 2,
                    )
                else:
                    ax.plot(
                        [X[i, 0], Y[j, 0]],
                        [X[i, 1], Y[j, 1]],
                        "k-",
                        alpha=alpha * 0.5,
                        linewidth=alpha * 2,
                    )

    # Plot points
    if d == 1:
        ax.scatter([0] * len(X), X[:, 0], c="blue", s=100, label="Source", zorder=5)
        ax.scatter([1] * len(Y), Y[:, 0], c="red", s=100, label="Target", zorder=5)
        ax.set_xlim(-0.2, 1.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
    else:
        ax.scatter(X[:, 0], X[:, 1], c="blue", s=100, label="Source", zorder=5)
        ax.scatter(Y[:, 0], Y[:, 1], c="red", s=100, label="Target", zorder=5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax.legend()
    ax.set_title("Transport Plan")
    ax.grid(True, alpha=0.3)

    return ax


def plot_samples(
    X: np.ndarray,
    Y: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    labels: Optional[Tuple[str, str]] = ("Source", "Target"),
    **kwargs,
) -> plt.Axes:
    """
    Plot two sets of samples.

    Parameters
    ----------
    X : array-like, shape (n, d)
        First set of points (d must be 1 or 2)
    Y : array-like, shape (m, d)
        Second set of points (d must be 1 or 2)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, default=(8, 6)
        Figure size
    labels : tuple of str, optional
        Labels for X and Y
    **kwargs
        Additional arguments passed to scatter

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    d = X.shape[1]
    if d > 2:
        raise ValueError("Can only visualize 1D or 2D point clouds")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if d == 1:
        ax.scatter([0] * len(X), X[:, 0], c="blue", s=50, label=labels[0], alpha=0.6)
        ax.scatter([1] * len(Y), Y[:, 0], c="red", s=50, label=labels[1], alpha=0.6)
        ax.set_xlim(-0.2, 1.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
    else:
        ax.scatter(X[:, 0], X[:, 1], c="blue", s=50, label=labels[0], alpha=0.6)
        ax.scatter(Y[:, 0], Y[:, 1], c="red", s=50, label=labels[1], alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_trajectories(
    trajectories: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    n_plot: Optional[int] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot sampled trajectories from Schrödinger Bridge.

    Parameters
    ----------
    trajectories : array-like, shape (n_samples, n_steps, d)
        Sampled trajectories (d must be 1 or 2)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, default=(10, 8)
        Figure size
    n_plot : int, optional
        Number of trajectories to plot (all if None)
    **kwargs
        Additional arguments passed to plot

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object
    """
    trajectories = np.asarray(trajectories, dtype=np.float64)

    n_samples, n_steps, d = trajectories.shape
    if d > 2:
        raise ValueError("Can only visualize 1D or 2D trajectories")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if n_plot is None:
        n_plot = n_samples
    else:
        n_plot = min(n_plot, n_samples)

    # Plot trajectories
    for i in range(n_plot):
        if d == 1:
            t = np.linspace(0, 1, n_steps)
            ax.plot(t, trajectories[i, :, 0], alpha=0.3, **kwargs)
        else:
            ax.plot(
                trajectories[i, :, 0],
                trajectories[i, :, 1],
                alpha=0.3,
                **kwargs,
            )

    # Mark start and end points
    if d == 1:
        ax.scatter([0] * n_plot, trajectories[:n_plot, 0, 0], c="blue", s=50, zorder=5)
        ax.scatter([1] * n_plot, trajectories[:n_plot, -1, 0], c="red", s=50, zorder=5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
    else:
        ax.scatter(
            trajectories[:n_plot, 0, 0],
            trajectories[:n_plot, 0, 1],
            c="blue",
            s=50,
            label="Start",
            zorder=5,
        )
        ax.scatter(
            trajectories[:n_plot, -1, 0],
            trajectories[:n_plot, -1, 1],
            c="red",
            s=50,
            label="End",
            zorder=5,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

    ax.set_title(f"Schrödinger Bridge Trajectories (n={n_plot})")
    ax.grid(True, alpha=0.3)

    return ax


def plot_cost_matrix(
    C: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    **kwargs,
) -> plt.Axes:
    """
    Visualize a cost matrix.

    Parameters
    ----------
    C : array-like, shape (n, m)
        Cost matrix
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, default=(8, 6)
        Figure size
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object
    """
    C = np.asarray(C, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(C, cmap="viridis", aspect="auto", **kwargs)
    ax.set_xlabel("Target index")
    ax.set_ylabel("Source index")
    ax.set_title("Cost Matrix")
    plt.colorbar(im, ax=ax)

    return ax
