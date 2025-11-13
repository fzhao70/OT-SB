"""Tests for Schrödinger Bridge solver."""

import numpy as np
import pytest
from otsb.solvers.schrodinger_bridge import SchrodingerBridgeSolver


def test_schrodinger_bridge_fit():
    """Test that Schrödinger Bridge can be fitted."""
    np.random.seed(42)
    n = 20

    X0 = np.random.randn(n, 2)
    X1 = np.random.randn(n, 2) + 2

    sb = SchrodingerBridgeSolver(n_steps=20, sigma=1.0, max_iter=50)
    sb.fit(X0, X1)

    # Check that it converged
    assert hasattr(sb, "history_")
    assert sb.history_["num_iter"] <= 50


def test_schrodinger_bridge_sample():
    """Test trajectory sampling from Schrödinger Bridge."""
    np.random.seed(42)
    n = 20

    X0 = np.random.randn(n, 2)
    X1 = np.random.randn(n, 2) + 2

    sb = SchrodingerBridgeSolver(n_steps=20, sigma=1.0, max_iter=50)
    sb.fit(X0, X1)

    # Sample trajectories
    trajectories = sb.sample_trajectory(n_samples=10, random_state=42)

    # Check shape
    assert trajectories.shape == (10, 20, 2)

    # Check that trajectories are finite
    assert np.all(np.isfinite(trajectories))


def test_schrodinger_bridge_transport_plan():
    """Test transport plan from Schrödinger Bridge."""
    np.random.seed(42)
    n = 20

    X0 = np.random.randn(n, 2)
    X1 = np.random.randn(n, 2) + 2

    sb = SchrodingerBridgeSolver(n_steps=20, sigma=1.0, max_iter=50)
    sb.fit(X0, X1)

    # Get transport plan
    P = sb.get_transport_plan()

    # Check shape
    assert P.shape == (n, n)

    # Check that it's a valid coupling (approximately)
    assert np.allclose(P.sum(), 1.0, atol=1e-4)
    assert np.all(P >= -1e-10)


def test_schrodinger_bridge_1d():
    """Test Schrödinger Bridge on 1D distributions."""
    np.random.seed(42)
    n = 30

    X0 = np.random.randn(n)
    X1 = np.random.randn(n) + 3

    sb = SchrodingerBridgeSolver(n_steps=20, sigma=1.0, max_iter=50)
    sb.fit(X0, X1)

    # Sample trajectories
    trajectories = sb.sample_trajectory(n_samples=10, random_state=42)

    # Check shape (should be (10, 20, 1) for 1D)
    assert trajectories.shape == (10, 20, 1)
