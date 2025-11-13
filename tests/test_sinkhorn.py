"""Tests for Sinkhorn algorithm."""

import numpy as np
import pytest
from otsb.core.sinkhorn import sinkhorn, sinkhorn_log, sinkhorn_cost


def test_sinkhorn_convergence():
    """Test that Sinkhorn converges to a valid transport plan."""
    np.random.seed(42)
    n, m = 10, 15

    a = np.ones(n) / n
    b = np.ones(m) / m
    C = np.random.rand(n, m)

    P, log_dict = sinkhorn(a, b, C, reg=0.1, log=True)

    # Check marginals
    assert np.allclose(P.sum(axis=1), a, atol=1e-6)
    assert np.allclose(P.sum(axis=0), b, atol=1e-6)

    # Check convergence
    assert log_dict["num_iter"] < 1000


def test_sinkhorn_log_stability():
    """Test log-domain Sinkhorn for numerical stability."""
    np.random.seed(42)
    n, m = 10, 15

    a = np.ones(n) / n
    b = np.ones(m) / m
    C = np.random.rand(n, m)

    P, log_dict = sinkhorn_log(a, b, C, reg=0.01, log=True)

    # Check marginals
    assert np.allclose(P.sum(axis=1), a, atol=1e-5)
    assert np.allclose(P.sum(axis=0), b, atol=1e-5)


def test_sinkhorn_cost():
    """Test Sinkhorn cost computation."""
    np.random.seed(42)
    n, m = 10, 15

    a = np.ones(n) / n
    b = np.ones(m) / m
    C = np.random.rand(n, m)

    cost = sinkhorn_cost(a, b, C, reg=0.1)
    assert cost > 0
    assert np.isfinite(cost)


def test_sinkhorn_input_validation():
    """Test input validation."""
    a = np.ones(10) / 10
    b = np.ones(15) / 15
    C = np.random.rand(10, 15)

    # Test invalid regularization
    with pytest.raises(ValueError):
        sinkhorn(a, b, C, reg=-0.1)

    # Test invalid distributions (not summing to 1)
    with pytest.raises(ValueError):
        sinkhorn(np.ones(10), b, C, reg=0.1)
