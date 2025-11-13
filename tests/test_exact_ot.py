"""Tests for exact optimal transport."""

import numpy as np
import pytest
from otsb.core.exact_ot import emd, emd2, wasserstein_distance


def test_emd_convergence():
    """Test that EMD produces valid transport plan."""
    np.random.seed(42)
    n, m = 10, 10

    a = np.ones(n) / n
    b = np.ones(m) / m
    C = np.random.rand(n, m)

    P, result = emd(a, b, C)

    # Check marginals
    assert np.allclose(P.sum(axis=1), a, atol=1e-6)
    assert np.allclose(P.sum(axis=0), b, atol=1e-6)

    # Check non-negativity
    assert np.all(P >= -1e-10)

    # Check success
    assert result["success"]


def test_emd2():
    """Test EMD cost computation."""
    np.random.seed(42)
    n, m = 10, 10

    a = np.ones(n) / n
    b = np.ones(m) / m
    C = np.random.rand(n, m)

    cost = emd2(a, b, C)
    assert cost > 0
    assert np.isfinite(cost)


def test_wasserstein_distance_1d():
    """Test Wasserstein distance on 1D distributions."""
    # Two identical uniform distributions
    a = np.ones(10) / 10
    b = np.ones(10) / 10
    X = np.arange(10)
    Y = np.arange(10)

    W = wasserstein_distance(a, b, X, Y, p=2)
    assert np.allclose(W, 0, atol=1e-6)


def test_wasserstein_distance_2d():
    """Test Wasserstein distance on 2D distributions."""
    np.random.seed(42)
    n = 20

    a = np.ones(n) / n
    b = np.ones(n) / n
    X = np.random.randn(n, 2)
    Y = np.random.randn(n, 2) + 1

    W = wasserstein_distance(a, b, X, Y, p=2)
    assert W > 0
    assert np.isfinite(W)
