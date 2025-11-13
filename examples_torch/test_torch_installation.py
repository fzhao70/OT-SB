"""
Simple test to verify PyTorch implementation works correctly.
Run this after installing: pip install torch
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_torch_import():
    """Test that torch modules can be imported."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed. Install with: pip install torch")
        return False

    return True


def test_otsb_torch_import():
    """Test that otsb_torch can be imported."""
    try:
        from otsb_torch import sinkhorn, SchrodingerBridgeSolver
        from otsb_torch.utils import squared_euclidean_cost
        print("✓ otsb_torch imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import otsb_torch: {e}")
        return False


def test_basic_sinkhorn():
    """Test basic Sinkhorn functionality."""
    try:
        import torch
        from otsb_torch import sinkhorn, squared_euclidean_cost

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Small problem
        n = 10
        X = torch.randn(n, 2, device=device)
        Y = torch.randn(n, 2, device=device) + 1

        a = torch.ones(n, device=device) / n
        b = torch.ones(n, device=device) / n

        C = squared_euclidean_cost(X, Y)
        P, log_dict = sinkhorn(a, b, C, reg=0.1, max_iter=100, log=True, device=device)

        # Verify output
        assert P.shape == (n, n), f"Wrong shape: {P.shape}"
        assert torch.allclose(P.sum(), torch.tensor(1.0, device=device), atol=1e-3), "Transport plan doesn't sum to 1"
        assert log_dict['num_iter'] > 0, "No iterations performed"

        print(f"✓ Sinkhorn works (converged in {log_dict['num_iter']} iterations)")
        print(f"  Transport cost: {(P * C).sum().item():.4f}")
        print(f"  Device: {device}")
        return True

    except Exception as e:
        print(f"✗ Sinkhorn test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_schrodinger_bridge():
    """Test basic Schrödinger Bridge functionality."""
    try:
        import torch
        from otsb_torch import SchrodingerBridgeSolver

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Small problem
        n = 20
        X0 = torch.randn(n, 2, device=device) * 0.3
        X1 = torch.randn(n, 2, device=device) * 0.3 + 2

        sb = SchrodingerBridgeSolver(
            n_steps=20,
            sigma=0.5,
            max_iter=30,
            tol=1e-5,
            device=device
        )
        sb.fit(X0, X1)

        # Test trajectory sampling
        trajectories = sb.sample_trajectory(n_samples=10, random_state=42)

        assert trajectories.shape == (10, 20, 2), f"Wrong shape: {trajectories.shape}"
        assert sb.history_['num_iter'] > 0, "No iterations performed"

        print(f"✓ Schrödinger Bridge works (converged in {sb.history_['num_iter']} iterations)")
        print(f"  Sampled {trajectories.shape[0]} trajectories")
        print(f"  Device: {device}")
        return True

    except Exception as e:
        print(f"✗ Schrödinger Bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("Testing PyTorch Implementation of OT-SB")
    print("=" * 70)

    results = []

    # Test 1: PyTorch installation
    print("\n1. Checking PyTorch installation...")
    results.append(("PyTorch Install", test_torch_import()))

    if not results[-1][1]:
        print("\nSkipping remaining tests (PyTorch not available)")
        return 1

    # Test 2: Import otsb_torch
    print("\n2. Testing otsb_torch imports...")
    results.append(("otsb_torch Import", test_otsb_torch_import()))

    if not results[-1][1]:
        print("\nSkipping remaining tests (imports failed)")
        return 1

    # Test 3: Basic Sinkhorn
    print("\n3. Testing Sinkhorn algorithm...")
    results.append(("Sinkhorn", test_basic_sinkhorn()))

    # Test 4: Basic Schrödinger Bridge
    print("\n4. Testing Schrödinger Bridge...")
    results.append(("Schrödinger Bridge", test_basic_schrodinger_bridge()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! PyTorch implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
