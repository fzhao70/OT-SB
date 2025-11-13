"""
Quick script to run all examples and verify they work without installation.
"""

import sys
import os
# Add parent directory to path to import otsb without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import time


def run_example(script_name):
    """Run a single example script."""
    print(f"\n{'=' * 70}")
    print(f"Running: {script_name}")
    print('=' * 70)

    start = time.time()
    try:
        # Run the script using Python
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=60
        )

        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"✓ SUCCESS ({elapsed:.2f}s)")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                print("Output (last 5 lines):")
                for line in lines[-5:]:
                    print(f"  {line}")
            return True
        else:
            print(f"✗ FAILED ({elapsed:.2f}s)")
            print("Error output:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT (>60s)")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def main():
    """Run all example scripts."""
    print("=" * 70)
    print("OT-SB Examples Test Suite")
    print("Testing that all examples run without installation...")
    print("=" * 70)

    # Note: We'll just test imports since matplotlib.show() blocks
    examples = [
        'example_sinkhorn.py',
        'example_exact_ot.py',
        'example_schrodinger_bridge.py',
        'simple_1d_ot.py',
        'comparison_methods.py',
        'gaussian_mixture_transport.py',
    ]

    print("\nNote: Examples are configured to display plots.")
    print("Close plot windows to continue to the next example.")
    print("\nTo run without displaying plots, modify the examples")
    print("by commenting out plt.show() calls.")

    results = {}
    for example in examples:
        success = run_example(example)
        results[example] = success

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print('=' * 70)

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    for example, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {example}")

    print(f"\nTotal: {total} examples")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 All examples passed!")
        return 0
    else:
        print(f"\n⚠ {failed} example(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
