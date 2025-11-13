"""
Example: Using Schrödinger Bridge solver.
This example can run directly after cloning without installation.
"""

import sys
import os
# Add parent directory to path to import otsb without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from otsb import SchrodingerBridgeSolver
from otsb.utils import plot_samples, plot_trajectories


def main():
    # Set random seed
    np.random.seed(42)

    # Create source and target distributions
    n_samples = 50

    # Source: Gaussian at origin
    X0 = np.random.randn(n_samples, 2) * 0.3

    # Target: Gaussian at (3, 0)
    X1 = np.random.randn(n_samples, 2) * 0.3 + np.array([3, 0])

    # Create and fit Schrödinger Bridge solver
    print("Fitting Schrödinger Bridge...")
    sb = SchrodingerBridgeSolver(n_steps=50, sigma=0.5, max_iter=100, tol=1e-6)
    sb.fit(X0, X1)

    print(f"Converged in {sb.history_['num_iter']} iterations")

    # Sample trajectories
    print("Sampling trajectories...")
    trajectories = sb.sample_trajectory(n_samples=100, random_state=42)

    # Get transport plan
    P = sb.get_transport_plan()
    print(f"Transport plan sum: {P.sum():.4f}")

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot source and target
    plot_samples(X0, X1, ax=axes[0])
    axes[0].set_title("Source and Target Distributions")

    # Plot trajectories
    plot_trajectories(trajectories, ax=axes[1], n_plot=50)

    # Plot transport plan
    im = axes[2].imshow(P, cmap="viridis", aspect="auto")
    axes[2].set_xlabel("Target index")
    axes[2].set_ylabel("Source index")
    axes[2].set_title("Transport Plan from Schrödinger Bridge")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig("schrodinger_bridge_example.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to schrodinger_bridge_example.png")
    plt.show()


if __name__ == "__main__":
    main()
