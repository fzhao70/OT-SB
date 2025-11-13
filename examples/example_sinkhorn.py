"""
Example: Using Sinkhorn algorithm for entropic regularized optimal transport.
"""

import numpy as np
import matplotlib.pyplot as plt
from otsb import sinkhorn, squared_euclidean_cost
from otsb.utils import plot_transport_plan, plot_cost_matrix


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create source and target distributions in 2D
    n_source = 20
    n_target = 30

    # Source: samples from a Gaussian
    X = np.random.randn(n_source, 2) * 0.5 + np.array([0, 0])

    # Target: samples from a mixture of Gaussians
    Y1 = np.random.randn(n_target // 2, 2) * 0.3 + np.array([2, 2])
    Y2 = np.random.randn(n_target // 2, 2) * 0.3 + np.array([2, -2])
    Y = np.vstack([Y1, Y2])

    # Uniform weights
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    # Compute cost matrix
    C = squared_euclidean_cost(X, Y)

    # Solve with Sinkhorn algorithm
    print("Solving with Sinkhorn algorithm...")
    P, log_dict = sinkhorn(a, b, C, reg=0.1, max_iter=1000, tol=1e-9, log=True)

    print(f"Converged in {log_dict['num_iter']} iterations")
    print(f"Transport cost: {np.sum(P * C):.4f}")

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot cost matrix
    plot_cost_matrix(C, ax=axes[0])
    axes[0].set_title("Cost Matrix")

    # Plot transport plan matrix
    im = axes[1].imshow(P, cmap="viridis", aspect="auto")
    axes[1].set_xlabel("Target index")
    axes[1].set_ylabel("Source index")
    axes[1].set_title("Transport Plan")
    plt.colorbar(im, ax=axes[1])

    # Plot transport plan in 2D
    plot_transport_plan(X, Y, P, ax=axes[2], threshold=1e-3)

    plt.tight_layout()
    plt.savefig("sinkhorn_example.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to sinkhorn_example.png")
    plt.show()


if __name__ == "__main__":
    main()
