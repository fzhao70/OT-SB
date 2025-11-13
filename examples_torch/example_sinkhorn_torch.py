"""
Example: Using Sinkhorn algorithm with PyTorch (GPU-accelerated).
This example can run directly after cloning without installation.
"""

import sys
import os
# Add parent directory to path to import otsb_torch without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from otsb_torch import sinkhorn, squared_euclidean_cost


def main():
    print("=" * 60)
    print("PyTorch Sinkhorn Example (GPU-Accelerated)")
    print("=" * 60)

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create source and target distributions in 2D
    n_source = 50
    n_target = 60

    # Source: samples from a Gaussian
    X = torch.randn(n_source, 2, device=device) * 0.5

    # Target: samples from a mixture of Gaussians
    Y1 = torch.randn(n_target // 2, 2, device=device) * 0.3 + torch.tensor([2, 2], device=device)
    Y2 = torch.randn(n_target // 2, 2, device=device) * 0.3 + torch.tensor([2, -2], device=device)
    Y = torch.cat([Y1, Y2], dim=0)

    # Uniform weights
    a = torch.ones(n_source, device=device) / n_source
    b = torch.ones(n_target, device=device) / n_target

    # Compute cost matrix
    print(f"\nProblem size: {n_source} x {n_target}")
    C = squared_euclidean_cost(X, Y)

    # Solve with Sinkhorn algorithm
    print("\nSolving with Sinkhorn algorithm...")
    import time
    start = time.time()
    P, log_dict = sinkhorn(a, b, C, reg=0.1, max_iter=1000, tol=1e-9, log=True, device=device)
    elapsed = time.time() - start

    cost = (P * C).sum().item()
    print(f"Converged in {log_dict['num_iter']} iterations")
    print(f"Transport cost: {cost:.4f}")
    print(f"Time elapsed: {elapsed:.4f}s")

    # Compare with different regularization
    print("\nComparing different regularization parameters:")
    for reg in [0.01, 0.05, 0.1, 0.5]:
        P_reg, log_reg = sinkhorn(a, b, C, reg=reg, log=True, device=device)
        cost_reg = (P_reg * C).sum().item()
        print(f"  reg={reg:.2f}: cost={cost_reg:.6f}, iters={log_reg['num_iter']}")

    # Visualize results (move to CPU for plotting)
    X_cpu = X.cpu().numpy()
    Y_cpu = Y.cpu().numpy()
    P_cpu = P.cpu().numpy()
    C_cpu = C.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot cost matrix
    im0 = axes[0].imshow(C_cpu, cmap="viridis", aspect="auto")
    axes[0].set_xlabel("Target index")
    axes[0].set_ylabel("Source index")
    axes[0].set_title("Cost Matrix")
    plt.colorbar(im0, ax=axes[0])

    # Plot transport plan matrix
    im1 = axes[1].imshow(P_cpu, cmap="viridis", aspect="auto")
    axes[1].set_xlabel("Target index")
    axes[1].set_ylabel("Source index")
    axes[1].set_title(f"Transport Plan (PyTorch on {device})")
    plt.colorbar(im1, ax=axes[1])

    # Plot 2D distribution
    axes[2].scatter(X_cpu[:, 0], X_cpu[:, 1], c='blue', s=50, alpha=0.6, label='Source')
    axes[2].scatter(Y_cpu[:, 0], Y_cpu[:, 1], c='red', s=50, alpha=0.6, label='Target')
    axes[2].set_xlabel('x₁')
    axes[2].set_ylabel('x₂')
    axes[2].set_title('Source and Target Distributions')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "sinkhorn_torch_example.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved visualization to {output_file}")
    plt.show()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
