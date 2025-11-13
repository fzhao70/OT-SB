"""
Example: Using Schrödinger Bridge solver with PyTorch (GPU-accelerated).
This example can run directly after cloning without installation.
"""

import sys
import os
# Add parent directory to path to import otsb_torch without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from otsb_torch import SchrodingerBridgeSolver


def main():
    print("=" * 60)
    print("PyTorch Schrödinger Bridge Example (GPU-Accelerated)")
    print("=" * 60)

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Create source and target distributions
    n_samples = 100

    # Source: Gaussian at origin
    X0 = torch.randn(n_samples, 2, device=device) * 0.3

    # Target: Gaussian at (3, 0)
    X1 = torch.randn(n_samples, 2, device=device) * 0.3 + torch.tensor([3, 0], device=device, dtype=torch.float32)

    print(f"\nProblem size: {n_samples} samples, 2D space")

    # Create and fit Schrödinger Bridge solver
    print("\nFitting Schrödinger Bridge...")
    import time
    start = time.time()

    sb = SchrodingerBridgeSolver(
        n_steps=50,
        sigma=0.5,
        max_iter=100,
        tol=1e-6,
        device=device
    )
    sb.fit(X0, X1)

    elapsed = time.time() - start

    print(f"Converged in {sb.history_['num_iter']} iterations")
    print(f"Time elapsed: {elapsed:.4f}s")

    # Sample trajectories
    print("\nSampling trajectories...")
    trajectories = sb.sample_trajectory(n_samples=50, random_state=42)

    # Get transport plan
    P = sb.get_transport_plan()
    print(f"Transport plan sum: {P.sum().item():.4f}")

    # Move to CPU for visualization
    X0_cpu = X0.cpu().numpy()
    X1_cpu = X1.cpu().numpy()
    trajectories_cpu = trajectories.cpu().numpy()
    P_cpu = P.cpu().numpy()

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot source and target
    axes[0].scatter(X0_cpu[:, 0], X0_cpu[:, 1], c='blue', s=50, alpha=0.6, label='Source', edgecolors='k', linewidth=0.5)
    axes[0].scatter(X1_cpu[:, 0], X1_cpu[:, 1], c='red', s=50, alpha=0.6, label='Target', edgecolors='k', linewidth=0.5)
    axes[0].set_xlabel('x₁')
    axes[0].set_ylabel('x₂')
    axes[0].set_title('Source and Target Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # Plot trajectories
    n_plot = min(30, len(trajectories_cpu))
    for i in range(n_plot):
        axes[1].plot(
            trajectories_cpu[i, :, 0],
            trajectories_cpu[i, :, 1],
            'gray',
            alpha=0.3,
            linewidth=1
        )

    axes[1].scatter(
        trajectories_cpu[:n_plot, 0, 0],
        trajectories_cpu[:n_plot, 0, 1],
        c='blue',
        s=50,
        alpha=0.7,
        label='Start',
        zorder=5,
        edgecolors='k',
        linewidth=0.5
    )
    axes[1].scatter(
        trajectories_cpu[:n_plot, -1, 0],
        trajectories_cpu[:n_plot, -1, 1],
        c='red',
        s=50,
        alpha=0.7,
        label='End',
        zorder=5,
        edgecolors='k',
        linewidth=0.5
    )
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].set_title(f'Schrödinger Bridge Trajectories (PyTorch on {device})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    # Plot transport plan
    im = axes[2].imshow(P_cpu, cmap="viridis", aspect="auto")
    axes[2].set_xlabel("Target index")
    axes[2].set_ylabel("Source index")
    axes[2].set_title("Transport Plan from Schrödinger Bridge")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    output_file = "schrodinger_bridge_torch_example.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved visualization to {output_file}")
    plt.show()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
