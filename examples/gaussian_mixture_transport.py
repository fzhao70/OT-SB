"""
Example: Gaussian Mixture Transport
This example demonstrates OT between complex distributions.
Runs directly after cloning without installation.
"""

import sys
import os
# Add parent directory to path to import otsb without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from otsb import sinkhorn, SchrodingerBridgeSolver
from otsb.utils import squared_euclidean_cost, plot_transport_plan


def sample_gaussian_mixture(means, covs, weights, n_samples, random_state=None):
    """Sample from a Gaussian mixture model."""
    if random_state is not None:
        np.random.seed(random_state)

    n_components = len(means)
    # Sample component assignments
    components = np.random.choice(n_components, size=n_samples, p=weights)

    # Sample from each component
    samples = np.zeros((n_samples, means[0].shape[0]))
    for i, (mean, cov) in enumerate(zip(means, covs)):
        mask = components == i
        n_comp_samples = mask.sum()
        if n_comp_samples > 0:
            samples[mask] = multivariate_normal.rvs(mean=mean, cov=cov, size=n_comp_samples)

    return samples


def main():
    print("=" * 60)
    print("Gaussian Mixture Optimal Transport")
    print("=" * 60)

    np.random.seed(42)

    # Define source distribution: 3-component mixture
    source_means = [
        np.array([-2, -2]),
        np.array([2, -2]),
        np.array([0, 2])
    ]
    source_covs = [
        0.3 * np.eye(2),
        0.3 * np.eye(2),
        0.3 * np.eye(2)
    ]
    source_weights = [0.3, 0.3, 0.4]

    # Define target distribution: 2-component mixture
    target_means = [
        np.array([-1, 0]),
        np.array([1, 0])
    ]
    target_covs = [
        0.5 * np.eye(2),
        0.5 * np.eye(2)
    ]
    target_weights = [0.5, 0.5]

    # Sample from distributions
    n_source = 100
    n_target = 100

    print(f"\nSampling {n_source} source points from 3-component mixture...")
    X = sample_gaussian_mixture(source_means, source_covs, source_weights,
                                n_source, random_state=42)

    print(f"Sampling {n_target} target points from 2-component mixture...")
    Y = sample_gaussian_mixture(target_means, target_covs, target_weights,
                                n_target, random_state=43)

    # Uniform weights
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    # Compute cost matrix
    C = squared_euclidean_cost(X, Y)

    # Solve with Sinkhorn
    print("\nSolving OT with Sinkhorn...")
    P, log_dict = sinkhorn(a, b, C, reg=0.1, log=True)
    cost = np.sum(P * C)
    print(f"  Converged in {log_dict['num_iter']} iterations")
    print(f"  Transport cost: {cost:.4f}")

    # Solve with Schrödinger Bridge
    print("\nSolving Schrödinger Bridge...")
    sb = SchrodingerBridgeSolver(n_steps=30, sigma=0.3, max_iter=50, tol=1e-5)
    sb.fit(X, Y)
    print(f"  Converged in {sb.history_['num_iter']} iterations")

    # Sample trajectories
    print("\nSampling trajectories...")
    trajectories = sb.sample_trajectory(n_samples=50, random_state=42)

    # Create visualizations
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Distributions
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(X[:, 0], X[:, 1], c='blue', s=40, alpha=0.6, label='Source', edgecolors='k', linewidth=0.5)
    ax1.scatter(Y[:, 0], Y[:, 1], c='red', s=40, alpha=0.6, label='Target', edgecolors='k', linewidth=0.5)

    # Plot component centers
    for mean in source_means:
        ax1.plot(mean[0], mean[1], 'b*', markersize=15, markeredgecolor='darkblue', markeredgewidth=1.5)
    for mean in target_means:
        ax1.plot(mean[0], mean[1], 'r*', markersize=15, markeredgecolor='darkred', markeredgewidth=1.5)

    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('Source (3 components) and Target (2 components)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Transport plan
    ax2 = plt.subplot(1, 3, 2)
    plot_transport_plan(X, Y, P, ax=ax2, threshold=2e-3)
    ax2.set_title(f'Sinkhorn Transport Plan\n(cost = {cost:.4f})')

    # Plot 3: Schrödinger Bridge trajectories
    ax3 = plt.subplot(1, 3, 3)

    # Plot subset of trajectories
    n_plot = 30
    for i in range(n_plot):
        ax3.plot(trajectories[i, :, 0], trajectories[i, :, 1],
                'gray', alpha=0.3, linewidth=1)

    # Plot start and end points
    ax3.scatter(trajectories[:n_plot, 0, 0], trajectories[:n_plot, 0, 1],
               c='blue', s=50, alpha=0.7, label='Start', zorder=5, edgecolors='k', linewidth=0.5)
    ax3.scatter(trajectories[:n_plot, -1, 0], trajectories[:n_plot, -1, 1],
               c='red', s=50, alpha=0.7, label='End', zorder=5, edgecolors='k', linewidth=0.5)

    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_title(f'Schrödinger Bridge Trajectories\n(n={n_plot})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    plt.tight_layout()
    output_file = 'gaussian_mixture_transport.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_file}")

    # Additional plot: Transport plan matrix
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap of transport plan
    im1 = axes[0].imshow(P, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Target sample index')
    axes[0].set_ylabel('Source sample index')
    axes[0].set_title('Transport Plan Matrix')
    plt.colorbar(im1, ax=axes[0], label='Transport mass')

    # Marginal distributions
    marginal_source = P.sum(axis=1)
    marginal_target = P.sum(axis=0)

    axes[1].plot(marginal_source, 'b-', linewidth=2, label='Row marginals (should = a)', alpha=0.7)
    axes[1].plot(a, 'b--', linewidth=2, label='True source (a)', alpha=0.7)
    axes[1].plot(marginal_target, 'r-', linewidth=2, label='Col marginals (should = b)', alpha=0.7)
    axes[1].plot(b, 'r--', linewidth=2, label='True target (b)', alpha=0.7)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Mass')
    axes[1].set_title('Transport Plan Marginals (Validation)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file2 = 'transport_plan_details.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved details to {output_file2}")

    plt.show()

    # Compute and display statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print("-" * 60)
    print(f"Source distribution:")
    print(f"  Mean: [{X.mean(axis=0)[0]:.3f}, {X.mean(axis=0)[1]:.3f}]")
    print(f"  Std:  [{X.std(axis=0)[0]:.3f}, {X.std(axis=0)[1]:.3f}]")
    print(f"\nTarget distribution:")
    print(f"  Mean: [{Y.mean(axis=0)[0]:.3f}, {Y.mean(axis=0)[1]:.3f}]")
    print(f"  Std:  [{Y.std(axis=0)[0]:.3f}, {Y.std(axis=0)[1]:.3f}]")
    print(f"\nTransport plan validation:")
    print(f"  Row marginals error: {np.abs(marginal_source - a).max():.2e}")
    print(f"  Col marginals error: {np.abs(marginal_target - b).max():.2e}")
    print(f"  Total mass: {P.sum():.6f} (should be 1.0)")
    print("=" * 60)


if __name__ == "__main__":
    main()
