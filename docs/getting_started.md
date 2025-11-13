# Getting Started with OT-SB

This guide will help you get started with the OT-SB library for Optimal Transport and Schrödinger Bridge problems.

## Installation

```bash
git clone https://github.com/fzhao70/OT-SB.git
cd OT-SB
pip install -e .
```

## Basic Concepts

### Optimal Transport

Optimal transport finds the most efficient way to transform one probability distribution into another. Given:
- Source distribution `a` with support on points `X`
- Target distribution `b` with support on points `Y`
- Cost function `C` (e.g., squared Euclidean distance)

OT finds a coupling `P` that minimizes the total transport cost.

### Entropic Regularization

Adding entropy regularization smooths the problem and enables fast algorithms like Sinkhorn:
- Regularization parameter `ε` controls smoothness
- Larger `ε` → smoother solution, faster convergence
- Smaller `ε` → closer to exact OT, slower convergence

### Schrödinger Bridge

The Schrödinger Bridge extends OT to the dynamic setting, finding the most likely stochastic process connecting two distributions.

## First Example: 2D Gaussian Transport

```python
import numpy as np
import matplotlib.pyplot as plt
from otsb import sinkhorn, squared_euclidean_cost
from otsb.utils import plot_transport_plan

# Create source and target samples
np.random.seed(42)
n_samples = 50

# Source: Gaussian at origin
X = np.random.randn(n_samples, 2) * 0.5

# Target: Gaussian at (2, 0)
Y = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 0])

# Uniform weights
a = np.ones(n_samples) / n_samples
b = np.ones(n_samples) / n_samples

# Compute cost matrix
C = squared_euclidean_cost(X, Y)

# Solve with Sinkhorn
P, log_dict = sinkhorn(a, b, C, reg=0.1, log=True)

print(f"Converged in {log_dict['num_iter']} iterations")
print(f"Transport cost: {np.sum(P * C):.4f}")

# Visualize
plot_transport_plan(X, Y, P)
plt.show()
```

## Choosing the Right Algorithm

### Use Sinkhorn when:
- You need fast computation
- Approximate solution is acceptable
- Working with large-scale problems
- Want differentiable solutions (for ML applications)

### Use Exact OT (EMD) when:
- Need the exact optimal solution
- Problem size is small to moderate (< 1000 samples)
- Computing Wasserstein distances

### Use Schrödinger Bridge when:
- Interested in dynamic transport
- Need sample trajectories
- Modeling stochastic processes

## Regularization Parameter Selection

The regularization parameter `ε` (reg) is crucial for Sinkhorn:

```python
# Small ε: more accurate but slower
P1, _ = sinkhorn(a, b, C, reg=0.01, log=True)

# Medium ε: balanced
P2, _ = sinkhorn(a, b, C, reg=0.1, log=True)

# Large ε: fast but smoother
P3, _ = sinkhorn(a, b, C, reg=1.0, log=True)
```

**Rule of thumb**: Start with `reg = 0.1` and adjust based on your needs.

## Common Workflows

### 1. Computing Wasserstein Distance

```python
from otsb import wasserstein_distance

# Between empirical distributions
X = np.random.randn(100, 2)
Y = np.random.randn(100, 2) + 1

W2 = wasserstein_distance(
    a=np.ones(100)/100,
    b=np.ones(100)/100,
    X=X, Y=Y, p=2
)
```

### 2. Custom Cost Functions

```python
from otsb.utils import cost_matrix

# Use different metrics
C_euclidean = cost_matrix(X, Y, metric='euclidean')
C_manhattan = cost_matrix(X, Y, metric='manhattan')
C_cosine = cost_matrix(X, Y, metric='cosine')
```

### 3. Sampling Transport Trajectories

```python
from otsb import SchrodingerBridgeSolver

# Fit bridge
sb = SchrodingerBridgeSolver(n_steps=50, sigma=0.5)
sb.fit(X, Y)

# Sample trajectories
trajectories = sb.sample_trajectory(n_samples=100)
# trajectories.shape = (100, 50, 2)
# 100 trajectories, 50 time steps, 2D space
```

## Next Steps

- Check out the [examples/](../examples/) directory for more examples
- Read the [API Reference](api_reference.md) for detailed documentation
- Explore advanced features in the library

## Troubleshooting

### Sinkhorn doesn't converge
- Increase `max_iter`
- Increase `reg` (regularization)
- Check that distributions sum to 1

### Memory issues
- Reduce number of samples
- Use Sinkhorn instead of exact OT for large problems

### Numerical instability
- Use `sinkhorn_log` instead of `sinkhorn`
- Increase regularization parameter
