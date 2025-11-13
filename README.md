# OT-SB: Optimal Transport and Schrödinger Bridge Library

A comprehensive Python library for Optimal Transport (OT) and Schrödinger Bridge (SB) problems, providing efficient implementations of state-of-the-art algorithms for applications in machine learning, statistics, and scientific computing.

## Features

- **Entropic Regularized OT**: Fast Sinkhorn algorithm with log-domain stabilization
- **Exact Optimal Transport**: Linear programming-based EMD (Earth Mover's Distance) solver
- **Schrödinger Bridge**: Dynamic optimal transport solver using Iterative Proportional Fitting
- **Flexible Cost Functions**: Multiple distance metrics (Euclidean, Manhattan, Cosine, etc.)
- **Visualization Tools**: Built-in plotting for transport plans and trajectories
- **Well-tested**: Comprehensive test suite ensuring reliability
- **🔥 GPU Support**: PyTorch implementation with CUDA acceleration (see `otsb_torch/`)

## Try It Without Installation!

Want to try OT-SB immediately? All examples work directly after cloning - **no pip install needed**!

```bash
git clone https://github.com/fzhao70/OT-SB.git
cd OT-SB
pip install numpy scipy matplotlib  # Install dependencies only
python examples/simple_1d_ot.py      # Run examples directly!
python examples/comparison_methods.py
```

See [examples/README.md](examples/README.md) for more details.

## Installation

### From source

```bash
git clone https://github.com/fzhao70/OT-SB.git
cd OT-SB
pip install -e .
```

### Development installation

For development with additional dependencies:

```bash
pip install -e ".[dev]"
```

### PyTorch / GPU version

For GPU-accelerated optimal transport:

```bash
# Install with PyTorch support
pip install -e ".[torch]"

# Or install PyTorch separately for specific CUDA version
# See https://pytorch.org for installation instructions
pip install torch
pip install -e .
```

See `examples_torch/` and `otsb_torch/` for PyTorch implementation.

## Quick Start

### Sinkhorn Algorithm (Entropic OT)

```python
import numpy as np
from otsb import sinkhorn, squared_euclidean_cost

# Create source and target distributions
n, m = 50, 60
a = np.ones(n) / n  # uniform source
b = np.ones(m) / m  # uniform target

# Sample points
X = np.random.randn(n, 2)
Y = np.random.randn(m, 2) + 2

# Compute cost matrix
C = squared_euclidean_cost(X, Y)

# Solve OT with Sinkhorn
P, log_dict = sinkhorn(a, b, C, reg=0.1, log=True)
print(f"Converged in {log_dict['num_iter']} iterations")
```

### Exact Optimal Transport

```python
from otsb import emd, wasserstein_distance

# Solve exact OT (no regularization)
P, result = emd(a, b, C)
print(f"Optimal cost: {result['cost']}")

# Compute Wasserstein distance
W2 = wasserstein_distance(a, b, X, Y, p=2)
print(f"Wasserstein-2 distance: {W2}")
```

### Schrödinger Bridge

```python
from otsb import SchrodingerBridgeSolver
from otsb.utils import plot_trajectories
import matplotlib.pyplot as plt

# Create source and target samples
X0 = np.random.randn(100, 2) * 0.5
X1 = np.random.randn(100, 2) * 0.5 + np.array([3, 0])

# Fit Schrödinger Bridge
sb = SchrodingerBridgeSolver(n_steps=50, sigma=0.5, max_iter=100)
sb.fit(X0, X1)

# Sample trajectories
trajectories = sb.sample_trajectory(n_samples=50)

# Visualize
plot_trajectories(trajectories)
plt.show()
```

## Library Structure

```
otsb/
├── core/               # Core OT algorithms
│   ├── sinkhorn.py    # Sinkhorn algorithm
│   └── exact_ot.py    # Exact OT solvers (EMD)
├── solvers/           # Advanced solvers
│   └── schrodinger_bridge.py  # Schrödinger Bridge
└── utils/             # Utilities
    ├── distances.py   # Distance metrics and cost functions
    └── visualization.py  # Plotting tools
```

## API Reference

### Core Algorithms

#### `sinkhorn(a, b, C, reg=1.0, max_iter=1000, tol=1e-9, log=False)`

Solve entropic regularized optimal transport using the Sinkhorn algorithm.

**Parameters:**
- `a`: Source distribution (array of shape (n,))
- `b`: Target distribution (array of shape (m,))
- `C`: Cost matrix (array of shape (n, m))
- `reg`: Regularization parameter ε (float, default=1.0)
- `max_iter`: Maximum iterations (int, default=1000)
- `tol`: Convergence tolerance (float, default=1e-9)
- `log`: Return convergence information (bool, default=False)

**Returns:**
- `P`: Transport plan (array of shape (n, m))
- `log_dict`: Dictionary with convergence info (if log=True)

#### `emd(a, b, C, method="highs")`

Solve exact optimal transport using linear programming.

**Parameters:**
- `a`: Source distribution
- `b`: Target distribution
- `C`: Cost matrix
- `method`: LP solver method (str, default="highs")

**Returns:**
- `P`: Optimal transport plan
- `result_dict`: Solver information

### Schrödinger Bridge

#### `SchrodingerBridgeSolver`

```python
sb = SchrodingerBridgeSolver(
    n_steps=100,      # Time discretization steps
    sigma=1.0,        # Diffusion coefficient
    max_iter=100,     # Maximum IPF iterations
    tol=1e-6          # Convergence tolerance
)
```

**Methods:**
- `fit(X0, X1, weights0=None, weights1=None)`: Fit the bridge
- `sample_trajectory(n_samples=100, random_state=None)`: Sample trajectories
- `get_transport_plan()`: Get the transport plan matrix

### Utility Functions

#### Distance Metrics

```python
from otsb.utils import (
    squared_euclidean_cost,
    euclidean_cost,
    manhattan_cost,
    cosine_cost,
    cost_matrix,  # Flexible cost matrix computation
)
```

#### Visualization

```python
from otsb.utils import (
    plot_transport_plan,
    plot_samples,
    plot_trajectories,
    plot_cost_matrix,
)
```

## Examples

The library includes 6 comprehensive examples that **run without installation**:

**Basic Examples:**
- `example_sinkhorn.py`: Entropic OT with Sinkhorn algorithm
- `example_exact_ot.py`: Exact OT and Wasserstein distance
- `example_schrodinger_bridge.py`: Schrödinger Bridge trajectories

**Advanced Examples:**
- `simple_1d_ot.py`: 1D transport with method comparison
- `comparison_methods.py`: Detailed algorithm comparison and analysis
- `gaussian_mixture_transport.py`: Complex multi-modal distributions

Run any example directly after cloning:

```bash
python examples/simple_1d_ot.py
python examples/comparison_methods.py
```

Or run all examples at once:

```bash
python examples/run_all_examples.py
```

See [examples/README.md](examples/README.md) for detailed documentation.

## Mathematical Background

### Optimal Transport

The optimal transport problem seeks to find the most efficient way to transport mass from distribution `a` to distribution `b`:

```
min_{P ∈ Π(a,b)} ⟨P, C⟩
```

where `Π(a,b)` is the set of joint distributions with marginals `a` and `b`.

### Entropic Regularization

Adding entropic regularization makes the problem smooth and enables fast algorithms:

```
min_{P ∈ Π(a,b)} ⟨P, C⟩ + ε H(P)
```

where `H(P)` is the entropy of `P`. The Sinkhorn algorithm efficiently solves this.

### Schrödinger Bridge

The Schrödinger Bridge is the most likely stochastic process connecting two distributions, given a reference diffusion:

```
min_{π} KL(π || π_ref)
subject to: π_0 = a, π_1 = b
```

This generalizes optimal transport to the dynamic setting.

## Testing

Run the test suite:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=otsb --cov-report=html
```

## Dependencies

Core dependencies:
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0

Development dependencies:
- pytest >= 7.0.0
- pytest-cov >= 3.0.0
- black >= 22.0.0
- mypy >= 0.950

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{otsb2024,
  title={OT-SB: A Library for Optimal Transport and Schrödinger Bridge},
  author={OT-SB Contributors},
  year={2024},
  url={https://github.com/fzhao70/OT-SB}
}
```

## References

1. Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport. Foundations and Trends in Machine Learning.
2. Chen, Y., Georgiou, T. T., & Pavon, M. (2021). Optimal transport in systems and control. Annual Review of Control, Robotics, and Autonomous Systems.
3. Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. NeurIPS.

## Acknowledgments

This library implements algorithms from the optimal transport and Schrödinger Bridge literature. We thank the research community for their foundational work in this area.
