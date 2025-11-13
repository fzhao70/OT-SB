# Quick Start Guide - OT-SB

Get started with OT-SB in 2 minutes!

## Option 1: Try Without Installation (Recommended for First Time)

```bash
# 1. Clone the repository
git clone https://github.com/fzhao70/OT-SB.git
cd OT-SB

# 2. Install only the dependencies (not the package itself)
pip install numpy scipy matplotlib

# 3. Run an example immediately!
python examples/simple_1d_ot.py
```

That's it! The example will:
- Run optimal transport on 1D distributions
- Compare Sinkhorn vs Exact OT
- Generate visualizations
- Save output as `simple_1d_ot.png`

## Option 2: Install the Package

```bash
# Clone and install
git clone https://github.com/fzhao70/OT-SB.git
cd OT-SB
pip install -e .

# Now you can use it anywhere in Python
python -c "from otsb import sinkhorn; print('Success!')"
```

## Try Different Examples

```bash
# Basic examples
python examples/example_sinkhorn.py              # Entropic OT
python examples/example_exact_ot.py              # Exact OT
python examples/example_schrodinger_bridge.py   # Schrödinger Bridge

# Advanced examples
python examples/simple_1d_ot.py                  # 1D comparison
python examples/comparison_methods.py            # Algorithm analysis
python examples/gaussian_mixture_transport.py    # Complex distributions

# Run all examples
python examples/run_all_examples.py
```

## Your First Script

Create `my_ot_example.py`:

```python
import sys
import os
# Add this if you haven't installed the package
sys.path.insert(0, 'path/to/OT-SB')

import numpy as np
from otsb import sinkhorn, squared_euclidean_cost

# Create simple distributions
n = 20
X = np.random.randn(n, 2)      # Source
Y = np.random.randn(n, 2) + 2  # Target

# Uniform weights
a = np.ones(n) / n
b = np.ones(n) / n

# Compute cost and solve OT
C = squared_euclidean_cost(X, Y)
P, log_dict = sinkhorn(a, b, C, reg=0.1, log=True)

print(f"Converged in {log_dict['num_iter']} iterations")
print(f"Transport cost: {(P * C).sum():.4f}")
```

Run it:
```bash
python my_ot_example.py
```

## Common Use Cases

### Computing Wasserstein Distance
```python
from otsb import wasserstein_distance
import numpy as np

X = np.random.randn(100, 2)
Y = np.random.randn(100, 2) + 1

W2 = wasserstein_distance(
    a=np.ones(100)/100,
    b=np.ones(100)/100,
    X=X, Y=Y, p=2
)
print(f"Wasserstein-2 distance: {W2:.4f}")
```

### Sampling Trajectories
```python
from otsb import SchrodingerBridgeSolver
import numpy as np

# Create distributions
X0 = np.random.randn(50, 2)
X1 = np.random.randn(50, 2) + 3

# Fit bridge
sb = SchrodingerBridgeSolver(n_steps=50, sigma=0.5)
sb.fit(X0, X1)

# Sample trajectories
trajectories = sb.sample_trajectory(n_samples=100)
# Shape: (100, 50, 2) - 100 paths, 50 time steps, 2D space
```

### Custom Cost Functions
```python
from otsb.utils import cost_matrix

# Different distance metrics
C_euclidean = cost_matrix(X, Y, metric='euclidean')
C_manhattan = cost_matrix(X, Y, metric='manhattan')
C_cosine = cost_matrix(X, Y, metric='cosine')
```

## Next Steps

1. **Explore Examples**: See `examples/README.md` for detailed documentation
2. **Read API Reference**: Check main `README.md` for complete API
3. **Run Tests**: `pytest tests/` to verify installation
4. **Read Docs**: See `docs/getting_started.md` for in-depth guide

## Need Help?

- Examples not working? Make sure dependencies are installed: `pip install numpy scipy matplotlib`
- Import errors? Check you added the sys.path line or installed the package
- Questions? Check the main README.md for detailed documentation

## Troubleshooting

**"No module named 'otsb'"**
- Either add `sys.path.insert(0, 'path/to/OT-SB')` at the top of your script
- Or install the package: `pip install -e .`

**"No module named 'numpy'"**
- Install dependencies: `pip install numpy scipy matplotlib`

**Examples don't display plots**
- Check matplotlib backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`
- Make sure you're not running in a headless environment
- Plots are saved as PNG files anyway

**Sinkhorn doesn't converge**
- Try increasing regularization: `reg=0.5` instead of `reg=0.1`
- Increase iterations: `max_iter=5000`
- Use log-domain version: `from otsb import sinkhorn_log`

Happy transporting! 🚀
