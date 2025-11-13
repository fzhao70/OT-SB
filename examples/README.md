# OT-SB Examples

This directory contains example scripts demonstrating the usage of the OT-SB library.

## Running the Examples

Make sure you have installed the library first:

```bash
pip install -e .
```

Then run any example:

```bash
python examples/example_sinkhorn.py
python examples/example_exact_ot.py
python examples/example_schrodinger_bridge.py
```

## Examples Overview

### 1. Sinkhorn Algorithm (`example_sinkhorn.py`)

Demonstrates entropic regularized optimal transport using the Sinkhorn algorithm.
- Creates 2D source and target distributions
- Computes optimal transport plan with entropic regularization
- Visualizes cost matrix and transport plan

### 2. Exact Optimal Transport (`example_exact_ot.py`)

Shows how to compute exact optimal transport using linear programming.
- Creates 1D distributions
- Solves exact OT problem
- Computes Wasserstein distance
- Visualizes the transport plan

### 3. Schrödinger Bridge (`example_schrodinger_bridge.py`)

Illustrates dynamic optimal transport using the Schrödinger Bridge.
- Creates Gaussian source and target distributions
- Solves the Schrödinger Bridge problem
- Samples and visualizes trajectories
- Shows the resulting transport plan

## Output

Each example generates a PNG file with visualizations:
- `sinkhorn_example.png`
- `exact_ot_example.png`
- `schrodinger_bridge_example.png`
