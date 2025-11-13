# OT-SB Examples

This directory contains example scripts demonstrating the usage of the OT-SB library.

## Running Examples WITHOUT Installation

All examples can run directly after cloning the repository - **no pip install required**!

```bash
# Clone the repository
git clone https://github.com/fzhao70/OT-SB.git
cd OT-SB

# Run any example directly
python examples/example_sinkhorn.py
python examples/simple_1d_ot.py
python examples/comparison_methods.py
```

The examples automatically add the parent directory to Python's path, so they can import the `otsb` module without installation.

## Running Examples WITH Installation (Optional)

If you prefer to install the library:

```bash
pip install -e .
python examples/example_sinkhorn.py
```

## Examples Overview

### Basic Examples

#### 1. Sinkhorn Algorithm (`example_sinkhorn.py`)
Demonstrates entropic regularized optimal transport using the Sinkhorn algorithm.
- Creates 2D source and target distributions
- Computes optimal transport plan with entropic regularization
- Visualizes cost matrix and transport plan
- **Output**: `sinkhorn_example.png`

#### 2. Exact Optimal Transport (`example_exact_ot.py`)
Shows how to compute exact optimal transport using linear programming.
- Creates 1D distributions
- Solves exact OT problem
- Computes Wasserstein distance
- Visualizes the transport plan
- **Output**: `exact_ot_example.png`

#### 3. Schrödinger Bridge (`example_schrodinger_bridge.py`)
Illustrates dynamic optimal transport using the Schrödinger Bridge.
- Creates Gaussian source and target distributions
- Solves the Schrödinger Bridge problem
- Samples and visualizes trajectories
- Shows the resulting transport plan
- **Output**: `schrodinger_bridge_example.png`

### Advanced Examples

#### 4. Simple 1D OT (`simple_1d_ot.py`)
Comprehensive 1D optimal transport demonstration.
- Compares Sinkhorn vs Exact OT on 1D distributions
- Visualizes bimodal to unimodal transport
- Shows differences between methods
- **Output**: `simple_1d_ot.png`

#### 5. Method Comparison (`comparison_methods.py`)
Detailed comparison of different OT algorithms.
- Tests multiple regularization parameters
- Compares Sinkhorn, Log-Sinkhorn, and Exact OT
- Analyzes convergence speed and accuracy
- Shows cost vs regularization trade-offs
- **Output**: `comparison_ot_methods.png`, `comparison_analysis.png`

#### 6. Gaussian Mixture Transport (`gaussian_mixture_transport.py`)
Complex distribution transport with Schrödinger Bridge.
- Transports between multi-modal Gaussian mixtures
- Demonstrates both Sinkhorn and Schrödinger Bridge
- Validates transport plan marginals
- Shows trajectory sampling
- **Output**: `gaussian_mixture_transport.png`, `transport_plan_details.png`

## Running All Examples

To test all examples at once:

```bash
python examples/run_all_examples.py
```

This will run all examples sequentially and report which ones pass/fail.

## Dependencies

The examples require:
- numpy
- scipy
- matplotlib

These are specified in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Tips

- **Close plot windows** to continue to the next visualization
- To run examples **without displaying plots**, comment out `plt.show()` calls
- All examples save their visualizations as PNG files
- Use `--help` flag for individual example options (if implemented)

## Example Output Files

After running the examples, you'll find these visualization files:
- `sinkhorn_example.png`
- `exact_ot_example.png`
- `schrodinger_bridge_example.png`
- `simple_1d_ot.png`
- `comparison_ot_methods.png`
- `comparison_analysis.png`
- `gaussian_mixture_transport.png`
- `transport_plan_details.png`

## Customizing Examples

All examples are well-commented and easy to modify. Key parameters you can adjust:

- **Number of samples**: `n_source`, `n_target`, `n_samples`
- **Regularization**: `reg` parameter in Sinkhorn
- **Convergence**: `max_iter`, `tol` parameters
- **Random seed**: `np.random.seed()` for reproducibility
- **Visualization**: `threshold` parameter for cleaner plots
