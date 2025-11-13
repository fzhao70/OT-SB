# PyTorch Examples for OT-SB

GPU-accelerated examples using the PyTorch implementation of OT-SB.

## Features

- **GPU Acceleration**: Run on CUDA-enabled GPUs for significant speedup
- **Automatic Differentiation**: Integrate with deep learning pipelines
- **Batched Operations**: Efficient tensor operations
- **Device Flexibility**: Seamlessly switch between CPU and GPU

## Requirements

```bash
pip install torch numpy scipy matplotlib
```

Or install PyTorch from the official website for CUDA support:
https://pytorch.org/get-started/locally/

## Running Examples WITHOUT Installation

All examples work directly after cloning:

```bash
# Clone the repository
git clone https://github.com/fzhao70/OT-SB.git
cd OT-SB

# Install dependencies
pip install torch numpy scipy matplotlib

# Run examples directly
python examples_torch/example_sinkhorn_torch.py
python examples_torch/example_schrodinger_bridge_torch.py
python examples_torch/gpu_benchmark.py
```

## Examples Overview

### 1. Sinkhorn Algorithm (`example_sinkhorn_torch.py`)

GPU-accelerated Sinkhorn algorithm demonstration.

**Features:**
- Automatic GPU detection and usage
- Comparison of different regularization parameters
- Performance timing
- Visualization of transport plans

**Usage:**
```bash
python examples_torch/example_sinkhorn_torch.py
```

**Output:** `sinkhorn_torch_example.png`

### 2. Schrödinger Bridge (`example_schrodinger_bridge_torch.py`)

GPU-accelerated Schrödinger Bridge solver.

**Features:**
- GPU-accelerated trajectory sampling
- Fast IPF iterations on GPU
- Transport plan visualization
- Performance benchmarking

**Usage:**
```bash
python examples_torch/example_schrodinger_bridge_torch.py
```

**Output:** `schrodinger_bridge_torch_example.png`

### 3. GPU Benchmark (`gpu_benchmark.py`)

Comprehensive CPU vs GPU performance comparison.

**Features:**
- Benchmarks multiple problem sizes
- Compares Sinkhorn and Schrödinger Bridge
- Visualizes speedup curves
- Prints detailed speedup statistics

**Usage:**
```bash
python examples_torch/gpu_benchmark.py
```

**Output:** `gpu_benchmark.png`

**Note:** Requires CUDA-enabled GPU for meaningful comparison.

## CPU vs GPU Performance

Expected speedups with GPU (varies by hardware):

| Algorithm | Problem Size | Typical GPU Speedup |
|-----------|--------------|---------------------|
| Sinkhorn | 100 × 100 | 2-3x |
| Sinkhorn | 500 × 500 | 5-10x |
| Sinkhorn | 1000 × 1000 | 10-20x |
| Schrödinger Bridge | 200 samples | 3-5x |
| Schrödinger Bridge | 500 samples | 5-10x |

## Using PyTorch OT-SB in Your Code

### Basic Example

```python
import torch
from otsb_torch import sinkhorn, squared_euclidean_cost

# Specify device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create data on GPU
X = torch.randn(100, 2, device=device)
Y = torch.randn(100, 2, device=device) + 2

a = torch.ones(100, device=device) / 100
b = torch.ones(100, device=device) / 100

# Compute OT
C = squared_euclidean_cost(X, Y)
P, log_dict = sinkhorn(a, b, C, reg=0.1, log=True, device=device)

print(f"Transport cost: {(P * C).sum().item():.4f}")
```

### Schrödinger Bridge with GPU

```python
import torch
from otsb_torch import SchrodingerBridgeSolver

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create distributions
X0 = torch.randn(200, 2, device=device)
X1 = torch.randn(200, 2, device=device) + 3

# Fit bridge on GPU
sb = SchrodingerBridgeSolver(
    n_steps=50,
    sigma=0.5,
    max_iter=100,
    device=device
)
sb.fit(X0, X1)

# Sample trajectories (on GPU)
trajectories = sb.sample_trajectory(n_samples=100)
# trajectories is a GPU tensor: shape (100, 50, 2)
```

### Integration with Deep Learning

```python
import torch
import torch.nn as nn
from otsb_torch import sinkhorn, squared_euclidean_cost

class OTLoss(nn.Module):
    """Optimal Transport loss for neural networks."""

    def __init__(self, reg=0.1):
        super().__init__()
        self.reg = reg

    def forward(self, X, Y):
        """
        Compute OT distance between two point clouds.

        Parameters
        ----------
        X, Y : torch.Tensor, shape (batch_size, n, d)
            Point clouds

        Returns
        -------
        loss : torch.Tensor
            OT distance (differentiable)
        """
        batch_size = X.shape[0]
        n = X.shape[1]

        loss = 0
        for i in range(batch_size):
            a = torch.ones(n, device=X.device) / n
            b = torch.ones(n, device=X.device) / n
            C = squared_euclidean_cost(X[i], Y[i])

            P, _ = sinkhorn(a, b, C, reg=self.reg, device=X.device)
            loss += (P * C).sum()

        return loss / batch_size

# Use in training
model = ...  # Your neural network
optimizer = torch.optim.Adam(model.parameters())
ot_loss = OTLoss(reg=0.1)

for epoch in range(num_epochs):
    pred = model(input)
    target = ...
    loss = ot_loss(pred, target)

    optimizer.zero_grad()
    loss.backward()  # OT loss is differentiable!
    optimizer.step()
```

## Device Selection

The PyTorch implementation automatically handles device placement:

```python
# Automatic device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Or specify explicitly
device = 'cuda:0'  # First GPU
device = 'cuda:1'  # Second GPU
device = 'cpu'     # CPU only

# Pass to functions
P, _ = sinkhorn(a, b, C, device=device)

# Or to solvers
sb = SchrodingerBridgeSolver(device=device)
```

## Memory Considerations

GPU memory is limited. For large problems:

- **Sinkhorn**: Scales as O(n × m) for cost matrix
- **Schrödinger Bridge**: Scales as O(n × m × T) for T time steps

Tips for large-scale problems:
1. Use smaller regularization (fewer Sinkhorn iterations)
2. Reduce number of time steps for Schrödinger Bridge
3. Process in batches
4. Use gradient checkpointing for very large problems

## Performance Tips

1. **Keep data on GPU**: Avoid CPU ↔ GPU transfers
2. **Batch operations**: Process multiple problems simultaneously
3. **Use appropriate dtype**: `float32` is usually sufficient and faster than `float64`
4. **Warm-up GPU**: First iteration is slower due to CUDA initialization
5. **Profile your code**: Use `torch.cuda.synchronize()` for accurate timing

## Troubleshooting

**CUDA out of memory:**
- Reduce problem size
- Decrease batch size
- Use smaller time steps
- Clear cache: `torch.cuda.empty_cache()`

**Slow performance on GPU:**
- Problem might be too small (GPU overhead > speedup)
- Check data is actually on GPU: `tensor.device`
- Ensure CUDA is properly installed: `torch.cuda.is_available()`

**Import errors:**
- Make sure PyTorch is installed: `pip install torch`
- Add parent directory to path (for standalone examples)
- Or install the package: `pip install -e .`

## Comparison with NumPy Version

| Feature | NumPy (`otsb`) | PyTorch (`otsb_torch`) |
|---------|----------------|------------------------|
| CPU Performance | ✓ Fast | ✓ Fast |
| GPU Support | ✗ No | ✓ Yes |
| Automatic Differentiation | ✗ No | ✓ Yes |
| Deep Learning Integration | ✗ Limited | ✓ Native |
| Memory Efficiency | ✓✓ Excellent | ✓ Good |
| Exact OT | ✓ Native | ⚠ Via scipy/POT |

**When to use PyTorch version:**
- You have a GPU
- Need gradients for deep learning
- Working with PyTorch pipelines
- Large-scale problems that benefit from GPU

**When to use NumPy version:**
- CPU-only environment
- Need exact OT (better support)
- Simpler deployment
- Smaller problems

## See Also

- Main examples: `../examples/`
- NumPy version: `../otsb/`
- Documentation: `../README.md`
- PyTorch docs: https://pytorch.org/docs/
