# PyTorch Installation Guide for OT-SB

This guide explains how to install and use the PyTorch version of OT-SB for GPU-accelerated optimal transport.

## Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 1.10.0 or higher (tested up to 2.x)
- **CUDA**: Optional, for GPU acceleration

## Installation Options

### Option 1: CPU-Only Installation

For CPU-only usage (no GPU):

```bash
# Install PyTorch CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install OT-SB
cd OT-SB
pip install -e .
```

### Option 2: GPU Installation (Recommended)

For GPU-accelerated computing:

```bash
# Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for your specific system

# Example for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install OT-SB
cd OT-SB
pip install -e .
```

### Option 3: Using pip extras

```bash
# This will install PyTorch (default version)
pip install -e ".[torch]"

# Note: This installs the default PyTorch version, which may not
# be optimized for your CUDA version. Use Option 2 for best GPU performance.
```

## Verifying Installation

Run the verification script:

```bash
python examples_torch/test_torch_installation.py
```

This will check:
1. ✓ PyTorch is installed and working
2. ✓ CUDA is available (if you have a GPU)
3. ✓ otsb_torch can be imported
4. ✓ Sinkhorn algorithm works
5. ✓ Schrödinger Bridge works

Expected output:
```
======================================================================
Testing PyTorch Implementation of OT-SB
======================================================================

1. Checking PyTorch installation...
✓ PyTorch version: 2.x.x
✓ CUDA available: True
✓ CUDA version: 11.8
✓ GPU: NVIDIA GeForce RTX 3090

2. Testing otsb_torch imports...
✓ otsb_torch imports successfully

3. Testing Sinkhorn algorithm...
✓ Sinkhorn works (converged in 42 iterations)
  Transport cost: 0.8234
  Device: cuda

4. Testing Schrödinger Bridge...
✓ Schrödinger Bridge works (converged in 28 iterations)
  Sampled 10 trajectories
  Device: cuda

======================================================================
Test Summary
======================================================================
✓ PASS: PyTorch Install
✓ PASS: otsb_torch Import
✓ PASS: Sinkhorn
✓ PASS: Schrödinger Bridge

Total: 4/4 tests passed

🎉 All tests passed! PyTorch implementation is working correctly.
```

## Version Compatibility

### Tested Configurations

| PyTorch | Python | CUDA | Status |
|---------|--------|------|--------|
| 1.10.x | 3.8-3.9 | 11.1 | ✓ Supported |
| 1.12.x | 3.8-3.10 | 11.3, 11.6 | ✓ Supported |
| 1.13.x | 3.8-3.11 | 11.7 | ✓ Supported |
| 2.0.x | 3.8-3.11 | 11.7, 11.8 | ✓ Supported |
| 2.1.x | 3.8-3.11 | 11.8, 12.1 | ✓ Supported |
| 2.2.x+ | 3.8-3.12 | 11.8, 12.1 | ✓ Supported |

### Minimum Requirements

- **PyTorch >= 1.10.0** (older versions not tested)
- **CUDA >= 11.0** (for GPU support)
- Compatible with PyTorch 2.x

### API Compatibility

The implementation uses standard PyTorch APIs that are stable across versions:
- `torch.as_tensor` (since 1.0)
- `torch.logsumexp` (since 0.4.1)
- Device management (since 0.4)
- All tensor operations are standard

No breaking changes expected in future PyTorch versions.

## Troubleshooting

### "No module named 'torch'"

PyTorch is not installed. Install it:
```bash
pip install torch
```

### "CUDA not available" but you have a GPU

1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Install PyTorch with correct CUDA version:
   ```bash
   # Check your CUDA version from nvidia-smi
   # Then install matching PyTorch from pytorch.org
   ```

3. Verify CUDA in Python:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

### Import errors with otsb_torch

Make sure you're in the OT-SB directory or have installed the package:
```bash
cd OT-SB
pip install -e .
```

Or use sys.path in examples (already included):
```python
import sys
sys.path.insert(0, '/path/to/OT-SB')
```

### Performance is slow on GPU

1. Check GPU is actually being used:
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   print(f"Using device: {device}")
   ```

2. Problem might be too small for GPU benefit:
   - GPU overhead dominates for small problems (< 50×50)
   - Use CPU for small problems
   - Use GPU for large problems (> 100×100)

3. Enable GPU in your code:
   ```python
   # Explicitly specify device
   P, _ = sinkhorn(a, b, C, device='cuda')
   ```

### Version conflicts

If you have version conflicts:
```bash
# Create fresh environment
conda create -n otsb python=3.10
conda activate otsb

# Install PyTorch
pip install torch

# Install OT-SB
pip install -e .
```

## Performance Tips

### 1. Keep Data on GPU
```python
# Good: Create tensors directly on GPU
X = torch.randn(1000, 2, device='cuda')

# Bad: Create on CPU then move
X = torch.randn(1000, 2).to('cuda')  # Extra copy
```

### 2. Use Appropriate Problem Sizes

- **CPU**: Good for n < 100
- **GPU**: Best for n > 100
- **Speedup**: Increases with problem size

### 3. Batch Processing
```python
# Process multiple problems at once
# (Future feature - not yet implemented)
```

### 4. Mixed Precision (Advanced)
```python
# Use float32 (default) for most cases
# float16 for very large problems (future feature)
```

## Next Steps

1. **Run examples**: See `examples_torch/README.md`
2. **Benchmark your system**: Run `examples_torch/gpu_benchmark.py`
3. **Integrate with your code**: See examples in main README.md
4. **Report issues**: https://github.com/fzhao70/OT-SB/issues

## Additional Resources

- PyTorch Installation: https://pytorch.org/get-started/locally/
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- OT-SB Documentation: See `README.md`
- PyTorch Examples: See `examples_torch/`
