# CUDA to AMD Migration Guide

## Step 1: Analyze Your CUDA Code

```bash
cuda-scribe analyze my_kernel.cu
```

This tells you:
- Kernel complexity (trivial → extreme)
- Shared memory usage
- Warp-level primitives that need wavefront conversion
- Estimated translation difficulty

## Step 2: Choose Your Path

### Path A: Triton (Recommended)
Best for standard patterns. Same kernel runs on NVIDIA and AMD.

```bash
cuda-scribe translate my_kernel.cu --backend both
```

### Path B: HIP (Direct)
Best for complex kernels with CUDA-specific features.

```bash
cuda-scribe hipify my_kernel.cu --arch gfx942
```

### Path C: Full Project HIPIFY
For entire CUDA codebases with library dependencies.

```python
from cuda_scribe import hipify, HipifyConfig

config = HipifyConfig(target_arch="gfx942")
result = hipify(open("main.cu").read(), config=config)

# Check translation diagnostics
for d in result.diagnostics:
    print(f"[{d.level.value}] Line {d.line}: {d.message}")
```

## Step 3: Optimize for AMD

```bash
cuda-scribe optimize my_kernel.cu --target mi300x
```

## Common Migration Patterns

### Warp Shuffle → Wavefront Shuffle
```cpp
// CUDA (32 threads)
val = __shfl_down_sync(0xffffffff, val, offset);

// HIP (64 threads)
val = __shfl_down(val, offset);  // No mask needed on AMD
```

### cuBLAS → rocBLAS
```cpp
// CUDA
cublasCreate(&handle);
cublasSgemm(handle, ...);

// HIP (auto-translated)
rocblas_create_handle(&handle);
rocblas_sgemm(handle, ...);
```

### Shared Memory → LDS
```cpp
// CUDA
__shared__ float smem[256];

// HIP (identical syntax, but 64KB LDS limit per CU)
__shared__ float smem[256];
```

## Library Mapping Reference

| CUDA | ROCm | Notes |
|---|---|---|
| cuBLAS | rocBLAS | API similar, some enum differences |
| cuDNN | MIOpen | Different API surface, auto-mapped |
| cuFFT | rocFFT | Compatible API |
| cuSPARSE | rocSPARSE | Compatible API |
| cuRAND | rocRAND | Compatible API |
| NCCL | RCCL | Drop-in replacement |
| Thrust | rocThrust | Drop-in replacement |
| CUB | hipCUB | Drop-in replacement |
