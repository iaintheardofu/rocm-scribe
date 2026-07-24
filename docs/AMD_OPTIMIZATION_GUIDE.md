# AMD Optimization Guide

## Critical Differences: NVIDIA vs AMD

### 1. Wavefront Width (Most Important)
- NVIDIA warp: **32 threads**
- AMD wavefront: **64 threads** (CDNA), 32 or 64 (RDNA)

**Impact:** Reduction steps (6 vs 5), shared memory access patterns, ballot mask width.

```python
from cuda_scribe import convert_warp_to_wavefront

# Automatically adjust warp-level code for AMD
amd_code = convert_warp_to_wavefront(cuda_source)
```

### 2. LDS vs Shared Memory
- NVIDIA: shared memory per SM (configurable with L1)
- AMD: LDS per CU (fixed 64KB on CDNA)

**Bank conflicts:** AMD LDS has 32 banks with 4-byte stride (same as NVIDIA), but 64-wide wavefronts mean different conflict patterns.

```python
from cuda_scribe import apply_lds_bank_conflict_padding

padded_size = apply_lds_bank_conflict_padding(array_size, element_bytes=4)
```

### 3. MFMA Instructions (Matrix Cores)
AMD's equivalent of NVIDIA Tensor Cores:

| Instruction | Input | Output | Throughput |
|---|---|---|---|
| `v_mfma_f32_32x32x8_f16` | FP16 | FP32 | 512 ops/cycle |
| `v_mfma_f32_16x16x16_f16` | FP16 | FP32 | 512 ops/cycle |
| `v_mfma_f32_32x32x16_bf16` | BF16 | FP32 | 512 ops/cycle |
| `v_mfma_f32_32x32x16_fp8` | FP8 | FP32 | 1024 ops/cycle |
| `v_mfma_i32_32x32x16_i8` | INT8 | INT32 | 1024 ops/cycle |

```python
from cuda_scribe import select_mfma_instruction

instr = select_mfma_instruction(dtype="fp16", m=32, n=32)
print(instr)  # v_mfma_f32_32x32x8_f16
```

### 4. Occupancy
AMD occupancy is VGPR-limited (not register-limited like NVIDIA):

| VGPRs per wavefront | Max wavefronts per SIMD |
|---|---|
| 48 | 10 |
| 96 | 5 |
| 128 | 4 |
| 256 | 2 |
| 512 | 1 |

```python
from cuda_scribe import compute_occupancy

occ = compute_occupancy(vgprs_per_thread=48, arch="cdna3")
print(occ)  # {'wavefronts_per_simd': 10, 'occupancy_pct': 100.0}
```

## Compiler Flags

### MI300X (gfx942)
```bash
hipcc --offload-arch=gfx942 -O3 -ffast-math \
      -mllvm -amdgpu-early-inline-all=true \
      -mllvm -amdgpu-function-calls=false \
      kernel.hip.cpp -o kernel
```

### MI250X (gfx90a)
```bash
hipcc --offload-arch=gfx90a -O3 -ffast-math kernel.hip.cpp -o kernel
```

## Triton on AMD

Triton natively supports AMD GPUs via the HIP backend. The same Triton kernel runs on both NVIDIA and AMD:

```python
from cuda_scribe import generate_triton_amd_config

# Get AMD-optimal Triton autotuning configs
configs = generate_triton_amd_config(kernel_type="gemm", arch="cdna3")
```

Key differences when writing Triton for AMD:
- Use `BLOCK_SIZE` multiples of 64 (not 32)
- Prefer larger tile sizes (AMD has more registers)
- Pipeline stages may differ (AMD HBM latency profile)

## Roofline Analysis

```python
from cuda_scribe import roofline_analysis

analysis = roofline_analysis(
    flops=1e12,           # Operations
    bytes_moved=1e10,     # Memory traffic
    arch="cdna3",         # Target GPU
)
print(analysis)
# {'bound': 'compute', 'achieved_pct': 76.5, 'peak_tflops': 1307}
```
