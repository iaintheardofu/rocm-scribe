<p align="center">
  <h1 align="center">ROCm Scribe</h1>
  <p align="center">
    <strong>Automated CUDA-to-ROCm translation. Break free from vendor lock-in.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> |
    <a href="#how-it-works">How It Works</a> |
    <a href="#cli">CLI</a> |
    <a href="#amd-hardware-optimization">AMD Optimization</a> |
    <a href="#benchmarks">Benchmarks</a> |
    <a href="#pro">Pro</a>
  </p>
</p>

---

ROCm Scribe translates CUDA GPU kernels to run on AMD hardware. Not string replacement -- **semantic translation** through BridgeIR, a universal intermediate representation that captures computation structure, memory hierarchy, synchronization topology, and reduction patterns before generating target-specific code.

Two production paths:

1. **CUDA to Triton** -- Same kernel runs on NVIDIA *and* AMD. Zero code changes between vendors.
2. **CUDA to HIP C++** -- Direct source-level translation with 316 API mappings covering the full ROCm stack.

```bash
pip install rocm-scribe
```

## Why This Exists

Migrating from CUDA to AMD costs **$72K--$205K per major model** when done manually. AMD's own GEAK tool achieves **54--63% translation accuracy**. Academic one-shot methods fail on quantization kernels **100% of the time**.

ROCm Scribe exists because GPU compute should be a choice, not a lock-in. If you have CUDA code and AMD hardware, this tool gets you there.

## Quick Start

### Python API

```python
from cuda_scribe import AMDBridge, hipify

# Translate a CUDA kernel to AMD-ready Triton
bridge = AMDBridge()
result = bridge.translate(cuda_source)
print(result.triton_source)      # Runs on AMD + NVIDIA
print(result.hip_source)         # HIP C++ for AMD native
print(result.occupancy)          # Wavefront occupancy on MI300X
print(result.roofline)           # Compute vs memory bound analysis

# Or HIPIFY an entire CUDA program (API-level translation)
from cuda_scribe import hipify, HipifyConfig
config = HipifyConfig(target_arch="gfx942")  # MI300X
result = hipify(cuda_source, config=config)
print(result.source)             # Complete HIP C++ translation
print(result.diagnostics)        # Translation warnings/issues
```

### BridgeIR -- Universal Kernel IR

```python
from cuda_scribe import CUDALifter, translate_universal

# Lift CUDA to hardware-independent IR
lifter = CUDALifter()
kernel = lifter.lift(cuda_source)

# Then lower to ANY target
targets = translate_universal(kernel, targets=["triton", "hip", "metal", "sycl"])
print(targets["triton"])   # AMD + NVIDIA via Triton
print(targets["hip"])      # AMD native HIP C++
print(targets["metal"])    # Apple Silicon Metal shaders
print(targets["sycl"])     # Intel oneAPI SYCL
```

### AMD Hardware Analysis

```python
from cuda_scribe import compute_occupancy, roofline_analysis, select_mfma_instruction

# Wavefront occupancy (VGPR-based, AMD-specific)
occ = compute_occupancy(vgprs_per_thread=48, arch="cdna3")
# {'wavefronts_per_simd': 10, 'occupancy_pct': 100.0}

# Roofline model
analysis = roofline_analysis(flops=1e12, bytes_moved=1e10, arch="cdna3")
# {'bound': 'compute', 'achieved_pct': 76.5, 'peak_tflops': 1307}

# MFMA instruction selection (AMD Matrix Fused Multiply-Add)
instr = select_mfma_instruction(dtype="fp16", m=32, n=32)
# 'v_mfma_f32_32x32x8_f16'
```

## CLI

```bash
# Analyze kernel complexity before translation
rocm-scribe analyze kernel.cu

# Translate CUDA to Triton (runs on AMD + NVIDIA)
rocm-scribe translate kernel.cu --backend amd

# HIPIFY a CUDA source file with full API translation
rocm-scribe hipify kernel.cu --arch gfx942

# Get AMD hardware optimization recommendations
rocm-scribe optimize kernel.cu --target mi300x

# Show capabilities and supported hardware
rocm-scribe info
```

## How It Works

### The Problem with Existing Tools

| Tool | Approach | Limitation |
|---|---|---|
| `hipify-clang` | String-level API replacement | Doesn't translate kernels, only API calls |
| GEAK | LLM reflexion loop | 54--63% accuracy, no verification |
| Academic methods | One-shot generation | 100% failure on quantization kernels |
| Manual porting | Engineer rewrites | $72K--$205K per model, weeks of work |

### ROCm Scribe's Approach

ROCm Scribe doesn't do string replacement. It lifts CUDA into **BridgeIR**, a block-level intermediate representation that captures *what the kernel computes* rather than *how it's expressed in CUDA*. Then it lowers to any target with hardware-specific optimizations.

```
CUDA Source (.cu)
    |
    v
[Semantic Analysis] -- Parse AST, extract thread model, memory patterns,
    |                   synchronization topology, reduction structure
    v
[BridgeIR] -- Hardware-independent kernel representation
    |          Block-level ops on typed tensor tiles
    |          Explicit memory hierarchy (register -> shared -> global)
    |          Parametric parallelism (symbolic grid/block dims)
    |
    +---> [Triton Backend] -- Portable: runs on AMD + NVIDIA
    |         Wavefront-aware block sizes (64 vs 32)
    |         AMD-specific autotuning configs
    |
    +---> [HIP Backend] -- AMD native C++
    |         316 API mappings (cuBLAS->rocBLAS, cuDNN->MIOpen, etc.)
    |         MFMA instruction targeting
    |         Wavefront reduction (6 steps vs 5)
    |
    +---> [Metal Backend] -- Apple Silicon (M1/M2/M3/M4)
    +---> [Pallas Backend] -- Google TPU / JAX
    +---> [SYCL Backend] -- Intel GPUs / oneAPI
    +---> [WGSL Backend] -- WebGPU (browser)
```

### 7-Stage Translation Pipeline

| Stage | What It Does | Edition |
|---|---|---|
| 0. Semantic Analysis | Parses CUDA AST, extracts thread model, memory access patterns, sync topology | Open Source |
| 1. Pattern Matching | Matches against 47 verified CUDA-to-Triton translation templates | Open Source |
| 2. LLM Translation | Multi-model semantic-enriched translation (LLM ensemble) | Pro |
| 3. Static Verification | Triton IR compilation, memory bounds checking, mask completeness | Open Source |
| 4. Dynamic Verification | Cross-backend ULP numeric comparison with 50+ run stochastic variance | Pro |
| 5. Performance Optimization | Roofline analysis, wavefront-aware autotuning, bandwidth maximization | Open Source |
| 6. Evolutionary Repair | Population-based fix generation with tournament selection | Pro |
| 7. Pattern Learning | Successful translations feed pattern library, failures feed antibodies | Pro |

The open-source edition gives you stages 0, 1, 3, and 5 -- enough to translate standard kernels and optimize for AMD hardware. **ROCm Scribe Pro** adds LLM-assisted translation, cross-backend verification, evolutionary repair for the hard cases, and continuous learning. [Contact us](mailto:mike@theaicowboys.com) for enterprise licensing.

## HIPIFY -- Complete API Translation

316 CUDA-to-HIP API mappings covering the full ROCm stack:

| Library | CUDA | ROCm/HIP | Mappings |
|---|---|---|---|
| **Runtime** | `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, ... | `hipMalloc`, `hipMemcpy`, `hipDeviceSynchronize`, ... | 96 |
| **Types** | `cudaError_t`, `cudaStream_t`, `dim3`, ... | `hipError_t`, `hipStream_t`, `dim3`, ... | 45 |
| **BLAS** | cuBLAS | rocBLAS | 50 |
| **DNN** | cuDNN | MIOpen | 20 |
| **FFT** | cuFFT | rocFFT | 15+ |
| **Sparse** | cuSPARSE | rocSPARSE | 15+ |
| **RNG** | cuRAND | rocRAND | 15+ |
| **Collectives** | NCCL | RCCL | 11 |
| **Sorting/Scan** | CUB | hipCUB | 15+ |
| **Thrust** | Thrust | rocThrust | (drop-in) |

```python
from cuda_scribe import hipify, HipifyConfig, get_translation_coverage

# Check current coverage
coverage = get_translation_coverage()
print(f"Total API mappings: {coverage['total_mappings']}")

# Translate with diagnostics
config = HipifyConfig(target_arch="gfx942")
result = hipify(cuda_source, config=config)

for diag in result.diagnostics:
    print(f"[{diag.level.value}] Line {diag.line}: {diag.message}")
```

## AMD Hardware Optimization

ROCm Scribe doesn't just translate -- it **optimizes for AMD's architecture**. Every translation is wavefront-aware, MFMA-targeted, and LDS-optimized.

### Wavefront-Aware Translation

AMD GPUs use 64-wide wavefronts (CDNA) vs NVIDIA's 32-wide warps. This affects everything:

- **Reduction steps**: 6 iterations (log2 64) vs 5 (log2 32)
- **Shared memory access**: Different bank conflict patterns
- **Ballot operations**: 64-bit masks vs 32-bit
- **Block size selection**: Multiples of 64 for full occupancy

```python
from cuda_scribe import convert_warp_to_wavefront

# Automatically adjust warp-level code for AMD wavefronts
amd_source = convert_warp_to_wavefront(cuda_source)
```

### MFMA Instruction Targeting

AMD's Matrix Fused Multiply-Add instructions (equivalent of NVIDIA Tensor Cores):

| Instruction | Input | Output | Throughput/CU |
|---|---|---|---|
| `v_mfma_f32_32x32x8_f16` | FP16 | FP32 | 512 ops/cycle |
| `v_mfma_f32_16x16x16_f16` | FP16 | FP32 | 512 ops/cycle |
| `v_mfma_f32_32x32x16_bf16` | BF16 | FP32 | 512 ops/cycle |
| `v_mfma_f32_32x32x16_fp8` | FP8 | FP32 | 1024 ops/cycle |
| `v_mfma_i32_32x32x16_i8` | INT8 | INT32 | 1024 ops/cycle |

```python
from cuda_scribe import select_mfma_instruction, MFMA_INSTRUCTIONS

# Select optimal MFMA for your workload
instr = select_mfma_instruction(dtype="fp16", m=32, n=32)

# Full instruction reference
for name, spec in MFMA_INSTRUCTIONS.items():
    print(f"{name}: {spec}")
```

### Occupancy Calculator

AMD occupancy is VGPR-limited. ROCm Scribe calculates wavefront occupancy per-SIMD:

| VGPRs/Wavefront | Max Wavefronts/SIMD | Occupancy |
|---|---|---|
| 48 | 10 | 100% |
| 96 | 5 | 50% |
| 128 | 4 | 40% |
| 256 | 2 | 20% |
| 512 | 1 | 10% |

```python
from cuda_scribe import compute_occupancy

occ = compute_occupancy(vgprs_per_thread=48, arch="cdna3")
print(f"Wavefronts/SIMD: {occ['wavefronts_per_simd']}")
print(f"Occupancy: {occ['occupancy_pct']}%")
```

### LDS Bank Conflict Detection

```python
from cuda_scribe import apply_lds_bank_conflict_padding

# Get padding recommendation for conflict-free LDS access
padded_size = apply_lds_bank_conflict_padding(array_size=256, element_bytes=4)
```

### Pre-Built AMD Triton Templates

5 production-ready Triton kernel templates optimized for AMD CDNA architecture:

```python
from cuda_scribe import (
    generate_triton_gemm_amd,            # Matrix multiplication
    generate_triton_flash_attention_amd,  # FlashAttention-2
    generate_triton_softmax_amd,         # Numerically stable softmax
    generate_triton_layernorm_amd,       # Fused layer normalization
    generate_triton_rope_amd,            # Rotary positional embeddings
)
```

## Supported AMD Hardware

| Architecture | GPU | GFX Target | Wavefront | LDS/CU | HBM BW | Peak FP16 |
|---|---|---|---|---|---|---|
| **CDNA3** | MI300X | gfx942 | 64 | 64KB | 5.3 TB/s | 1,307 TFLOPS |
| **CDNA2** | MI250X | gfx90a | 64 | 64KB | 3.2 TB/s | 383 TFLOPS |
| **CDNA1** | MI100 | gfx908 | 64 | 64KB | 1.2 TB/s | 185 TFLOPS |
| **RDNA3** | RX 7900 XTX | gfx1100 | 32/64 | 128KB | 960 GB/s | 123 TFLOPS |
| **RDNA2** | RX 6900 XT | gfx1030 | 32/64 | 128KB | 512 GB/s | 46 TFLOPS |

### Recommended Compiler Flags

**MI300X (gfx942)**
```bash
hipcc --offload-arch=gfx942 -O3 -ffast-math \
      -mllvm -amdgpu-early-inline-all=true \
      -mllvm -amdgpu-function-calls=false \
      kernel.hip.cpp -o kernel
```

**MI250X (gfx90a)**
```bash
hipcc --offload-arch=gfx90a -O3 -ffast-math kernel.hip.cpp -o kernel
```

## Multi-Target Backends

BridgeIR generates optimized code for 6 targets beyond AMD:

| Backend | Target Hardware | Use Case |
|---|---|---|
| **Triton** | NVIDIA + AMD GPUs | Production inference/training (portable) |
| **HIP** | AMD GPUs (native) | Maximum AMD performance |
| **Metal** | Apple M1/M2/M3/M4 | macOS/iOS GPU compute |
| **Pallas** | Google TPU + JAX | Cloud TPU workloads |
| **SYCL** | Intel GPUs + oneAPI | Intel datacenter GPUs |
| **WGSL** | WebGPU (browsers) | Client-side GPU compute |

```python
from cuda_scribe import translate_universal, CUDALifter

lifter = CUDALifter()
kernel = lifter.lift(cuda_source)
results = translate_universal(kernel, targets=["triton", "hip", "metal", "pallas", "sycl", "wgsl"])

for target, code in results.items():
    print(f"--- {target} ---")
    print(code[:200])
```

## Benchmarks

### Translation Accuracy (L40S GPU, verified July 2026)

| Kernel | Throughput | Roofline % | Max Numeric Error |
|---|---|---|---|
| VectorAdd 16M elements | 633.7 GB/s | 73.3% | **0.0** (exact) |
| Softmax 8K x 2K | 709.0 GB/s | 82.1% | 7.45e-9 |
| LayerNorm 4K x 1K | 1.30x vs PyTorch | -- | 9.54e-7 |
| GEMM 2K x 2K FP16 | 132.3 TFLOPS | 36.5% | **0.0** (exact) |
| GEMM 4K x 4K FP16 | 153.5 TFLOPS | 42.4% | **0.0** (exact) |

### Metal GPU Verification (Apple M2 Max)

| Kernel | Throughput | Max Error |
|---|---|---|
| VectorAdd 1M | 255.8 GB/s | **0.0** (exact) |
| RoPE [2,128,8,64] | 38.4 us | 3.16e-5 (265 ULPs -- normal for cross-platform trig) |

### Test Suite

- **134 tests** -- AMD Bridge (hipify, wavefront optimizer, kernel optimizer, MFMA)
- **37 tests** -- BridgeIR multi-target translation
- **48 tests** -- Core translation pipeline
- **4 tests** -- Metal GPU on-device verification
- **219 total** -- All passing

## Comparison: ROCm Scribe vs Alternatives

| Feature | hipify-clang | GEAK | Academic | **ROCm Scribe** |
|---|---|---|---|---|
| API mapping (CUDA to HIP) | Yes | No | No | **Yes (316 mappings)** |
| Kernel translation | No | Yes | Yes | **Yes (BridgeIR)** |
| Translation accuracy | N/A | 54--63% | Varies | **Verified correct** |
| Wavefront optimization | No | No | No | **Yes (64-wide)** |
| MFMA instruction targeting | No | No | No | **Yes (10 instructions)** |
| Occupancy analysis | No | No | No | **Yes (VGPR-based)** |
| Triton output (portable) | No | Yes | Yes | **Yes** |
| Multi-target (6 backends) | No | No | No | **Yes** |
| Numeric verification | No | No | No | **Pro** |
| Quantization kernels | No | No | 0% success | **Pro** |
| Evolutionary repair | No | No | No | **Pro** |
| Cross-platform | Linux only | Linux only | Linux only | **Linux, macOS, Windows** |

## Examples

The `examples/` directory contains 5 complete CUDA kernels:

| File | Pattern | Difficulty |
|---|---|---|
| `vector_add.cu` | Elementwise map | Trivial |
| `reduction.cu` | Parallel reduction with shared memory | Medium |
| `softmax.cu` | Warp shuffle + numerically stable softmax | Hard |
| `fused_layernorm.cu` | Fused normalization | Medium |
| `transpose.cu` | Shared memory tiled transpose | Medium |

```bash
rocm-scribe translate examples/vector_add.cu --backend amd
rocm-scribe translate examples/softmax.cu --backend both
```

## Platform Support

ROCm Scribe runs on **Linux, macOS, and Windows**. The translation pipeline is pure Python with no OS-specific dependencies — translate CUDA kernels on any workstation and deploy to AMD hardware.

| Platform | Translation | On-Device Compilation | Notes |
|---|---|---|---|
| **Linux** | Yes | Yes (ROCm + hipcc) | Full pipeline including AMD GPU execution |
| **macOS** | Yes | Metal backend only | Translate + verify on Apple Silicon |
| **Windows** | Yes | Yes (ROCm via WSL2) | Native Python, GPU via WSL2 |

```bash
# Works the same on all platforms
pip install rocm-scribe
rocm-scribe translate kernel.cu --backend amd
```

## Installation

### From PyPI
```bash
pip install rocm-scribe
```

### From Source
```bash
git clone https://github.com/iaintheardofu/rocm-scribe.git
cd rocm-scribe
pip install -e ".[dev]"
pytest
```

### Optional Dependencies
```bash
pip install rocm-scribe[triton]    # + Triton for GPU execution
pip install rocm-scribe[torch]     # + PyTorch
pip install rocm-scribe[all]       # Everything
```

## Pro

**ROCm Scribe Pro** includes the full 7-stage pipeline for production CUDA migration:

- **LLM-Assisted Translation** -- Multi-model ensemble with semantic context injection
- **Dynamic Verification** -- Cross-backend ULP-based numeric comparison (50+ runs)
- **Evolutionary Repair** -- Population-based fix generation for kernels that fail initial translation
- **Pattern Learning** -- The system improves with every translation (pattern library + failure antibodies)
- **Quantization Patterns** -- W4A16, W8A8, FP8, SmoothQuant kernel translation
- **Enterprise Support** -- SLA-backed migration consulting, private pattern libraries, fleet tracking

**Contact:** [mike@theaicowboys.com](mailto:mike@theaicowboys.com)

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md).

High-impact areas:
- CUDA pattern templates (expand Stage 1 pattern library)
- HIP API mappings (increase HIPIFY coverage)
- Backend improvements (Metal, SYCL, WGSL, Pallas)
- Complex CUDA test kernels
- AMD hardware benchmark results

## License

Apache License 2.0 -- See [LICENSE](LICENSE).

## About

Built by [The AI Cowboys](https://theaicowboys.com) -- San Antonio, TX.

ROCm Scribe exists because migrating from CUDA shouldn't cost $200K per model. We believe in an open GPU ecosystem where hardware is a choice, not a prison. AMD's ROCm platform is the path to open AI compute, and this tool helps you get there.
