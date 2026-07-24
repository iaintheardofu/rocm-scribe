#!/usr/bin/env python3
"""
HIP/ROCm Backend — AMD GPU kernel generation from BridgeIR.

Translates BridgeIR kernels to HIP C++ code that runs on:
  - AMD MI300X / MI250X (CDNA architecture, 64-wide wavefronts)
  - AMD Instinct MI100 / MI200 series
  - AMD Radeon RX 7900 XTX (RDNA 3)
  - NVIDIA GPUs (HIP is source-compatible with CUDA)

HIP mapping from CUDA:
  - hipLaunchKernelGGL ≈ <<<grid, block>>>
  - __shared__ → __shared__ (same syntax)
  - __syncthreads() → __syncthreads() (same)
  - atomicAdd → atomicAdd (same)
  - warpSize → warpSize (but 64 on AMD, 32 on NVIDIA)
  - __shfl_down_sync → __shfl_down (no mask on AMD)
  - __ballot_sync → __ballot (no mask on AMD)
  - cudaMalloc → hipMalloc
  - cudaMemcpy → hipMemcpy

Key AMD wavefront differences (CRITICAL for performance):
  - Wavefront = 64 threads (vs NVIDIA warp = 32 threads)
  - LDS (Local Data Share) = 64KB per CU (vs 48-100KB shared on NVIDIA)
  - AMD Matrix Core (CDNA) = MFMA instructions
  - HBM3 bandwidth on MI300X: 5.3 TB/s (vs 3.35 TB/s on H100)
  - Occupancy: different register file sizes, different LDS constraints

Algorithmica HPC principles applied:
  - Memory hierarchy: registers → LDS (shared) → L1 → L2 → HBM
  - Tiling for cache-oblivious GPU patterns (Section 8.7)
  - SoA layout for coalesced memory access (Section 9.11)
  - Branchless programming for wavefront divergence reduction (Section 3.3)
  - SIMD reduction patterns mapped to wavefront-wide operations (Section 10.3)

References:
  - https://rocm.docs.amd.com/
  - https://github.com/ROCm/HIPIFY
  - https://en.algorithmica.org/hpc/
"""
from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List

from ..bridge_ir import (
    BackendPlugin,
    BridgeKernel,
    DType,
    HardwareRequirement,
    ParallelPattern,
    register_backend,
)


class HIPBackend(BackendPlugin):
    """Generate HIP C++ code from BridgeIR for AMD GPUs."""

    # BridgeIR DType → HIP/C++ type
    DTYPE_MAP = {
        DType.FLOAT16: "half",           # hip_fp16.h
        DType.BFLOAT16: "hip_bfloat16",  # hip_bfloat16.h
        DType.FLOAT32: "float",
        DType.FLOAT64: "double",
        DType.INT8: "int8_t",
        DType.INT16: "int16_t",
        DType.INT32: "int32_t",
        DType.INT64: "int64_t",
        DType.UINT8: "uint8_t",
        DType.UINT16: "uint16_t",
        DType.UINT32: "uint32_t",
        DType.UINT64: "uint64_t",
        DType.BOOL: "bool",
    }

    # CUDA math → HIP math (most are identical, but document explicitly)
    MATH_MAP = {
        "cosf": "cosf", "sinf": "sinf", "expf": "expf", "logf": "logf",
        "sqrtf": "sqrtf", "fabsf": "fabsf", "powf": "powf", "tanhf": "tanhf",
        "fmaxf": "fmaxf", "fminf": "fminf",
        # HIP-specific fast math
        "__cosf": "__cosf", "__sinf": "__sinf", "__expf": "__expf",
    }

    # CUDA API → HIP API
    CUDA_TO_HIP_API = {
        "cudaMalloc": "hipMalloc",
        "cudaFree": "hipFree",
        "cudaMemcpy": "hipMemcpy",
        "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
        "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
        "cudaDeviceSynchronize": "hipDeviceSynchronize",
        "cudaGetLastError": "hipGetLastError",
        "cudaStreamCreate": "hipStreamCreate",
        "cudaStreamSynchronize": "hipStreamSynchronize",
        "cudaEventCreate": "hipEventCreate",
        "cudaEventRecord": "hipEventRecord",
        "cudaEventElapsedTime": "hipEventElapsedTime",
        "cudaError_t": "hipError_t",
        "cudaSuccess": "hipSuccess",
        "cudaGetDeviceProperties": "hipGetDeviceProperties",
        "cudaDeviceProp": "hipDeviceProp_t",
    }

    # Warp intrinsics: CUDA (masked) → HIP (no mask on AMD)
    WARP_INTRINSICS = {
        "__shfl_down_sync": "__shfl_down",
        "__shfl_up_sync": "__shfl_up",
        "__shfl_xor_sync": "__shfl_xor",
        "__shfl_sync": "__shfl",
        "__ballot_sync": "__ballot",
        "__any_sync": "__any",
        "__all_sync": "__all",
        "__activemask": "__ballot(1)",
    }

    def name(self) -> str:
        return "hip"

    def supported_hardware(self) -> List[str]:
        return ["amd", "nvidia"]  # HIP is portable to both

    def translate(self, kernel: BridgeKernel, cuda_source: str = "") -> str:
        """Generate HIP code from BridgeIR."""
        tensor_params = [p for p in kernel.params if p.is_tensor]
        scalar_params = [p for p in kernel.params if not p.is_tensor]

        # Build kernel function signature
        func_params = []
        for p in tensor_params:
            hip_type = self.DTYPE_MAP.get(
                p.tile_type.dtype if p.tile_type else DType.FLOAT32, "float"
            )
            if p.is_output:
                func_params.append(f"    {hip_type}* __restrict__ {p.name}")
            else:
                func_params.append(f"    const {hip_type}* __restrict__ {p.name}")
        for p in scalar_params:
            hip_type = self.DTYPE_MAP.get(p.scalar_dtype or DType.INT32, "int32_t")
            func_params.append(f"    const {hip_type} {p.name}")

        # Build kernel body
        body = self._translate_body(kernel, cuda_source)

        # Wavefront-aware block size
        block_size = kernel.block_size if isinstance(kernel.block_size, int) else 256
        # AMD wavefronts are 64-wide: prefer multiples of 64 for full occupancy
        if block_size % 64 != 0:
            block_size = ((block_size + 63) // 64) * 64
            if block_size > 1024:
                block_size = 1024

        # Determine includes
        includes = ["#include <hip/hip_runtime.h>"]
        if any(p.tile_type and p.tile_type.dtype == DType.FLOAT16
               for p in kernel.params if p.tile_type):
            includes.append("#include <hip/hip_fp16.h>")
        if any(p.tile_type and p.tile_type.dtype == DType.BFLOAT16
               for p in kernel.params if p.tile_type):
            includes.append("#include <hip/hip_bfloat16.h>")

        # Shared memory declaration if needed
        shared_decl = ""
        if HardwareRequirement.SHARED_MEMORY in kernel.hardware_requirements:
            shared_decl = self._generate_shared_memory(kernel)

        # Wavefront-aware optimization comments
        opt_comments = self._generate_optimization_comments(kernel)

        code = textwrap.dedent(f"""\
        // Bridge v2.0 — HIP/ROCm translation of {kernel.name}
        // Pattern: {kernel.pattern.value}
        // Target: AMD MI300X/MI250X (CDNA) + Radeon (RDNA)
        // Wavefront: 64 threads (AMD) / 32 threads (NVIDIA)
        // Block size: {block_size} (wavefront-aligned)
        {opt_comments}

        {chr(10).join(includes)}
        #include <cstdint>

        __global__ void {kernel.name}(
        {(',' + chr(10)).join(func_params)}
        ) {{
        {shared_decl}{textwrap.indent(body, '    ')}
        }}

        // Host launcher with wavefront-aware configuration
        void launch_{kernel.name}(
        {', '.join(self._host_param_list(kernel))},
            int total_elements,
            hipStream_t stream = 0
        ) {{
            const int block = {block_size};
            const int grid = (total_elements + block - 1) / block;
            hipLaunchKernelGGL({kernel.name}, dim3(grid), dim3(block), 0, stream,
                {', '.join(self._host_arg_list(kernel))});
        }}
        """)
        return code

    def _translate_body(self, kernel: BridgeKernel, cuda_source: str) -> str:
        """Translate CUDA kernel body to HIP.

        HIP is ~95% source-compatible with CUDA. Key differences:
        1. Warp intrinsics lose the mask parameter (AMD wavefronts don't use masks)
        2. warpSize is 64 on AMD (vs 32 on NVIDIA) — CRITICAL for reduction kernels
        3. Cooperative groups API differs slightly
        4. Some CUDA-specific PTX intrinsics have no HIP equivalent
        """
        if not cuda_source:
            return "// TODO: LLM-assisted body translation"

        body_match = re.search(r'\{(.*)\}', cuda_source, re.DOTALL)
        if not body_match:
            return "// Could not extract kernel body"

        body = body_match.group(1).strip()

        # === CUDA → HIP warp intrinsic translation ===
        # Remove mask parameter from warp intrinsics
        # __shfl_down_sync(0xffffffff, val, offset) → __shfl_down(val, offset)
        for cuda_fn, hip_fn in self.WARP_INTRINSICS.items():
            # Pattern: __shfl_down_sync(MASK, value, offset[, width])
            body = re.sub(
                rf'{re.escape(cuda_fn)}\s*\(\s*(?:0x[0-9a-fA-F]+|~0|0xFFFFFFFF|\w+)\s*,\s*',
                f'{hip_fn}(',
                body
            )
            # Also handle cases where the function is used without mask (already HIP-compatible)
            body = body.replace(cuda_fn, hip_fn)

        # === CUDA API → HIP API ===
        for cuda_api, hip_api in self.CUDA_TO_HIP_API.items():
            body = body.replace(cuda_api, hip_api)

        # === Wavefront-aware warpSize handling ===
        # Replace hardcoded warp size of 32 with warpSize (which is 64 on AMD)
        # This is critical for reduction kernels and shuffle operations
        body = re.sub(
            r'\b32\b(?=\s*(?://.*warp|/\*.*warp))',
            'warpSize',
            body
        )

        # === CUDA cooperative groups → HIP cooperative groups ===
        body = body.replace(
            'cooperative_groups::this_thread_block()',
            'cooperative_groups::this_thread_block()'  # Same in HIP
        )

        # === Handle CUDA __launch_bounds__ → HIP __launch_bounds__ (same syntax) ===
        # No change needed — HIP supports __launch_bounds__ natively

        # === Memory fence instructions ===
        body = body.replace('__threadfence_block()', '__threadfence_block()')
        body = body.replace('__threadfence()', '__threadfence()')

        return body

    def _generate_shared_memory(self, kernel: BridgeKernel) -> str:
        """Generate shared memory declarations.

        AMD LDS (Local Data Share) is 64KB per CU on MI300X.
        Algorithmica principle: maximize LDS utilization for tiling (Section 8.7).
        """
        return "    // AMD LDS: 64KB per CU — use for tiling and reductions\n"

    def _generate_optimization_comments(self, kernel: BridgeKernel) -> str:
        """Generate AMD-specific optimization hints based on kernel pattern."""
        comments = []

        if kernel.pattern == ParallelPattern.REDUCE:
            comments.append(
                "// AMD Optimization: Use wavefront-wide __shfl_down for first 64 elements\n"
                "// (vs 32 on NVIDIA). Single wavefront reduction handles 64 values."
            )
        elif kernel.pattern == ParallelPattern.MATMUL:
            comments.append(
                "// AMD Optimization: Use MFMA (Matrix Fused Multiply-Add) instructions\n"
                "// on CDNA. MI300X: 1307.4 TFLOPS FP8, 653.7 TFLOPS FP16.\n"
                "// Tile sizes: 16x16, 32x32 for MFMA; 32x32x8 for FP16 MFMA."
            )
        elif kernel.pattern == ParallelPattern.MAP:
            comments.append(
                "// AMD Optimization: Coalesced access width = 64 threads (wavefront)\n"
                "// Ensure consecutive threads access consecutive memory addresses.\n"
                "// MI300X HBM3 bandwidth: 5.3 TB/s (58% more than H100 3.35 TB/s)."
            )
        elif kernel.pattern == ParallelPattern.ATTENTION:
            comments.append(
                "// AMD Optimization: FlashAttention-2 available via composable_kernel library.\n"
                "// Use ck::tensor_operation for fused attention on CDNA.\n"
                "// MI300X: 192GB HBM3 unified memory (vs 80GB on H100)."
            )

        if HardwareRequirement.SHARED_MEMORY in kernel.hardware_requirements:
            comments.append(
                "// LDS Tiling (Algorithmica §8.7): AMD LDS is 64KB per CU.\n"
                "// Tile dimensions should maximize LDS reuse while fitting constraints.\n"
                "// Use __builtin_amdgcn_ds_permute for LDS-level data exchange."
            )

        if HardwareRequirement.ATOMICS in kernel.hardware_requirements:
            comments.append(
                "// AMD Atomics: Global atomics use PCIe/Infinity Fabric.\n"
                "// Prefer LDS atomics (__shared__ atomicAdd) where possible.\n"
                "// MI300X uses Infinity Fabric for cross-chiplet atomics."
            )

        return "\n".join(comments) if comments else ""

    def _host_param_list(self, kernel: BridgeKernel) -> List[str]:
        """Generate host function parameter list."""
        params = []
        for p in kernel.params:
            if p.is_tensor:
                hip_type = self.DTYPE_MAP.get(
                    p.tile_type.dtype if p.tile_type else DType.FLOAT32, "float"
                )
                params.append(f"{'const ' if not p.is_output else ''}{hip_type}* {p.name}")
            else:
                hip_type = self.DTYPE_MAP.get(p.scalar_dtype or DType.INT32, "int32_t")
                params.append(f"{hip_type} {p.name}")
        return params

    def _host_arg_list(self, kernel: BridgeKernel) -> List[str]:
        """Generate host function argument list for kernel launch."""
        return [p.name for p in kernel.params]

    def verify(self, source: str, reference=None) -> Dict[str, Any]:
        """Verify HIP source by attempting compilation with hipcc."""
        import shutil
        if not shutil.which("hipcc"):
            return {"status": "skipped", "note": "hipcc not found (ROCm not installed)"}
        try:
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
                f.write(source)
                f.flush()
                result = subprocess.run(
                    ["hipcc", "--syntax-only", f.name],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    return {"status": "pass"}
                return {"status": "fail", "error": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def benchmark(self, source: str) -> Dict[str, Any]:
        return {"status": "not_implemented", "note": "Requires AMD GPU + ROCm runtime"}

    def is_hardware_available(self) -> bool:
        """Check for AMD GPU via ROCm."""
        import shutil
        return shutil.which("hipcc") is not None


# --- Wavefront-Aware Optimization Passes ---

def wavefront_optimize_reduction(cuda_source: str) -> str:
    """Optimize reduction kernels for 64-wide AMD wavefronts.

    Algorithmica §10.3: SIMD reductions should match the hardware vector width.
    On AMD, a single wavefront reduces 64 values (vs 32 on NVIDIA).

    Before (CUDA/NVIDIA-tuned):
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);

    After (AMD wavefront-optimized):
        for (int offset = 32; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
    """
    # Replace warp-level reduction starting from 16 with wavefront-level from 32
    result = re.sub(
        r'for\s*\(\s*int\s+(\w+)\s*=\s*16\s*;',
        r'for (int \1 = warpSize/2;',  # warpSize = 64 on AMD
        cuda_source
    )
    return result


def wavefront_optimize_shared_memory(cuda_source: str, pattern: ParallelPattern) -> str:
    """Optimize shared memory usage for AMD LDS constraints.

    AMD MI300X: 64KB LDS per CU, shared among active wavefronts.
    Algorithmica §8.1: Memory hierarchy optimization — maximize locality.

    Key insight: With 64-wide wavefronts, shared memory bank conflicts
    differ from NVIDIA. AMD has 32 LDS banks (same as NVIDIA), but
    64 threads access simultaneously → potential for 2x more conflicts.
    Padding shared memory arrays by 1 element per bank mitigates this.
    """
    if pattern in (ParallelPattern.REDUCE, ParallelPattern.MATMUL):
        # Add bank conflict avoidance padding
        # __shared__ float smem[256] → __shared__ float smem[256 + 1]
        result = re.sub(
            r'__shared__\s+(\w+)\s+(\w+)\[(\d+)\]',
            lambda m: f'__shared__ {m.group(1)} {m.group(2)}[{m.group(3)} + 1]  '
                      f'// +1 padding for AMD LDS bank conflict avoidance',
            cuda_source
        )
        return result
    return cuda_source


def generate_hip_reduction_kernel(dtype: str = "float", op: str = "add") -> str:
    """Generate an optimized wavefront-aware reduction kernel.

    Applies Algorithmica HPC principles:
    - §10.3: SIMD reductions with proper vector width
    - §3.3: Branchless programming for divergence reduction
    - §9.11: SoA layout for coalesced memory access
    - §8.7: Cache-oblivious tiling for LDS utilization
    """
    op_symbol = {"add": "+", "max": "max", "min": "min"}.get(op, "+")
    op_identity = {"add": "0", "max": f"-INFINITY", "min": f"INFINITY"}.get(op, "0")

    if op in ("max", "min"):
        combine = f"val = f{op}f(val, other);"
    else:
        combine = f"val {op_symbol}= other;"

    return textwrap.dedent(f"""\
    // Bridge v2.0 — Wavefront-optimized {op} reduction for AMD ROCm
    // Algorithmica HPC: §10.3 SIMD reductions, §3.3 branchless, §8.7 cache-oblivious
    #include <hip/hip_runtime.h>

    __device__ __forceinline__ {dtype} warp_reduce_{op}({dtype} val) {{
        // AMD wavefront = 64 threads: reduce in log2(64) = 6 steps
        // NVIDIA warp = 32 threads: only 5 steps needed
        // Using warpSize makes this portable across both
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {{
            {dtype} other = __shfl_down(val, offset);
            {combine}
        }}
        return val;
    }}

    __global__ void reduce_{op}(
        const {dtype}* __restrict__ input,
        {dtype}* __restrict__ output,
        const int n
    ) {{
        // Phase 1: Coalesced global load with grid-stride loop
        // Algorithmica §9.1: maximize memory bandwidth utilization
        {dtype} val = {op_identity};
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < n;
             i += blockDim.x * gridDim.x) {{
            {dtype} loaded = input[i];
            {combine.replace('other', 'loaded')}
        }}

        // Phase 2: Wavefront-level reduction (no shared memory needed)
        val = warp_reduce_{op}(val);

        // Phase 3: Cross-wavefront reduction via LDS
        // AMD: blockDim.x / 64 wavefronts per block
        // NVIDIA: blockDim.x / 32 warps per block
        __shared__ {dtype} shared[32];  // max 32 wavefronts per block (2048/64)
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;

        if (lane == 0) shared[wid] = val;
        __syncthreads();

        // Final reduction: only first wavefront participates
        int num_warps = blockDim.x / warpSize;
        val = (threadIdx.x < num_warps) ? shared[lane] : ({dtype}){op_identity};
        if (wid == 0) val = warp_reduce_{op}(val);

        // Branchless write (Algorithmica §3.3)
        if (threadIdx.x == 0) atomicAdd(&output[0], val);
    }}
    """)


def generate_hip_gemm_kernel() -> str:
    """Generate a tiled GEMM kernel optimized for AMD CDNA.

    Applies Algorithmica HPC principles:
    - §11.15: Matrix multiplication tiling and register blocking
    - §8.7: Cache-oblivious blocking for LDS
    - §9.11: SoA-style layout for coalesced access
    - §10.3: Wavefront-level reductions for partial sums
    """
    return textwrap.dedent("""\
    // Bridge v2.0 — Tiled GEMM for AMD CDNA (MI300X/MI250X)
    // Algorithmica HPC: §11.15 matrix mult, §8.7 cache-oblivious, §9.11 SoA
    // For production use, prefer rocBLAS or composable_kernel library.
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>

    #define TILE_M 64
    #define TILE_N 64
    #define TILE_K 16

    __global__ void gemm_tiled(
        const float* __restrict__ A,  // [M, K]
        const float* __restrict__ B,  // [K, N]
        float* __restrict__ C,        // [M, N]
        const int M, const int N, const int K
    ) {
        // LDS tiles — padded for bank conflict avoidance (AMD: 32 banks, 64-wide access)
        __shared__ float As[TILE_M][TILE_K + 1];  // +1 padding
        __shared__ float Bs[TILE_K][TILE_N + 1];  // +1 padding

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int bx = blockIdx.x * TILE_N;
        const int by = blockIdx.y * TILE_M;

        float acc = 0.0f;

        // Tile over K dimension
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            // Cooperative load of A tile into LDS
            if ((by + ty) < M && (k0 + tx) < K)
                As[ty][tx] = A[(by + ty) * K + (k0 + tx)];
            else
                As[ty][tx] = 0.0f;

            // Cooperative load of B tile into LDS
            if ((k0 + ty) < K && (bx + tx) < N)
                Bs[ty][tx] = B[(k0 + ty) * N + (bx + tx)];
            else
                Bs[ty][tx] = 0.0f;

            __syncthreads();

            // Compute partial dot product from LDS
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                acc += As[ty][k] * Bs[k][tx];
            }

            __syncthreads();
        }

        // Store result with bounds check
        if ((by + ty) < M && (bx + tx) < N) {
            C[(by + ty) * N + (bx + tx)] = acc;
        }
    }
    """)


register_backend(HIPBackend())
