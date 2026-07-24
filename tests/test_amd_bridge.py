#!/usr/bin/env python3
"""
Tests for AMD Bridge — Complete CUDA-to-AMD ROCm translation pipeline.

Tests cover:
  1. HIPIFY translation engine (CUDA→HIP API mapping)
  2. AMD wavefront optimizer (occupancy, roofline, block size)
  3. AMD kernel optimizer (divergence elimination, LDS padding, coalescing)
  4. HIP backend (BridgeIR → HIP C++ generation)
  5. AMD bridge orchestrator (end-to-end CUDA → AMD)
  6. Triton AMD kernel generation (map, reduce, matmul, softmax, rope)
  7. Quantization patterns for AMD
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cuda_scribe.bridge_ir import (
    BridgeKernel, CUDALifter, DType, HardwareRequirement,
    MemoryLevel, ParallelPattern, TileType,
    available_backends, get_backend, translate_universal,
)

from cuda_scribe.hipify import (
    hipify, HipifyConfig, HipifyResult, HipifyWarningLevel,
    get_translation_coverage,
    RUNTIME_API_MAP, TYPE_MAP, HEADER_MAP,
    CUBLAS_TO_ROCBLAS, CUDNN_TO_MIOPEN,
)

from cuda_scribe.backends.amd_wavefront_optimizer import (
    AMDArch, AMD_SPECS, AMDHardwareSpec,
    compute_occupancy, roofline_analysis, optimize_block_size,
    apply_lds_bank_conflict_padding, convert_warp_to_wavefront,
    generate_triton_amd_config,
)

from cuda_scribe.backends.amd_kernel_optimizer import (
    optimize_hip_kernel, OptimizationLevel, OptimizationReport,
    generate_triton_autotune_configs,
    generate_triton_softmax_amd, generate_triton_layernorm_amd,
    generate_triton_rope_amd,
)

from cuda_scribe.amd_bridge import (
    AMDBridge, AMDTranslationResult, TranslationPath,
)

# Trigger backend registration
import cuda_scribe.backends.hip_backend


# ═══════════════════════════════════════════════════════════════════
# TEST CUDA SOURCES
# ═══════════════════════════════════════════════════════════════════

VECTOR_ADD_CUDA = """
__global__ void vector_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    c[tid] = a[tid] + b[tid];
}
"""

REDUCTION_CUDA = """
__global__ void sum_reduce(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    __shared__ float shared[256];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ltid = threadIdx.x;

    shared[ltid] = (tid < n) ? input[tid] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (ltid < stride) {
            shared[ltid] += shared[ltid + stride];
        }
        __syncthreads();
    }

    if (ltid == 0) {
        atomicAdd(&output[0], shared[0]);
    }
}
"""

WARP_REDUCTION_CUDA = """
__global__ void warp_reduce(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    float val = input[blockIdx.x * blockDim.x + threadIdx.x];

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (threadIdx.x % 32 == 0) // warp lane 0
        atomicAdd(output, val);
}
"""

CUBLAS_CUDA = """
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gemm(float* A, float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, &alpha, A, M, B, K, &beta, C, M);
    cublasDestroy(handle);
}
"""

FUSED_ROPE_CUDA = """
__global__ void fused_rope_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ positions,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float base,
    const float scaling_factor
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    if (tid >= total_pairs) return;

    const int half_dim = head_dim / 2;
    const int pair_idx = tid % half_dim;
    const int head_idx = (tid / half_dim) % num_heads;
    const int seq_idx  = (tid / (half_dim * num_heads)) % seq_len;
    const int batch_idx = tid / (half_dim * num_heads * seq_len);

    const float freq = 1.0f / powf(base, (2.0f * pair_idx) / (float)head_dim);
    const float scaled_freq = freq / scaling_factor;
    const int pos = positions[batch_idx * seq_len + seq_idx];
    const float angle = pos * scaled_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const int base_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
    const float x0 = input[base_offset + 2 * pair_idx];
    const float x1 = input[base_offset + 2 * pair_idx + 1];

    output[base_offset + 2 * pair_idx]     = x0 * cos_val - x1 * sin_val;
    output[base_offset + 2 * pair_idx + 1] = x0 * sin_val + x1 * cos_val;
}
"""

LAUNCH_SYNTAX_CUDA = """
#include <cuda_runtime.h>

__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void launch() {
    float *d_a, *d_b, *d_c;
    int n = 1024;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    add_kernel<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
"""

INLINE_PTX_CUDA = """
__device__ uint32_t extract_bits(uint32_t packed, int start, int num_bits) {
    uint32_t result;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(packed), "r"(start), "r"(num_bits));
    return result;
}
"""


# ═══════════════════════════════════════════════════════════════════
# HIPIFY ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestHipifyHeaders:
    def test_cuda_runtime_header(self):
        source = '#include <cuda_runtime.h>'
        result = hipify(source)
        assert "hip/hip_runtime.h" in result.source

    def test_cublas_header(self):
        source = '#include <cublas_v2.h>'
        result = hipify(source)
        assert "rocblas/rocblas.h" in result.source

    def test_cufft_header(self):
        source = '#include <cufft.h>'
        result = hipify(source)
        assert "rocfft/rocfft.h" in result.source

    def test_cudnn_header(self):
        source = '#include <cudnn.h>'
        result = hipify(source)
        assert "miopen/miopen.h" in result.source

    def test_nccl_header(self):
        source = '#include <nccl.h>'
        result = hipify(source)
        assert "rccl/rccl.h" in result.source

    def test_curand_header(self):
        source = '#include <curand.h>'
        result = hipify(source)
        assert "rocrand/rocrand.h" in result.source

    def test_cub_header(self):
        source = '#include <cub/cub.cuh>'
        result = hipify(source)
        assert "hipcub/hipcub.hpp" in result.source


class TestHipifyRuntimeAPI:
    def test_cudaMalloc(self):
        source = 'cudaMalloc(&d_ptr, size);'
        result = hipify(source)
        assert "hipMalloc" in result.source

    def test_cudaFree(self):
        source = 'cudaFree(d_ptr);'
        result = hipify(source)
        assert "hipFree" in result.source

    def test_cudaMemcpy(self):
        source = 'cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);'
        result = hipify(source)
        assert "hipMemcpy" in result.source
        assert "hipMemcpyHostToDevice" in result.source

    def test_cudaDeviceSynchronize(self):
        source = 'cudaDeviceSynchronize();'
        result = hipify(source)
        assert "hipDeviceSynchronize" in result.source

    def test_cudaStreamCreate(self):
        source = 'cudaStream_t stream; cudaStreamCreate(&stream);'
        result = hipify(source)
        assert "hipStream_t" in result.source
        assert "hipStreamCreate" in result.source

    def test_cudaEventElapsedTime(self):
        source = 'cudaEventElapsedTime(&ms, start, stop);'
        result = hipify(source)
        assert "hipEventElapsedTime" in result.source

    def test_cudaGetDeviceProperties(self):
        source = 'cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);'
        result = hipify(source)
        assert "hipDeviceProp_t" in result.source
        assert "hipGetDeviceProperties" in result.source

    def test_cudaMallocManaged(self):
        source = 'cudaMallocManaged(&ptr, size);'
        result = hipify(source)
        assert "hipMallocManaged" in result.source

    def test_error_types(self):
        source = 'cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) {}'
        result = hipify(source)
        assert "hipError_t" in result.source
        assert "hipGetLastError" in result.source
        assert "hipSuccess" in result.source


class TestHipifyWarpIntrinsics:
    def test_shfl_down_sync(self):
        source = 'val = __shfl_down_sync(0xffffffff, val, offset);'
        result = hipify(source)
        assert "__shfl_down(" in result.source
        assert "0xffffffff" not in result.source

    def test_shfl_up_sync(self):
        source = 'val = __shfl_up_sync(0xffffffff, val, 1);'
        result = hipify(source)
        assert "__shfl_up(" in result.source

    def test_shfl_xor_sync(self):
        source = 'val = __shfl_xor_sync(~0u, val, 16);'
        result = hipify(source)
        assert "__shfl_xor(" in result.source

    def test_ballot_sync(self):
        source = 'uint32_t mask = __ballot_sync(0xFFFFFFFF, pred);'
        result = hipify(source)
        assert "__ballot(" in result.source

    def test_any_sync(self):
        source = 'if (__any_sync(0xffffffff, pred)) {}'
        result = hipify(source)
        assert "__any(" in result.source

    def test_all_sync(self):
        source = 'if (__all_sync(0xffffffff, pred)) {}'
        result = hipify(source)
        assert "__all(" in result.source

    def test_activemask(self):
        source = 'uint32_t mask = __activemask();'
        result = hipify(source)
        assert "__ballot(1)" in result.source


class TestHipifyLaunchSyntax:
    def test_simple_launch(self):
        result = hipify(LAUNCH_SYNTAX_CUDA)
        assert "hipLaunchKernelGGL" in result.source
        assert "<<<" not in result.source

    def test_launch_with_shared_memory(self):
        source = 'kernel<<<grid, block, shared_size>>>(args);'
        result = hipify(source)
        assert "hipLaunchKernelGGL" in result.source

    def test_launch_with_stream(self):
        source = 'kernel<<<grid, block, 0, stream>>>(args);'
        result = hipify(source)
        assert "hipLaunchKernelGGL" in result.source

    def test_full_cuda_program(self):
        result = hipify(LAUNCH_SYNTAX_CUDA)
        assert "hipMalloc" in result.source
        assert "hipFree" in result.source
        assert "hipDeviceSynchronize" in result.source


class TestHipifyLibraries:
    def test_cublas_to_rocblas(self):
        result = hipify(CUBLAS_CUDA)
        assert "rocblas" in result.source
        assert "rocblas_create_handle" in result.source
        assert "rocblas_sgemm" in result.source
        assert "rocblas_destroy_handle" in result.source

    def test_cublas_operations(self):
        source = 'cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha, A, M, B, K, &beta, C, M);'
        result = hipify(source)
        assert "rocblas_sgemm" in result.source
        assert "rocblas_operation_none" in result.source
        assert "rocblas_operation_transpose" in result.source

    def test_cublas_types(self):
        source = 'cublasHandle_t h; cublasStatus_t s = CUBLAS_STATUS_SUCCESS;'
        result = hipify(source)
        assert "rocblas_handle" in result.source
        assert "rocblas_status_success" in result.source

    def test_cudnn_to_miopen(self):
        source = 'cudnnHandle_t h; cudnnCreate(&h); cudnnConvolutionForward(h);'
        result = hipify(source)
        assert "miopenHandle_t" in result.source
        assert "miopenCreate" in result.source
        assert "miopenConvolutionForward" in result.source

    def test_cusparse_to_rocsparse(self):
        source = 'cusparseHandle_t h; cusparseCreate(&h);'
        result = hipify(source)
        assert "rocsparse_handle" in result.source
        assert "rocsparse_create_handle" in result.source

    def test_curand_to_rocrand(self):
        source = 'curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);'
        result = hipify(source)
        assert "rocrand_generator" in result.source
        assert "rocrand_create_generator" in result.source
        assert "ROCRAND_RNG_PSEUDO_DEFAULT" in result.source

    def test_cub_to_hipcub(self):
        source = 'cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);'
        result = hipify(source)
        assert "hipcub::DeviceReduce" in result.source


class TestHipifyWavefrontFixes:
    def test_lane_id_fix(self):
        source = 'int lane = threadIdx.x % 32;'
        result = hipify(source)
        assert "threadIdx.x % warpSize" in result.source

    def test_warp_id_fix(self):
        source = 'int wid = threadIdx.x / 32;'
        result = hipify(source)
        assert "threadIdx.x / warpSize" in result.source

    def test_num_warps_fix(self):
        source = 'int nw = blockDim.x / 32;'
        result = hipify(source)
        assert "blockDim.x / warpSize" in result.source

    def test_reduction_loop_fix(self):
        source = 'for (int offset = 16; offset > 0; offset >>= 1) {'
        result = hipify(source)
        assert "warpSize/2" in result.source

    def test_ballot_return_type_64bit(self):
        source = 'uint32_t mask = __ballot(pred);'
        config = HipifyConfig(wavefront_size=64)
        result = hipify(source, config=config)
        assert "uint64_t" in result.source


class TestHipifyUnsupported:
    def test_inline_ptx_detected(self):
        result = hipify(INLINE_PTX_CUDA)
        assert len(result.unsupported_features) > 0
        assert any("PTX" in f for f in result.unsupported_features)
        assert result.errors_count > 0

    def test_ldg_removed(self):
        source = 'float val = __ldg(&input[tid]);'
        result = hipify(source)
        assert "__ldg" not in result.source  # Should be replaced with direct access

    def test_wmma_detected(self):
        source = '#include <mma.h>\nnvcuda::wmma::fragment<> frag;'
        result = hipify(source)
        assert len(result.unsupported_features) > 0


class TestHipifyCoverage:
    def test_coverage_report(self):
        coverage = get_translation_coverage()
        assert coverage["runtime_api"] >= 90
        assert coverage["types"] >= 30
        assert coverage["cublas_to_rocblas"] >= 30
        assert coverage["total"] >= 280

    def test_api_calls_counted(self):
        result = hipify(LAUNCH_SYNTAX_CUDA)
        assert result.api_calls_translated > 0

    def test_config_options(self):
        config = HipifyConfig(
            target_arch="gfx942",
            wavefront_size=64,
            translate_libraries=False,
        )
        result = hipify(CUBLAS_CUDA, config=config)
        # Libraries should NOT be translated with translate_libraries=False
        assert "cublasSgemm" in result.source  # Not translated


class TestHipifyFullProgram:
    def test_warp_reduction_full(self):
        result = hipify(WARP_REDUCTION_CUDA)
        assert "__shfl_down(" in result.source
        assert "0xffffffff" not in result.source
        assert "warpSize" in result.source

    def test_rope_full(self):
        result = hipify(FUSED_ROPE_CUDA)
        # cosf/sinf stay the same in HIP
        assert "cosf" in result.source
        assert "sinf" in result.source


# ═══════════════════════════════════════════════════════════════════
# AMD WAVEFRONT OPTIMIZER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAMDSpecs:
    def test_mi300x_specs(self):
        spec = AMD_SPECS[AMDArch.CDNA3]
        assert spec.compute_units == 304
        assert spec.wavefront_width == 64
        assert spec.lds_per_cu_kb == 64
        assert spec.hbm_bandwidth_tbs == 5.3
        assert spec.has_mfma

    def test_mi250x_specs(self):
        spec = AMD_SPECS[AMDArch.CDNA2]
        assert spec.compute_units == 220
        assert spec.wavefront_width == 64
        assert spec.has_mfma

    def test_rx7900_specs(self):
        spec = AMD_SPECS[AMDArch.RDNA3]
        assert spec.wavefront_width == 32  # RDNA3 wave32 mode
        assert not spec.has_mfma  # No MFMA on RDNA

    def test_all_archs_present(self):
        for arch in AMDArch:
            assert arch in AMD_SPECS


class TestOccupancy:
    def test_high_occupancy(self):
        occ = compute_occupancy(vgpr_per_thread=32, lds_bytes=0, block_size=256)
        assert occ.occupancy_pct > 50
        assert occ.waves_per_cu > 0

    def test_low_occupancy_high_vgprs(self):
        occ = compute_occupancy(vgpr_per_thread=128, lds_bytes=0, block_size=256)
        assert occ.occupancy_pct <= 50
        assert "vgpr" in occ.limiting_factor.lower() or len(occ.recommendations) > 0

    def test_lds_limited(self):
        occ = compute_occupancy(vgpr_per_thread=32, lds_bytes=60000, block_size=256)
        assert occ.occupancy_pct < 100

    def test_unaligned_block_warning(self):
        occ = compute_occupancy(vgpr_per_thread=32, lds_bytes=0, block_size=100)
        assert any("aligned" in r.lower() for r in occ.recommendations)

    def test_cdna3_vs_rdna3(self):
        occ_cdna = compute_occupancy(32, 0, 256, AMDArch.CDNA3)
        occ_rdna = compute_occupancy(32, 0, 256, AMDArch.RDNA3)
        # Both should be valid but different
        assert occ_cdna.waves_per_cu > 0
        assert occ_rdna.waves_per_cu > 0


class TestRoofline:
    def test_memory_bound(self):
        result = roofline_analysis(1, 12)  # 1 FLOP, 12 bytes → low AI
        assert result["bound"] == "memory"
        assert result["arithmetic_intensity"] < result["ridge_point"]

    def test_compute_bound(self):
        result = roofline_analysis(1000, 4)  # 1000 FLOPs, 4 bytes → high AI
        assert result["bound"] == "compute"
        assert result["arithmetic_intensity"] > result["ridge_point"]

    def test_mi300x_bandwidth(self):
        result = roofline_analysis(1, 4, AMDArch.CDNA3)
        assert result["bandwidth_tbs"] == 5.3
        assert result["peak_tflops"] > 1000


class TestBlockSizeOptimizer:
    def test_returns_wavefront_multiple(self):
        bs = optimize_block_size(1000000, AMDArch.CDNA3)
        assert bs % 64 == 0

    def test_rdna3_wave32(self):
        bs = optimize_block_size(1000000, AMDArch.RDNA3)
        assert bs % 32 == 0

    def test_within_limits(self):
        bs = optimize_block_size(1000000)
        assert 64 <= bs <= 1024


class TestLDSBankConflictPadding:
    def test_adds_padding(self):
        source = '__shared__ float smem[256]'
        result = apply_lds_bank_conflict_padding(source)
        assert "256 + 1" in result or "257" in result

    def test_preserves_non_shared(self):
        source = 'float array[256];'
        result = apply_lds_bank_conflict_padding(source)
        assert result == source


class TestWarpToWavefront:
    def test_shfl_down_sync_removal(self):
        source = '__shfl_down_sync(0xffffffff, val, offset)'
        result = convert_warp_to_wavefront(source)
        assert "__shfl_down(" in result
        assert "0xffffffff" not in result

    def test_ballot_sync_removal(self):
        source = '__ballot_sync(0xFFFFFFFF, pred)'
        result = convert_warp_to_wavefront(source)
        assert "__ballot(" in result

    def test_preserves_non_warp_code(self):
        source = 'float x = a + b;'
        result = convert_warp_to_wavefront(source)
        assert result == source


class TestTritonAMDConfig:
    def test_matmul_config(self):
        config = generate_triton_amd_config("matmul")
        assert "BLOCK_M" in config
        assert "BLOCK_N" in config
        assert "BLOCK_K" in config
        assert 16 in config["BLOCK_K"]  # MFMA alignment

    def test_reduce_config(self):
        config = generate_triton_amd_config("reduce")
        assert "BLOCK_SIZE" in config

    def test_attention_config(self):
        config = generate_triton_amd_config("attention")
        assert "BLOCK_DHEAD" in config


# ═══════════════════════════════════════════════════════════════════
# AMD KERNEL OPTIMIZER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestKernelOptimizer:
    def test_lds_padding_applied(self):
        source = """
__global__ void kern() {
    __shared__ float smem[256];
    __shared__ float tile[64][32];
}"""
        optimized, report = optimize_hip_kernel(source)
        assert "256 + 1" in optimized  # 1D array padded
        assert "32 + 1" in optimized   # 2D inner dimension padded

    def test_divergence_elimination(self):
        source = """
__global__ void kern() {
    float val;
    if (threadIdx.x < 128) { val = 1.0f; } else { val = 0.0f; }
}"""
        optimized, report = optimize_hip_kernel(source)
        assert "?" in optimized  # Ternary operator

    def test_ldg_removal(self):
        source = """
__global__ void kern(const float* input) {
    float val = __ldg(&input[threadIdx.x]);
}"""
        optimized, report = optimize_hip_kernel(source)
        assert "__ldg" not in optimized

    def test_compiler_flags_generated(self):
        _, report = optimize_hip_kernel("__global__ void k() {}")
        assert len(report.recommended_compiler_flags) > 0
        assert any("gfx942" in f for f in report.recommended_compiler_flags)

    def test_optimization_levels(self):
        source = "__global__ void k() {}"
        _, r1 = optimize_hip_kernel(source, level=OptimizationLevel.CONSERVATIVE)
        _, r3 = optimize_hip_kernel(source, level=OptimizationLevel.AGGRESSIVE)
        # Aggressive should have more flags
        assert len(r3.recommended_compiler_flags) >= len(r1.recommended_compiler_flags)


class TestTritonAutotuneConfigs:
    def test_matmul_configs(self):
        config = generate_triton_autotune_configs("matmul")
        assert "triton.Config" in config
        assert "BLOCK_M" in config
        assert "num_warps" in config

    def test_attention_configs(self):
        config = generate_triton_autotune_configs("attention")
        assert "BLOCK_DHEAD" in config

    def test_reduce_configs(self):
        config = generate_triton_autotune_configs("reduce")
        assert "BLOCK_SIZE" in config

    def test_softmax_configs(self):
        config = generate_triton_autotune_configs("softmax")
        assert "BLOCK_SIZE" in config

    def test_elementwise_configs(self):
        config = generate_triton_autotune_configs("elementwise")
        assert "BLOCK_SIZE" in config
        assert "4096" in config  # Large blocks for bandwidth


class TestTritonKernelTemplates:
    def test_softmax_template(self):
        source = generate_triton_softmax_amd()
        assert "softmax_kernel" in source
        assert "@triton.jit" in source
        assert "tl.max" in source
        assert "tl.exp" in source
        assert "tl.sum" in source

    def test_layernorm_template(self):
        source = generate_triton_layernorm_amd()
        assert "layernorm_kernel" in source
        assert "mean" in source
        assert "var" in source or "rstd" in source

    def test_rope_template(self):
        source = generate_triton_rope_amd()
        assert "rope_kernel" in source
        assert "cos_val" in source
        assert "sin_val" in source


# ═══════════════════════════════════════════════════════════════════
# AMD BRIDGE ORCHESTRATOR TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAMDBridge:
    def setup_method(self):
        self.bridge = AMDBridge()

    def test_translate_vector_add(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA)
        assert result.success
        assert result.triton_source is not None
        assert result.hip_source is not None
        assert result.bridge_ir is not None
        assert result.amd_config is not None

    def test_translate_rope(self):
        result = self.bridge.translate(FUSED_ROPE_CUDA)
        assert result.success
        assert result.triton_source is not None
        assert result.hip_source is not None
        assert "cos" in result.triton_source or "cos" in result.hip_source

    def test_translate_reduction(self):
        result = self.bridge.translate(REDUCTION_CUDA)
        assert result.success
        assert result.hip_source is not None
        # Should detect shared memory and atomics in HIP output
        assert "__shared__" in result.hip_source or "LDS" in result.hip_source

    def test_translate_hip_path(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA, path=TranslationPath.HIP)
        assert result.success
        assert result.path == TranslationPath.HIP
        assert result.hip_source is not None
        assert "hip_runtime.h" in result.hip_source

    def test_translate_triton_path(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA, path=TranslationPath.TRITON)
        assert result.success
        assert result.path == TranslationPath.TRITON
        assert result.triton_source is not None

    def test_auto_path_selection(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA, path=TranslationPath.AUTO)
        assert result.success
        # MAP pattern should select Triton
        assert result.path == TranslationPath.TRITON

    def test_translate_hip_direct(self):
        hip_source = self.bridge.translate_hip(VECTOR_ADD_CUDA)
        assert "hip" in hip_source.lower() or "__global__" in hip_source

    def test_occupancy_in_result(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA)
        assert result.occupancy is not None
        assert "waves_per_cu" in result.occupancy
        assert "occupancy_pct" in result.occupancy
        assert result.occupancy["target_arch"] == "cdna3"

    def test_roofline_in_result(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA)
        assert result.roofline is not None
        assert "bound" in result.roofline
        assert "arithmetic_intensity" in result.roofline

    def test_amd_config_in_result(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA)
        assert result.amd_config is not None

    def test_compatibility_warnings(self):
        result = self.bridge.translate(INLINE_PTX_CUDA)
        assert len(result.warnings) > 0
        assert any("PTX" in w or "asm" in w.lower() for w in result.warnings)

    def test_deploy_placeholder(self):
        result = self.bridge.deploy_to_amd(VECTOR_ADD_CUDA)
        assert result["status"] == "placeholder"
        assert result["ready_for_deployment"]
        assert result["triton_source"] is not None
        assert result["hip_source"] is not None

    def test_status(self):
        status = self.bridge.status()
        assert status["name"] == "CUDA Scribe AMD Bridge"
        assert "triton" in status["translation_paths"]
        assert "hip" in status["translation_paths"]
        assert len(status["supported_kernels"]) > 5
        assert len(status["algorithmica_optimizations"]) > 0

    def test_target_arch_configurable(self):
        bridge_mi250 = AMDBridge(target_arch=AMDArch.CDNA2)
        result = bridge_mi250.translate(VECTOR_ADD_CUDA)
        assert result.success
        assert result.occupancy["target_arch"] == "cdna2"


class TestAMDBridgeTritonGeneration:
    def setup_method(self):
        self.bridge = AMDBridge()

    def test_map_kernel(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA)
        assert "triton" in result.triton_source.lower()
        assert "tl.load" in result.triton_source
        assert "tl.store" in result.triton_source

    def test_reduce_kernel(self):
        result = self.bridge.translate(REDUCTION_CUDA)
        assert "tl.sum" in result.triton_source or "reduce" in result.triton_source.lower()

    def test_rope_kernel(self):
        result = self.bridge.translate(FUSED_ROPE_CUDA)
        assert result.triton_source is not None
        assert "import triton" in result.triton_source


class TestAMDBridgeHIPGeneration:
    def setup_method(self):
        self.bridge = AMDBridge()

    def test_hip_vector_add(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA, path=TranslationPath.HIP)
        source = result.hip_source
        assert "hip_runtime.h" in source
        assert "__global__" in source
        assert "hipLaunchKernelGGL" in source

    def test_hip_wavefront_alignment(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA, path=TranslationPath.HIP)
        source = result.hip_source
        # Block size should be wavefront-aligned
        assert "wavefront" in source.lower() or "64" in source

    def test_hip_params_preserved(self):
        result = self.bridge.translate(VECTOR_ADD_CUDA, path=TranslationPath.HIP)
        source = result.hip_source
        assert "a" in source
        assert "b" in source
        assert "c" in source


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_pipeline_vector_add(self):
        """Test the complete CUDA → AMD pipeline."""
        bridge = AMDBridge()

        # Step 1: Translate
        result = bridge.translate(VECTOR_ADD_CUDA)
        assert result.success

        # Step 2: Verify HIP has proper structure
        assert "hipLaunchKernelGGL" in result.hip_source
        assert "__global__" in result.hip_source

        # Step 3: Verify Triton has proper structure
        assert "import triton" in result.triton_source
        assert "@triton.jit" in result.triton_source

        # Step 4: Verify analysis
        assert result.occupancy["waves_per_cu"] > 0
        assert result.roofline["bound"] in ("memory", "compute")

    def test_full_pipeline_rope(self):
        """Test complex kernel (RoPE) through pipeline."""
        bridge = AMDBridge()
        result = bridge.translate(FUSED_ROPE_CUDA)
        assert result.success
        assert result.bridge_ir["name"] == "fused_rope_forward"
        assert result.bridge_ir["pattern"] == "map"

    def test_hipify_then_optimize(self):
        """Test HIPIFY → kernel optimizer pipeline."""
        # Step 1: HIPIFY the warp reduction
        hipified = hipify(WARP_REDUCTION_CUDA)
        assert "__shfl_down(" in hipified.source

        # Step 2: Optimize for AMD
        optimized, report = optimize_hip_kernel(hipified.source)
        assert len(report.recommended_compiler_flags) > 0

    def test_bridge_ir_to_all_backends(self):
        """Test that BridgeIR feeds into HIP backend alongside others."""
        results = translate_universal(VECTOR_ADD_CUDA)
        assert "hip" in results
        assert results["hip"]["success"]

    def test_bridge_and_hipify_agree(self):
        """Bridge HIP output and HIPIFY should produce similar results."""
        bridge = AMDBridge()
        result = bridge.translate(VECTOR_ADD_CUDA)

        # Both should contain HIP runtime header
        assert "hip_runtime.h" in result.hip_source

        # HIPIFY on raw CUDA
        hipified = hipify(VECTOR_ADD_CUDA)
        # Both approaches should handle the kernel


class TestEdgeCases:
    def test_empty_source(self):
        result = hipify("")
        assert result.source == ""
        assert result.api_calls_translated == 0

    def test_non_cuda_code(self):
        source = "int main() { return 0; }"
        result = hipify(source)
        assert result.source == source  # No CUDA APIs to translate

    def test_bridge_invalid_cuda(self):
        bridge = AMDBridge()
        result = bridge.translate("not valid cuda")
        # Should still succeed (lifter is lenient) or fail gracefully
        assert isinstance(result, AMDTranslationResult)

    def test_multiple_kernels_hipify(self):
        source = """
__global__ void kern1(float* a) { a[threadIdx.x] = 1.0f; }
__global__ void kern2(float* b) { b[threadIdx.x] = 2.0f; }
"""
        result = hipify(source)
        # No CUDA APIs to translate, but should not error
        assert result.source is not None


# ═══════════════════════════════════════════════════════════════════
# MFMA REFERENCE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestMFMAInstructions:
    """Tests for MFMA instruction reference table."""

    def test_mfma_fp16_instructions_exist(self):
        from cuda_scribe.backends.amd_kernel_optimizer import MFMA_INSTRUCTIONS
        fp16_instrs = {k: v for k, v in MFMA_INSTRUCTIONS.items() if v["in"] == "fp16"}
        assert len(fp16_instrs) >= 3  # 32x32, 16x16, 4x4

    def test_mfma_fp8_instructions_exist(self):
        from cuda_scribe.backends.amd_kernel_optimizer import MFMA_INSTRUCTIONS
        fp8_instrs = {k: v for k, v in MFMA_INSTRUCTIONS.items() if v["in"] == "fp8"}
        assert len(fp8_instrs) >= 2  # 32x32, 16x16

    def test_mfma_register_budget(self):
        from cuda_scribe.backends.amd_kernel_optimizer import MFMA_INSTRUCTIONS
        instr = MFMA_INSTRUCTIONS["v_mfma_f32_32x32x8_f16"]
        total_vgprs = instr["vgpr_a"] + instr["vgpr_b"] + instr["vgpr_c"]
        assert total_vgprs == 24  # 4+4+16
        assert instr["M"] == 32
        assert instr["N"] == 32
        assert instr["K"] == 8

    def test_mfma_16x16_lower_latency(self):
        from cuda_scribe.backends.amd_kernel_optimizer import MFMA_INSTRUCTIONS
        large = MFMA_INSTRUCTIONS["v_mfma_f32_32x32x8_f16"]
        small = MFMA_INSTRUCTIONS["v_mfma_f32_16x16x16_f16"]
        assert small["latency"] <= large["latency"]  # 8 <= 16

    def test_select_mfma_fp16_large(self):
        from cuda_scribe.backends.amd_kernel_optimizer import select_mfma_instruction
        instr = select_mfma_instruction("fp16", prefer_large=True)
        assert instr["M"] == 32 and instr["N"] == 32

    def test_select_mfma_fp16_small(self):
        from cuda_scribe.backends.amd_kernel_optimizer import select_mfma_instruction
        instr = select_mfma_instruction("fp16", prefer_large=False)
        assert instr["M"] == 16 and instr["N"] == 16

    def test_select_mfma_bf16(self):
        from cuda_scribe.backends.amd_kernel_optimizer import select_mfma_instruction
        instr = select_mfma_instruction("bf16")
        assert instr["in"] == "bf16"

    def test_select_mfma_unknown_falls_back(self):
        from cuda_scribe.backends.amd_kernel_optimizer import select_mfma_instruction
        instr = select_mfma_instruction("fp64")  # doesn't exist
        assert instr["in"] == "fp16"  # falls back


class TestVGPROccupancyLookup:
    """Tests for VGPR→occupancy lookup table."""

    def test_low_vgpr_max_occupancy(self):
        from cuda_scribe.backends.amd_kernel_optimizer import lookup_vgpr_occupancy
        assert lookup_vgpr_occupancy(32) == 10  # ≤48 → 10

    def test_medium_vgpr(self):
        from cuda_scribe.backends.amd_kernel_optimizer import lookup_vgpr_occupancy
        assert lookup_vgpr_occupancy(96) == 5

    def test_high_vgpr_gemm(self):
        from cuda_scribe.backends.amd_kernel_optimizer import lookup_vgpr_occupancy
        assert lookup_vgpr_occupancy(256) == 2  # GEMM register blocking

    def test_max_vgpr(self):
        from cuda_scribe.backends.amd_kernel_optimizer import lookup_vgpr_occupancy
        assert lookup_vgpr_occupancy(512) == 1

    def test_overflow_vgpr(self):
        from cuda_scribe.backends.amd_kernel_optimizer import lookup_vgpr_occupancy
        assert lookup_vgpr_occupancy(600) == 0  # cannot launch

    def test_granularity_rounding(self):
        from cuda_scribe.backends.amd_kernel_optimizer import lookup_vgpr_occupancy
        # 49 rounds up to 56 → 9 wavefronts
        assert lookup_vgpr_occupancy(49) == 9


class TestTritonGEMMTemplate:
    """Tests for Triton GEMM kernel template."""

    def test_gemm_template_generates(self):
        from cuda_scribe.backends.amd_kernel_optimizer import generate_triton_gemm_amd
        code = generate_triton_gemm_amd()
        assert "gemm_kernel" in code
        assert "tl.dot" in code
        assert "MFMA" in code or "mfma" in code.lower() or "BLOCK_M" in code

    def test_gemm_has_autotune(self):
        from cuda_scribe.backends.amd_kernel_optimizer import generate_triton_gemm_amd
        code = generate_triton_gemm_amd()
        assert "@triton.autotune" in code
        assert "BLOCK_M" in code
        assert "BLOCK_K" in code

    def test_gemm_has_l2_friendly_ordering(self):
        from cuda_scribe.backends.amd_kernel_optimizer import generate_triton_gemm_amd
        code = generate_triton_gemm_amd()
        assert "GROUP_SIZE_M" in code  # L2-friendly grouped PID remapping

    def test_gemm_wrapper_function(self):
        from cuda_scribe.backends.amd_kernel_optimizer import generate_triton_gemm_amd
        code = generate_triton_gemm_amd()
        assert "def matmul(" in code


class TestTritonFlashAttentionTemplate:
    """Tests for Triton FlashAttention kernel template."""

    def test_flash_attention_generates(self):
        from cuda_scribe.backends.amd_kernel_optimizer import generate_triton_flash_attention_amd
        code = generate_triton_flash_attention_amd()
        assert "flash_attn_fwd_kernel" in code
        assert "sm_scale" in code

    def test_flash_attention_has_online_softmax(self):
        from cuda_scribe.backends.amd_kernel_optimizer import generate_triton_flash_attention_amd
        code = generate_triton_flash_attention_amd()
        assert "m_i" in code  # row max tracker
        assert "l_i" in code  # row sum tracker
        assert "tl.exp" in code

    def test_flash_attention_has_autotune(self):
        from cuda_scribe.backends.amd_kernel_optimizer import generate_triton_flash_attention_amd
        code = generate_triton_flash_attention_amd()
        assert "@triton.autotune" in code
        assert "BLOCK_M" in code
        assert "BLOCK_N" in code

    def test_flash_attention_wrapper(self):
        from cuda_scribe.backends.amd_kernel_optimizer import generate_triton_flash_attention_amd
        code = generate_triton_flash_attention_amd()
        assert "def flash_attention(" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
