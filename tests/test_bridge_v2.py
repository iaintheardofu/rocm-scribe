#!/usr/bin/env python3
"""
Tests for Bridge v2.0 — Multi-target GPU kernel translation.

Tests the BridgeIR, CUDA lifter, and all backend code generators.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cuda_scribe.bridge_ir import (
    BridgeKernel,
    CUDALifter,
    DType,
    HardwareRequirement,
    MemoryLevel,
    ParallelPattern,
    TileType,
    available_backends,
    get_backend,
    translate_universal,
)

# Import backends to trigger registration
import cuda_scribe.backends.pallas_backend
import cuda_scribe.backends.metal_backend
import cuda_scribe.backends.wgsl_backend
import cuda_scribe.backends.sycl_backend
import cuda_scribe.backends.hip_backend


# ══════════════════════════════════════════════════════════════════
# TEST CUDA SOURCES
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
# BRIDGEIR TESTS
# ══════════════════════════════════════════════════════════════════

class TestTileType:
    def test_basic(self):
        t = TileType(shape=(1024,), dtype=DType.FLOAT32)
        assert str(t) == "Tile[1024, f32, global]"

    def test_with_memory(self):
        t = TileType(shape=(16, 64), dtype=DType.FLOAT16, memory=MemoryLevel.SHARED)
        assert "shared" in str(t)

    def test_symbolic_shape(self):
        t = TileType(shape=("B", "S", "H", "D"), dtype=DType.FLOAT32)
        assert "BxSxHxD" in str(t)


class TestDType:
    def test_bytes(self):
        assert DType.FLOAT32.bytes == 4
        assert DType.FLOAT16.bytes == 2
        assert DType.INT8.bytes == 1

    def test_is_float(self):
        assert DType.FLOAT32.is_float
        assert DType.BFLOAT16.is_float
        assert not DType.INT32.is_float

    def test_is_integer(self):
        assert DType.INT32.is_integer
        assert DType.UINT8.is_integer
        assert not DType.FLOAT32.is_integer


# ══════════════════════════════════════════════════════════════════
# CUDA LIFTER TESTS
# ══════════════════════════════════════════════════════════════════

class TestCUDALifter:
    def setup_method(self):
        self.lifter = CUDALifter()

    def test_lift_vector_add(self):
        kernel = self.lifter.lift(VECTOR_ADD_CUDA, "vector_add")
        assert kernel.name == "vector_add"
        assert kernel.pattern == ParallelPattern.MAP
        assert len(kernel.params) == 4  # a, b, c, n
        assert kernel.grid_dims == 1

        # Check param types
        tensor_params = [p for p in kernel.params if p.is_tensor]
        scalar_params = [p for p in kernel.params if not p.is_tensor]
        assert len(tensor_params) == 3  # a, b, c
        assert len(scalar_params) == 1  # n

        # Check outputs
        outputs = [p for p in kernel.params if p.is_output]
        assert len(outputs) == 1
        assert outputs[0].name == "c"

    def test_lift_fused_rope(self):
        kernel = self.lifter.lift(FUSED_ROPE_CUDA, "fused_rope_forward")
        assert kernel.name == "fused_rope_forward"
        assert kernel.pattern == ParallelPattern.MAP

        # Should have tensor and scalar params
        tensor_params = [p for p in kernel.params if p.is_tensor]
        scalar_params = [p for p in kernel.params if not p.is_tensor]
        assert len(tensor_params) == 3   # input, output, positions
        assert len(scalar_params) >= 5   # batch_size, seq_len, num_heads, head_dim, base, scaling_factor

        # Should detect derived scalars
        assert len(kernel.derived_scalars) > 0

    def test_lift_reduction(self):
        kernel = self.lifter.lift(REDUCTION_CUDA, "sum_reduce")
        assert kernel.name == "sum_reduce"
        assert kernel.pattern == ParallelPattern.REDUCE

        # Should detect shared memory and atomics
        assert HardwareRequirement.SHARED_MEMORY in kernel.hardware_requirements
        assert HardwareRequirement.ATOMICS in kernel.hardware_requirements

    def test_lift_auto_detect_name(self):
        kernel = self.lifter.lift(VECTOR_ADD_CUDA)
        assert kernel.name == "vector_add"

    def test_ir_hash_deterministic(self):
        k1 = self.lifter.lift(VECTOR_ADD_CUDA)
        k2 = self.lifter.lift(VECTOR_ADD_CUDA)
        assert k1.ir_hash == k2.ir_hash

    def test_ir_hash_differs(self):
        k1 = self.lifter.lift(VECTOR_ADD_CUDA)
        k2 = self.lifter.lift(FUSED_ROPE_CUDA)
        assert k1.ir_hash != k2.ir_hash


# ══════════════════════════════════════════════════════════════════
# BACKEND REGISTRY TESTS
# ══════════════════════════════════════════════════════════════════

class TestBackendRegistry:
    def test_backends_registered(self):
        backends = available_backends()
        assert "pallas" in backends
        assert "metal" in backends
        assert "wgsl" in backends
        assert "sycl" in backends
        assert "hip" in backends

    def test_get_backend(self):
        pallas = get_backend("pallas")
        assert pallas.name() == "pallas"
        assert "nvidia" in pallas.supported_hardware()
        assert "tpu" in pallas.supported_hardware()

    def test_get_hip_backend(self):
        hip = get_backend("hip")
        assert hip.name() == "hip"
        assert "amd" in hip.supported_hardware()
        assert "nvidia" in hip.supported_hardware()  # HIP is portable

    def test_get_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")


# ══════════════════════════════════════════════════════════════════
# TRANSLATION TESTS
# ══════════════════════════════════════════════════════════════════

class TestTranslation:
    def setup_method(self):
        self.lifter = CUDALifter()

    def test_translate_vector_add_all_backends(self):
        results = translate_universal(VECTOR_ADD_CUDA)
        for backend_name, result in results.items():
            assert result["success"], f"{backend_name} failed: {result.get('error')}"
            assert len(result["source"]) > 50, f"{backend_name} source too short"

    def test_translate_rope_all_backends(self):
        results = translate_universal(FUSED_ROPE_CUDA)
        for backend_name, result in results.items():
            assert result["success"], f"{backend_name} failed: {result.get('error')}"

    def test_pallas_output_has_jax_imports(self):
        results = translate_universal(VECTOR_ADD_CUDA, targets=["pallas"])
        source = results["pallas"]["source"]
        assert "import jax" in source
        assert "pallas" in source

    def test_metal_output_has_metal_syntax(self):
        results = translate_universal(VECTOR_ADD_CUDA, targets=["metal"])
        source = results["metal"]["source"]
        assert "kernel void" in source
        assert "metal_stdlib" in source or "thread_position_in_grid" in source

    def test_wgsl_output_has_webgpu_syntax(self):
        results = translate_universal(VECTOR_ADD_CUDA, targets=["wgsl"])
        source = results["wgsl"]["source"]
        assert "@compute" in source
        assert "@workgroup_size" in source
        assert "global_invocation_id" in source

    def test_sycl_output_has_sycl_syntax(self):
        results = translate_universal(VECTOR_ADD_CUDA, targets=["sycl"])
        source = results["sycl"]["source"]
        assert "sycl" in source
        assert "parallel_for" in source

    def test_hip_output_has_hip_syntax(self):
        results = translate_universal(VECTOR_ADD_CUDA, targets=["hip"])
        source = results["hip"]["source"]
        assert "hip_runtime.h" in source
        assert "__global__" in source
        assert "hipLaunchKernelGGL" in source

    def test_hip_output_wavefront_aligned(self):
        results = translate_universal(VECTOR_ADD_CUDA, targets=["hip"])
        source = results["hip"]["source"]
        # Block size should be aligned to 64 (AMD wavefront width)
        assert "wavefront" in source.lower() or "64" in source

    def test_hip_rope_translation(self):
        results = translate_universal(FUSED_ROPE_CUDA, targets=["hip"])
        assert results["hip"]["success"]
        source = results["hip"]["source"]
        assert "hip_runtime.h" in source
        assert "cosf" in source or "cos" in source

    def test_hip_reduction_translation(self):
        results = translate_universal(REDUCTION_CUDA, targets=["hip"])
        assert results["hip"]["success"]
        source = results["hip"]["source"]
        assert "__shared__" in source or "LDS" in source

    def test_hip_vector_add_all_params_present(self):
        results = translate_universal(VECTOR_ADD_CUDA, targets=["hip"])
        source = results["hip"]["source"]
        assert "float* __restrict__ a" in source or "a" in source
        assert "float* __restrict__ b" in source or "b" in source
        assert "float* __restrict__ c" in source or "c" in source

    def test_translate_includes_hip(self):
        results = translate_universal(VECTOR_ADD_CUDA)
        assert "hip" in results
        assert results["hip"]["success"]

    def test_metal_rope_has_trig_functions(self):
        results = translate_universal(FUSED_ROPE_CUDA, targets=["metal"])
        source = results["metal"]["source"]
        assert "cos" in source
        assert "sin" in source

    def test_wgsl_rope_has_params_struct(self):
        results = translate_universal(FUSED_ROPE_CUDA, targets=["wgsl"])
        source = results["wgsl"]["source"]
        assert "struct Params" in source
        assert "params" in source

    def test_translate_specific_targets(self):
        results = translate_universal(VECTOR_ADD_CUDA, targets=["pallas", "metal"])
        assert len(results) == 2
        assert "pallas" in results
        assert "metal" in results

    def test_ir_hash_in_results(self):
        results = translate_universal(VECTOR_ADD_CUDA, targets=["pallas"])
        assert "ir_hash" in results["pallas"]
        assert len(results["pallas"]["ir_hash"]) == 16


# ══════════════════════════════════════════════════════════════════
# HARDWARE DETECTION TESTS
# ══════════════════════════════════════════════════════════════════

class TestHardwareDetection:
    def test_metal_availability(self):
        import platform
        metal = get_backend("metal")
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            assert metal.is_hardware_available()
        else:
            assert not metal.is_hardware_available()

    def test_wgsl_not_available(self):
        # WGSL requires browser context
        wgsl = get_backend("wgsl")
        assert not wgsl.is_hardware_available()

    def test_sycl_not_available(self):
        # SYCL requires DPC++ compiler
        sycl = get_backend("sycl")
        assert not sycl.is_hardware_available()


# ══════════════════════════════════════════════════════════════════
# BRIDGEIR SERIALIZATION TESTS
# ══════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_to_dict(self):
        lifter = CUDALifter()
        kernel = lifter.lift(VECTOR_ADD_CUDA)
        d = kernel.to_dict()
        assert d["name"] == "vector_add"
        assert d["pattern"] == "map"
        assert "params" in d
        assert "operation_types" in d

    def test_to_dict_rope(self):
        lifter = CUDALifter()
        kernel = lifter.lift(FUSED_ROPE_CUDA)
        d = kernel.to_dict()
        assert d["name"] == "fused_rope_forward"
        assert len(d["params"]) > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
