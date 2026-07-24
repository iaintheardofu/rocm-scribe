#!/usr/bin/env python3
"""
SYCL Backend — Intel GPU + multi-vendor kernel generation from BridgeIR.

Translates BridgeIR kernels to SYCL C++ code that runs on:
  - Intel Arc / Xe Max / Ponte Vecchio GPUs
  - Intel Gaudi (Habana) accelerators
  - NVIDIA GPUs (via SYCL-CUDA interop)
  - AMD GPUs (via SYCL-HIP interop)
  - CPUs (via OpenCL/Level Zero)

SYCL mapping from CUDA:
  - sycl::id<1>(idx) ≈ threadIdx.x + blockIdx.x * blockDim.x
  - sycl::local_accessor ≈ __shared__
  - item.barrier() ≈ __syncthreads()
  - sycl::atomic_ref ≈ atomicAdd

References:
  - https://github.com/oneapi-src/SYCLomatic (Intel's CUDA→SYCL tool)
  - https://www.khronos.org/sycl/
"""
from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List

from ..bridge_ir import (
    BackendPlugin,
    BridgeKernel,
    DType,
    ParallelPattern,
    register_backend,
)


class SYCLBackend(BackendPlugin):
    """Generate SYCL C++ code from BridgeIR."""

    DTYPE_MAP = {
        DType.FLOAT16: "sycl::half",
        DType.BFLOAT16: "sycl::ext::oneapi::bfloat16",
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

    def name(self) -> str:
        return "sycl"

    def supported_hardware(self) -> List[str]:
        return ["intel", "nvidia", "amd", "cpu"]

    def translate(self, kernel: BridgeKernel, cuda_source: str = "") -> str:
        """Generate SYCL code from BridgeIR."""
        # Collect params
        tensor_params = [p for p in kernel.params if p.is_tensor]
        scalar_params = [p for p in kernel.params if not p.is_tensor]

        # Build function signature
        func_params = []
        for p in tensor_params:
            sycl_type = self.DTYPE_MAP.get(
                p.tile_type.dtype if p.tile_type else DType.FLOAT32, "float"
            )
            func_params.append(f"{sycl_type}* {p.name}")
        for p in scalar_params:
            sycl_type = self.DTYPE_MAP.get(p.scalar_dtype or DType.INT32, "int32_t")
            func_params.append(f"{sycl_type} {p.name}")

        # Determine total elements
        total_expr = "total_elements"
        if kernel.derived_scalars:
            for k, v in kernel.derived_scalars.items():
                if 'total' in k.lower():
                    total_expr = k
                    break

        # Build kernel body
        body = self._translate_body(kernel, cuda_source)

        code = textwrap.dedent(f"""\
        // Bridge v2.0 — SYCL translation of {kernel.name}
        // Pattern: {kernel.pattern.value}
        // Target: Intel Arc/Gaudi, NVIDIA, AMD via SYCL

        #include <sycl/sycl.hpp>
        #include <cstdint>

        void {kernel.name}(sycl::queue& q, {', '.join(func_params)}) {{
            q.parallel_for(sycl::range<1>({total_expr}), [=](sycl::id<1> idx) {{
                const auto tid = idx[0];
        {textwrap.indent(body, '        ')}
            }}).wait();
        }}
        """)
        return code

    def _translate_body(self, kernel: BridgeKernel, cuda_source: str) -> str:
        """Translate CUDA kernel body to SYCL."""
        if not cuda_source:
            return "// TODO: LLM-assisted body translation"

        body_match = re.search(r'\{(.*)\}', cuda_source, re.DOTALL)
        if not body_match:
            return "// Could not extract kernel body"

        body = body_match.group(1).strip()

        # Remove tid computation (we compute it from sycl::id above)
        body = re.sub(
            r'const\s+int\s+tid\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x\s*;',
            '', body
        )

        # CUDA → SYCL replacements
        body = body.replace('__syncthreads()', 'sycl::group_barrier(item.get_group())')

        # CUDA math functions (cosf → sycl::cos, etc.)
        math_replacements = {
            'cosf': 'sycl::cos', 'sinf': 'sycl::sin', 'expf': 'sycl::exp',
            'logf': 'sycl::log', 'sqrtf': 'sycl::sqrt', 'fabsf': 'sycl::fabs',
            'powf': 'sycl::pow', 'tanhf': 'sycl::tanh',
            'fmaxf': 'sycl::fmax', 'fminf': 'sycl::fmin',
        }
        for cuda_fn, sycl_fn in math_replacements.items():
            body = body.replace(cuda_fn, sycl_fn)

        # atomicAdd → sycl atomic
        body = re.sub(
            r'atomicAdd\s*\(\s*&?\s*(\w+)\[([^\]]+)\]\s*,\s*([^)]+)\)',
            r'sycl::atomic_ref<float, sycl::memory_order::relaxed, '
            r'sycl::memory_scope::device, sycl::access::address_space::global_space>'
            r'(\1[\2]).fetch_add(\3)',
            body
        )

        return body

    def verify(self, source: str, reference=None) -> Dict[str, Any]:
        return {"status": "not_implemented", "note": "Requires Intel oneAPI DPC++ compiler"}

    def benchmark(self, source: str) -> Dict[str, Any]:
        return {"status": "not_implemented"}

    def is_hardware_available(self) -> bool:
        return False  # Would need to check for Intel GPU + DPC++ compiler


register_backend(SYCLBackend())
