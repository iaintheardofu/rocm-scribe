#!/usr/bin/env python3
"""
Metal Backend — Apple Silicon GPU kernel generation from BridgeIR.

Translates BridgeIR kernels to Metal Shading Language (MSL) compute shaders
that run on Apple M1/M2/M3/M4 and future Apple Silicon.

Key differences from CUDA:
  - threadgroup_position_in_grid ≈ blockIdx
  - thread_position_in_threadgroup ≈ threadIdx
  - threadgroup memory ≈ __shared__
  - simdgroup ≈ warp (32 threads on Apple GPU)
  - threadgroup_barrier(mem_flags::mem_threadgroup) ≈ __syncthreads()

References:
  - https://developer.apple.com/metal/
  - https://developer.apple.com/documentation/metal/compute_passes
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List

from ..bridge_ir import (
    BackendPlugin,
    BridgeKernel,
    DType,
    HardwareRequirement,
    LoadOp,
    ParallelPattern,
    StoreOp,
    register_backend,
)


class MetalBackend(BackendPlugin):
    """Generate Metal Shading Language compute shaders from BridgeIR."""

    # BridgeIR DType → Metal type
    DTYPE_MAP = {
        DType.FLOAT16: "half",
        DType.BFLOAT16: "bfloat",
        DType.FLOAT32: "float",
        DType.FLOAT64: "float",  # Metal has limited double support
        DType.INT8: "char",
        DType.INT16: "short",
        DType.INT32: "int",
        DType.INT64: "long",
        DType.UINT8: "uchar",
        DType.UINT16: "ushort",
        DType.UINT32: "uint",
        DType.UINT64: "ulong",
        DType.BOOL: "bool",
    }

    # Math function mapping
    MATH_MAP = {
        "cos": "cos", "sin": "sin", "exp": "exp", "log": "log",
        "sqrt": "sqrt", "abs": "abs", "tanh": "tanh",
        "max": "max", "min": "min", "pow": "pow",
        "exp_fast": "fast::exp", "log_fast": "fast::log",
    }

    def name(self) -> str:
        return "metal"

    def supported_hardware(self) -> List[str]:
        return ["apple_silicon"]

    def translate(self, kernel: BridgeKernel, cuda_source: str = "") -> str:
        """Generate Metal compute shader from BridgeIR."""
        # Build parameter list
        buffer_params = []
        buffer_idx = 0
        for p in kernel.params:
            if p.is_tensor:
                metal_type = self.DTYPE_MAP.get(
                    p.tile_type.dtype if p.tile_type else DType.FLOAT32, "float"
                )
                if p.is_output:
                    buffer_params.append(
                        f"    device {metal_type}* {p.name} [[buffer({buffer_idx})]]"
                    )
                else:
                    buffer_params.append(
                        f"    device const {metal_type}* {p.name} [[buffer({buffer_idx})]]"
                    )
                buffer_idx += 1

        # Add params struct for scalars
        scalar_params = [p for p in kernel.params if not p.is_tensor]
        has_params_struct = len(scalar_params) > 0

        # Build the params struct
        struct_code = ""
        if has_params_struct:
            struct_fields = []
            for p in scalar_params:
                metal_type = self.DTYPE_MAP.get(p.scalar_dtype or DType.INT32, "int")
                struct_fields.append(f"    {metal_type} {p.name};")
            struct_code = textwrap.dedent(f"""\
            struct {kernel.name.title()}Params {{
            {chr(10).join(struct_fields)}
            }};
            """)
            buffer_params.append(
                f"    constant {kernel.name.title()}Params& params [[buffer({buffer_idx})]]"
            )

        # Thread position attributes
        thread_attrs = [
            "    uint tid [[thread_position_in_grid]]",
            "    uint tgid [[threadgroup_position_in_grid]]",
            "    uint ltid [[thread_position_in_threadgroup]]",
        ]

        # Build kernel body from CUDA source pattern
        body = self._translate_body(kernel, cuda_source, has_params_struct)

        param_str = ",\n".join(buffer_params + thread_attrs)

        code = textwrap.dedent(f"""\
        // Bridge v2.0 — Metal translation of {kernel.name}
        // Pattern: {kernel.pattern.value}
        // Hardware: Apple Silicon (M1/M2/M3/M4)

        #include <metal_stdlib>
        using namespace metal;

        {struct_code}
        kernel void {kernel.name}(
        {param_str}
        ) {{
        {textwrap.indent(body, '    ')}
        }}
        """)
        return code

    def _translate_body(self, kernel: BridgeKernel, cuda_source: str,
                        has_params: bool) -> str:
        """Translate kernel body to Metal.

        For recognized patterns, generates directly.
        For complex kernels, provides a template for LLM completion.
        """
        if not cuda_source:
            return "// TODO: LLM-assisted body translation"

        # Simple source-level translation for common patterns
        body = cuda_source

        # Replace CUDA thread indexing
        replacements = {
            'blockIdx.x * blockDim.x + threadIdx.x': 'tid',
            'threadIdx.x': 'ltid',
            'blockIdx.x': 'tgid',
            'blockDim.x': 'threads_per_threadgroup',
            '__syncthreads()': 'threadgroup_barrier(mem_flags::mem_threadgroup)',
            '__shared__': 'threadgroup',
            'atomicAdd': 'atomic_fetch_add_explicit',
        }

        # Extract just the kernel body (between first { and last })
        import re
        body_match = re.search(r'\{(.*)\}', cuda_source, re.DOTALL)
        if body_match:
            body = body_match.group(1).strip()

        for old, new in replacements.items():
            body = body.replace(old, new)

        # Remove redundant tid computation (Metal provides tid via attribute)
        body = re.sub(
            r'\s*const\s+int\s+tid\s*=\s*tid\s*;\s*',
            '\n', body
        )

        # Replace CUDA math functions
        for cuda_fn, metal_fn in self.MATH_MAP.items():
            body = re.sub(rf'\b{cuda_fn}f?\b', metal_fn, body)

        # Prefix scalar params with params. if struct exists
        if has_params:
            for p in kernel.params:
                if not p.is_tensor:
                    body = re.sub(rf'\b{p.name}\b', f'params.{p.name}', body)

        return body

    def verify(self, source: str, reference=None) -> Dict[str, Any]:
        """Compile Metal shader to verify syntax correctness on macOS."""
        if not self.is_hardware_available():
            return {"status": "skipped", "note": "Not on Apple Silicon"}
        try:
            import Metal as _Metal
            device = _Metal.MTLCreateSystemDefaultDevice()
            options = _Metal.MTLCompileOptions.new()
            library, error = device.newLibraryWithSource_options_error_(
                source, options, None
            )
            if error:
                return {"status": "fail", "error": str(error)}
            return {
                "status": "pass",
                "gpu": str(device.name()),
                "functions": [
                    str(library.functionNames().objectAtIndex_(i))
                    for i in range(library.functionNames().count())
                ],
            }
        except ImportError:
            return {"status": "skipped", "note": "pyobjc-framework-Metal not installed"}

    def benchmark(self, source: str) -> Dict[str, Any]:
        return {"status": "not_implemented", "note": "Requires Metal runtime setup"}

    def is_hardware_available(self) -> bool:
        import platform
        return platform.system() == "Darwin" and platform.machine() == "arm64"


# Register the backend
register_backend(MetalBackend())
