#!/usr/bin/env python3
"""
WGSL Backend — WebGPU compute shader generation from BridgeIR.

Translates BridgeIR kernels to WGSL (WebGPU Shading Language) compute shaders
that run in any modern browser (Chrome, Firefox, Edge, Safari).

Key use case: Browser-based ML inference with zero server costs.

WGSL mapping from CUDA:
  - workgroup_id ≈ blockIdx
  - local_invocation_id ≈ threadIdx
  - global_invocation_id ≈ blockIdx * blockDim + threadIdx
  - var<workgroup> ≈ __shared__
  - workgroupBarrier() ≈ __syncthreads()
  - @workgroup_size(N) ≈ blockDim

References:
  - https://www.w3.org/TR/WGSL/
  - https://tianpan.co/blog/2026-04-17-browser-native-llm-inference-webgpu
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


class WGSLBackend(BackendPlugin):
    """Generate WGSL compute shaders from BridgeIR."""

    DTYPE_MAP = {
        DType.FLOAT16: "f16",
        DType.BFLOAT16: "f32",  # WGSL has no bfloat16, upcast to f32
        DType.FLOAT32: "f32",
        DType.FLOAT64: "f32",   # WGSL has no f64 in most browsers
        DType.INT8: "i32",      # WGSL minimum is 32-bit
        DType.INT16: "i32",
        DType.INT32: "i32",
        DType.INT64: "i32",     # WGSL has no i64 in most implementations
        DType.UINT8: "u32",
        DType.UINT16: "u32",
        DType.UINT32: "u32",
        DType.UINT64: "u32",
        DType.BOOL: "bool",
    }

    MATH_MAP = {
        "cos": "cos", "sin": "sin", "exp": "exp", "log": "log",
        "sqrt": "sqrt", "abs": "abs", "tanh": "tanh",
        "max": "max", "min": "min", "pow": "pow",
    }

    def name(self) -> str:
        return "wgsl"

    def supported_hardware(self) -> List[str]:
        return ["nvidia", "amd", "intel", "apple_silicon", "browser"]

    def translate(self, kernel: BridgeKernel, cuda_source: str = "") -> str:
        """Generate WGSL compute shader from BridgeIR."""

        # Collect params
        tensor_inputs = [p for p in kernel.params if p.is_tensor and not p.is_output]
        tensor_outputs = [p for p in kernel.params if p.is_tensor and p.is_output]
        scalars = [p for p in kernel.params if not p.is_tensor]

        # Build bindings
        bindings = []
        group = 0
        binding_idx = 0

        for p in tensor_inputs:
            wgsl_type = self.DTYPE_MAP.get(
                p.tile_type.dtype if p.tile_type else DType.FLOAT32, "f32"
            )
            bindings.append(
                f"@group({group}) @binding({binding_idx}) "
                f"var<storage, read> {p.name}: array<{wgsl_type}>;"
            )
            binding_idx += 1

        for p in tensor_outputs:
            wgsl_type = self.DTYPE_MAP.get(
                p.tile_type.dtype if p.tile_type else DType.FLOAT32, "f32"
            )
            bindings.append(
                f"@group({group}) @binding({binding_idx}) "
                f"var<storage, read_write> {p.name}: array<{wgsl_type}>;"
            )
            binding_idx += 1

        # Params struct for scalars
        struct_code = ""
        if scalars:
            fields = []
            for p in scalars:
                wgsl_type = self.DTYPE_MAP.get(p.scalar_dtype or DType.UINT32, "u32")
                fields.append(f"    {p.name}: {wgsl_type},")
            struct_code = f"struct Params {{\n" + "\n".join(fields) + "\n};\n"
            bindings.append(
                f"@group({group}) @binding({binding_idx}) "
                f"var<uniform> params: Params;"
            )

        # Workgroup size
        wg_size = kernel.block_size if isinstance(kernel.block_size, int) else 256

        # Build kernel body
        body = self._translate_body(kernel, cuda_source, bool(scalars))

        code = textwrap.dedent(f"""\
        // Bridge v2.0 — WGSL translation of {kernel.name}
        // Pattern: {kernel.pattern.value}
        // Target: WebGPU (Chrome/Firefox/Edge/Safari)

        {struct_code}
        {chr(10).join(bindings)}

        @compute @workgroup_size({wg_size})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let tid = global_id.x;
        {textwrap.indent(body, '    ')}
        }}
        """)
        return code

    def _translate_body(self, kernel: BridgeKernel, cuda_source: str,
                        has_params: bool) -> str:
        """Translate kernel body to WGSL."""
        if not cuda_source:
            return "// TODO: LLM-assisted body translation"

        # Extract kernel body
        body_match = re.search(r'\{(.*)\}', cuda_source, re.DOTALL)
        if not body_match:
            return "// Could not extract kernel body"

        body = body_match.group(1).strip()

        # CUDA → WGSL replacements
        replacements = {
            'blockIdx.x * blockDim.x + threadIdx.x': 'tid',
            'threadIdx.x': 'global_id.x % WORKGROUP_SIZE',
            'blockIdx.x': 'global_id.x / WORKGROUP_SIZE',
            '__syncthreads()': 'workgroupBarrier()',
        }
        for old, new in replacements.items():
            body = body.replace(old, new)

        # Type replacements
        body = re.sub(r'\bconst\s+float\b', 'let', body)
        body = re.sub(r'\bconst\s+int\b', 'let', body)
        body = re.sub(r'\bconst\s+unsigned\s+int\b', 'let', body)
        body = re.sub(r'\bfloat\b(?!\s*\*)', 'f32', body)
        body = re.sub(r'\bint\b(?!\s*\*)', 'i32', body)

        # CUDA math → WGSL math
        for cuda_fn, wgsl_fn in self.MATH_MAP.items():
            body = re.sub(rf'\b{cuda_fn}f?\b', wgsl_fn, body)

        # Array access: arr[idx] stays the same in WGSL
        # Param prefix
        if has_params:
            for p in kernel.params:
                if not p.is_tensor:
                    body = re.sub(rf'\b{p.name}\b', f'params.{p.name}', body)

        # WGSL uses 'return' not 'return;' for void, and 'if (x) return' → 'if (x) { return; }'
        body = re.sub(r'if\s*\(([^)]+)\)\s*return\s*;',
                       r'if (\1) { return; }', body)

        return body

    def verify(self, source: str, reference=None) -> Dict[str, Any]:
        return {"status": "not_implemented", "note": "Requires WebGPU runtime (browser/Dawn)"}

    def benchmark(self, source: str) -> Dict[str, Any]:
        return {"status": "not_implemented", "note": "Requires WebGPU runtime"}

    def is_hardware_available(self) -> bool:
        return False  # WebGPU requires browser context


register_backend(WGSLBackend())
