#!/usr/bin/env python3
"""
Pallas Backend — JAX/XLA GPU + TPU kernel generation from BridgeIR.

Translates BridgeIR kernels to JAX Pallas code that runs on:
  - NVIDIA/AMD GPUs (via Triton underneath)
  - Google TPUs (via Mosaic)

Pallas is JAX's kernel authoring DSL. It operates at the block level,
making the mapping from BridgeIR nearly 1:1.

Key references:
  - https://docs.jax.dev/en/latest/pallas/index.html
  - https://docs.jax.dev/en/latest/pallas/gpu/reference.html
  - https://docs.jax.dev/en/latest/pallas/design/design.html
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional

from ..bridge_ir import (
    BackendPlugin,
    BridgeKernel,
    ComputeOp,
    DotOp,
    DType,
    HardwareRequirement,
    IndexDecomposition,
    LoadOp,
    MemoryLevel,
    ParallelPattern,
    ReduceOp,
    StoreOp,
    register_backend,
)


class PallasBackend(BackendPlugin):
    """Generate JAX Pallas kernel code from BridgeIR."""

    # BridgeIR DType → JAX dtype string
    DTYPE_MAP = {
        DType.FLOAT16: "jnp.float16",
        DType.BFLOAT16: "jnp.bfloat16",
        DType.FLOAT32: "jnp.float32",
        DType.FLOAT64: "jnp.float64",
        DType.INT8: "jnp.int8",
        DType.INT16: "jnp.int16",
        DType.INT32: "jnp.int32",
        DType.INT64: "jnp.int64",
        DType.UINT8: "jnp.uint8",
        DType.UINT16: "jnp.uint16",
        DType.UINT32: "jnp.uint32",
        DType.UINT64: "jnp.uint64",
        DType.BOOL: "jnp.bool_",
    }

    # BridgeIR math ops → JAX/Pallas equivalents
    MATH_OP_MAP = {
        "cos": "jnp.cos",
        "sin": "jnp.sin",
        "exp": "jnp.exp",
        "log": "jnp.log",
        "sqrt": "jnp.sqrt",
        "abs": "jnp.abs",
        "tanh": "jnp.tanh",
        "max": "jnp.maximum",
        "min": "jnp.minimum",
        "pow": "jnp.power",
        "exp_fast": "jnp.exp",  # JAX has no fast math variants
        "log_fast": "jnp.log",
    }

    def name(self) -> str:
        return "pallas"

    def supported_hardware(self) -> List[str]:
        return ["nvidia", "amd", "tpu"]

    def translate(self, kernel: BridgeKernel, cuda_source: str = "") -> str:
        """Generate Pallas kernel from BridgeIR + original CUDA source.

        For complex kernels, uses LLM translation with BridgeIR context.
        For recognized patterns, uses direct template generation.
        """
        if kernel.pattern == ParallelPattern.MAP:
            return self._generate_elementwise(kernel, cuda_source)
        else:
            return self._generate_generic(kernel, cuda_source)

    def _generate_elementwise(self, kernel: BridgeKernel, cuda_source: str) -> str:
        """Generate Pallas code for element-wise (map) kernels."""
        # Collect input/output params
        inputs = [p for p in kernel.params if p.is_tensor and not p.is_output]
        outputs = [p for p in kernel.params if p.is_tensor and p.is_output]
        scalars = [p for p in kernel.params if not p.is_tensor]

        kernel_body = self._generate_kernel_body(kernel)

        code = textwrap.dedent(f"""\
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl

        def {kernel.name}_kernel({', '.join(p.name + '_ref' for p in inputs + outputs)}):
            \"\"\"Pallas kernel body for {kernel.name}.\"\"\"
            pid = pl.program_id(axis=0)
        {textwrap.indent(kernel_body, '    ')}

        def {kernel.name}({', '.join(p.name for p in inputs + scalars)}) -> {self._return_type(outputs)}:
            \"\"\"Apply {kernel.name} using Pallas.\"\"\"
            BLOCK_SIZE = {kernel.block_size}
            n_elements = {inputs[0].name}.size if hasattr({inputs[0].name}, 'size') else {inputs[0].name}.shape[0]
            grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

            return pl.pallas_call(
                {kernel.name}_kernel,
                out_shape=[jax.ShapeDtypeStruct({outputs[0].name}.shape, {outputs[0].name}.dtype)
                           for _ in range({len(outputs)})],
                grid=(grid,),
            )({', '.join(p.name for p in inputs)})
        """)
        return code

    def _generate_generic(self, kernel: BridgeKernel, cuda_source: str) -> str:
        """Generate Pallas code for complex kernels using LLM-assisted translation."""
        # Build rich context from BridgeIR for LLM
        ir_context = self._build_ir_context(kernel)

        code = textwrap.dedent(f"""\
        # Bridge v2.0 — Pallas translation of {kernel.name}
        # Pattern: {kernel.pattern.value}
        # Hardware requirements: {', '.join(r.name for r in kernel.hardware_requirements)}
        # IR hash: {kernel.ir_hash}
        #
        # BridgeIR Context:
        # {ir_context}

        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from functools import partial

        # TODO: LLM-assisted translation pending
        # The BridgeIR has been extracted — pass to LLM with Pallas examples
        # for translation of {kernel.pattern.value} pattern kernels.
        #
        # Key mappings for this kernel:
        #   CUDA threadIdx → Pallas pl.program_id()
        #   CUDA __shared__ → Pallas SMEM scratch space
        #   CUDA __syncthreads() → implicit in Pallas
        #   CUDA atomicAdd → pl.atomic_add()
        """)
        return code

    def _generate_kernel_body(self, kernel: BridgeKernel) -> str:
        """Generate the body of a Pallas kernel function."""
        lines = []

        for op in kernel.operations:
            if isinstance(op, IndexDecomposition):
                for var_name, dim_size in op.dimensions:
                    lines.append(f"{var_name} = pid % {dim_size}")
            elif isinstance(op, LoadOp):
                idx_str = ", ".join(ie.expr for ie in op.indices)
                lines.append(f"{op.result_var} = {op.source_param}_ref[{idx_str}]")
            elif isinstance(op, StoreOp):
                idx_str = ", ".join(ie.expr for ie in op.indices)
                lines.append(f"{op.target_param}_ref[{idx_str}] = {op.value_expr}")
            elif isinstance(op, ComputeOp):
                if op.op in self.MATH_OP_MAP:
                    fn = self.MATH_OP_MAP[op.op]
                    args = ", ".join(op.operands)
                    lines.append(f"{op.result_var} = {fn}({args})")
                elif op.op in ("mul", "add", "sub", "div"):
                    ops_map = {"mul": "*", "add": "+", "sub": "-", "div": "/"}
                    a, b = op.operands[0], op.operands[1]
                    lines.append(f"{op.result_var} = {a} {ops_map[op.op]} {b}")

        return "\n".join(lines)

    def _build_ir_context(self, kernel: BridgeKernel) -> str:
        """Build a human-readable IR context string for LLM translation."""
        parts = [
            f"Kernel: {kernel.name}",
            f"Pattern: {kernel.pattern.value}",
            f"Grid dims: {kernel.grid_dims}",
            f"Params: {len(kernel.params)} ({sum(1 for p in kernel.params if p.is_tensor)} tensors)",
        ]
        for p in kernel.params:
            if p.is_tensor:
                parts.append(f"  {p.name}: {p.tile_type} {'(output)' if p.is_output else '(input)'}")
            else:
                parts.append(f"  {p.name}: scalar {p.scalar_dtype.value if p.scalar_dtype else '?'}")

        if kernel.derived_scalars:
            parts.append(f"Derived scalars: {kernel.derived_scalars}")

        parts.append(f"Operations: {len(kernel.operations)}")
        for op in kernel.operations:
            parts.append(f"  {type(op).__name__}")

        return "\n# ".join(parts)

    def _return_type(self, outputs: list) -> str:
        if len(outputs) == 1:
            return "jnp.ndarray"
        return f"tuple[{''.join('jnp.ndarray, ' for _ in outputs)}]"

    def verify(self, source: str, reference: Any = None) -> Dict[str, Any]:
        """Verify Pallas kernel against reference."""
        return {"status": "not_implemented", "note": "Requires JAX runtime"}

    def benchmark(self, source: str) -> Dict[str, Any]:
        """Benchmark Pallas kernel."""
        return {"status": "not_implemented", "note": "Requires JAX runtime"}

    def is_hardware_available(self) -> bool:
        try:
            import jax
            return len(jax.devices()) > 0
        except ImportError:
            return False


# Register the backend
register_backend(PallasBackend())
