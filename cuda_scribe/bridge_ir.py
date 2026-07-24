#!/usr/bin/env python3
"""
BridgeIR — Universal Intermediate Representation for GPU Kernels

The core abstraction that sits between CUDA source parsing and multi-target
code generation. Captures block-level computation semantics in a
target-independent form that maps naturally to Triton, Pallas, Metal, SYCL,
WGSL, Vulkan, and Mojo backends.

Design principles:
  1. Block-level (not thread-level) — operations on tiles, not individual elements
  2. Typed tensor tiles with known shapes, dtypes, and memory locations
  3. Explicit memory hierarchy (register → shared → global)
  4. Parametric parallelism (grid/block dims are symbolic for autotuning)
  5. Hardware capability annotations (tensor cores, matrix cores, etc.)
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


# ══════════════════════════════════════════════════════════════════
# TYPES AND ENUMS
# ══════════════════════════════════════════════════════════════════

class DType(Enum):
    """Data types supported across all backends."""
    FLOAT16 = "f16"
    BFLOAT16 = "bf16"
    FLOAT32 = "f32"
    FLOAT64 = "f64"
    INT8 = "i8"
    INT16 = "i16"
    INT32 = "i32"
    INT64 = "i64"
    UINT8 = "u8"
    UINT16 = "u16"
    UINT32 = "u32"
    UINT64 = "u64"
    BOOL = "bool"

    @property
    def bytes(self) -> int:
        sizes = {
            "f16": 2, "bf16": 2, "f32": 4, "f64": 8,
            "i8": 1, "i16": 2, "i32": 4, "i64": 8,
            "u8": 1, "u16": 2, "u32": 4, "u64": 8,
            "bool": 1,
        }
        return sizes[self.value]

    @property
    def is_float(self) -> bool:
        return self.value.startswith(("f", "bf"))

    @property
    def is_integer(self) -> bool:
        return self.value.startswith(("i", "u")) or self == DType.BOOL


class MemoryLevel(Enum):
    """Memory hierarchy levels."""
    REGISTER = "register"
    SHARED = "shared"        # threadgroup (Metal), LDS (AMD), SRAM (TPU)
    GLOBAL = "global"        # HBM, DRAM
    CONSTANT = "constant"    # read-only, often cached


class ParallelPattern(Enum):
    """Recognized parallel computation patterns."""
    MAP = "map"              # Element-wise / embarassingly parallel
    REDUCE = "reduce"        # Tree reduction (sum, max, etc.)
    SCAN = "scan"            # Prefix sum / exclusive scan
    STENCIL = "stencil"      # Neighbor-dependent (conv, blur)
    MATMUL = "matmul"        # Matrix multiplication (tiled)
    ATTENTION = "attention"  # Softmax + matmul (FlashAttention pattern)
    HISTOGRAM = "histogram"  # Bin counting with atomics
    SORT = "sort"            # Parallel sorting
    GATHER = "gather"        # Indirect indexed read
    SCATTER = "scatter"      # Indirect indexed write with atomics
    CUSTOM = "custom"        # Unrecognized pattern


class HardwareRequirement(Enum):
    """Hardware features that may be required by a kernel."""
    BASIC_COMPUTE = auto()       # Any GPU
    SHARED_MEMORY = auto()       # All modern GPUs
    ATOMICS = auto()             # All modern GPUs
    WARP_SHUFFLE = auto()        # NVIDIA warp, AMD wavefront, Metal SIMD group
    TENSOR_CORES = auto()        # NVIDIA WMMA/MMA
    MATRIX_CORES = auto()        # AMD MFMA
    XMX = auto()                 # Intel Xe Matrix Extensions
    DOUBLE_PRECISION = auto()    # Some mobile GPUs lack this
    INT8_DOT = auto()            # Quantized inference


# ══════════════════════════════════════════════════════════════════
# IR NODE TYPES
# ══════════════════════════════════════════════════════════════════

@dataclass
class TileType:
    """Type descriptor for a tile (block of data)."""
    shape: Tuple[Union[int, str], ...]   # Dimensions — int or symbolic name
    dtype: DType
    memory: MemoryLevel = MemoryLevel.GLOBAL

    def __str__(self):
        shape_str = "x".join(str(s) for s in self.shape)
        return f"Tile[{shape_str}, {self.dtype.value}, {self.memory.value}]"


@dataclass
class KernelParam:
    """A kernel parameter (input tensor, scalar, etc.)."""
    name: str
    tile_type: Optional[TileType] = None   # None for scalars
    scalar_dtype: Optional[DType] = None   # For scalar params
    is_output: bool = False
    description: str = ""

    @property
    def is_tensor(self) -> bool:
        return self.tile_type is not None


@dataclass
class IndexExpr:
    """An index expression for load/store operations.

    Can represent:
    - Simple: "batch_idx * seq_len + seq_idx"
    - Compound: decomposition from linear index
    """
    expr: str                            # Python expression string
    depends_on: List[str] = field(default_factory=list)  # Variables referenced
    is_linear_decomposition: bool = False


@dataclass
class LoadOp:
    """Load a tile from memory."""
    result_var: str
    source_param: str                    # Which kernel parameter to load from
    indices: List[IndexExpr]
    mask: Optional[str] = None           # Mask expression (e.g., "offsets < N")
    default_value: Optional[float] = None  # Value for masked-out elements
    cast_to: Optional[DType] = None      # Optional cast after load


@dataclass
class StoreOp:
    """Store a tile to memory."""
    target_param: str
    value_expr: str                      # Expression producing the value
    indices: List[IndexExpr]
    mask: Optional[str] = None


@dataclass
class ComputeOp:
    """An arithmetic/math computation."""
    result_var: str
    op: str                              # "mul", "add", "sub", "div", "exp", "cos", "sin", etc.
    operands: List[str]                  # Variable names or literals
    dtype: Optional[DType] = None


@dataclass
class DotOp:
    """Matrix multiply operation on tiles."""
    result_var: str
    lhs: str
    rhs: str
    accumulator: Optional[str] = None    # For fused multiply-accumulate


@dataclass
class ReduceOp:
    """Reduction operation across a tile dimension."""
    result_var: str
    input_var: str
    op: str                              # "sum", "max", "min", "prod"
    axis: int


@dataclass
class BarrierOp:
    """Synchronization barrier."""
    memory_scope: MemoryLevel = MemoryLevel.SHARED


@dataclass
class AtomicOp:
    """Atomic memory operation."""
    target_param: str
    op: str                              # "add", "max", "min", "cas"
    value_expr: str
    indices: List[IndexExpr]
    mask: Optional[str] = None


@dataclass
class IndexDecomposition:
    """Decompose a linear program ID into multi-dimensional indices.

    Common pattern: tid → (batch, seq, head, pair) via modular arithmetic.
    """
    linear_var: str                      # Source variable (e.g., "offsets")
    dimensions: List[Tuple[str, str]]    # (var_name, dim_size_expr) pairs
    # E.g., [("pair_idx", "half_dim"), ("head_idx", "num_heads"),
    #        ("seq_idx", "seq_len"), ("batch_idx", "batch_size * seq_len * num_heads * half_dim")]
    # Inner-most dimension first (rightmost in memory layout)


# ══════════════════════════════════════════════════════════════════
# KERNEL IR
# ══════════════════════════════════════════════════════════════════

@dataclass
class BridgeKernel:
    """Complete intermediate representation of a GPU kernel."""
    name: str
    description: str = ""

    # Parameters
    params: List[KernelParam] = field(default_factory=list)

    # Grid configuration
    grid_dims: int = 1                   # 1D, 2D, or 3D grid
    grid_expr: List[str] = field(default_factory=list)  # Grid size expressions
    block_size: Union[int, str] = 256    # Block/workgroup size (or symbolic)
    autotunable_block_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    # Scalar constants derived from parameters
    derived_scalars: Dict[str, str] = field(default_factory=dict)
    # E.g., {"half_dim": "head_dim // 2", "total_pairs": "batch_size * seq_len * num_heads * half_dim"}

    # Operations (in order)
    operations: List[Any] = field(default_factory=list)
    # Can contain: LoadOp, StoreOp, ComputeOp, DotOp, ReduceOp, BarrierOp, AtomicOp, IndexDecomposition

    # Metadata
    pattern: ParallelPattern = ParallelPattern.CUSTOM
    hardware_requirements: List[HardwareRequirement] = field(default_factory=list)
    estimated_arithmetic_intensity: Optional[float] = None  # FLOPS per byte
    cuda_source_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for caching/debugging."""
        return {
            "name": self.name,
            "description": self.description,
            "params": [
                {
                    "name": p.name,
                    "tile_type": str(p.tile_type) if p.tile_type else None,
                    "scalar_dtype": p.scalar_dtype.value if p.scalar_dtype else None,
                    "is_output": p.is_output,
                }
                for p in self.params
            ],
            "grid_dims": self.grid_dims,
            "grid_expr": self.grid_expr,
            "block_size": self.block_size,
            "pattern": self.pattern.value,
            "hardware_requirements": [h.name for h in self.hardware_requirements],
            "num_operations": len(self.operations),
            "operation_types": [type(op).__name__ for op in self.operations],
        }

    @property
    def ir_hash(self) -> str:
        """Content-based hash for caching."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════
# CUDA → BridgeIR LIFTER
# ══════════════════════════════════════════════════════════════════

class CUDALifter:
    """Lifts CUDA source code to BridgeIR representation.

    Extends the existing CUDASemanticAnalyzer with structured IR output.
    """

    # Regex patterns for CUDA constructs
    KERNEL_DECL = r'__global__\s+void\s+(\w+)\s*\('
    THREAD_IDX = {
        'threadIdx.x': ('tid_x', 0),
        'threadIdx.y': ('tid_y', 1),
        'threadIdx.z': ('tid_z', 2),
    }
    BLOCK_IDX = {
        'blockIdx.x': ('bid_x', 0),
        'blockIdx.y': ('bid_y', 1),
        'blockIdx.z': ('bid_z', 2),
    }

    # CUDA → BridgeIR type mapping
    CUDA_TYPE_MAP = {
        'float': DType.FLOAT32,
        'double': DType.FLOAT64,
        'half': DType.FLOAT16,
        '__half': DType.FLOAT16,
        '__nv_bfloat16': DType.BFLOAT16,
        'int': DType.INT32,
        'int32_t': DType.INT32,
        'int64_t': DType.INT64,
        'uint32_t': DType.UINT32,
        'unsigned int': DType.UINT32,
        'int8_t': DType.INT8,
        'uint8_t': DType.UINT8,
        'bool': DType.BOOL,
    }

    # CUDA math functions → generic ops
    CUDA_MATH_MAP = {
        'cosf': 'cos', 'sinf': 'sin', 'expf': 'exp', 'logf': 'log',
        'sqrtf': 'sqrt', 'powf': 'pow', 'fabsf': 'abs', 'fmaxf': 'max',
        'fminf': 'min', 'tanhf': 'tanh',
        'cos': 'cos', 'sin': 'sin', 'exp': 'exp', 'log': 'log',
        'sqrt': 'sqrt', 'pow': 'pow', 'fabs': 'abs', 'fmax': 'max',
        'fmin': 'min', 'tanh': 'tanh',
        '__expf': 'exp_fast', '__logf': 'log_fast',
    }

    def lift(self, cuda_source: str, kernel_name: Optional[str] = None) -> BridgeKernel:
        """Lift CUDA source to BridgeIR.

        This is the primary entry point. For complex kernels, the LLM translation
        stage will also receive this IR as context.
        """
        import re

        # Extract kernel name
        if kernel_name is None:
            match = re.search(self.KERNEL_DECL, cuda_source)
            kernel_name = match.group(1) if match else "unknown_kernel"

        kernel = BridgeKernel(name=kernel_name)
        kernel.cuda_source_hash = hashlib.sha256(cuda_source.encode()).hexdigest()[:16]

        # Parse parameters
        kernel.params = self._extract_params(cuda_source, kernel_name)

        # Detect thread model and grid dimensions
        kernel.grid_dims = self._detect_grid_dims(cuda_source)

        # Detect parallel pattern
        kernel.pattern = self._detect_pattern(cuda_source)

        # Detect hardware requirements
        kernel.hardware_requirements = self._detect_hardware_reqs(cuda_source)

        # Extract index decomposition patterns
        decomps = self._extract_index_decompositions(cuda_source)
        if decomps:
            kernel.operations.extend(decomps)

        # Extract derived scalars
        kernel.derived_scalars = self._extract_derived_scalars(cuda_source)

        # Extract load/compute/store pattern
        self._extract_operations(cuda_source, kernel)

        return kernel

    def _extract_params(self, source: str, kernel_name: str) -> List[KernelParam]:
        """Extract kernel parameters from function signature."""
        import re

        # Find the parameter list
        pattern = rf'(?:__global__|void)\s+{re.escape(kernel_name)}\s*\(([^)]+)\)'
        match = re.search(pattern, source, re.DOTALL)
        if not match:
            return []

        param_str = match.group(1)
        params = []

        for p in param_str.split(','):
            p = p.strip()
            if not p:
                continue

            # Remove __restrict__ and const
            clean = p.replace('__restrict__', '').replace('const ', '').strip()

            # Detect pointer (tensor) vs scalar
            is_pointer = '*' in clean
            clean = clean.replace('*', '').strip()

            # Split type and name
            parts = clean.split()
            if len(parts) >= 2:
                type_str = ' '.join(parts[:-1])
                name = parts[-1]
            else:
                type_str = "float"
                name = parts[0] if parts else "unknown"

            dtype = self.CUDA_TYPE_MAP.get(type_str, DType.FLOAT32)

            if is_pointer:
                # Detect if this is an output by checking for writes
                is_output = bool(re.search(rf'{re.escape(name)}\s*\[', source) and
                                 re.search(rf'{re.escape(name)}\s*\[[^\]]+\]\s*=', source))
                params.append(KernelParam(
                    name=name,
                    tile_type=TileType(shape=("N",), dtype=dtype, memory=MemoryLevel.GLOBAL),
                    is_output=is_output,
                ))
            else:
                params.append(KernelParam(
                    name=name,
                    scalar_dtype=dtype,
                ))

        return params

    def _detect_grid_dims(self, source: str) -> int:
        """Detect 1D/2D/3D grid usage."""
        dims = 1
        if 'blockIdx.y' in source or 'threadIdx.y' in source:
            dims = 2
        if 'blockIdx.z' in source or 'threadIdx.z' in source:
            dims = 3
        return dims

    def _detect_pattern(self, source: str) -> ParallelPattern:
        """Detect the dominant parallel pattern."""
        import re

        # Check for matmul patterns
        if re.search(r'(wmma|mma|cublas|__hmma|nvcuda::wmma)', source, re.I):
            return ParallelPattern.MATMUL

        # Check for reduction
        if ('atomicAdd' in source or '__shfl_down_sync' in source or
            '__shfl_xor_sync' in source):
            if 'softmax' in source.lower():
                return ParallelPattern.ATTENTION
            return ParallelPattern.REDUCE

        # Check for scan
        if '__shfl_up_sync' in source or 'inclusive_scan' in source.lower():
            return ParallelPattern.SCAN

        # Check for stencil
        if re.search(r'\[.*[+-]\s*1\s*\]', source) and '__shared__' in source:
            return ParallelPattern.STENCIL

        # Check for histogram
        if 'atomicAdd' in source and 'bin' in source.lower():
            return ParallelPattern.HISTOGRAM

        # Check for gather/scatter
        if re.search(r'\[\s*\w+\s*\[\s*\w+\s*\]\s*\]', source):
            return ParallelPattern.GATHER

        return ParallelPattern.MAP

    def _detect_hardware_reqs(self, source: str) -> List[HardwareRequirement]:
        """Detect hardware requirements from CUDA features used."""
        reqs = [HardwareRequirement.BASIC_COMPUTE]

        if '__shared__' in source:
            reqs.append(HardwareRequirement.SHARED_MEMORY)
        if 'atomicAdd' in source or 'atomicCAS' in source:
            reqs.append(HardwareRequirement.ATOMICS)
        if '__shfl' in source:
            reqs.append(HardwareRequirement.WARP_SHUFFLE)
        if 'wmma' in source or '__hmma' in source or 'mma' in source:
            reqs.append(HardwareRequirement.TENSOR_CORES)
        if 'double' in source:
            reqs.append(HardwareRequirement.DOUBLE_PRECISION)
        if '__dp4a' in source or '__dp2a' in source:
            reqs.append(HardwareRequirement.INT8_DOT)

        return reqs

    def _extract_index_decompositions(self, source: str) -> List[IndexDecomposition]:
        """Extract modular arithmetic index decompositions."""
        import re

        decomps = []
        # Pattern: var = (tid / divisor) % modulus
        # or: var = tid % modulus
        mod_pattern = r'(\w+)\s*=\s*(?:\(?\s*(\w+)\s*/\s*(\([^)]+\)|\w+)\s*\)?\s*%\s*(\w+)|(\w+)\s*%\s*(\w+))'
        for match in re.finditer(mod_pattern, source):
            result_var = match.group(1)
            if result_var.endswith('_idx') or result_var.endswith('_id'):
                decomps.append(IndexDecomposition(
                    linear_var="tid",
                    dimensions=[(result_var, match.group(4) or match.group(6))],
                ))

        return decomps

    def _extract_derived_scalars(self, source: str) -> Dict[str, str]:
        """Extract const scalar computations."""
        import re
        scalars = {}
        pattern = r'const\s+(?:int|float|unsigned\s+int)\s+(\w+)\s*=\s*([^;]+);'
        for match in re.finditer(pattern, source):
            name = match.group(1)
            expr = match.group(2).strip()
            if name not in ('tid',):  # Skip thread index
                scalars[name] = expr
        return scalars

    def _extract_operations(self, source: str, kernel: BridgeKernel):
        """Extract load/compute/store operations from kernel body."""
        import re

        # Find array loads: var = array[index]
        load_pattern = r'const\s+float\s+(\w+)\s*=\s*(\w+)\[([^\]]+)\]'
        for match in re.finditer(load_pattern, source):
            result_var = match.group(1)
            source_param = match.group(2)
            index_expr = match.group(3)
            kernel.operations.append(LoadOp(
                result_var=result_var,
                source_param=source_param,
                indices=[IndexExpr(expr=index_expr)],
            ))

        # Find stores: array[index] = expr
        store_pattern = r'(\w+)\[([^\]]+)\]\s*=\s*([^;]+);'
        for match in re.finditer(store_pattern, source):
            target_param = match.group(1)
            index_expr = match.group(2)
            value_expr = match.group(3).strip()
            # Only include actual output stores (not local arrays)
            if any(p.name == target_param and p.is_output for p in kernel.params):
                kernel.operations.append(StoreOp(
                    target_param=target_param,
                    value_expr=value_expr,
                    indices=[IndexExpr(expr=index_expr)],
                ))


# ══════════════════════════════════════════════════════════════════
# BACKEND PLUGIN INTERFACE
# ══════════════════════════════════════════════════════════════════

class BackendPlugin:
    """Abstract base for target code generation backends."""

    def name(self) -> str:
        raise NotImplementedError

    def supported_hardware(self) -> List[str]:
        raise NotImplementedError

    def translate(self, kernel: BridgeKernel, cuda_source: str = "") -> str:
        """Generate target code from BridgeIR."""
        raise NotImplementedError

    def verify(self, source: str, reference: Any = None) -> Dict[str, Any]:
        """Verify translated code for correctness."""
        raise NotImplementedError

    def benchmark(self, source: str) -> Dict[str, Any]:
        """Benchmark translated code performance."""
        raise NotImplementedError

    def is_hardware_available(self) -> bool:
        """Check if target hardware is accessible."""
        return False


# ══════════════════════════════════════════════════════════════════
# BACKEND REGISTRY
# ══════════════════════════════════════════════════════════════════

_BACKENDS: Dict[str, BackendPlugin] = {}


def register_backend(backend: BackendPlugin):
    """Register a new backend plugin."""
    _BACKENDS[backend.name()] = backend


def get_backend(name: str) -> BackendPlugin:
    """Get a registered backend by name."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}")
    return _BACKENDS[name]


def available_backends() -> List[str]:
    """List all registered backends."""
    return list(_BACKENDS.keys())


# ══════════════════════════════════════════════════════════════════
# UNIVERSAL TRANSLATE FUNCTION
# ══════════════════════════════════════════════════════════════════

def translate_universal(
    cuda_source: str,
    targets: Optional[List[str]] = None,
    kernel_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Translate CUDA source to multiple target backends.

    Args:
        cuda_source: CUDA kernel source code
        targets: List of backend names (None = all available)
        kernel_name: Optional kernel name override

    Returns:
        Dict mapping backend name → translation result dict
    """
    if targets is None:
        targets = available_backends()

    # Stage 0: Lift to BridgeIR (shared across all targets)
    lifter = CUDALifter()
    bridge_ir = lifter.lift(cuda_source, kernel_name)

    results = {}
    for target in targets:
        try:
            backend = get_backend(target)
            translated = backend.translate(bridge_ir, cuda_source)
            results[target] = {
                "success": True,
                "source": translated,
                "backend": target,
                "ir_hash": bridge_ir.ir_hash,
                "pattern": bridge_ir.pattern.value,
            }
        except Exception as e:
            results[target] = {
                "success": False,
                "error": str(e),
                "backend": target,
            }

    return results
