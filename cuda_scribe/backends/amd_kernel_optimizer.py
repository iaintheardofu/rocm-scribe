#!/usr/bin/env python3
"""
AMD Kernel Optimizer — Compiler-level optimization passes for AMD GPUs.

Applies source-level transformations that align HIP/Triton kernels with
AMD GPU microarchitecture for maximum performance. These passes operate
on the translated HIP C++ or Triton source code.

Optimization categories:
  1. Wavefront divergence elimination (branchless patterns)
  2. LDS tiling optimization (bank conflict avoidance)
  3. Memory coalescing for 64-wide wavefronts
  4. Register pressure management (VGPR/SGPR budgeting)
  5. Instruction scheduling hints (MFMA latency hiding)
  6. Occupancy-driven block size selection
  7. Data layout transformation (AoS→SoA)

AMD Architecture Reference (CDNA3 / MI300X / gfx942):
  - 304 Compute Units, each with 4 SIMD units
  - 512 VGPRs per SIMD unit (architectural), 256 addressable per wave
  - 106 SGPRs per wavefront
  - 64KB LDS per CU, 32 banks, 4-byte bank width
  - 256MB L2 cache (unified across all XCDs)
  - 5.3 TB/s HBM3 aggregate bandwidth (8 stacks)
  - MFMA: v_mfma_f32_32x32x8f16 (peak instruction)

Compiler flags for MI300X:
  hipcc --offload-arch=gfx942 -O3 -ffast-math

Algorithmica HPC principles applied:
  §3.3  Branchless → predicated wavefront operations
  §8.1  Memory hierarchy → register > LDS > L1 > L2 > HBM
  §8.7  Cache-oblivious → recursive tiling for LDS
  §9.1  Bandwidth → coalesced 256-byte transactions (64 threads × 4 bytes)
  §9.11 SoA → structure-of-arrays for GPU coalescing
  §10.3 SIMD → wavefront-wide reductions (6 steps for 64 threads)
  §10.6 Auto-vectorization → block sizes as multiples of 64
  §11.15 GEMM → MFMA tiling with register blocking
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .amd_wavefront_optimizer import AMDArch, AMD_SPECS, AMDHardwareSpec


# ═══════════════════════════════════════════════════════════════════
# MFMA INSTRUCTION REFERENCE (CDNA3 / MI300X / gfx942)
# ═══════════════════════════════════════════════════════════════════
# Each entry: (output_shape, k_dim, input_type, accum_type,
#              vgprs_a, vgprs_b, vgprs_c, latency_cycles, throughput_per_cu)
MFMA_INSTRUCTIONS = {
    # FP16 input, FP32 accumulate
    "v_mfma_f32_32x32x8_f16":  {"M": 32, "N": 32, "K": 8,  "in": "fp16", "acc": "fp32",
                                 "vgpr_a": 4, "vgpr_b": 4, "vgpr_c": 16, "latency": 16, "flops": 16384},
    "v_mfma_f32_16x16x16_f16": {"M": 16, "N": 16, "K": 16, "in": "fp16", "acc": "fp32",
                                 "vgpr_a": 4, "vgpr_b": 4, "vgpr_c": 4,  "latency": 8,  "flops": 8192},
    "v_mfma_f32_4x4x4_f16":   {"M": 4,  "N": 4,  "K": 4,  "in": "fp16", "acc": "fp32",
                                 "vgpr_a": 4, "vgpr_b": 4, "vgpr_c": 4,  "latency": 8,  "flops": 128},
    # BF16 input, FP32 accumulate
    "v_mfma_f32_32x32x16_bf16": {"M": 32, "N": 32, "K": 16, "in": "bf16", "acc": "fp32",
                                  "vgpr_a": 8, "vgpr_b": 8, "vgpr_c": 16, "latency": 16, "flops": 32768},
    "v_mfma_f32_16x16x32_bf16": {"M": 16, "N": 16, "K": 32, "in": "bf16", "acc": "fp32",
                                  "vgpr_a": 8, "vgpr_b": 8, "vgpr_c": 4,  "latency": 8,  "flops": 16384},
    # FP8 input, FP32 accumulate (CDNA3 / MI300X only)
    "v_mfma_f32_32x32x16_fp8": {"M": 32, "N": 32, "K": 16, "in": "fp8", "acc": "fp32",
                                 "vgpr_a": 4, "vgpr_b": 4, "vgpr_c": 16, "latency": 16, "flops": 32768},
    "v_mfma_f32_16x16x32_fp8": {"M": 16, "N": 16, "K": 32, "in": "fp8", "acc": "fp32",
                                 "vgpr_a": 4, "vgpr_b": 4, "vgpr_c": 4,  "latency": 8,  "flops": 16384},
    # INT8 input, INT32 accumulate
    "v_mfma_i32_32x32x16_i8":  {"M": 32, "N": 32, "K": 16, "in": "int8", "acc": "int32",
                                 "vgpr_a": 4, "vgpr_b": 4, "vgpr_c": 16, "latency": 16, "flops": 32768},
    "v_mfma_i32_16x16x32_i8":  {"M": 16, "N": 16, "K": 32, "in": "int8", "acc": "int32",
                                 "vgpr_a": 4, "vgpr_b": 4, "vgpr_c": 4,  "latency": 8,  "flops": 16384},
}

# VGPR → max wavefronts per SIMD lookup (CDNA3, 512 VGPRs/SIMD, granularity=8)
VGPR_OCCUPANCY_TABLE = [
    # (max_vgprs, max_wavefronts_per_simd)
    (48,  10),  # ≤48 VGPRs → 10 wavefronts (max)
    (56,   9),
    (64,   8),
    (80,   6),
    (96,   5),
    (128,  4),
    (168,  3),
    (256,  2),
    (512,  1),
]


def lookup_vgpr_occupancy(vgprs_per_thread: int) -> int:
    """Look up max wavefronts per SIMD from VGPR count (CDNA3)."""
    vgpr_alloc = ((vgprs_per_thread + 7) // 8) * 8  # round up to granularity of 8
    for max_v, max_wf in VGPR_OCCUPANCY_TABLE:
        if vgpr_alloc <= max_v:
            return max_wf
    return 0  # > 512 VGPRs: cannot launch


def select_mfma_instruction(dtype: str = "fp16", prefer_large: bool = True) -> dict:
    """Select optimal MFMA instruction for given data type.

    Args:
        dtype: Input data type (fp16, bf16, fp8, int8)
        prefer_large: If True, prefer 32x32 tiles (more compute per instruction);
                      if False, prefer 16x16 tiles (lower latency, higher occupancy)
    """
    candidates = {k: v for k, v in MFMA_INSTRUCTIONS.items()
                  if v["in"] == dtype}
    if not candidates:
        # Default to fp16
        candidates = {k: v for k, v in MFMA_INSTRUCTIONS.items()
                      if v["in"] == "fp16"}

    if prefer_large:
        # Prefer largest output tile (most compute per instruction)
        return max(candidates.values(), key=lambda v: v["M"] * v["N"])
    else:
        # Prefer 16x16 (lower latency, better for small matrices)
        candidates_16 = {k: v for k, v in candidates.items() if v["M"] == 16}
        if candidates_16:
            return max(candidates_16.values(), key=lambda v: v["K"])
        return max(candidates.values(), key=lambda v: v["K"])


class OptimizationLevel(Enum):
    """Optimization aggressiveness."""
    CONSERVATIVE = "O1"   # Safe, correctness-preserving only
    BALANCED = "O2"       # Performance + correctness (default)
    AGGRESSIVE = "O3"     # Maximum performance, may change numerical results


@dataclass
class OptimizationReport:
    """Report of optimizations applied to a kernel."""
    passes_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    estimated_speedup: float = 1.0
    register_pressure: Optional[Dict] = None
    lds_usage_bytes: int = 0
    recommended_block_size: int = 256
    recommended_compiler_flags: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# OPTIMIZATION PASSES
# ═══════════════════════════════════════════════════════════════════

def optimize_hip_kernel(
    source: str,
    arch: AMDArch = AMDArch.CDNA3,
    level: OptimizationLevel = OptimizationLevel.BALANCED,
) -> Tuple[str, OptimizationReport]:
    """Apply all AMD-specific optimizations to a HIP kernel.

    Args:
        source: HIP C++ kernel source code
        arch: Target AMD architecture
        level: Optimization aggressiveness

    Returns:
        Tuple of (optimized_source, optimization_report)
    """
    spec = AMD_SPECS[arch]
    report = OptimizationReport()

    # Compiler flags
    report.recommended_compiler_flags = _get_compiler_flags(arch, level)

    # Pass 1: Wavefront divergence elimination
    source = _eliminate_wavefront_divergence(source, spec, report)

    # Pass 2: LDS bank conflict avoidance
    source = _optimize_lds_banking(source, spec, report)

    # Pass 3: Memory coalescing
    source = _optimize_memory_coalescing(source, spec, report)

    # Pass 4: Register pressure management
    source = _optimize_register_pressure(source, spec, report)

    # Pass 5: Block size optimization
    source = _optimize_block_size(source, spec, report)

    # Pass 6: Loop unrolling hints
    if level in (OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE):
        source = _add_unroll_hints(source, report)

    # Pass 7: Instruction scheduling hints
    if level == OptimizationLevel.AGGRESSIVE:
        source = _add_scheduling_hints(source, spec, report)

    # Pass 8: AoS → SoA transformation hints
    source = _detect_aos_patterns(source, report)

    return source, report


def _get_compiler_flags(arch: AMDArch, level: OptimizationLevel) -> List[str]:
    """Get recommended hipcc compiler flags."""
    arch_to_gfx = {
        AMDArch.CDNA3: "gfx942",
        AMDArch.CDNA2: "gfx90a",
        AMDArch.CDNA1: "gfx908",
        AMDArch.RDNA3: "gfx1100",
        AMDArch.RDNA2: "gfx1030",
    }
    gfx = arch_to_gfx.get(arch, "gfx942")

    flags = [
        f"--offload-arch={gfx}",
        f"-O{level.value[-1]}",  # O1, O2, O3
    ]

    if level == OptimizationLevel.AGGRESSIVE:
        flags.extend([
            "-ffast-math",
            "-funsafe-math-optimizations",
            "-fno-hip-fp32-correctly-rounded-divide-sqrt",
            "-mllvm", "-amdgpu-early-inline-all=true",
            "-mllvm", "-amdgpu-function-calls=false",
        ])
    elif level == OptimizationLevel.BALANCED:
        flags.extend([
            "-ffast-math",
        ])

    return flags


def _eliminate_wavefront_divergence(
    source: str, spec: AMDHardwareSpec, report: OptimizationReport
) -> str:
    """Eliminate wavefront divergence using branchless patterns.

    Algorithmica §3.3: Branchless programming — on GPU, divergent branches
    in a wavefront cause serialization. Prefer predicated/select operations.

    AMD CDNA: All 64 threads in a wavefront execute in lockstep. A branch
    where some threads take one path and others take another forces the
    wavefront to execute BOTH paths sequentially.

    Pattern 1: if (tid < N) val = x; else val = 0;
           →  val = (tid < N) ? x : 0;  // Compiler can use v_cndmask_b32

    Pattern 2: if (cond) atomicAdd(...);
           →  Keep as-is (atomics are inherently conditional)
    """
    # Detect simple if/else that can be converted to ternary
    # This is a conservative pattern — only matches simple assignments
    simple_branch = re.compile(
        r'if\s*\(([^)]+)\)\s*\{\s*(\w+)\s*=\s*([^;]+);\s*\}'
        r'\s*else\s*\{\s*\2\s*=\s*([^;]+);\s*\}',
        re.MULTILINE,
    )
    matches = list(simple_branch.finditer(source))
    if matches:
        for m in reversed(matches):
            cond = m.group(1)
            var = m.group(2)
            true_val = m.group(3)
            false_val = m.group(4)
            replacement = f"{var} = ({cond}) ? ({true_val}) : ({false_val});"
            source = source[:m.start()] + replacement + source[m.end():]
        report.passes_applied.append(
            f"wavefront_divergence_elimination: {len(matches)} branches → ternary"
        )
        report.estimated_speedup *= 1.05

    return source


def _optimize_lds_banking(
    source: str, spec: AMDHardwareSpec, report: OptimizationReport
) -> str:
    """Optimize LDS (shared memory) to avoid bank conflicts.

    AMD LDS: 32 banks, 4-byte bank width.
    With 64-wide wavefronts, 2 threads per bank access simultaneously.

    Bank conflict condition:
      Two threads in the same wavefront access the same bank
      but different addresses within that bank.

    Solution: Add +1 padding to shared memory array dimensions
    that are multiples of 32 (the bank count).

    Example:
      __shared__ float smem[256];      // 256 % 32 == 0 → bank conflicts!
      __shared__ float smem[256 + 1];  // No bank conflicts for stride-1

    Algorithmica §9.3: Cache line optimization — equivalent principle
    applied to LDS banks.
    """
    # 2D shared memory arrays — pad the innermost (column) dimension
    # MUST run before 1D pattern to avoid partial matches
    shared_2d_pattern = re.compile(
        r'__shared__\s+(\w+)\s+(\w+)\[(\d+)\]\[(\d+)\]',
    )

    def _pad_2d_if_needed(match):
        dtype = match.group(1)
        name = match.group(2)
        dim0 = int(match.group(3))
        dim1 = int(match.group(4))
        if dim1 >= 32 and dim1 % 32 == 0:
            report.passes_applied.append(
                f"lds_bank_padding_2d: {name}[{dim0}][{dim1}] → {name}[{dim0}][{dim1}+1]"
            )
            return f"__shared__ {dtype} {name}[{dim0}][{dim1} + 1]  /* AMD LDS bank padding */"
        return match.group(0)

    source = shared_2d_pattern.sub(_pad_2d_if_needed, source)

    # 1D shared memory declarations (only match single-dimension arrays)
    shared_pattern = re.compile(
        r'__shared__\s+(\w+)\s+(\w+)\[(\d+)\](?!\s*[\[+])',
    )

    def _pad_if_needed(match):
        dtype = match.group(1)
        name = match.group(2)
        size = int(match.group(3))
        if size >= 32 and size % 32 == 0:
            report.passes_applied.append(
                f"lds_bank_padding: {name}[{size}] → {name}[{size}+1]"
            )
            return f"__shared__ {dtype} {name}[{size} + 1]  /* AMD LDS bank conflict padding */"
        return match.group(0)

    source = shared_pattern.sub(_pad_if_needed, source)

    # Calculate LDS usage
    lds_usage = 0
    for match in re.finditer(r'__shared__\s+(\w+)\s+\w+\[([^\]]+)\]', source):
        dtype = match.group(1)
        dtype_size = {"float": 4, "double": 8, "half": 2, "int": 4, "int32_t": 4}.get(dtype, 4)
        try:
            size_expr = match.group(2).replace('+', ' + ')
            # Simple evaluation of "N" or "N + 1"
            parts = size_expr.split('+')
            total = sum(int(p.strip()) for p in parts if p.strip().isdigit())
            lds_usage += total * dtype_size
        except (ValueError, TypeError):
            pass
    report.lds_usage_bytes = lds_usage

    return source


def _optimize_memory_coalescing(
    source: str, spec: AMDHardwareSpec, report: OptimizationReport
) -> str:
    """Ensure memory access patterns are coalesced for 64-wide wavefronts.

    AMD coalescing rules (CDNA3):
    - A wavefront issues a single memory request for consecutive addresses
    - Coalesced: thread i accesses address base + i * sizeof(element)
    - Uncoalesced: thread i accesses address base + i * stride (stride > 1)
    - Transaction size: 64 bytes (16 float32s) per cache line
    - A fully coalesced wavefront needs 4 cache lines (64 × 4B = 256B)

    Algorithmica §9.1: Memory bandwidth — coalesced access is the single
    most important optimization for memory-bound kernels.
    """
    # Detect strided access patterns
    strided_access = re.findall(
        r'(\w+)\[\s*(\w+)\s*\*\s*(\d+)\s*\+\s*(\w+)\s*\]',
        source,
    )
    for array, outer, stride, inner in strided_access:
        stride_val = int(stride)
        if stride_val > 1 and inner in ('threadIdx.x', 'tid', 'ltid', 'lane'):
            report.warnings.append(
                f"Non-coalesced access detected: {array}[{outer}*{stride}+{inner}]. "
                f"Consider transposing data layout for coalesced access."
            )

    # Detect __ldg usage (NVIDIA-specific, not needed on AMD)
    if '__ldg(' in source:
        source = re.sub(r'__ldg\(([^)]+)\)', r'*(\1)', source)
        report.passes_applied.append("ldg_removal: __ldg() → direct load (AMD L2 handles caching)")

    return source


def _optimize_register_pressure(
    source: str, spec: AMDHardwareSpec, report: OptimizationReport
) -> str:
    """Manage register pressure for AMD occupancy.

    AMD CDNA3 (MI300X):
    - 512 architectural VGPRs per SIMD unit, 256 addressable per wave
    - 32 max wavefronts per CU
    - VGPRs allocated in granularity of 8 (for wave64)

    Occupancy vs VGPRs (CDNA3, per SIMD unit):
      VGPRs ≤ 64:  8 waves (100% occupancy)
      VGPRs ≤ 128: 4 waves (50%)
      VGPRs ≤ 256: 2 waves (25%)

    Trade-off: More VGPRs = more ILP but lower occupancy.
    For bandwidth-bound kernels: prefer high occupancy (fewer VGPRs).
    For compute-bound kernels: prefer more VGPRs for register blocking.
    """
    # Count approximate VGPR usage by counting local float variables
    float_vars = len(re.findall(r'\bfloat\s+\w+\s*[=;]', source))
    double_vars = len(re.findall(r'\bdouble\s+\w+\s*[=;]', source))
    half_vars = len(re.findall(r'\bhalf\s+\w+\s*[=;]', source))

    # Rough estimate: each float var ≈ 1 VGPR, double ≈ 2 VGPRs
    estimated_vgprs = float_vars + double_vars * 2 + half_vars + 8  # +8 for implicit

    report.register_pressure = {
        "estimated_vgprs": estimated_vgprs,
        "float_vars": float_vars,
        "double_vars": double_vars,
        "max_vgprs_per_wave": 256,
        "occupancy_estimate": _estimate_occupancy_from_vgprs(estimated_vgprs, spec),
    }

    if estimated_vgprs > 128:
        report.warnings.append(
            f"High register pressure ({estimated_vgprs} est. VGPRs). "
            f"Occupancy likely ≤ 50%. Consider: "
            f"(1) __launch_bounds__(256, 4) to hint compiler, "
            f"(2) Spilling infrequently-used values to LDS, "
            f"(3) Reducing live variable count."
        )
        # Add launch_bounds if not present
        if '__launch_bounds__' not in source:
            source = re.sub(
                r'__global__\s+void\s+(\w+)',
                r'__global__ __launch_bounds__(256, 4) void \1',
                source,
                count=1,
            )
            report.passes_applied.append("launch_bounds_added: __launch_bounds__(256, 4)")

    return source


def _estimate_occupancy_from_vgprs(vgprs: int, spec: AMDHardwareSpec) -> str:
    """Estimate occupancy category from VGPR count."""
    if vgprs <= 64:
        return "high (≥75%)"
    elif vgprs <= 128:
        return "medium (50%)"
    elif vgprs <= 192:
        return "low (33%)"
    else:
        return "very low (25%)"


def _optimize_block_size(
    source: str, spec: AMDHardwareSpec, report: OptimizationReport
) -> str:
    """Ensure block sizes are aligned to wavefront width.

    AMD wavefront width: 64 (CDNA) or 32 (RDNA wave32).
    Block sizes not multiples of wavefront width waste threads.

    Algorithmica §10.6: Auto-vectorization — match hardware vector width.
    """
    wavefront = spec.wavefront_width

    # Find block size definitions
    block_defs = re.finditer(
        r'(?:const\s+int\s+|int\s+|#define\s+)(\w*block\w*|BLOCK_SIZE|THREADS)\s*[=\s]+\s*(\d+)',
        source,
        re.IGNORECASE,
    )
    for match in block_defs:
        name = match.group(1)
        size = int(match.group(2))
        if size % wavefront != 0:
            aligned = ((size + wavefront - 1) // wavefront) * wavefront
            if aligned <= 1024:
                report.warnings.append(
                    f"{name}={size} not aligned to wavefront width ({wavefront}). "
                    f"Recommend {aligned}."
                )
                report.recommended_block_size = aligned

    return source


def _add_unroll_hints(source: str, report: OptimizationReport) -> str:
    """Add #pragma unroll to tight inner loops.

    AMD compiler (amd-clang) respects #pragma unroll for loop unrolling.
    This is particularly important for:
    - GEMM inner K loop
    - Reduction accumulation loops
    - Small, fixed-iteration loops
    """
    # Find loops with small, known trip counts
    small_loop = re.compile(
        r'(for\s*\([^;]+;\s*\w+\s*<\s*(\d+)\s*;[^)]+\)\s*\{)',
    )
    for match in small_loop.finditer(source):
        trip_count = int(match.group(2))
        if trip_count <= 32 and '#pragma unroll' not in source[max(0, match.start()-30):match.start()]:
            source = source[:match.start()] + f"#pragma unroll\n    {match.group(1)}" + source[match.end():]
            report.passes_applied.append(f"unroll_hint: loop with trip count {trip_count}")

    return source


def _add_scheduling_hints(
    source: str, spec: AMDHardwareSpec, report: OptimizationReport
) -> str:
    """Add instruction scheduling hints for AMD GPUs.

    AMD CDNA MFMA instructions have multi-cycle latency:
    - v_mfma_f32_16x16x16f16: 8 cycles
    - v_mfma_f32_32x32x8f16: 16 cycles

    To hide this latency, interleave MFMA with memory loads
    (software pipelining). The compiler does some of this automatically
    with -O3, but hints help.
    """
    if spec.has_mfma and 'mfma' in source.lower():
        report.passes_applied.append(
            "scheduling_hint: MFMA detected — ensure software pipelining"
        )
        report.warnings.append(
            "MFMA instructions detected. Ensure loads from next K-tile are "
            "issued BEFORE current MFMA to hide latency. Use num_stages ≥ 2 "
            "in Triton autotune configs."
        )

    return source


def _detect_aos_patterns(source: str, report: OptimizationReport) -> str:
    """Detect Array-of-Structures (AoS) patterns that hurt coalescing.

    Algorithmica §9.11: SoA (Structure of Arrays) is critical for GPU
    memory coalescing. AoS layouts cause strided access.

    AoS: struct { float x, y, z; } particles[N];
    SoA: struct { float x[N]; float y[N]; float z[N]; } particles;

    With 64-wide wavefronts on AMD, AoS impact is 2x worse than NVIDIA
    because each transaction covers 64 threads instead of 32.
    """
    # Detect struct array access patterns
    struct_access = re.findall(
        r'(\w+)\[(\w+)\]\.(\w+)',
        source,
    )
    if struct_access:
        arrays = set(s[0] for s in struct_access)
        for arr in arrays:
            fields = set(s[2] for s in struct_access if s[0] == arr)
            if len(fields) > 1:
                report.warnings.append(
                    f"AoS pattern detected: {arr}[i].{{{'|'.join(fields)}}}. "
                    f"Consider SoA layout for {len(fields)}x better coalescing "
                    f"on 64-wide AMD wavefronts."
                )

    return source


# ═══════════════════════════════════════════════════════════════════
# TRITON AMD AUTOTUNING
# ═══════════════════════════════════════════════════════════════════

def generate_triton_autotune_configs(
    pattern: str,
    arch: AMDArch = AMDArch.CDNA3,
) -> str:
    """Generate Triton autotuning configurations optimized for AMD.

    AMD-specific Triton tuning considerations:
    1. num_warps: In Triton, a "warp" is always 32 threads.
       On AMD with wave64, Triton handles the mapping:
       - 4 Triton warps = 128 threads = 2 AMD wavefronts
       - 8 Triton warps = 256 threads = 4 AMD wavefronts

    2. BLOCK sizes: Must be multiples of 64 for wave64 mode.
       Powers of 2 preferred for efficient memory addressing.

    3. num_stages: Software pipelining depth.
       More stages = more register pressure but better latency hiding.
       AMD MI300X has deep memory pipeline, 3-4 stages optimal.

    4. waves_per_eu: AMD-specific hint for occupancy.
       Lower = more registers per wave, higher = better occupancy.
    """
    spec = AMD_SPECS[arch]

    if pattern == "matmul":
        return textwrap.dedent(f"""\
        # AMD MI300X-optimized GEMM autotuning configs
        # Target: {spec.compute_units} CUs, {spec.lds_per_cu_kb}KB LDS, MFMA
        @triton.autotune(
            configs=[
                # Large tiles — high compute intensity, MFMA-optimized
                triton.Config({{'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32}},
                              num_warps=8, num_stages=3),
                triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}},
                              num_warps=8, num_stages=3),
                triton.Config({{'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}},
                              num_warps=8, num_stages=3),
                # Medium tiles — balanced compute/memory
                triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}},
                              num_warps=4, num_stages=4),
                triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}},
                              num_warps=4, num_stages=3),
                # Small tiles — high occupancy, for small matrices
                triton.Config({{'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}},
                              num_warps=4, num_stages=4),
                triton.Config({{'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}},
                              num_warps=4, num_stages=4),
            ],
            key=['M', 'N', 'K'],
        )""")

    elif pattern == "attention":
        return textwrap.dedent(f"""\
        # AMD MI300X-optimized FlashAttention autotuning configs
        # 192GB HBM3 unified memory — can handle larger sequences
        @triton.autotune(
            configs=[
                triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DHEAD': 64}},
                              num_warps=4, num_stages=2),
                triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_DHEAD': 64}},
                              num_warps=4, num_stages=3),
                triton.Config({{'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_DHEAD': 64}},
                              num_warps=4, num_stages=3),
                triton.Config({{'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_DHEAD': 128}},
                              num_warps=4, num_stages=2),
            ],
            key=['N_CTX', 'DHEAD'],
        )""")

    elif pattern == "reduce":
        return textwrap.dedent(f"""\
        # AMD MI300X-optimized reduction autotuning configs
        # 64-wide wavefronts: larger blocks are more efficient
        @triton.autotune(
            configs=[
                triton.Config({{'BLOCK_SIZE': 1024}}, num_warps=8),
                triton.Config({{'BLOCK_SIZE': 512}}, num_warps=8),
                triton.Config({{'BLOCK_SIZE': 256}}, num_warps=4),
                triton.Config({{'BLOCK_SIZE': 2048}}, num_warps=16),
            ],
            key=['n_elements'],
        )""")

    elif pattern == "softmax":
        return textwrap.dedent(f"""\
        # AMD MI300X-optimized softmax autotuning configs
        @triton.autotune(
            configs=[
                triton.Config({{'BLOCK_SIZE': 1024}}, num_warps=8),
                triton.Config({{'BLOCK_SIZE': 2048}}, num_warps=16),
                triton.Config({{'BLOCK_SIZE': 512}}, num_warps=4),
                triton.Config({{'BLOCK_SIZE': 4096}}, num_warps=32),
            ],
            key=['n_cols'],
        )""")

    elif pattern == "layernorm":
        return textwrap.dedent(f"""\
        # AMD MI300X-optimized LayerNorm autotuning configs
        @triton.autotune(
            configs=[
                triton.Config({{'BLOCK_SIZE': 1024}}, num_warps=8),
                triton.Config({{'BLOCK_SIZE': 2048}}, num_warps=16),
                triton.Config({{'BLOCK_SIZE': 512}}, num_warps=4),
            ],
            key=['n_cols'],
        )""")

    else:  # elementwise / map
        return textwrap.dedent(f"""\
        # AMD MI300X-optimized elementwise autotuning configs
        # Maximize bandwidth utilization: 5.3 TB/s HBM3
        @triton.autotune(
            configs=[
                triton.Config({{'BLOCK_SIZE': 1024}}, num_warps=4),
                triton.Config({{'BLOCK_SIZE': 2048}}, num_warps=8),
                triton.Config({{'BLOCK_SIZE': 512}}, num_warps=4),
                triton.Config({{'BLOCK_SIZE': 4096}}, num_warps=16),
            ],
            key=['n_elements'],
        )""")


# ═══════════════════════════════════════════════════════════════════
# AMD-SPECIFIC TRITON KERNEL TEMPLATES
# ═══════════════════════════════════════════════════════════════════

def generate_triton_softmax_amd() -> str:
    """Generate AMD-optimized Triton softmax kernel."""
    return textwrap.dedent("""\
    # AMD Bridge — Triton Softmax (MI300X-optimized)
    # Wavefront-portable: auto-adapts to 64-wide (AMD) and 32-wide (NVIDIA)
    import triton
    import triton.language as tl
    import torch

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        ],
        key=['n_cols'],
    )
    @triton.jit
    def softmax_kernel(
        output_ptr, input_ptr,
        input_row_stride, output_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols

        # Load row — coalesced for both 32-wide and 64-wide
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # Numerically stable softmax
        row_max = tl.max(row, axis=0)
        numerator = tl.exp(row - row_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # Store — coalesced
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

    def softmax(x: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)  # Minimum for AMD wavefront efficiency
        y = torch.empty_like(x)
        softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols)
        return y
    """)


def generate_triton_layernorm_amd() -> str:
    """Generate AMD-optimized Triton LayerNorm kernel."""
    return textwrap.dedent("""\
    # AMD Bridge — Triton LayerNorm (MI300X-optimized)
    import triton
    import triton.language as tl
    import torch

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        ],
        key=['n_cols'],
    )
    @triton.jit
    def layernorm_kernel(
        output_ptr, input_ptr, weight_ptr, bias_ptr,
        input_row_stride, output_row_stride,
        n_cols, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load input row (coalesced)
        row_start = input_ptr + row_idx * input_row_stride
        x = tl.load(row_start + col_offsets, mask=mask, other=0.0)

        # Compute mean and variance
        mean = tl.sum(x, axis=0) / n_cols
        x_centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(x_centered * x_centered, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)

        # Normalize and apply affine
        x_norm = x_centered * rstd
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
        output = x_norm * weight + bias

        # Store (coalesced)
        out_start = output_ptr + row_idx * output_row_stride
        tl.store(out_start + col_offsets, output, mask=mask)
    """)


def generate_triton_rope_amd() -> str:
    """Generate AMD-optimized Triton RoPE kernel."""
    return textwrap.dedent("""\
    # AMD Bridge — Triton Rotary Positional Encoding (MI300X-optimized)
    import triton
    import triton.language as tl
    import torch
    import math

    @triton.jit
    def rope_kernel(
        output_ptr, input_ptr, cos_ptr, sin_ptr,
        seq_len, n_heads, head_dim,
        input_batch_stride, input_seq_stride,
        input_head_stride, input_dim_stride,
        output_batch_stride, output_seq_stride,
        output_head_stride, output_dim_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        half_dim = head_dim // 2

        # Decompose program ID into (batch, seq, head) indices
        num_per_batch = seq_len * n_heads
        batch_idx = pid // num_per_batch
        remainder = pid % num_per_batch
        seq_idx = remainder // n_heads
        head_idx = remainder % n_heads

        # Load paired elements
        dim_offsets = tl.arange(0, BLOCK_SIZE)
        mask = dim_offsets < half_dim

        base = (batch_idx * input_batch_stride +
                seq_idx * input_seq_stride +
                head_idx * input_head_stride)

        x0 = tl.load(input_ptr + base + dim_offsets * 2 * input_dim_stride,
                      mask=mask, other=0.0)
        x1 = tl.load(input_ptr + base + (dim_offsets * 2 + 1) * input_dim_stride,
                      mask=mask, other=0.0)

        # Load cos/sin (precomputed)
        cos_val = tl.load(cos_ptr + seq_idx * half_dim + dim_offsets, mask=mask, other=1.0)
        sin_val = tl.load(sin_ptr + seq_idx * half_dim + dim_offsets, mask=mask, other=0.0)

        # Apply rotation
        y0 = x0 * cos_val - x1 * sin_val
        y1 = x0 * sin_val + x1 * cos_val

        # Store
        out_base = (batch_idx * output_batch_stride +
                    seq_idx * output_seq_stride +
                    head_idx * output_head_stride)

        tl.store(output_ptr + out_base + dim_offsets * 2 * output_dim_stride,
                 y0, mask=mask)
        tl.store(output_ptr + out_base + (dim_offsets * 2 + 1) * output_dim_stride,
                 y1, mask=mask)
    """)


def generate_triton_gemm_amd() -> str:
    """Generate AMD-optimized Triton GEMM kernel with MFMA tiling.

    Uses 128x128x32 tile config optimized for MI300X:
    - Compiles to v_mfma_f32_32x32x8_f16 (4x4 register block)
    - 32KB LDS (fits in 64KB with double buffering)
    - ~256 VGPRs → 2 wavefronts/SIMD occupancy
    """
    return textwrap.dedent("""\
    # AMD Bridge — Triton GEMM (MI300X-optimized, MFMA tiling)
    # C[M,N] = A[M,K] @ B[K,N]  (FP16 input, FP32 accumulate, FP16 output)
    import triton
    import triton.language as tl
    import torch

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
                           'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,
                           'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32,
                           'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64,
                           'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
                           'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        # L2-friendly program ID remapping (grouped ordering)
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Tile pointers
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # FP32 accumulator (maps to VGPRs, MFMA accumulates in FP32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k_start < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k_start < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)  # Compiles to MFMA on AMD CDNA
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # Store as FP16
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.float16), mask=mask)

    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
        return c
    """)


def generate_triton_flash_attention_amd() -> str:
    """Generate AMD-optimized Triton FlashAttention kernel.

    Implements FlashAttention-2 algorithm with AMD MI300X optimizations:
    - Tile sizes aligned to MFMA 32x32 or 16x16 instruction dimensions
    - Exploits 192GB HBM3 unified memory for larger sequence support
    - Software pipelining with 2 stages for MFMA latency hiding
    """
    return textwrap.dedent("""\
    # AMD Bridge — Triton FlashAttention-2 (MI300X-optimized)
    # Standard attention: O = softmax(Q @ K^T / sqrt(d)) @ V
    # FlashAttention: tiled, memory-efficient, O(N) memory
    import triton
    import triton.language as tl
    import torch
    import math

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        ],
        key=['N_CTX', 'HEAD_DIM'],
    )
    @triton.jit
    def flash_attn_fwd_kernel(
        Q, K, V, Out,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        N_CTX,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        num_heads = stride_qb // stride_qh  # infer from strides

        batch_idx = pid_bh // num_heads
        head_idx = pid_bh % num_heads

        # Offsets for this block of queries
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        # Base pointers for this batch/head
        q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
        k_base = K + batch_idx * stride_kb + head_idx * stride_kh
        v_base = V + batch_idx * stride_vb + head_idx * stride_vh
        o_base = Out + batch_idx * stride_ob + head_idx * stride_oh

        # Load Q block [BLOCK_M, HEAD_DIM]
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q_mask = offs_m[:, None] < N_CTX
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # row max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # row sum
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)      # output acc

        # Iterate over K/V blocks
        for start_n in range(0, N_CTX, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Load K block [BLOCK_N, HEAD_DIM]
            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_mask = offs_n[:, None] < N_CTX
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # QK^T [BLOCK_M, BLOCK_N] — compiles to MFMA on AMD
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(offs_m[:, None] >= 0, qk, float('-inf'))
            # Causal mask (optional): qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))

            # Online softmax
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)

            # Rescale accumulator
            acc = acc * alpha[:, None]

            # Load V block [BLOCK_N, HEAD_DIM]
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

            # P @ V — compiles to MFMA
            p = tl.exp(qk - m_ij[:, None]) * beta[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            m_i = m_new
            l_i = l_new

        # Final normalization
        acc = acc / l_i[:, None]

        # Store output
        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)

    def flash_attention(q, k, v, causal=False):
        B, H, N, D = q.shape
        sm_scale = 1.0 / math.sqrt(D)
        o = torch.empty_like(q)

        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), B * H)
        flash_attn_fwd_kernel[grid](
            q, k, v, o, sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            N, HEAD_DIM=D,
        )
        return o
    """)
