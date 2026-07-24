#!/usr/bin/env python3
"""
AMD Wavefront Optimizer — Hardware-aware kernel optimization for AMD GPUs.

Applies Algorithmica HPC performance engineering principles to GPU kernels:
  - §3.3  Branchless programming → wavefront divergence elimination
  - §8.1  Memory hierarchy → registers → LDS → L1 → L2 → HBM optimization
  - §8.7  Cache-oblivious algorithms → tiling for LDS
  - §9.1  Memory bandwidth → coalesced access patterns for 64-wide wavefronts
  - §9.11 AoS vs SoA → structure of arrays for GPU memory coalescing
  - §10.3 SIMD reductions → wavefront-wide shuffle reduction
  - §11.15 Matrix multiplication → tiled GEMM with MFMA instructions

AMD Architecture Key Parameters:
  - MI300X (CDNA 3): 304 CUs, 64KB LDS/CU, 5.3 TB/s HBM3, 1307 TFLOPS FP8
  - MI250X (CDNA 2): 220 CUs, 64KB LDS/CU, 3.2 TB/s HBM2e, 383 TFLOPS FP16
  - RX 7900 XTX (RDNA 3): 96 CUs, 128KB LDS/WGP, 960 GB/s GDDR6X
  - Wavefront width: 64 threads (CDNA), 32 or 64 (RDNA 3 wave32/wave64)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class AMDArch(Enum):
    """AMD GPU architecture targets."""
    CDNA3 = "cdna3"    # MI300X
    CDNA2 = "cdna2"    # MI250X
    CDNA1 = "cdna1"    # MI100
    RDNA3 = "rdna3"    # RX 7900 XTX
    RDNA2 = "rdna2"    # RX 6900 XT


@dataclass
class AMDHardwareSpec:
    """Hardware specifications for AMD GPU targets."""
    arch: AMDArch
    compute_units: int
    lds_per_cu_kb: int             # Local Data Share per CU in KB
    wavefront_width: int           # 64 for CDNA, 32 or 64 for RDNA
    max_waves_per_cu: int          # Max wavefronts per CU
    hbm_bandwidth_tbs: float       # TB/s
    vgpr_count: int                # Vector GPRs per SIMD unit
    sgpr_count: int                # Scalar GPRs per SIMD unit
    has_mfma: bool                 # Matrix Fused Multiply-Add
    lds_banks: int = 32            # LDS bank count
    l2_cache_mb: int = 0           # L2 cache in MB
    peak_tflops_fp16: float = 0.0  # Peak FP16 TFLOPS


# Known AMD GPU specifications
AMD_SPECS: Dict[AMDArch, AMDHardwareSpec] = {
    AMDArch.CDNA3: AMDHardwareSpec(
        arch=AMDArch.CDNA3, compute_units=304, lds_per_cu_kb=64,
        wavefront_width=64, max_waves_per_cu=32,
        hbm_bandwidth_tbs=5.3, vgpr_count=512, sgpr_count=106,
        has_mfma=True, l2_cache_mb=256, peak_tflops_fp16=1307.4,
    ),
    AMDArch.CDNA2: AMDHardwareSpec(
        arch=AMDArch.CDNA2, compute_units=220, lds_per_cu_kb=64,
        wavefront_width=64, max_waves_per_cu=32,
        hbm_bandwidth_tbs=3.2, vgpr_count=512, sgpr_count=106,
        has_mfma=True, l2_cache_mb=32, peak_tflops_fp16=383.0,
    ),
    AMDArch.CDNA1: AMDHardwareSpec(
        arch=AMDArch.CDNA1, compute_units=120, lds_per_cu_kb=64,
        wavefront_width=64, max_waves_per_cu=32,
        hbm_bandwidth_tbs=1.228, vgpr_count=256, sgpr_count=106,
        has_mfma=True, l2_cache_mb=16, peak_tflops_fp16=184.6,
    ),
    AMDArch.RDNA3: AMDHardwareSpec(
        arch=AMDArch.RDNA3, compute_units=96, lds_per_cu_kb=128,
        wavefront_width=32, max_waves_per_cu=16,  # wave32 mode
        hbm_bandwidth_tbs=0.96, vgpr_count=384, sgpr_count=106,
        has_mfma=False, l2_cache_mb=96, peak_tflops_fp16=123.0,
    ),
    AMDArch.RDNA2: AMDHardwareSpec(
        arch=AMDArch.RDNA2, compute_units=80, lds_per_cu_kb=128,
        wavefront_width=32, max_waves_per_cu=16,
        hbm_bandwidth_tbs=0.512, vgpr_count=256, sgpr_count=106,
        has_mfma=False, l2_cache_mb=128, peak_tflops_fp16=46.1,
    ),
}


@dataclass
class OccupancyAnalysis:
    """GPU occupancy analysis for a kernel."""
    waves_per_cu: int
    occupancy_pct: float
    limiting_factor: str       # "vgpr", "sgpr", "lds", "wavefront_slots"
    lds_per_wave_bytes: int
    vgpr_per_thread: int
    recommendations: List[str] = field(default_factory=list)


def compute_occupancy(
    vgpr_per_thread: int,
    lds_bytes: int,
    block_size: int,
    arch: AMDArch = AMDArch.CDNA3,
) -> OccupancyAnalysis:
    """Compute GPU occupancy for an AMD target.

    Algorithmica §8.1: Memory hierarchy — occupancy determines how well
    we can hide memory latency via wavefront switching.
    """
    spec = AMD_SPECS[arch]
    wavefront_width = spec.wavefront_width
    waves_per_block = block_size // wavefront_width

    # VGPR limit: VGPRs are allocated per SIMD unit
    # Each CU has 4 SIMD units (CDNA), each with vgpr_count VGPRs
    # AMD VGPRs are per-wavefront (each VGPR = wavefront_width lanes)
    # So vgpr_per_thread IS vgpr_per_wave in AMD terminology
    # VGPRs allocated in granularity of 8 (wave64)
    vgpr_alloc = ((vgpr_per_thread + 7) // 8) * 8
    max_waves_per_simd = spec.vgpr_count // max(vgpr_alloc, 1)
    max_waves_vgpr = max_waves_per_simd * 4  # 4 SIMDs per CU

    # LDS limit
    lds_total = spec.lds_per_cu_kb * 1024
    if lds_bytes > 0:
        max_waves_lds = (lds_total // lds_bytes) * waves_per_block
    else:
        max_waves_lds = spec.max_waves_per_cu

    # Wavefront slot limit
    max_waves_slots = spec.max_waves_per_cu

    # Actual occupancy
    waves = min(max_waves_vgpr, max_waves_lds, max_waves_slots)
    occupancy = waves / spec.max_waves_per_cu * 100

    # Determine limiting factor
    if waves == max_waves_vgpr:
        limiting = "vgpr"
    elif waves == max_waves_lds:
        limiting = "lds"
    else:
        limiting = "wavefront_slots"

    # Generate recommendations
    recs = []
    if limiting == "vgpr" and vgpr_per_thread > 64:
        recs.append(
            f"Reduce VGPR usage (currently {vgpr_per_thread}/thread). "
            f"Consider register spilling or reducing live variables."
        )
    if limiting == "lds" and lds_bytes > lds_total // 2:
        recs.append(
            f"LDS usage ({lds_bytes}B) limits occupancy. "
            f"Consider smaller tiles or multi-pass approach."
        )
    if occupancy < 50:
        recs.append(
            f"Low occupancy ({occupancy:.0f}%). "
            f"Increase parallelism or reduce resource usage."
        )
    if block_size % wavefront_width != 0:
        recs.append(
            f"Block size ({block_size}) not aligned to wavefront width "
            f"({wavefront_width}). Wasting threads."
        )

    return OccupancyAnalysis(
        waves_per_cu=waves,
        occupancy_pct=occupancy,
        limiting_factor=limiting,
        lds_per_wave_bytes=lds_bytes,
        vgpr_per_thread=vgpr_per_thread,
        recommendations=recs,
    )


def roofline_analysis(
    flops_per_element: int,
    bytes_per_element: int,
    arch: AMDArch = AMDArch.CDNA3,
) -> Dict[str, float]:
    """Compute roofline model parameters for an AMD GPU.

    Algorithmica §9.1: Memory bandwidth is the primary bottleneck for
    most GPU kernels. The arithmetic intensity determines whether a
    kernel is compute-bound or memory-bound.
    """
    spec = AMD_SPECS[arch]

    # Arithmetic intensity (FLOP/byte)
    ai = flops_per_element / max(bytes_per_element, 1)

    # Ridge point: where compute meets bandwidth
    peak_tflops = spec.peak_tflops_fp16
    bandwidth_tbs = spec.hbm_bandwidth_tbs
    ridge_point = peak_tflops / bandwidth_tbs  # FLOP/byte at ridge

    # Attainable performance
    attainable_tflops = min(peak_tflops, ai * bandwidth_tbs)
    bound = "compute" if ai >= ridge_point else "memory"

    return {
        "arithmetic_intensity": ai,
        "ridge_point": ridge_point,
        "attainable_tflops": attainable_tflops,
        "peak_tflops": peak_tflops,
        "bandwidth_tbs": bandwidth_tbs,
        "bound": bound,
        "efficiency_pct": (attainable_tflops / peak_tflops) * 100,
    }


def optimize_block_size(
    total_elements: int,
    arch: AMDArch = AMDArch.CDNA3,
    lds_bytes: int = 0,
    vgpr_per_thread: int = 32,
) -> int:
    """Choose optimal block size for AMD GPU.

    Algorithmica §10.6: Auto-vectorization — match the hardware vector width.
    For AMD CDNA: multiples of 64 (wavefront width).
    For AMD RDNA3: multiples of 32 (wave32 mode) or 64 (wave64 mode).
    """
    spec = AMD_SPECS[arch]
    wavefront = spec.wavefront_width

    # Try block sizes from 64 to 1024
    best_size = 256
    best_occupancy = 0.0

    for bs in range(wavefront, 1025, wavefront):
        occ = compute_occupancy(vgpr_per_thread, lds_bytes, bs, arch)
        if occ.occupancy_pct > best_occupancy:
            best_occupancy = occ.occupancy_pct
            best_size = bs

    return best_size


def apply_lds_bank_conflict_padding(source: str) -> str:
    """Add LDS bank conflict padding to shared memory arrays.

    AMD has 32 LDS banks. With 64-wide wavefronts, 2 threads per bank
    access simultaneously. Adding +1 padding eliminates bank conflicts
    for stride-1 access patterns.

    Algorithmica §9.3: Cache line optimization — avoid false sharing.
    """
    return re.sub(
        r'__shared__\s+(\w+)\s+(\w+)\[(\d+)\]',
        lambda m: (
            f'__shared__ {m.group(1)} {m.group(2)}'
            f'[{m.group(3)} + 1]  // AMD LDS bank conflict padding'
        ),
        source
    )


def convert_warp_to_wavefront(source: str) -> str:
    """Convert NVIDIA warp-specific code to AMD wavefront-portable code.

    Key changes:
    - Remove mask parameter from sync intrinsics
    - Replace hardcoded warpSize=32 with runtime warpSize
    - Adjust reduction loop bounds

    Algorithmica §10.3: SIMD reductions must match hardware vector width.
    """
    # Remove mask from __shfl_*_sync
    warp_fns = {
        "__shfl_down_sync": "__shfl_down",
        "__shfl_up_sync": "__shfl_up",
        "__shfl_xor_sync": "__shfl_xor",
        "__shfl_sync": "__shfl",
        "__ballot_sync": "__ballot",
        "__any_sync": "__any",
        "__all_sync": "__all",
    }

    for cuda_fn, hip_fn in warp_fns.items():
        # Remove the mask argument (first arg)
        source = re.sub(
            rf'{re.escape(cuda_fn)}\s*\(\s*(?:0x[0-9a-fA-F]+|~0u?|0xFFFFFFFF[uU]?|\w+)\s*,\s*',
            f'{hip_fn}(',
            source
        )

    # Replace hardcoded 32 in warp-related contexts
    source = re.sub(r'\bwarpSize\s*==\s*32\b', 'true /* warpSize check */', source)
    source = re.sub(
        r'(?<!\d)32(?!\d)(?=\s*//\s*(?:warp|WARP))',
        'warpSize',
        source
    )

    return source


def generate_triton_amd_config(kernel_pattern: str) -> Dict[str, any]:
    """Generate Triton autotuning configuration optimized for AMD GPUs.

    Triton on AMD uses the ROCm backend. Key differences from NVIDIA:
    - Wavefront width: 64 (affects BLOCK_SIZE choices)
    - LDS: 64KB (affects tile sizes)
    - MFMA instructions for matrix ops (CDNA only)
    - Different L2 cache sizes affect tile reuse

    Algorithmica §11.15: Matrix multiplication — tile size selection
    determines the balance between compute and memory access.
    """
    if kernel_pattern == "matmul":
        return {
            "BLOCK_M": [64, 128, 256],   # AMD prefers larger M tiles
            "BLOCK_N": [64, 128, 256],
            "BLOCK_K": [16, 32, 64],     # K=16 aligns with MFMA
            "num_warps": [4, 8],         # 4 warps = 256 threads (4x64)
            "num_stages": [2, 3, 4],
            "waves_per_eu": [1, 2],      # AMD-specific: waves per execution unit
        }
    elif kernel_pattern == "reduce":
        return {
            "BLOCK_SIZE": [256, 512, 1024],
            "num_warps": [4, 8, 16],
            "REDUCTION_DIM": [0, 1],
        }
    elif kernel_pattern == "attention":
        return {
            "BLOCK_M": [64, 128],
            "BLOCK_N": [64, 128],
            "BLOCK_DHEAD": [32, 64, 128],
            "num_warps": [4, 8],
            "num_stages": [1, 2],
        }
    else:  # elementwise / map
        return {
            "BLOCK_SIZE": [256, 512, 1024, 2048],
            "num_warps": [4, 8],
        }
