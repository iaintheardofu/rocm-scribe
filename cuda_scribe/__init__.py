"""
cuda-scribe -- Automated CUDA-to-AMD GPU kernel translation.

Translates CUDA kernels to run on AMD hardware through BridgeIR,
a universal intermediate representation for GPU computation.

Two translation paths:
  1. CUDA -> Triton (portable: runs on NVIDIA + AMD)
  2. CUDA -> HIP C++ (AMD native via hipcc)

Quick start:
    from cuda_scribe import AMDBridge, hipify

    bridge = AMDBridge()
    result = bridge.translate(cuda_source)

    hip_source = hipify(cuda_source)
"""
__version__ = "1.0.1"

from cuda_scribe.amd_bridge import AMDBridge, AMDTranslationResult, TranslationPath
from cuda_scribe.hipify import hipify, HipifyConfig, HipifyResult, get_translation_coverage
from cuda_scribe.bridge_ir import (
    BridgeKernel,
    CUDALifter,
    DType,
    HardwareRequirement,
    ParallelPattern,
    translate_universal,
)
from cuda_scribe.backends.amd_wavefront_optimizer import (
    AMDArch,
    AMD_SPECS,
    compute_occupancy,
    roofline_analysis,
    optimize_block_size,
    apply_lds_bank_conflict_padding,
    convert_warp_to_wavefront,
    generate_triton_amd_config,
)
from cuda_scribe.backends.amd_kernel_optimizer import (
    optimize_hip_kernel,
    OptimizationLevel,
    generate_triton_autotune_configs,
    generate_triton_softmax_amd,
    generate_triton_layernorm_amd,
    generate_triton_rope_amd,
    generate_triton_gemm_amd,
    generate_triton_flash_attention_amd,
    MFMA_INSTRUCTIONS,
    VGPR_OCCUPANCY_TABLE,
    lookup_vgpr_occupancy,
    select_mfma_instruction,
)

__all__ = [
    # Core
    "AMDBridge",
    "AMDTranslationResult",
    "TranslationPath",
    # HIPIFY
    "hipify",
    "HipifyConfig",
    "HipifyResult",
    "get_translation_coverage",
    # BridgeIR
    "BridgeKernel",
    "CUDALifter",
    "DType",
    "HardwareRequirement",
    "ParallelPattern",
    "translate_universal",
    # AMD Hardware
    "AMDArch",
    "AMD_SPECS",
    "compute_occupancy",
    "roofline_analysis",
    "optimize_block_size",
    "apply_lds_bank_conflict_padding",
    "convert_warp_to_wavefront",
    "generate_triton_amd_config",
    # Kernel Optimizer
    "optimize_hip_kernel",
    "OptimizationLevel",
    "generate_triton_autotune_configs",
    "generate_triton_softmax_amd",
    "generate_triton_layernorm_amd",
    "generate_triton_rope_amd",
    "generate_triton_gemm_amd",
    "generate_triton_flash_attention_amd",
    "MFMA_INSTRUCTIONS",
    "VGPR_OCCUPANCY_TABLE",
    "lookup_vgpr_occupancy",
    "select_mfma_instruction",
]
