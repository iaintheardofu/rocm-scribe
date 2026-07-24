#!/usr/bin/env python3
"""
cuda-scribe CLI -- Command-line interface for CUDA-to-AMD translation.

Usage:
    cuda-scribe analyze kernel.cu
    cuda-scribe translate kernel.cu [--backend amd|nvidia|both]
    cuda-scribe hipify kernel.cu [--arch gfx942]
    cuda-scribe optimize kernel.cu [--target mi300x]
    cuda-scribe info
"""
import argparse
import json
import sys
from pathlib import Path


def cmd_info(args):
    """Show cuda-scribe status and capabilities."""
    from cuda_scribe import __version__
    from cuda_scribe.hipify import get_translation_coverage
    from cuda_scribe.backends.amd_wavefront_optimizer import AMD_SPECS

    coverage = get_translation_coverage()
    print(f"cuda-scribe v{__version__}")
    print(f"\nHIPIFY coverage: {coverage.get('total_mappings', 'N/A')} API mappings")
    print(f"\nSupported AMD targets:")
    for arch_name, spec in AMD_SPECS.items():
        print(f"  {arch_name}: {spec.get('gpu', 'N/A')} ({spec.get('gfx', 'N/A')})")


def cmd_analyze(args):
    """Analyze a CUDA kernel for translation complexity."""
    from cuda_scribe.bridge_ir import CUDALifter

    cuda_source = Path(args.file).read_text()
    lifter = CUDALifter()
    kernel = lifter.lift(cuda_source)
    print(json.dumps(kernel.to_dict(), indent=2, default=str))


def cmd_translate(args):
    """Translate CUDA kernel to Triton (AMD + NVIDIA)."""
    from cuda_scribe import AMDBridge

    bridge = AMDBridge()
    cuda_source = Path(args.file).read_text()
    result = bridge.translate(cuda_source)

    output_file = args.output or args.file.replace(".cu", "_triton.py")
    if result.triton_source:
        Path(output_file).write_text(result.triton_source)
        print(f"Triton kernel written to: {output_file}")
    if result.hip_source:
        hip_file = args.file.replace(".cu", ".hip.cpp")
        Path(hip_file).write_text(result.hip_source)
        print(f"HIP source written to: {hip_file}")
    if result.occupancy:
        print(f"\nOccupancy: {json.dumps(result.occupancy, indent=2, default=str)}")
    if result.roofline:
        print(f"Roofline: {json.dumps(result.roofline, indent=2, default=str)}")


def cmd_hipify(args):
    """HIPIFY a CUDA source file (API-level translation)."""
    from cuda_scribe import hipify, HipifyConfig

    cuda_source = Path(args.file).read_text()
    config = HipifyConfig(target_arch=args.arch) if args.arch else HipifyConfig()
    result = hipify(cuda_source, config=config)

    output_file = args.output or args.file.replace(".cu", ".hip.cpp")
    if hasattr(result, "source"):
        Path(output_file).write_text(result.source)
        print(f"HIP source written to: {output_file}")
        if hasattr(result, "diagnostics") and result.diagnostics:
            print(f"\nDiagnostics ({len(result.diagnostics)}):")
            for d in result.diagnostics[:10]:
                print(f"  [{d.level.value}] {d.message}")
    else:
        Path(output_file).write_text(result)
        print(f"HIP source written to: {output_file}")


def cmd_optimize(args):
    """Get AMD optimization recommendations for a kernel."""
    from cuda_scribe.backends.amd_wavefront_optimizer import (
        roofline_analysis,
        optimize_block_size,
        compute_occupancy,
    )

    target_map = {
        "mi300x": "cdna3",
        "mi250x": "cdna2",
        "mi100": "cdna1",
        "rx7900": "rdna3",
    }
    arch = target_map.get(args.target.lower(), "cdna3")

    block_rec = optimize_block_size(arch)
    print(f"Target: {args.target} ({arch})")
    print(f"\nRecommended block sizes: {json.dumps(block_rec, indent=2, default=str)}")


def main():
    parser = argparse.ArgumentParser(
        prog="cuda-scribe",
        description="Automated CUDA-to-AMD GPU kernel translation",
    )
    subs = parser.add_subparsers(dest="command")

    subs.add_parser("info", help="Show engine status and capabilities")

    p_analyze = subs.add_parser("analyze", help="Analyze CUDA kernel complexity")
    p_analyze.add_argument("file", help="CUDA source file (.cu)")

    p_translate = subs.add_parser("translate", help="Translate CUDA to Triton/HIP")
    p_translate.add_argument("file", help="CUDA source file (.cu)")
    p_translate.add_argument("--backend", default="both", choices=["nvidia", "amd", "both"])
    p_translate.add_argument("--output", help="Output file path")

    p_hipify = subs.add_parser("hipify", help="HIPIFY CUDA source (API-level translation)")
    p_hipify.add_argument("file", help="CUDA source file (.cu)")
    p_hipify.add_argument("--arch", default=None, help="Target arch (e.g., gfx942)")
    p_hipify.add_argument("--output", help="Output file path")

    p_optimize = subs.add_parser("optimize", help="AMD optimization recommendations")
    p_optimize.add_argument("file", help="CUDA source file (.cu)", nargs="?")
    p_optimize.add_argument("--target", default="mi300x",
                            choices=["mi300x", "mi250x", "mi100", "rx7900"])

    args = parser.parse_args()

    commands = {
        "info": cmd_info,
        "analyze": cmd_analyze,
        "translate": cmd_translate,
        "hipify": cmd_hipify,
        "optimize": cmd_optimize,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
