#!/bin/bash
# Sync open-source files from workforce repo into cuda-scribe.
# Run this to pull latest changes from the internal codebase.
# Does NOT copy proprietary stages (evolutionary repair, pattern learning, LLM translation).

set -euo pipefail

SRC="/Users/michaelpendleton/workforce/integrations/cuda_scribe"
DST="/Users/michaelpendleton/cuda-scribe/cuda_scribe"
TESTS_SRC="/Users/michaelpendleton/workforce/tests"
TESTS_DST="/Users/michaelpendleton/cuda-scribe/tests"
EXAMPLES_SRC="$SRC/examples"
EXAMPLES_DST="/Users/michaelpendleton/cuda-scribe/examples"

echo "=== Syncing open-source files from workforce ==="

# Core translation files (open-source)
cp "$SRC/bridge_ir.py" "$DST/bridge_ir.py"
cp "$SRC/hipify.py" "$DST/hipify.py"
cp "$SRC/amd_bridge.py" "$DST/amd_bridge.py"

# Backends (open-source)
cp "$SRC/backends/hip_backend.py" "$DST/backends/hip_backend.py"
cp "$SRC/backends/amd_wavefront_optimizer.py" "$DST/backends/amd_wavefront_optimizer.py"
cp "$SRC/backends/amd_kernel_optimizer.py" "$DST/backends/amd_kernel_optimizer.py"
cp "$SRC/backends/metal_backend.py" "$DST/backends/metal_backend.py"
cp "$SRC/backends/pallas_backend.py" "$DST/backends/pallas_backend.py"
cp "$SRC/backends/sycl_backend.py" "$DST/backends/sycl_backend.py"
cp "$SRC/backends/wgsl_backend.py" "$DST/backends/wgsl_backend.py"

# Examples
cp "$EXAMPLES_SRC"/*.cu "$EXAMPLES_DST/"

# Tests (open-source subset)
cp "$TESTS_SRC/test_amd_bridge.py" "$TESTS_DST/test_amd_bridge.py"
cp "$TESTS_SRC/test_bridge_v2.py" "$TESTS_DST/test_bridge_v2.py"

# Fix imports: integrations.cuda_scribe -> cuda_scribe
find "$DST" "$TESTS_DST" -name "*.py" -exec sed -i '' \
    -e 's/from integrations\.cuda_scribe/from cuda_scribe/g' \
    -e 's/import integrations\.cuda_scribe/import cuda_scribe/g' \
    {} +

# Remove workforce-specific paths
find "$DST" -name "*.py" -exec sed -i '' \
    -e '/WORKFORCE_DIR/d' \
    -e '/RUNTIME_DIR.*=.*WORKFORCE_DIR/d' \
    -e '/STATE_DIR.*=.*RUNTIME_DIR/d' \
    -e '/AUDIT_LOG.*=.*STATE_DIR/d' \
    {} +

echo "=== Sync complete ==="
echo "Files synced. Review changes with: git diff"
echo ""
echo "NOT synced (proprietary):"
echo "  - connector.py (full 7-stage pipeline with evolutionary repair)"
echo "  - quantization_patterns.py (W4A16, W8A8, FP8, SmoothQuant)"
echo "  - venus_harness.py (internal GPU server)"
echo "  - benchmark_l40s.py (internal benchmark harness)"
