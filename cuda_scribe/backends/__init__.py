"""cuda-scribe backends -- target-specific code generators from BridgeIR."""

# Import all backends so they register with the BridgeIR backend registry.
from cuda_scribe.backends import (  # noqa: F401
    hip_backend,
    metal_backend,
    pallas_backend,
    sycl_backend,
    wgsl_backend,
)
