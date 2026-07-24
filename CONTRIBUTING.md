# Contributing to rocm-scribe

Thank you for your interest in contributing to rocm-scribe.

## Getting Started

```bash
git clone https://github.com/iaintheardofu/rocm-scribe.git
cd rocm-scribe
pip install -e ".[dev]"
pytest
```

## What We Need

### High Impact
- **CUDA pattern templates** — Add new CUDA-to-Triton translation patterns to `cuda_scribe/patterns/`. Each pattern needs a CUDA input, expected Triton output, and test case.
- **HIP API mappings** — Expand coverage in `cuda_scribe/hipify.py`. We track coverage gaps in issues.
- **Test kernels** — Complex real-world CUDA kernels that stress the translation pipeline. Add to `examples/`.

### Backend Improvements
- Metal shader generation accuracy
- SYCL 2020 feature coverage
- WGSL compute shader patterns
- Pallas/JAX integration testing

### Documentation
- Translation guides for common CUDA patterns
- AMD hardware optimization tips
- Migration case studies

## Development

### Running Tests
```bash
pytest                          # All tests
pytest tests/test_hipify.py     # Specific module
pytest -k "test_wavefront"     # Pattern match
```

### Code Style
We use `ruff` for formatting and linting:
```bash
ruff check .
ruff format .
```

### Commit Messages
Use conventional commits:
- `feat: add cuSPARSE-to-rocSPARSE mappings`
- `fix: correct wavefront reduction step count`
- `test: add FlashAttention CUDA kernel`
- `docs: add MI300X optimization guide`

## Pull Request Process

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Ensure all tests pass
4. Submit PR with clear description of changes

## Code of Conduct

Be respectful. We're here to make GPU computing accessible to everyone.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
