# Tribunus Compute Kernel

Core 0.31.2 compatibility fork gate.

## Dependency Tuple

| Layer | Version | Notes |
|-------|---------|-------|
| tribunus-mlx-rs | 0.25.3-tribunus.1 | Fork of upstream mlx-rs 0.25.3 (93ed8db) |
| tribunus-mlx-sys | 0.6.0-tribunus.1 | Bindings generated from MLX C 0.6.0 |
| MLX C | 0.6.0 (patched) | FFTNorm, quantization mode fixes |
| MLX Core | v0.31.2 | Confirmed from mlx/version.h |

## Structure

- `mlx-rs-fork/` — Tribunus compatibility fork (no model-specific code)
- `compute-native/` — Tribunus Compute Kernel (Gemma, ComputeImage, sessions)

## Fork commits

Branch: `compat/mlx-core-0.31.2`

1. Update vendored MLX C to v0.6.0, pin MLX Core to v0.31.2
2. Version bump: mlx-rs 0.25.3-tribunus.1, mlx-sys 0.6.0-tribunus.1
3. Fix FFT API for MLX Core 0.31.2: add FFTNorm::Backward
4. Fix mlx-sys version pin to 0.6.0-tribunus.1
5. Fix mlx-rs wrapper for MLX C 0.6.0: quantization API changes
6. Fix SIGSEGV: pass non-null mode string to quantize ops
7. Fix quantization mode: use 'affine' string

## Capability Report

- `supports_quantized_matmul`: true
- `supports_dequantize`: true
- `supports_memory_telemetry`: true (mlx-sys FFI)
- `supports_cache_control`: true
- `supports_multithreaded_execution`: true (qualified: 4×50 heavy matmul)
- `supports_external_array`: false (no-copy not in MLX C API)
- `metal_available`: true
- `accelerate_available`: true

## Test Status

All 4 compute_image fixture tests pass:
- `handles=4→29→4` — residency invariant
- `mlx_active=72128→71288` — active memory bounded
- Quantized matmul parity: fused vs dequantize+matmul
- Concurrent: 2×100 + 4×50 heavy quantized matmul, no SIGSEGV

## Legacy Baselines

- `legacy-mlx-021-baseline`: mlx-rs 0.21.2, Core 0.21.0
- `mlx-core-025-compat`: mlx-rs 0.25.3, Core 0.25.1
