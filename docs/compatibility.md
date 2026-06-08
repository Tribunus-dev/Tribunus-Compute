# MLX Compatibility Tuple

This document records the pinned dependency versions and fork identity for the
Tribunus Compute Kernel. The kernel is not version-pinned to upstream MLX
releases — it targets a specific fork that backports patches and stabilises the
ABI for production deployment.

## Current Tuple (v0.1.0)

| Layer | Version | Notes |
|-------|---------|-------|
| `tribunus-mlx-rs` | `0.25.3-tribunus.1` | Fork of upstream `mlx-rs` 0.25.3 at commit `93ed8db` |
| `tribunus-mlx-sys` | `0.6.0-tribunus.1` | Bindings generated from MLX C 0.6.0 |
| MLX C | `0.6.0` (patched) | FFTNorm, quantization mode fixes |
| MLX Core | `v0.31.2` | Confirmed from `mlx/version.h` |
| Rust MSRV | `1.82.0` | Required by `mlx-rs` job queue API |
| napi-rs | `3.9.0` | N-API v8, serde-json feature |

## Fork Location

The MLX fork lives at `mlx-rs-fork/` (submodule):

```
url = https://github.com/oxideai/mlx-rs.git
branch = compat/mlx-core-0.31.2
```

The fork contains no model-specific code — only compatibility shims and
patches required to build against MLX Core 0.31.2 / MLX C 0.6.0.

## Fork Commit History

1. Update vendored MLX C to v0.6.0, pin MLX Core to v0.31.2
2. Version bump: `mlx-rs` 0.25.3-tribunus.1, `mlx-sys` 0.6.0-tribunus.1
3. Fix FFT API for MLX Core 0.31.2: add `FFTNorm::Backward`
4. Fix `mlx-sys` version pin to 0.6.0-tribunus.1
5. Fix `mlx-rs` wrapper for MLX C 0.6.0: quantization API changes
6. Fix SIGSEGV: pass non-null mode string to quantize ops
7. Fix quantization mode: use `'affine'` string

## Legacy Baselines

| Branch / Tag | mlx-rs | MLX Core | Notes |
|--------------|--------|----------|-------|
| `legacy-mlx-021-baseline` | 0.21.2 | 0.21.0 | First kernel prototype |
| `mlx-core-025-compat` | 0.25.3 | 0.25.1 | Intermediate fork gate |
| `compat/mlx-core-0.31.2` | 0.25.3-tribunus.1 | 0.31.2 | **Current** |

## Forked Cargo Workspace

The MLX fork is a Rust workspace with these crates:

| Crate | Role |
|-------|------|
| `mlx-sys` | Low-level C bindings (bindgen-generated) |
| `mlx-rs` | Safe Rust wrapper |
| `mlx-macros` | Proc-macro helpers |
| `mlx-internal-macros` | Internal proc-macros |
| `mlx-lm` | Language-model utilities (unused by kernel) |
| `mlx-lm-utils` | LM helper crate (unused by kernel) |

## Compatibility Guarantee

Within the same kernel major version, the MLX dependency tuple is stable and
will not change without a minor- or major-version bump of the kernel itself.
Minor MLX C patches (0.6.x) may be pulled in as patch releases of the kernel
when they fix bugs without changing the ABI.
