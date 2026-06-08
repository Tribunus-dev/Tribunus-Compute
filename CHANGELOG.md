# Changelog

All notable changes to the Tribunus Compute Kernel are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with the `compute-vX.Y.Z` scheme (see [RELEASING.md](docs/RELEASING.md)).

## [Unreleased]

### Current State

The kernel is at v0.1.0 — the monorepo extraction is complete and the
SharedTensorArena phases 0–4 (IOSurface-backed FP16 storage, MLX external
arrays, Core ML IOSurface input, Core ML output backings, verified
zero-application-copy round trips) are verified. Phases 5–10 infrastructure
(Tokio supervisor, arena pool, capability reports, hybrid profile schema,
structured errors, receipts) is implemented. The kernel compiles and passes
38 tests across 9 modules.

### In Progress

- **Phase E: Structural Tests** — synthetic plan validation, malformed plan
  rejection, segment corruption gate.
- **KV-cache integration** — prefill + cached decode with parity verification.
- **Mapped no-copy segment residency** — mmap segments with external MLX
  arrays over mapped memory.
- **Core ML stateful prediction qualification** — bridge compiles but is not
  runtime-qualified; stateful prediction crashes on `coreml_state.mm` stubs.

### Known Gaps

- Deferred to v1.x / v2: additional dtypes beyond FP16, cross-process IOSurface
  transport, full Gemma Core ML islands, ANE autotuning, video-frame formats,
  multiple concurrent readers per arena, arbitrary strided views.

---

## [compute-v0.1.0] — 2026-06-08

### Added

- **Monorepo extraction** — Tribunus Compute Kernel separated from the parent
  Tribunus monorepo into its own workspace. The kernel becomes the single
  native MLX backend for Tribunus.
- **compute-native crate** (`tribunus-compute-native`) — napi-rs based
  `cdylib` + `rlib` targeting `aarch64-apple-darwin`. Provides `run_full_model_from_image`
  N-API binding and a `tribunus-compute-worker` binary.
- **mlx-rs fork** (`mlx-rs-fork/`) — Tribunus compatibility fork at
  `tribunus-mlx-rs` 0.25.3-tribunus.1 pinned to MLX Core v0.31.2.
- **Patched MLX C 0.6.0** — vendored bindings with FFTNorm::Backward fix and
  quantization mode (`affine` string) patch for MLX Core 0.31.2 compatibility.
- **SharedTensorArena v1** (phases 0–4) — IOSurface-backed FP16 contiguous
  storage with `ArenaCreationReceipt`. Verified zero-application-copy
  MLX → Core ML → MLX round trip through five identity-model tests.
- **Arena pool** — bounded per-key pooling with acquire, release, reuse,
  and budget enforcement. 4 tests.
- **Arena lifecycle** — deterministic state machine (`Uninitialized → Allocated
  → BoundToIOSurface → MappedByMLX → OwnedByCoreML → Released → Recycled`)
  with illegal-transition rejection. 3 tests.
- **MLX external array bridge** — `ExternalArrayHandle` wrapping
  `mlx_array_new_data_managed` for MLX to consume IOSurface-backed memory.
- **Core ML bridge** — stateless prediction through `coreml_exec.mm`
  (ObjC++). Supports IOSurface input via `MLMultiArray(pixelBuffer:shape:)`
  and `outputBackings`.
- **Core ML stateful bridge** — `coreml_state.mm` compiles but is not
  runtime-qualified (stub methods pending Phase 12).
- **Plan-driven execution engine** — `prologue`, `layer`, `epilogue` executor
  with `ModelExecutionPlan`, `LayerPlan`, `build_execution_plan()`, and
  `validate()` (~1250 lines across `executor.rs` and `config.rs`).
- **ComputeImage compiler** — `ImageRuntime::run_full_model()` with plan
  embedding (~2800 lines in `compute_image.rs`).
- **Capability report** — 17 frozen capability names (iosurface_creation,
  fp16_pixelbuffer_multiarrays, mlx_iosurface_external_array,
  mlx_coreml_round_trip, hybrid_compute_image, etc.) with
  `CAP_*` constants, auto-detection, and serde round-trip. 2 tests.
- **Hybrid profile** — `HybridDeploymentProfile` schema with tensor flow
  validation for MLX / Core ML hybrid execution. 3 tests.
- **Frozen receipt schemas** — `ArenaCreationReceipt`, `CoreMlPredictionReceipt`,
  `HybridJobReceipt` with copy classification
  (`application_copy_free`, `copied_fallback`,
  `materialized_layout_conversion`, `internal_coreml_staging_unknown`).
  Structurally typed with stable field names. 3 receipt tests.
- **Frozen ABIs** — three boundary contracts documented:
  `tribunus-iosurface-fp16-arena-v1`,
  `tribunus-coreml-stateful-island-v1`,
  `tribunus-hybrid-compute-image-v1`.
- **Tokio supervisor** — job lifecycle, worker spawning, cancellation, and
  graceful shutdown. 9 tests.
- **KV-cache** — sliding window and global eviction with concurrent access.
  6 tests. Not yet runtime-qualified.
- **Structured errors** — typed error variants with builder pattern and
  Display impl. 3 tests.
- **Build infrastructure** — `build.rs` with Core ML / IOSurface / CoreVideo
  framework linking, `napi-build` setup, and `-undefined,dynamic_lookup`
  for macOS test harness compatibility.
- **Release profile** — LTO, symbol stripping, single codegen unit for
  minimal binary size on `aarch64-apple-darwin`.

### Changed

- MLX C bindings updated from mlx-core 0.21.x (legacy baseline) to
  v0.6.0 → Core 0.31.2.
- MLX quantization API calls updated for MLX C 0.6.0 (non-null mode string,
  `affine` quantization mode).
- FFT API updated for MLX Core 0.31.2 (added `FFTNorm::Backward`).

### Fixed

- SIGSEGV in quantized matmul: pass non-null mode string to quantize ops.
- Quantization mode string: use `'affine'` instead of null/default.
- `mlx-sys` version pin enforced to `0.6.0-tribunus.1`.

### Qualified

- `supports_quantized_matmul`: true
- `supports_dequantize`: true
- `supports_memory_telemetry`: true (mlx-sys FFI)
- `supports_cache_control`: true
- `supports_multithreaded_execution`: true (4×50 heavy matmul qualified)
- `metal_available`: true, `accelerate_available`: true
- `supports_external_array`: false (no-copy not in MLX C API)
- 4 compute_image fixture tests pass: handles invariant (4→29→4),
  active memory bounded (72128→71288), quantized matmul parity,
  concurrent 2×100 + 4×50 heavy quantized matmul (no SIGSEGV).

### Removed

- Legacy MLX 0.21.x baseline (preserved as git branch
  `legacy-mlx-021-baseline` for reference).
- MLX Core 0.25.x compat baseline (preserved as `mlx-core-025-compat`).

[Unreleased]: https://github.com/Tribunus-dev/Tribunus-Compute/compare/compute-v0.1.0...HEAD
[compute-v0.1.0]: https://github.com/Tribunus-dev/Tribunus-Compute/releases/tag/compute-v0.1.0
