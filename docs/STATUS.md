# Tribunus Compute — Capability Qualification Matrix

**Date:** 2026-06-08
**Version:** 0.1.0
**Build target:** `aarch64-apple-darwin` (Apple Silicon)
**Runtime targets:** Metal GPU, ANE (Core ML), CPU (NEON + Accelerate)

## Legend

| Tier | Meaning |
|---|---|
| ✅ | Confirmed and verifiable |
| ⬜ | Implemented but not tested at this tier |
| ❌ | Not applicable / not supported |
| 🚧 | In progress |

Tiers are cumulative: a capability cannot be Runtime-Qualified unless also Unit-Tested, etc.

---

## Capability Matrix

| Capability | Implemented | Unit-Tested | Runtime-Qualified | Benchmark-Qualified | Production-Supported | Hardware Required | Evidence |
|---|---|---|---|---|---|---|---|
| **MLX primitives** | | | | | | | |
| MLX matmul (f32) | ✅ | ✅ | 🚧 | ⬜ | ❌ | Metal GPU | `lib.rs` napi `matmul`; `bridge.rs` tests; canary benchmarks in `mlx_executor.rs` (3 tests); f32 matmul tested synthetically but not at production scale |
| MLX matmul (quantized) | ✅ | ✅ | 🚧 | ✅ | ❌ | Metal GPU | `quantized.rs` `QuantizedLinearBinding::forward` (3 tests); GPU canary in `mlx_executor.rs` measures quantized matmul CPU vs GPU at Gemma scale (3840); parity oracle `dequantize_then_matmul` |
| MLX dequantize | ✅ | ⬜ | ❌ | ❌ | ❌ | Metal GPU | `quantized.rs` `dequantize_then_matmul` exists but is `#[allow(dead_code)]`; no dedicated dequantize-only tests |
| MLX memory telemetry | ✅ | ⬜ | 🚧 | ❌ | ❌ | Metal GPU | `compute_image.rs` exports `mlx_active_memory_bytes`, `mlx_cache_memory_bytes`, `mlx_peak_memory_bytes`; `worker_memory.rs` `sample_mlx_memory()`; no dedicated unit tests — exercised indirectly |
| MLX cache control | ✅ | ⬜ | 🚧 | ❌ | ❌ | Metal GPU | `compute_image.rs` `clear_mlx_cache`, `set_mlx_cache_limit`, `set_mlx_memory_limit`; `worker_memory.rs` `configure_mlx_memory_limits`; no unit tests for cache-control-only scenarios |
| MLX multithreaded execution | ✅ | ⬜ | 🚧 | ❌ | ❌ | Metal GPU | MLX itself manages GPU streams; `worker_supervisor.rs` manages concurrent worker processes; no isolated multithreading tests |
| MLX external array (no-copy) | ✅ | ✅ | ✅ | ❌ | ❌ | Metal GPU | `external_array.rs` `new_external_array` (9 tests); verified zero-copy round-trip with IOSurface-backed FP16 storage in `arena.rs` phases 0–4; no no-copy benchmarks at Gemma 12B model scale |
| **Core ML bridge** | | | | | | | |
| Core ML state bridge | 🚧 | 🚧 | ❌ | ❌ | ❌ | ANE (M1+) | `coreml_state.rs` compiles; `bridge/coreml_state.mm` has stubs — stateful prediction crashes at runtime; stateful model loading confirmed working with toy model per STATUS.md |
| Core ML execution (stateless) | ✅ | ✅ | ✅ | ⬜ | ❌ | ANE (M1+) | `coreml_bridge.rs` + `coreml_exec.mm`; tested via `arena.rs` (IOSurface identity models); stateless prediction works end-to-end |
| Core ML arena / IOSurface | ✅ | ✅ | ✅ | ❌ | ❌ | ANE (M1+) | `arena.rs` (11 tests), `arena_lifecycle.rs` (3 tests), `arena_pool.rs` (4 tests); phases 0–4 fully verified with 5 identity model tests passing without SIGSEGV |
| **ComputeImage pipeline** | | | | | | | |
| ComputeImage compilation | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `compute_image.rs` `compile()` (14 tests); compilation produces manifest, segments, receipts; requires safetensors files for full-path testing |
| ComputeImage validation | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `validator.rs` `validate_bindings`; cross-references every tensor in ExecutionSpec against safetensors headers; tested via compute_image tests |
| ComputeImage verification | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `compute_image.rs` `verify()` (14 tests); checks manifest integrity, segment accessibility, cryptographic hashes |
| Model loading (safetensors) | ✅ | ⬜ | 🚧 | ❌ | ❌ | Apple Silicon | `loader.rs` `load_safetensors_json`, `inspect_safetensors`; `config.rs` `parse_config`; no dedicated unit tests for loader — tested indirectly through compile pipeline |
| Full model forward pass (Gemma 4 12B) | ✅ | ⬜ | ❌ | ❌ | ❌ | Metal GPU, ~16 GB+ RAM | `gemma.rs` `GemmaModel::forward` (48-layer decoder); `run_full_model_from_image` via napi; full model execution requires checkpoint safetensors not available in CI; synthetic parity tests exist in `primitives.rs` (6 tests) |
| **KV cache** | | | | | | | |
| KV cache (packed, quantized) | ✅ | ✅ | 🚧 | ❌ | ❌ | Metal GPU | `kv_cache.rs` `KvCache` (18 tests); sliding/global eviction, concurrency, clear; per STATUS.md: "implemented but not yet runtime-qualified" |
| KV cache Metal fused QK | ❌ | ❌ | ❌ | ❌ | ❌ | Metal GPU | Not implemented |
| KV cache prerotation | ✅ | ⬜ | ❌ | ❌ | ❌ | Metal GPU | `kv_cache.rs` integrates rotation tables; no isolated prerotation test |
| KV cache sparse attention | ❌ | ❌ | ❌ | ❌ | ❌ | Metal GPU | Not implemented |
| **Session management** | | | | | | | |
| Session isolation | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `session.rs` (32 tests); per-request inference sessions with independent state |
| Session cancellation | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `session.rs` cancellation flag mechanism; `streaming.rs` cancellation integration (21 tests) |
| Session memory accounting | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `worker_memory.rs` process RSS sampling + MLX allocator state (9 tests); `ModelAdmissionEstimate` in `model_runtime.rs` |
| **Worker infrastructure** | | | | | | | |
| Worker subprocess spawn | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `worker_supervisor.rs` worker process lifecycle (34 tests); `bin/tribunus-compute-worker.rs` binary |
| Worker protocol (TCP/pipe) | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `worker_protocol.rs` framed JSON protocol (13 tests); length-prefixed framing, stateful protocol validation |
| Worker health checks | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `worker_supervisor.rs` heartbeat monitoring, timeout enforcement (34 tests) |
| Worker graceful shutdown | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `worker_supervisor.rs` shutdown sequencing; `worker_protocol.rs` terminal message handling |
| **Streaming** | | | | | | | |
| Streaming token emission | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `streaming.rs` bounded event channels + token emission (21 tests) |
| Streaming cancellation | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `streaming.rs` cancellation integration |
| **Observability** | | | | | | | |
| Execution receipts | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `receipts.rs` (3 tests); `engine_receipts.rs` (15 tests); model load, request admission, phase, step, terminal, and worker exit receipts |
| Runtime tracing | ✅ | ✅ | 🚧 | ❌ | ❌ | Metal GPU | `runtime_trace.rs` kernel trace hooks (11 tests); quantized matmul, elementwise, sync markers, timeline, decode steps; trace entries collected but not yet validated against real Metal execution |
| **Compiler & profiling** | | | | | | | |
| Fusion region formation | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `fusion_region.rs` (11 tests); 7 hardcoded Gemma 4 12B fusion regions (QKV, gate+up, SiLU+mul, down_proj, RMS norm+residual, self-attn megakernel) |
| Layout compilation | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `layout_compiler.rs` `compile_layouts` (12 tests); weight/activation segment planning, zero-copy eligibility, offset alignment; pure computation, no runtime validation |
| Placement profiling | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `placement_profile.rs` deployment profiles + `select_profile` (5 tests); M1 latency/throughput/constrained/fallback profiles |
| Hybrid profile (MLX + CoreML) | ✅ | ✅ | ❌ | ❌ | ❌ | ANE (M1+) | `hybrid_profile.rs` `HybridProfile` (3 tests); MLX regions, Core ML stateful islands, boundary tensors, execution order, fallback policy; serde + validation only — no runtime hybrid execution |
| Profile compilation | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `profile_compiler.rs` (7 tests); compiles operation audit → `ExecutionPlacementProfile`; M1 baseline partitions |
| **Engine** | | | | | | | |
| Engine error propagation | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `engine_error.rs` (19 tests); stable kebab-case error codes, structured error builder, failure/cancellation funnels, napi JSON conversion |
| Engine policy enforcement | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `engine_policy.rs` (9 tests); budget resolution, admission checks, deadline guard, qualification policy; pure-function, no runtime integration tests |
| Engine lifecycle | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `engine.rs` (6 tests); `engine_receipts.rs` (15 tests); model install, generate, cancel, worker exit |
| **Miscellaneous** | | | | | | | |
| Fake worker mode | ✅ | ⬜ | ❌ | ❌ | ❌ | None | `bin/tribunus-fake-worker.rs` binary exists; no unit tests |
| N-API surface (all exports) | ✅ | ⬜ | 🚧 | ❌ | ❌ | Apple Silicon | `lib.rs` 20+ `#[napi]` exports; smoke test `test/compute_image_smoke.test.js` (1 test); cross-checked against `index.d.ts` per CODE_REVIEW.md |
| Arena allocation/deallocation | ✅ | ✅ | ✅ | ❌ | ❌ | ANE (M1+) | `arena.rs` (11 tests); IOSurface-backed FP16 contiguous storage with lifecycle tracking |
| Arena residency tracking | ✅ | 🚧 | ❌ | ❌ | ❌ | ANE (M1+) | `arena.rs` tracks generation + IOSurface ID; receipt schema includes arena lifecycle state; no dedicated residency-boundary tests |
| Arena pool management | ✅ | ✅ | 🚧 | ❌ | ❌ | ANE (M1+) | `arena_pool.rs` acquire/release/reuse/max-per-key (4 tests) |
| Mapped image loading | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `mapped_image.rs` (13 tests); mmap MAP_PRIVATE segment loading, SegmentState lifecycle, SegmentView metadata |
| External array lifecycle | ✅ | ✅ | ✅ | ❌ | ❌ | Metal GPU | `external_array.rs` (9 tests); new_external_array, ExternalStorage trait, SegmentSlice adapter in `profiled_executor.rs` |
| Requalification | ✅ | ✅ | ❌ | ❌ | ❌ | Apple Silicon | `requalification.rs` `Requalifier` (17 tests); sliding-window evidence collection, promotion/deprecation thresholds; pure computation |
| Copy ledger | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `copy_ledger.rs` (6 tests); CopyEntry, SyncBoundary, audit_mapped_weight_layout, audit_runtime_path, fusion opportunity reporting |
| Operation catalog | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `operation_catalog.rs` `OperationCatalog` (7 tests); all 20 OperationKind variants, generate_from_plan |
| Residency tracking | ✅ | ✅ | 🚧 | ❌ | ❌ | Metal GPU | `residency.rs` `ResidencyManager` (16 tests); segment lifecycle state machine (absent → prefetched → bound → in-flight → retired), bounded budget admission |
| MLX patch register | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `mlx_patch_register.rs` (6 tests) |
| MLX inventory | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `mlx_inventory.rs` (6 tests); `MlxInventory` with inventory entries for all Gemma operations, completion heuristics |
| Core ML audit reports | ✅ | ✅ | ❌ | ❌ | ❌ | ANE (M1+) | `coreml_audit.rs` (8 tests); MLP island, decoder layer, stateful decode reports; ANE op classification; all hardcoded — no real Core ML artifact introspection |
| Projection tests (parity) | ✅ | ✅ | 🚧 | ❌ | ❌ | Metal GPU | `projection_tests.rs` (1 test); synthetic QKV/MLP projection parity between MLX and reference shapes |
| Attention primitives | ✅ | ✅ | 🚧 | ❌ | ❌ | Metal GPU | `attention.rs` (2 tests) |
| Primitive parity tests | ✅ | ✅ | 🚧 | ❌ | ❌ | Metal GPU | `primitives.rs` (6 tests); RMSNorm, GELU, RoPE, quantized embedding lookup, parity with synthetic data |
| Config parse + plan build | ✅ | ✅ | 🚧 | ❌ | ❌ | Apple Silicon | `config.rs` (1 test); parse_config, build_execution_plan, resolve_namespace |
| Profiled executor (cold path) | ✅ | ✅ | 🚧 | ❌ | ❌ | Metal GPU, 16 GB+ | `profiled_executor.rs` (1 test); cold full-model execution via MappedImage segments; requires model weights |

---

## Summary by Tier

### Counts by capability

| Tier | Count |
|---|---|
| Total capabilities assessed | 50 |
| Implemented (✅) | 45 |
| Unit-Tested (✅) | 37 |
| Runtime-Qualified (✅) | 5 |
| Benchmark-Qualified (✅) | 1 |
| Production-Supported (✅) | 0 |
| In Progress / Not Implemented (🚧/❌) | 5 |

### Runtime-qualified capabilities (✅ — proven on real hardware)

These five capabilities have been verified end-to-end on Apple M-series hardware with IOSurface-backed FP16 storage, zero-application-copy MLX ↔ Core ML round trip:

1. MLX external array (no-copy)
2. Core ML execution (stateless)
3. Core ML arena / IOSurface
4. Arena allocation/deallocation
5. External array lifecycle

### Benchmark-qualified capabilities (✅)

1. MLX matmul (quantized) — GPU canary benchmark measures quantized matmul CPU vs GPU at Gemma scale (3840 hidden dim)

### Capabilities not implemented or in progress (🚧/❌)

1. Core ML state bridge — 🚧 bridge compiles, stateful prediction crashes (ObjC++ stubs)
2. KV cache Metal fused QK — ❌ not implemented
3. KV cache sparse attention — ❌ not implemented
4. Full model forward pass (Gemma 4 12B) — 🚧 code complete + compiles, but requires checkpoint safetensors (~24 GB) not available in CI
5. Fake worker mode — ✅ binary exists but no tests

### Capabilities requiring real hardware for qualification

Most capabilities depend on Metal GPU or ANE (Core ML) for runtime qualification. The test suite runs on CI (no GPU, no ANE), so the following are **🚧 Runtime-Qualified** — compiled and unit-tested but never exercised against real hardware in an automated pipeline:

- All MLX capabilities (matmul, dequantize, telemetry, cache control)
- All KV cache capabilities
- All streaming capabilities
- Engine lifecycle (requires running workers)
- Mapped image loading (requires segment files)
- Residency tracking (requires MLX arrays on device)
- Attention/primitives/parity tests (require MLX eval on GPU)

### Production support

**Zero.** Version 0.1.0 — pre-alpha. No capabilities are production-supported. The engine pipeline compiles, the IOSurface arena path is verified on hardware, but the full 48-layer Gemma 4 12B model has not been executed end-to-end from a compiled ComputeImage.

---

## Qualification gaps by component

### What blocks each gap

| Gap | Blocker | Workaround |
|---|---|---|
| Full model forward pass not runtime-qualified | Gemma 4 12B checkpoint safetensors (~24 GB) not in CI | Can run locally with downloaded weights; `gemma_forward` + `run_full_model_from_image` tested synthetically |
| Core ML stateful prediction crashes | `coreml_state.mm` has ObjC++ stubs; MLState bridge not implemented | Stateless prediction works; stateful island execution deferred to Phase 12 |
| KV cache not runtime-qualified | No Metal GPU in CI | Integration test requires M-series runner; pure-logic tests pass |
| Streaming requires running worker process | End-to-end integration not wired in CI | `streaming.rs` unit tests pass; protocol tests pass in isolation |
| No benchmark numbers at model scale | Requires full model + GPU to measure | `cpu_benchmarks.rs` provides CPU-only micro-benchmarks; GPU canary in `mlx_executor.rs` provides projection-scale matmul latency |
