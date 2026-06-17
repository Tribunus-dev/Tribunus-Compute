<!--
GitHub description:
Rust-native execution kernel for Tribunus, providing governed MLX, Metal,
Accelerate, and Core ML inference with isolated sessions, bounded scheduling,
streaming, cancellation, memory accounting, and runtime qualification.
-->

# Tribunus Compute Kernel

Tribunus Compute Kernel is the native execution runtime for Tribunus. It
provides governed, hardware-accelerated model execution across MLX, Metal,
Accelerate, and Core ML, with isolated sessions, bounded scheduling, streaming,
cancellation, memory accounting, capability qualification, and evidence-bearing
execution receipts.

---

## 1. Overview

The kernel is the lowest layer of the Tribunus inference stack. It owns the MLX
runtime, N-API bridge, worker subprocesses, session state machines, KV cache,
model store, compile pipeline, and the IOSurface-backed SharedTensorArena for
zero-copy MLX/Core ML boundary transfers.

```
Tribunus App (JS/TS) ---> N-API (napi-rs) ---> Compute Kernel (Rust)
                                                     |
                                          +----------+----------+
                                          |          |          |
                                          v          v          v
                                        MLX C    Core ML    Accelerate
                                        (Metal)   (ANE/GPU)    (vDSP)
```

**Platforms:** macOS ARM64 (Metal, ANE, Accelerate); Linux ARM64/x86_64 (worker
protocol only, CPU inference via MLX CPU).

**Maturity:** pre-production (v0.1.0). Core subsystems implemented and
unit-tested. Full 48-layer Gemma 4 12B execution via ComputeImage is
structurally wired but needs real checkpoint weights. Core ML stateful
prediction bridge compiles but is not yet runtime-qualified.

## 2. Current Maturity

| Subsystem | Impl | Tested | Fixture-Q | Locally-E | Bench | Prod |
|---|---|---|---|---|---|---|
| Session state machines | ✓ | ✓ (3 modules) | ✓ | — | — | — |
| Worker supervisor + protocol | ✓ | ✓ (9 tests) | ✓ | — | — | — |
| Streaming event channels | ✓ | ✓ | — | ✓ | — | — |
| KV cache (sliding/global) | ✓ | ✓ (6 tests) | — | — | — | — |
| IOSurface SharedTensorArena | ✓ | ✓ (11 tests) | ✓ | — | — | — |
| Arena lifecycle | ✓ | ✓ (3 tests) | ✓ | — | — | — |
| Arena pool | ✓ | ✓ (4 tests) | ✓ | — | — | — |
| MLX external arrays | ✓ | — | — | — | — | — |
| Core ML stateless prediction | ✓ | ✓ | — | ✓ | — | — |
| Core ML stateful prediction | ✓ (compiles) | — | — | — | — | — |
| Capability report | ✓ | ✓ (2 tests) | ✓ | — | — | — |
| Receipt emission | ✓ | ✓ (3 tests) | ✓ | — | — | — |
| ComputeImage compile pipeline | ✓ | Smoke | — | ✓ | — | — |
| Mapped-image no-copy storage | ✓ | — | — | — | — | — |
| Residency manager | ✓ | — | — | — | — | — |
| Execution planner (MLX-only) | ✓ | — | — | — | — | — |
| Hybrid profile (MLX/Core ML) | ✓ | ✓ (3 tests) | — | — | — | — |
| Engine orchestrator | ✓ | — | — | — | — | — |
| Model store | ✓ | — | — | — | — | — |
| Runtime trace | ✓ | — | — | — | — | — |
| Re-qualification system | ✓ | — | — | — | — | — |
| Full 48-layer model execution | ✓ (structural) | — | — | — | — | — |
| Core ML MLP placement | ✓ (compiler) | — | — | — | — | — |

**Columns:** Fixture-Qualified = unit-tested with hw identity + runtime tuple +
commit SHA. Locally Exercised = real hardware but no formal repetition/receipts.
Runtime-Qualified (none yet) = full 7-element evidence tuple. See `STATUS.md`.

**Key qualifiers:** Multithreaded (4x50 quantised matmul, no SIGSEGV); Memory
telemetry (mlx-sys FFI); Quantised matmul (fused vs dequantised parity);
Concurrent generation (2x100 + 4x50); Residency (4->29->4 handles, MLX active
72128->71288).

**Gaps:** `supports_external_array` is **false** (no no-copy external arrays in
MLX C API). Core ML stateful prediction crashes at predict time (stubs). Full
48-layer ComputeImage execution needs weights not available in CI.

## 3. Architecture

Four major layers. Top: **N-API** (~50 exports). Middle-left: **Worker
Protocol** (length-prefixed JSON IPC, no tensor data crosses boundary).
Middle-right: **ComputeEngine** (policy resolution -> WorkerSupervisor ->
GenerationStream). Lower: **Session Runtime** (two FSMs), **Execution
Planners/Executors**, **SharedTensorArena / KV Cache / Residency / Receipts**.

```
+--- N-API Boundary (lib.rs) ---+
| create_array, matmul,         |
| compile_image, engine_gen,    |
| capability_report, ...        |
+------+---------+------+------+
       |         |      |
       v         v      v
+--------+  +-----------+  +---------+
| Worker  |  |ComputeEng |  |Session  |
|Protocol |  |Supervisor |  |Runtime  |
+--------+  +-----------+  +---------+
       |         |      |
       v         v      v
+-----------+  +-----------+  +-------------+
|MLX Exec   |  |Core ML    |  |Arena / KV   |
|(GPU/CPU)  |  |Bridge     |  |Cache/Resid  |
+-----------+  +-----------+  +-------------+
```

### Entry Points

1. **N-API bridge** (`lib.rs`): ~50 exports (array ops, safetensors, Gemma
   forward/sample, ComputeImage lifecycle, engine generate/cancel, capability
   report, memory telemetry). `napi-rs` 3.9.0, napi-8 ABI. See
   [ARCHITECTURE.md SS2](docs/ARCHITECTURE.md#2-entry-points).

2. **Worker protocol** (`worker_protocol.rs`): length-prefixed JSON IPC over
   stdin/stdout. Only token IDs and metadata cross the boundary — no tensor
   payloads. Protocol V1_0. See
   [ARCHITECTURE.md SS2.2](docs/ARCHITECTURE.md#22-worker-protocol).

3. **ComputeEngine** (`engine.rs`): resolves policy, checks model state,
   delegates to `WorkerSupervisor`. Returns `GenerationHandle` (stream + id).

### Worker Subprocess

`WorkerSupervisor`: **command** thread (stdin frame parsing + routing),
**inference** thread (model, prefill/decode), **heartbeat** thread (250 ms
telemetry). Automatic restart (configurable `restart_limit`), cancellation
grace periods, soft/hard RSS ceilings, stuck-worker watchdog.

## 4. Capabilities

**Inference** — Gemma 4 12B (48-layer, GQA, RMSNorm, RoPE, SwiGLU, quantised
matmul, sampling). ComputeImage: `run_full_model_from_image()` returns next
token directly. Hybrid MLX/Core ML profiles. GPU-canary-gated profiled executor.

**Storage** — Copied-v0 (CPU baseline), Mapped-no-copy-v1 (mmap, in progress),
SharedTensorArena (IOSurface-backed FP16, zero-copy MLX/Core ML boundary). See
[ARCHITECTURE.md SS18](docs/ARCHITECTURE.md#18-storage-capabilities).

**Memory** — Machine profile (`hw.memsize`), MLX active/cache limits, RSS
ceilings, residency manager (Prefetched->Bound->InFlight->Retired), copy
ledger. See [ARCHITECTURE.md SS19](docs/ARCHITECTURE.md#19-memory-management-detail).

**Security** — Per-UUID session FSMs, model runtime in separate process, no
tensor data across IPC, cancellation with grace period + SIGKILL, memory
ceilings. See [ARCHITECTURE.md SS16](docs/ARCHITECTURE.md#16-security-and-trust-model)
and `SECURITY.md`.

**Observability** — JSON receipts per lifecycle event (ModelLoad, ArenaCreation,
CoreMlPrediction, HybridJob, Generation, EngineError). Copy classification
taxonomy. See [ARCHITECTURE.md SS17](docs/ARCHITECTURE.md#17-observability).

**Compilation** — 4-lane parallel relocation pipeline, layout compiler, 7
Gemma 4 fusion topologies, profile compiler, hermetic Core ML toolchain (Python
3.13 ARM64 + coremltools 9.0). See
[ARCHITECTURE.md SS5](docs/ARCHITECTURE.md#5-execution-planner).

## 5. Supported Platforms

| Platform | Arch | MLX | Metal | Core ML | Accel | Worker | Status |
|---|---|---|---|---|---|---|---|
| macOS 15+ ARM64 (M1) | arm64 | ✓ | ✓ | ✓ | ✓ | ✓ | Actively tested |
| macOS 15+ ARM64 (M2+) | arm64 | ✓ | ✓ | ✓ | ✓ | ✓ | Expected compatible, unqualified |
| macOS 15+ | x86_64 | — | — | — | — | ✓ | Planned |
| Linux | arm64/x86_64 | CPU | — | — | — | ✓ | Implemented, unqualified |
| Windows | x86_64 | — | — | — | — | — | Planned |
| CUDA / ROCm / other | — | — | — | — | — | — | Planned |

**Requirements:** Rust >= 1.82.0, Xcode CLT (macOS), MLX C 0.6.0 (from
`mlx-rs-fork` submodule), Python 3.13 ARM64 + coremltools 9.0 (Core ML
compilation only).

### Platform Qualification Scope

Tribunus Compute is currently bootstrapped and qualified on a first-generation
Apple Silicon Mac with an M1 processor. macOS ARM64 is therefore the only
actively developed and tested hardware target at this stage. Support for
additional Apple Silicon generations, Intel macOS, Linux, Windows, discrete
GPUs, and other accelerator configurations is planned after the kernel reaches
a stable release baseline.

Current platform claims should be understood as implementation targets unless
they are backed by published qualification evidence for that specific operating
system, architecture, and hardware configuration.

Hardware donations, hosted test-machine access, CI runner sponsorships, and
monetary contributions are welcome. Contributions will be used to expand the
qualification matrix, acquire representative hardware, fund signing and release
infrastructure, and validate the kernel across additional operating systems and
accelerator configurations.

## 6. Repository Structure

```
.
+-- Cargo.toml, CHANGELOG.md, LICENSE
+-- .github/
|   +-- CODEOWNERS, PULL_REQUEST_TEMPLATE.md
|   +-- ISSUE_TEMPLATE/ (4 templates)
|   +-- workflows/ci.yml
+-- docs/
|   +-- ARCHITECTURE.md, compatibility.md, CONTRIBUTING.md
|   +-- LICENSE-COMMERCIAL.md, RELEASING.md, SECURITY.md
|   +-- STATUS.md, SUPPORT.md
+-- compute-native/           # Kernel crate (cdylib + rlib + 2 bins)
|   +-- Cargo.toml, package.json, build.rs
|   +-- src/
|   |   +-- lib.rs, bridge.rs
|   |   +-- bridge/           # ObjC++: coreml_arena.{h,mm}, coreml_exec.mm, coreml_state.{h,mm}
|   |   +-- bin/              # tribunus-compute-worker, tribunus-fake-worker
|   |   +-- engine.*, session.*, streaming.*, worker_*.rs
|   |   +-- compute_image.*, config.*, executor.*, mlx_executor.*
|   |   +-- mapped_image.*, residency.*, kv_cache.*, attention.*
|   |   +-- primitives.*, quantized.*, gemma.*
|   |   +-- arena.*, arena_lifecycle.*, arena_pool.*, external_array.*
|   |   +-- compile_*.rs, compute_ir.*, layout_compiler.*, fusion_region.*
|   |   +-- profile_compiler.*, placement_profile.*, hybrid_profile.*
|   |   +-- coreml_bridge.*, coreml_state.*, coreml_pipeline.*, coreml_audit.*
|   |   +-- gpu_worker.*, cpu_benchmarks.*, projection_tests.*
|   |   +-- capability.*, receipts.*, runtime_trace.*, requalification.*
|   |   +-- loader.*, model.*, validator.*, errors.*, cli.*
|   +-- models/, test/, tools/coreml-compiler/
+-- mlx-rs-fork/              # Git submodule (mlx-sys, mlx-rs, etc.)
```

## 7. Building

```bash
git submodule update --init --recursive
cd compute-native && npm run build       # N-API addon (macOS)
cargo build --release -p tribunus-compute-native  # All targets
cargo build --release --no-default-features --features linux-compat --bin tribunus-compute-worker  # Linux worker only
```

Feature gate `private-development` relaxes memory limits (local only, never CI).

## 8. Testing

```bash
cargo test -p tribunus-compute-native              # 38 unit tests
cargo test -p tribunus-compute-native -- --ignored  # Qual tests (needs Apple Silicon)
cargo test all_real_projection_parity -- --nocapture  # Safetensors parity
cd compute-native && bun test                       # N-API smoke test
```

**Modules:** arena (11), arena_lifecycle (3), arena_pool (4), capability (2),
worker_supervisor (9), kv_cache (6), errors (3), hybrid_profile (3), receipts (3).

**Fake worker** (`tribunus-fake-worker`): 12 fault modes (normal,
identity-mismatch, no-handshake, model-load-hang, slow-prefill, ignored-cancel,
heartbeat-loss, malformed-frames, sequence-gap, duplicate-terminal, crash,
memory-alloc).

## 9. Runtime Qualification

Two-phase: compile-time `ModelExecutionPlan` (structural validity), then
evidence-based promotion via `requalification.rs` (oracle vs candidate
throughput, boundary latency, device activity). Qualification mode: max_tokens
coerced to 8, prompt capped to 64, deadline 30s.

Receipts are JSON with frozen schemas per major version. Types: ModelLoad,
ArenaCreation, CoreMlPrediction, HybridJob, Generation, EngineError.
CopyClassification: `application_copy_free`, `copied_fallback`,
`materialized_layout_conversion`, `internal_coreml_staging_unknown`. See
[ARCHITECTURE.md SS17](docs/ARCHITECTURE.md#17-observability).

## 10. Security & Isolation

Per-UUID session FSMs with strict transitions. Model runtime in separate OS
process — no tensor data crosses IPC. `ProtocolValidator` enforces frame
ordering. Watchdog (250 ms heartbeat). MLX active memory FFI, RSS sampling,
soft ceiling (MemoryPressure protocol), hard ceiling (SIGKILL). Cancellation
(`AtomicBool` + grace period).

| Boundary | Trust Model |
|---|---|
| JS <-> N-API | Untrusted caller; inputs validated |
| Host <-> Worker | Worker untrusted; frames validated |
| Worker <-> MLX C | MLX C trusted; crash terminates worker |
| Worker <-> Core ML | Core ML trusted (process-scoped) |
| Worker <-> IOSurface | Not cross-process in v0.1.0 |

See [ARCHITECTURE.md SS16](docs/ARCHITECTURE.md#16-security-and-trust-model)
and `SECURITY.md`.

## 11. Versioning

| Component | Current | Scheme |
|---|---|---|
| Kernel | v0.1.0 | Semver |
| N-API ABI | napi8 | Stable across patch versions |
| MLX fork | 0.25.3-tribunus.1 | Upstream + `.tribunus.N` patch |
| MLX C | 0.6.0 | Bound to `mlx-sys` |
| Worker protocol | 1.0 | Bumped on incompatible changes |
| Storage ABI | copied-v0, mapped-no-copy-v1 | Frozen; new types get new IDs |
| Capability names | Frozen | Add-only |

Every release records version + fork commit + MLX C + macOS + protocol +
capability hash in the receipt chain. See `docs/compatibility.md`.

## 12. License

Available under AGPL-3.0-only; separate commercial licenses from the copyright
holder.

### Open Source — AGPL-3.0-only

```
Copyright (c) 2026 Tribunus, Inc.
This program is free software under AGPL-3.0-only.
```

The licensor's stated interpretation is that any application linking through the
N-API addon (including `require('@tribunus/compute-native')`) must comply with
AGPL-3.0 terms. If uncertain, seek legal counsel or obtain a commercial license.
AGPL Section 13 applies to modified versions supporting remote network
interaction.

### Commercial License

Proprietary incorporation — closed-source product, distribution under non-AGPL
terms, or use where AGPL network obligations are not acceptable — requires a
separate commercial license from the copyright holder.
May include warranties, indemnification, and separately negotiated patent rights subject to negotiation.
Inquiries: `license@tribunus.io` | See
[docs/LICENSE-COMMERCIAL.md](docs/LICENSE-COMMERCIAL.md).

### Contributor Agreement

External contributions require a CLA granting the copyright holder sufficient
rights for both AGPL and commercial licensing. See
[docs/CONTRIBUTING.md SS10](docs/CONTRIBUTING.md#10-license-and-contributor-agreement).

### Third-Party Components

`mlx-rs`/`mlx-sys`: MIT OR Apache-2.0 (fork at `mlx-rs-fork/`). `napi-rs`:
MIT. `safetensors`: Apache-2.0.

## 13. Roadmap

| Milestone | Scope | Target |
|---|---|---|
| **v0.2 — KV cache** | Prefill + cached decode, parity verification | Near-term |
| **v0.3 — No-copy residency** | mmap segments, MLX external arrays, flat plateau | Near-term |
| **v0.4 — Core ML state qual** | Mutation, isolation, cancellation | Near-term |
| **v0.5 — Streaming** | Event channels, token emission, supervisor lifecycle | Near-term |
| **v1.0 — Production** | Full 48-layer execution, receipts, stress, restart | Medium-term |
| **v1.1 — MLP placement** | Boundary latency benchmark (MLX/arena/Core ML) | Medium-term |
| **v1.2 — Hybrid** | Profile-compiler-driven MLX/Core ML | Medium-term |
| **v2.0 — ANE autotuning** | Island detection, placement optimisation | Medium-term |
| **v2.x — Xproc IOSurface** | Shared IOSurface across processes | Future |

**Blocked:** Full 48-layer execution needs weights (not in CI). Core ML
stateful (`coreml_state.mm` compiles but stubbed). Core ML compiler preflight
needs pinned Python 3.13 + coremltools 9.0.

## 14. Contributing

See `docs/CONTRIBUTING.md` for workflow, branch conventions, commit format,
review guidelines, test expectations, and MLX fork maintenance.

- New subsystems require unit tests before merging.
- MLX fork / worker protocol changes: backward compat or bump version.
- Capability report: frozen ABI (add-only).
- Receipt schemas: frozen per major version; new fields `#[serde(default)]`.
- Formatting: `cargo fmt`; linting: `cargo clippy --all-targets`.
