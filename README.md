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

The kernel is the lowest layer of the Tribunus inference stack. It owns the
MLX runtime, N-API bridge, worker subprocesses, session state machines, KV
cache, model store, compile pipeline, and the IOSurface-backed SharedTensorArena
for zero-copy MLX/Core ML boundary transfers.

**Where it sits:**

```
Tribunus App (JS/TS) ──→ N-API (napi-rs) ──→ Compute Kernel (Rust)
                                                     │
                                          ┌──────────┼──────────┐
                                          ↓          ↓          ↓
                                        MLX C    Core ML    Accelerate
                                        (Metal)   (ANE/GPU)    (vDSP)
```

**Platforms:**

- **macOS ARM64** — primary target. Metal GPU, ANE, Accelerate all available.
- **Linux ARM64 / x86_64** — worker protocol only (CPU inference via MLX CPU).

**Maturity:**

The kernel is pre-production (v0.1.0). Core subsystems (session isolation,
worker protocol, streaming, KV cache, compilation pipeline, SharedTensorArena,
receipts, capability reports, residency management) are implemented and
unit-tested. Full 48-layer Gemma 4 12B execution with the compiled ComputeImage
path is structurally wired but requires real checkpoint weights for end-to-end
validation. Core ML stateful prediction bridge compiles but is not yet
runtime-qualified.

## 2. Current Maturity

| Subsystem | Implemented | Unit-Tested | Runtime-Qualified | Benchmark-Qualified | Production-Supported |
|---|---|---|---|---|---|
| Session state machines | ✓ | ✓ (3 state-machine modules) | ✓ | — | — |
| Worker supervisor + protocol | ✓ | ✓ (9 tests) | ✓ | — | — |
| Streaming event channels | ✓ | ✓ | ✓ | — | — |
| KV cache (sliding/global) | ✓ | ✓ (6 tests) | — | — | — |
| IOSurface SharedTensorArena | ✓ | ✓ (11 tests) | ✓ | — | — |
| Arena lifecycle | ✓ | ✓ (3 tests) | ✓ | — | — |
| Arena pool (acquire/release/reuse) | ✓ | ✓ (4 tests) | ✓ | — | — |
| MLX external arrays | ✓ | — | — | — | — |
| Core ML stateless prediction | ✓ | ✓ | ✓ | — | — |
| Core ML stateful prediction | ✓ (bridge compiles) | — | — | — | — |
| Capability report | ✓ | ✓ (2 tests) | ✓ | — | — |
| Receipt emission | ✓ | ✓ (3 tests) | ✓ | — | — |
| ComputeImage compile pipeline | ✓ | Smoke | ✓ | — | — |
| Mapped-image no-copy storage | ✓ | — | — | — | — |
| Residency manager | ✓ | — | — | — | — |
| Execution planner (MLX-only) | ✓ | — | — | — | — |
| Hybrid profile (MLX/Core ML) | ✓ | ✓ (3 tests) | — | — | — |
| Engine orchestrator | ✓ | — | — | — | — |
| Model store | ✓ | — | — | — | — |
| Runtime trace | ✓ | — | — | — | — |
| Re-qualification system | ✓ | — | — | — | — |
| Full 48-layer model execution | ✓ (structural) | — | — | — | — |
| Core ML MLP placement | ✓ (compiler) | — | — | — | — |

**Key subsystem qualifiers:**

- **Multithreaded execution** — qualified: 4×50 heavy quantised matmul, no SIGSEGV.
- **Memory telemetry** — `mlx-sys` FFI for `mlx_active_memory()`, `mlx_clear_cache()`.
- **Quantised matmul** — fused path verified against dequantize+matmul parity.
- **Concurrent generation** — 2×100 + 4×50 quantised matmul, no SIGSEGV.
- **ComputeImage residency** — 4→29→4 handles, MLX active memory tracked (72128→71288).

Notable gaps: `supports_external_array` is **false** — MLX C API does not support
no-copy external arrays yet. Core ML stateful prediction bridge is compiled but
crashes at predict time (`coreml_state.mm` stubs). Full 48-layer ComputeImage
execution requires checkpoint weights not available in CI.

## 3. Architecture

The kernel is decomposed into four major layers:

```
┌────────────────────────────────────────────────────────────────┐
│                     N-API Boundary (lib.rs)                     │
│  create_array, matmul, load_safetensors, compile_image,        │
│  engine_generate, engine_cancel, capability_report, ...        │
└──────────┬──────────────────────────────────┬──────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐    ┌──────────────────────────────────┐
│   Worker Protocol    │    │       ComputeEngine              │
│   (framed JSON IPC)  │    │  resolve_policy → check_store →  │
│                      │    │  WorkerSupervisor.generate()     │
│   Host ↔ Worker      │    │                                  │
│   over stdin/stdout  │    │  ┌─ WorkerSupervisor ──────────┐ │
│                      │    │  │ Process lifecycle           │ │
│   Commands:          │    │  │ Handshake, heartbeats,      │ │
│   Hello, LoadModel,  │    │  │ watchdog, cancellation,     │ │
│   StartGeneration,   │    │  │ restart with backoff        │ │
│   CancelGeneration,  │    │  └─────────────────────────────┘ │
│   UnloadModel,       │    │                                  │
│   Ping, Shutdown     │    │  ┌─ GenerationStream ──────────┐ │
│                      │    │  │ Event channel (Mutex+Cdv)   │ │
│   Events:            │    │  │ Started, Token, Chunk,      │ │
│   HelloAck, Token,   │    │  │ Progress, Metrics,          │ │
│   PrefillStarted,    │    │  │ Done, Error, Cancelled      │ │
│   StepMetrics,       │    │  └─────────────────────────────┘ │
│   GenerationCompleted│    │                                  │
└──────┬───────────────┘    └──────────┬───────────────────────┘
       │                               │
       ▼                               ▼
┌────────────────────────────────────────────────────────────────┐
│                   Session Runtime (session.rs)                  │
│  ControlSessionState: Created → Admitted → Submitted →        │
│    PrefillRunning → Decoding → Completed / Cancelled / Failed │
│  InferenceSessionState: Created → Prefill → Decode → Done     │
│  SamplerConfig, token-buffer management                        │
└────────────────────────────────────────────────────────────────┘
       │                               │
       ▼                               ▼
┌────────────────────────────────────────────────────────────────┐
│              Execution Planner (config.rs)                       │
│  ModelManifest → TextArchitecture → ModelExecutionPlan         │
│  ModelExecutionPlan: ProloguePlan + Vec<LayerPlan> + Epilogue  │
│  LayerPlan: dimensions, attention_kind, quant metadata         │
│                                                                 │
│              Executors (executor.rs)                            │
│  run_prologue(), run_layer(), run_epilogue()                    │
│  Storage-neutral: consume Plan + resolved Array references     │
│  Caller must eval() before dropping weight leases              │
└────────────────────────────────────────────────────────────────┘
       │                               │
       ▼                               ▼
┌────────────────────────────────────────────────────────────────┐
│              MLX Executor (mlx_executor.rs)                     │
│  Explicit Device+Stream pair (GPU or CPU)                      │
│  run_gpu_canary: measure quantised matmul latency diff         │
│                                                                 │
│              Core ML Bridge (coreml_bridge.rs)                  │
│  Stateless prediction via ObjC++ (coreml_exec.mm)              │
│  Stateful prediction (coreml_state.mm — bridge compiles)       │
│                                                                 │
│              Accelerate (via MLX feature flag)                  │
└────────────────────────────────────────────────────────────────┘
       │                               │
       ▼                               ▼
┌────────────────────────────────────────────────────────────────┐
│              SharedTensorArena (arena.rs, coreml_arena.mm)      │
│  IOSurface-backed FP16 arena for MLX/Core ML boundary          │
│  CVPixelBuffer + MLMultiArray zero-copy path                   │
│                                                                 │
│              KV Cache (kv_cache.rs)                             │
│  Per-layer sliding ring buffer or global concat                 │
│  Commit/rollback staging, byte accounting                       │
│                                                                 │
│              Residency Manager (residency.rs)                   │
│  Prefetched → Bound → InFlight → Retired lifecycle             │
│  Budget + safety reserve enforcement                           │
│                                                                 │
│              Receipt Emitter (receipts.rs, engine_receipts.rs)  │
│  ArenaCreationReceipt, CoreMlPredictionReceipt,                │
│  HybridJobReceipt, ModelLoadReceipt, GenerationReceipt,        │
│  WorkerExitReceipt — all with CopyClassification               │
└────────────────────────────────────────────────────────────────┘
```

### Entry Points

1. **N-API bridge** (`lib.rs`, `bridge.rs`): ~50 exported functions covering
   array ops, safetensors loading, Gemma forward/sample, ComputeImage
   compile/read/verify, engine generate/cancel, capability report, and memory
   telemetry. Hosted by `napi-rs` 3.9.0 with napi-8 ABI. Handles are `i64`
   indices into a global `ArrayRegistry`.

2. **Worker protocol** (`worker_protocol.rs`): framed length-prefixed JSON IPC
   over stdin/stdout. No tensor payload crosses the process boundary — only
   metadata, token IDs, and control frames. Protocol version V1_0. Max frame
   size is configurable via `ExecutionPolicy`.

3. **ComputeEngine** (`engine.rs`): thin orchestrator that resolves policy,
   checks model state, and delegates to `WorkerSupervisor`. Accepts
   `GenerationRequest` with prompt, session_id, max_tokens, temperature, top_k,
   top_p, seed, and stop-token sequences. Returns `GenerationHandle`
   (event stream + job_id).

### Worker Subprocess

The `WorkerSupervisor` (`worker_supervisor.rs`) manages a worker subprocess
with three independent threads:

- **Command thread**: reads stdin, parses frames, validates with
  `ProtocolValidator`. Forwards LoadModel / StartGeneration / CancelGeneration /
  UnloadModel via mpsc channel. Handles Ping and Shutdown directly.
- **Inference thread**: owns `LoadedProfiledModel` and
  `ProfiledInferenceSession`. Runs prefill + decode loop. Publishes phase,
  layer, and token progress to shared atomics.
- **Heartbeat thread**: emits `Heartbeat` frames every 250 ms with telemetry
  from shared atomics.

The supervisor supports automatic worker restart (up to `restart_limit`),
cancellation grace periods, soft/hard RSS ceilings, and a watchdog thread for
stuck-worker detection.

## 4. Capabilities

### Inference

- **Gemma 4 12B decoder** — config-driven 48-layer architecture with sliding
  (local) and full (global) attention scheduling, GQA, RMSNorm, RoPE, SwiGLU,
  quantised matmul, top-k/top-p/greedy sampling. See `gemma.rs`, `config.rs`,
  `attention.rs`, `primitives.rs`.
- **ComputeImage execution**: `compile_image()` produces a validated,
  execution-ordered artifact (`manifest.json` + segment files). `run_full_model_from_image()` executes the full 48-layer model and returns the next token
  ID directly — no logits cross the FFI boundary.
- **Hybrid MLX/Core ML deployment** via `hybrid_profile.rs`: profiles describing
  MLX regions, Core ML stateful islands, boundary tensors, arena profiles,
  execution order, and fallback policy.
- **Profiled heterogeneous executor** (`profiled_executor.rs`): GPU-canary-gated
  execution with explicit receipts, KV cache integration, and runtime timeline
  tracing.

### Storage

- **Copied-v0** (baseline): CPU-allocated, runtime-ready image.
- **Mapped-no-copy-v1** (in progress): mmap'd segment files with TensorEntry
  offset tables, designed for direct Metal buffer wrapping.
- **SharedTensorArena**: IOSurface-backed FP16 boundary tensors with
  `ArenaInfo` C-compatible struct. FFI to ObjC++ bridge for IOSurfaceCreate,
  CVPixelBufferCreate, MLMultiArray wrapping.
- **IOSurface FP16 canonical boundary path**: MLX writes arena A through
  external array → evaluation completes → ownership transfers to Core ML →
  Core ML reads via MLMultiArray → Core ML writes arena B through
  outputBackings → ownership transfers to MLX → MLX consumes via external
  array. Copy classification: `application_copy_free = true`.
- **ExternalHostMemory fallback**: arena allocated via posix_memalign, MLX wraps
  via `mlx_array_new_data_managed`, Core ML reads via `initWithDataPointer`.

### Memory Management

- **Machine profile detection**: `hw.memsize` sysctl, system reserve
  subtraction, GPU family identification.
- **MLX limit configuration**: active memory ceiling, cache limit, RSS soft/hard
  ceilings. Configured via `ExecutionPolicy`.
- **Residency manager**: `Prefetched → Bound → InFlight → Retired` lifecycle
  with configurable budget and safety reserve.
- **Memory telemetry**: `mlx_active_memory()` (MLX Metal active bytes),
  `mlx_clear_cache()` (MIL cache flush), `sample_process_rss_self()` via mach.
- **Copy ledger**: per-operation copy tracking with `CopyClass` taxonomy
  (zero-copy-mapped, IO-surface-shared, explicit-CPU-staging, etc.).

### Security and Isolation

- **Session isolation**: full state machine (`ControlSessionState`), per-session
  UUID, illegal-transition rejection.
- **Worker process boundaries**: model runtime runs in a separate OS process.
  No tensor data crosses the IPC boundary — only token IDs and metadata.
- **Cancellation**: prompt cancellation via `CancelGeneration`, cancellation
  grace period, forced worker teardown on timeout.
- **Memory ceilings**: hard RSS ceiling terminates the worker; soft ceiling
  triggers memory-pressure protocol frame.
- **Protocol validation**: `ProtocolValidator` checks frame ordering, message
  kinds, and `MAX_FRAME_SIZE_BYTES`.
- **Integrity seals**: `InstallationSeal` records per-segment SHA-256 hashes,
  file sizes, and installation timestamps.

### Observability

- **Receipts**: structured JSON receipts for every lifecycle event:
  - `ModelLoadReceipt` — image_hash, worker PID, model_open_ms, mapped_virtual_bytes,
    persistent_resident_bytes, segment_count, mlx_active_limit_bytes, admission
    estimate.
  - `ArenaCreationReceipt` — arena_id, IO surface ID, logical shape, physical
    dimensions, bytes_per_row, pixel format.
  - `CoreMlPredictionReceipt` — model_hash, input/output arena IDs, IOSurface
    IDs, duration_ms, copy classification.
  - `HybridJobReceipt` — full job lifecycle with lease transitions, Core ML
    predictions, state mutations, finalizer count, and combined copy
    classification.
  - `EngineError` — stable kebab-case error codes (`model-busy`,
    `worker-crashed`, `deadline-exceeded`, etc.).
- **Runtime trace**: per-operation trace entries recording device, stream,
  kernel family, graph-build time, eval time, temporary copies, and memory
  state.
- **Capability report**: `SharedTensorCapabilityReport` with 14 frozen
  capability booleans (iosurface_creation, fp16_pixelbuffer_multiarrays,
  mlx_coreml_round_trip, etc.).

### Compilation

- **ComputeImage pipeline** — 4-lane parallel relocation:
  source-read → relocate → write → hash, connected by bounded Tokio channels.
  Backpressure propagates from slow writer to source reader.
- **Layout compiler** (`layout_compiler.rs`) — plans tensor memory layout from
  IR + MLX primitive inventory. Validates zero-copy eligibility (page alignment,
  contiguity, non-aliasing).
- **Fusion regions** (`fusion_region.rs`) — 7 hardcoded Gemma 4 decoder-layer
  fusion topologies with per-backend implementation candidates.
- **Profile compiler** (`profile_compiler.rs`) — translates execution-plan
  outputs into `ExecutionPlacementProfile` with per-backend candidate regions.
- **Core ML compiler toolchain** — hermetic Python 3.13 ARM64 environment
  with coremltools 9.0. Generates ML Programs with `ct.TensorType(dtype=np.float16)`.

## 5. Supported Platforms

| Platform | Architecture | MLX | Metal | Core ML | Accelerate | Worker Protocol | Status |
|---|---|---|---|---|---|---|---|
| macOS 15+ | ARM64 | ✓ | ✓ | ✓ | ✓ | ✓ | **Primary** |
| macOS 15+ | x86_64 | — (no GPU) | — | ✓ | ✓ | ✓ | Worker only |
| Linux | ARM64 / x86_64 | CPU-only | — | — | — | ✓ | Worker only |

**macOS target:** `aarch64-apple-darwin`. The N-API binary is built only for
this triple. Linux builds skip the N-API cdylib and build the worker binaries
with CPU-only MLX.

### Build Requirements

- **Rust toolchain** ≥ 1.82.0 (MSRV required by `mlx-rs`)
- **macOS** with Xcode Command Line Tools (for Metal, Core ML, IOSurface)
- **MLX C 0.6.0** — built from the `mlx-rs-fork` submodule; requires Metal
  framework headers
- **Python 3.13 ARM64** with `coremltools==9.0` (for Core ML compilation only;
  runtime does not require Python)

## 6. Repository Structure

```
.
├── Cargo.toml                  # Workspace root (members: [compute-native])
├── package.json                # Root workspace package.json (monorepo entry)
├── AGENTS.md                   # Development conventions for AI agents
├── BRANDING.md                 # Brand guidelines
├── CONTRIBUTING.md             # Contribution guide
├── TYPECHECK_LEDGER.md         # Typecheck migration records
│
├── docs/
│   └── compatibility.md        # MLX dependency tuple and fork provenance
│
├── compute-native/             # 👈 The kernel crate
│   ├── Cargo.toml              #   cdylib + rlib + two bin targets
│   ├── package.json            #   @tribunus/compute-native NPM package
│   ├── index.d.ts              #   Auto-generated N-API type declarations
│   ├── build.rs                #   napi-build + ObjC++ bridge compilation
│   ├── STATUS.md               #   Detailed implementation status
│   ├── CODE_REVIEW.md          #   API alignment review against mlx-rs 0.21.2
│   ├── environment.json        #   Python / coremltools environment snapshot
│   │
│   ├── src/
│   │   ├── lib.rs              #     ~50 N-API exports
│   │   ├── bridge.rs           #     ArrayRegistry, opaque handle bridge
│   │   │
│   │   ├── bridge/             #     ObjC++ FFI files
│   │   │   ├── coreml_arena.mm #       IOSurface/CVPixelBuffer allocation
│   │   │   ├── coreml_arena.h
│   │   │   ├── coreml_exec.mm  #       Core ML stateless prediction
│   │   │   ├── coreml_state.mm #       Core ML stateful prediction (stubs)
│   │   │   ├── coreml_state.h
│   │   │   ├── coreml_arena.mm
│   │   │   └── coreml_arena.h
│   │   │
│   │   ├── bin/
│   │   │   ├── tribunus-compute-worker.rs  # Real inference subprocess
│   │   │   └── tribunus-fake-worker.rs     # Deterministic test harness (12 fault modes)
│   │   │
│   │   ├── engine.rs            #   ComputeEngine orchestrator
│   │   ├── engine_error.rs      #   Typed error taxonomy (19 codes)
│   │   ├── engine_policy.rs     #   ExecutionPolicy, budget resolution
│   │   ├── engine_receipts.rs   #   ModelLoad/Generation/WorkerExit receipts
│   │   ├── session.rs           #   Control + Inference session state machines
│   │   ├── streaming.rs         #   Mutex+Cdv event channels (GenerationEvent)
│   │   ├── worker_protocol.rs   #   Framed JSON IPC protocol
│   │   ├── worker_supervisor.rs #   Subprocess lifecycle, heartbeats, watchdog
│   │   ├── worker_memory.rs     #   RSS sampling, MLX limit config
│   │   │
│   │   ├── compute_image.rs     #   Manifest, segments, full-model runner
│   │   ├── config.rs            #   TextArchitecture, ModelExecutionPlan
│   │   ├── executor.rs          #   prologue/layer/epilogue executors
│   │   ├── mlx_executor.rs      #   Explicit Device+Stream executor
│   │   ├── profiled_executor.rs #   GPU-canary-gated profiled executor
│   │   ├── model_runtime.rs     #   Persistent model-instance handle
│   │   ├── model_store.rs       #   InstallationSeal, integrity verification
│   │   ├── mapped_image.rs      #   Mapped-no-copy segment view
│   │   ├── residency.rs         #   Segment lease lifecycle manager
│   │   ├── kv_cache.rs          #   Sliding/global per-layer KV cache
│   │   ├── attention.rs         #   Gemma 4 sliding/global attention
│   │   ├── primitives.rs        #   RMSNorm, RoPE, GELU, quantised embedding
│   │   ├── quantized.rs         #   QuantizedLinearBinding wrapper
│   │   ├── gemma.rs             #   Gemma 4 12B decoder
│   │   │
│   │   ├── arena.rs             #   IOSurface-backed SharedTensorArena
│   │   ├── arena_lifecycle.rs   #   Arena state lifecycle
│   │   ├── arena_pool.rs        #   Pooled arena acquire/release/reuse
│   │   ├── external_array.rs    #   MLX external array wrappers
│   │   ├── copy_ledger.rs       #   Per-operation copy audit trail
│   │   │
│   │   ├── compile_pipeline.rs  #   4-lane parallel relocation pipeline
│   │   ├── compile_state.rs     #   Compilation state machine
│   │   ├── compile_progress.rs  #   Progress tracking
│   │   ├── compute_ir.rs        #   Intermediate representation
│   │   ├── layout_compiler.rs   #   Memory layout planning
│   │   ├── fusion_region.rs     #   7 fusion topologies
│   │   ├── profile_compiler.rs  #   ExecutionPlacementProfile compiler
│   │   ├── placement_profile.rs #   CandidateClass, PlaceRegion
│   │   ├── hybrid_profile.rs    #   MLX/Core ML hybrid deployment spec
│   │   │
│   │   ├── coreml_bridge.rs     #   ObjC++ stateless prediction bridge
│   │   ├── coreml_state.rs      #   ObjC++ stateful prediction bridge (stubs)
│   │   ├── coreml_pipeline.rs   #   Core ML compilation pipeline
│   │   ├── coreml_audit.rs      #   Island-level deployment audit reports
│   │   ├── gpu_worker.rs        #   GPU transform worker thread
│   │   ├── transform_recipe.rs  #   Tensor transform recipes
│   │   ├── cpu_benchmarks.rs    #   CPU-side benchmark harness
│   │   │
│   │   ├── capability.rs        #   SharedTensorCapabilityReport (14 caps)
│   │   ├── receipts.rs          #   Arena/CoreML/Hybrid job receipts
│   │   ├── runtime_trace.rs     #   Execution trace instrumentation
│   │   ├── requalification.rs   #   Evidence-based profile promotion
│   │   │
│   │   ├── loader.rs            #   Safetensors loading
│   │   ├── model.rs             #   Model identity
│   │   ├── validator.rs         #   Binding validation against safetensors
│   │   ├── projection_tests.rs  #   Real-checkpoint projection parity tests
│   │   ├── operation_catalog.rs #   Operation catalog
│   │   ├── mlx_inventory.rs     #   MLX primitive inventory
│   │   ├── mlx_patch_register.rs#   Patch register for MLX compatibility
│   │   ├── cli.rs               #   CLI argument handling
│   │   └── errors.rs            #   SharedTensorArena structured errors
│   │
│   └── test/
│       └── compute_image_smoke.test.js  # N-API addon smoke test (Bun)
│
├── tools/
│   └── coreml-compiler/         # Hermetic Core ML compilation environment
│       ├── compile_region.py
│       ├── verify_artifact.py
│       ├── preflight.py
│       ├── requirements.lock
│       └── environment.json
│
├── mlx-rs-fork/                 # Git submodule — MLX Rust bindings fork
│   ├── mlx-sys/                 #   Low-level C bindings (mlx-c 0.6.0)
│   ├── mlx-rs/                  #   Safe Rust wrapper (v0.25.3-tribunus.1)
│   ├── mlx-macros/
│   ├── mlx-internal-macros/
│   ├── mlx-lm/
│   ├── mlx-lm-utils/
│   └── mlx-tests/
└── perf/
    └── test-suite.md            # Performance test definitions
```

## 7. Building

### Prerequisites

- Rust ≥ 1.82.0 (`rustup install 1.82.0`)
- macOS 15+ with Xcode Command Line Tools (`xcode-select --install`)
- The `mlx-rs-fork` submodule initialised:

```bash
git submodule update --init --recursive
```

### Build the N-API addon

```bash
cd compute-native
npm run build            # napi build --platform --release (aarch64-apple-darwin)
```

This compiles `lib.rs` as a `cdylib`, the three ObjC++ bridge files
(`coreml_arena.mm`, `coreml_exec.mm`, `coreml_state.mm`), links against
CoreML, CoreVideo, and IOSurface frameworks, and produces
`tribunus-compute-native.{platform}.node`.

### Build with Cargo (for testing)

```bash
cargo build --release -p tribunus-compute-native
```

The rlib target includes all Rust modules. The cdylib is the N-API addon.
Two binaries are produced:

- `tribunus-compute-worker` — real inference subprocess
- `tribunus-fake-worker` — deterministic test harness (12 fault modes)

### Build the worker binaries only (Linux)

```bash
cargo build --release --bin tribunus-compute-worker
```

On Linux the Objective-C bridge files are excluded by `#[cfg(target_os = "macos")]`
in `build.rs`, and `-Wl,-undefined,dynamic_lookup` is not applied.

### Debug build (N-API)

```bash
npm run build:debug       # napi build --platform (debug profile)
```

### Feature gates

- `private-development` — relaxes memory limits and disables qualification
  mode for local development. Enabled only for local builds, never in CI.

## 8. Testing

### Cargo unit tests

```bash
cargo test -p tribunus-compute-native
```

**Test inventory** (38 tests across 9 modules):

| Module | Tests | What it covers |
|---|---|---|
| `arena` | 11 | SharedTensorArena allocation, IOSurface creation, FP16 pixel buffer, MLX external array, Core ML round trip, stateful island, ping-pong, host memory, Gemma MLP prediction |
| `arena_lifecycle` | 3 | All valid state transitions + illegal transitions |
| `arena_pool` | 4 | Acquire, release, reuse, budget enforcement, max-per-key |
| `capability` | 2 | Capability detection + serde roundtrip |
| `worker_supervisor` | 9 | Job lifecycle, worker lifecycle, cancellation, graceful/forceful shutdown |
| `kv_cache` | 6 | Sliding eviction, global concatenation, concurrent access, clear |
| `errors` | 3 | Builder pattern, Display impl, variant enumeration |
| `hybrid_profile` | 3 | Serde roundtrip, validation, tensor flow analysis |
| `receipts` | 3 | Emitter builder, copy classification, serde roundtrip |

### Qualification tests (need hardware)

These require a physical Apple Silicon machine with Metal GPU. They are not
run in CI.

```bash
cargo test -p tribunus-compute-native -- --ignored
```

Qualification tests include:

- Quantised matmul fused vs dequantised parity (synthetic)
- Concurrent generation: 2×100 + 4×50 heavy quantised matmul
- Residency invariants: `handles=4→29→4`, `mlx_active=72128→71288`
- GPU canary latency measurement
- Core ML prediction with real model artifacts

### Fixture-based tests

The `projection_tests.rs` module loads real safetensors checkpoints from
`models/gemma4-12b-8bit/` and verifies quantised-matmul projection parity
against dequantise + matmul for every Gemma 4 12B projection tensor (Q, K,
V, O, gate, up, down, embedding, LM head). Run:

```bash
cargo test -p tribunus-compute-native all_real_projection_parity -- --nocapture
```

### N-API smoke test

```bash
cd compute-native
bun test
```

The `compute_image_smoke.test.js` test verifies that the N-API addon loads and
that `compileImage`, `readCompiledImage`, and `verifyCompiledImage` are
callable (they are expected to throw on empty temp directories — this is the
intended failure mode).

### Fake worker integration tests

The `tribunus-fake-worker` binary supports 12 fault modes for testing
supervisor resilience:

| Mode | Behaviour |
|---|---|
| `normal` | Clean handshake, generation, terminal ack |
| `identity-mismatch` | Wrong protocol version in HelloAck |
| `no-handshake` | Never responds to Hello |
| `model-load-hang` | Responds to Hello but never completes LoadModel |
| `slow-prefill` | Responds with prefill taking > timeout |
| `ignored-cancel` | Receives CancelGeneration but continues decoding |
| `heartbeat-loss` | Drops heartbeat frames after initial handshake |
| `malformed-frames` | Sends invalid JSON or wrong-length frames |
| `sequence-gap` | Sends Token after GenerationCompleted |
| `duplicate-terminal` | Sends two GenerationCompleted frames |
| `crash` | Exits with non-zero at configurable time |
| `memory-alloc` | Simulates memory exhaustion |

## 9. Runtime Qualification

### How qualification works

Qualification is a two-phase process:

1. **Compile-time qualification**: the compiler produces a
   `ModelExecutionPlan` with known tensor shapes, attention schedules, and
   quantisation metadata. This is structural — it proves the plan is valid but
   not that execution produces correct outputs.

2. **Runtime qualification**: an `ExecutionPolicy` with `unqualified: true`
   (the default) is replaced with a qualified policy after benchmarking.
   The `requalification.rs` module implements evidence-based promotion:
   - Collect `ProfileEvidence` observations (oracle tokens/s vs candidate
     tokens/s, boundary latency, device activity).
   - Call `promote_to_preferred()` when aggregate throughput meets or exceeds
     the oracle prediction.
   - Call `deprecate_profile()` when degradation is detected.

Qualification mode is triggered by `max_tokens == 0`, which is coerced to
`SAFE_ZERO_MAX_TOKENS = 8`. The prompt is capped to
`QUALIFICATION_PROMPT_TOKEN_CEILING = 64`, and a
`QUALIFICATION_WALL_CLOCK_DEADLINE = 30s` is enforced.

### What receipts contain

Receipts are serialised as JSON. Frozen schemas:

**ModelLoadReceipt** — `image_hash`, `storage_abi`, `runtime_abi`, `worker_pid`,
`model_open_ms`, `mapped_virtual_bytes`, `persistent_resident_bytes`,
`materialized_bytes`, `copied_bytes`, `tensor_binding_count`, `segment_count`,
`mlx_active_limit_bytes`, `mlx_cache_limit_bytes`, `rss_before_bytes`,
`rss_after_bytes`, `admission_estimate_json`.

**ArenaCreationReceipt** — `arena_id` (UUID), `generation` (monotonic counter),
`io_surface_id` (IOSurfaceGetID()), `logical_shape`, `physical_width`,
`physical_height`, `bytes_per_row`, `total_bytes`, `pixel_format`
(0x4C303068 = kCVPixelFormatType_OneComponent16Half), `profile`
(IOSurfaceFp16ContiguousV1), `created_at`.

**CoreMlPredictionReceipt** — `job_id`, `model_hash`, `island_id`,
`input_arena_id`, `input_io_surface_id`, `input_shape`, `output_arena_id`,
`output_io_surface_id`, `output_shape`, `output_backing_feature`, `duration_ms`,
`copy_classification`, `internal_coreml_staging`, `success`.

**HybridJobReceipt** — full job lifecycle: `job_id`, `session_id`,
`compute_image_hash`, `coreml_artifact_hash`, `macos_version`,
`capability_report_hash`, `state_id`, arena IDs, `lease_transitions`
(Vec<LeaseReceipt>), `coreml_predictions` (Vec<CoreMlPredictionReceipt>),
`state_mutations`, `total_duration_ms`, `application_copy_free`,
`finalizer_count`, `final_arena_state`, `success`, `error`.

**CopyClassification** values: `application_copy_free`, `copied_fallback`,
`materialized_layout_conversion`, `internal_coreml_staging_unknown`.

### How to read receipts

Receipts are produced by the receipt emitter module (`receipts.rs` engine_receipts)
and returned as JSON strings through the N-API bridge. The JS side can
parse and accumulate them:

```typescript
import compute from '@tribunus/compute-native';
const report = JSON.parse(compute.nativeCapabilityReport());
// report contains SharedTensorCapabilityReport with all 14 booleans
```

Receipts are the source of truth for:

- Proving whether a generation was `application_copy_free` (no app-level copies)
- Verifying IOSurface IDs and arena shapes match expectations
- Auditing worker memory consumption and MLX active-memory limits
- Detecting internal Core ML staging (always `true` / unknown for IOSurface path)

## 10. Security and Isolation Model

### Session Isolation

Every generation runs in its own `ControlSessionState` / `InferenceSessionState`
machine. Session IDs are UUIDs. The state machine enforces a strict transition
graph:

```
Created → Admitted → Submitted → PrefillRunning → Decoding → Completed
                                  └─ Cancelled ──┘
Any non-terminal state → Failed
```

Terminal states (Completed, Cancelled, Failed) reject all transitions. Failed
is reachable from any non-terminal state. The engine maintains a concurrent
map of active sessions and refuses operations on unknown or terminal sessions.

### Worker Boundaries

The model runtime executes in a separate OS subprocess. Communication is via
framed JSON IPC over stdin/stdout:

- **No tensor payload crosses the process boundary.** Only metadata, token IDs,
  and control frames are serialised.
- The `ProtocolValidator` enforces frame ordering, message-kind contracts,
  and `MAX_FRAME_SIZE_BYTES`.
- A watchdog thread monitors heartbeat frames (250 ms interval). Missing
  heartbeats trigger cascade: HealthCheckTimeout → SIGKILL → restart.
- The engine supports transparent worker restart (up to `restart_limit`).

### Memory Accounting

- **MLX active memory** queried via `mlx_active_memory()` FFI.
- **Process RSS** sampled via `task_info` on macOS (mach API).
- **Soft ceiling**: crossing `worker_rss_soft_ceiling_bytes` sends a
  `MemoryPressure` host command. If RSS does not fall below the reset
  threshold within the watchdog interval, the worker is killed.
- **Hard ceiling**: crossing `worker_rss_hard_ceiling_bytes` terminates the
  worker immediately.
- **Model admission**: `ModelAdmissionEstimate` computed at runtime-open time.
  The scheduler refuses admission if the estimate exceeds available budget.

### Cancellation

- `engine_cancel_generation()` sends `CancelGeneration` via the host command
  channel.
- The worker responds by setting an `AtomicBool` cancellation flag. The
  inference thread checks the flag between decode steps.
- Cancellation grace period (`cancellation_grace_period`, default 5s). After
  the grace period, the worker is SIGKILL'd and restarted.
- The stream emits `Cancelled` followed by no further events.

### Trust Boundaries

| Boundary | Trust Model |
|---|---|
| JS ↔ N-API | Untrusted caller. All inputs validated; arrays checked for bounds and dtype. |
| Host ↔ Worker | Worker is untrusted. Host validates all frames. Protocol version checked on handshake. |
| Worker ↔ MLX C | MLX C is trusted to not corrupt host memory. MLX C crash terminates worker. |
| Worker ↔ Core ML | Core ML is trusted for isolation (ANE/GPU memory is process-scoped). |
| Worker ↔ IOSurface | IOSurface is shared across processes only if explicitly exported (not in v0.1.0). |

See `SECURITY.md` for detailed threat model, incident response, and disclosure
process.

## 11. Versioning

### Kernel version

The kernel follows `vMAJOR.MINOR.PATCH`:

| Component | Current | Scheme |
|---|---|---|
| Kernel | v0.1.0 | Semantic versioning |
| N-API ABI | napi8 | Stable across kernel patch versions |
| MLX fork | 0.25.3-tribunus.1 | Tracks upstream MLX with `.tribunus.N` patch suffix |
| MLX C | 0.6.0 | Bound to `mlx-sys` version |
| Worker protocol | 1.0 | Incremented on incompatible frame changes |
| Storage ABI | copied-v0, mapped-no-copy-v1 | Frozen; new storage types get new ABI IDs |
| Capability names | Frozen | Adding a capability does not change existing names |

### Release Provenance

Every release records in the receipt chain:

```
Kernel version + MLX fork commit + MLX C version
+ macOS version + worker protocol version + capability report hash
```

The `HybridJobReceipt` captures `macos_version` and
`capability_report_hash` so that any generation can be traced to the exact
software and hardware configuration that produced it.

The dependency tuple is documented in `docs/compatibility.md`.

## 12. License

```
Copyright (c) 2026 Tribunus, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, version 3 only.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
```

The kernel is licensed under **AGPL-3.0-only**. Both the `tribunus-compute-native`
crate and the `tribunus-mlx-rs` fork (MIT / Apache-2.0) are linked together in
a single binary — the combined work is distributed under AGPL-3.0 as required
by the AGPL's "Corresponding Source" provisions.

**Linking implications:** Any application that links against the N-API addon
(including via `require('@tribunus/compute-native')`) must comply with AGPL-3.0
terms. If you are considering using the kernel in a proprietary or closed-source
product, please contact Tribunus, Inc. about commercial licensing options.

**Third-party components:**
- `mlx-rs` / `mlx-sys`: MIT OR Apache-2.0 (used under fork at `mlx-rs-fork/`)
- `napi-rs` / `napi-derive`: MIT
- `safetensors`: Apache-2.0
- Various Rust crates under MIT, Apache-2.0, BSD, or ISC (see `Cargo.lock`)

## 13. Roadmap

| Milestone | Scope | Target |
|---|---|---|
| **v0.2 — KV cache integration** | Prefill + cached decode; cached vs uncached parity over several decode steps | Near-term |
| **v0.3 — Mapped no-copy residency** | mmap segment files, MLX external arrays over mapped memory, flat residency plateau | Near-term |
| **v0.4 — Core ML state qualification** | Repeated mutation, two-session isolation, concurrency rejection, cancellation | Near-term |
| **v0.5 — Streaming + cancellation** | Bounded event channels, token emission, EOS, supervisor lifecycle, cancellation at boundaries | Near-term |
| **v1.0 — Production baseline** | Full 48-layer execution, all receipts, stress testing, worker restart, memory ceilings | Medium-term |
| **v1.1 — Core ML MLP placement** | Boundary latency benchmark (MLX eval → arena → Core ML → arena → MLX consume) | Medium-term |
| **v1.2 — Hybrid execution** | Profile-compiler-driven MLX/Core ML hybrid deployment | Medium-term |
| **v2.0 — ANE autotuning** | Automatic island detection, placement optimisation, video-frame formats | Medium-term |
| **v2.x — Cross-process IOSurface** | Shared IOSurface transport across process boundaries | Future |

### Currently blocked items

- **Full 48-layer ComputeImage execution**: requires Gemma 4 12B safetensors
  checkpoint (not available in CI). The structural path is wired and the
  config-driven layer_types scheduler works — execution at scale awaits
  weights.
- **Core ML stateful prediction**: `coreml_state.mm` bridge compiles but
  stateful prediction crashes because the MLState bridge implementation is
  stubbed. Deferred to Phase 12.
- **Core ML compiler preflight**: requires a pinned Python 3.13 ARM64
  environment with coremltools 9.0 binary wheel. The hermetic toolchain at
  `tools/coreml-compiler/` enforces this contract.

## 14. Contributing

See `CONTRIBUTING.md` for:

- Development workflow and branch conventions
- Commit message format
- Code review guidelines
- Test expectations (unit tests required for all new subsystems)
- Performance gate requirements
- MLX compatibility fork maintenance

Key points:

- All new subsystems must have unit tests before merging.
- Changes that touch the MLX fork or worker protocol must maintain backward
  compatibility or bump the protocol version.
- The capability report is a frozen ABI: new capabilities are added, never
  removed or renamed.
- Receipt schemas are frozen per kernel major version. New fields are added
  with `#[serde(default)]`.
- Formatting is enforced by `cargo fmt`; linting by `cargo clippy --all-targets`.
