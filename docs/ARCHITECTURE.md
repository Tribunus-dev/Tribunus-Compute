# Tribunus Compute Kernel — Architecture

## 1. Overview

The Tribunus Compute Kernel is a Rust-native, multi-process inference runtime that sits at the bottom of the Tribunus stack. It owns the MLX runtime, N-API bridge, worker subprocesses, model store, KV cache, compile pipeline, and the IOSurface-backed SharedTensorArena for zero-copy MLX/Core ML boundary transfers.

**Runtime boundaries:**
- **Host process** (Node.js via N-API, or SDK via IPC): session management, policy enforcement, worker supervision.
- **Worker subprocess** (spawned per model image): owns MLX arrays, KV cache, generated tokens, arenas. No tensor payload crosses the IPC boundary — only token IDs and metadata.
- **Core ML** (in-process via ObjC++ bridge): stateless and stateful prediction on the host or worker side, sharing IOSurface-backed FP16 buffers.

## 2. Entry Points

### 2.1 N-API Surface

`lib.rs` exposes ~20 `#[napi]` functions to the Tribunus host (Node.js / Bun). Array handles are `i64` indices into a global `ArrayRegistry`.

| Export | Purpose |
|---|---|
| `createArrayF32`, `createArrayRaw` | Create MLX arrays from JS buffers |
| `matmul`, `multiply`, `add` | Scalar math |
| `loadSafetensors`, `inspectSafetensors` | Model weight loading |
| `compileImage`, `readCompiledImage`, `verifyCompiledImage`, `validateFromMetadata` | ComputeImage lifecycle |
| `runFullModelFromImage` | Execute 48-layer model, return next token |
| `gemmaForward`, `gemmaSampleGreedy` | Direct Gemma inference |
| `nativeCapabilityReport` | Runtime capability detection |
| `mlxActiveMemory`, `mlxClearCache` | Memory telemetry |
| `arrayDataF32`, `arrayEval`, `arrayNbytes`, `arrayShape`, `arraySize` | Array introspection |
| `drainArrays`, `freeArray`, `handleCount` | Resource lifecycle |
| `gemma412BConfig`, `parseConfigOnly` | Config utilities |
| `detectDefaultDevice` | Device detection |

### 2.2 Worker Protocol

The worker protocol (`worker_protocol.rs`) is a length-prefixed JSON IPC over stdin/stdout. Protocol version `V1_0`. Max frame size 1 MiB.

**Commands** (host → worker):

| Command | Payload |
|---|---|
| `Hello` | `worker_instance_id`, `protocol_version` |
| `LoadModel` | `model_path`, `execution_plan_hash` |
| `StartGeneration` | `prompt_tokens`, `session_id`, `sampler_config`, `max_tokens`, `kv_state_keys` |
| `CancelGeneration` | `job_id` |
| `UnloadModel` | — |
| `Ping` | — |
| `Shutdown` | — |

**Events** (worker → host):

| Event | Payload |
|---|---|
| `HelloAck` | `worker_version`, `device_info` |
| `ModelLoaded` | `image_hash`, `memory_estimate` |
| `PrefillStarted` | `job_id` |
| `Token` | `job_id`, `token_id`, `index`, `log_prob` |
| `StepMetrics` | `job_id`, `step_ms`, `mlx_active_bytes` |
| `Progress` | `job_id`, `phase`, `layer_current`, `layer_total` |
| `GenerationCompleted` | `job_id`, `total_tokens`, `total_ms`, `stop_reason` |
| `GenerationCancelled` | `job_id` |
| `MemoryPressure` | `rss_bytes` |
| `Heartbeat` | `uptime_secs`, `rss_bytes`, `mlx_active_bytes`, `active_jobs` |
| `Error` | `code`, `message`, `retryable`, `worker_terminated` |
| `Goodbye` | `reason` |

**Frame format:** `[4-byte LE length][JSON payload bytes]`

## 3. Supervisor

The `WorkerSupervisor` (`worker_supervisor.rs`, ~2500 lines) manages the worker subprocess lifecycle.

### 3.1 Process Lifecycle

```
Spawn → Handshake → LoadModel → Ready
                                  │
                    ┌─────────────┼─────────────┐
                    ↓             ↓             ↓
              StartGeneration  Ping         UnloadModel
                    │                          │
                    ↓                          ↓
              [Prefill→Decode→Done]       ModelReleased
                    │
                    ↓
              CancelGeneration → GenerationCancelled
```

### 3.2 Thread Model

The supervisor spawns the worker as a child process and runs three internal threads:

- **Event reader thread**: owns `stdout` read. Parses frames, validates with `ProtocolValidator`, routes events to the host through an `mpsc` channel.
- **Command writer thread**: owns `stdin` write (`BufWriter`). Receives commands from the host via `mpsc` and serializes to framed JSON.
- **Watchdog thread**: polls RSS every ~150 ms. Enforces soft/hard ceilings. Detects stuck workers (no heartbeat for `watchdog_timeout`).

### 3.3 Health Checks

- **Heartbeat**: worker emits `Heartbeat` every 250 ms with RSS, MLX active memory, and active job count.
- **Ping/Pong**: supervisor sends `Ping` on idle workers; expects `Heartbeat` within timeout.
- **RSS ceilings**: `rss_soft_ceiling_bytes` triggers `MemoryPressure` event. `rss_hard_ceiling_bytes` triggers SIGKILL.
- **Worker crash**: detected via `Child::wait()` on the process handle. Automatic restart with exponential backoff (configurable `restart_limit`).

### 3.4 Cancellation Propagation

1. Host calls `CancelGeneration(job_id)` through the N-API or IPC surface.
2. Supervisor sends `CancelGeneration` frame to worker.
3. Worker\'s decode loop checks `AtomicBool` cancellation flag each iteration.
4. If worker acknowledges within `cancellation_grace_period` (2 s), supervisor receives `GenerationCancelled` event.
5. If grace period expires, supervisor sends SIGKILL and starts worker restart sequence.

## 4. Session Runtime

`session.rs` defines two state machines.

### 4.1 ControlSessionState (host-side)

```
Created → Admitted → Submitted → PrefillRunning → Decoding → Completed
                                                              Cancelled
                                                              Failed
```

- `Admitted`: passed budget checks (token ceiling, deadline, RSS headroom).
- `Submitted`: dispatched to worker, awaiting first event.
- `PrefillRunning` → `Decoding`: state transitions driven by worker events.

Transitions to `Completed`, `Cancelled`, or `Failed` are terminal.

### 4.2 InferenceSessionState (worker-side)

```
Created → Prefill → Decode → Done
```

The worker-side session owns:
- KV cache (per-layer sliding windows or global concat buffers)
- Cancellation flag (`AtomicBool`)
- Sampler config (temperature, top_k, top_p, seed)
- Generated token buffer

The host-side session owns **no** MLX arrays and **no** KV cache — those belong exclusively to the worker process.

## 5. Execution Planner

`config.rs` builds the execution plan from model configuration.

### 5.1 Model Execution Plan

```
ModelExecutionPlan {
    prologue: ProloguePlan      // embedding lookup
    layers: Vec<LayerPlan>      // 48 decoder layers for Gemma 4 12B
    epilogue: EpiloguePlan      // final norm, output projection, softcapping
}
```

Each `LayerPlan` carries:
- `dim`, `hidden_dim`, `ffn_dim` — tensor dimensions
- `num_kv_heads`, `num_attention_heads`, `head_dim` — attention geometry
- `attention_kind` — `Gqa` (grouped-query attention)
- `rope_theta` — RoPE base frequency
- `quant_meta` — group size, scales/zeros dtype
- `segment_assignments` — tensor → ComputeImage segment mapping

### 5.2 Supporting Compilation Pipeline

- **Profile compiler** (`profile_compiler.rs`): compiles operation audit → `ExecutionPlacementProfile` with M1 baseline partitions.
- **Layout compiler** (`layout_compiler.rs`): plans weight/activation segment layouts, zero-copy eligibility, offset alignment.
- **Fusion regions** (`fusion_region.rs`): 7 hardcoded Gemma 4 12B fusion regions (QKV, gate+up, SiLU+mul, down_proj, RMS norm+residual, self-attn megakernel).
- **Placement profiles** (`placement_profile.rs`): M1 latency/throughput/constrained/fallback deployment profiles.
- **Hybrid profiles** (`hybrid_profile.rs`): MLX regions, Core ML stateful islands, boundary tensors, execution order, fallback policy.

## 6. MLX Executor

`mlx_executor.rs` and `profiled_executor.rs` run operations on MLX.

### 6.1 Function Execution

- Operations are dispatched to an explicit `Device`+`Stream` pair (GPU or CPU).
- `run_gpu_canary`: measures quantized matmul CPU vs GPU latency to validate GPU availability.
- Array lifecycle: create (from data or external storage) → eval (synchronize to device) → free.

### 6.2 Memory Telemetry

- `mlx_active_memory_bytes()` — MLX Metal active bytes.
- `mlx_cache_memory_bytes()` — MLX Metal cache bytes.
- `mlx_peak_memory_bytes()` — peak since last clear.
- `mlx_clear_cache()` — flush MIL compile cache.
- `set_mlx_memory_limit()`, `set_mlx_cache_limit()` — budget configuration.

### 6.3 External Arrays

`external_array.rs` wraps `mlx_array_new_data_managed` for no-copy consumption of IOSurface-backed memory. The external data pointer must remain valid until MLX evaluates the array.

## 7. Core ML Executor

The Core ML bridge consists of three ObjC++ files compiled by `build.rs`.

### 7.1 Stateless Prediction (`coreml_exec.mm`)

- Loads a compiled `.mlmodelc` bundle.
- Accepts input via `MLMultiArray` backed by IOSurface pixel buffers.
- Returns output via `outputBackings` (IOSurface-backed `MLMultiArray`).
- Feature validation (`_assertFeature`) checks dtype and shape before every prediction.
- Error codes: `-11` input mismatch, `-12` output mismatch, `-2` memory error, `-10`/`-20` ObjC exception.

### 7.2 Stateful Prediction (`coreml_state.mm`)

- Compiles but not runtime-qualified. Stateful prediction uses `MLState` API.
- When fully implemented: operator model islands execute on ANE with state fed from MLX via IOSurface.

### 7.3 Arena Bridge (`coreml_arena.mm`)

- Allocates IOSurface-backed FP16 contiguous storage.
- Creates `CVPixelBuffer` from IOSurface for Core ML input.
- Returns `ArenaInfo` C-compatible struct with IOSurface ID, pixel format, logical/physical dimensions.

## 8. Shared Tensor Arena

`arena.rs` + `arena_lifecycle.rs` + `arena_pool.rs` manage the IO surface-backed zero-copy boundary storage.

### 8.1 Arena Lifecycle

```
Uninitialized → Allocated → BoundToIOSurface → MappedByMLX → OwnedByCoreML → Released → Recycled
```

Each transition is validated; illegal transitions are rejected at runtime.

### 8.2 Data Flow (IOSurface FP16 canonical path)

1. MLX writes arena A through external array (`mlx_array_new_data_managed`).
2. Evaluation completes — MLX data is on GPU.
3. Ownership transfers to Core ML — arena A becomes Core ML input `MLMultiArray`.
4. Core ML reads via `MLMultiArray(pixelBuffer:shape:)`.
5. Core ML writes arena B through `outputBackings`.
6. Ownership transfers to MLX — arena B becomes MLX external array.
7. MLX consumes via external array.

Copy classification: `application_copy_free = true` — no application-level copies.

### 8.3 Arena Pool

Bounded per-key pooling: acquire (get or create), release (return to pool), reuse (avoid allocation). Max-per-key budget enforced. Used for repeated inference cycles to avoid IOSurface allocation overhead.

## 9. KV Cache

`kv_cache.rs` provides per-layer key-value storage with two eviction strategies.

### 9.1 Sliding Window

- Fixed-capacity ring buffer per layer.
- Oldest entries evicted when window is full.
- Used for local attention layers (layers 0-41 in Gemma 4 12B).

### 9.2 Global Concat

- Unbounded concatenation of all past keys/values.
- Used for global attention layers (layers 42-47 in Gemma 4 12B).

### 9.3 Operations

- **Commit/rollback**: transactional semantics — reads see committed state only.
- **Byte accounting**: per-entry byte tracking for memory budgeting.
- **Concurrent access**: `RwLock`-protected per-layer storage.

### 9.4 Quantization

K/V tensors stored in quantized format (grouped affine quantization). Quantized matmul against incoming query with fused dequantize+matmul path.

## 10. Mapped Images

`mapped_image.rs` provides no-copy segment access via `mmap`.

### 10.1 Storage Modes

- **Copied-v0** (baseline): CPU-allocated, runtime-ready image. Segments read into heap.
- **Mapped-no-copy-v1** (in progress): `mmap`\'d segment files with `TensorEntry` offset tables. Designed for direct Metal buffer wrapping.

### 10.2 Segment Lifecycle

```
Absent → Prefetched → Bound → InFlight → Retired
```

`ResidencyManager` (`residency.rs`) enforces budgeted admission and safety reserve.

## 11. Streaming

`streaming.rs` implements bounded event channels for token-by-token emission.

- **Event channel**: `Mutex<VecDeque>` + `Condvar`, bounded capacity.
- **Phases**: `Started` → token events → `Done` / `Error` / `Cancelled`.
- **Cancellation**: `AtomicBool` checked per iteration by decode loop. Cancellation flag set by supervisor, picked up by worker.
- **Flow control**: channel capacity prevents unbounded buffering.

## 12. Receipts

`receipts.rs` and `engine_receipts.rs` emit structured JSON receipts for every lifecycle event.

### 12.1 Receipt Types

| Receipt | Fields |
|---|---|
| `ModelLoadReceipt` | `image_hash`, `worker_pid`, `model_open_ms`, `mapped_virtual_bytes`, `persistent_resident_bytes`, `segment_count`, `mlx_active_limit_bytes`, `admission_estimate` |
| `ArenaCreationReceipt` | `arena_id`, `io_surface_id`, `logical_shape`, `physical_dimensions`, `bytes_per_row`, `pixel_format` |
| `CoreMlPredictionReceipt` | `model_hash`, `input_arena_id`, `output_arena_id`, `io_surface_ids`, `duration_ms`, `copy_classification` |
| `HybridJobReceipt` | Full job lifecycle: lease transitions, Core ML predictions, state mutations, finalizer count, combined copy classification |
| `GenerationReceipt` | `job_id`, `prompt_tokens`, `output_tokens`, `total_ms`, `tokens_per_second`, `stop_reason`, `mlx_peak_bytes` |
| `EngineError` | Stable kebab-case codes (`model-busy`, `worker-crashed`, `deadline-exceeded`, etc.), retryability, worker-termination flag |

### 12.2 Copy Classification Taxonomy

| Class | Description |
|---|---|
| `zero-copy-mapped` | mmap\'d segment → MLX external array, no copy |
| `io-surface-shared` | IOSurface-backed, shared between MLX and Core ML |
| `explicit-cpu-staging` | Copied through CPU-visible staging buffer |
| `internal-coreml-staging-unknown` | Core ML internal copy of unknown kind |
| `application-copy-free` | No application-level copy in the round trip |

## 13. Cancellation

Cancellation propagates through three layers:

1. **Host API**: SDK caller invokes `Engine::cancel(job_id)`.
2. **Supervisor → Worker IPC**: `CancelGeneration` frame sent over stdin.
3. **Worker decode loop**: `AtomicBool` checked each iteration.

After `cancellation_grace_period` (2 s in qualification mode), the supervisor sends SIGKILL. Resource cleanup: arena pool releases, KV cache dropped, file handles closed.

## 14. Memory Model

### 14.1 Ownership Rules

- **MLX arrays**: owned by the worker process. Not shared across processes.
- **KV cache**: owned by the worker\'s `InferenceSession`. Dropped on session termination.
- **IOSurface arenas**: allocated by the host or worker. Reference-counted by the kernel. Shared between processes via IOSurface ID.
- **External arrays**: wrap externally-owned memory. The memory must outlive the MLX array.
- **No-copy paths**: MMAP segments → MLX external arrays. IOSurface → `CVPixelBuffer` → `MLMultiArray`.

### 14.2 Memory Budgeting

| Budget | Mechanism |
|---|---|
| MLX Metal active | `set_mlx_memory_limit()` |
| MLX Metal cache | `set_mlx_cache_limit()` + `clear_mlx_cache()` |
| Worker RSS soft | `rss_soft_ceiling_bytes` → `MemoryPressure` event |
| Worker RSS hard | `rss_hard_ceiling_bytes` → SIGKILL |
| Model admission | `model_admission_ceiling_bytes` → reject load |
| Request budget | `prompt_token_ceiling`, `output_token_ceiling`, `request_deadline` |
| System reserve | `physical_memory_reserve_bytes` subtracted from machine total |

## 15. Thread Model

### 15.1 Tokio Runtime

The host process uses a Tokio multi-threaded runtime for:
- Worker supervisor (spawn, health check timers, restart backoff).
- Event routing from workers to host.
- Concurrent generation requests across sessions.

### 15.2 Worker Threads

Within the worker subprocess:

- **Command thread**: reads stdin, parses frames, validates protocol.
- **Inference thread**: owns `LoadedProfiledModel`, runs prefill+decode loop.
- **Heartbeat thread**: emits telemetry every 250 ms.

### 15.3 MLX Internal Threads

MLX manages its own Metal command queues and GPU streams. The kernel does not create or manage MLX threads directly — it dispatches operations to streams via the `mlx-rs` API.

### 15.4 Core ML

Core ML prediction runs on the calling thread (typically the inference thread). ANE scheduling is managed by Core ML internally.
