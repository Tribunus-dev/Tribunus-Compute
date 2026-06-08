# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes (development)  |

Security fixes target the current minor version. There is no LTS release channel
yet. Users building from `main` receive fixes on the next commit.

---

## 1. Vulnerability Reporting

If you discover a security vulnerability in Tribunus Compute — including but not
limited to crafted ComputeImage files that bypass validation, remote-code-execution
vectors via the worker protocol, memory exhaustion in the ObjC++ bridge, or supply-chain
compromise of the mlx-rs fork — please report it privately.

- **Email:** `security@tribunus.io`
- **PGP key:** Available via `openpgp4fpr:9E3B4A2C1D8F6E7A` (keys.openpgp.org).
  Fingerprint `9E3B 4A2C 1D8F 6E7A`.
- **Response SLA:** Initial acknowledgement within **48 hours**. A triage
  assessment within **5 business days**.
- **Process:** We follow coordinated disclosure. We will work with you to
  understand the impact, prepare a fix, and release. We request a 90-day
  embargo from the date of the fix release before full public disclosure.

Please include:

1. A minimal reproduction case (model file, config, or protocol trace).
2. Affected component and version.
3. Impact description (denial of service, information disclosure, remote code
   execution, etc.).
4. Any proposed fix or mitigation, if available.

---

## 2. Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                     HOST / SDK CALLER                            │
│  (trusted for session creation; session isolation assumes        │
│   cooperative host; no sandbox between caller and engine)        │
└────────────────┬────────────────────────────────────────────────┘
                 │  N-API (napi-rs v3) ──── framed IPC (1 MiB max)
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    WORKER SUPERVISOR (host Rust)                   │
│  • Spawns worker subprocess per model image                       │
│  • Validates every IPC frame (sequence, version, size, worker ID) │
│  • Enforces policy: RSS limits, heartbeats, deadlines, restarts   │
│  • Routes validated events to SDK caller                          │
│  • Process boundary: crash isolation, no shared address space     │
└────────────────┬──────────────────────────────────────────────────┘
                 │  4-byte LE length prefix + JSON frame (stdin/stdout)
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    WORKER SUBPROCESS                               │
│  (trusted once spawned; protocol is the trust boundary)           │
│  • Owned: MLX arrays, KV cache, generated tokens, arenas         │
│  • No tensor payload crosses the IPC channel                      │
│  • Cancellation via AtomicBool + SIGKILL grace period             │
│  • Supervisor enforces RSS ceiling; hard kill on violation        │
└───────────┬───────────────────────────────────────────────────────┘
            │  mlx-rs (Rust FFI)    ObjC++ bridge (compiled from source)
            ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│       MLX C Library       │   │  Core ML / IOSurface (macOS kernel)  │
│  (trusted; system-        │   │  (trusted; OS-provided, sandboxed)   │
│   installed; linked via   │   │                                      │
│   mlx-sys FFI)            │   │  • IOSurface-backed FP16 arenas      │
│                           │   │  • CVPixelBuffer for GPU texture I/O │
│                           │   │  • MLModel prediction via Core ML    │
└───────────────────────────┘   └──────────────────────────────────────┘
```

### Trusted Components

| Component | Reason |
|-----------|--------|
| **Host/SDK caller** | Session creation API. No sandbox between caller and engine. Session isolation assumes the host does not tamper with its own session state. |
| **Worker subprocesses** | Once spawned, the worker is trusted to execute inference. The IPC protocol is the trust boundary — validated frames carry only metadata and token IDs, never tensor payloads. |
| **MLX C library** | System-installed via the mlx-rs fork. Trusted for numerical computation and Metal GPU memory management. |
| **Core ML / IOSurface** | macOS kernel-provided frameworks (CoreML, CoreVideo, IOSurface). Trusted for model prediction and pixel-buffer management. |
| **Native bridge code** | `coreml_exec.mm`, `coreml_arena.mm`, `coreml_state.mm` — compiled from source by `build.rs` using `cc` crate. No runtime dependency injection. |

### Untrusted Components

| Component | Mitigation |
|-----------|------------|
| **ComputeImage artifacts** (manifest.json + segment files) | Full validation at open time: manifest hash, segment SHA-256, receipt cross-check, storage ABI conformance, tensor layout constraints. |
| **Model weights** (safetensors shards) | Binding validation against the execution spec: header parsing, dtype/shape cross-checks, unexpected tensor classification. No array data is materialized during validation. |
| **Model config** (config.json) | Config validation during `ExecutionSpec` construction — architecture normalization, quantization meta verification, unsupported feature detection. |

---

## 3. Model-File Trust

Model weights enter the system as safetensors shards. The validation pipeline
operates on metadata only — no tensor data is loaded into MLX during validation.

### Safetensors Header Validation

`validator.rs` [`validate_bindings`] performs:

1. **Shard discovery** — enumerates all `.safetensors` files in the model directory
   (`discover_shards`).
2. **Header parsing** — reads each safetensors header without materializing tensor
   data (`read_safetensors_header`).
3. **Binding cross-reference** — every tensor declared in the `ExecutionSpec` is
   looked up by name in the safetensors index. Mismatches in shape or dtype are
   reported per-tensor with pass/fail status.
4. **Unexpected tensor classification** — tensors present in the files but not
   referenced by the spec are classified (multimodal wrappers, optimizer state,
   etc.). Unknown tensors are flagged.
5. **Executable verdict** — a boolean `executable` flag gates further processing.
   No model can be compiled into a ComputeImage without passing validation.

### Config Validation

`config.rs` validates the model's `config.json` during `ExecutionSpec` construction:

- Architecture must be a supported type (e.g., `gemma3`, `gemma4`).
- `AttentionKind` (mla, gqa, mha) must be consistent with the tensor layout.
- `QuantizationMeta` group sizes must not overflow logical element counts.
- Unsupported features are collected into `unsupported_features` and gate
    compilation with a clear error.

### Malformed Input Rejection

- Zero-length tensor entries in safetensors headers produce a validation failure.
- Shape mismatches between the spec and the on-disk header are caught per-binding.
- Dtype mismatches (e.g., spec expects `f16` but file declares `f32`) are caught.
- `SourceIdentity` captures shard hashes at compile time and cross-references
    them at load time.

---

## 4. ComputeImage Integrity

A ComputeImage is the compiled runtime artifact containing:

```
image_dir/
  manifest.json    — architecture, tensor table, aliases, residency plan
  receipt.json     — compile receipts, segment hashes, provenance
  segment_000.bin  — execution-ordered tensor bytes
  segment_001.bin  — ...
```

### Manifest Validation

`compute_image.rs` [`CompiledImageReader::verify`] runs at open time:

1. **Manifest hash** — `manifest.image_hash` is recomputed from the serialized
   manifest and compared against the stored value. Mismatch → error.
2. **Receipt cross-check** — `receipt.complete_image_hash` must equal
   `manifest.image_hash`. Every segment in the manifest must have a corresponding
   receipt entry with matching `id`, `filename`, `sha256`, and `byte_size`.
3. **Segment hash verification** — every segment file is read from disk and its
   SHA-256 is computed. Each hash must match the manifest's `segment.sha256`.
   Segment bytes are read only during verification and dropped immediately after.

### Storage ABI Validation

For `mapped-no-copy-v1` images, additional checks enforce:

- Segment alignment (`alignment_bytes` must be a multiple of 4096).
- Tensor offset alignment (must be a multiple of 16 bytes).
- `storage_dtype` must be in the supported set: `U8`, `I8`, `F16`, `BF16`, `F32`, `U32`.
- Quantized tensor dimensions must not overflow logical element counts.

### Receipt Chain

`CompileReceipt` captures at compile time:

- `source_config_hash` — hash of the source config.json.
- `source_shard_hashes` — SHA-256 of every safetensors shard used.
- `complete_image_hash` — deterministic hash of the full compiled image.
- `segment_hashes` — per-segment `SegmentReceipt` with SHA-256 and byte size.
- `byte_provenance` — per-tensor provenance tracing source SHA-256 → emitted SHA-256,
  with a `preserved_byte_for_byte` flag.

### Image Publishing

`publish_image` writes a `.publishing` marker during atomic rename of the staging
directory to its final destination. On failure, a `.failed` marker is left for
inspection. This prevents partial images from being served.

---

## 5. Native-Code Risk

Tribunus Compute runs three categories of native code:

### Rust (safe, compiled from source)

All Rust code is in `compute-native/src/`:

- **`worker_protocol.rs`** — framed JSON IPC with fixed-size length prefix.
  Maximum frame size: 1 MiB (`MAX_FRAME_SIZE_BYTES`). Sequence numbers are
  validated monotonically, protocol version is checked, worker instance IDs
  are enforced by the `ProtocolValidator` state machine.
- **`worker_supervisor.rs`** — decomposed into independently locked components:
  `WorkerProcessControl` (Mutex on the Child handle, killed atomic),
  `WorkerCommandWriter` (Mutex on BufWriter), `WorkerEventReader` (lock-free,
  owned by event-reader thread), `ActiveRequestRegistry` (Mutex on request map),
  `ActiveRequest` (per-field synchronization with atomic terminal flag).
- **`engine_policy.rs`** — immutable policy surface. Budget resolution is a pure
  function with no side effects. `DeadlineGuard` uses an injectable clock.
- **`session.rs`** — two state machines: host-side `ControlSessionState` and
  worker-side `InferenceSessionState`. Terminal states are irreversible.
  `InferenceSession` owns the KV cache, cancellation flag (`AtomicBool`), and
  generated tokens. The host-side session owns **no** MLX arrays and **no**
  KV cache — those belong exclusively to the worker.
- **`engine_error.rs`** — typed error taxonomy with 19 machine-readable error
  codes. Every error carries a retryability flag and a worker-termination flag.
  The `failure_funnel` ensures `worker_terminated` is set correctly.
- **`validator.rs`** — safetensors header-only parsing. No tensor arrays are
  materialized during validation.
- **`config.rs`** — `ExecutionSpec` validation during compilation plan construction.

### Objective-C++ Bridge (compiled from source by `build.rs`)

- **`coreml_exec.mm`** — stateless Core ML prediction path. Feature validation
  (`_assertFeature`) checks model description against the caller's expected
  dtype and shape before every prediction. Returns negative error codes on
  mismatch (`-11` input mismatch, `-12` output mismatch, `-2` memory error,
  `-10`/`-20` Objective-C exception).
- **`coreml_arena.mm`** — IOSurface-backed FP16 arena allocation.
- **`coreml_state.mm`** — stateful Core ML prediction (compiled; not
  runtime-qualified).
- Compiled with `-fobjc-arc` (automatic reference counting) and `-fblocks`.
  Linked against CoreML, CoreVideo, and IOSurface frameworks.

### Metal Shaders (MLX internal)

MLX manages Metal Shading Language kernels through the `mlx-sys` FFI. These are
not authored in this repository. GPU memory is bounded by `mlx_active_memory_limit`
and `mlx_cache_limit` set via `mlx-rs` APIs.

### Build-Time Security

- `build.rs` compiles the ObjC++ bridge with `cc` crate. No prebuilt binaries
  are vendored.
- On macOS, the linker uses `-undefined,dynamic_lookup` so N-API symbols are
  resolved at runtime by Node.js — no machine-specific `libnode` bindings are
  baked into the `.node` binary.
- Release profile uses LTO, symbol stripping, and single codegen unit.

---

## 6. Resource Exhaustion

### Memory Accounting

| Mechanism | Description |
|-----------|-------------|
| `worker_rss_soft_ceiling_bytes` | RSS monitored by watchdog at ~150 ms intervals. When exceeded, the supervisor sends `MemoryPressure` to the worker. |
| `worker_rss_hard_ceiling_bytes` | Hard limit. Worker is SIGKILL'd if RSS exceeds this value. |
| `mlx_active_memory_limit_bytes` | MLX Metal active memory budget, set via `set_mlx_memory_limit()`. |
| `mlx_cache_limit_bytes` | MLX Metal cache limit, set via `set_mlx_cache_limit()`. Can be cleared (`clear_mlx_cache()`). |
| `model_admission_ceiling_bytes` | Rejects model loading if estimated peak exceeds this ceiling. |
| `physical_memory_reserve_bytes` | Reservation for guard pages, signal handling, and process overhead. |
| `soft_pressure_reset_threshold_bytes` | RSS must fall below this to clear a soft-pressure episode. |

Memory is monitored per-worker via `read_process_rss()` (procfs on Linux,
`proc_pid_rusage` on macOS). MLX memory is queried through `mlx-rs` FFI calls:
`mlx_active_memory_bytes()`, `mlx_cache_memory_bytes()`, `mlx_peak_memory_bytes()`.

### Arena Limits (IOSurface)

IOSurface-backed arenas (`coreml_arena.mm`) are allocated at fixed FP16 sizes
and managed through the arena pool. The pool supports acquire/release/reuse
semantics with a maximum-per-key budget. No unbounded arena growth is possible.

### Session Quotas

Each generation request is admitted through `resolve_generation_budget()`:

- `prompt_token_ceiling` — oversize prompts are rejected before submission.
- `output_token_ceiling` — output is clamped to this maximum.
- `request_deadline` — hard wall-clock deadline. The `DeadlineGuard` is checked
  by the decode loop. Expiry produces `DeadlineExceeded`.

### Cancellation

- The worker's `InferenceSession` has an `AtomicBool` `cancellation_flag` checked
  by the decode loop.
- The supervisor sends `HostCommand::CancelGeneration` over IPC.
- After `cancellation_grace_period` (2 s in qualification mode), the supervisor
  sends SIGKILL if the worker has not acknowledged cancellation.
- `cancellation_funnel` produces a structured `Cancelled` error with the reason.

### Timeout

| Timeout | Duration | Action |
|---------|----------|--------|
| Handshake (`HANDSHAKE_TIMEOUT`) | 5 s | `WorkerHandshakeFailed` |
| Model load (`model_load_timeout`) | 120 s | `WorkerUnresponsive` |
| Request deadline (`request_deadline`) | 30 s (qual) | `DeadlineExceeded` |
| Heartbeat (`worker_heartbeat_timeout`) | 2 s (qual) | `WorkerUnresponsive` |
| Shutdown grace (`SHUTDOWN_TIMEOUT`) | 10 s | SIGKILL then `WorkerCrashed` |

### Restart Limit

`restart_limit: 3` — after this many worker restarts within a session, the
supervisor produces `WorkerRestartLimitExceeded` and does not retry.

### Stderr Diagnostics

Worker stderr is captured in a ring buffer (`stderr_diagnostic_ceiling_bytes`,
default 64 KiB). The ring prevents unbounded memory consumption from verbose
worker output.

---

## 7. Worker Isolation

### Process Boundary

Every inference worker runs as a **separate OS process**, spawned by the
`WorkerSupervisor` via `std::process::Command`. The worker binary is
`tribunus-compute-worker`.

- **No shared address space** — the host cannot read worker memory and vice
  versa (modulo OS-level debugging facilities).
- **Crash containment** — a worker segfault, abort, or SIGKILL does not
  affect the host process. The supervisor detects the exit via `try_wait()`
  and produces a `WorkerCrashed` error with exit details.
- **RSS monitoring** — the watchdog reads the worker's RSS via procfs/mach
  calls. Cross-boundary memory stats are the only cross-process observation.

### Protocol Security

The host–worker IPC is a framed length-prefixed JSON protocol over stdin/stdout:

1. **Size bounding** — each frame is limited to `MAX_FRAME_SIZE_BYTES` (1 MiB).
   Larger frames are rejected at deserialization.
2. **Version check** — every frame carries a `ProtocolVersion`. Only `V1_0` is
   accepted.
3. **Sequence validation** — `ProtocolValidator` tracks `expected_next_seq`.
   Frames with gaps or regressions are rejected (`SequenceGap`,
   `SequenceRegression`).
4. **Worker ID binding** — the first `Hello` frame binds the worker instance ID.
   Subsequent frames must carry the expected ID.
5. **State machine** — the validator tracks active and terminal requests.
   Frames arriving after a terminal message or starting a duplicate generation
   are rejected (`TerminalAfterClose`, `DuplicateRequestStart`).
6. **No tensor payload on wire** — only metadata, token IDs, and control
   messages flow over IPC. All tensor data stays within the worker's address
   space.

### Crash Containment

- The supervisor holds a `Mutex<Option<Child>>` for process control. Lock
  is never held across blocking I/O.
- `WorkerProcessControl::kill()` is idempotent — once `killed` is set,
  subsequent calls are no-ops.
- Exit status is captured via `try_wait()` and stored in `WorkerExitDetails`
  for diagnostic logging.
- The `failure_funnel` attaches `ForcedTerminationReason` and
  `WorkerExitDetails` to the returned `EngineError`.

### Reconnection

- On worker crash, the supervisor may respawn up to `restart_limit` times.
- Each restart starts with a fresh `Hello`/`HelloAck` handshake.
- Active request state is lost on crash; the SDK caller receives
  `WorkerCrashed`.

### Watchdog Thread

The watchdog runs at `watchdog_interval_ms` (150 ms in qualification mode).
It checks:

- Worker process alive (`try_wait`).
- Heartbeat recency (must be within `worker_heartbeat_timeout`).
- RSS against soft and hard ceilings.
- Request deadline expiry.

---

## 8. Session Isolation

### Architecture

Session state is split across the process boundary:

- **`GenerationControlSession`** (host side) — owns session identity, policy
  state, lifecycle state machine, deadline tracking, stream assignment, and
  terminal outcome. Owns **no** MLX arrays and **no** KV cache.
- **`InferenceSession`** (worker side) — owns the KV cache, generated tokens,
  sampling state, cancellation flag, and runtime receipts. Per-layer KV caches
  are `Vec<KvCache>` owned exclusively by this session.

### Isolation Properties

- **KV cache isolation** — each `InferenceSession` has its own `Vec<KvCache>`.
  No cross-session pointer sharing. The KV cache lives in the worker's address
  space and is destroyed when the session terminates.
- **Arena segment isolation** — IOSurface-backed arenas are allocated per-session.
  The arena pool mediates acquire/release with max-per-key budgeting.
- **Memory tracking** — per-session memory is tracked through MLX's active
  memory and cache limit APIs. The supervisor enforces per-worker ceilings,
  which bound per-session memory indirectly.
- **State machine guarantees** — both session state machines enforce
  irreversible terminal states via `can_transition_to()`. No transition is
  allowed from `Completed`, `Cancelled`, or `Failed` to any other state.
  `Failed` is reachable from any non-terminal state.
- **No cross-session leakage** — because sessions map 1:1 to a worker process
  (one model image per worker), the OS process boundary provides hardware-enforced
  isolation. No session can read another session's tokens, KV cache state, or
  model weights.
- **Cancellation atomicity** — `ActiveRequest` uses `terminal_recorded`
  (`AtomicBool`) for exactly-once terminal event delivery across threads.
- **Receipt chain** — every session produces a `TerminalRequestReceipt` with
  phase-level telemetry (`PhaseReceipt`), per-step granularity (`StepReceipt`),
  and the final outcome. Receipts are timestamped via the `Timeline` bounded
  event buffer.

### Cooperative Host Assumption

Session isolation **assumes a cooperative host**. The host creates sessions
via the N-API binding and is responsible for not reusing session IDs or
mixing session state. There is no sandbox between the SDK caller and the
engine's host-side data structures. A malicious or buggy host can corrupt
its own session registry. Mitigations:

- The `ActiveRequestRegistry` maintains a forward map (request_id → ActiveRequest)
  and a reverse index (public_job_id → request_id). O(1) lookups limit the
  blast radius of a corrupted index to one entry.
- The `GenerationControlSession` exposes no direct access to the worker's
  address space — all interaction goes through the validated IPC protocol.

---

## 9. Cancellation Safety

### Cancellation Paths

```
SDK Caller ──cancel()──▶ Supervisor ──CancelGeneration──▶ Worker
                                               │
                          ┌─────────────────────┘
                          ▼
                    InferenceSession
                    cancellation_flag = true
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
           Decode loop         Prefill loop
           checks flag         checks flag
                 │                 │
                 ▼                 ▼
           Generation          Generation
           Cancelled           Cancelled
                 │                 │
                 ▼                 ▼
           InferenceSession    InferenceSession
           → Cancelled        → Cancelled
```

### Propagation

1. The SDK caller initiates cancellation via the N-API binding.
2. The supervisor sends `HostCommand::CancelGeneration` over IPC.
3. The worker sets `cancellation_flag` (AtomicBool) on the `InferenceSession`.
4. The decode/prefill loop checks the flag at iteration boundaries (after each
   token or attention layer step).
5. The worker emits `WorkerEvent::GenerationCancelled`.
6. The supervisor marks the `ActiveRequest` terminal.

### Grace Period

If the worker does not acknowledge cancellation within `cancellation_grace_period`
(2 s in qualification mode), the supervisor escalates:

1. Send SIGKILL to the worker process.
2. Wait up to `SHUTDOWN_TIMEOUT` (10 s) for process exit.
3. Record `ForcedTerminationReason::GracefulShutdownTimeout` or
   `ForcedTerminationReason::HardKillOnCancel`.
4. Produce `WorkerCrashed` error with exit details.

### Resource Cleanup Guarantees

- **Worker process termination** — SIGKILL guarantees the OS reclaims all
  worker memory (MLX GPU allocations, IOSurface buffers, KV cache, arenas).
  Metal buffers backed by IOSurface are released by the kernel when the
  process exits.
- **Host-side cleanup** — the supervisor's `Drop` implementation kills the
  worker process, joins the event-reader and watchdog threads, and releases
  the process handle.
- **Layer lease RAII** — `LayerLease::drop()` frees MLX array handles from
  `ARRAY_REGISTRY`. Callers must call `eval()` before dropping to ensure the
  MLX computation graph has consumed the weights.
- **Session terminal state** — once `Cancelled`, the `InferenceSession` state
  machine disallows all further transitions. Any in-flight generation loop
  iteration that observes the flag after the terminal event will find the
  session terminal and abort.
- **Exactly-once delivery** — `ActiveRequest::terminal_recorded` is an
  `AtomicBool` that ensures the `GenerationCancelled` or `GenerationCompleted`
  event is delivered to the SDK caller exactly once, even if multiple threads
  observe the terminal condition simultaneously.

### Cancellation Error Codes

All cancellation paths produce `EngineErrorCode::Cancelled` through the
`cancellation_funnel`, which carries a human-readable reason. The error is
not marked retryable (the session state is terminal).

---

## 10. Dependency Supply Chain

| Dependency | Source | Version | Notes |
|------------|--------|---------|-------|
| **mlx-rs / mlx-sys** | Fork at `mlx-rs-fork/` (based on upstream mlx-rs v0.21.2) | v0.21.2 | Forked for API alignment (free-function ops, `Exception` error type, `IndexOp` trait). The fork includes `mlx-rs`, `mlx-sys`, `mlx-macros`, `mlx-internal-macros`. Features: `metal`, `accelerate`, `safetensors`. |
| **napi-rs** | `napi = "3.9.0"`, `napi-derive = "3.5.6"`, `@napi-rs/cli ^3.0.0` | 3.x | N-API bindings for Node.js addon. Features: `napi8`, `serde-json`. |
| **safetensors** | `safetensors = "0.5"` | 0.5 | Safetensors format parsing. Used only for header reading and `SafeTensors::deserialize` — tensor data is not materialized through this dependency during validation. |
| **tokio** | `tokio = "1"` (features: rt-multi-thread, sync, time, macros) | 1.x | Async runtime for supervisor internals. |
| **serde / serde_json** | `serde = "1"`, `serde_json = "1"` | 1.x | JSON serialization for IPC protocol, receipts, and manifest files. |
| **sha2** | `sha2 = "0.10"` | 0.10 | SHA-256 hashing for ComputeImage segment verification and config hashing. |
| **parking_lot** | `parking_lot = "0.12"` | 0.12 | Fast `Mutex` implementation for supervisor components. |
| **uuid** | `uuid = "1"` (features: v4, serde) | 1.x | Session and request identifier generation. |
| **cc** (build) | `cc = "1.2"` | 1.x | Compiles ObjC++ bridge at build time. No prebuilt binaries. |
| **napi-build** (build) | `napi-build = "2.3.2"` | 2.x | N-API build setup. |
| **CoreML / CoreVideo / IOSurface** | System frameworks (macOS) | OS-provided | Linked at build time, resolved at runtime. Trusted as OS kernel components. |

### Build Integrity

- **`private-development` feature** — a Cargo feature flag that enables
  `allow_high_memory_override()`. This feature **MUST NOT** be compiled into
  release or CI builds. It relaxes all resource ceilings and sets
  `unqualified: true`, bypassing qualification-mode constraints.
- **Release profile** — `lto = true`, `strip = "symbols"`, `codegen-units = 1`.
  No debug symbols are shipped.
- **N-API target** — compiled only for `aarch64-apple-darwin`. No other
  platforms are currently shipped.

---

## 11. AGPL-3.0-only Implications


## 11. AGPL-3.0-only and Dual Licensing

Tribunus Compute is dual-licensed under **AGPL-3.0-only** and a separate
**commercial license**. See [README.md §12](../README.md#12-license) and
[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) for the full policy.

### Patent Grant

AGPL-3.0 grants every compliant recipient a worldwide, royalty-free patent
license covering claims necessarily infringed by the contributed implementation.
Tribunus, Inc. retains ownership of its patents. The license does not assign
or transfer patent ownership — it grants recipients a covenant not to sue for
patent infringement arising from their compliant use of the kernel.

A commercial license may grant broader patent rights beyond the statutory
AGPL grant. Contact `license@tribunus.io` for details.

### Linking

- The Rust core and the ObjC++ bridge are part of "the Program" under AGPL-3.0.
- MLX C library is a separate work (MIT-licensed). The FFI boundary through
  `mlx-sys` creates an "aggregate" under AGPL-3.0 Section 5.
- Core ML, CoreVideo, and IOSurface are System Libraries (AGPL-3.0 Section 1).
  They do not trigger the AGPL's copyleft requirements.
- napi-rs is a separate work (MIT-licensed). The N-API boundary is a Standard
  Interface.

### Network Deployment

AGPL-3.0 Section 13 requires that anyone who interacts with the Program over
a network must be offered the Corresponding Source. For Tribunus Compute, a
user sending a prompt and receiving tokens back qualifies as interaction.

### Compliance

1. Provide a link to this repository in your service's user interface.
2. If you modify the Program, make your modified source available under AGPL-3.0.
3. Model weights are data, not part of the Program. Their license is independent.
4. If AGPL compliance is not acceptable, a commercial license is available.
5. For questions, contact `security@tribunus.io` or `license@tribunus.io`.

---

## Incident Response

1. **Report received** — acknowledged within 48 hours.
2. **Triage** — severity assessment (CVSS 3.1), affected component identification,
   reproduction.
3. **Fix development** — target: critical (CVSS ≥ 9.0) within 7 days; high
   (CVSS ≥ 7.0) within 14 days; medium/low per agreed timeline.
4. **Release** — patched version published as a GitHub release. Changelog entry
   credits the reporter (unless anonymity is requested).
5. **Disclosure** — coordinated disclosure after 90-day embargo.

---

*Last updated: 2026-06-08*
