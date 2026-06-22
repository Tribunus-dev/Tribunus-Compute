# Fused Metal Kernel Dispatch — Production Implementation Map

## Objective

Implement the Metal dispatch path so that `MetalFusedKernelRunner` actually launches
Metal compute via `MTLComputePipelineState`, measures timing, records
`FusedMetalExecutionEvidence`, and falls back cleanly on failure.

---

## 1. Metal Artifact Loading at Runtime

**Current state:** `worker_dispatch.rs` has `LoadedMetalKernel` (artifact + pipeline_state)
but `MetalPipelineState` is a private stub (`library_data`, `function_name`).

### Required changes

### 1a. MetalPipelineState — real MTL pipeline state (worker_dispatch.rs)

Replace the private `struct MetalPipelineState` with:

```rust
pub struct MetalPipelineState {
    pub library: metal::MetalLibrary,
    pub pipeline: metal::ComputePipelineState,
    pub entry_point: String,
}
```

Requires the `metal` crate (`metal-rs`). Add to `Cargo.toml` if not present.

Files to modify:
- `worker_dispatch.rs` — replace `MetalPipelineState`, update `LoadedMetalKernel`
- `worker_dispatch.rs` — add `fn load_metallib(path: &Path, entry_point: &str) -> Result<MetalPipelineState, String>`

### 1b. load_metallib implementation

```rust
pub fn load_metallib(path: &Path, entry_point: &str) -> Result<MetalPipelineState, String> {
    let device = metal::Device::system_default()
        .ok_or_else(|| "no Metal device".to_string())?;
    let lib_data = std::fs::read(path)
        .map_err(|e| format!("read metallib: {}", e))?;
    let lib = device.new_library_with_data(&lib_data)
        .map_err(|e| format!("create library: {}", e))?;
    let func = lib.get_function(entry_point, None)
        .ok_or_else(|| format!("entry point '{}' not found", entry_point))?;
    let pipeline = device.new_compute_pipeline_state_with_function(&func)
        .map_err(|e| format!("create pipeline state: {}", e))?;
    Ok(MetalPipelineState { library: lib, pipeline, entry_point: entry_point.into() })
}
```

### 1c. Wire into runtime loading

In `profiled_executor.rs` or `profiled_model.rs::new()`, after loading manifests:

```rust
let metal_kernels: Vec<LoadedMetalKernel> = if let Some(dag) = &manifest.phase_dag {
    let mut kernels = Vec::new();
    for artifact in &manifest.metal_kernel_artifacts {
        let metallib_path = image_dir.join(&artifact.metallib_relpath);
        match load_metallib(&metallib_path, &artifact.dispatch.entry_point) {
            Ok(state) => kernels.push(LoadedMetalKernel {
                artifact: artifact.clone(),
                pipeline_state: state,
            }),
            Err(e) => eprintln!("[metal] failed to load '{}': {}", artifact.artifact_id, e),
        }
    }
    kernels
} else {
    Vec::new()
};
```

Load these into `RuntimeBackends.metal_kernels`.

---

## 2. Metal Kernel Dispatch

**Current state:** `MetalFusedKernelRunner::run()` downcasts `RuntimeBackends`, finds matching
kernel, logs — does not actually dispatch.

### Required changes

### 2a. Metal dispatch function (worker_dispatch.rs or new metal_launcher.rs)

```rust
use metal::*;

pub fn dispatch_fused_kernel(
    kernel: &LoadedMetalKernel,
    buffers: &[&metal::Buffer],
    command_buffer: &metal::CommandBuffer,
) -> Result<u64, String> {
    let start = std::time::Instant::now();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&kernel.pipeline_state.pipeline);

    // Bind buffers according to the dispatch recipe's buffer_slot_map
    for (slot, buf) in kernel.artifact.dispatch.buffer_slot_map.iter() {
        let idx = *slot as usize;
        if idx < buffers.len() {
            encoder.set_buffer(*buffers[idx] as *const _ as u64, 0, idx as u64);
        }
    }

    // Dispatch
    let tg = kernel.artifact.dispatch.threads_per_threadgroup;
    let gg = kernel.artifact.dispatch.threadgroups_per_grid;
    encoder.dispatch_thread_groups(
        MTLSize { width: gg[0] as u64, height: gg[1] as u64, depth: gg[2] as u64 },
        MTLSize { width: tg[0] as u64, height: tg[1] as u64, depth: tg[2] as u64 },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(start.elapsed().as_micros() as u64)
}
```

### 2b. Wire into MetalFusedKernelRunner (phase_runner.rs)

Replace the current `MetalFusedKernelRunner::run()` body with:

```rust
fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
    let region = phase.metadata.get("fusion_region")
        .cloned()
        .unwrap_or_else(|| phase.phase_id.clone());

    let backends = ctx.backend.as_ref()
        .and_then(|b| b.downcast_ref::<RuntimeBackends>())
        .ok_or_else(|| "no runtime backends".to_string())?;

    // Find the matching loaded kernel
    let kernel = backends.metal_kernels.iter()
        .find(|k| k.artifact.artifact_id == region)
        .ok_or_else(|| format!("kernel '{}' not loaded", region))?;

    // Get the Metal command buffer (from context or create one)
    let device = metal::Device::system_default().ok_or("no Metal device")?;
    let cmd_buf = device.new_command_buffer();

    // --- Admission gate ---
    let metallib_bytes = std::fs::read(
        /* resolve from manifest */ "placeholder"
    ).unwrap_or_default();
    let verdict = crate::benchmark::admission::check_fused_metal_benchmark_admission(
        /* need sealed artifact */ todo!(), &metallib_bytes, "m1",
    );
    if let AdmissionVerdict::Rejected(reason) = verdict {
        return Err(format!("admission rejected: {}", reason));
    }

    // --- Dispatch ---
    let start = std::time::Instant::now();
    // ... call dispatch_fused_kernel ...
    let duration_us = start.elapsed().as_micros() as u64;

    // Record evidence
    ctx.evidence = Some(FusedMetalExecutionEvidence {
        region_id: region.clone(),
        artifact_name: kernel.artifact.artifact_id.clone(),
        metallib_hash: kernel.artifact.metallib_blake3.clone(),
        launch_contract: todo!(),
        performed_at: crate::now_iso8601(),
        duration_us,
        numerical_summary: None,
        fallback_reason: None,
    });

    Ok(())
}
```

---

## 3. Benchmark Admission Gate Integration

**Current state:** `check_fused_metal_benchmark_admission()` exists in
`benchmark/admission.rs` but is never called.

### Required changes

### 3a. Wire admission gate into runner

Add admission check call in `MetalFusedKernelRunner::run()` before dispatch (see 2b).

### 3b. Admission gate hardening (admission.rs)

- Add `hardware_compatibility()` check: compare `artifact.gpu_family` against actual device name
- Add `numerical_parity()` check: dispatch kernel on a fixed test input, compare output with CPU reference
- Add `qualification_cache()`: cache admission verdicts by artifact hash to avoid re-checking every decode step

---

## 4. Fallback Path Implementation

**Current state:** When `MetalFusedKernelRunner` returns `Err`, the PhaseEngine checks
for `FallbackDecomposition` edges in the DAG and sets status to `FallbackUsed`.

### Required changes

### 4a. Ensure fallback edges exist in emitted DAG

In `compile.rs::emit_phase_graph()` (around line 2310), add a `FallbackDecomposition`
edge for every `MetalFusedKernelPhase` pointing to the corresponding `MlxDecodePhase`.

```rust
edges.push(EmittedPhaseEdge {
    from_phase: phase_id.clone(),
    to_phase: format!("{}_fallback", phase_id),
    semantic_kind: SemanticKind::FallbackDecomposition,
    label: Some("unfused_fallback".into()),
    metadata: HashMap::new(),
});
```

### 4b. Fallback dispatch modes

| Failure mode | PhaseEngine response | Observable in receipt |
|---|---|---|
| xcrun not available at compile time | Kernel not compiled, no artifact | `fallback_reason: XcrunNotAvailable` |
| Seal mismatch at load | Kernel not loaded into registry | Runner returns Err → `FallbackUsed(SealMismatch)` |
| No Metal device | `load_metallib` returns Err | Module not loaded → Runner Err → `FallbackUsed(HardwareIncompatible)` |
| Dispatch error (GPU timeout) | Metal dispatch returns Err | Runner Err → `FallbackUsed(DecompositionUsed)` |
| Admission gate rejects | `check_admission()` returns Rejected | Runner Err → `FallbackUsed(BenchmarkGateRejected)` |
| Runner crashes | Process-level | No receipt — crash log captures layer index |

---

## 5. Timing and Evidence Recording

**Current state:** `FusedMetalExecutionEvidence` struct exists, `from_artifact()` exists.

### Required changes

### 5a. Populate evidence from dispatch

After successful Metal dispatch:

```rust
let evidence = FusedMetalExecutionEvidence {
    region_id: region,
    artifact_name: kernel.artifact.artifact_id.clone(),
    metallib_hash: kernel.artifact.metallib_blake3.clone(),
    launch_contract: MetalLaunchContract {
        entry_point: kernel.pipeline_state.entry_point.clone(),
        threads_per_threadgroup: kernel.artifact.dispatch.threads_per_threadgroup,
        threadgroups_per_grid: kernel.artifact.dispatch.threadgroups_per_grid,
        buffer_bindings: kernel.artifact.dispatch.buffer_slot_map.clone(),
    },
    performed_at: crate::now_iso8601(),
    duration_us,
    numerical_summary: None,
    fallback_reason: None,
};
```

### 5b. Add numerical summary (optional, post-qualification)

After output buffer is ready, read the result back to CPU and compute
min/max/mean/std for the evidence record.

---

## 6. Benchmark Report Integration

**Current state:** `tribunus-bench decode` outputs JSON with decode throughput.
Phase receipts are not included.

### Required changes

### 6a. Add phase receipts to bench report (tribunus-bench.rs)

In `cmd_decode`, after the decode loop, collect receipts from the session:

```rust
let receipts = session.take_phase_receipts();
metrics.insert("phase_count".into(), json!(receipts.len()));
metrics.insert("fused_kernel_count".into(), json!(
    receipts.iter().filter(|r| r.fused_evidence.is_some()).count()
));
metrics.insert("fallback_count".into(), json!(
    receipts.iter().filter(|r| matches!(r.status, PhaseCompletionStatus::FallbackUsed(_))).count()
));
```

### 6b. Add `take_phase_receipts()` to `ProfiledInferenceSession`

A new field `phase_receipts: Vec<PhaseReceipt>` on the session, populated
during `execute_phase_dag()` calls.

---

## 7. File-by-file change list

| File | Change | Priority |
|---|---|---|
| `Cargo.toml` | Add `metal = "0.29"` dependency | P0 |
| `worker_dispatch.rs` | Replace `MetalPipelineState` with real `metal::ComputePipelineState` | P0 |
| `worker_dispatch.rs` | Add `load_metallib(path, entry_point)` | P0 |
| `profiled_model.rs` | Load metal kernels during model init | P0 |
| `profiled_executor.rs` | Wire `RuntimeBackends.metal_kernels` | P0 |
| `profiled_executor.rs` | Add `phase_receipts` field + collector | P1 |
| `metal_launcher.rs` (new) | `dispatch_fused_kernel()` | P0 |
| `phase_runner.rs` | MetalFusedKernelRunner → real dispatch | P0 |
| `admission.rs` | Wire into dispatch, add hardware check | P1 |
| `compile.rs` | Add fallback edges to emitted DAG | P2 |
| `tribunus-bench.rs` | Include receipts in report | P2 |

## 8. Testing strategy

| Test type | What it validates | File |
|---|---|---|
| Unit | `load_metallib` returns Ok with valid pipeline state for a known .metallib | `worker_dispatch_test.rs` |
| Unit | `dispatch_fused_kernel` returns non-zero duration | `metal_launcher_test.rs` |
| Integration | Fused QKV kernel output matches unfused Q+K+V (FP16 tolerance) | `tests/fused_metal_dispatch_test.rs` |
| Integration | Admission gate correctly rejects tampered artifacts | `tests/admission_integration_test.rs` |
| E2E | `tribunus-bench decode --phase-dag` produces receipts with fused_evidence | `tests/bench_e2e_test.rs` |
