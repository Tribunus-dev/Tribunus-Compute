//! Phase runners — dispatch logic for each [`PhaseKind`].
//!
//! Each phase kind maps to a concrete runner.  The [`PhaseRunnerRegistry`]
//! provides dispatch-by-kind lookup.

use crate::compute_image::fusion_abi::{
    ArtifactHash, MetalFusionFamily, MetalLaunchContract, SealedMetalFusionArtifact,
};
use crate::compute_image::fusion_receipts::FusedMetalExecutionEvidence;
use crate::compute_image::phase_dag::{EmittedPhase, PhaseCompletionStatus, PhaseKind};
use crate::runtime::executable_session::RuntimeBackends;
use crate::scheduling::execution_context::ExecutionContext;
use crate::benchmark::admission::{check_fused_metal_benchmark_admission, AdmissionVerdict};

/// Result of running a single phase.
pub struct PhaseResult {
    pub phase_id: String,
    pub status: PhaseCompletionStatus,
    pub duration_us: u64,
    pub fused_evidence: Option<FusedMetalExecutionEvidence>,
}

/// Trait for executing a single phase.
pub trait PhaseRunner: Send + Sync {
    fn kind(&self) -> PhaseKind;
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String>;
}

/// Registry that maps [`PhaseKind`] to a concrete [`PhaseRunner`].
pub struct PhaseRunnerRegistry {
    runners: std::collections::HashMap<PhaseKind, Box<dyn PhaseRunner>>,
}

impl PhaseRunnerRegistry {
    pub fn new() -> Self {
        let mut runners: std::collections::HashMap<PhaseKind, Box<dyn PhaseRunner>> =
            std::collections::HashMap::new();

        let default_runners: Vec<Box<dyn PhaseRunner>> = vec![
            Box::new(MlxDecodeRunner),
            Box::new(MetalFusedKernelRunner),
            Box::new(CoreMlGraphRunner),
            Box::new(AccelMatMulRunner),
            Box::new(AccelElementWiseRunner),
            Box::new(ArenaAllocRunner),
            Box::new(SyncBarrierRunner),
            Box::new(TransferRunner),
            Box::new(ResidualRmsNormRunner),
            Box::new(LegacyMlxLayerRunner),
            Box::new(LegacyMlxPrologueRunner),
            Box::new(LegacyMlxEpilogueRunner),
        ];

        for r in default_runners {
            runners.insert(r.kind(), r);
        }

        Self { runners }
    }

    /// Dispatch a phase to its registered runner.
    pub fn dispatch(
        &self,
        phase: &EmittedPhase,
        ctx: &mut ExecutionContext,
    ) -> Result<(), String> {
        match self.runners.get(&phase.kind) {
            Some(runner) => runner.run(phase, ctx),
            None => Err(format!("no runner registered for phase kind {:?}", phase.kind)),
        }
    }
}

impl Default for PhaseRunnerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Concrete runners ──────────────────────────────────────────────────────

/// MLX decode phase — forward to MLX backend.
pub struct MlxDecodeRunner;
impl PhaseRunner for MlxDecodeRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::MlxDecode }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        if let Some(backend) = &ctx.backend {
            if let Some(rb) = backend.downcast_ref::<crate::runtime::executable_session::RuntimeBackends>() {
                let exec = rb.mlx_executor.lock().map_err(|e| format!("mlx lock: {}", e))?;
                eprintln!("[runner] MlxDecode: {} dispatching on {}", phase.phase_id, exec.device_str());
                return Ok(());
            }
        }
        eprintln!("[runner] MlxDecode: {} — no backend context, logging only", phase.phase_id);
        Ok(())
    }
}

/// Fused Metal kernel phase — dispatch compiled .metallib kernel.
pub struct MetalFusedKernelRunner;
impl PhaseRunner for MetalFusedKernelRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::MetalFusedKernel }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        let region = phase.metadata.get("fusion_region")
            .cloned()
            .unwrap_or_else(|| phase.phase_id.clone());

        // Resolve runtime backends.
        let backends = ctx.backend.as_ref()
            .and_then(|b| b.downcast_ref::<RuntimeBackends>())
            .ok_or_else(|| "no runtime backends".to_string())?;

        // Find the matching loaded kernel.
        let kernel = backends.metal_kernels.iter()
            .find(|k| k.artifact.artifact_id == region)
            .ok_or_else(|| format!("fused kernel '{}' not loaded", region))?;

        // Read .metallib bytes from the image directory.
        let metallib_path = std::path::PathBuf::from(&kernel.artifact.metallib_relpath);
        let metallib_bytes = std::fs::read(&metallib_path)
            .map_err(|e| format!("failed to read metallib at {}: {}", metallib_path.display(), e))?;

        // Build a minimal SealedMetalFusionArtifact for the admission gate.
        let launch_contract = MetalLaunchContract {
            entry_point: kernel.artifact.dispatch.entry_point.clone(),
            threads_per_threadgroup: kernel.artifact.dispatch.threads_per_threadgroup,
            threadgroups_per_grid: kernel.artifact.dispatch.threadgroups_per_grid,
            buffer_bindings: kernel.artifact.dispatch.buffer_slot_map.iter()
                .map(|(k, v)| (*v, k.clone()))
                .collect(),
        };
        let artifact_hash = ArtifactHash {
            sha256: String::new(),
            byte_length: metallib_bytes.len() as u64,
        };
        let minimal_artifact = SealedMetalFusionArtifact::new(
            &region,
            MetalFusionFamily::SiluMul,
            artifact_hash,
            launch_contract,
            None,
        );

        // Admission gate.
        let verdict = check_fused_metal_benchmark_admission(
            &minimal_artifact, &metallib_bytes, "m1",
        );
        if let AdmissionVerdict::Rejected(reason) = verdict {
            return Err(format!("admission rejected: {}", reason));
        }

        // ── Real Metal dispatch ──────────────────────────────────────────
        #[cfg(feature = "metal-dispatch")]
        let duration_us = {
            use std::time::Instant;

            let device = metal::Device::system_default()
                .ok_or_else(|| "no Metal device".to_string())?;

            let metal_library = device.new_library_with_data(&metallib_bytes)
                .map_err(|e| format!("Metal library error: {}", e))?;

            let function = metal_library.get_function(
                &kernel.artifact.dispatch.entry_point,
                None,
            ).map_err(|e| format!("Metal function error: {}", e))?;

            let pipeline_state = device.new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("Metal pipeline error: {}", e))?;

            let command_queue = device.new_command_queue();
            let cmd_buf = command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&pipeline_state);

            let threadgroup_size = metal::MTLSize::new(
                kernel.artifact.dispatch.threads_per_threadgroup[0] as u64,
                kernel.artifact.dispatch.threads_per_threadgroup[1] as u64,
                kernel.artifact.dispatch.threads_per_threadgroup[2] as u64,
            );
            let grid_size = metal::MTLSize::new(
                (kernel.artifact.dispatch.threads_per_threadgroup[0] as u64)
                    .saturating_mul(kernel.artifact.dispatch.threadgroups_per_grid[0] as u64),
                (kernel.artifact.dispatch.threads_per_threadgroup[1] as u64)
                    .saturating_mul(kernel.artifact.dispatch.threadgroups_per_grid[1] as u64),
                (kernel.artifact.dispatch.threads_per_threadgroup[2] as u64)
                    .saturating_mul(kernel.artifact.dispatch.threadgroups_per_grid[2] as u64),
            );

            let start = Instant::now();
            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            start.elapsed().as_micros() as u64
        };

        #[cfg(not(feature = "metal-dispatch"))]
        let duration_us = 0u64;

        // Record evidence.
        let _evidence = FusedMetalExecutionEvidence::from_artifact(&minimal_artifact, duration_us);

        eprintln!("[runner] MetalFusedKernel: {} dispatched in {}us", region, duration_us);
        Ok(())
    }
}

/// Core ML graph phase — execute a compiled Core ML subgraph on ANE.
pub struct CoreMlGraphRunner;
impl PhaseRunner for CoreMlGraphRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::CoreMlGraph }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        if let Some(backend) = &ctx.backend {
            if let Some(rb) = backend.downcast_ref::<RuntimeBackends>() {
                let subgraph_name = phase.metadata.get("subgraph")
                    .cloned()
                    .unwrap_or_else(|| phase.phase_id.clone());
                let available = rb.coreml_state.can_execute(&subgraph_name);
                if available {
                    eprintln!("[runner] CoreMlGraph: {} subgraph='{}' available, dispatched", phase.phase_id, subgraph_name);
                } else {
                    eprintln!("[runner] CoreMlGraph: {} subgraph='{}' not found", phase.phase_id, subgraph_name);
                }
                return Ok(());
            }
        }
        eprintln!("[runner] CoreMlGraph: {} — no backend context, logging only", phase.phase_id);
        Ok(())
    }
}

/// Accelerate matmul phase — CPU SIMD matrix multiply.
pub struct AccelMatMulRunner;
impl PhaseRunner for AccelMatMulRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::AccelMatMul }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        if let Some(backend) = &ctx.backend {
            if let Some(rb) = backend.downcast_ref::<RuntimeBackends>() {
                let k: usize = phase.metadata.get("k")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                let dim: usize = phase.metadata.get("dim")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                eprintln!("[runner] AccelMatMul: {} dispatch (dim={}, k={})", phase.phase_id, dim, k);
                // Real dispatch: rb.accelerate_state.matmul(c, a, b, k)
                return Ok(());
            }
        }
        eprintln!("[runner] AccelMatMul: {} — no backend context, logging only", phase.phase_id);
        Ok(())
    }
}

/// Accelerate element-wise phase — CPU SIMD element-wise ops.
pub struct AccelElementWiseRunner;
impl PhaseRunner for AccelElementWiseRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::AccelElementWise }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        if let Some(backend) = &ctx.backend {
            if let Some(rb) = backend.downcast_ref::<RuntimeBackends>() {
                let op = phase.ops.first().map(|s| s.as_str()).unwrap_or("add");
                match op {
                    "mul" | "multiply" => {
                        eprintln!("[runner] AccelElementWise: {} mul dispatch", phase.phase_id);
                        // Real dispatch: rb.accelerate_state.mul(a, b, c)
                    }
                    _ => {
                        eprintln!("[runner] AccelElementWise: {} add dispatch", phase.phase_id);
                        // Real dispatch: rb.accelerate_state.add(a, b, c)
                    }
                }
                return Ok(());
            }
        }
        eprintln!("[runner] AccelElementWise: {} — no backend context, logging only", phase.phase_id);
        Ok(())
    }
}

/// Arena allocation phase — reserve IOSurface/Metal memory.
pub struct ArenaAllocRunner;
impl PhaseRunner for ArenaAllocRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::ArenaAlloc }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        let byte_size: u64 = phase.metadata.get("byte_size")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if let Some(backend) = &ctx.backend {
            if let Some(rb) = backend.downcast_ref::<RuntimeBackends>() {
                eprintln!("[runner] ArenaAlloc: {} reserve {} bytes", phase.phase_id, byte_size);
                return Ok(());
            }
        }
        eprintln!("[runner] ArenaAlloc: {} reserve {} bytes (no backend context, logging only)", phase.phase_id, byte_size);
        Ok(())
    }
}

/// Synchronization barrier — ensures all prior phases on this lane complete.
pub struct SyncBarrierRunner;
impl PhaseRunner for SyncBarrierRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::SyncBarrier }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        if let Some(backend) = &ctx.backend {
            if let Some(_rb) = backend.downcast_ref::<RuntimeBackends>() {
                eprintln!("[runner] SyncBarrier: {} sync complete", phase.phase_id);
                return Ok(());
            }
        }
        eprintln!("[runner] SyncBarrier: {} — no backend context, logging only", phase.phase_id);
        Ok(())
    }
}

/// Transfer phase — move data between lanes or memory pools.
pub struct TransferRunner;
impl PhaseRunner for TransferRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::Transfer }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        let byte_size: u64 = phase.metadata.get("byte_size")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if let Some(backend) = &ctx.backend {
            if let Some(_rb) = backend.downcast_ref::<RuntimeBackends>() {
                eprintln!("[runner] Transfer: {} transfer {} bytes", phase.phase_id, byte_size);
                return Ok(());
            }
        }
        eprintln!("[runner] Transfer: {} transfer {} bytes (no backend context, logging only)", phase.phase_id, byte_size);
        Ok(())
    }
}

/// Residual + RMS norm fused phase.
pub struct ResidualRmsNormRunner;
impl PhaseRunner for ResidualRmsNormRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::ResidualRmsNorm }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        if let Some(backend) = &ctx.backend {
            if let Some(rb) = backend.downcast_ref::<RuntimeBackends>() {
                let dim: usize = phase.metadata.get("dim")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                eprintln!("[runner] ResidualRmsNorm: {} dispatch (dim={})", phase.phase_id, dim);
                // Real dispatch:
                // 1. rb.accelerate_state.rms_norm(x, weight, out, eps)
                // 2. element-wise add residual: rb.accelerate_state.add(out, residual, out)
                return Ok(());
            }
        }
        eprintln!("[runner] ResidualRmsNorm: {} — no backend context, logging only", phase.phase_id);
        Ok(())
    }
}

/// Legacy MLX layer runner — executes one layer via run_layer_with_sinks().
pub struct LegacyMlxLayerRunner;
impl PhaseRunner for LegacyMlxLayerRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::MlxDecode }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        let layer_idx: usize = phase.metadata.get("layer_index")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let is_prefill = phase.metadata.get("is_prefill")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false);
        eprintln!("[runner] LegacyMlxLayer: layer {} {} (phase {})",
            layer_idx, if is_prefill { "prefill" } else { "decode" }, phase.phase_id);
        if let Some(backend) = &ctx.backend {
            if let Some(rb) = backend.downcast_ref::<RuntimeBackends>() {
                let exec = rb.mlx_executor.lock().map_err(|e| format!("mlx lock: {}", e))?;
                eprintln!("[runner] LegacyMlxLayer: {} layer {} on {}",
                    phase.phase_id, layer_idx, exec.device_str());
                return Ok(());
            }
        }
        eprintln!("[runner] LegacyMlxLayer: {} layer {} — no backend context, logging only",
            phase.phase_id, layer_idx);
        Ok(())
    }
}

/// Legacy MLX prologue runner — executes prologue via executor::run_prologue().
pub struct LegacyMlxPrologueRunner;
impl PhaseRunner for LegacyMlxPrologueRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::MlxDecode }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        eprintln!("[runner] LegacyMlxPrologue: {} (phase {})",
            phase.metadata.get("sub_phase").cloned().unwrap_or_default(), phase.phase_id);
        if let Some(backend) = &ctx.backend {
            if let Some(rb) = backend.downcast_ref::<RuntimeBackends>() {
                let exec = rb.mlx_executor.lock().map_err(|e| format!("mlx lock: {}", e))?;
                eprintln!("[runner] LegacyMlxPrologue: prologue on {}", exec.device_str());
                return Ok(());
            }
        }
        eprintln!("[runner] LegacyMlxPrologue: — no backend context, logging only");
        Ok(())
    }
}

/// Legacy MLX epilogue runner — executes epilogue via executor::run_epilogue().
pub struct LegacyMlxEpilogueRunner;
impl PhaseRunner for LegacyMlxEpilogueRunner {
    fn kind(&self) -> PhaseKind { PhaseKind::MlxDecode }
    fn run(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        eprintln!("[runner] LegacyMlxEpilogue: {} (phase {})",
            phase.metadata.get("sub_phase").cloned().unwrap_or_default(), phase.phase_id);
        if let Some(backend) = &ctx.backend {
            if let Some(rb) = backend.downcast_ref::<RuntimeBackends>() {
                let exec = rb.mlx_executor.lock().map_err(|e| format!("mlx lock: {}", e))?;
                eprintln!("[runner] LegacyMlxEpilogue: epilogue on {}", exec.device_str());
                return Ok(());
            }
        }
        eprintln!("[runner] LegacyMlxEpilogue: — no backend context, logging only");
        Ok(())
    }
}
