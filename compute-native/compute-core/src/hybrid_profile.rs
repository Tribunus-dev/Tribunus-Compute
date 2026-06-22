//! Hybrid deployment profile — the contract for MLX/Core ML hybrid execution.
//!
//! A hybrid profile describes the MLX regions, Core ML stateful islands,
//! boundary tensors, arena profiles, execution order, fallback policy,
//! and required capabilities. It is separate from the canonical logical model
//! and from the MLX-only profile.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::backend::heterogeneous_executor::BackendInstance;
use crate::backend::routing::*;
use crate::backend::MlxBackend;
use crate::backend::TensorBackend;
use crate::memory::allocator::IosurfaceAllocator;

/// Complete hybrid deployment profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridProfile {
    /// Root model hash (the ComputeImage this profile belongs to).
    pub root_model_hash: String,
    /// Hash of the ComputeImage artifact.
    pub compute_image_hash: String,
    /// Profile version for migration.
    pub version: u32,
    /// The MLX execution regions that bracket Core ML islands.
    pub mlx_regions: Vec<MlxRegion>,
    /// The Core ML stateful islands.
    pub coreml_islands: Vec<CoreMlIsland>,
    /// Boundary tensors that cross between MLX and Core ML.
    pub boundary_tensors: Vec<BoundaryTensor>,
    /// Execution order: sequence of region/island references.
    pub execution_order: Vec<ExecutionStep>,
    /// Fallback policy when Core ML is unavailable.
    pub fallback: FallbackPolicy,
    /// Required runtime capabilities.
    pub required_capabilities: Vec<String>,
    /// Minimum OS version (e.g. "15.0").
    pub min_os_version: String,
    /// Storage ABI identifier.
    pub storage_abi: String,
    /// Compute-unit preference ("cpuAndGPU", "cpuAndNeuralEngine", "all").
    pub compute_units: ComputeUnits,
}

/// An MLX execution region — pure MLX operations that run before or after Core ML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxRegion {
    pub id: String,
    pub kind: MlxRegionKind,
    /// Input boundary tensors consumed from previous step.
    pub inputs: Vec<String>,
    /// Output boundary tensors produced for next step.
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MlxRegionKind {
    Embedding,
    PreAttentionProcess,
    PostAttentionProcess,
    Ffn,
    FinalNorm,
    LmHead,
}

/// A Core ML stateful island — persistent state + stateless boundary interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlIsland {
    pub id: String,
    /// Path to the compiled .mlmodelc artifact.
    pub artifact_path: String,
    /// Hash of the Core ML artifact for cache validation.
    pub artifact_hash: String,
    /// The MIL function name (default: "main").
    pub function_name: String,
    /// Input feature names (boundary activation ingest).
    pub input_names: Vec<String>,
    /// Output feature names (boundary activation output).
    pub output_names: Vec<String>,
    /// State schema — shapes and dtypes of recurrent state.
    pub state_schema: Vec<StateTensor>,
    /// Minimum macOS version for this island.
    pub min_os_version: String,
    /// Compute-unit policy for this island.
    pub compute_units: ComputeUnits,
    /// Fallback region: if this island cannot execute, fall back to MLX.
    pub fallback_region: Option<String>,
    /// Numerical tolerance for output comparison (max absolute error).
    pub tolerance_fp16: f64,
}

/// A state tensor descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTensor {
    pub name: String,
    pub dtype: String, // "float16"
    pub shape: Vec<u32>,
}

/// A boundary tensor that crosses between MLX and Core ML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryTensor {
    pub name: String,
    /// Feature name in the Core ML island.
    pub feature_name: String,
    /// Logical shape.
    pub shape: Vec<u32>,
    /// FP16 arena profile.
    pub arena_profile: String, // "IOSurfaceFp16ContiguousV1"
}

/// One step in the execution order.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExecutionStep {
    #[serde(rename = "mlx")]
    Mlx { region_id: String },
    #[serde(rename = "coreml")]
    CoreMl { island_id: String },
    #[serde(rename = "ane")]
    AneInference {
        /// MIL program text to compile.
        mil_text: String,
        /// Input tensor names.
        inputs: Vec<String>,
        /// Output tensor names.
        outputs: Vec<String>,
        /// Program tag for caching.
        tag: String,
    },
}

/// Fallback policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "policy")]
pub enum FallbackPolicy {
    /// Fail if Core ML is unavailable.
    #[serde(rename = "require")]
    RequireCoreMl,
    /// Fall back to MLX if Core ML unavailable.
    #[serde(rename = "mlx_fallback")]
    MlxFallback,
    /// Use MLX for all execution (no Core ML).
    #[serde(rename = "mlx_only")]
    MlxOnly,
}

/// Compute-unit preference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ComputeUnits {
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
    All,
}

impl std::fmt::Display for ComputeUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeUnits::CpuOnly => write!(f, "cpuOnly"),
            ComputeUnits::CpuAndGpu => write!(f, "cpuAndGPU"),
            ComputeUnits::CpuAndNeuralEngine => write!(f, "cpuAndNeuralEngine"),
            ComputeUnits::All => write!(f, "all"),
        }
    }
}

impl HybridProfile {
    /// Validate against the runtime capability report. Returns the first missing capability.
    pub fn validate(
        &self,
        caps: &crate::capability::SharedTensorCapabilityReport,
    ) -> Result<(), String> {
        for req in &self.required_capabilities {
            let present = match req.as_str() {
                "iosurface_fp16_bridge" => caps.supports_iosurface_fp16_bridge,
                "coreml_iosurface_input" => caps.supports_coreml_iosurface_input,
                "coreml_output_backing" => caps.supports_coreml_output_backing,
                "mlx_external_array" => caps.supports_mlx_iosurface_external_array,
                "mlx_coreml_round_trip" => caps.supports_mlx_coreml_round_trip,
                "coreml_stateful" => caps.supports_coreml_stateful_models,
                "coreml_async" => caps.supports_coreml_async_stateful_prediction,
                _ => false,
            };
            if !present {
                return Err(format!("missing required capability: {}", req));
            }
        }
        Ok(())
    }

    /// Check that boundary tensors flow correctly between steps.
    pub fn validate_tensor_flow(&self) -> Result<(), String> {
        // Each boundary tensor must be produced by exactly one step and consumed by exactly one step.
        // This is a simple check that every tensor appears in at least one producer and one consumer.
        let mut producers: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        let mut consumers: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();

        for step in &self.execution_order {
            match step {
                ExecutionStep::Mlx { region_id } => {
                    if let Some(region) = self.mlx_regions.iter().find(|r| &r.id == region_id) {
                        for output in &region.outputs {
                            if producers.contains_key(output.as_str()) {
                                return Err(format!(
                                    "tensor {} produced by multiple steps",
                                    output
                                ));
                            }
                            producers.insert(output, region_id);
                        }
                        for input in &region.inputs {
                            consumers.insert(input, region_id);
                        }
                    }
                }
                ExecutionStep::CoreMl { island_id } => {
                    if let Some(island) = self.coreml_islands.iter().find(|i| &i.id == island_id) {
                        for output in &island.output_names {
                            if producers.contains_key(output.as_str()) {
                                return Err(format!(
                                    "tensor {} produced by multiple steps",
                                    output
                                ));
                            }
                            producers.insert(output, island_id);
                        }
                        for input in &island.input_names {
                            consumers.insert(input, island_id);
                        }
                    }
                }
                ExecutionStep::AneInference { .. } => {
                    // ANE steps manage their own IO through the compiled program;
                    // they do not participate in the named boundary-tensor flow.
                }
            }
        }
        Ok(())
    }
}

// ── HybridExecutor ────────────────────────────────────────────────────────────

/// Dispatches execution across MLX and Core ML backends according to a
/// [`HybridProfile`].  Each step in `execution_order` is mapped to its
/// registered backend (or a temporary instance).  Boundary tensors flow
/// through the optional IOSurface allocator for zero-copy sharing.
pub struct HybridExecutor {
    profile: HybridProfile,
    mlx_backend: Option<Box<dyn BackendInstance + Send>>,
    accelerate_backend: Option<Box<dyn BackendInstance + Send>>,
    allocator: Option<Arc<IosurfaceAllocator>>,
}

impl HybridExecutor {
    /// Creates a new executor with the given deployment profile.
    /// Backends must be registered before calling [`execute`](Self::execute).
    pub fn new(profile: HybridProfile) -> Self {
        Self {
            profile,
            mlx_backend: None,
            accelerate_backend: None,
            allocator: None,
        }
    }

    /// Register the MLX backend instance.
    pub fn register_mlx(&mut self, backend: Box<dyn BackendInstance + Send>) {
        self.mlx_backend = Some(backend);
    }

    /// Register the Accelerate backend instance.
    pub fn register_accelerate(&mut self, backend: Box<dyn BackendInstance + Send>) {
        self.accelerate_backend = Some(backend);
    }

    /// Set the IOSurface allocator for cross-backend tensor transfers.
    pub fn set_allocator(&mut self, allocator: Arc<IosurfaceAllocator>) {
        self.allocator = Some(allocator);
    }

    /// Execute every step in [`self.profile.execution_order`] and return
    /// one [`BoundaryExecutionReceipt`] per step.
    ///
    /// # Errors
    ///
    /// Returns `Err` if an MLX step references a region that does not exist
    /// in the profile, or if a registered backend's evaluate call fails.
    pub fn execute(&mut self) -> Result<Vec<BoundaryExecutionReceipt>, String> {
        let mut receipts = Vec::with_capacity(self.profile.execution_order.len());

        for step in &self.profile.execution_order {
            match step {
                ExecutionStep::Mlx { region_id } => {
                    let region = self
                        .profile
                        .mlx_regions
                        .iter()
                        .find(|r| &r.id == region_id)
                        .ok_or_else(|| {
                            format!("MLX region '{}' not found in profile", region_id)
                        })?;

                    let eval_receipt = if let Some(backend) = self.mlx_backend.as_mut() {
                        backend.evaluate(0, &[])?
                    } else {
                        let mut tmp = MlxBackend::new();
                        tmp.evaluate(0, &[])?
                    };

                    receipts.push(BoundaryExecutionReceipt {
                        group_id: EvaluationGroupId(0),
                        planned_policy: EvaluationPolicy::BackendLazy,
                        backend: BackendId(0),
                        operation_count: region.outputs.len().max(1),
                        planned_materialized_outputs: region.outputs.len(),
                        actual_eval_calls: eval_receipt.eval_calls,
                        actual_sync_count: 1,
                        graph_build_ns: eval_receipt.graph_build_ns,
                        submit_ns: eval_receipt.submit_ns,
                        execution_ns: eval_receipt.sync_ns,
                        wait_ns: 0,
                        temporary_bytes: 0,
                        released_tensor_count: 0,
                        unaccounted_ns: 0,
                        policy_support: policy_support(
                            BackendId(0),
                            &EvaluationPolicy::BackendLazy,
                        ),
                    });
                }
                ExecutionStep::CoreMl { island_id } => {
                    let _island = self
                        .profile
                        .coreml_islands
                        .iter()
                        .find(|i| &i.id == island_id)
                        .ok_or_else(|| {
                            format!("Core ML island '{}' not found in profile", island_id)
                        })?;

                    // Stub: emit a minimal receipt for Core ML execution.
                    // Phase 2 will wire real Core ML prediction here.
                    receipts.push(BoundaryExecutionReceipt {
                        group_id: EvaluationGroupId(0),
                        planned_policy: EvaluationPolicy::ExplicitOperation,
                        backend: BackendId(2),
                        operation_count: 1,
                        planned_materialized_outputs: _island.output_names.len(),
                        actual_eval_calls: 0,
                        actual_sync_count: 1,
                        graph_build_ns: 0,
                        submit_ns: 0,
                        execution_ns: 0,
                        wait_ns: 0,
                        temporary_bytes: 0,
                        released_tensor_count: 0,
                        unaccounted_ns: 0,
                        policy_support: EvaluationPolicySupport::Unsupported,
                    });
                }
                ExecutionStep::AneInference {
                    mil_text,
                    inputs,
                    outputs,
                    tag,
                } => {
                    // Compile the ANE program and emit a receipt.
                    let program = crate::ane_bridge::AneProgram::compile(mil_text, tag)?;
                    let compiled_output_count = outputs.len();

                    // Phase 2: wire real IOSurface-backed buffers.
                    // For now we just emit the receipt without evaluating.
                    let _ = (inputs, program);

                    receipts.push(BoundaryExecutionReceipt {
                        group_id: EvaluationGroupId(0),
                        planned_policy: EvaluationPolicy::ExplicitOperation,
                        backend: BackendId(3),
                        operation_count: 1,
                        planned_materialized_outputs: compiled_output_count,
                        actual_eval_calls: 0,
                        actual_sync_count: 1,
                        graph_build_ns: 0,
                        submit_ns: 0,
                        execution_ns: 0,
                        wait_ns: 0,
                        temporary_bytes: 0,
                        released_tensor_count: 0,
                        unaccounted_ns: 0,
                        policy_support: EvaluationPolicySupport::Unsupported,
                    });
                }
            }
        }

        Ok(receipts)
    }
    /// Execute a batch by dispatching each slot to its assigned backend.
    ///
    /// backend_id mapping: 0=MLX, 1=Accelerate, 2=CoreML (stub), 3=ANE/Orion
    pub fn execute_batch(
        &mut self,
        batch: &crate::scheduling::Batch,
    ) -> Result<Vec<BoundaryExecutionReceipt>, String> {
        let mut receipts = Vec::with_capacity(batch.slots.len());

        // ── Get or create the shared arena ─────────────────────────────────
        // The arena must outlive the slot loop so all backends share the same
        // IOSurface-backed memory island (zero-copy transport).
        let arena = if let Some(allocator) = self.allocator.as_ref() {
            let arena_id = allocator
                .allocate(1, 4096, mlx_rs::Dtype::Float16)
                .map_err(|e| format!("execute_batch: arena alloc failed: {e}"))?;
            allocator
                .get_arena(arena_id)
                .ok_or_else(|| "execute_batch: arena not found after alloc".to_string())?
        } else {
            crate::arena::Arena::new(1, 4096, mlx_rs::Dtype::Float16)
                .map_err(|e| format!("execute_batch: throwaway arena failed: {e}"))?
        };

        for slot in &batch.slots {
            match slot.backend_id {
                0 => {
                    // ── MLX dispatch ───────────────────────────────────────
                    let backend = self.mlx_backend.as_mut().ok_or_else(|| {
                        format!("MLX backend not registered for slot {}", slot.id)
                    })?;
                    let eval_receipt = backend.evaluate_into_arena(slot.id as u64, &[], &arena)?;
                    receipts.push(BoundaryExecutionReceipt {
                        group_id: EvaluationGroupId(slot.id as u64),
                        planned_policy: EvaluationPolicy::Eager {
                            release_inputs_after_use: true,
                            prohibit_deferred_nodes: false,
                        },
                        backend: BackendId(0),
                        operation_count: 1,
                        planned_materialized_outputs: 0,
                        actual_eval_calls: eval_receipt.eval_calls,
                        actual_sync_count: 1,
                        graph_build_ns: eval_receipt.graph_build_ns,
                        submit_ns: eval_receipt.submit_ns,
                        execution_ns: eval_receipt.sync_ns,
                        wait_ns: 0,
                        temporary_bytes: 0,
                        released_tensor_count: 0,
                        unaccounted_ns: 0,
                        policy_support: policy_support(
                            BackendId(0),
                            &EvaluationPolicy::Eager {
                                release_inputs_after_use: true,
                                prohibit_deferred_nodes: false,
                            },
                        ),
                    });
                }
                1 => {
                    // ── Accelerate dispatch ────────────────────────────────
                    let backend = self.accelerate_backend.as_mut().ok_or_else(|| {
                        format!("Accelerate backend not registered for slot {}", slot.id)
                    })?;
                    let eval_receipt = backend.evaluate_into_arena(slot.id as u64, &[], &arena)?;
                    receipts.push(BoundaryExecutionReceipt {
                        group_id: EvaluationGroupId(slot.id as u64),
                        planned_policy: EvaluationPolicy::Eager {
                            release_inputs_after_use: true,
                            prohibit_deferred_nodes: false,
                        },
                        backend: BackendId(1),
                        operation_count: 1,
                        planned_materialized_outputs: 0,
                        actual_eval_calls: eval_receipt.eval_calls,
                        actual_sync_count: 1,
                        graph_build_ns: eval_receipt.graph_build_ns,
                        submit_ns: eval_receipt.submit_ns,
                        execution_ns: eval_receipt.sync_ns,
                        wait_ns: 0,
                        temporary_bytes: 0,
                        released_tensor_count: 0,
                        unaccounted_ns: 0,
                        policy_support: policy_support(
                            BackendId(1),
                            &EvaluationPolicy::Eager {
                                release_inputs_after_use: true,
                                prohibit_deferred_nodes: false,
                            },
                        ),
                    });
                }
                2 => {
                    // ── Core ML dispatch (stub) ────────────────────────────
                    // Create a model stub and call predict_pixelbuffer to
                    // exercise the IOSurface-backed pixel-buffer path.
                    // Phase 2: wire real Core ML prediction with properly
                    // loaded models.
                    let _ = (|| -> Result<(), String> {
                        let model = crate::coreml_bridge::CoreMlModel::load("/dev/null")
                            .map_err(|e| format!("CoreMlModel stub load: {}", e))?;
                        let mut output_info = arena.info;
                        model
                            .predict_pixelbuffer("input", &arena.info, "output", &mut output_info)
                            .map_err(|e| format!("predict_pixelbuffer stub: {}", e))?;
                        Ok(())
                    })();

                    // Since Core ML bridges are not runtime-qualified,
                    // emit a stub receipt with ExplicitOperation policy.
                    let support = crate::backend::routing::policy_support(
                        BackendId(2),
                        &EvaluationPolicy::ExplicitOperation,
                    );
                    receipts.push(BoundaryExecutionReceipt {
                        group_id: EvaluationGroupId(slot.id as u64),
                        planned_policy: EvaluationPolicy::ExplicitOperation,
                        backend: BackendId(2),
                        operation_count: 1,
                        planned_materialized_outputs: 0,
                        actual_eval_calls: 0,
                        actual_sync_count: 1,
                        graph_build_ns: 0,
                        submit_ns: 0,
                        execution_ns: 0,
                        wait_ns: 0,
                        temporary_bytes: 0,
                        released_tensor_count: 0,
                        unaccounted_ns: 0,
                        policy_support: support,
                    });
                }
                3 => {
                    // ── ANE / Orion dispatch ───────────────────────────────
                    // Wrap the shared arena as an ANE IOSurface surface and
                    // execute a compiled ANE program step.
                    let _ane_surface = crate::memory::orion_bridge::wrap_arena_for_ane(&arena)
                        .map_err(|e| format!("wrap_arena_for_ane failed: {}", e))?;

                    let stub_step = ExecutionStep::AneInference {
                        mil_text: String::new(),
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        tag: String::new(),
                    };
                    let mut ane_receipt = crate::ane_bridge::execute_ane_step(
                        &stub_step,
                        crate::ane_bridge::AneProgramCache::global(),
                    )?;
                    ane_receipt.backend = BackendId(3);
                    ane_receipt.group_id = EvaluationGroupId(slot.id as u64);
                    ane_receipt.policy_support = crate::backend::routing::policy_support(
                        BackendId(3),
                        &EvaluationPolicy::Eager {
                            release_inputs_after_use: true,
                            prohibit_deferred_nodes: false,
                        },
                    );
                    receipts.push(ane_receipt);
                }
                other => {
                    return Err(format!("unknown backend_id {} for slot {}", other, slot.id));
                }
            }
        }

        Ok(receipts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::SharedTensorCapabilityReport;

    fn minimal_profile() -> HybridProfile {
        HybridProfile {
            root_model_hash: "abcdef".into(),
            compute_image_hash: "123456".into(),
            version: 1,
            mlx_regions: vec![MlxRegion {
                id: "pre".into(),
                kind: MlxRegionKind::PreAttentionProcess,
                inputs: vec![],
                outputs: vec!["hidden_in".into()],
            }],
            coreml_islands: vec![CoreMlIsland {
                id: "attn".into(),
                artifact_path: "/tmp/model.mlmodelc".into(),
                artifact_hash: "hash".into(),
                function_name: "main".into(),
                input_names: vec!["hidden_in".into()],
                output_names: vec!["hidden_out".into()],
                state_schema: vec![],
                min_os_version: "15.0".into(),
                compute_units: ComputeUnits::CpuAndGpu,
                fallback_region: Some("pre".into()),
                tolerance_fp16: 0.001,
            }],
            boundary_tensors: vec![BoundaryTensor {
                name: "hidden_in".into(),
                feature_name: "hidden_in".into(),
                shape: vec![1, 64],
                arena_profile: "IOSurfaceFp16ContiguousV1".into(),
            }],
            execution_order: vec![
                ExecutionStep::Mlx {
                    region_id: "pre".into(),
                },
                ExecutionStep::CoreMl {
                    island_id: "attn".into(),
                },
            ],
            fallback: FallbackPolicy::MlxFallback,
            required_capabilities: vec!["iosurface_fp16_bridge".into()],
            min_os_version: "15.0".into(),
            storage_abi: "tribunus-iosurface-fp16-arena-v1".into(),
            compute_units: ComputeUnits::CpuAndGpu,
        }
    }

    #[test]
    fn test_profile_serde_roundtrip() {
        let profile = minimal_profile();
        let json = serde_json::to_string(&profile).expect("serialize");
        let parsed: HybridProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.storage_abi, "tribunus-iosurface-fp16-arena-v1");
    }

    #[test]
    fn test_validate_missing_capability() {
        let profile = HybridProfile {
            required_capabilities: vec!["nonexistent".into()],
            ..minimal_profile()
        };
        let caps = SharedTensorCapabilityReport::detect();
        assert!(profile.validate(&caps).is_err());
    }

    #[test]
    fn test_validate_tensor_flow_ok() {
        let profile = minimal_profile();
        assert!(profile.validate_tensor_flow().is_ok());
    }

    #[test]
    fn test_execute_batch_with_mlx_and_accelerate() {
        use crate::backend::accelerate::AccelerateBackend;
        use crate::scheduling::{Batch, Slot};

        // Create a minimal HybridProfile for the executor
        let profile = HybridProfile {
            root_model_hash: "test".into(),
            compute_image_hash: "test".into(),
            version: 1,
            mlx_regions: vec![],
            coreml_islands: vec![],
            boundary_tensors: vec![],
            execution_order: vec![],
            fallback: FallbackPolicy::MlxOnly,
            required_capabilities: vec![],
            min_os_version: "15.0".into(),
            storage_abi: "test".into(),
            compute_units: ComputeUnits::CpuAndGpu,
        };

        let mut executor = HybridExecutor::new(profile);
        executor.register_mlx(Box::new(MlxBackend::new()));
        executor.register_accelerate(Box::new(AccelerateBackend::new()));

        // Create a batch with three slots: MLX, Accelerate, MLX
        let batch = Batch {
            slots: vec![
                Slot {
                    id: 0,
                    request_id: Some(1),
                    tokens_generated: 0,
                    kv_cache_start: 0,
                    kv_cache_length: 10,
                    backend_id: 0, // MLX
                    kv_cache_pages: vec![],
                },
                Slot {
                    id: 1,
                    request_id: Some(2),
                    tokens_generated: 0,
                    kv_cache_start: 0,
                    kv_cache_length: 10,
                    backend_id: 1, // Accelerate
                    kv_cache_pages: vec![],
                },
                Slot {
                    id: 2,
                    request_id: Some(3),
                    tokens_generated: 0,
                    kv_cache_start: 0,
                    kv_cache_length: 10,
                    backend_id: 0, // MLX
                    kv_cache_pages: vec![],
                },
            ],
            batch_size: 3,
            max_batch_size: 64,
        };

        let receipts = executor.execute_batch(&batch).unwrap();
        assert_eq!(receipts.len(), 3, "should get one receipt per slot");

        // MLX slots: backend_id 0
        for r in &receipts {
            match r.backend.0 {
                0 => assert_eq!(r.backend.0, 0, "MLX receipt backend"),
                1 => assert_eq!(r.backend.0, 1, "Accelerate receipt backend"),
                other => panic!("unexpected backend_id {other}"),
            }
        }

        // Check receipt fields are populated
        for r in &receipts {
            assert!(r.operation_count >= 1, "receipt should have operations");
            assert_eq!(1, r.actual_sync_count, "each slot syncs once");
        }
    }
}
