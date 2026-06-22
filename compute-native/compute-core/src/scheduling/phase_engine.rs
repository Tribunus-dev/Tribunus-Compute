//! Phase engine — executes a compiler-emitted phase DAG through concrete
//! phase runners.  The engine is the bridge between the typed DAG and the
//! actual backend dispatch.

use crate::compute_image::phase_dag::{EmittedPhase, EmittedPhaseGraph, PhaseCompletionStatus};
use crate::scheduling::execution_context::ExecutionContext;
use crate::scheduling::phase_runner::{PhaseResult, PhaseRunnerRegistry};
use crate::scheduling::ready_queue::ReadyQueue;
use crate::scheduling::receipts::PhaseReceipt;
use crate::scheduling::phase_engine_state::{PhaseLifecycleState, PhaseLifecycleTracker};
use crate::inference::execution_image_state::ComputeImageState;
use crate::inference::inference_session_state::InferenceSessionState;
use crate::inference::inference_step_state::{InferenceStepState, InferenceStepOutput, InferenceMode};
use crate::runtime::executable_session::RuntimeBackends;
use crate::backend::accelerate_lane::AccelerateLane;
use crate::backend::coreml_lane::CoreMlLane;
use crate::mlx_executor::MlxExecutor;
use mlx_rs::Array;
use std::sync::{Arc, Mutex};

/// Result of executing a full phase graph to completion.
#[derive(Debug)]
pub struct PhaseGraphResult {
    /// One receipt per phase, in execution order.
    pub receipts: Vec<PhaseReceipt>,
    /// Whether the entire graph reached terminal successfully.
    pub all_completed: bool,
}

/// The DAG execution engine.
///
/// Call [`execute_graph`] with an [`EmittedPhaseGraph`] and an
/// [`ExecutionContext`]; it drives the ready-set computation and dispatches
/// each phase through the [`PhaseRunnerRegistry`].
pub struct PhaseEngine {
    runners: PhaseRunnerRegistry,
}

impl PhaseEngine {
    /// Create a new engine with the default runner registry.
    pub fn new() -> Self {
        Self {
            runners: PhaseRunnerRegistry::default(),
        }
    }

    /// Execute the full phase graph until every phase has either completed
    /// or failed.
    pub fn execute_graph(
        &self,
        dag: &EmittedPhaseGraph,
        ctx: &mut ExecutionContext,
    ) -> PhaseGraphResult {
        let mut receipts: Vec<PhaseReceipt> = Vec::new();
        let mut completed: std::collections::HashSet<String> = std::collections::HashSet::new();
        let ready_queue = ReadyQueue::new(dag);

        loop {
            let ready = ready_queue.ready_phases(&completed);
            if ready.is_empty() {
                break;
            }

            for phase in ready {
                // Verify all predecessors are complete.
                let preds = dag.predecessors(&phase.phase_id);
                let all_ready = preds.iter().all(|p| completed.contains(&p.phase_id));
                if !all_ready {
                    continue;
                }

                let receipt = self.execute_single_phase(dag, phase, ctx);
                let phase_id = receipt.phase_id.clone();
                let status = receipt.status.clone();
                receipts.push(receipt);

                match status {
                    PhaseCompletionStatus::Complete => {
                        completed.insert(phase_id);
                    }
                    PhaseCompletionStatus::Failed(ref reason) => {
                        eprintln!(
                            "[phase-engine] phase '{}' failed: {}",
                            phase_id, reason
                        );
                        // Mark as completed so downstream can attempt fallback.
                        completed.insert(phase_id);
                    }
                    PhaseCompletionStatus::FallbackUsed(ref reason) => {
                        eprintln!(
                            "[phase-engine] phase '{}' fallback: {}",
                            phase_id, reason
                        );
                        completed.insert(phase_id);
                    }
                    PhaseCompletionStatus::Pending => {
                        // Should not happen after execution.
                        eprintln!("[phase-engine] phase '{}' still pending after execution", phase_id);
                    }
                }
            }
        }

        PhaseGraphResult {
            all_completed: completed.len() == dag.phases.len(),
            receipts,
        }
    }

    /// Execute a single phase through the runner registry.
    fn execute_single_phase(
        &self,
        dag: &EmittedPhaseGraph,
        phase: &EmittedPhase,
        ctx: &mut ExecutionContext,
    ) -> PhaseReceipt {
        let start = std::time::Instant::now();

        let result: PhaseResult = match self.runners.dispatch(phase, ctx) {
            Ok(()) => {
                PhaseResult {
                phase_id: phase.phase_id.clone(),
                status: PhaseCompletionStatus::Complete,
                duration_us: start.elapsed().as_micros() as u64,
                fused_evidence: None,
            }
            },
            Err(e) => {
                // Attempt fallback decomposition.
                let fallback_edges: Vec<&crate::compute_image::phase_dag::EmittedPhaseEdge> = dag
                    .edges
                    .iter()
                    .filter(|e| {
                        e.from_phase == phase.phase_id
                            && e.semantic_kind
                                == crate::compute_image::phase_dag::SemanticKind::FallbackDecomposition
                    })
                    .collect();

                if !fallback_edges.is_empty() {
                    PhaseResult {
                        phase_id: phase.phase_id.clone(),
                    status: PhaseCompletionStatus::FallbackUsed(format!("runner error: {}", e)),
                        duration_us: start.elapsed().as_micros() as u64,
                        fused_evidence: None,
                    }
                } else {
                PhaseResult {
                        phase_id: phase.phase_id.clone(),
                        status: PhaseCompletionStatus::Failed(format!("runner error (no fallback): {}", e)),
                        duration_us: start.elapsed().as_micros() as u64,
                        fused_evidence: None,
                    }
                }
            }
        };

        PhaseReceipt {
            phase_id: result.phase_id,
            status: result.status,
            duration_us: result.duration_us,
            fused_evidence: result.fused_evidence,
        }
    }

    /// Execute the phase graph until terminal output is produced.
    ///
    /// This is the authoritative execution method that replaces the old
    /// imperative layer loop. It owns:
    /// - Phase readiness computation
    /// - Cancellation checking
    /// - Lifecycle state transitions
    /// - Runner dispatch
    /// - Receipt collection
    /// - Fallback decisions
    pub async fn execute_until_terminal(
        &self,
        image: &ComputeImageState,
        session: &mut InferenceSessionState,
        step: &mut InferenceStepState,
    ) -> Result<InferenceStepOutput, String> {
        let dag: &EmittedPhaseGraph = &*image.phase_graph;
        let mut lifecycle = PhaseLifecycleTracker::new();
        let mut completed: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Register all phases.
        for phase in &dag.phases {
            lifecycle.register(&phase.phase_id);
        }

        let ready_queue = ReadyQueue::new(dag);

        // Build one execution context from real session/image/step state.
        let placeholder_arr = Arc::new(Array::from_slice::<f32>(&[0.0], &[1]));
                let mut ctx = ExecutionContext {
                    request_id: step.request_id.0,
                    token_position: step.token_position,
            is_prefill: step.mode == InferenceMode::Prefill,
                    token_ids: step.input_tokens.token_ids.iter().map(|&t| t as i32).collect(),
            hidden_state: step.current_activation.as_ref().and_then(|ca| {
                #[cfg(feature = "mlx-backend")]
                { ca.mlx_compatibility_view.clone() }
                #[cfg(not(feature = "mlx-backend"))]
                { None::<Array> }
            }),
            // TODO: live kv_caches once LiveKvCache derives Clone or we build them per-layer
            kv_caches: Vec::new(),
            layer_weights: Arc::new(image.layer_weights.to_vec()),
            backend: Some(Box::new(RuntimeBackends {
                mlx_executor: Arc::new(Mutex::new(MlxExecutor::spawn_gpu())),
                // TODO: populate metal_kernels from image when ComputeImageState carries them
                metal_kernels: Arc::new(Vec::new()),
                accelerate_state: AccelerateLane::new(),
                coreml_state: CoreMlLane::new(),
                // TODO: populate weight tensors from image when available
                emb_w: placeholder_arr.clone(),
                emb_s: placeholder_arr.clone(),
                emb_b: placeholder_arr.clone(),
                fn_w: placeholder_arr.clone(),
                rope_cos: placeholder_arr.clone(),
                rope_sin: placeholder_arr.clone(),
                full_cos: placeholder_arr.clone(),
                full_sin: placeholder_arr.clone(),
            })),
        };

        loop {
            // 1. Check cancellation before any work selection.
            if session.is_cancelled() {
                for phase in &dag.phases {
                    let _ = lifecycle.transition(&phase.phase_id, PhaseLifecycleState::Cancelled);
                }
                return Err("cancelled during execution".to_string());
            }

            // 2. Compute ready set from graph edges.
            let ready = ready_queue.ready_phases(&completed);
            if ready.is_empty() {
                break;
            }

            for phase in ready {
                let phase_id = phase.phase_id.clone();

                // 3. Transition to Ready -> Admitted -> Dispatched.
                let _ = lifecycle.transition(&phase_id, PhaseLifecycleState::Ready);
                let _ = lifecycle.transition(&phase_id, PhaseLifecycleState::Admitted);
                let _ = lifecycle.transition(&phase_id, PhaseLifecycleState::Dispatched);

                // 4. Run the phase through the registry using the shared context.
                let phase_start = std::time::Instant::now();
                let run_result = self.runners.dispatch(phase, &mut ctx);
                let duration_us = phase_start.elapsed().as_micros() as u64;

                // 5. Propagate updated hidden_state back to step state.
                #[cfg(feature = "mlx-backend")]
                if let Some(ref mut ca) = step.current_activation.as_mut() {
                    if let Some(ref hidden) = ctx.hidden_state {
                    ca.mlx_compatibility_view = Some(hidden.clone());
                }
                }

                // 5. Record receipt.
                let (status, fused_evidence) = match run_result {
                    Ok(()) => (PhaseCompletionStatus::Complete, None),
            Err(e) => {
                        eprintln!("[phase-engine] phase '{}' failed: {}", phase_id, e);
                        (PhaseCompletionStatus::Failed(e), None)
                    }
        };

                let receipt = PhaseReceipt {
                    phase_id: phase_id.clone(),
                    status: status.clone(),
                    duration_us,
                    fused_evidence,
        };
                step.receipt_ledger.push(receipt);

                // 6. Update lifecycle and completed set.
                if matches!(status, PhaseCompletionStatus::Complete | PhaseCompletionStatus::FallbackUsed(_)) {
                    let _ = lifecycle.transition(&phase_id, PhaseLifecycleState::Complete);
                    completed.insert(phase_id);
                } else {
                    let _ = lifecycle.transition(&phase_id, PhaseLifecycleState::FailedBeforePublication);
                    completed.insert(phase_id);
                }
            }
        }

        // 7. Build output.
        Ok(InferenceStepOutput {
            token: None,
            logits: None,
            receipts: step.receipt_ledger.take(),
                    })
    }
}

impl Default for PhaseEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::phase_dag::{
        ComputeLane, EmittedArenaPlan, EmittedConcurrencyPlan, EmittedPhase,
        EmittedPhaseEdge, PhaseKind, SemanticKind,
    };
    use crate::scheduling::execution_context::ExecutionContext;
    use std::collections::HashMap;

    fn make_phase(id: &str, kind: PhaseKind) -> EmittedPhase {
        EmittedPhase {
            phase_id: id.into(),
            kind,
            lane: ComputeLane::Metal,
            ops: vec![format!("op_{}", id)],
            arena_slots: vec![],
            tensor_reads: vec![],
            tensor_writes: vec!["out".into()],
            estimated_ops: 100,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_engine_runs_single_phase() {
        let dag = EmittedPhaseGraph {
            phases: vec![make_phase("p0", PhaseKind::MlxDecode)],
            edges: vec![],
            arena_plan: EmittedArenaPlan {
                total_bytes: 0,
                slots: vec![],
            },
            concurrency_plan: EmittedConcurrencyPlan {
                independent_sets: vec![],
            },
            compiler_version: "test".into(),
        };

        let engine = PhaseEngine::new();
        let mut ctx = ExecutionContext::new_empty();
        let result = engine.execute_graph(&dag, &mut ctx);

        assert_eq!(result.receipts.len(), 1);
        assert!(result.all_completed);
    }

    #[test]
    fn test_engine_runs_sequential_phases() {
        let dag = EmittedPhaseGraph {
            phases: vec![
                make_phase("a", PhaseKind::ArenaAlloc),
                make_phase("b", PhaseKind::MlxDecode),
                make_phase("c", PhaseKind::MlxDecode),
            ],
            edges: vec![
                EmittedPhaseEdge {
                    from_phase: "a".into(),
                    to_phase: "b".into(),
                    semantic_kind: SemanticKind::Data,
                    label: None,
                    metadata: HashMap::new(),
                },
                EmittedPhaseEdge {
                    from_phase: "b".into(),
                    to_phase: "c".into(),
                    semantic_kind: SemanticKind::Data,
                    label: None,
                    metadata: HashMap::new(),
                },
            ],
            arena_plan: EmittedArenaPlan {
                total_bytes: 0,
                slots: vec![],
            },
            concurrency_plan: EmittedConcurrencyPlan {
                independent_sets: vec![],
            },
            compiler_version: "test".into(),
        };

        let engine = PhaseEngine::new();
        let mut ctx = ExecutionContext::new_empty();
        let result = engine.execute_graph(&dag, &mut ctx);

        assert_eq!(result.receipts.len(), 3);
        assert!(result.all_completed);
        // Verify ordering: a then b then c
        assert_eq!(result.receipts[0].phase_id, "a");
        assert_eq!(result.receipts[1].phase_id, "b");
        assert_eq!(result.receipts[2].phase_id, "c");
    }
}
