//! Backend assessment pass — scores each operation against all available
//! backends and produces sealed ExecutionBoundaryPlans with cross-backend
//! tensor transfer plans.

use std::collections::HashMap;
use std::time::Instant;

use crate::backend::routing::{
    BackendId, ConversionKind, EvaluationGroupId, EvaluationPolicy, EvidenceDigest,
    ExecutionBoundaryPlan, OperationFamily, OperationId, PhysicalLayout,
    SealedExecutionBoundaryPlan, SynchronizationPolicy, TensorId, TensorTransferPlan,
};
use crate::compiler::pass::{PassIdentity, TransformPass, TransformReceipt};

// ── Graph types ────────────────────────────────────────────────────────────

/// An operation in the model graph with its characteristics.
#[derive(Debug, Clone)]
pub struct GraphOperation {
    pub id: OperationId,
    pub family: OperationFamily,
    pub m: Option<u32>,
    pub n: Option<u32>,
    pub k: Option<u32>,
    pub quantized: bool,
}

/// The input IR to the assessment pass.
#[derive(Debug, Clone)]
pub struct ModelOperationGraph {
    pub operations: Vec<GraphOperation>,
    pub operand_shapes: HashMap<OperationId, Vec<i32>>,
}

// ── Internal grouping types ────────────────────────────────────────────────

/// One group of consecutive operations on the same backend.
#[derive(Debug, Clone)]
struct BackendBlock {
    pub backend: BackendId,
    pub operation_ids: Vec<OperationId>,
}

// ── Scoring ────────────────────────────────────────────────────────────────

/// Score a single operation against a given backend.
///
/// Higher values indicate a better match.  Scores range from 10 (poor) to
/// 100 (ideal).  The default baseline of 70 for MLX reflects that MLX is the
/// primary GPU-accelerated backend; other backends have lower baselines when
/// their strengths do not apply.
fn score_backend(op: &GraphOperation, backend: BackendId) -> u32 {
    match backend.0 {
        0 => {
            // MLX — GPU-accelerated, best for matmul and fused activation ops
            match op.family {
                OperationFamily::Matmul => 100,
                OperationFamily::QuantizedMatmul => 100,
                OperationFamily::Softmax => 90,
                OperationFamily::RmsNorm => 80,
                OperationFamily::RoPE => 80,
                OperationFamily::Silu => 85,
                _ => 70,
            }
        }
        1 => {
            // Accelerate — CPU BLAS / BNNS, good for element-wise and layout ops
            match op.family {
                OperationFamily::Add | OperationFamily::Multiply => 90,
                OperationFamily::Silu => 85,
                OperationFamily::RmsNorm => 75,
                OperationFamily::Matmul => 60,
                OperationFamily::QuantizedMatmul => 80,
                OperationFamily::Transpose => 90,
                OperationFamily::Reshape => 95,
                OperationFamily::Softmax => 70,
                OperationFamily::Reduction => 75,
                _ => 40,
            }
        }
        2 => {
            // Core ML — ANE islands for attention-heavy regions
            match op.family {
                OperationFamily::AttentionBlock => 90,
                OperationFamily::MlpBlock => 70,
                OperationFamily::DecoderLayer => 90,
                _ => 30,
            }
        }
        3 => {
            // Orion / ANE private runtime — lowest-level ANE access
            match op.family {
                OperationFamily::AttentionBlock => 95,
                OperationFamily::MlpBlock => 75,
                OperationFamily::DecoderLayer => 95,
                _ => 25,
            }
        }
        _ => 10,
    }
}

// ── Assessment pass ────────────────────────────────────────────────────────

/// Compiler pass that assigns each operation group to the optimal backend and
/// produces sealed [`ExecutionBoundaryPlan`]s.
pub struct BackendAssessmentPass {
    identity: PassIdentity,
    available_backends: Vec<BackendId>,
}

impl BackendAssessmentPass {
    pub fn new(available_backends: Vec<BackendId>) -> Self {
        Self {
            identity: PassIdentity {
                name: "backend:assess".into(),
                version: "0.1.0".into(),
                implementation_digest: EvidenceDigest("backend-assessment-v0.1.0".into()),
            },
            available_backends,
        }
    }

    /// Return the backends this pass was configured with.
    pub fn available_backends(&self) -> &[BackendId] {
        &self.available_backends
    }
}

impl TransformPass<ModelOperationGraph> for BackendAssessmentPass {
    fn identity(&self) -> &PassIdentity {
        &self.identity
    }

    fn applies_to(&self, ir: &ModelOperationGraph) -> bool {
        !ir.operations.is_empty()
    }

    fn apply(
        &self,
        ir: &ModelOperationGraph,
        input_digest: EvidenceDigest,
    ) -> (ModelOperationGraph, TransformReceipt) {
        let start = Instant::now();

        let mut assignments: Vec<(OperationId, BackendId)> =
            Vec::with_capacity(ir.operations.len());
        for op in &ir.operations {
            let best = self
                .available_backends
                .iter()
                .copied()
                .max_by_key(|&b| score_backend(op, b))
                .unwrap_or(BackendId(0));
            assignments.push((op.id, best));
        }

        // Group consecutive same-backend ops
        let groups = build_groups(&assignments);

        let plan_count = groups.len() as u64;
        let mut backend_ids: Vec<u32> = groups.iter().map(|g| g.backend.0).collect();
        backend_ids.dedup();
        let unique_backends = backend_ids.len() as u64;

        let duration_ns = start.elapsed().as_nanos() as u64;

        let receipt = TransformReceipt {
            pass: self.identity.clone(),
            input_digest,
            output_digest: EvidenceDigest("assessed".into()),
            rewrites_applied: plan_count,
            rewrites_rejected: 0,
            rewrite_descriptions: vec![format!(
                "assigned {} ops to {} backends in {} groups",
                ir.operations.len(),
                unique_backends,
                plan_count,
            )],
            reached_fixpoint: true,
            duration_ns,
            equivalence_claimed: true,
            equivalence_evidence: None,
        };

        (ir.clone(), receipt)
    }
}

// ── Public entry point ─────────────────────────────────────────────────────

/// Run the assessment pass on a model graph and produce sealed plans together
/// with cross-backend transfer plans.
///
/// This is the public entry point for backend assessment outside the compiler
/// pipeline.  Returns the sealed execution-boundary plans and any tensor
/// transfer plans needed for cross-backend data movement.
pub fn assess_and_route(
    graph: &ModelOperationGraph,
    available_backends: &[BackendId],
) -> Result<(Vec<SealedExecutionBoundaryPlan>, Vec<TensorTransferPlan>), String> {
    if graph.operations.is_empty() {
        return Err("cannot assess an empty operation graph".into());
    }
    if available_backends.is_empty() {
        return Err("at least one backend must be available".into());
    }

    // 1. Score and assign each operation to its best backend
    let mut assignments: Vec<(OperationId, BackendId)> = Vec::with_capacity(graph.operations.len());
    for op in &graph.operations {
        let best = available_backends
            .iter()
            .copied()
            .max_by_key(|&b| score_backend(op, b))
            .unwrap_or(BackendId(0));
        assignments.push((op.id, best));
    }

    // 2. Group consecutive same-backend ops
    let groups = build_groups(&assignments);

    // 3. Create ExecutionBoundaryPlans
    let mut plans: Vec<SealedExecutionBoundaryPlan> = Vec::with_capacity(groups.len());
    for (i, block) in groups.iter().enumerate() {
        let mut materialized_outputs: Vec<TensorId> = Vec::new();

        // Cross-backend boundary: materialise the last op's output
        if i + 1 < groups.len() && groups[i + 1].backend != block.backend {
            if let Some(&last_op) = block.operation_ids.last() {
                materialized_outputs.push(TensorId(last_op.0));
            }
        }

        // Last group: materialise every output so the host can read them
        if i == groups.len() - 1 {
            for op_id in &block.operation_ids {
                materialized_outputs.push(TensorId(op_id.0));
            }
        }

        let plan = ExecutionBoundaryPlan {
            group_id: EvaluationGroupId(i as u64),
            backend_id: block.backend,
            operations: block.operation_ids.clone(),
            materialized_outputs,
            policy: EvaluationPolicy::ExplicitRegion,
            synchronization: if i > 0 {
                SynchronizationPolicy::Barrier
            } else {
                SynchronizationPolicy::None
            },
            release_after: Vec::new(),
            content_digest: None,
        };

        plans.push(SealedExecutionBoundaryPlan::seal(plan));
    }

    // 4. Create transfer plans for cross-backend boundaries
    let mut transfers: Vec<TensorTransferPlan> = Vec::new();
    for i in 1..groups.len() {
        if groups[i].backend != groups[i - 1].backend {
            if let Some(&output_op) = groups[i - 1].operation_ids.last() {
                transfers.push(TensorTransferPlan {
                    source_backend: groups[i - 1].backend,
                    destination_backend: groups[i].backend,
                    tensor_id: TensorId(output_op.0),
                    source_layout: PhysicalLayout::RowMajor,
                    destination_layout: PhysicalLayout::RowMajor,
                    conversion: ConversionKind::SharedReference,
                    expected_bytes: 0,
                    synchronization_before: true,
                    synchronization_after: true,
                });
            }
        }
    }

    Ok((plans, transfers))
}

// ── Re-exports ─────────────────────────────────────────────────────────────

pub use self::assess_and_route as assess_model_ops;

// ── Internal helpers ───────────────────────────────────────────────────────

/// Split a sequence of (op, backend) assignments into consecutive same-backend
/// blocks.
fn build_groups(assignments: &[(OperationId, BackendId)]) -> Vec<BackendBlock> {
    if assignments.is_empty() {
        return Vec::new();
    }

    let mut groups: Vec<BackendBlock> = Vec::new();
    let mut current_backend = assignments[0].1;
    let mut current_ids: Vec<OperationId> = Vec::new();

    for &(op_id, backend) in assignments {
        if backend != current_backend {
            groups.push(BackendBlock {
                backend: current_backend,
                operation_ids: std::mem::take(&mut current_ids),
            });
            current_backend = backend;
        }
        current_ids.push(op_id);
    }

    if !current_ids.is_empty() {
        groups.push(BackendBlock {
            backend: current_backend,
            operation_ids: current_ids,
        });
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::routing::{EvaluationPolicy, SynchronizationPolicy};

    // Helper to create a test operation
    fn make_op(
        id: u64,
        family: OperationFamily,
        m: Option<u32>,
        n: Option<u32>,
        k: Option<u32>,
        quantized: bool,
    ) -> GraphOperation {
        GraphOperation {
            id: OperationId(id),
            family,
            m,
            n,
            k,
            quantized,
        }
    }

    #[test]
    fn test_score_backend_mlx_matmul_ideal() {
        let op = make_op(
            1,
            OperationFamily::Matmul,
            Some(4096),
            Some(4096),
            Some(4096),
            false,
        );
        assert_eq!(score_backend(&op, BackendId(0)), 100);
    }

    #[test]
    fn test_score_backend_accelerate_transpose() {
        let op = make_op(1, OperationFamily::Transpose, None, None, None, false);
        assert_eq!(score_backend(&op, BackendId(1)), 90);
    }

    #[test]
    fn test_score_backend_coreml_attention() {
        let op = make_op(1, OperationFamily::AttentionBlock, None, None, None, false);
        assert_eq!(score_backend(&op, BackendId(2)), 90);
    }

    #[test]
    fn test_score_backend_orion_decoder_layer() {
        let op = make_op(1, OperationFamily::DecoderLayer, None, None, None, false);
        assert_eq!(score_backend(&op, BackendId(3)), 95);
    }

    #[test]
    fn test_score_backend_unknown_backend() {
        let op = make_op(1, OperationFamily::Matmul, None, None, None, false);
        assert_eq!(score_backend(&op, BackendId(99)), 10);
    }

    #[test]
    fn test_empty_graph_rejected() {
        let graph = ModelOperationGraph {
            operations: vec![],
            operand_shapes: HashMap::new(),
        };
        assert!(assess_and_route(&graph, &[BackendId(0)]).is_err());
    }

    #[test]
    fn test_no_backends_rejected() {
        let graph = ModelOperationGraph {
            operations: vec![make_op(1, OperationFamily::Matmul, None, None, None, false)],
            operand_shapes: HashMap::new(),
        };
        assert!(assess_and_route(&graph, &[]).is_err());
    }

    #[test]
    fn test_single_op_single_backend() {
        let graph = ModelOperationGraph {
            operations: vec![make_op(1, OperationFamily::Matmul, None, None, None, false)],
            operand_shapes: HashMap::new(),
        };
        let (plans, transfers) = assess_and_route(&graph, &[BackendId(0)]).unwrap();
        assert_eq!(plans.len(), 1);
        assert!(transfers.is_empty());
        assert_eq!(plans[0].plan.backend_id, BackendId(0));
        assert_eq!(plans[0].plan.operations.len(), 1);
        assert_eq!(plans[0].plan.group_id.0, 0);
    }

    #[test]
    fn test_consecutive_same_backend_grouped() {
        let graph = ModelOperationGraph {
            operations: vec![
                make_op(1, OperationFamily::Matmul, None, None, None, false),
                make_op(2, OperationFamily::Silu, None, None, None, false),
                make_op(3, OperationFamily::Softmax, None, None, None, false),
            ],
            operand_shapes: HashMap::new(),
        };
        let (plans, transfers) = assess_and_route(&graph, &[BackendId(0)]).unwrap();
        // All ops on MLX (0) → one group
        assert_eq!(plans.len(), 1);
        assert!(transfers.is_empty());
        assert_eq!(plans[0].plan.operations.len(), 3);
    }

    #[test]
    fn test_cross_backend_groups_and_transfers() {
        // Force a cross-backend scenario: op1 is best on MLX, op2 on Accelerate,
        // op3 on MLX again.  This creates three groups with two transfers.
        let op1 = make_op(1, OperationFamily::Matmul, None, None, None, false); // best: MLX(100) > Accel(60)
        let op2 = make_op(2, OperationFamily::Transpose, None, None, None, false); // best: Accel(90) > MLX(70)
        let op3 = make_op(3, OperationFamily::Silu, None, None, None, false); // best: MLX(85) > Accel(85) — MLX wins default

        let graph = ModelOperationGraph {
            operations: vec![op1, op2, op3],
            operand_shapes: HashMap::new(),
        };

        let (plans, transfers) = assess_and_route(&graph, &[BackendId(0), BackendId(1)]).unwrap();
        // MLX → Accelerate → MLX = 3 groups, 2 transfers
        assert_eq!(plans.len(), 3);
        assert_eq!(transfers.len(), 2);

        // First group: MLX
        assert_eq!(plans[0].plan.backend_id, BackendId(0));
        assert_eq!(plans[0].plan.operations, vec![OperationId(1)]);
        assert_eq!(plans[0].plan.synchronization, SynchronizationPolicy::None);

        // Second group: Accelerate
        assert_eq!(plans[1].plan.backend_id, BackendId(1));
        assert_eq!(plans[1].plan.operations, vec![OperationId(2)]);
        assert_eq!(
            plans[1].plan.synchronization,
            SynchronizationPolicy::Barrier
        );

        // Third group: MLX
        assert_eq!(plans[2].plan.backend_id, BackendId(0));
        assert_eq!(plans[2].plan.operations, vec![OperationId(3)]);
        assert_eq!(
            plans[2].plan.synchronization,
            SynchronizationPolicy::Barrier
        );

        // Ti: MLX→Accel transfer for tensor of op1 (id=1)
        assert_eq!(transfers[0].source_backend, BackendId(0));
        assert_eq!(transfers[0].destination_backend, BackendId(1));
        assert_eq!(transfers[0].tensor_id, TensorId(1));

        // Transfer: Accel→MLX for tensor of op2 (id=2)
        assert_eq!(transfers[1].source_backend, BackendId(1));
        assert_eq!(transfers[1].destination_backend, BackendId(0));
        assert_eq!(transfers[1].tensor_id, TensorId(2));
    }

    #[test]
    fn test_last_group_materializes_all_outputs() {
        let graph = ModelOperationGraph {
            operations: vec![
                make_op(1, OperationFamily::Matmul, None, None, None, false),
                make_op(2, OperationFamily::Silu, None, None, None, false),
            ],
            operand_shapes: HashMap::new(),
        };
        let (plans, _) = assess_and_route(&graph, &[BackendId(0)]).unwrap();
        assert_eq!(plans.len(), 1);
        // Last (and only) group materialises all outputs
        assert!(plans[0].plan.materialized_outputs.contains(&TensorId(1)));
        assert!(plans[0].plan.materialized_outputs.contains(&TensorId(2)));
    }

    #[test]
    fn test_assess_model_ops_reexport() {
        let graph = ModelOperationGraph {
            operations: vec![make_op(1, OperationFamily::Matmul, None, None, None, false)],
            operand_shapes: HashMap::new(),
        };
        let result = assess_model_ops(&graph, &[BackendId(0)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transform_pass_applies_to() {
        let pass = BackendAssessmentPass::new(vec![BackendId(0)]);

        let empty = ModelOperationGraph {
            operations: vec![],
            operand_shapes: HashMap::new(),
        };
        assert!(!pass.applies_to(&empty));

        let non_empty = ModelOperationGraph {
            operations: vec![make_op(1, OperationFamily::Matmul, None, None, None, false)],
            operand_shapes: HashMap::new(),
        };
        assert!(pass.applies_to(&non_empty));
    }

    #[test]
    fn test_transform_pass_apply() {
        let pass = BackendAssessmentPass::new(vec![BackendId(0)]);
        let input_digest = EvidenceDigest("test-input".into());

        let graph = ModelOperationGraph {
            operations: vec![
                make_op(1, OperationFamily::Matmul, None, None, None, false),
                make_op(2, OperationFamily::Silu, None, None, None, false),
            ],
            operand_shapes: HashMap::new(),
        };

        let (_output, receipt) = pass.apply(&graph, input_digest);
        assert_eq!(receipt.pass.name, "backend:assess");
        assert!(receipt.rewrites_applied > 0);
        assert!(receipt.reached_fixpoint);
        assert!(receipt.equivalence_claimed);
        assert_eq!(receipt.input_digest, EvidenceDigest("test-input".into()));
    }

    #[test]
    fn test_sealed_plan_verifies() {
        let graph = ModelOperationGraph {
            operations: vec![make_op(1, OperationFamily::Matmul, None, None, None, false)],
            operand_shapes: HashMap::new(),
        };
        let (plans, _) = assess_and_route(&graph, &[BackendId(0)]).unwrap();
        assert_eq!(plans.len(), 1);
        assert!(plans[0].verify());
    }
}
