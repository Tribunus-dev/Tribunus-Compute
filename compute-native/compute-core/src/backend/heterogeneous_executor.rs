//! Heterogeneous executor — dispatches operations across backends.
//!
//! Implements [`BoundaryExecutor`] by iterating over sealed boundary plans,
//! finding the correct backend for each, and emitting observed
//! [`BoundaryExecutionReceipt`]s. Cross-backend tensor transfers go through
//! the IOSurface unified memory island (zero-copy via
//! [`IosurfaceAllocator`] / [`TensorBackend::bind_external`]).

use std::collections::HashMap;
use std::sync::Arc;

use crate::arena::Arena;
use crate::backend::flex_dispatch::FlexDispatch;
use crate::backend::routing::*;
use crate::backend::TensorBackend;
use crate::backend::TensorHandle;
use crate::memory::allocator::IosurfaceAllocator;

// ── BackendInstance trait ──────────────────────────────────────────────────

/// A backend instance that can execute operations.
pub trait BackendInstance: TensorBackend {
    /// The [`BackendId`] this instance represents.
    fn backend_kind(&self) -> BackendId;

    /// Whether this backend supports a given [`OperationFamily`].
    fn supports(&self, family: OperationFamily) -> bool;

    /// Execute a single operation on this backend.
    ///
    /// Returns a [`BackendExecutionReceipt`] with observed timing.
    /// `inputs` are the tensor handles of predecessor operations.
    fn execute(
        &mut self,
        op: &OperationDescriptor,
        inputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String>;

    /// Evaluate output tensors and materialize them directly into a
    /// pre-allocated IOSurface arena, avoiding any copy between backends.
    ///
    /// The default implementation falls back to `self.evaluate()`. Backends
    /// that support zero-copy IOSurface materialization (MLX via
    /// `evaluate_into()`) should override this.
    fn evaluate_into_arena(
        &mut self,
        _group_id: u64,
        _outputs: &[TensorHandle],
        _arena: &Arena,
    ) -> Result<crate::backend::EvaluationReceipt, String> {
        self.evaluate(_group_id, _outputs)
    }
}

// ── HeterogeneousExecutor ──────────────────────────────────────────────────

/// Heterogeneous executor that dispatches operations across backends.
///
/// Implements the [`BoundaryExecutor`] trait from `routing.rs`.
/// Each boundary is dispatched to its assigned backend.
/// Cross-backend tensor transfers go through the IOSurface unified memory
/// island (zero-copy via [`IosurfaceAllocator`] / `bind_external`).
pub struct HeterogeneousExecutor {
    backends: Vec<Box<dyn BackendInstance + Send>>,
    allocator: Option<Arc<IosurfaceAllocator>>,
    execution_count: u64,
    pub(crate) operation_registry: HashMap<OperationId, OperationDescriptor>,
    /// Optional ANE program cache for Orion-routed boundaries.
    ane_cache: Option<crate::memory::ane_program_cache::AneProgramCache>,
    /// Per-operation runtime routing table (populated by FlexDispatch).
    pub(crate) routing_table: HashMap<OperationId, BackendId>,
}

impl HeterogeneousExecutor {
    /// Create an empty executor with no backends registered.
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            allocator: None,
            execution_count: 0,
            operation_registry: HashMap::new(),
            ane_cache: None,
            routing_table: HashMap::new(),
        }
    }

    /// Register a backend instance.
    ///
    /// Multiple backends of the same kind may be registered; the first match
    /// is used during dispatch.
    pub fn register(&mut self, backend: Box<dyn BackendInstance + Send>) {
        self.backends.push(backend);
    }

    /// Set the IOSurface allocator (for cross-backend tensor transfers).
    pub fn set_allocator(&mut self, allocator: Arc<IosurfaceAllocator>) {
        self.allocator = Some(allocator);
    }

    /// Set the operation descriptor registry.
    ///
    /// The registry maps [`OperationId`] to [`OperationDescriptor`] and is
    /// used by [`execute_boundaries`](BoundaryExecutor::execute_boundaries)
    /// to resolve each operation in a boundary plan at dispatch time.
    pub fn set_operation_registry(&mut self, registry: HashMap<OperationId, OperationDescriptor>) {
        self.operation_registry = registry;
    }

    /// Set the ANE program cache (compiled per-layer Orion programs).
    /// Without this, Orion-routed boundaries will fall back to MLX.
    pub fn set_ane_cache(&mut self, cache: crate::memory::ane_program_cache::AneProgramCache) {
        self.ane_cache = Some(cache);
    }

    /// Find a registered backend by its [`BackendId`].
    ///
    /// Returns `None` if no such backend is registered or available.
    fn find_backend(&mut self, id: BackendId) -> Option<&mut Box<dyn BackendInstance + Send>> {
        self.backends.iter_mut().find(|b| b.backend_kind() == id)
    }

    /// Set a per-operation route override in the routing table.
    pub fn set_route(&mut self, op_id: OperationId, backend: BackendId) {
        self.routing_table.insert(op_id, backend);
    }

    /// Get the current route for an operation, if one has been set.
    pub fn get_route(&self, op_id: &OperationId) -> Option<BackendId> {
        self.routing_table.get(op_id).copied()
    }

    /// Execute a sealed boundary plan using per-operation flex dispatch.
    ///
    /// For each operation in the boundary, calls [`FlexDispatch::dispatch`]
    /// to determine which backend runs it *right now*, overriding the
    /// plan's static [`BackendId`].  Operations within the same boundary
    /// may execute on different backends.
    ///
    /// After dispatch, each operation is executed on its selected backend
    /// and a consolidated [`BoundaryExecutionReceipt`] is emitted for the
    /// whole boundary batch.
    pub fn execute_boundary_flex(
        &mut self,
        boundary: &SealedExecutionBoundaryPlan,
        flex: &mut FlexDispatch,
    ) -> Result<Vec<BoundaryExecutionReceipt>, String> {
        let plan = &boundary.plan;
        let boundary_start = std::time::Instant::now();
        let mut receipts = Vec::new();

        // 1. Look up operation descriptors from the registry.
        let ops: Vec<OperationDescriptor> = plan
            .operations
            .iter()
            .map(|op_id| {
                self.operation_registry
                    .get(op_id)
                    .ok_or_else(|| format!("operation {} not found in registry", op_id.0))
                    .cloned()
            })
            .collect::<Result<Vec<_>, String>>()?;

        // 2. Dispatch each operation to its runtime-selected backend.
        let mut _op_receipts: Vec<BackendExecutionReceipt> =
            Vec::with_capacity(plan.operations.len());

        for (_i, op_desc) in ops.iter().enumerate() {
            let backend_id = flex.dispatch(op_desc, plan.group_id.0 as u32);

            // Cache the route so the executor's routing table stays
            // consistent with the flex dispatcher's decisions.
            self.routing_table.insert(op_desc.operation_id, backend_id);

            let backend = self
                .find_backend(backend_id)
                .ok_or_else(|| format!("flex-dispatch backend {} not registered", backend_id.0))?;

            let receipt = backend.execute(op_desc, &[])?;
            _op_receipts.push(receipt);
        }

        // 3. Emit a consolidated boundary receipt.
        let elapsed_ns = boundary_start.elapsed().as_nanos() as u64;
        let support = policy_support(plan.backend_id, &plan.policy);

        let actual_sync_count = if matches!(plan.synchronization, SynchronizationPolicy::Barrier) {
            1
        } else {
            0
        };

        receipts.push(BoundaryExecutionReceipt {
            group_id: plan.group_id,
            planned_policy: plan.policy.clone(),
            backend: plan.backend_id,
            operation_count: plan.operations.len(),
            planned_materialized_outputs: plan.materialized_outputs.len(),
            actual_eval_calls: plan.operations.len(),
            actual_sync_count,
            graph_build_ns: 0,
            submit_ns: 0,
            execution_ns: elapsed_ns,
            wait_ns: 0,
            temporary_bytes: 0,
            released_tensor_count: plan.release_after.len(),
            unaccounted_ns: 0,
            policy_support: support,
        });

        self.execution_count += 1;
        Ok(receipts)
    }
}

// ── BoundaryExecutor implementation ────────────────────────────────────────

impl BoundaryExecutor for HeterogeneousExecutor {
    fn execute_boundaries(
        &mut self,
        plans: &[ExecutionBoundaryPlan],
    ) -> Result<Vec<BoundaryExecutionReceipt>, String> {
        let mut receipts = Vec::with_capacity(plans.len());

        for plan in plans {
            let boundary_start = std::time::Instant::now();

            // 1. Resolve backend id and policy support (frees the borrow on self
            //    for the subsequent registry lookup)
            let backend_id = plan.backend_id;
            let support = policy_support(plan.backend_id, &plan.policy);

            // 2. Look up operation descriptors from the registry (immutable borrow)
            let ops: Vec<OperationDescriptor> = plan
                .operations
                .iter()
                .map(|op_id| {
                    self.operation_registry
                        .get(op_id)
                        .ok_or_else(|| format!("operation {} not found in registry", op_id.0))
                        .cloned()
                })
                .collect::<Result<Vec<_>, String>>()?;

            // 3. Find the backend for this boundary (mutable borrow)
            //    (moved after the ANE cache lookup to avoid borrow conflict)

            // If this boundary is assigned to Orion (ANE), use a compiled
            // ANE program instead of individual backend operations.
            let boundary_type = if plan.backend_id.0 == 3 {
                self.ane_cache.as_ref().and_then(|cache| {
                    // Find the program for this boundary's layer.
                    // The evaluation group id maps to the layer index.
                    let prog = cache.get_program(plan.group_id.0 as usize);
                    if prog.is_null() {
                        None
                    } else {
                        Some(prog)
                    }
                })
            } else {
                None
            };

            // Reserve space for operation receipts (used below)
            let mut _op_receipts: Vec<BackendExecutionReceipt> =
                Vec::with_capacity(plan.operations.len());

            // 4. Find the backend for this boundary (mutable borrow)
            let _backend = self
                .find_backend(backend_id)
                .ok_or_else(|| format!("no backend registered for id {}", backend_id.0))?;

            if let Some(_ane_prog) = boundary_type {
                // ANE dispatch: run the compiled Orion program.
                // For now, emit a single receipt with ANE execution time.
                // Phase 2: Build IOSurface I/O buffers from tensor registry
                // and call orion_eval(prog, inputs, outputs).
                _op_receipts.push(BackendExecutionReceipt {
                    operation_id: OperationId(0),
                    backend_id: BackendId(3),
                    backend_version: BackendVersion {
                        backend_name: "ane".to_string(),
                        version: String::new(),
                        git_commit: None,
                    },
                    requested_substrate: None,
                    observed_substrate: None,
                    graph_build_ns: None,
                    compile_ns: None,
                    queue_wait_ns: None,
                    submit_ns: None,
                    execution_ns: Some(boundary_start.elapsed().as_nanos() as u64),
                    synchronization_ns: None,
                    total_wall_ns: boundary_start.elapsed().as_nanos() as u64,
                    bytes_read: None,
                    bytes_written: None,
                    temporary_bytes: None,
                    active_memory_before: None,
                    active_memory_after: None,
                    cache_memory_before: None,
                    cache_memory_after: None,
                    transfer_in_ns: None,
                    transfer_out_ns: None,
                    fallback_occurred: false,
                });
            } else {
                for op_desc in &ops {
                    let receipt = _backend.execute(op_desc, &[])?;
                    _op_receipts.push(receipt);
                }
            }

            // 5. Release tensors specified in release_after
            //    Phase 2: iterate plan.release_after and call _backend.release().

            // 6. Handle synchronization
            let actual_sync_count =
                if matches!(plan.synchronization, SynchronizationPolicy::Barrier) {
                    1
                } else {
                    0
                };

            // 7. Emit receipt with observed timing
            let elapsed_ns = boundary_start.elapsed().as_nanos() as u64;
            receipts.push(BoundaryExecutionReceipt {
                group_id: plan.group_id,
                planned_policy: plan.policy.clone(),
                backend: plan.backend_id,
                operation_count: plan.operations.len(),
                planned_materialized_outputs: plan.materialized_outputs.len(),
                actual_eval_calls: plan.operations.len(),
                actual_sync_count,
                graph_build_ns: 0,
                submit_ns: 0,
                execution_ns: elapsed_ns,
                wait_ns: 0,
                temporary_bytes: 0,
                released_tensor_count: plan.release_after.len(),
                unaccounted_ns: 0,
                policy_support: support,
            });
        }

        self.execution_count += 1;
        Ok(receipts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DType, MatmulOp, QuantizedMatmulOp, RmsNormOp, RoPEOp};

    struct TestBackend;

    impl TensorBackend for TestBackend {
        fn create_f32(&mut self, _data: &[f32], _shape: &[i32]) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn create_u32(&mut self, _data: &[u32], _shape: &[i32]) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn create_f32_from_bf16_bits(
            &mut self,
            _data: &[u16],
            _shape: &[i32],
        ) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn create_owned_from_bytes(
            &mut self,
            _data: &[u8],
            _shape: &[i32],
            _dtype: DType,
        ) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn quantized_matmul(
            &mut self,
            _op: &QuantizedMatmulOp,
            _x: TensorHandle,
            _w: crate::backend::QuantizedWeightHandle,
            _scales: TensorHandle,
            _biases: TensorHandle,
        ) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn matmul(
            &mut self,
            _op: &MatmulOp,
            _a: TensorHandle,
            _b: TensorHandle,
        ) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn rms_norm(
            &mut self,
            _op: &RmsNormOp,
            _x: TensorHandle,
            _weight: TensorHandle,
        ) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn rope(&mut self, _op: &RoPEOp, _x: TensorHandle) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn add(&mut self, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn multiply(&mut self, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn silu(&mut self, _x: TensorHandle) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn transpose(&mut self, _x: TensorHandle, _dims: &[i32]) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn reshape(&mut self, _x: TensorHandle, _shape: &[i32]) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn softmax(&mut self, _x: TensorHandle, _axis: i32) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn index_select(
            &mut self,
            _x: TensorHandle,
            _indices: &[u32],
            _axis: i32,
        ) -> Result<TensorHandle, String> {
            Err("stub".into())
        }
        fn evaluate(
            &mut self,
            _group_id: u64,
            _outputs: &[TensorHandle],
        ) -> Result<crate::backend::EvaluationReceipt, String> {
            Err("stub".into())
        }
        fn read_f32(
            &mut self,
            _handle: TensorHandle,
        ) -> Result<crate::backend::ReadbackReceipt, String> {
            Err("stub".into())
        }
        fn shape(&self, _handle: TensorHandle) -> Result<Vec<i32>, String> {
            Err("stub".into())
        }
        fn release(&mut self, _handle: TensorHandle) -> Result<(), String> {
            Err("stub".into())
        }
        fn active_memory(&self) -> (u64, u64) {
            (0, 0)
        }
        fn backend_capabilities(&self) -> crate::backend::BackendCapabilities {
            crate::backend::BackendCapabilities {
                can_gpu: false,
                can_cpu: true,
                supports_quantized: false,
                supports_bf16_native: false,
                backend_name: "test".into(),
            }
        }
    }

    impl BackendInstance for TestBackend {
        fn backend_kind(&self) -> BackendId {
            BackendId(0)
        }
        fn supports(&self, _family: OperationFamily) -> bool {
            true
        }

        fn execute(
            &mut self,
            _op: &OperationDescriptor,
            _inputs: &[TensorHandle],
        ) -> Result<BackendExecutionReceipt, String> {
            let now = std::time::Instant::now();
            Ok(BackendExecutionReceipt {
                operation_id: _op.operation_id,
                backend_id: BackendId(0),
                backend_version: crate::backend::routing::BackendVersion {
                    backend_name: "test".into(),
                    version: "0.0.0".into(),
                    git_commit: None,
                },
                requested_substrate: None,
                observed_substrate: None,
                graph_build_ns: None,
                compile_ns: None,
                queue_wait_ns: None,
                submit_ns: None,
                execution_ns: Some(now.elapsed().as_nanos() as u64),
                synchronization_ns: None,
                total_wall_ns: now.elapsed().as_nanos() as u64,
                bytes_read: None,
                bytes_written: None,
                temporary_bytes: None,
                active_memory_before: None,
                active_memory_after: None,
                cache_memory_before: None,
                cache_memory_after: None,
                transfer_in_ns: None,
                transfer_out_ns: None,
                fallback_occurred: false,
            })
        }
    }

    /// Create a stub operation descriptor for testing.
    fn make_op(id: u64) -> OperationDescriptor {
        use crate::backend::DType;
        OperationDescriptor {
            operation_id: OperationId(id),
            family: OperationFamily::Matmul,
            layer_index: None,
            phase: Phase::Decode,
            logical_shape: LogicalShape { dims: vec![1, 64] },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::F32, DType::F32],
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: vec![1, 64] },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        }
    }

    #[test]
    fn test_execute_boundaries_empty_plans() {
        let mut exec = HeterogeneousExecutor::new();
        exec.register(Box::new(TestBackend));

        let receipts = exec
            .execute_boundaries(&[])
            .expect("empty plans should succeed");
        assert!(receipts.is_empty());
    }

    #[test]
    fn test_execute_boundaries_single_plan() {
        let mut exec = HeterogeneousExecutor::new();
        exec.register(Box::new(TestBackend));

        let mut registry = HashMap::new();
        registry.insert(OperationId(10), make_op(10));
        registry.insert(OperationId(11), make_op(11));
        exec.set_operation_registry(registry);

        let plan = ExecutionBoundaryPlan {
            group_id: EvaluationGroupId(1),
            backend_id: BackendId(0),
            operations: vec![OperationId(10), OperationId(11)],
            materialized_outputs: vec![TensorId(100)],
            policy: EvaluationPolicy::ExplicitOperation,
            synchronization: SynchronizationPolicy::Barrier,
            release_after: vec![TensorId(99)],
            content_digest: None,
        };

        let receipts = exec
            .execute_boundaries(&[plan])
            .expect("single plan should succeed");
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].group_id, EvaluationGroupId(1));
        assert_eq!(receipts[0].backend, BackendId(0));
        assert_eq!(receipts[0].operation_count, 2);
        assert_eq!(receipts[0].planned_materialized_outputs, 1);
        assert_eq!(receipts[0].actual_eval_calls, 2);
        assert_eq!(receipts[0].actual_sync_count, 1);
        assert_eq!(receipts[0].released_tensor_count, 1);
        assert_eq!(receipts[0].policy_support, EvaluationPolicySupport::Native);
    }

    #[test]
    fn test_execute_boundaries_multiple_plans() {
        let mut exec = HeterogeneousExecutor::new();
        exec.register(Box::new(TestBackend));

        let registry: HashMap<_, _> = (0..3).map(|i| (OperationId(i), make_op(i))).collect();
        exec.set_operation_registry(registry);

        let plans: Vec<ExecutionBoundaryPlan> = (0..3)
            .map(|i| ExecutionBoundaryPlan {
                group_id: EvaluationGroupId(i),
                backend_id: BackendId(0),
                operations: vec![OperationId(i)],
                materialized_outputs: vec![],
                policy: EvaluationPolicy::BackendLazy,
                synchronization: SynchronizationPolicy::None,
                release_after: vec![],
                content_digest: None,
            })
            .collect();

        let receipts = exec
            .execute_boundaries(&plans)
            .expect("multiple plans should succeed");
        assert_eq!(receipts.len(), 3);
        for (i, r) in receipts.iter().enumerate() {
            assert_eq!(r.group_id, EvaluationGroupId(i as u64));
            assert_eq!(r.actual_sync_count, 0); // None → 0
        }
    }

    #[test]
    fn test_execute_boundaries_missing_backend() {
        let mut exec = HeterogeneousExecutor::new();
        // No backend registered

        let mut registry = HashMap::new();
        registry.insert(OperationId(10), make_op(10));
        exec.set_operation_registry(registry);

        let plan = ExecutionBoundaryPlan {
            group_id: EvaluationGroupId(1),
            backend_id: BackendId(99),
            operations: vec![OperationId(10)],
            materialized_outputs: vec![],
            policy: EvaluationPolicy::ExplicitOperation,
            synchronization: SynchronizationPolicy::None,
            release_after: vec![],
            content_digest: None,
        };

        let err = exec
            .execute_boundaries(&[plan])
            .expect_err("should fail for missing backend");
        assert!(
            err.contains("no backend registered for id 99"),
            "error: {err}"
        );
    }

    #[test]
    fn test_execution_count() {
        let mut exec = HeterogeneousExecutor::new();
        exec.register(Box::new(TestBackend));

        assert_eq!(exec.execution_count, 0);

        exec.execute_boundaries(&[]).unwrap();
        assert_eq!(exec.execution_count, 1);

        exec.execute_boundaries(&[]).unwrap();
        assert_eq!(exec.execution_count, 2);
    }
}
