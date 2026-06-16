//! Accelerate execution definitions.
//!
//! This module defines the execution plan and receipt types for Accelerate backend.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{Duration, Instant};

use super::{dtype::AccelerateDType, layout::AccelerateLayout, ops::CanonicalOp, subsystem::AccelerateSubsystem, lowering::AccelerateLoweringKind};

/// Accelerate execution receipt.
///
/// This receipt captures the result of executing a canonical phase through Accelerate,
/// including timing, numerical status, and execution details.
///
/// # Subsystem Distinction
///
/// - `lowering_subsystem`: The subsystem that was *intended* to execute (based on lowering decision)
/// - `executed_subsystem`: The subsystem that *actually* executed
///
/// For example, on Linux: lowering_subsystem=vDSP, executed_subsystem=Reference
/// On macOS with native calls: lowering_subsystem=vDSP, executed_subsystem=vDSP
/// This distinction is critical for truthful evidence reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerateExecutionReceipt {
    /// Unique identifier for this execution.
    pub execution_id: String,
    /// The canonical phase/operation that was executed.
    pub phase_id: String,
    /// The backend that executed this phase.
    pub backend: String,
    /// The subsystem that was *intended* to execute (from lowering decision).
    pub lowering_subsystem: AccelerateSubsystem,
    /// The subsystem that *actually* executed.
    pub executed_subsystem: AccelerateSubsystem,
    /// The canonical operation.
    pub op: CanonicalOp,
    /// The data type used.
    pub dtype: AccelerateDType,
    /// The shape key (string representation of shape).
    pub shape_key: String,
    /// The layout key.
    pub layout_key: String,
    /// Number of allocations performed (if measurable).
    pub allocation_count: Option<usize>,
    /// Wall clock time in nanoseconds.
    pub wall_time_ns: u64,
    /// Numerical status of the execution.
    pub numerical_status: NumericalStatus,
    /// Whether fallback was used.
    pub fallback_used: bool,
    /// Fallback reason. MANDATORY when fallback_used=true.
    pub fallback_reason: Option<String>,
    /// Graph hash (if applicable).
    pub graph_hash: Option<String>,
    /// Timestamp of execution start.
    pub start_timestamp: String,
    /// Timestamp of execution end.
    pub end_timestamp: String,
}

impl AccelerateExecutionReceipt {
    /// Creates a new execution receipt.
    ///
    /// # Arguments
    ///
    /// * `execution_id` - Unique identifier for this execution
    /// * `phase_id` - The canonical phase/operation identifier
    /// * `op` - The canonical operation
    /// * `dtype` - The data type used
    /// * `shape_key` - String representation of shape
    /// * `layout_key` - String representation of layout
    /// * `lowering_subsystem` - The subsystem that was *intended* to execute
    /// * `executed_subsystem` - The subsystem that *actually* executed
    pub fn new(
        execution_id: String,
        phase_id: String,
        op: CanonicalOp,
        dtype: AccelerateDType,
        shape_key: String,
        layout_key: String,
        lowering_subsystem: AccelerateSubsystem,
        executed_subsystem: AccelerateSubsystem,
    ) -> Self {
        let start = Instant::now();
        Self {
            execution_id,
            phase_id,
            backend: super::BACKEND_ACCELERATE.to_string(),
            lowering_subsystem,
            executed_subsystem,
            op,
            dtype,
            shape_key,
            layout_key,
            allocation_count: None,
            wall_time_ns: 0,
            numerical_status: NumericalStatus::NotComputed,
            fallback_used: false,
            fallback_reason: None,
            graph_hash: None,
            start_timestamp: Self::format_timestamp(start),
            end_timestamp: Self::format_timestamp(start), // Will be updated
        }
    }

    /// Updates the receipt with execution results.
    pub fn with_results(
        mut self,
        wall_time: Duration,
        numerical_status: NumericalStatus,
        allocation_count: Option<usize>,
    ) -> Self {
        self.wall_time_ns = wall_time.as_nanos() as u64;
        self.numerical_status = numerical_status;
        self.allocation_count = allocation_count;
        self.end_timestamp = Self::format_timestamp(Instant::now());
        self
    }

    /// Marks this execution as using fallback.
    /// 
    /// # Arguments
    ///
    /// * `reason` - The reason for using fallback. This is MANDATORY and cannot be empty.
    /// 
    /// # Panics
    ///
    /// Panics if reason is empty, as fallback_reason must always be provided when fallback is used.
    pub fn with_fallback(mut self, reason: &str) -> Self {
        assert!(!reason.is_empty(), "fallback_reason cannot be empty when fallback_used=true");
        self.fallback_used = true;
        self.fallback_reason = Some(reason.to_string());
        self
    }

    /// Sets the graph hash.
    pub fn with_graph_hash(mut self, hash: String) -> Self {
        self.graph_hash = Some(hash);
        self
    }

    /// Formats a timestamp as a string.
    fn format_timestamp(instant: Instant) -> String {
        let duration = instant.elapsed();
        format!("{:?}", duration)
    }

    /// Returns true if the execution was successful.
    /// 
    /// Success means: no fallback used AND numerical status is OK.
    /// This ensures that only actual native execution (not reference fallback) counts as success.
    pub fn is_success(&self) -> bool {
        !self.fallback_used && self.numerical_status.is_ok()
    }

    /// Returns true if this execution used native Accelerate (not reference fallback).
    pub fn is_native_execution(&self) -> bool {
        !self.fallback_used && self.executed_subsystem != AccelerateSubsystem::Reference
    }

    /// Returns the execution time in milliseconds.
    pub fn wall_time_ms(&self) -> f64 {
        self.wall_time_ns as f64 / 1_000_000.0
    }

    /// Returns a summary of the receipt.
    pub fn summary(&self) -> String {
        format!(
            "AccelerateExecutionReceipt: op={}, dtype={}, shape={}, lowering={}, executed={}, time={:.3}ms, status={}, fallback={}",
            self.op,
            self.dtype,
            self.shape_key,
            self.lowering_subsystem,
            self.executed_subsystem,
            self.wall_time_ms(),
            self.numerical_status,
            self.fallback_used
        )
    }
}

impl fmt::Display for AccelerateExecutionReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Receipt[{}] op={} backend={} lowering={} executed={} dtype={} shape={} time={}ns status={}",
            self.execution_id,
            self.op,
            self.backend,
            self.lowering_subsystem,
            self.executed_subsystem,
            self.dtype,
            self.shape_key,
            self.wall_time_ns,
            self.numerical_status
        )
    }
}

/// Numerical status of an execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalStatus {
    /// Numerical result has not been computed yet.
    NotComputed,
    /// Execution passed all numerical checks.
    Passed,
    /// Execution failed numerical checks.
    Failed,
    /// Execution passed with warnings (e.g., within tolerance but not perfect).
    PassedWithWarnings,
    /// Numerical check was skipped.
    Skipped,
}

impl fmt::Display for NumericalStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericalStatus::NotComputed => write!(f, "not_computed"),
            NumericalStatus::Passed => write!(f, "passed"),
            NumericalStatus::Failed => write!(f, "failed"),
            NumericalStatus::PassedWithWarnings => write!(f, "passed_with_warnings"),
            NumericalStatus::Skipped => write!(f, "skipped"),
        }
    }
}

impl NumericalStatus {
    /// Returns true if this status indicates success.
    pub fn is_ok(&self) -> bool {
        matches!(self, NumericalStatus::Passed | NumericalStatus::PassedWithWarnings)
    }

    /// Returns true if this status indicates failure.
    pub fn is_err(&self) -> bool {
        matches!(self, NumericalStatus::Failed)
    }

    /// Returns true if computation has been performed.
    pub fn is_computed(&self) -> bool {
        !matches!(self, NumericalStatus::NotComputed | NumericalStatus::Skipped)
    }
}

impl Default for NumericalStatus {
    fn default() -> Self {
        NumericalStatus::NotComputed
    }
}

/// Accelerate execution plan.
///
/// This plan takes canonical phase IR plus input buffers and returns output buffers
/// plus execution receipts. It represents a single execution unit for Accelerate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerateExecutionPlan {
    /// Unique identifier for this plan.
    pub plan_id: String,
    /// The canonical operation to execute.
    pub op: CanonicalOp,
    /// The lowering kind to use.
    pub lowering_kind: AccelerateLoweringKind,
    /// The subsystem that was *intended* to execute (from lowering decision).
    pub lowering_subsystem: AccelerateSubsystem,
    /// Input buffer information.
    pub input_buffers: Vec<BufferInfo>,
    /// Output buffer information.
    pub output_buffers: Vec<BufferInfo>,
    /// Execution receipt (filled after execution).
    pub receipt: Option<AccelerateExecutionReceipt>,
    /// Whether this plan uses fallback.
    pub uses_fallback: bool,
}

impl AccelerateExecutionPlan {
    /// Creates a new execution plan.
    pub fn new(plan_id: String, op: CanonicalOp, lowering_kind: AccelerateLoweringKind) -> Self {
        let lowering_subsystem = lowering_kind.subsystem();
        Self {
            plan_id,
            op,
            lowering_kind,
            lowering_subsystem,
            input_buffers: Vec::new(),
            output_buffers: Vec::new(),
            receipt: None,
            uses_fallback: lowering_kind == AccelerateLoweringKind::Reference,
        }
    }

    /// Adds an input buffer.
    pub fn with_input_buffer(mut self, buffer: BufferInfo) -> Self {
        self.input_buffers.push(buffer);
        self
    }

    /// Adds an output buffer.
    pub fn with_output_buffer(mut self, buffer: BufferInfo) -> Self {
        self.output_buffers.push(buffer);
        self
    }

    /// Sets the execution receipt.
    pub fn with_receipt(mut self, receipt: AccelerateExecutionReceipt) -> Self {
        self.receipt = Some(receipt);
        self
    }

    /// Returns true if this plan has been executed.
    pub fn is_executed(&self) -> bool {
        self.receipt.is_some()
    }

    /// Returns the execution receipt, if available.
    pub fn get_receipt(&self) -> Option<&AccelerateExecutionReceipt> {
        self.receipt.as_ref()
    }

    /// Returns the total input buffer size in bytes.
    pub fn input_size_bytes(&self) -> usize {
        self.input_buffers.iter().map(|b| b.size_bytes()).sum()
    }

    /// Returns the total output buffer size in bytes.
    pub fn output_size_bytes(&self) -> usize {
        self.output_buffers.iter().map(|b| b.size_bytes()).sum()
    }
}

impl Default for AccelerateExecutionPlan {
    fn default() -> Self {
        Self::new("default".to_string(), CanonicalOp::Identity, AccelerateLoweringKind::Reference)
    }
}

/// Buffer information for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferInfo {
    /// Buffer identifier.
    pub buffer_id: String,
    /// Data type.
    pub dtype: AccelerateDType,
    /// Shape of the buffer.
    pub shape: Vec<usize>,
    /// Layout of the buffer.
    pub layout: AccelerateLayout,
    /// Strides (if not contiguous).
    pub strides: Option<Vec<usize>>,
    /// Size in bytes.
    pub size_bytes: usize,
}

impl BufferInfo {
    /// Creates a new buffer info.
    pub fn new(
        buffer_id: String,
        dtype: AccelerateDType,
        shape: Vec<usize>,
        layout: AccelerateLayout,
    ) -> Self {
        let size_bytes = dtype.size_in_bytes() * shape.iter().product::<usize>();
        Self {
            buffer_id,
            dtype,
            shape,
            layout,
            strides: None,
            size_bytes,
        }
    }

    /// Creates a new buffer info with explicit strides.
    pub fn with_strides(
        buffer_id: String,
        dtype: AccelerateDType,
        shape: Vec<usize>,
        layout: AccelerateLayout,
        strides: Vec<usize>,
    ) -> Self {
        let size_bytes = dtype.size_in_bytes() * shape.iter().product::<usize>();
        Self {
            buffer_id,
            dtype,
            shape,
            layout,
            strides: Some(strides),
            size_bytes,
        }
    }

    /// Returns the shape as a string key.
    pub fn shape_key(&self) -> String {
        format!("[{}]", self.shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","))
    }

    /// Returns the layout as a string key.
    pub fn layout_key(&self) -> String {
        self.layout.to_string()
    }

    /// Returns the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Returns the number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Execution context for Accelerate operations.
#[derive(Debug, Clone)]
pub struct AccelerateExecutionContext {
    /// The backend capabilities.
    pub capabilities: super::capabilities::AccelerateBackendCapabilities,
    /// The support table.
    pub support_table: super::support::OpSupportTable,
    /// Whether to enable evidence collection.
    pub collect_evidence: bool,
    /// Whether to enable timing.
    pub enable_timing: bool,
}

impl AccelerateExecutionContext {
    /// Creates a new execution context.
    pub fn new() -> Self {
        Self {
            capabilities: super::capabilities::AccelerateBackendCapabilities::detect(),
            support_table: super::support::OpSupportTable::v0(),
            collect_evidence: true,
            enable_timing: true,
        }
    }

    /// Creates a new execution context with the given capabilities.
    pub fn with_capabilities(capabilities: super::capabilities::AccelerateBackendCapabilities) -> Self {
        Self {
            capabilities,
            support_table: super::support::OpSupportTable::v0(),
            collect_evidence: true,
            enable_timing: true,
        }
    }

    /// Returns true if the given operation is supported.
    pub fn is_supported(&self, op: CanonicalOp) -> bool {
        self.support_table.is_supported(op)
    }

    /// Gets the support information for the given operation.
    pub fn get_support(&self, op: CanonicalOp) -> Option<&super::support::AccelerateSupport> {
        self.support_table.get(op)
    }

    /// Creates an execution plan for the given operation.
    pub fn create_plan(&self, plan_id: String, op: CanonicalOp) -> Option<AccelerateExecutionPlan> {
        let support = self.get_support(op)?;
        if !support.supported {
            return None;
        }

        Some(AccelerateExecutionPlan::new(
            plan_id,
            op,
            support.lowering_kind,
        ))
    }
}

impl Default for AccelerateExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_status_display() {
        assert_eq!(NumericalStatus::NotComputed.to_string(), "not_computed");
        assert_eq!(NumericalStatus::Passed.to_string(), "passed");
        assert_eq!(NumericalStatus::Failed.to_string(), "failed");
        assert_eq!(NumericalStatus::PassedWithWarnings.to_string(), "passed_with_warnings");
        assert_eq!(NumericalStatus::Skipped.to_string(), "skipped");
    }

    #[test]
    fn test_numerical_status_predicates() {
        assert!(NumericalStatus::Passed.is_ok());
        assert!(NumericalStatus::PassedWithWarnings.is_ok());
        assert!(!NumericalStatus::Failed.is_ok());

        assert!(NumericalStatus::Failed.is_err());
        assert!(!NumericalStatus::Passed.is_err());

        assert!(NumericalStatus::Passed.is_computed());
        assert!(!NumericalStatus::NotComputed.is_computed());
        assert!(!NumericalStatus::Skipped.is_computed());
    }

    #[test]
    fn test_numerical_status_default() {
        assert_eq!(NumericalStatus::default(), NumericalStatus::NotComputed);
    }

    #[test]
    fn test_buffer_info() {
        let buffer = BufferInfo::new(
            "test".to_string(),
            AccelerateDType::F32,
            vec![2, 3],
            AccelerateLayout::RowMajor,
        );

        assert_eq!(buffer.buffer_id, "test");
        assert_eq!(buffer.dtype, AccelerateDType::F32);
        assert_eq!(buffer.shape, vec![2, 3]);
        assert_eq!(buffer.layout, AccelerateLayout::RowMajor);
        assert_eq!(buffer.size_bytes(), 2 * 3 * 4); // 2x3 f32 = 24 bytes
        assert_eq!(buffer.num_elements(), 6);
    }

    #[test]
    fn test_buffer_info_shape_key() {
        let buffer = BufferInfo::new(
            "test".to_string(),
            AccelerateDType::F32,
            vec![2, 3],
            AccelerateLayout::RowMajor,
        );

        assert_eq!(buffer.shape_key(), "[2,3]");
        assert_eq!(buffer.layout_key(), "row_major");
    }

    #[test]
    fn test_execution_receipt() {
        let receipt = AccelerateExecutionReceipt::new(
            "exec1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
            AccelerateSubsystem::Vdsp,
        );

        assert_eq!(receipt.execution_id, "exec1");
        assert_eq!(receipt.phase_id, "phase1");
        assert_eq!(receipt.backend, "accelerate");
        assert_eq!(receipt.lowering_subsystem, AccelerateSubsystem::Vdsp);
        assert_eq!(receipt.executed_subsystem, AccelerateSubsystem::Vdsp);
        assert_eq!(receipt.op, CanonicalOp::Add);
        assert!(!receipt.is_success()); // Not computed yet
    }

    #[test]
    fn test_execution_receipt_with_results() {
        let receipt = AccelerateExecutionReceipt::new(
            "exec1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
            AccelerateSubsystem::Vdsp,
        )
        .with_results(
            Duration::from_millis(10),
            NumericalStatus::Passed,
            Some(2),
        );

        assert!(receipt.is_success());
        assert!(receipt.is_native_execution());
        assert_eq!(receipt.wall_time_ns, 10_000_000); // 10ms in ns
        assert_eq!(receipt.allocation_count, Some(2));
    }

    #[test]
    fn test_execution_receipt_with_fallback() {
        let receipt = AccelerateExecutionReceipt::new(
            "exec1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,  // lowering intended vDSP
            AccelerateSubsystem::Reference,  // but executed with Reference
        )
        .with_fallback("Accelerate unavailable");

        assert!(receipt.fallback_used);
        assert_eq!(receipt.fallback_reason, Some("Accelerate unavailable".to_string()));
        assert!(!receipt.is_success());
        assert!(!receipt.is_native_execution());
        assert_eq!(receipt.lowering_subsystem, AccelerateSubsystem::Vdsp);
        assert_eq!(receipt.executed_subsystem, AccelerateSubsystem::Reference);
    }

    #[test]
    #[should_panic(expected = "fallback_reason cannot be empty")]
    fn test_execution_receipt_empty_fallback_reason_panics() {
        let receipt = AccelerateExecutionReceipt::new(
            "exec1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
            AccelerateSubsystem::Reference,
        )
        .with_fallback(""); // Empty reason should panic
    }

    #[test]
    fn test_execution_plan() {
        let plan = AccelerateExecutionPlan::new(
            "plan1".to_string(),
            CanonicalOp::Add,
            AccelerateLoweringKind::VdspVector,
        );

        assert_eq!(plan.plan_id, "plan1");
        assert_eq!(plan.op, CanonicalOp::Add);
        assert_eq!(plan.lowering_kind, AccelerateLoweringKind::VdspVector);
        assert_eq!(plan.lowering_subsystem, AccelerateSubsystem::Vdsp);
        assert!(!plan.is_executed());
    }

    #[test]
    fn test_execution_plan_with_buffers() {
        let plan = AccelerateExecutionPlan::new(
            "plan1".to_string(),
            CanonicalOp::Add,
            AccelerateLoweringKind::VdspVector,
        )
        .with_input_buffer(BufferInfo::new(
            "input1".to_string(),
            AccelerateDType::F32,
            vec![2, 3],
            AccelerateLayout::RowMajor,
        ))
        .with_output_buffer(BufferInfo::new(
            "output1".to_string(),
            AccelerateDType::F32,
            vec![2, 3],
            AccelerateLayout::RowMajor,
        ));

        assert_eq!(plan.input_buffers.len(), 1);
        assert_eq!(plan.output_buffers.len(), 1);
        assert_eq!(plan.input_size_bytes(), 24); // 2x3 f32 = 24 bytes
        assert_eq!(plan.output_size_bytes(), 24);
    }

    #[test]
    fn test_execution_context() {
        let ctx = AccelerateExecutionContext::new();
        
        assert!(ctx.collect_evidence);
        assert!(ctx.enable_timing);
        assert!(ctx.is_supported(CanonicalOp::Add));
        assert!(!ctx.is_supported(CanonicalOp::KvCacheView));
    }

    #[test]
    fn test_execution_context_create_plan() {
        let ctx = AccelerateExecutionContext::new();
        
        let plan = ctx.create_plan("plan1".to_string(), CanonicalOp::Add);
        assert!(plan.is_some());
        
        let plan = plan.unwrap();
        assert_eq!(plan.op, CanonicalOp::Add);
        assert_eq!(plan.subsystem, AccelerateSubsystem::Vdsp);
    }

    #[test]
    fn test_execution_context_unsupported_plan() {
        let ctx = AccelerateExecutionContext::new();
        
        let plan = ctx.create_plan("plan1".to_string(), CanonicalOp::KvCacheView);
        assert!(plan.is_none()); // KV-cache not supported in v0
    }

    #[test]
    fn test_execution_plan_default() {
        let plan = AccelerateExecutionPlan::default();
        assert_eq!(plan.op, CanonicalOp::Identity);
        assert_eq!(plan.lowering_kind, AccelerateLoweringKind::Reference);
    }
}
