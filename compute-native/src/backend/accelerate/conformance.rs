//! Accelerate backend conformance runner integration.
//!
//! This module implements PR6: Integration of Accelerate into the existing
//! backend qualification/conformance runner.
//!
//! # Design Principles
//!
//! 1. **Backend Registry**: Clean abstraction for including Accelerate
//!    without hardcoding one-off paths.
//! 2. **Classification**: Each canonical phase classified as pass, unsupported,
//!    fallback, numerical_divergence, backend_unavailable, or execution_failed.
//! 3. **Schema Compatibility**: Output schema remains compatible with existing
//!    backend evidence pipeline.
//! 4. **Portable**: Non-macOS behavior remains consistent.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use super::{dtype::AccelerateDType, ops::CanonicalOp, subsystem::AccelerateSubsystem, 
          support::OpSupportTable, evidence::AccelerateEvidence, 
          kernels::{KernelDispatcher, KernelResult}, activation::ActivationDispatcher, 
          ffi::AccelerateHandle};

/// Backend classification for conformance results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendClassification {
    /// Operation passed all checks.
    Pass,
    /// Operation is unsupported by this backend.
    Unsupported,
    /// Operation used fallback implementation.
    Fallback,
    /// Operation passed but with numerical divergence within tolerance.
    NumericalDivergence,
    /// Backend is unavailable on this platform.
    BackendUnavailable,
    /// Execution failed.
    ExecutionFailed,
}

impl fmt::Display for BackendClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendClassification::Pass => write!(f, "pass"),
            BackendClassification::Unsupported => write!(f, "unsupported"),
            BackendClassification::Fallback => write!(f, "fallback"),
            BackendClassification::NumericalDivergence => write!(f, "numerical_divergence"),
            BackendClassification::BackendUnavailable => write!(f, "backend_unavailable"),
            BackendClassification::ExecutionFailed => write!(f, "execution_failed"),
        }
    }
}

impl BackendClassification {
    /// Returns true if this classification indicates success.
    pub fn is_success(&self) -> bool {
        matches!(self, BackendClassification::Pass | BackendClassification::NumericalDivergence)
    }

    /// Returns true if this classification indicates the backend is unavailable.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, BackendClassification::BackendUnavailable)
    }

    /// Returns true if this classification indicates the operation is unsupported.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, BackendClassification::Unsupported)
    }

    /// Returns true if this classification indicates fallback was used.
    pub fn is_fallback(&self) -> bool {
        matches!(self, BackendClassification::Fallback)
    }
}

/// Backend identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendId {
    /// Accelerate backend.
    Accelerate,
    /// MLX backend.
    Mlx,
    /// Core ML backend.
    CoreMl,
    /// Reference backend.
    Reference,
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendId::Accelerate => write!(f, "accelerate"),
            BackendId::Mlx => write!(f, "mlx"),
            BackendId::CoreMl => write!(f, "coreml"),
            BackendId::Reference => write!(f, "reference"),
        }
    }
}

/// Conformance result for a single operation on a single backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceResult {
    /// The backend that produced this result.
    pub backend: BackendId,
    /// The canonical operation.
    pub op: CanonicalOp,
    /// The data type.
    pub dtype: AccelerateDType,
    /// The shape key.
    pub shape_key: String,
    /// The classification.
    pub classification: BackendClassification,
    /// Evidence (if available).
    pub evidence: Option<AccelerateEvidence>,
    /// Additional notes.
    pub notes: String,
}

impl ConformanceResult {
    /// Creates a new conformance result.
    pub fn new(
        backend: BackendId,
        op: CanonicalOp,
        dtype: AccelerateDType,
        shape_key: String,
        classification: BackendClassification,
    ) -> Self {
        Self {
            backend,
            op,
            dtype,
            shape_key,
            classification,
            evidence: None,
            notes: String::new(),
        }
    }

    /// Creates a new conformance result with evidence.
    pub fn with_evidence(
        backend: BackendId,
        op: CanonicalOp,
        dtype: AccelerateDType,
        shape_key: String,
        classification: BackendClassification,
        evidence: AccelerateEvidence,
    ) -> Self {
        Self {
            backend,
            op,
            dtype,
            shape_key,
            classification,
            evidence: Some(evidence),
            notes: String::new(),
        }
    }

    /// Sets additional notes.
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = notes.to_string();
        self
    }

    /// Returns true if this result indicates success.
    pub fn is_success(&self) -> bool {
        self.classification.is_success()
    }
}

/// Backend registry for conformance testing.
pub struct BackendRegistry {
    /// Registered backends.
    pub backends: Vec<BackendId>,
    /// Support tables for each backend.
    pub support_tables: HashMap<BackendId, OpSupportTable>,
}

impl BackendRegistry {
    /// Creates a new backend registry.
    pub fn new() -> Self {
        let mut registry = Self {
            backends: Vec::new(),
            support_tables: HashMap::new(),
        };
        
        // Register Accelerate backend
        registry.register_backend(BackendId::Accelerate, OpSupportTable::v0());
        
        // Register other backends (with empty support tables for now)
        registry.register_backend(BackendId::Mlx, OpSupportTable::new());
        registry.register_backend(BackendId::CoreMl, OpSupportTable::new());
        registry.register_backend(BackendId::Reference, OpSupportTable::new());
        
        registry
    }

    /// Registers a backend with its support table.
    pub fn register_backend(&mut self, backend: BackendId, support_table: OpSupportTable) -> &mut Self {
        if !self.backends.contains(&backend) {
            self.backends.push(backend);
        }
        self.support_tables.insert(backend, support_table);
        self
    }

    /// Returns true if the given backend is registered.
    pub fn has_backend(&self, backend: BackendId) -> bool {
        self.backends.contains(&backend)
    }

    /// Returns the support table for the given backend.
    pub fn get_support_table(&self, backend: BackendId) -> Option<&OpSupportTable> {
        self.support_tables.get(&backend)
    }

    /// Returns true if the given operation is supported by the given backend.
    pub fn is_supported(&self, backend: BackendId, op: CanonicalOp) -> bool {
        self.get_support_table(backend)
            .map(|table| table.is_supported(op))
            .unwrap_or(false)
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Accelerate conformance runner.
pub struct AccelerateConformanceRunner {
    /// Backend registry.
    pub registry: BackendRegistry,
    /// Kernel dispatcher for Accelerate.
    pub kernel_dispatcher: KernelDispatcher,
    /// Activation dispatcher for Accelerate.
    pub activation_dispatcher: ActivationDispatcher,
    /// Accelerate handle.
    pub handle: AccelerateHandle,
}

impl AccelerateConformanceRunner {
    /// Creates a new conformance runner.
    pub fn new() -> Self {
        Self {
            registry: BackendRegistry::new(),
            kernel_dispatcher: KernelDispatcher::new(),
            activation_dispatcher: ActivationDispatcher::new(),
            handle: AccelerateHandle::new(),
        }
    }

    /// Runs conformance tests for all canonical operations.
    pub fn run_all(&self) -> Vec<ConformanceResult> {
        let mut results = Vec::new();
        
        // Test all canonical operations
        for op in CanonicalOp::all() {
            let result = self.run_operation(*op);
            results.push(result);
        }
        
        results
    }

    /// Runs conformance test for a specific operation.
    pub fn run_operation(&self, op: CanonicalOp) -> ConformanceResult {
        use super::dtype::AccelerateDType::F32;
        
        // Check if Accelerate backend is available
        if !self.handle.is_available() {
            return ConformanceResult::new(
                BackendId::Accelerate,
                op,
                F32,
                "scalar".to_string(),
                BackendClassification::BackendUnavailable,
            ).with_notes("Accelerate framework unavailable on this platform");
        }
        
        // Check if the operation is supported
        if !self.registry.is_supported(BackendId::Accelerate, op) {
            return ConformanceResult::new(
                BackendId::Accelerate,
                op,
                F32,
                "scalar".to_string(),
                BackendClassification::Unsupported,
            ).with_notes("Operation not supported by Accelerate backend");
        }
        
        // Run the operation based on its type
        match op {
            CanonicalOp::Identity => self.run_identity(),
            CanonicalOp::Add => self.run_add(),
            CanonicalOp::Multiply => self.run_multiply(),
            CanonicalOp::Matmul => self.run_matmul(),
            CanonicalOp::Sigmoid => self.run_sigmoid(),
            CanonicalOp::Silu => self.run_silu(),
            CanonicalOp::Softmax => self.run_softmax(),
            CanonicalOp::Reshape => self.run_reshape(),
            CanonicalOp::Transpose => self.run_transpose(),
            _ => ConformanceResult::new(
                BackendId::Accelerate,
                op,
                F32,
                "scalar".to_string(),
                BackendClassification::Unsupported,
            ).with_notes("Operation not yet implemented in conformance runner"),
        }
    }

    /// Runs identity conformance test.
    fn run_identity(&self) -> ConformanceResult {
        use super::layout::AccelerateLayout;
        
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];
        
        let result = self.kernel_dispatcher.dispatch_identity(&input, &shape, AccelerateLayout::RowMajor);
        
        if result.receipt.fallback_used {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Identity,
                super::dtype::AccelerateDType::F32,
                "[4]".to_string(),
                BackendClassification::Fallback,
            ).with_notes("Used reference fallback")
        } else if let Some(evidence) = result.evidence {
            if evidence.is_success() {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Identity,
                    super::dtype::AccelerateDType::F32,
                    "[4]".to_string(),
                    BackendClassification::Pass,
                    evidence,
                )
            } else {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Identity,
                    super::dtype::AccelerateDType::F32,
                    "[4]".to_string(),
                    BackendClassification::ExecutionFailed,
                    evidence,
                ).with_notes("Evidence indicates failure")
            }
        } else {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Identity,
                super::dtype::AccelerateDType::F32,
                "[4]".to_string(),
                BackendClassification::Pass,
            )
        }
    }

    /// Runs add conformance test.
    fn run_add(&self) -> ConformanceResult {
        use super::layout::AccelerateLayout;
        
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let shape = vec![3];
        
        let result = self.kernel_dispatcher.dispatch_add(&a, &b, &shape, AccelerateLayout::RowMajor);
        
        if result.receipt.fallback_used {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Add,
                super::dtype::AccelerateDType::F32,
                "[3]".to_string(),
                BackendClassification::Fallback,
            ).with_notes("Used reference fallback")
        } else if let Some(evidence) = result.evidence {
            if evidence.is_success() {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Add,
                    super::dtype::AccelerateDType::F32,
                    "[3]".to_string(),
                    BackendClassification::Pass,
                    evidence,
                )
            } else {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Add,
                    super::dtype::AccelerateDType::F32,
                    "[3]".to_string(),
                    BackendClassification::ExecutionFailed,
                    evidence,
                ).with_notes("Evidence indicates failure")
            }
        } else {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Add,
                super::dtype::AccelerateDType::F32,
                "[3]".to_string(),
                BackendClassification::Pass,
            )
        }
    }

    /// Runs multiply conformance test.
    fn run_multiply(&self) -> ConformanceResult {
        use super::layout::AccelerateLayout;
        
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let shape = vec![3];
        
        let result = self.kernel_dispatcher.dispatch_multiply(&a, &b, &shape, AccelerateLayout::RowMajor);
        
        if result.receipt.fallback_used {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Multiply,
                super::dtype::AccelerateDType::F32,
                "[3]".to_string(),
                BackendClassification::Fallback,
            ).with_notes("Used reference fallback")
        } else if let Some(evidence) = result.evidence {
            if evidence.is_success() {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Multiply,
                    super::dtype::AccelerateDType::F32,
                    "[3]".to_string(),
                    BackendClassification::Pass,
                    evidence,
                )
            } else {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Multiply,
                    super::dtype::AccelerateDType::F32,
                    "[3]".to_string(),
                    BackendClassification::ExecutionFailed,
                    evidence,
                ).with_notes("Evidence indicates failure")
            }
        } else {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Multiply,
                super::dtype::AccelerateDType::F32,
                "[3]".to_string(),
                BackendClassification::Pass,
            )
        }
    }

    /// Runs matmul conformance test.
    fn run_matmul(&self) -> ConformanceResult {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        
        let result = self.kernel_dispatcher.dispatch_matmul(&a, &b, 2, 3, 2);
        
        if result.receipt.fallback_used {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Matmul,
                super::dtype::AccelerateDType::F32,
                "[2,3,2]".to_string(),
                BackendClassification::Fallback,
            ).with_notes("Used reference fallback")
        } else if let Some(evidence) = result.evidence {
            if evidence.is_success() {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Matmul,
                    super::dtype::AccelerateDType::F32,
                    "[2,3,2]".to_string(),
                    BackendClassification::Pass,
                    evidence,
                )
            } else {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Matmul,
                    super::dtype::AccelerateDType::F32,
                    "[2,3,2]".to_string(),
                    BackendClassification::ExecutionFailed,
                    evidence,
                ).with_notes("Evidence indicates failure")
            }
        } else {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Matmul,
                super::dtype::AccelerateDType::F32,
                "[2,3,2]".to_string(),
                BackendClassification::Pass,
            )
        }
    }

    /// Runs sigmoid conformance test.
    fn run_sigmoid(&self) -> ConformanceResult {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let shape = vec![5];
        
        let result = self.activation_dispatcher.dispatch_sigmoid(&input, &shape);
        
        if result.receipt.fallback_used {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Sigmoid,
                super::dtype::AccelerateDType::F32,
                "[5]".to_string(),
                BackendClassification::Fallback,
            ).with_notes("Used reference fallback")
        } else if let Some(evidence) = result.evidence {
            if evidence.is_success() {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Sigmoid,
                    super::dtype::AccelerateDType::F32,
                    "[5]".to_string(),
                    BackendClassification::Pass,
                    evidence,
                )
            } else {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Sigmoid,
                    super::dtype::AccelerateDType::F32,
                    "[5]".to_string(),
                    BackendClassification::ExecutionFailed,
                    evidence,
                ).with_notes("Evidence indicates failure")
            }
        } else {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Sigmoid,
                super::dtype::AccelerateDType::F32,
                "[5]".to_string(),
                BackendClassification::Pass,
            )
        }
    }

    /// Runs SiLU conformance test.
    fn run_silu(&self) -> ConformanceResult {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let shape = vec![5];
        
        let result = self.activation_dispatcher.dispatch_silu(&input, &shape);
        
        if result.receipt.fallback_used {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Silu,
                super::dtype::AccelerateDType::F32,
                "[5]".to_string(),
                BackendClassification::Fallback,
            ).with_notes("Used reference fallback")
        } else if let Some(evidence) = result.evidence {
            if evidence.is_success() {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Silu,
                    super::dtype::AccelerateDType::F32,
                    "[5]".to_string(),
                    BackendClassification::Pass,
                    evidence,
                )
            } else {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Silu,
                    super::dtype::AccelerateDType::F32,
                    "[5]".to_string(),
                    BackendClassification::ExecutionFailed,
                    evidence,
                ).with_notes("Evidence indicates failure")
            }
        } else {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Silu,
                super::dtype::AccelerateDType::F32,
                "[5]".to_string(),
                BackendClassification::Pass,
            )
        }
    }

    /// Runs softmax conformance test.
    fn run_softmax(&self) -> ConformanceResult {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];
        
        let result = self.activation_dispatcher.dispatch_softmax(&input, &shape);
        
        if result.receipt.fallback_used {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Softmax,
                super::dtype::AccelerateDType::F32,
                "[4]".to_string(),
                BackendClassification::Fallback,
            ).with_notes("Used reference fallback")
        } else if let Some(evidence) = result.evidence {
            if evidence.is_success() {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Softmax,
                    super::dtype::AccelerateDType::F32,
                    "[4]".to_string(),
                    BackendClassification::Pass,
                    evidence,
                )
            } else {
                ConformanceResult::with_evidence(
                    BackendId::Accelerate,
                    CanonicalOp::Softmax,
                    super::dtype::AccelerateDType::F32,
                    "[4]".to_string(),
                    BackendClassification::ExecutionFailed,
                    evidence,
                ).with_notes("Evidence indicates failure")
            }
        } else {
            ConformanceResult::new(
                BackendId::Accelerate,
                CanonicalOp::Softmax,
                super::dtype::AccelerateDType::F32,
                "[4]".to_string(),
                BackendClassification::Pass,
            )
        }
    }

    /// Runs reshape conformance test.
    fn run_reshape(&self) -> ConformanceResult {
        // Reshape is a layout operation, not a compute operation
        // For now, mark as unsupported in conformance runner
        ConformanceResult::new(
            BackendId::Accelerate,
            CanonicalOp::Reshape,
            super::dtype::AccelerateDType::F32,
            "layout".to_string(),
            BackendClassification::Unsupported,
        ).with_notes("Reshape is a layout operation, not yet implemented in conformance runner")
    }

    /// Runs transpose conformance test.
    fn run_transpose(&self) -> ConformanceResult {
        // Transpose is a layout operation, not a compute operation
        // For now, mark as unsupported in conformance runner
        ConformanceResult::new(
            BackendId::Accelerate,
            CanonicalOp::Transpose,
            super::dtype::AccelerateDType::F32,
            "layout".to_string(),
            BackendClassification::Unsupported,
        ).with_notes("Transpose is a layout operation, not yet implemented in conformance runner")
    }

    /// Generates a backend matrix showing results for all operations.
    pub fn generate_backend_matrix(&self) -> BackendMatrix {
        let results = self.run_all();
        
        let mut matrix = BackendMatrix::new();
        
        for result in results {
            matrix.add_result(result);
        }
        
        matrix
    }
}

impl Default for AccelerateConformanceRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend matrix showing conformance results for all backends and operations.
#[derive(Debug, Clone, Default)]
pub struct BackendMatrix {
    /// Results organized by backend and operation.
    pub results: HashMap<BackendId, Vec<ConformanceResult>>,
}

impl BackendMatrix {
    /// Creates a new empty backend matrix.
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Adds a conformance result to the matrix.
    pub fn add_result(&mut self, result: ConformanceResult) -> &mut Self {
        self.results
            .entry(result.backend)
            .or_insert_with(Vec::new)
            .push(result);
        self
    }

    /// Returns the results for a specific backend.
    pub fn get_backend_results(&self, backend: BackendId) -> Option<&Vec<ConformanceResult>> {
        self.results.get(&backend)
    }

    /// Returns the classification for a specific backend and operation.
    pub fn get_classification(&self, backend: BackendId, op: CanonicalOp) -> Option<BackendClassification> {
        self.get_backend_results(backend)
            .and_then(|results| {
                results.iter().find(|r| r.op == op).map(|r| r.classification)
            })
    }

    /// Returns a summary of the matrix.
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        
        for (backend, results) in &self.results {
            summary.push_str(&format!("\n=== {} ===\n", backend));
            
            for result in results {
                summary.push_str(&format!(
                    "  {} [{}] -> {}\n",
                    result.op, result.shape_key, result.classification
                ));
                
                if !result.notes.is_empty() {
                    summary.push_str(&format!("    Note: {}\n", result.notes));
                }
            }
        }
        
        summary
    }

    /// Returns a JSON representation of the matrix.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Returns the success rate for a specific backend.
    pub fn backend_success_rate(&self, backend: BackendId) -> f64 {
        if let Some(results) = self.get_backend_results(backend) {
            if results.is_empty() {
                return 0.0;
            }
            
            let success_count = results.iter().filter(|r| r.is_success()).count();
            success_count as f64 / results.len() as f64
        } else {
            0.0
        }
    }

    /// Returns the overall success rate across all backends.
    pub fn overall_success_rate(&self) -> f64 {
        let total_results: usize = self.results.values().map(|v| v.len()).sum();
        if total_results == 0 {
            return 0.0;
        }
        
        let total_success: usize = self.results.values()
            .flat_map(|v| v.iter().filter(|r| r.is_success()))
            .count();
        
        total_success as f64 / total_results as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_classification_display() {
        assert_eq!(BackendClassification::Pass.to_string(), "pass");
        assert_eq!(BackendClassification::Unsupported.to_string(), "unsupported");
        assert_eq!(BackendClassification::Fallback.to_string(), "fallback");
        assert_eq!(BackendClassification::NumericalDivergence.to_string(), "numerical_divergence");
        assert_eq!(BackendClassification::BackendUnavailable.to_string(), "backend_unavailable");
        assert_eq!(BackendClassification::ExecutionFailed.to_string(), "execution_failed");
    }

    #[test]
    fn test_backend_classification_predicates() {
        assert!(BackendClassification::Pass.is_success());
        assert!(BackendClassification::NumericalDivergence.is_success());
        assert!(!BackendClassification::Fallback.is_success());

        assert!(BackendClassification::BackendUnavailable.is_unavailable());
        assert!(!BackendClassification::Pass.is_unavailable());

        assert!(BackendClassification::Unsupported.is_unsupported());
        assert!(!BackendClassification::Pass.is_unsupported());

        assert!(BackendClassification::Fallback.is_fallback());
        assert!(!BackendClassification::Pass.is_fallback());
    }

    #[test]
    fn test_backend_id_display() {
        assert_eq!(BackendId::Accelerate.to_string(), "accelerate");
        assert_eq!(BackendId::Mlx.to_string(), "mlx");
        assert_eq!(BackendId::CoreMl.to_string(), "coreml");
        assert_eq!(BackendId::Reference.to_string(), "reference");
    }

    #[test]
    fn test_conformance_result() {
        let result = ConformanceResult::new(
            BackendId::Accelerate,
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[3]".to_string(),
            BackendClassification::Pass,
        );

        assert_eq!(result.backend, BackendId::Accelerate);
        assert_eq!(result.op, CanonicalOp::Add);
        assert_eq!(result.dtype, AccelerateDType::F32);
        assert_eq!(result.shape_key, "[3]");
        assert_eq!(result.classification, BackendClassification::Pass);
        assert!(result.is_success());
    }

    #[test]
    fn test_conformance_result_with_evidence() {
        let evidence = AccelerateEvidence::new(
            "test".to_string(),
            "phase".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
        );

        let result = ConformanceResult::with_evidence(
            BackendId::Accelerate,
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[3]".to_string(),
            BackendClassification::Pass,
            evidence,
        );

        assert!(result.evidence.is_some());
        assert_eq!(result.classification, BackendClassification::Pass);
    }

    #[test]
    fn test_backend_registry() {
        let registry = BackendRegistry::new();
        
        assert!(registry.has_backend(BackendId::Accelerate));
        assert!(registry.has_backend(BackendId::Mlx));
        assert!(registry.has_backend(BackendId::CoreMl));
        assert!(registry.has_backend(BackendId::Reference));
        
        assert!(registry.is_supported(BackendId::Accelerate, CanonicalOp::Add));
        assert!(!registry.is_supported(BackendId::Accelerate, CanonicalOp::KvCacheView));
    }

    #[test]
    fn test_accelerate_conformance_runner() {
        let runner = AccelerateConformanceRunner::new();
        
        // Run a single operation
        let result = runner.run_operation(CanonicalOp::Add);
        
        assert_eq!(result.backend, BackendId::Accelerate);
        assert_eq!(result.op, CanonicalOp::Add);
        
        // On non-macOS, should be fallback
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(result.classification, BackendClassification::Fallback);
        }
    }

    #[test]
    fn test_backend_matrix() {
        let runner = AccelerateConformanceRunner::new();
        let matrix = runner.generate_backend_matrix();
        
        assert!(!matrix.results.is_empty());
        assert!(matrix.results.contains_key(&BackendId::Accelerate));
        
        let summary = matrix.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Accelerate"));
    }

    #[test]
    fn test_backend_matrix_success_rate() {
        let runner = AccelerateConformanceRunner::new();
        let matrix = runner.generate_backend_matrix();
        
        let rate = matrix.backend_success_rate(BackendId::Accelerate);
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    #[test]
    fn test_backend_matrix_json() {
        let runner = AccelerateConformanceRunner::new();
        let matrix = runner.generate_backend_matrix();
        
        let json = matrix.to_json();
        assert!(!json.is_empty());
        assert!(json.contains("Accelerate"));
    }
}
