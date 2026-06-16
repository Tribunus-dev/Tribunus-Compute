//! Accelerate evidence definitions.
//!
//! This module defines the evidence system for Accelerate backend,
//! including numerical validation and JSON evidence output.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::{dtype::AccelerateDType, ops::CanonicalOp, subsystem::AccelerateSubsystem};
use super::execution::NumericalStatus;
use super::native::NativeSymbol;

/// Accelerate evidence record.
///
/// This record captures the evidence for a single Accelerate execution,
/// including numerical validation results and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerateEvidence {
    /// Unique identifier for this evidence record.
    pub evidence_id: String,
    /// The backend that produced this evidence.
    pub backend: String,
    /// The subsystem that executed the operation.
    pub subsystem: AccelerateSubsystem,
    /// The canonical operation.
    pub op: CanonicalOp,
    /// The data type used.
    pub dtype: AccelerateDType,
    /// The shape key.
    pub shape_key: String,
    /// The layout key.
    pub layout_key: String,
    /// Graph hash (if applicable).
    pub graph_hash: Option<String>,
    /// Phase identifier.
    pub phase_id: String,
    /// Numerical status.
    pub numerical_status: NumericalStatus,
    /// Maximum absolute error (for elementwise operations).
    pub max_abs_error: Option<f64>,
    /// Maximum relative error (for elementwise operations).
    pub max_rel_error: Option<f64>,
    /// Cosine similarity (for matmul/softmax operations).
    pub cosine_similarity: Option<f64>,
    /// Whether the result passed numerical validation.
    pub passed: bool,
    /// Fallback used flag.
    pub fallback_used: bool,
    /// Fallback reason (if applicable).
    pub fallback_reason: Option<String>,
    /// Wall clock time in nanoseconds.
    pub wall_time_ns: u64,
    /// Allocation count (if measurable).
    pub allocation_count: Option<usize>,
    /// The native symbol that was actually called (if any).
    /// Only populated when the operation executed natively.
    /// Examples: "vDSP_vadd", "vDSP_vmul", "cblas_sgemm"
    pub native_symbol: Option<String>,
    /// Evidence generation timestamp.
    pub timestamp: String,
}

impl AccelerateEvidence {
    /// Creates a new evidence record.
    pub fn new(
        evidence_id: String,
        phase_id: String,
        op: CanonicalOp,
        dtype: AccelerateDType,
        shape_key: String,
        layout_key: String,
        subsystem: AccelerateSubsystem,
    ) -> Self {
        Self {
            evidence_id,
            backend: super::BACKEND_ACCELERATE.to_string(),
            subsystem,
            op,
            dtype,
            shape_key,
            layout_key,
            graph_hash: None,
            phase_id,
            numerical_status: NumericalStatus::NotComputed,
            max_abs_error: None,
            max_rel_error: None,
            cosine_similarity: None,
            passed: false,
            fallback_used: false,
            fallback_reason: None,
            native_symbol: None,
            wall_time_ns: 0,
            allocation_count: None,
            timestamp: Self::current_timestamp(),
        }
    }

    /// Updates the evidence with numerical validation results.
    pub fn with_numerical_results(
        mut self,
        max_abs_error: Option<f64>,
        max_rel_error: Option<f64>,
        cosine_similarity: Option<f64>,
        passed: bool,
    ) -> Self {
        self.max_abs_error = max_abs_error;
        self.max_rel_error = max_rel_error;
        self.cosine_similarity = cosine_similarity;
        self.passed = passed;
        
        if passed {
            self.numerical_status = NumericalStatus::Passed;
        } else {
            self.numerical_status = NumericalStatus::Failed;
        }
        
        self
    }

    /// Sets the native symbol that was called.
    pub fn with_native_symbol(mut self, symbol: NativeSymbol) -> Self {
        self.native_symbol = Some(symbol.to_string());
        self
    }

    /// Marks this evidence as using fallback.
    pub fn with_fallback(mut self, reason: &str) -> Self {
        self.fallback_used = true;
        self.fallback_reason = Some(reason.to_string());
        self.passed = false;
        self.numerical_status = NumericalStatus::Skipped;
        self
    }

    /// Sets the graph hash.
    pub fn with_graph_hash(mut self, hash: String) -> Self {
        self.graph_hash = Some(hash);
        self
    }

    /// Sets timing information.
    pub fn with_timing(mut self, wall_time_ns: u64, allocation_count: Option<usize>) -> Self {
        self.wall_time_ns = wall_time_ns;
        self.allocation_count = allocation_count;
        self
    }

    /// Returns true if the evidence indicates success.
    pub fn is_success(&self) -> bool {
        self.passed && !self.fallback_used
    }

    /// Returns the current timestamp.
    fn current_timestamp() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("{}", secs)
    }

    /// Returns a JSON representation of the evidence.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Returns a summary of the evidence.
    pub fn summary(&self) -> String {
        format!(
            "AccelerateEvidence: op={} dtype={} shape={} subsystem={} passed={} fallback={} time={}ns",
            self.op,
            self.dtype,
            self.shape_key,
            self.subsystem,
            self.passed,
            self.fallback_used,
            self.wall_time_ns
        )
    }
}

impl fmt::Display for AccelerateEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let native_info = if let Some(symbol) = &self.native_symbol {
            format!(" native={}", symbol)
        } else {
            String::new()
        };
        write!(
            f,
            "Evidence[{}] op={} backend={} subsystem={} dtype={} shape={} passed={}{}",
            self.evidence_id,
            self.op,
            self.backend,
            self.subsystem,
            self.dtype,
            self.shape_key,
            self.passed,
            native_info
        )
    }
}

/// Evidence validation thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceThresholds {
    /// Maximum absolute error threshold for elementwise operations.
    pub max_abs_error_threshold: f64,
    /// Maximum relative error threshold for elementwise operations.
    pub max_rel_error_threshold: f64,
    /// Minimum cosine similarity threshold for matmul/softmax operations.
    pub min_cosine_similarity_threshold: f64,
    /// Special tolerance for identity/reshape/transpose operations.
    pub identity_tolerance: f64,
}

impl EvidenceThresholds {
    /// Creates the v0 thresholds.
    pub fn v0() -> Self {
        Self {
            max_abs_error_threshold: super::DEFAULT_F32_ELEMENTWISE_TOLERANCE as f64,
            max_rel_error_threshold: super::DEFAULT_F32_ELEMENTWISE_TOLERANCE as f64,
            min_cosine_similarity_threshold: super::DEFAULT_COSINE_SIMILARITY_THRESHOLD as f64,
            identity_tolerance: super::DEFAULT_F32_IDENTITY_TOLERANCE as f64,
        }
    }

    /// Returns true if the given absolute error is within threshold.
    pub fn is_abs_error_ok(&self, error: f64) -> bool {
        error <= self.max_abs_error_threshold
    }

    /// Returns true if the given relative error is within threshold.
    pub fn is_rel_error_ok(&self, error: f64) -> bool {
        error <= self.max_rel_error_threshold
    }

    /// Returns true if the given cosine similarity is within threshold.
    pub fn is_cosine_similarity_ok(&self, similarity: f64) -> bool {
        similarity >= self.min_cosine_similarity_threshold
    }

    /// Returns true if the given identity error is within threshold.
    pub fn is_identity_error_ok(&self, error: f64) -> bool {
        error <= self.identity_tolerance
    }
}

impl Default for EvidenceThresholds {
    fn default() -> Self {
        Self::v0()
    }
}

/// Evidence validator for checking numerical results.
#[derive(Debug, Clone)]
pub struct EvidenceValidator {
    /// The thresholds to use for validation.
    pub thresholds: EvidenceThresholds,
}

impl EvidenceValidator {
    /// Creates a new validator with v0 thresholds.
    pub fn new() -> Self {
        Self {
            thresholds: EvidenceThresholds::v0(),
        }
    }

    /// Creates a new validator with custom thresholds.
    pub fn with_thresholds(thresholds: EvidenceThresholds) -> Self {
        Self { thresholds }
    }

    /// Validates elementwise operation results.
    pub fn validate_elementwise(
        &self,
        max_abs_error: f64,
        max_rel_error: f64,
        op: CanonicalOp,
    ) -> bool {
        // For identity operations, use tighter tolerance
        if op.is_memory() || op == CanonicalOp::Identity {
            return self.thresholds.is_identity_error_ok(max_abs_error);
        }

        // For other elementwise operations, use standard tolerance
        self.thresholds.is_abs_error_ok(max_abs_error) && 
        self.thresholds.is_rel_error_ok(max_rel_error)
    }

    /// Validates matmul/softmax operation results.
    pub fn validate_matmul_softmax(
        &self,
        cosine_similarity: f64,
    ) -> bool {
        self.thresholds.is_cosine_similarity_ok(cosine_similarity)
    }

    /// Creates an evidence record from validation results.
    pub fn create_evidence(
        &self,
        evidence_id: String,
        phase_id: String,
        op: CanonicalOp,
        dtype: AccelerateDType,
        shape_key: String,
        layout_key: String,
        subsystem: AccelerateSubsystem,
        max_abs_error: Option<f64>,
        max_rel_error: Option<f64>,
        cosine_similarity: Option<f64>,
        wall_time_ns: u64,
        allocation_count: Option<usize>,
    ) -> AccelerateEvidence {
        let passed = if op.is_memory() || op.is_layout() {
            // For memory/layout ops, we expect perfect results
            max_abs_error.map(|e| self.thresholds.is_identity_error_ok(e)).unwrap_or(true)
        } else if op.is_matmul() || op.is_reduction() {
            // For matmul/softmax, use cosine similarity
            cosine_similarity.map(|cs| self.thresholds.is_cosine_similarity_ok(cs)).unwrap_or(false)
        } else {
            // For elementwise/activation ops, use error thresholds
            max_abs_error.map(|e| self.thresholds.is_abs_error_ok(e)).unwrap_or(false) &&
            max_rel_error.map(|e| self.thresholds.is_rel_error_ok(e)).unwrap_or(false)
        };

        AccelerateEvidence::new(
            evidence_id,
            phase_id,
            op,
            dtype,
            shape_key,
            layout_key,
            subsystem,
        )
        .with_numerical_results(
            max_abs_error,
            max_rel_error,
            cosine_similarity,
            passed,
        )
        .with_timing(wall_time_ns, allocation_count)
    }

    /// Creates an evidence record with native symbol from validation results.
    pub fn create_evidence_with_native_symbol(
        &self,
        evidence_id: String,
        phase_id: String,
        op: CanonicalOp,
        dtype: AccelerateDType,
        shape_key: String,
        layout_key: String,
        subsystem: AccelerateSubsystem,
        max_abs_error: Option<f64>,
        max_rel_error: Option<f64>,
        cosine_similarity: Option<f64>,
        wall_time_ns: u64,
        allocation_count: Option<usize>,
        native_symbol: NativeSymbol,
    ) -> AccelerateEvidence {
        let passed = if op.is_memory() || op.is_layout() {
            // For memory/layout ops, we expect perfect results
            max_abs_error.map(|e| self.thresholds.is_identity_error_ok(e)).unwrap_or(true)
        } else if op.is_matmul() || op.is_reduction() {
            // For matmul/softmax, use cosine similarity
            cosine_similarity.map(|cs| self.thresholds.is_cosine_similarity_ok(cs)).unwrap_or(false)
        } else {
            // For elementwise/activation ops, use error thresholds
            max_abs_error.map(|e| self.thresholds.is_abs_error_ok(e)).unwrap_or(false) &&
            max_rel_error.map(|e| self.thresholds.is_rel_error_ok(e)).unwrap_or(false)
        };

        AccelerateEvidence::new(
            evidence_id,
            phase_id,
            op,
            dtype,
            shape_key,
            layout_key,
            subsystem,
        )
        .with_numerical_results(
            max_abs_error,
            max_rel_error,
            cosine_similarity,
            passed,
        )
        .with_timing(wall_time_ns, allocation_count)
        .with_native_symbol(native_symbol)
    }
}

impl Default for EvidenceValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Evidence collector for aggregating multiple evidence records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvidenceCollector {
    /// Collected evidence records.
    pub records: Vec<AccelerateEvidence>,
}

impl EvidenceCollector {
    /// Creates a new empty collector.
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    /// Adds an evidence record.
    pub fn add(&mut self, evidence: AccelerateEvidence) -> &mut Self {
        self.records.push(evidence);
        self
    }

    /// Returns the number of successful evidence records.
    pub fn success_count(&self) -> usize {
        self.records.iter().filter(|e| e.is_success()).count()
    }

    /// Returns the number of failed evidence records.
    pub fn failure_count(&self) -> usize {
        self.records.iter().filter(|e| !e.is_success()).count()
    }

    /// Returns the overall success rate.
    pub fn success_rate(&self) -> f64 {
        if self.records.is_empty() {
            0.0
        } else {
            self.success_count() as f64 / self.records.len() as f64
        }
    }

    /// Returns a summary of all collected evidence.
    pub fn summary(&self) -> String {
        format!(
            "EvidenceCollector: total={}, success={}, failure={}, rate={:.2}%",
            self.records.len(),
            self.success_count(),
            self.failure_count(),
            self.success_rate() * 100.0
        )
    }

    /// Returns a JSON representation of all collected evidence.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_creation() {
        let evidence = AccelerateEvidence::new(
            "evid1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
        );

        assert_eq!(evidence.evidence_id, "evid1");
        assert_eq!(evidence.backend, "accelerate");
        assert_eq!(evidence.subsystem, AccelerateSubsystem::Vdsp);
        assert!(!evidence.is_success()); // Not computed yet
    }

    #[test]
    fn test_evidence_with_results() {
        let evidence = AccelerateEvidence::new(
            "evid1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
        )
        .with_numerical_results(
            Some(1e-5),
            Some(1e-6),
            None,
            true,
        );

        assert!(evidence.is_success());
        assert_eq!(evidence.max_abs_error, Some(1e-5));
        assert_eq!(evidence.numerical_status, NumericalStatus::Passed);
    }

    #[test]
    fn test_evidence_with_fallback() {
        let evidence = AccelerateEvidence::new(
            "evid1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Reference,
        )
        .with_fallback("unsupported");

        assert!(!evidence.is_success());
        assert!(evidence.fallback_used);
        assert_eq!(evidence.fallback_reason, Some("unsupported".to_string()));
    }

    #[test]
    fn test_evidence_display() {
        let evidence = AccelerateEvidence::new(
            "evid1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
        );

        let display = format!("{}", evidence);
        assert!(display.contains("evid1"));
        assert!(display.contains("add"));
        assert!(display.contains("accelerate"));
    }

    #[test]
    fn test_evidence_json() {
        let evidence = AccelerateEvidence::new(
            "evid1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
        );

        let json = evidence.to_json();
        assert!(!json.is_empty());
        assert!(json.contains("accelerate"));
    }

    #[test]
    fn test_thresholds_v0() {
        let thresholds = EvidenceThresholds::v0();
        
        assert_eq!(thresholds.max_abs_error_threshold, 1e-4);
        assert_eq!(thresholds.min_cosine_similarity_threshold, 0.9999);
        assert_eq!(thresholds.identity_tolerance, 1e-6);
    }

    #[test]
    fn test_thresholds_validation() {
        let thresholds = EvidenceThresholds::v0();
        
        assert!(thresholds.is_abs_error_ok(1e-5));
        assert!(!thresholds.is_abs_error_ok(1e-3));
        
        assert!(thresholds.is_cosine_similarity_ok(0.99995));
        assert!(!thresholds.is_cosine_similarity_ok(0.999));
        
        assert!(thresholds.is_identity_error_ok(1e-7));
        assert!(!thresholds.is_identity_error_ok(1e-5));
    }

    #[test]
    fn test_validator_elementwise() {
        let validator = EvidenceValidator::new();
        
        // Should pass for elementwise ops
        assert!(validator.validate_elementwise(1e-5, 1e-6, CanonicalOp::Add));
        assert!(validator.validate_elementwise(1e-7, 1e-8, CanonicalOp::Multiply));
        
        // Should fail for large errors
        assert!(!validator.validate_elementwise(1e-3, 1e-4, CanonicalOp::Add));
    }

    #[test]
    fn test_validator_identity() {
        let validator = EvidenceValidator::new();
        
        // Identity ops should use tighter tolerance
        assert!(validator.validate_elementwise(1e-7, 1e-8, CanonicalOp::Identity));
        assert!(!validator.validate_elementwise(1e-5, 1e-6, CanonicalOp::Identity));
    }

    #[test]
    fn test_validator_matmul() {
        let validator = EvidenceValidator::new();
        
        // Should pass for high cosine similarity
        assert!(validator.validate_matmul_softmax(0.99995));
        assert!(validator.validate_matmul_softmax(0.99999));
        
        // Should fail for low cosine similarity
        assert!(!validator.validate_matmul_softmax(0.999));
    }

    #[test]
    fn test_validator_create_evidence() {
        let validator = EvidenceValidator::new();
        
        let evidence = validator.create_evidence(
            "evid1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
            Some(1e-5),
            Some(1e-6),
            None,
            1000,
            Some(2),
        );

        assert!(evidence.is_success());
        assert_eq!(evidence.wall_time_ns, 1000);
    }

    #[test]
    fn test_collector() {
        let mut collector = EvidenceCollector::new();
        
        collector.add(AccelerateEvidence::new(
            "evid1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
        )
        .with_numerical_results(Some(1e-5), Some(1e-6), None, true));

        collector.add(AccelerateEvidence::new(
            "evid2".to_string(),
            "phase2".to_string(),
            CanonicalOp::Multiply,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
        )
        .with_numerical_results(Some(1e-3), Some(1e-4), None, false));

        assert_eq!(collector.records.len(), 2);
        assert_eq!(collector.success_count(), 1);
        assert_eq!(collector.failure_count(), 1);
        assert_eq!(collector.success_rate(), 0.5);
    }

    #[test]
    fn test_collector_summary() {
        let mut collector = EvidenceCollector::new();
        
        collector.add(AccelerateEvidence::new(
            "evid1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
        )
        .with_numerical_results(Some(1e-5), Some(1e-6), None, true));

        let summary = collector.summary();
        assert!(summary.contains("EvidenceCollector"));
        assert!(summary.contains("success=1"));
    }

    #[test]
    fn test_collector_json() {
        let mut collector = EvidenceCollector::new();
        
        collector.add(AccelerateEvidence::new(
            "evid1".to_string(),
            "phase1".to_string(),
            CanonicalOp::Add,
            AccelerateDType::F32,
            "[2,3]".to_string(),
            "row_major".to_string(),
            AccelerateSubsystem::Vdsp,
        ));

        let json = collector.to_json();
        assert!(!json.is_empty());
        assert!(json.contains("records"));
    }
}
