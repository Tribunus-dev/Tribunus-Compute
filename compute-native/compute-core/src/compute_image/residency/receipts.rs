//! Receipt types for residency admission and execution.
//!
//! [`ResidencyAdmissionReceipt`] captures the outcome of checking a
//! [`CompiledResidencyPlan`](super::plan::CompiledResidencyPlan) against
//! a device memory budget.  [`ResidencyExecutionReceipt`] captures the
//! runtime-observed I/O and memory footprint during execution of the plan.

/// Receipt emitted when a [`CompiledResidencyPlan`](super::plan::CompiledResidencyPlan)
/// is admitted or refused for execution.
///
/// Captures the admission decision together with the peak memory footprint,
/// available budget, and a summary of mandatory object coverage.
#[derive(Debug, Clone)]
pub struct ResidencyAdmissionReceipt {
    /// Identifier of the residency plan that was evaluated.
    pub plan_id: String,
    /// `true` when the plan was admitted; `false` when refused.
    pub admitted: bool,
    /// Peak memory estimate from the plan (total_resident_bytes).
    pub peak_memory_bytes: u64,
    /// Device memory budget supplied to the admission check.
    pub available_memory_bytes: u64,
    /// Human-readable refusal reason when `admitted` is `false`.
    pub refusal_reason: Option<String>,
    /// Number of mandatory weight objects that were present.
    ///
    /// For admission, this is always equal to `mandatory_objects_total`
    /// since the admission check evaluates all mandatory objects by
    /// definition.
    pub mandatory_objects_loaded: u32,
    /// Total number of mandatory weight objects in the plan.
    pub mandatory_objects_total: u32,
}

/// Receipt emitted after executing a residency plan.
///
/// Tracks object-level I/O (loaded, evicted, prefetched), total
/// weight throughput, and observed peak activation / KV-cache usage.
#[derive(Debug, Clone)]
pub struct ResidencyExecutionReceipt {
    /// Identifier of the residency plan that was executed.
    pub plan_id: String,
    /// Number of weight objects loaded during execution.
    pub total_objects_loaded: u32,
    /// Number of weight objects evicted during execution.
    pub total_objects_evicted: u32,
    /// Number of weight objects prefetched during execution.
    pub total_objects_prefetched: u32,
    /// Total bytes of weight data loaded from storage.
    pub total_weight_bytes_loaded: u64,
    /// Highest activation arena bytes observed during execution.
    pub peak_activation_bytes_seen: u64,
    /// Total KV cache bytes allocated during execution.
    pub kv_cache_bytes_allocated: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ResidencyAdmissionReceipt tests ─────────────────────────────────

    #[test]
    fn test_admission_receipt_admitted() {
        let receipt = ResidencyAdmissionReceipt {
            plan_id: "plan_admit_1".into(),
            admitted: true,
            peak_memory_bytes: 2_000_000_000,
            available_memory_bytes: 4_000_000_000,
            refusal_reason: None,
            mandatory_objects_loaded: 42,
            mandatory_objects_total: 42,
        };

        assert_eq!(receipt.plan_id, "plan_admit_1");
        assert!(receipt.admitted);
        assert_eq!(receipt.peak_memory_bytes, 2_000_000_000);
        assert_eq!(receipt.available_memory_bytes, 4_000_000_000);
        assert!(receipt.refusal_reason.is_none());
        assert_eq!(receipt.mandatory_objects_loaded, 42);
        assert_eq!(receipt.mandatory_objects_total, 42);
    }

    #[test]
    fn test_admission_receipt_refused() {
        let receipt = ResidencyAdmissionReceipt {
            plan_id: "plan_refuse_1".into(),
            admitted: false,
            peak_memory_bytes: 3_000_000_000,
            available_memory_bytes: 1_000_000_000,
            refusal_reason: Some("InsufficientMemory: required 3GB, available 1GB".into()),
            mandatory_objects_loaded: 0,
            mandatory_objects_total: 42,
        };

        assert_eq!(receipt.plan_id, "plan_refuse_1");
        assert!(!receipt.admitted);
        assert_eq!(receipt.peak_memory_bytes, 3_000_000_000);
        assert_eq!(receipt.available_memory_bytes, 1_000_000_000);
        assert!(receipt.refusal_reason.is_some());
        assert_eq!(receipt.mandatory_objects_loaded, 0);
        assert_eq!(receipt.mandatory_objects_total, 42);
    }

    #[test]
    fn test_admission_receipt_all_optional() {
        let receipt = ResidencyAdmissionReceipt {
            plan_id: "plan_no_mandatory".into(),
            admitted: true,
            peak_memory_bytes: 500_000,
            available_memory_bytes: 10_000_000,
            refusal_reason: None,
            mandatory_objects_loaded: 0,
            mandatory_objects_total: 0,
        };

        assert_eq!(receipt.plan_id, "plan_no_mandatory");
        assert!(receipt.admitted);
        assert_eq!(receipt.mandatory_objects_loaded, 0);
        assert_eq!(receipt.mandatory_objects_total, 0);
    }

    // ── ResidencyExecutionReceipt tests ────────────────────────────────

    #[test]
    fn test_execution_receipt_defaults() {
        let receipt = ResidencyExecutionReceipt {
            plan_id: "exec_1".into(),
            total_objects_loaded: 100,
            total_objects_evicted: 10,
            total_objects_prefetched: 20,
            total_weight_bytes_loaded: 500_000_000,
            peak_activation_bytes_seen: 50_000_000,
            kv_cache_bytes_allocated: 200_000_000,
        };

        assert_eq!(receipt.plan_id, "exec_1");
        assert_eq!(receipt.total_objects_loaded, 100);
        assert_eq!(receipt.total_objects_evicted, 10);
        assert_eq!(receipt.total_objects_prefetched, 20);
        assert_eq!(receipt.total_weight_bytes_loaded, 500_000_000);
        assert_eq!(receipt.peak_activation_bytes_seen, 50_000_000);
        assert_eq!(receipt.kv_cache_bytes_allocated, 200_000_000);
    }

    #[test]
    fn test_execution_receipt_zero_values() {
        let receipt = ResidencyExecutionReceipt {
            plan_id: "exec_zero".into(),
            total_objects_loaded: 0,
            total_objects_evicted: 0,
            total_objects_prefetched: 0,
            total_weight_bytes_loaded: 0,
            peak_activation_bytes_seen: 0,
            kv_cache_bytes_allocated: 0,
        };

        assert_eq!(receipt.plan_id, "exec_zero");
        assert_eq!(receipt.total_objects_loaded, 0);
        assert_eq!(receipt.total_objects_evicted, 0);
        assert_eq!(receipt.total_objects_prefetched, 0);
        assert_eq!(receipt.total_weight_bytes_loaded, 0);
        assert_eq!(receipt.peak_activation_bytes_seen, 0);
        assert_eq!(receipt.kv_cache_bytes_allocated, 0);
    }

    #[test]
    fn test_execution_receipt_large_values() {
        let receipt = ResidencyExecutionReceipt {
            plan_id: "exec_large".into(),
            total_objects_loaded: u32::MAX,
            total_objects_evicted: 0,
            total_objects_prefetched: 0,
            total_weight_bytes_loaded: u64::MAX,
            peak_activation_bytes_seen: u64::MAX / 2,
            kv_cache_bytes_allocated: u64::MAX / 3,
        };

        assert_eq!(receipt.total_objects_loaded, u32::MAX);
        assert_eq!(receipt.total_weight_bytes_loaded, u64::MAX);
    }
}
