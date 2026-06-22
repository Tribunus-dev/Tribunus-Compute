//! CompiledResidencyPlan — the compiled memory residency schedule for a
//! SealedComputeImageExecutable.
//!
//! The residency plan captures every decision the compiler makes about
//! memory management during execution: which weight objects must be
//! resident at each phase, when to prefetch or evict, how much
//! activation and KV-cache memory is needed, and the peak memory
//! envelope that the runtime must satisfy for admission.
//!
//! This is a **compiled** artifact: every field is determined by the
//! compiler's analysis (memory analysis, weight classification, arena
//! requirements, prefetch scheduling, and residency admission). The
//! runtime does not make residency decisions — it reads this plan and
//! executes it.

use serde::{Deserialize, Serialize};

use crate::compute_image::execution_shape::ExecutionShapeClass;
use crate::integration::ContentHash;

// ── Type aliases ──────────────────────────────────────────────────────────

/// Opaque identifier for a compiled residency plan.
pub type ResidencyPlanId = String;

/// Identifier for a weight object that the runtime must make resident.
pub type RequiredWeightObjectId = String;

// ── Core plan ─────────────────────────────────────────────────────────────

/// Compiled residency plan — the authoritative contract between the
/// compiler and the runtime for memory management during execution of
/// one program variant.
///
/// Every field is computed at compile time and frozen into the plan.
/// The runtime reads the plan to determine:
///   - which weight objects to load before each phase,
///   - when prefetch and eviction occur,
///   - how much activation / KV-cache memory to reserve,
///   - the peak memory envelope and admission gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledResidencyPlan {
    /// Unique identifier for this plan.
    pub plan_id: ResidencyPlanId,

    /// Content hash covering the entire plan — used for integrity
    /// verification and cache invalidation.
    pub plan_hash: ContentHash,

    /// Execution shape class this plan was compiled for (e.g., Decode1,
    /// PrefillBucket { tokens: 4096 }).
    pub shape_class: ExecutionShapeClass,

    /// Weight objects that must be resident (in some form) at some
    /// point during execution.  The runtime uses this list to drive
    /// its prefetch / evict schedule.
    pub required_weight_objects: Vec<RequiredWeightObject>,

    /// Ordered prefetch schedule that tells the runtime in which order
    /// to bring weight objects to device memory before they are needed.
    pub prefetch_schedule: Vec<PrefetchAction>,

    /// Weight objects the runtime may evict after a given phase,
    /// following the declared eviction policy.
    pub evictable_weight_objects: Vec<EvictableWeightObject>,

    /// Compiler-computed activation arena dimensions.
    pub activation_arena_requirements: ActivationArenaRequirements,

    /// Compiler-computed KV-cache reservation.
    pub kv_cache_requirements: KvCacheRequirements,

    /// Aggregate peak memory estimate across all categories.
    pub peak_memory_estimate: PeakMemoryEstimate,

    /// Admission contract — the minimum and recommended memory budgets
    /// the runtime must satisfy before executing this plan.
    pub memory_admission_contract: MemoryAdmissionContract,
}

// ── Peak-memory analyzer ───────────────────────────────────────────────────

// ── Required weight objects ───────────────────────────────────────────────

/// A weight object that must be resident at some point during execution.
///
/// The compiler assigns each required weight a residency class that
/// determines when and how it is loaded, and an estimated byte size
/// that feeds into memory budgeting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredWeightObject {
    /// Stable identifier for this weight object within the compute image.
    pub object_id: RequiredWeightObjectId,

    /// Residency class that governs the load/lifecycle strategy.
    pub residency_class: ResidencyClass,

    /// Compiler-estimated byte size of this weight object.
    pub estimated_bytes: u64,
}

// ── Prefetch schedule ─────────────────────────────────────────────────────

/// A single prefetch action in the compiled schedule.
///
/// The runtime executes prefetch actions in order, respecting their
/// priority.  High-priority actions must complete before the target
/// phase begins; low-priority actions may be deferred if I/O bandwidth
/// is constrained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchAction {
    /// Weight object to prefetch.
    pub object_id: RequiredWeightObjectId,

    /// Identifier of the phase before which the prefetch must be
    /// initiated (or completed, depending on priority).
    pub prefetch_before_phase: String,

    /// Urgency of this prefetch action.
    pub priority: PrefetchPriority,
}

/// Urgency of a prefetch action.
///
/// The runtime scheduler uses this to prioritise I/O bandwidth:
/// High actions block phase start; Low actions fill idle bandwidth.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PrefetchPriority {
    /// Must be resident before the target phase starts.  The runtime
    /// will not begin the phase until the prefetch completes.
    High,

    /// Prefetch opportunistically when I/O bandwidth is available.
    /// The runtime may skip or defer low-priority prefetches under
    /// memory pressure.
    Low,
}

// ── Evictable weight objects ──────────────────────────────────────────────

/// A weight object that the runtime may evict after a given phase.
///
/// The eviction policy tells the runtime what to do with the freed
/// resources: discard only the device buffer, discard everything, or
/// keep a compressed representation in place.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictableWeightObject {
    /// Weight object eligible for eviction.
    pub object_id: RequiredWeightObjectId,

    /// Phase identifier after which the object may be evicted.
    pub evict_after_phase: String,

    /// Policy governing eviction behaviour.
    pub eviction_policy: EvictionPolicy,
}

/// Policy for evicting a weight object from device memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Release the device (GPU/ANE) buffer but keep the mmap'd backing
    /// store alive for fast re-load.  The object may be restored to
    /// device memory without re-reading from disk.
    DiscardView,

    /// Release both the device buffer and the mmap'd backing store.
    /// Re-loading requires a full I/O from disk.
    DiscardAll,

    /// Keep the weight data on device in a compressed form.  The
    /// on-device footprint is smaller than the uncompressed size,
    /// but decompression is required before use.
    CompressInPlace,
}

// ── Activation arena requirements ─────────────────────────────────────────

/// Compiler-computed requirements for the activation memory arena.
///
/// The arena is the scratch space used for intermediate activations
/// during execution.  The compiler computes the total byte requirement
/// and the number of distinct arena regions (one per execution lane
/// or pipeline stage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationArenaRequirements {
    /// Total number of bytes required for activation storage across
    /// all phases and regions.
    pub total_activation_bytes: u64,

    /// Number of distinct activation arena regions.  Each region is
    /// an independently allocated buffer that may be reused across
    /// phases via the arena's sub-allocation scheme.
    pub arena_region_count: u32,
}

// ── KV-cache requirements ─────────────────────────────────────────────────

/// Compiler-computed requirements for the key-value cache.
///
/// The KV-cache is the primary memory consumer during decode.  The
/// compiler computes the budget from the model's maximum context
/// length, the per-token cache byte footprint, and the total.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheRequirements {
    /// Maximum number of context tokens this plan supports.
    pub max_context_tokens: u32,

    /// Number of bytes consumed per token in the KV cache.
    pub cache_bytes_per_token: u64,

    /// Total KV-cache byte requirement (`max_context_tokens *
    /// cache_bytes_per_token`).
    pub total_cache_bytes: u64,

    #[serde(default)]
    pub total_kv_cache_bytes: u64,
    #[serde(default)]
    pub kv_cache_per_layer_bytes: u64,
    #[serde(default)]
    pub n_layers: u32,
    #[serde(default)]
    pub n_kv_heads: u32,
    #[serde(default)]
    pub head_dim: u32,
    #[serde(default)]
    pub max_context: u32,

}

// ── Peak memory estimate ──────────────────────────────────────────

/// Aggregate peak memory estimate across all memory categories.
///
/// The runtime uses this to validate that the device has sufficient
/// capacity before admitting the plan.  The compiler also publishes a
/// separate admission contract (see [`MemoryAdmissionContract`]) that
/// accounts for runtime overheads and optional degradation budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakMemoryEstimate {
    /// Total estimated resident memory at peak, covering activation
    /// arenas, KV cache, resident weights, and runtime overhead.
    pub total_resident_bytes: u64,

    /// Peak activation arena memory (a sub-component of
    /// `total_resident_bytes`).
    pub activation_peak_bytes: u64,

    /// Peak KV-cache memory (a sub-component of `total_resident_bytes`).
    pub kv_cache_bytes: u64,

    /// Sum of bytes for all weight objects expected to be resident at
    /// peak (a sub-component of `total_resident_bytes`).
    pub resident_weight_bytes: u64,

    /// Runtime overhead bytes — internal bookkeeping, I/O buffers,
    /// dispatch metadata, etc.  This is the remainder after summing
    /// the three categories above.
    pub overhead_bytes: u64,
}

// ── Memory admission contract ─────────────────────────────────────────────

/// The compiled memory admission contract that gates execution.
///
/// The runtime MUST compare its available device memory against these
/// thresholds.  If `minimum_required_bytes` cannot be satisfied, the
/// runtime MUST refuse execution.  If `recommended_bytes` can be
/// satisfied, the runtime SHOULD proceed at full quality.  If only
/// the minimum is available, the runtime MAY degrade gracefully if
/// `graceful_degradation` is set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAdmissionContract {
    /// Absolute lower bound: execution MUST NOT start unless the
    /// device has at least this many bytes available.
    pub minimum_required_bytes: u64,

    /// Recommended budget at which the plan operates at full quality
    /// (no degradation, no OOM risk).
    pub recommended_bytes: u64,

    /// If `true`, the runtime MAY degrade behaviour (e.g., reduce KV
    /// cache size, use smaller batch, opportunistically recompute
    /// activations) when memory falls between `minimum_required_bytes`
    /// and `recommended_bytes`.  If `false`, the runtime MUST fail
    /// cleanly rather than degrade.
    pub graceful_degradation: bool,
}

// ── Residency class ───────────────────────────────────────────────────────

/// Classifies the residency strategy for a weight object.
///
/// Each variant tells the runtime when and how the weight object is
/// loaded, pinned, or evicted.  The compiler assigns one residency
/// class per weight object based on its usage pattern across phases.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResidencyClass {
    /// Must be resident from session start until the end of the
    /// session.  Cannot be evicted.  Examples: embedding tables,
    /// lm_head, norm parameters.
    MandatoryAtSessionStart,

    /// Must be resident before a specific phase begins and stays
    /// resident at least until the phase completes.  The compiler
    /// provides the phase boundary in the prefetch schedule.
    MandatoryBeforePhase,

    /// Strong candidate for prefetch — the compiler believes it
    /// should be resident before its first use, but the runtime MAY
    /// defer or skip under memory pressure.  Not required for
    /// correctness, only for performance.
    PrefetchCandidate,

    /// may be evicted between reuse windows.  The runtime SHOULD keep
    /// it resident when possible and repin on next use.
    ReusablePinned,

    /// May be evicted after the declared phase.  The runtime follows
    /// the eviction policy specified in the corresponding
    /// [`EvictableWeightObject`] entry.
    EvictableAfterPhase,

    /// The weight data lives on disk only (or in the content store)
    /// and is never made resident on device.  The runtime accesses it
    /// via mmap or streaming I/O on demand.
    DiskOnly,
}

// ── Peak-memory analyzer ───────────────────────────────────────────────────

/// Static peak-memory analyzer for compiled residency plans.
///
/// Computes worst-case memory pressure from weight objects, activation
/// arenas, and KV cache, then produces an admission contract that gates
/// execution against a runtime budget.
#[derive(Debug, Clone, Default)]
pub struct PeakMemoryAnalyzer;

impl PeakMemoryAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Compute the peak memory estimate from required weight objects and
    /// activation / KV-cache requirements.
    ///
    /// Only weight objects whose residency class is
    /// [`MandatoryAtSessionStart`](ResidencyClass::MandatoryAtSessionStart)
    /// or [`MandatoryBeforePhase`](ResidencyClass::MandatoryBeforePhase) are
    /// included in `resident_weight_bytes`.  Overhead is estimated at 10 %
    /// of the sum of weights + activation + KV cache.
    pub fn estimate_peak(
        &self,
        required_weights: &[RequiredWeightObject],
        activation_reqs: &ActivationArenaRequirements,
        kv_reqs: &KvCacheRequirements,
    ) -> PeakMemoryEstimate {
        let resident_weight_bytes: u64 = required_weights
            .iter()
            .filter(|w| {
                matches!(
                    w.residency_class,
                    ResidencyClass::MandatoryAtSessionStart | ResidencyClass::MandatoryBeforePhase
                )
            })
            .map(|w| w.estimated_bytes)
            .sum();

        let activation_peak_bytes = activation_reqs.total_activation_bytes;
        let kv_cache_bytes = kv_reqs.total_cache_bytes;

        // Overhead: 10 % of the sum of weights, activation, and KV cache.
        let overhead_bytes = (resident_weight_bytes + activation_peak_bytes + kv_cache_bytes) / 10;

        let total_resident_bytes =
            resident_weight_bytes + activation_peak_bytes + kv_cache_bytes + overhead_bytes;

        PeakMemoryEstimate {
            total_resident_bytes,
            activation_peak_bytes,
            kv_cache_bytes,
            resident_weight_bytes,
            overhead_bytes,
        }
    }

    /// Build the admission contract that gates execution within a given
    /// memory budget.
    ///
    /// If the plan already carries a [`PeakMemoryEstimate`] the analyzer
    /// re-uses it; otherwise it computes one from the plan's components.
    /// The returned contract exposes the peak as
    /// [`minimum_required_bytes`](MemoryAdmissionContract::minimum_required_bytes)
    /// (the absolute lower bound for admission) and sets
    /// [`graceful_degradation`](MemoryAdmissionContract::graceful_degradation)
    /// to `true` when the peak is within 10 % of the available budget.
    pub fn check_admission(
        &self,
        plan: &CompiledResidencyPlan,
        available_bytes: u64,
    ) -> MemoryAdmissionContract {
        // Re-use the pre-computed estimate from the plan.
        let peak_total = plan.peak_memory_estimate.total_resident_bytes;
        let minimum_required_bytes = peak_total;
        let recommended_bytes = peak_total;

        // Graceful degradation hint: peak ≥ 90 % of the available budget.
        let graceful_degradation = peak_total as f64 >= (available_bytes as f64) * 0.9;

        MemoryAdmissionContract {
            minimum_required_bytes,
            recommended_bytes,
            graceful_degradation,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────

    fn weight(estimated_bytes: u64, class: ResidencyClass) -> RequiredWeightObject {
        RequiredWeightObject {
            object_id: format!("w_{}", estimated_bytes),
            residency_class: class,
            estimated_bytes,
        }
    }

    fn arena_reqs(total_activation_bytes: u64) -> ActivationArenaRequirements {
        ActivationArenaRequirements {
            total_activation_bytes,
            arena_region_count: 1,
        }
    }

    fn kv_reqs(total_cache_bytes: u64) -> KvCacheRequirements {
        KvCacheRequirements {
            max_context_tokens: 4096,
            cache_bytes_per_token: total_cache_bytes / 4096,
            total_cache_bytes,
            total_kv_cache_bytes: total_cache_bytes,
            kv_cache_per_layer_bytes: total_cache_bytes / 32,
            n_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            max_context: 4096,
        }
    }

    fn make_plan(
        weights: Vec<RequiredWeightObject>,
        act: ActivationArenaRequirements,
        kv: KvCacheRequirements,
    ) -> CompiledResidencyPlan {
        // Compute estimate up front via the analyzer.
        let analyzer = PeakMemoryAnalyzer::new();
        let est = analyzer.estimate_peak(&weights, &act, &kv);
        CompiledResidencyPlan {
            plan_id: "test".into(),
            plan_hash: Default::default(),
            shape_class: ExecutionShapeClass::Decode1,
            required_weight_objects: weights,
            prefetch_schedule: Vec::new(),
            evictable_weight_objects: Vec::new(),
            activation_arena_requirements: act,
            kv_cache_requirements: kv,
            peak_memory_estimate: est,
            memory_admission_contract: MemoryAdmissionContract {
                minimum_required_bytes: 0,
                recommended_bytes: 0,
                graceful_degradation: false,
            },
        }
    }

    // ── Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_weights_estimate() {
        let analyzer = PeakMemoryAnalyzer::new();
        let estimate = analyzer.estimate_peak(&[], &arena_reqs(0), &kv_reqs(0));

        assert_eq!(estimate.resident_weight_bytes, 0);
        assert_eq!(estimate.activation_peak_bytes, 0);
        assert_eq!(estimate.kv_cache_bytes, 0);
        assert_eq!(estimate.overhead_bytes, 0);
        assert_eq!(estimate.total_resident_bytes, 0);
    }

    #[test]
    fn test_single_weight_object() {
        let analyzer = PeakMemoryAnalyzer::new();
        let weights = vec![weight(1024, ResidencyClass::MandatoryAtSessionStart)];

        let estimate = analyzer.estimate_peak(&weights, &arena_reqs(2048), &kv_reqs(4096));

        assert_eq!(estimate.resident_weight_bytes, 1024);
        assert_eq!(estimate.activation_peak_bytes, 2048);
        assert_eq!(estimate.kv_cache_bytes, 4096);
        // overhead = 10 % of (1024 + 2048 + 4096) = 10 % of 7168 = 716
        assert_eq!(estimate.overhead_bytes, 716);
        assert_eq!(estimate.total_resident_bytes, 1024 + 2048 + 4096 + 716);
    }

    #[test]
    fn test_multiple_weight_objects_summed() {
        let analyzer = PeakMemoryAnalyzer::new();
        let weights = vec![
            weight(1000, ResidencyClass::MandatoryAtSessionStart),
            weight(2000, ResidencyClass::MandatoryBeforePhase),
            weight(3000, ResidencyClass::MandatoryAtSessionStart),
        ];

        let estimate = analyzer.estimate_peak(&weights, &arena_reqs(500), &kv_reqs(500));

        assert_eq!(estimate.resident_weight_bytes, 6000);
        assert_eq!(estimate.activation_peak_bytes, 500);
        assert_eq!(estimate.kv_cache_bytes, 500);
        // overhead = 10 % of (6000 + 500 + 500) = 700
        assert_eq!(estimate.overhead_bytes, 700);
        assert_eq!(estimate.total_resident_bytes, 6000 + 500 + 500 + 700);
    }

    #[test]
    fn test_optional_weights_excluded() {
        let analyzer = PeakMemoryAnalyzer::new();
        let weights = vec![
            weight(1000, ResidencyClass::MandatoryAtSessionStart),
            weight(2000, ResidencyClass::PrefetchCandidate), // excluded
            weight(3000, ResidencyClass::MandatoryBeforePhase),
        ];

        let estimate = analyzer.estimate_peak(&weights, &arena_reqs(0), &kv_reqs(0));

        assert_eq!(estimate.resident_weight_bytes, 4000);
    }

    #[test]
    fn test_admission_passes_when_budget_exceeds_peak() {
        let analyzer = PeakMemoryAnalyzer::new();
        let weights = vec![weight(100, ResidencyClass::MandatoryAtSessionStart)];
        let act = arena_reqs(100);
        let kv = kv_reqs(100);
        // total = 100 + 100 + 100 + (300/10=30) = 330
        let plan = make_plan(weights, act, kv);

        let contract = analyzer.check_admission(&plan, 500);

        // peak (330) < 500 → no degradation.  330 < 450 (90% of 500) → false.
        assert_eq!(contract.minimum_required_bytes, 330);
        assert_eq!(contract.recommended_bytes, 330);
        assert!(!contract.graceful_degradation);
    }

    #[test]
    fn test_admission_fails_when_peak_exceeds_budget() {
        let analyzer = PeakMemoryAnalyzer::new();
        let weights = vec![weight(1000, ResidencyClass::MandatoryAtSessionStart)];
        let act = arena_reqs(1000);
        let kv = kv_reqs(1000);
        // total = 1000 + 1000 + 1000 + (3000/10=300) = 3300
        let plan = make_plan(weights, act, kv);

        let contract = analyzer.check_admission(&plan, 2000);

        // peak (3300) > budget (2000) → contract exposes peak as min/recommended.
        assert_eq!(contract.minimum_required_bytes, 3300);
        assert_eq!(contract.recommended_bytes, 3300);
    }

    #[test]
    fn test_graceful_degradation_within_10_percent() {
        let analyzer = PeakMemoryAnalyzer::new();
        let weights = vec![weight(400, ResidencyClass::MandatoryAtSessionStart)];
        let act = arena_reqs(400);
        let kv = kv_reqs(200);
        // total = 400 + 400 + 200 + (1000/10=100) = 1100
        // 1100 ≥ 90 % of 1200 (= 1080) → graceful_degradation = true
        let plan = make_plan(weights, act, kv);

        let contract = analyzer.check_admission(&plan, 1200);

        assert_eq!(contract.minimum_required_bytes, 1100);
        assert_eq!(contract.recommended_bytes, 1100);
        assert!(contract.graceful_degradation);
    }

    #[test]
    fn test_graceful_degradation_not_set_when_well_under_budget() {
        let analyzer = PeakMemoryAnalyzer::new();
        let weights = vec![weight(50, ResidencyClass::MandatoryAtSessionStart)];
        let act = arena_reqs(50);
        let kv = kv_reqs(0);
        // total = 50 + 50 + 0 + (100/10=10) = 110
        let plan = make_plan(weights, act, kv);

        let contract = analyzer.check_admission(&plan, 1000);

        assert_eq!(contract.minimum_required_bytes, 110);
        assert_eq!(contract.recommended_bytes, 110);
        assert!(!contract.graceful_degradation);
    }
}
