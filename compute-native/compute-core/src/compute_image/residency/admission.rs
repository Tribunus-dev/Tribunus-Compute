//! Runtime residency admission checker for [`CompiledResidencyPlan`].
//!
//! The admission controller evaluates a compiled residency plan against
//! a device memory budget.  It checks that mandatory weight objects fit,
//! that the activation arena peak is within budget, and that the KV cache
//! reservation does not exceed available memory.

use serde::{Deserialize, Serialize};

use crate::compute_image::residency::plan::{
    CompiledResidencyPlan, ResidencyClass,
};

/// Runtime residency admission controller.
///
/// Checks whether a [`CompiledResidencyPlan`] can be satisfied within
/// a given device memory budget.  The admission logic evaluates:
///
/// 1. **Mandatory weight objects** — the summed bytes of all
///    `MandatoryAtSessionStart` and `MandatoryBeforePhase` weights must
///    fit within `available_bytes`.
/// 2. **Activation arena** — the peak activation footprint must fit
///    within `available_bytes`.
/// 3. **KV cache** — the total KV cache reservation must fit within
///    `available_bytes`.
///
/// If all conditions pass the plan is
/// [`Admitted`](ResidencyAdmissionResult::Admitted); otherwise the
/// first violation is returned as
/// [`Refused`](ResidencyAdmissionResult::Refused) with a structured
/// [`ResidencyRefusalReason`].
#[derive(Debug, Clone)]
pub struct ResidencyAdmission;

impl ResidencyAdmission {
    /// Create a new admission controller.
    pub fn new() -> Self {
        Self
    }
}

impl ResidencyAdmission {
    /// Check if the residency plan can be satisfied with the available memory.
    ///
    /// Returns [`Admitted`](ResidencyAdmissionResult::Admitted) with
    /// the plan id and peak memory estimate when all checks pass, or
    /// [`Refused`](ResidencyAdmissionResult::Refused) with the first
    /// violation found.
    pub fn check_admission(
        &self,
        plan: &CompiledResidencyPlan,
        available_bytes: u64,
    ) -> ResidencyAdmissionResult {
        // ── 1. Sum mandatory weight bytes ───────────────────────────
        let mandatory_bytes: u64 = plan
            .required_weight_objects
            .iter()
            .filter(|w| {
                matches!(
                    w.residency_class,
                    ResidencyClass::MandatoryAtSessionStart
                        | ResidencyClass::MandatoryBeforePhase
                )
            })
            .map(|w| w.estimated_bytes)
            .sum();

        if mandatory_bytes > available_bytes {
            return ResidencyAdmissionResult::Refused(
                ResidencyRefusalReason::InsufficientMemory {
                    required: mandatory_bytes,
                    available: available_bytes,
                },
            );
        }

        // ── 2. Check activation arena requirements fit ──────────────
        let act_required = plan.activation_arena_requirements.total_activation_bytes;
        if act_required > available_bytes {
            return ResidencyAdmissionResult::Refused(
                ResidencyRefusalReason::ActivationArenaTooLarge {
                    required: act_required,
                    available: available_bytes,
                },
            );
        }

        // ── 3. Check KV cache requirements fit ──────────────────────
        let kv_required = plan.kv_cache_requirements.total_cache_bytes;
        if kv_required > available_bytes {
            return ResidencyAdmissionResult::Refused(
                ResidencyRefusalReason::KvCacheTooLarge {
                    required: kv_required,
                    available: available_bytes,
                },
            );
        }

        // ── All checks passed ───────────────────────────────────────
        let plan_id = plan.plan_id.clone();
        let peak_bytes = plan.peak_memory_estimate.total_resident_bytes;

        ResidencyAdmissionResult::Admitted {
            plan_id,
            peak_bytes,
        }
    }
}

/// Outcome of a residency admission check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencyAdmissionResult {
    /// The plan is admitted for execution.
    Admitted {
        /// Identifier of the admitted plan.
        plan_id: String,
        /// Peak memory estimate from the plan (in bytes).
        peak_bytes: u64,
    },
    /// The plan was refused with a structured reason.
    Refused(ResidencyRefusalReason),
}

/// Structured reason why a residency plan was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencyRefusalReason {
    /// The sum of mandatory weight bytes exceeds the available budget.
    InsufficientMemory {
        /// Total bytes required for mandatory weight objects.
        required: u64,
        /// Bytes available in the device memory budget.
        available: u64,
    },
    /// A mandatory weight object is missing from the content store.
    MissingMandatoryObject(String),
    /// The activation arena peak exceeds the available budget.
    ActivationArenaTooLarge {
        /// Peak activation arena bytes required.
        required: u64,
        /// Bytes available in the device memory budget.
        available: u64,
    },
    /// The KV cache reservation exceeds the available budget.
    KvCacheTooLarge {
        /// Total KV cache bytes required.
        required: u64,
        /// Bytes available in the device memory budget.
        available: u64,
    },
}

impl Default for ResidencyAdmission {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::execution_shape::ExecutionShapeClass;
    use crate::compute_image::residency::plan::{
        ActivationArenaRequirements, CompiledResidencyPlan, KvCacheRequirements,
        PeakMemoryEstimate, RequiredWeightObject, ResidencyClass,
    };

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
        peak: u64,
    ) -> CompiledResidencyPlan {
        CompiledResidencyPlan {
            plan_id: "test_plan".into(),
            plan_hash: Default::default(),
            shape_class: ExecutionShapeClass::Decode1,
            required_weight_objects: weights,
            prefetch_schedule: Vec::new(),
            evictable_weight_objects: Vec::new(),
            activation_arena_requirements: act,
            kv_cache_requirements: kv,
            peak_memory_estimate: PeakMemoryEstimate {
                total_resident_bytes: peak,
                activation_peak_bytes: 0,
                kv_cache_bytes: 0,
                resident_weight_bytes: 0,
                overhead_bytes: 0,
            },
            memory_admission_contract: crate::compute_image::residency::plan::MemoryAdmissionContract {
                minimum_required_bytes: 0,
                recommended_bytes: 0,
                graceful_degradation: false,
            },
        }
    }

    // ── Admission tests ────────────────────────────────────────────────

    #[test]
    fn test_admitted_when_enough_memory() {
        let admission = ResidencyAdmission::new();
        let weights = vec![weight(500, ResidencyClass::MandatoryAtSessionStart)];
        let plan = make_plan(weights, arena_reqs(200), kv_reqs(300), 1000);

        let result = admission.check_admission(&plan, 2000);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Admitted {
                plan_id: "test_plan".into(),
                peak_bytes: 1000,
            }
        );
    }

    #[test]
    fn test_admitted_with_zero_weights() {
        let admission = ResidencyAdmission::new();
        let plan = make_plan(vec![], arena_reqs(0), kv_reqs(0), 0);

        let result = admission.check_admission(&plan, 1);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Admitted {
                plan_id: "test_plan".into(),
                peak_bytes: 0,
            }
        );
    }

    #[test]
    fn test_admitted_with_optional_weights_only() {
        let admission = ResidencyAdmission::new();
        // Only optional weights -- nothing mandatory, so admission passes
        // trivially even with tiny budget.
        let weights = vec![
            weight(10_000, ResidencyClass::PrefetchCandidate),
            weight(20_000, ResidencyClass::EvictableAfterPhase),
        ];
        let plan = make_plan(weights, arena_reqs(100), kv_reqs(100), 0);

        let result = admission.check_admission(&plan, 1000);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Admitted {
                plan_id: "test_plan".into(),
                peak_bytes: 0,
            }
        );
    }

    #[test]
    fn test_admitted_exact_boundary() {
        let admission = ResidencyAdmission::new();
        // mandatory = 500, activation = 200, kv = 300
        // all fit within 1000
        let weights = vec![weight(500, ResidencyClass::MandatoryAtSessionStart)];
        let plan = make_plan(weights, arena_reqs(200), kv_reqs(300), 1000);

        let result = admission.check_admission(&plan, 500);

        // mandatory (500) == available (500) → passes
        assert_eq!(
            result,
            ResidencyAdmissionResult::Admitted {
                plan_id: "test_plan".into(),
                peak_bytes: 1000,
            }
        );
    }

    // ── Refusal: InsufficientMemory ─────────────────────────────────────

    #[test]
    fn test_refused_insufficient_memory() {
        let admission = ResidencyAdmission::new();
        // mandatory = 1500, available = 1000
        let weights = vec![weight(1500, ResidencyClass::MandatoryAtSessionStart)];
        let plan = make_plan(weights, arena_reqs(0), kv_reqs(0), 1600);

        let result = admission.check_admission(&plan, 1000);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::InsufficientMemory {
                required: 1500,
                available: 1000,
            })
        );
    }

    #[test]
    fn test_refused_multiple_mandatory_summed() {
        let admission = ResidencyAdmission::new();
        // mandatory = 400 + 700 = 1100, available = 1000
        let weights = vec![
            weight(400, ResidencyClass::MandatoryAtSessionStart),
            weight(700, ResidencyClass::MandatoryBeforePhase),
        ];
        let plan = make_plan(weights, arena_reqs(0), kv_reqs(0), 1200);

        let result = admission.check_admission(&plan, 1000);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::InsufficientMemory {
                required: 1100,
                available: 1000,
            })
        );
    }

    #[test]
    fn test_refused_insufficient_memory_mixed_optional() {
        let admission = ResidencyAdmission::new();
        // mandatory = 300, available = 200, optional don't matter
        let weights = vec![
            weight(300, ResidencyClass::MandatoryAtSessionStart),
            weight(9999, ResidencyClass::PrefetchCandidate),
            weight(9999, ResidencyClass::DiskOnly),
        ];
        let plan = make_plan(weights, arena_reqs(0), kv_reqs(0), 300);

        let result = admission.check_admission(&plan, 200);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::InsufficientMemory {
                required: 300,
                available: 200,
            })
        );
    }

    // ── Refusal: ActivationArenaTooLarge ────────────────────────────────

    #[test]
    fn test_refused_activation_arena_too_large() {
        let admission = ResidencyAdmission::new();
        // mandatory = 100, activation = 500, available = 400
        let weights = vec![weight(100, ResidencyClass::MandatoryAtSessionStart)];
        let plan = make_plan(weights, arena_reqs(500), kv_reqs(0), 600);

        let result = admission.check_admission(&plan, 400);

        // mandatory (100) fits, activation (500) > available (400)
        assert_eq!(
            result,
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::ActivationArenaTooLarge {
                required: 500,
                available: 400,
            })
        );
    }

    #[test]
    fn test_activation_arena_check_first_after_weights() {
        let admission = ResidencyAdmission::new();
        // mandatory fits (200 < 500), activation fails (400 < 600)
        // but kv would also fail (300 < 500) -- activation is caught first
        let weights = vec![weight(200, ResidencyClass::MandatoryBeforePhase)];
        let plan = make_plan(weights, arena_reqs(600), kv_reqs(300), 1100);

        let result = admission.check_admission(&plan, 500);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::ActivationArenaTooLarge {
                required: 600,
                available: 500,
            })
        );
    }

    // ── Refusal: KvCacheTooLarge ────────────────────────────────────────

    #[test]
    fn test_refused_kv_cache_too_large() {
        let admission = ResidencyAdmission::new();
        // mandatory = 100, activation = 100, kv = 500, available = 400
        let weights = vec![weight(100, ResidencyClass::MandatoryAtSessionStart)];
        let plan = make_plan(weights, arena_reqs(100), kv_reqs(500), 700);

        let result = admission.check_admission(&plan, 400);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::KvCacheTooLarge {
                required: 500,
                available: 400,
            })
        );
    }

    #[test]
    fn test_refused_kv_cache_exact_boundary() {
        let admission = ResidencyAdmission::new();
        // mandatory = 100, activation = 100, kv = 301, available = 300
        // kv (301) > 300 → refused
        let weights = vec![weight(100, ResidencyClass::MandatoryAtSessionStart)];
        let plan = make_plan(weights, arena_reqs(100), kv_reqs(301), 501);

        let result = admission.check_admission(&plan, 300);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::KvCacheTooLarge {
                required: 301,
                available: 300,
            })
        );
    }

    // ── Cumulative: multiple violations, first returned ─────────────────

    #[test]
    fn test_first_violation_wins() {
        let admission = ResidencyAdmission::new();
        // mandatory = 300 fits (300 <= 300)
        // activation = 600 > 300 → caught FIRST
        // kv = 500 > 300 → would fail but not reached
        let weights = vec![
            weight(200, ResidencyClass::MandatoryAtSessionStart),
            weight(100, ResidencyClass::MandatoryBeforePhase),
        ];
        let plan = make_plan(weights, arena_reqs(600), kv_reqs(500), 1400);

        let result = admission.check_admission(&plan, 300);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::ActivationArenaTooLarge {
                required: 600,
                available: 300,
            })
        );
    }

    #[test]
    fn test_kv_cache_reached_only_after_weights_and_arena_pass() {
        let admission = ResidencyAdmission::new();
        // mandatory = 100 (fits in 500)
        // activation = 200 (fits in 500)
        // kv = 600 > 500 → refused here
        let weights = vec![weight(100, ResidencyClass::MandatoryAtSessionStart)];
        let plan = make_plan(weights, arena_reqs(200), kv_reqs(600), 900);

        let result = admission.check_admission(&plan, 500);

        assert_eq!(
            result,
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::KvCacheTooLarge {
                required: 600,
                available: 500,
            })
        );
    }

    // ── Default ────────────────────────────────────────────────────────

    #[test]
    fn test_default() {
        let admission = ResidencyAdmission::default();
        let plan = make_plan(vec![], arena_reqs(0), kv_reqs(0), 0);
        let result = admission.check_admission(&plan, 1);
        assert_eq!(
            result,
            ResidencyAdmissionResult::Admitted {
                plan_id: "test_plan".into(),
                peak_bytes: 0,
            }
        );
    }
}
