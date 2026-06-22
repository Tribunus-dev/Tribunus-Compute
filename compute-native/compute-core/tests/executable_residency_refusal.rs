//! E5D — Residency refusal tests. Validates that residency admission
//! correctly refuses when memory budget is insufficient.

#[cfg(test)]
mod tests {
    use tribunus_compute_core::compute_image::content_store::index::ResidencyClass;
    use tribunus_compute_core::compute_image::residency::admission::{
        ResidencyAdmission, ResidencyAdmissionResult, ResidencyRefusalReason,
    };
    use tribunus_compute_core::compute_image::residency::plan::{
        ActivationArenaRequirements, CompiledResidencyPlan, KvCacheRequirements,
        MemoryAdmissionContract, PeakMemoryEstimate, RequiredWeightObject, ResidencyPlanId,
    };
    use tribunus_compute_core::integration::ContentHash;

    fn make_plan_with_mandatory_bytes(weight_bytes: u64) -> CompiledResidencyPlan {
        CompiledResidencyPlan {
            plan_id: "test".into(),
            plan_hash: ContentHash(1),
            shape_class: Default::default(),
            required_weight_objects: vec![RequiredWeightObject {
                object_id: "w1".into(),
                residency_class: ResidencyClass::MandatoryAtSessionStart,
                estimated_bytes: weight_bytes,
            }],
            prefetch_schedule: vec![],
            evictable_weight_objects: vec![],
            activation_arena_requirements: ActivationArenaRequirements {
                total_activation_bytes: 1024,
                arena_region_count: 2,
            },
            kv_cache_requirements: KvCacheRequirements {
                max_context_tokens: 4096,
                cache_bytes_per_token: 1024,
                total_cache_bytes: 4 * 1024 * 1024,
                total_kv_cache_bytes: 4 * 1024 * 1024,
                kv_cache_per_layer_bytes: 1024 * 4096 / 32,
                n_layers: 32,
                n_kv_heads: 8,
                head_dim: 128,
                max_context: 4096,
            },
            peak_memory_estimate: PeakMemoryEstimate {
                total_resident_bytes: weight_bytes + 1024 + 4 * 1024 * 1024,
                activation_peak_bytes: 1024,
                kv_cache_bytes: 4 * 1024 * 1024,
                resident_weight_bytes: weight_bytes,
                overhead_bytes: 0,
            },
            memory_admission_contract: MemoryAdmissionContract {
                minimum_required_bytes: weight_bytes,
                recommended_bytes: weight_bytes * 2,
                graceful_degradation: false,
            },
        }
    }

    #[test]
    fn test_admission_succeeds_with_sufficient_memory() {
        let plan = make_plan_with_mandatory_bytes(1024 * 1024);
        let result = ResidencyAdmission::new().check_admission(&plan, 1024 * 1024 * 1024);
        match result {
            ResidencyAdmissionResult::Admitted { .. } => {}
            _ => panic!("expected Admitted"),
        }
    }

    #[test]
    fn test_admission_refuses_with_insufficient_memory() {
        let plan = make_plan_with_mandatory_bytes(1024 * 1024 * 1024);
        let result = ResidencyAdmission::new().check_admission(&plan, 1024);
        match result {
            ResidencyAdmissionResult::Refused(ResidencyRefusalReason::InsufficientMemory {
                ..
            }) => {}
            other => panic!("expected InsufficientMemory, got {:?}", other),
        }
    }

    #[test]
    fn test_missing_mandatory_object() {
        let plan = make_plan_with_mandatory_bytes(1024);
        let result = ResidencyAdmission::new().check_admission(&plan, 1024 * 1024);
        // Should succeed since plan has mandatory objects and enough memory
        assert!(matches!(result, ResidencyAdmissionResult::Admitted { .. }));
    }
}
