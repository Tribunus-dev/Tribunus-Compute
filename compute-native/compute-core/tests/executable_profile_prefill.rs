//! E5C — Prefill executable profile construction.

#[cfg(test)]
mod tests {
    use tribunus_compute_core::compute_image::executable::profile::{
        DefaultVariantSelection, ExecutableTargetProfile, HardwareTargetContract,
        RuntimeTargetContract,
    };
    use tribunus_compute_core::compute_image::executable::variant::{
        ShapeProfile, ShapeSpecializedProgram,
    };
    use tribunus_compute_core::compute_image::execution_shape::ExecutionShapeClass;
    use tribunus_compute_core::compute_image::program::phase_program::SerializedPhaseProgram;
    use tribunus_compute_core::integration::ContentHash;

    fn make_prefill_profile() -> ExecutableTargetProfile {
        let program = SerializedPhaseProgram {
            program_id: "prefill_4096".into(),
            program_hash: ContentHash(20),
            shape_class: ExecutionShapeClass::PrefillBucket { tokens: 4096 },
            execution_kind:
                tribunus_compute_core::compute_image::program::phase_program::ExecutionKind::Prefill,
            phases: vec![],
            edges: vec![],
            arena_plan_id: "arena_prefill_4k".into(),
            residency_plan_id: "res_prefill_4k".into(),
            default_artifact_selection: Default::default(),
            fallback_chains: vec![],
            proof_receipt_ids: vec![],
            program_bytes: vec![],
        };

        ExecutableTargetProfile {
            profile_id: "prefill-4k-apple-m1".into(),
            profile_hash: ContentHash(0xB),
            hardware_contract: HardwareTargetContract {
                hardware_family: "apple-m1".into(),
                gpu_core_count: 8,
                ane_count: 1,
                has_unified_memory: true,
                max_threadgroup_size: 256,
            },
            runtime_contract: RuntimeTargetContract {
                min_os_version: "14.0".into(),
                feature_flags: vec![],
            },
            shape_variants: vec![ShapeSpecializedProgram {
                variant_id: "prefill_4k".into(),
                shape_profile: ShapeProfile {
                    max_batch: 1,
                    max_tokens: 4096,
                    label: "prefill_4k".into(),
                },
                phase_program: program,
                program_hash: ContentHash(20),
            }],
            residency_plans: vec![],
            default_variant_selection: DefaultVariantSelection {
                decode_variant_id: "decode1".into(),
                prefill_variant_id: "prefill_4k".into(),
            },
        }
    }

    #[test]
    fn test_prefill_profile_constructs() {
        let _p = make_prefill_profile();
    }

    #[test]
    fn test_prefill_profile_shape() {
        let p = make_prefill_profile();
        assert_eq!(p.profile_id, "prefill-4k-apple-m1");
        assert_eq!(p.default_variant_selection.prefill_variant_id, "prefill_4k");
    }
}
