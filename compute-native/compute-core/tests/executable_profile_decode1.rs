//! E5B — Decode1 executable profile construction and basic validation.

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

    fn make_decode1_profile() -> ExecutableTargetProfile {
        let program = SerializedPhaseProgram {
            program_id: "decode1".into(),
            program_hash: ContentHash(10),
            shape_class: ExecutionShapeClass::Decode1,
            execution_kind:
                tribunus_compute_core::compute_image::program::phase_program::ExecutionKind::Decode,
            phases: vec![],
            edges: vec![],
            arena_plan_id: "arena_decode1".into(),
            residency_plan_id: "res_decode1".into(),
            default_artifact_selection: Default::default(),
            fallback_chains: vec![],
            proof_receipt_ids: vec![],
            program_bytes: vec![],
        };

        ExecutableTargetProfile {
            profile_id: "decode1-apple-m1".into(),
            profile_hash: ContentHash(0xA),
            hardware_contract: HardwareTargetContract {
                hardware_family: "apple-m1".into(),
                gpu_core_count: 8,
                ane_count: 1,
                has_unified_memory: true,
                max_threadgroup_size: 256,
            },
            runtime_contract: RuntimeTargetContract {
                min_os_version: "14.0".into(),
                feature_flags: vec!["ane".into()],
            },
            shape_variants: vec![ShapeSpecializedProgram {
                variant_id: "decode1".into(),
                shape_profile: ShapeProfile {
                    max_batch: 1,
                    max_tokens: 1,
                    label: "decode1".into(),
                },
                phase_program: program,
                program_hash: ContentHash(10),
            }],
            residency_plans: vec![],
            default_variant_selection: DefaultVariantSelection {
                decode_variant_id: "decode1".into(),
                prefill_variant_id: "prefill_small".into(),
            },
        }
    }

    #[test]
    fn test_decode1_profile_constructs() {
        let _p = make_decode1_profile();
    }

    #[test]
    fn test_decode1_profile_has_correct_lane_target() {
        let p = make_decode1_profile();
        assert!(p.hardware_contract.has_unified_memory);
        assert_eq!(p.hardware_contract.ane_count, 1);
    }
}
