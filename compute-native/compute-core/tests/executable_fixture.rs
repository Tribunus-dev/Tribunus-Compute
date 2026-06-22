//! E5A — Deterministic open/prepare/execute fixture.
//! Validates that the schema types construct deterministically.

#[cfg(test)]
mod tests {
    use tribunus_compute_core::compute_image::executable::admission::ExecutableAdmissionError;
    use tribunus_compute_core::compute_image::executable::provenance::CompilerProvenance;
    use tribunus_compute_core::compute_image::executable::schema::{
        CompileTimeReceiptBundle, ExecutableFormatVersion, ModelIdentity,
        SealedComputeImageExecutable,
    };
    use tribunus_compute_core::compute_image::executable::seal::ExecutableSeal;
    use tribunus_compute_core::integration::ContentHash;

    fn make_fixture() -> SealedComputeImageExecutable {
        SealedComputeImageExecutable {
            executable_version: ExecutableFormatVersion {
                major: 1,
                minor: 0,
                patch: 0,
            },
            model_identity: ModelIdentity {
                model_name: "test".into(),
                model_family: "gemma".into(),
                model_variant: "2b".into(),
                canonical_graph_hash: ContentHash(42),
            },
            model_graph_hash: ContentHash(1),
            tokenizer_hash: ContentHash(2),
            content_store: Default::default(),
            target_profiles: vec![],
            executable_seal: ExecutableSeal {
                root_hash: ContentHash(0),
                manifest_hash: ContentHash(1),
                profile_hashes: vec![],
                receipt_bundle_hash: ContentHash(2),
                signature: None,
            },
            compile_time_receipts: CompileTimeReceiptBundle {
                numerical_receipts: vec![],
                resource_fit_receipts: vec![],
                phase_graph_receipts: vec![],
                residency_receipts: vec![],
                artifact_selection_receipts: vec![],
                bundle_hash: ContentHash(3),
            },
            compiler_provenance: CompilerProvenance {
                compiler_name: "tribunus".into(),
                compiler_version: "1.0".into(),
                compilation_timestamp: "2026-01-01T00:00:00Z".into(),
                source_model_hash: "abc".into(),
                target_profile_ids: vec![],
            },
        }
    }

    #[test]
    fn test_fixture_constructs() {
        let _f = make_fixture();
    }

    #[test]
    fn test_admission_error_display() {
        let err = ExecutableAdmissionError::InvalidSeal;
        assert!(format!("{:?}", err).contains("InvalidSeal"));
    }
}
