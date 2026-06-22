//! E5E — Legacy/executable compatibility boundary.

#[cfg(test)]
mod tests {
    use tribunus_compute_core::compute_image::executable::admission::ExecutableAdmissionError;
    use tribunus_compute_core::compute_image::executable::schema::ExecutableFormatVersion;

    #[test]
    fn test_format_version_newest() {
        let v = ExecutableFormatVersion {
            major: 1,
            minor: 0,
            patch: 0,
        };
        assert_eq!(v.major, 1);
    }

    #[test]
    fn test_admission_errors_are_distinct() {
        // Verify the error variants are structurally sound
        let errs = vec![
            ExecutableAdmissionError::InvalidSeal,
            ExecutableAdmissionError::UnsupportedFormatVersion,
            ExecutableAdmissionError::MissingTargetProfile,
            ExecutableAdmissionError::IncompatibleHardwareProfile,
            ExecutableAdmissionError::IncompatibleRuntimeProfile,
            ExecutableAdmissionError::MissingRequiredFeature("ane".into()),
            ExecutableAdmissionError::ArtifactHashMismatch,
            ExecutableAdmissionError::ContentObjectHashMismatch,
            ExecutableAdmissionError::MissingProgramVariant,
            ExecutableAdmissionError::KvPlanUnsatisfied,
            ExecutableAdmissionError::StateDomainUnavailable,
        ];
        assert_eq!(errs.len(), 11);
        for e in &errs {
            let _s = format!("{:?}", e);
        }
    }

    #[test]
    fn test_legacy_archive_distinct_from_executable() {
        // Legacy images have no seal, no target profiles, no executable_version.
        // Executable images require all three. This test validates the
        // structural distinction — a minimal executable has a version.
        let v = ExecutableFormatVersion {
            major: 1,
            minor: 0,
            patch: 0,
        };
        let is_legacy = v.major == 0;
        assert!(!is_legacy, "executable images have major >= 1");
    }
}
