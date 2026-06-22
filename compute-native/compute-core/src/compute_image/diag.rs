//! Compile-time diagnostics, verification, and publishing for ComputeImage.

pub(crate) use super::compile::{
    build_compile_receipt, compute_manifest_hash, image_build_attestation, publish_image, read,
    run_diagnostics, verify, verify_image_build_profile, DiagnosticIssue, DiagnosticReport,
    GlobalDiagnostic, LayerDiagnostic,
};
