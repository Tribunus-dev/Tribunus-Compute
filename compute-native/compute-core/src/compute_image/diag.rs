//! Compile-time diagnostics, verification, and publishing for ComputeImage.

pub(crate) use super::compile::{
    verify_image_build_profile, image_build_attestation,
    DiagnosticReport, LayerDiagnostic, GlobalDiagnostic, DiagnosticIssue,
    run_diagnostics, publish_image, read, verify,
    build_compile_receipt, compute_manifest_hash,
};
