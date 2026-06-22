//! Executable admission — typed refusal for runtime compatibility checks.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutableAdmissionError {
    InvalidSeal,
    UnsupportedFormatVersion,
    MissingTargetProfile,
    IncompatibleHardwareProfile,
    IncompatibleRuntimeProfile,
    MissingRequiredFeature(String),
    ArtifactHashMismatch,
    ContentObjectHashMismatch,
    MissingProgramVariant,
    ArenaPlanUnsatisfied,
    ResidencyPlanUnsatisfied,
    KvPlanUnsatisfied,
    CoreMlArtifactUnavailable,
    MetalPipelineUnavailable,
    AccelerateArtifactUnavailable,
    StateDomainUnavailable,
}
