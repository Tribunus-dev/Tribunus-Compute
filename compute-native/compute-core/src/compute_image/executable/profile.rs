//! Executable target profile — binds a specific hardware/runtime configuration.

use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};

pub type TargetProfileId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableTargetProfile {
    pub profile_id: TargetProfileId,
    pub profile_hash: ContentHash,
    pub hardware_contract: HardwareTargetContract,
    pub runtime_contract: RuntimeTargetContract,
    pub shape_variants: Vec<super::variant::ShapeSpecializedProgram>,
    pub residency_plans: Vec<crate::compute_image::residency::plan::CompiledResidencyPlan>,
    pub default_variant_selection: DefaultVariantSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareTargetContract {
    pub hardware_family: String,
    pub gpu_core_count: u32,
    pub ane_count: u32,
    pub has_unified_memory: bool,
    pub max_threadgroup_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeTargetContract {
    pub min_os_version: String,
    pub feature_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultVariantSelection {
    pub decode_variant_id: String,
    pub prefill_variant_id: String,
}
