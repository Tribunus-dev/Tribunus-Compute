use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};

pub type KernelVariantId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreselectedKernelVariant {
    pub operation: String,
    pub shape_class: String,
    pub selected_artifact: String,
    pub selected_configuration: KernelConfiguration,
    pub candidate_evidence: Vec<KernelCandidateEvidence>,
    pub selection_receipt: KernelSelectionReceipt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfiguration {
    pub threadgroup_size: u32,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub pipeline_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelCandidateEvidence {
    pub candidate_id: String,
    pub operation: String,
    pub configuration: KernelConfiguration,
    pub median_latency_ns: u64,
    pub min_latency_ns: u64,
    pub resource_fit: bool,
    pub numerical_pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSelectionReceipt {
    pub target_profile_hash: ContentHash,
    pub candidate_artifacts: Vec<String>,
    pub candidate_count: u32,
    pub resource_fit_outcomes: Vec<String>,
    pub numerical_qualification_results: Vec<String>,
    pub selected_winner: String,
    pub selection_policy_version: String,
    pub benchmark_timestamp: String,
}
