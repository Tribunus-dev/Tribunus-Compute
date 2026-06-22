//! Top-level sealed executable object and receipt bundle types.

use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedComputeImageExecutable {
    pub executable_version: ExecutableFormatVersion,
    pub model_identity: ModelIdentity,
    pub model_graph_hash: ContentHash,
    pub tokenizer_hash: ContentHash,
    pub content_store: crate::compute_image::content_store::ContentAddressedContentStore,
    pub target_profiles: Vec<super::profile::ExecutableTargetProfile>,
    pub executable_seal: super::seal::ExecutableSeal,
    pub compile_time_receipts: CompileTimeReceiptBundle,
    pub compiler_provenance: super::provenance::CompilerProvenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableFormatVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_name: String,
    pub model_family: String,
    pub model_variant: String,
    pub canonical_graph_hash: ContentHash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileTimeReceiptBundle {
    pub numerical_receipts: Vec<NumericalVerificationReceipt>,
    pub resource_fit_receipts: Vec<ResourceFitReceipt>,
    pub phase_graph_receipts: Vec<PhaseGraphVerificationReceipt>,
    pub residency_receipts: Vec<ResidencyVerificationReceipt>,
    pub artifact_selection_receipts: Vec<KernelSelectionReceipt>,
    pub bundle_hash: ContentHash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalVerificationReceipt {
    pub artifact_identity: String,
    pub reference_graph_hash: ContentHash,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub cosine_similarity: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceFitReceipt {
    pub artifact_identity: String,
    pub resource_fit_ok: bool,
    pub peak_memory_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseGraphVerificationReceipt {
    pub artifact_identity: String,
    pub phase_count: u32,
    pub edge_count: u32,
    pub graph_valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyVerificationReceipt {
    pub artifact_identity: String,
    pub residency_ok: bool,
    pub total_weight_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSelectionReceipt {
    pub artifact_identity: String,
    pub selected_kernel_id: String,
    pub candidate_count: u32,
    pub selection_valid: bool,
}
