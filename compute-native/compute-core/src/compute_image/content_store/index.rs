//! Content-addressed index and object entry types.

use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};

pub type ContentObjectId = String;
pub type SegmentId = String;
pub type TargetLayoutId = String;
pub type TensorShape = Vec<i64>;
pub type TensorStrides = Vec<i64>;
pub type TensorDType = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentAddressedContentStore {
    pub store_version: ContentStoreVersion,
    pub segments: Vec<ImmutableSegment>,
    pub objects: Vec<ContentObjectEntry>,
    pub aliases: Vec<super::aliases::ContentAliasEntry>,
    pub index_hash: ContentHash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentStoreVersion {
    pub major: u32,
    pub minor: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmutableSegment {
    pub segment_id: SegmentId,
    pub payload_offset: u64,
    pub payload_length: u64,
    pub alignment: u64,
    pub checksum: ContentHash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentObjectEntry {
    pub object_id: ContentObjectId,
    pub content_hash: ContentHash,
    pub object_kind: ContentObjectKind,
    pub target_layout_id: TargetLayoutId,
    pub segment_id: SegmentId,
    pub segment_offset: u64,
    pub payload_bytes: u64,
    pub aligned_bytes: u64,
    pub alignment: u64,
    pub logical_shape: TensorShape,
    pub storage_shape: TensorShape,
    pub physical_strides: TensorStrides,
    pub dtype: TensorDType,
    pub quantization: Option<QuantizationDescriptor>,
    pub checksum: ContentHash,
    pub consumers: Vec<ArtifactConsumerRef>,
    pub residency_class: ResidencyClass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentObjectKind {
    CanonicalWeight,
    MetalPackedWeight,
    MetalQuantizationMetadata,
    AcceleratePackedWeight,
    CoreMlPackagePayload,
    TokenizerPayload,
    KernelArtifactPayload,
    PhaseProgramPayload,
    ArenaPlanPayload,
    ResidencyPlanPayload,
    VerificationReceiptPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactConsumerRef {
    pub artifact_id: String,
    pub artifact_kind: String,
    pub consumer_stage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationDescriptor {
    pub mode: String,
    pub bits: u8,
    pub group_size: u32,
    pub codebook: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResidencyClass {
    MandatoryAtSessionStart,
    MandatoryBeforePhase,
    PrefetchCandidate,
    ReusablePinned,
    EvictableAfterPhase,
    DiskOnly,
}
