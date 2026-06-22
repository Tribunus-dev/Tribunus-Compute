//! TensixComputeImage — compiled program artifact for Tenstorrent Tensix cores.
//! Mirrors the Metal2 Host API ProgramSpec pattern.

/// A compiled Tenstorrent device operation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensixComputeImage {
    /// Unique hash of the compute IR sequence
    pub program_hash: u64,
    /// Identity of this artifact in the cache
    pub artifact_identity: Option<TensixArtifactCacheKey>,
    /// Admission state into the active execution arena
    pub admission_state: Option<TensixAdmissionState>,
    /// Cache key for the image
    pub cache_key: Option<TensixArtifactCacheKey>,
    /// Placement plan for the cores
    pub placement_plan: Option<TensixPlacementPlan>,
    /// Number of Tensix cores this program uses
    pub core_count: u32,
    /// Total DRAM bytes required (weights + activations + CB buffers)
    pub dram_bytes: u64,
    /// SRAM bytes per core (circular buffer allocation)
    pub sram_per_core: u64,
    /// Compile-time kernel configurations
    pub kernel_configs: Vec<KernelConfig>,
    /// Tensor to DRAM buffer slot assignments
    pub tensor_bindings: Vec<TensorBinding>,
    /// Expected latency estimate (cycles, for profiling baseline)
    pub estimated_cycles: u64,
    /// Target device architecture
    pub target_arch: TensixArch,
    /// Serialized Metal2 ProgramSpec JSON (for C++ bridge)
    pub program_spec_json: String,
    /// Number of Tenstorrent cards in the mesh
    pub card_count: u32,
    /// All card coordinates in the interconnect mesh
    pub interconnect_map: Vec<CardCoord>,
    /// Predetermined golden-path dataflow through the card mesh
    pub golden_path: GoldenPath,
}

/// Coordinate of a Tensix core within a multi-card mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CardCoord {
    pub card_id: u32,
    pub noc_x: u32,
    pub noc_y: u32,
}

/// Precompiled kernel configuration.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelConfig {
    pub name: String,
    pub kernel_type: KernelType,
    pub math_fidelity: MathFidelity,
    pub tile_dims: (u32, u32),
    pub data_format: DataFormat,
}

/// Target Tensix architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TensixArch {
    WormholeB0,
    Blackhole,
    Quasar,
}

/// Kernel execution type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KernelType {
    Math,
    Unpack,
    Pack,
    Relu,
}

/// Math fidelity level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MathFidelity {
    LoFi, // fastest, lowest precision
    HiFi2,
    HiFi3,
    HiFi4, // slowest, highest precision
}

/// Data format for Tensix operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DataFormat {
    Float32,
    Float16,
    BFloat16,
    Int8,
    UInt8,
    Int32,
}

/// Predetermined dataflow path through the card mesh.
/// Fixed at compile time — no dynamic load balancing.
/// E.g. LLaMA attention: card0 (QKV) -> card1 (SDPA) -> card2 ...
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GoldenPath {
    /// Ordered card IDs forming the dataflow pipeline
    pub ordered_cards: Vec<u32>,
    /// Interconnect type between consecutive cards in the path
    pub interconnect: InterconnectType,
}

/// Interconnect type for data movement between cards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum InterconnectType {
    /// Intra-card NOC routing (within same card)
    Noc,
    /// High-speed Ethernet link between cards (Wormhole mesh)
    Ethernet,
    /// DRAM-based shared buffer pass-through
    Dram,
}

/// Binding of a tensor to a DRAM buffer slot
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorBinding {
    pub tensor_name: String,
    pub buffer_slot: u32,
    pub byte_offset: u64,
    pub byte_size: u64,
    pub tile_shape: (u32, u32),
}

impl TensixComputeImage {
    pub fn program_hash_short(&self) -> String {
        format!("{:016x}", self.program_hash)
    }
}

/// Unique key for caching compiled Tensix artifacts
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensixArtifactCacheKey {
    pub topology_hash: u64,
    pub arch: TensixArch,
    pub program_hash: u64,
}

/// Admission state for a Tensix compute image
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TensixAdmissionState {
    Pending,
    Admitted,
    Rejected,
    Evicted,
}

/// Policy for placing tensors on Tensix cores
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PlacementPolicy {
    /// Spread tensors across all available cores
    Spread,
    /// Pack tensors onto as few cores as possible
    Pack,
    /// Custom placement per tensor
    Custom,
    /// Distribute across multiple Tensix devices in a mesh
    MultiDeviceMesh,
}

/// Execution plan for a Tensix operation
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensixPlacementPlan {
    pub policy: PlacementPolicy,
    pub core_allocations: Vec<CardCoord>,
}

/// Variant manifest for decode operations
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodeVariantManifest {
    pub batch_size: u32,
    pub seq_len: u32,
    pub kv_cache_slots: u32,
}

/// Failure domain for topology-aware placement
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FailureDomain {
    None,
    Core,
    Card,
    Rack,
}

/// Description of a device mesh topology and placement policy
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyDescription {
    pub placement_policy: PlacementPolicy,
    pub failure_domain: FailureDomain,
    pub topology_hash: u64,
}
