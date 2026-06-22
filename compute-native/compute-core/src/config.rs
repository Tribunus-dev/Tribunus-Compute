//! Config-driven architecture for Tribunus Compute Kernel.
//!
//! Layer 1: Raw model manifest — captures config.json hash and structure.
//! Layer 2: Normalized architecture — strict Rust types from JSON.
//! Layer 3: Compiled execution specification — per-layer dimensions, policies, tensor shapes.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
pub mod operation_route;

// ── Layer 1: Raw Manifest ──────────────────────────────────────────────────

/// Raw model manifest read from config.json.
#[derive(Serialize)]
pub struct ModelManifest {
    pub config_path: String,
    pub config_hash: String,
    pub model_type: String,
    pub has_text_config: bool,
    pub has_vision_config: bool,
    pub has_audio_config: bool,
    pub has_quantization_metadata: bool,
    pub quantization_bits: Option<u32>,
    pub quantization_group_size: Option<u32>,
    pub quantization_mode: Option<String>,
    pub vision_config: Option<VisionArchitecture>,
    pub audio_config: Option<AudioArchitecture>,
    pub safetensors_shards: Vec<ShardManifest>,
}

#[derive(Serialize)]
pub struct ShardManifest {
    pub path: String,
    pub sha256: String,
    pub tensor_count: usize,
}

// ── Layer 2: Normalized Architecture ───────────────────────────────────────

/// Fully resolved text model architecture from config.json.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextArchitecture {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub global_head_dim: Option<u32>,
    pub num_global_key_value_heads: Option<u32>,
    pub num_hidden_layers: u32,
    pub vocab_size: u32,
    pub sliding_window: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub attention_k_eq_v: bool,
    pub final_logit_softcapping: Option<f64>,
    pub hidden_size_per_layer_input: u32,
    pub layer_types: Vec<AttentionKind>,
    pub rope_local: RopeSpec,
    pub rope_global: Option<RopeSpec>,
    pub model_type: String,

    /// Mixture-of-Experts configuration, if applicable.
    #[serde(default)]
    pub moe_config: Option<MoEConfig>,

    /// Diffusion model configuration, if applicable.
    #[serde(default)]
    pub diffusion_config: Option<DiffusionConfig>,
}

/// Vision encoder configuration from a model's vision_config.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisionArchitecture {
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub intermediate_size: u32,
    pub image_size: u32,
    pub patch_size: u32,
    pub num_channels: u32,
    pub projection_dim: u32,
}

/// Audio encoder configuration from a model's audio_config.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioArchitecture {
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub intermediate_size: u32,
    pub sample_rate: u32,        // e.g. 16000
    pub num_mel_bins: u32,       // e.g. 80
    pub hop_length: u32,         // e.g. 160
    pub max_audio_length_s: u32, // e.g. 30 (seconds)
    pub projection_dim: u32,     // audio_features -> text hidden dim
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RopeSpec {
    pub theta: f64,
    pub rope_type: String,
    pub partial_rotary_factor: Option<f64>,
}

/// Quantization metadata from the converted model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizationMeta {
    pub bits: u32,
    pub group_size: u32,
    pub mode: QuantizationMode,
    /// Per-layer overrides (if any layer has non-default group size or bits).
    pub overrides: HashMap<String, QuantizationMeta>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    None,
    Affine,
    Symmetric,
}

/// Mixture-of-Experts (MoE) routing configuration.
///
/// Describes how many experts exist, how many are active per token,
/// the intermediate (FFN) size within each expert, and whether shared
/// experts (present in every layer) are enabled.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoEConfig {
    /// Total number of experts in the MoE layer.
    pub num_experts: u32,
    /// Number of experts activated per token (top-K routing).
    pub top_k_experts: u32,
    /// FFN intermediate size within each expert.
    pub intermediate_size: u32,
    /// Whether shared (always-active) experts are used alongside routed experts.
    pub shared_experts: bool,
}

/// Diffusion model configuration from the model's config.json.
/// Used by DiffusionGemma for parallel denoising text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DiffusionConfig {
    /// Maximum number of diffusion tokens per batch (default 256).
    pub max_diffusion_tokens: u32,
    /// Default number of denoising steps (4-8 for text).
    pub default_denoising_steps: u32,
    /// Noise schedule type (cosine, sqrt, linear).
    pub noise_schedule: NoiseScheduleType,
    /// Number of tokens generated per forward pass (15-20).
    pub parallel_token_generation: u32,
    /// Whether the model supports image inputs natively.
    pub supports_images: bool,
    /// Whether the model supports video inputs natively.
    pub supports_video: bool,
    /// Input image size in pixels (e.g. 896).
    pub image_size: u32,
    /// Patch size for image/video processing.
    pub patch_size: u32,
    /// Maximum context length (e.g. 262144).
    pub max_context_length: u32,
    /// Token ID used for masking in diffusion (default 0).
    pub mask_token_id: u32,
    /// Padding token ID (default 0).
    pub pad_token_id: u32,
    /// End-of-sequence token ID (default 0).
    pub eos_token_id: u32,
    /// Maximum canvas tokens for diffusion generation (default 256).
    pub max_canvas_tokens: u32,
    /// Dimension of the timestep embedding (default 4096).
    pub timestep_embedding_dim: u32,
    /// Confidence type for token selection (default LogProb).
    pub confidence_type: ConfidenceType,
    /// Default confidence threshold for commit decisions (default 0.7).
    pub default_confidence_threshold: f32,
    /// Whether EOS collapse is enabled (default true).
    pub eos_collapse_enabled: bool,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            max_diffusion_tokens: 256,
            default_denoising_steps: 6,
            noise_schedule: NoiseScheduleType::Cosine,
            parallel_token_generation: 18,
            supports_images: true,
            supports_video: true,
            image_size: 896,
            patch_size: 16,
            max_context_length: 262_144,
            mask_token_id: 0,
            pad_token_id: 0,
            eos_token_id: 0,
            max_canvas_tokens: 256,
            timestep_embedding_dim: 4096,
            confidence_type: ConfidenceType::LogProb,
            default_confidence_threshold: 0.7,
            eos_collapse_enabled: true,
        }
    }
}

/// Confidence type for token selection during diffusion decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceType {
    /// Use log-probability of token.
    LogProb,
    /// Use softmax margin (top - second).
    SoftmaxMargin,
    /// Use normalized entropy.
    NormalizedEntropy,
}

/// Mask selection strategy for discrete diffusion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MaskSelection {
    /// Mask tokens below a confidence threshold.
    Threshold { confidence_threshold: f32 },
    /// Mask a fixed ratio of tokens.
    Ratio { mask_ratio: f32 },
    /// Adaptively schedule masking.
    AdaptiveSchedule,
    /// Mask the lowest-confidence tokens.
    LowestConfidence,
}

/// Sampling policy for discrete diffusion decoding steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerPolicy {
    pub temperature: f32,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub mask_selection: MaskSelection,
}

/// Policy for committing tokens during a diffusion step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitPolicy {
    pub min_confidence: f32,
    pub max_commits_per_step: Option<u32>,
}

/// Condition under which diffusion decoding stops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopCondition {
    /// Stop after a fixed number of steps.
    MaxSteps(u32),
    /// Stop after N steps with no new commits.
    ConvergedAfter(u32),
    /// Stop when all tokens are committed.
    AllCommitted,
    /// Stop on EOS collapse.
    EosCollapse,
    /// Hard ceiling on total steps.
    HardStepCeiling(u32),
}

/// Forward pass strategy for a diffusion route.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffusionForwardRoute {
    /// Full transformer forward pass.
    FullTransformer,
    /// Cached transformer forward pass with a KV cache strategy.
    CachedTransformer { cache_strategy: KvCacheMode },
}

/// A single stage in the diffusion execution pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionStage {
    pub stage_index: u32,
    pub timestep: u32,
    pub forward_route: DiffusionForwardRoute,
    pub sampler_policy: SamplerPolicy,
    pub commit_policy: CommitPolicy,
    pub stop_conditions: Vec<StopCondition>,
}

/// Complete execution plan for a diffusion model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionExecutionPlan {
    pub stages: Vec<DiffusionStage>,
    pub total_denoising_steps: u32,
    pub kv_cache_mode: KvCacheMode,
    pub max_canvas_tokens: u32,
    pub final_logit_softcapping: Option<f64>,
}

/// Generation regime: autoregressive (token-by-token) or discrete diffusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationRegime {
    /// Standard autoregressive generation (token-by-token).
    Autoregressive,
    /// Discrete diffusion / parallel decoding.
    DiscreteDiffusion,
}

impl Default for GenerationRegime {
    fn default() -> Self {
        Self::Autoregressive
    }
}

/// KV cache strategy for diffusion or autoregressive decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCacheMode {
    /// Append new tokens to KV cache only.
    AppendOnly,
    /// Recompute the full KV cache at each step.
    FullRecompute,
    /// Block-wise KV cache with fixed-size blocks.
    BlockCache,
}

impl Default for KvCacheMode {
    fn default() -> Self {
        Self::AppendOnly
    }
}

/// Attention masking strategy for diffusion decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffusionAttentionKind {
    /// Full bidirectional attention.
    BidirectionalFull,
    /// Sliding window bidirectional attention.
    BidirectionalSliding,
}

/// Noise schedule type for diffusion denoising.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NoiseScheduleType {
    /// Cosine schedule (cosine-based noise weighting).
    Cosine,
    /// Square-root schedule (sqrt-based noise weighting).
    Sqrt,
    /// Linear schedule (linear noise weighting).
    Linear,
}

/// Compile-time quantization mode for the ComputeImage compiler.
/// When set, FP16/BF16 source weights are quantized at compile time
/// into packed triplets (packed weight + F32 scales + F32 biases).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompileQuantMode {
    /// 4-bit NormalFloat (NF4) block quantization.
    Nf4 { group_size: u32 },
    /// 8-bit affine quantization.
    Af8 { group_size: u32 },
}

impl CompileQuantMode {
    /// Parse a quant mode name into a CompileQuantMode.
    /// Supports "nf4", "nf4-128", "8bit", and "none" (no quantization).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "nf4" => Some(Self::Nf4 { group_size: 64 }),
            "nf4-128" => Some(Self::Nf4 { group_size: 128 }),
            "8bit" => Some(Self::Af8 { group_size: 64 }),
            "none" => None,
            _ => None,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nf4 { group_size: 64 } => "nf4",
            Self::Nf4 { group_size: 128 } => "nf4-128",
            Self::Af8 { .. } => "8bit",
            _ => "nf4-64",
        }
    }
}

/// Target hardware for a ComputeImage compilation.
/// Determines quantization, segment layout, and feature set.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HardwareTarget {
    /// Apple M1 (16GB baseline) — max compression, streaming-friendly segments
    M1,
    /// Apple M1 Pro/Max (32-64GB) — moderate compression
    M1Pro,
    /// Apple M2/M2 Pro/Max (24-96GB) — balanced
    M2,
    /// Apple M2 Ultra/M3 Max (96-192GB) — high precision
    M2Ultra,
    /// Apple M3 Ultra (256-512GB) — maximum precision, batched layout
    M3Ultra,
}

impl HardwareTarget {
    /// Auto-detect the current machine's target.
    pub fn detect() -> Self {
        let ram_mb = crate::gpu_memory::total_physical_ram_mb();
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8);

        match (ram_mb, cpu_count) {
            (r, c) if r >= 393_216 && c >= 24 => Self::M3Ultra, // 384GB+ & 24+ cores
            (r, c) if r >= 131_072 && c >= 20 => Self::M2Ultra, // 128GB+
            (r, c) if r >= 65_536 && c >= 12 => Self::M2,       // 64GB+
            (r, _c) if r >= 32_768 => Self::M1Pro,              // 32GB+
            _ => Self::M1,                                      // 16GB
        }
    }

    /// Optimal quantization for this hardware.
    pub fn recommended_quant(&self) -> &'static str {
        match self {
            Self::M1 => "nf4-128",
            Self::M1Pro => "nf4-64",
            Self::M2 => "nf4-64",
            Self::M2Ultra => "8bit",
            Self::M3Ultra => "none", // keep BF16/FP16
        }
    }

    /// Whether weight streaming is beneficial (low RAM systems).
    pub fn needs_weight_streaming(&self) -> bool {
        matches!(self, Self::M1 | Self::M1Pro)
    }

    /// Recommended batch size for prefill+decode.
    pub fn recommended_batch(&self) -> u32 {
        match self {
            Self::M1 => 4,
            Self::M1Pro => 8,
            Self::M2 => 12,
            Self::M2Ultra => 20,
            Self::M3Ultra => 32,
        }
    }

    /// Number of ANE cores available for speculation.
    pub fn ane_cores(&self) -> u32 {
        match self {
            Self::M1 | Self::M1Pro | Self::M2 => 16,
            Self::M2Ultra | Self::M3Ultra => 32,
        }
    }

    /// Segment layout: small + many for streaming, large + few for batched.
    pub fn segment_target_size_mb(&self) -> u32 {
        match self {
            Self::M1 => 64, // small segments for streaming
            Self::M1Pro => 128,
            Self::M2 => 256,
            Self::M2Ultra => 512,
            Self::M3Ultra => 1024, // huge segments, fewer files
        }
    }
}

// ── Layer 3: Compiled Execution Specification ──────────────────────────────

/// Full execution plan: one spec per layer, plus global tensors.
#[derive(Debug, Serialize)]
pub struct ExecutionSpec {
    pub architecture: TextArchitecture,
    pub namespace: NamespaceBinding,
    pub global_tensors: Vec<TensorBinding>,
    pub layers: Vec<LayerSpec>,
    pub quantization: Option<QuantizationMeta>,
}

/// A layer's complete specification.
#[derive(Clone, Debug, Serialize)]
pub struct LayerSpec {
    pub index: u32,
    pub attention_kind: AttentionKind,
    pub q_out: u32,
    pub kv_out: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub global_kv_out: Option<u32>,
    pub n_global_kv_heads: Option<u32>,
    pub global_head_dim: Option<u32>,
    pub rope_theta: f64,
    pub rope_type: String,
    pub partial_rotary_factor: Option<f64>,
    pub sliding_window: Option<u32>,
    pub tensors: Vec<TensorBinding>,
}

/// A single tensor's expected identity in the safetensors file.
#[derive(Clone, Debug, Serialize)]
pub struct TensorBinding {
    pub name: String,
    pub role: TensorRole,
    pub logical_shape: Vec<u32>,
    /// If quantized: the packed weight shape (i8→u32 packing).
    pub packed_shape: Option<PackedLinearShapes>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PackedLinearShapes {
    pub weight: Vec<u32>,
    pub scales: Vec<u32>,
    pub biases: Vec<u32>,
    pub bits: u32,
    pub group_size: u32,
    pub groups: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum TensorRole {
    Embedding,
    FinalNorm,
    LmHead,
    AttentionNorm,
    FfnNorm,
    QProj,
    KProj,
    VProj,
    OProj,
    GlobalKProj,
    GlobalVProj,
    GateProj,
    UpProj,
    DownProj,
    QNorm,
    KNorm,
}

// ── Compiler

/// Compile a TextArchitecture into an ExecutionSpec.
/// Complete model execution plan emitted by the compiler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelExecutionPlan {
    pub prologue: ProloguePlan,
    pub layers: Vec<LayerPlan>,
    pub epilogue: EpiloguePlan,
    /// Fused ANE regions compiled to .mlmodelc artifacts.
    /// Populated by AneFusionPass during compute-image build.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fused_ane_islands: Vec<AneFusedIsland>,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub sliding_window: u32,
    pub final_logit_softcapping: Option<f64>,
    pub tie_word_embeddings: bool,
    pub rms_norm_eps: f64,
    /// Speculative decoding config when this image is a paired draft+target compile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_config: Option<SpeculativeModelConfig>,
    /// Generation regime (autoregressive or discrete diffusion).
    #[serde(default)]
    pub generation_regime: GenerationRegime,
    /// Diffusion configuration, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diffusion_config: Option<DiffusionConfig>,
    /// Diffusion execution plan, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diffusion_execution_plan: Option<DiffusionExecutionPlan>,
    /// KV cache mode for generation.
    #[serde(default)]
    pub kv_cache_mode: KvCacheMode,
}

/// Segment ID containing the embedding table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProloguePlan {
    /// Segment ID containing the embedding table.
    pub segment_id: String,
    /// Tensor entry ID for the embedding weights.
    pub embedding_tensor_id: u32,
    /// Name used for ARRAY_REGISTRY lookup (e.g. "model.embed_tokens.weight").
    pub embedding_name: String,
    /// Expected embedding shape [vocab_size, hidden_size].
    pub embedding_shape: Vec<u32>,
    pub embedding_dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPlan {
    pub layer_index: u32,
    pub attention_kind: String, // "sliding_attention" or "full_attention"
    pub segment_id: String,
    pub hidden_size: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// For global layers only.
    pub global_head_dim: Option<u32>,
    pub n_global_kv_heads: Option<u32>,
    pub sliding_window: u32,
    pub rope_theta: f32,
    pub partial_rotary_factor: Option<f32>,
    pub attention_k_eq_v: bool,
    pub q_norm_enabled: bool,
    pub k_norm_enabled: bool,
    /// Tensor IDs for this layer's weights in the tensor_table.
    pub q_proj_tensor_id: u32,
    pub k_proj_tensor_id: u32,
    pub v_proj_tensor_id: u32,
    pub o_proj_tensor_id: u32,
    pub q_norm_tensor_id: Option<u32>,
    pub k_norm_tensor_id: Option<u32>,
    pub gate_proj_tensor_id: u32,
    pub up_proj_tensor_id: u32,
    pub down_proj_tensor_id: u32,
    pub input_layernorm_tensor_id: u32,
    pub post_attention_layernorm_tensor_id: u32,
    pub pre_ffw_layernorm_tensor_id: Option<u32>,
    pub post_ffw_layernorm_tensor_id: Option<u32>,
    /// Layer scalars and other optional tensors.
    pub layer_scalar_ids: Vec<u32>,
    /// Quantization descriptor IDs for packed weight groups.
    pub quantization_ids: Vec<String>,
    /// Per-operation backend routing for heterogeneous dispatch.
    #[serde(default)]
    pub route: operation_route::OperationRoute,
    /// Fused operations detected at compile time.
    /// Populated by [`ModelExecutionPlan::apply_fusion_pass`] during
    /// compute-image build.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fused_operations: Vec<FusedOperation>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpiloguePlan {
    pub segment_id: String,
    pub final_norm_tensor_id: u32,
    pub final_norm_name: String,
    pub output_projection_tensor_id: Option<u32>,
    pub output_projection_name: Option<String>,
    pub final_logit_softcapping: Option<f64>,
    pub vocab_size: u32,
}

/// A fused ANE region compiled to a single .mlmodelc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneFusedIsland {
    pub island_id: String,
    pub modelc_relpath: String,
    pub layer_indices: Vec<u32>,
    pub compute_units: String,
    pub function_name: String,
    /// Semantic subgraph kind for this fused island.
    #[serde(default)]
    pub subgraph_kind: String,
}

/// A fused operation composed of multiple atomic operations.
/// The corresponding Metal kernel is precompiled and stored in the
/// model's .metallib.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusedOperation {
    /// rms_norm + q_proj matmul
    FusedNormQProj,
    /// rms_norm + k_proj matmul
    FusedNormKProj,
    /// rms_norm + v_proj matmul
    FusedNormVProj,
    /// silu(gate_proj(x)) * up_proj(x) + down_proj
    FusedFfnActivation,
    /// residual_add + rms_norm
    FusedResidualNorm,
    /// q @ k^T + softmax + @ v
    FusedFlashAttention,
    /// Gate -> Top-K -> Expert FFN -> Combine
    FusedMoERoute,
    /// Plugin-provided operation (kernel name looked up in PLUGIN_REGISTRY).
    Custom(String),
}

impl FusedOperation {
    /// Return the name of the precompiled Metal kernel.
    pub fn kernel_name(&self) -> &str {
        match self {
            Self::FusedNormQProj => "fused_norm_q_proj",
            Self::FusedNormKProj => "fused_norm_k_proj",
            Self::FusedNormVProj => "fused_norm_v_proj",
            Self::FusedFfnActivation => "fused_ffn_activation",
            Self::FusedResidualNorm => "fused_residual_norm",
            Self::FusedFlashAttention => "fused_flash_attention",
            Self::FusedMoERoute => "fused_moe_route",
            Self::Custom(name) => name.as_str(),
        }
    }
}

impl LayerPlan {
    /// Return the logical operation names for this layer in execution order,
    /// derived from which weight tensors are present.
    ///
    /// This is used by [`ModelExecutionPlan::apply_fusion_pass`] to detect
    /// fusible operation patterns.
    pub fn operation_names(&self) -> Vec<&'static str> {
        let mut ops = Vec::with_capacity(16);

        // Pre-attention: rms_norm + QKV projections
        if self.input_layernorm_tensor_id != 0 {
            ops.push("rms_norm");
        }
        if self.q_proj_tensor_id != 0 {
            ops.push("q_proj");
        }
        if self.k_proj_tensor_id != 0 {
            ops.push("k_proj");
        }
        if self.v_proj_tensor_id != 0 {
            ops.push("v_proj");
        }

        // Attention core: matmul(Q, K^T), softmax, matmul(probs, V)
        if self.q_proj_tensor_id != 0 && self.k_proj_tensor_id != 0 {
            ops.push("matmul");
            ops.push("softmax");
        }
        if self.q_proj_tensor_id != 0 && self.v_proj_tensor_id != 0 {
            ops.push("matmul");
        }

        // Post-attention residual + norm
        if self.o_proj_tensor_id != 0 {
            ops.push("add");
        }
        if self.post_attention_layernorm_tensor_id != 0 {
            ops.push("rms_norm");
        }

        // FFN: gate_proj, silu, multiply(up_proj, silu(gate)), add(sum)
        if self.gate_proj_tensor_id != 0 {
            ops.push("gate_proj");
            ops.push("silu");
        }
        if self.up_proj_tensor_id != 0 {
            ops.push("multiply");
        }
        if self.down_proj_tensor_id != 0 {
            ops.push("down_proj");
        }

        ops
    }
}

/// Check if `ops` contains the contiguous sequence of operation names in
/// `pattern`.  Returns `true` when every element of `pattern` appears in
/// order as a subsequence of `ops`.
fn has_pattern(ops: &[&str], pattern: &[&str]) -> bool {
    if pattern.is_empty() {
        return true;
    }
    let mut pi = 0;
    for &op in ops {
        if op == pattern[pi] {
            pi += 1;
            if pi == pattern.len() {
                return true;
            }
        }
    }
    false
}

impl ModelExecutionPlan {
    /// Scan adjacent layers with ANE-routed ops and populate fused_ane_islands.
    /// Called during compute-image build after routes are assigned.
    pub fn build_ane_fusion_plan(&mut self) {
        let mut islands: Vec<AneFusedIsland> = Vec::new();
        let mut i = 0;
        while i < self.layers.len() {
            let is_ane = self.layers[i].route.has_ane_backend();
            if !is_ane {
                i += 1;
                continue;
            }
            let mut layer_indices = vec![self.layers[i].layer_index];
            i += 1;
            while i < self.layers.len() && self.layers[i].route.has_ane_backend() {
                layer_indices.push(self.layers[i].layer_index);
                i += 1;
            }
            if layer_indices.len() >= 2 {
                let island_id = format!(
                    "ane_fused_layer{}-{}",
                    layer_indices[0],
                    layer_indices.last().unwrap()
                );
                let modelc_path = format!("{}.modelc", island_id);
                // Determine subgraph kind from the first layer's ops
                let first_idx = layer_indices[0] as usize;
                let first_ops = self.layers[first_idx].operation_names();
                let subgraph_kind = if first_ops.contains(&"gate_proj") && first_ops.contains(&"down_proj") {
                    "mlp_block".to_string()
                } else if first_ops.contains(&"q_proj")
                    && first_ops.contains(&"k_proj")
                    && first_ops.contains(&"v_proj")
                    && !first_ops.contains(&"rms_norm")
                {
                    "qkv_bundle".to_string()
                } else if first_ops.contains(&"rms_norm") && first_ops.contains(&"q_proj") {
                    "rmsnorm_qkv".to_string()
                } else if first_ops.contains(&"lm_head") {
                    "output_proj".to_string()
                } else {
                    "mlp_block".to_string()
                };
                islands.push(AneFusedIsland {
                    island_id,
                    modelc_relpath: modelc_path,
                    layer_indices,
                    compute_units: "cpuAndNeuralEngine".to_string(),
                    function_name: "main".to_string(),
                    subgraph_kind,
                });
            }
        }
        self.fused_ane_islands = islands;
    }
}

/// Config for a model pair compiled for speculative decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeModelConfig {
    /// Draft model architecture config
    pub draft_architecture: TextArchitecture,
    /// Target model architecture config
    pub target_architecture: TextArchitecture,
    /// Shared components
    pub shared_embedding: bool,
    pub shared_lm_head: bool,
    /// Segment ordering: draft layers come first for fast startup
    pub draft_first_segments: bool,
    /// Maximum draft speculation length
    pub speculation_length: u32,
}

impl ModelExecutionPlan {
    /// Post-plan fusion pass: detect common operation patterns and
    /// annotate layers with fused operations.
    ///
    /// Recognised patterns:
    /// - `rms_norm` + `q_proj` / `k_proj` / `v_proj` → single fused kernel
    /// - `silu(multiply)` → fused FFN activation
    /// - `add` + `rms_norm` → fused residual+norm
    /// - `matmul` + `softmax` + `matmul` → fused flash attention
    ///
    /// Called during compute-image build after the execution plan is
    /// assembled and before `build_ane_fusion_plan`.
    pub fn apply_fusion_pass(&mut self) {
        let pattern_norm_q = &["rms_norm", "q_proj"];
        let pattern_norm_k = &["rms_norm", "k_proj"];
        let pattern_norm_v = &["rms_norm", "v_proj"];
        let pattern_silu_mul = &["silu", "multiply"];
        let pattern_add_norm = &["add", "rms_norm"];
        let pattern_mm_soft_mm = &["matmul", "softmax", "matmul"];

        for layer in &mut self.layers {
            let mut fused = Vec::new();
            let ops = layer.operation_names();

            if has_pattern(&ops, pattern_norm_q) {
                fused.push(FusedOperation::FusedNormQProj);
            }
            if has_pattern(&ops, pattern_norm_k) {
                fused.push(FusedOperation::FusedNormKProj);
            }
            if has_pattern(&ops, pattern_norm_v) {
                fused.push(FusedOperation::FusedNormVProj);
            }
            if has_pattern(&ops, pattern_silu_mul) {
                fused.push(FusedOperation::FusedFfnActivation);
            }
            if has_pattern(&ops, pattern_add_norm) {
                fused.push(FusedOperation::FusedResidualNorm);
            }
            if has_pattern(&ops, pattern_mm_soft_mm) {
                fused.push(FusedOperation::FusedFlashAttention);
            }

            layer.fused_operations = fused;
        }
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.layers.is_empty() {
            errors.push("execution plan has zero layers".into());
        }
        for (i, plan) in self.layers.iter().enumerate() {
            if plan.layer_index != i as u32 {
                errors.push(format!("layer {} has index {}", i, plan.layer_index));
            }
            if plan.hidden_size != self.hidden_size {
                errors.push(format!(
                    "layer {} hidden_size {} != model {}",
                    i, plan.hidden_size, self.hidden_size
                ));
            }
            if plan.q_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero q_proj_tensor_id", i));
            }
            if plan.k_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero k_proj_tensor_id", i));
            }
            if plan.o_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero o_proj_tensor_id", i));
            }
            if plan.gate_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero gate_proj_tensor_id", i));
            }
            if plan.up_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero up_proj_tensor_id", i));
            }
            if plan.down_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero down_proj_tensor_id", i));
            }
            if plan.input_layernorm_tensor_id == 0 {
                errors.push(format!("layer {} has zero input_layernorm_tensor_id", i));
            }
            if plan.post_attention_layernorm_tensor_id == 0 {
                errors.push(format!(
                    "layer {} has zero post_attention_layernorm_tensor_id",
                    i
                ));
            }
            match plan.attention_kind.as_str() {
                "sliding_attention" => {
                    if plan.v_proj_tensor_id == 0 {
                        errors.push(format!("sliding layer {} has zero v_proj_tensor_id", i));
                    }
                }
                "full_attention" => {
                    if plan.global_head_dim.is_none() {
                        errors.push(format!(
                            "full-attention layer {} missing global_head_dim",
                            i
                        ));
                    }
                }
                other => {
                    errors.push(format!("layer {} has unknown attention_kind: {}", i, other));
                }
            }
            let expected_seg = format!("layer_{}", i);
            if plan.segment_id != expected_seg {
                errors.push(format!(
                    "layer {} segment_id '{}' != expected '{}'",
                    i, plan.segment_id, expected_seg
                ));
            }
        }
        // embedding_tensor_id can be 0 when it's the first tensor emitted.
        // if self.prologue.embedding_tensor_id == 0 {
        //     errors.push("prologue has zero embedding_tensor_id".into());
        // }
        if self.epilogue.final_norm_tensor_id == 0 {
            errors.push("epilogue has zero final_norm_tensor_id".into());
        }
        if self.epilogue.vocab_size == 0 {
            errors.push("epilogue has zero vocab_size".into());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Build a ModelExecutionPlan from the TextArchitecture, namespace, and emitted tensor IDs.
/// Called during ComputeImage compilation after all tensors have been assigned IDs.
pub fn build_execution_plan(
    arch: &TextArchitecture,
    namespace: &NamespaceBinding,
    emitted_ids: &std::collections::HashMap<String, u32>,
) -> ModelExecutionPlan {
    let root = &namespace.root;
    let mut layers = Vec::with_capacity(arch.layer_types.len());

    for (i, kind) in arch.layer_types.iter().enumerate() {
        let layer = i as u32;
        let base = format!("{}.layers.{}", root, layer);
        let is_full = *kind == AttentionKind::FullAttention;

        let get = |suffix: &str| -> u32 {
            let name = format!("{}.{}", base, suffix);
            emitted_ids.get(&name).copied().unwrap_or(0)
        };
        let get_opt = |suffix: &str| -> Option<u32> {
            let name = format!("{}.{}", base, suffix);
            emitted_ids.get(&name).copied()
        };

        let rope = if is_full {
            arch.rope_global.as_ref().unwrap_or(&arch.rope_local)
        } else {
            &arch.rope_local
        };

        let hdim = if is_full {
            arch.global_head_dim.unwrap_or(arch.head_dim)
        } else {
            arch.head_dim
        };
        let n_kv = if is_full {
            arch.num_global_key_value_heads
                .unwrap_or(arch.num_key_value_heads)
        } else {
            arch.num_key_value_heads
        };

        layers.push(LayerPlan {
            layer_index: layer,
            attention_kind: if is_full {
                "full_attention".into()
            } else {
                "sliding_attention".into()
            },
            segment_id: format!("layer_{}", layer),
            hidden_size: arch.hidden_size,
            n_heads: arch.num_attention_heads,
            n_kv_heads: n_kv,
            head_dim: hdim,
            global_head_dim: if is_full { arch.global_head_dim } else { None },
            n_global_kv_heads: if is_full {
                arch.num_global_key_value_heads
            } else {
                None
            },
            sliding_window: arch.sliding_window,
            rope_theta: rope.theta as f32,
            partial_rotary_factor: rope.partial_rotary_factor.map(|f| f as f32),
            attention_k_eq_v: arch.attention_k_eq_v && is_full,
            q_norm_enabled: true,
            k_norm_enabled: true,
            q_proj_tensor_id: get("self_attn.q_proj.weight"),
            k_proj_tensor_id: get("self_attn.k_proj.weight"),
            v_proj_tensor_id: if is_full {
                get("self_attn.k_proj.weight") // alias: K-equals-V
            } else {
                get("self_attn.v_proj.weight")
            },
            o_proj_tensor_id: get("self_attn.o_proj.weight"),
            q_norm_tensor_id: get_opt("self_attn.q_norm.weight"),
            k_norm_tensor_id: get_opt("self_attn.k_norm.weight"),
            gate_proj_tensor_id: get("mlp.gate_proj.weight"),
            up_proj_tensor_id: get("mlp.up_proj.weight"),
            down_proj_tensor_id: get("mlp.down_proj.weight"),
            input_layernorm_tensor_id: get("input_layernorm.weight"),
            post_attention_layernorm_tensor_id: get("post_attention_layernorm.weight"),
            pre_ffw_layernorm_tensor_id: None,
            post_ffw_layernorm_tensor_id: None,
            layer_scalar_ids: Vec::new(),
            quantization_ids: Vec::new(),
            route: Default::default(),
            fused_operations: Vec::new(),
        });
    }

    let embed_name = format!("{}.embed_tokens.weight", root);
    let fn_name = format!("{}.norm.weight", root);
    let lm_head_name = namespace.lm_head_key.clone();

    // ── Backend assessment ────────────────────────────────────────────
    // Assign per-operation backends for each layer.
    // Element-wise ops → Accelerate (vDSP), attention → Orion/MLX,
    // matmuls → MLX GPU.  This routes every operation to its optimal backend.
    for layer in &mut layers {
        let is_full = layer.attention_kind == "full_attention";
        layer.route = operation_route::OperationRoute {
            // RMS norm: Accelerate vDSP_vsma + vvfrsqrtf (fastest CPU path)
            rms_norm: 1,
            // SiLU: Accelerate vDSP_vsigmoid + vDSP_vmul (GPU overhead not worth it)
            // Benchmark: MLX wins at 256+ (3.8μs vs Accel 24μs at 1K)
            silu: 0,
            // Matmuls: MLX GPU (cblas_sgemm is slower on CPU)
            matmul: 0,
            // Dense attention: Orion ANE for full, MLX GPU for sliding
            attention: if is_full { 3 } else { 0 },
            // Softmax: MLX GPU (multiple vDSP calls on CPU are slower)
            softmax: 0,
            // RoPE: MLX GPU (trig table generation faster on GPU)
            rope: 0,
            // Add: Accelerate vDSP_vadd (trivial, no GPU overhead needed)
            add: 1,
            // Multiply: Accelerate vDSP_vmul (same rationale)
            multiply: 1,
            // Transpose: Accelerate vDSP_mtrans
            // Benchmark: MLX wins at 256+ (2.1μs vs Accel 971μs at 1024)
            transpose: 0,
            // Reshape: Accelerate storage layer (no-op)
            reshape: 1,
        };
    }

    ModelExecutionPlan {
        prologue: ProloguePlan {
            segment_id: "persistent".into(),
            embedding_tensor_id: emitted_ids.get(&embed_name).copied().unwrap_or(0),
            embedding_name: embed_name,
            embedding_shape: vec![arch.vocab_size, arch.hidden_size],
            embedding_dtype: if arch.model_type == "qwen2" {
                "BF16".into()
            } else {
                "F32".into()
            },
        },
        layers,
        epilogue: EpiloguePlan {
            segment_id: "persistent".into(),
            final_norm_tensor_id: emitted_ids.get(&fn_name).copied().unwrap_or(0),
            final_norm_name: fn_name,
            output_projection_tensor_id: emitted_ids.get(&lm_head_name).copied(),
            output_projection_name: Some(lm_head_name),
            final_logit_softcapping: arch.final_logit_softcapping,
            vocab_size: arch.vocab_size,
        },
        fused_ane_islands: vec![],
        hidden_size: arch.hidden_size,
        vocab_size: arch.vocab_size,
        sliding_window: arch.sliding_window,
        final_logit_softcapping: arch.final_logit_softcapping,
        tie_word_embeddings: arch.tie_word_embeddings,
        rms_norm_eps: arch.rms_norm_eps,
        ..Default::default()
    }
}

pub fn compile(
    arch: &TextArchitecture,
    namespace: &NamespaceBinding,
    q: Option<&QuantizationMeta>,
) -> ExecutionSpec {
    let mut spec = ExecutionSpec {
        architecture: arch.clone(),
        namespace: NamespaceBinding {
            root: namespace.root.clone(),
            discovery: namespace.discovery.clone(),
            lm_head_key: namespace.lm_head_key.clone(),
            lm_head_aliased: namespace.lm_head_aliased,
        },
        global_tensors: Vec::new(),
        layers: Vec::new(),
        quantization: q.cloned(),
    };

    let root = &namespace.root;
    // bits=0 and gs=0 means no quantization (quantized_linear returns None for packed_shape)
    let bits = q.as_ref().map(|m| m.bits).unwrap_or(0);
    // bits=0 means no quantization. quantized_linear checks bits>0.
    // bits=0 and gs=0 means no quantization (quantized_linear returns None for packed_shape)
    let bits = q.as_ref().map(|m| m.bits).unwrap_or(0);
    let gs = q.map(|m| m.group_size).unwrap_or(64);

    // Embedding (quantized in 8-bit models)
    spec.global_tensors.push(TensorBinding {
        name: format!("{}.embed_tokens.weight", root),
        role: TensorRole::Embedding,
        logical_shape: vec![arch.vocab_size, arch.hidden_size],
        packed_shape: if q.is_some() {
            let gs = q.as_ref().map(|m| m.group_size).unwrap_or(64);
            let bits = q.as_ref().map(|m| m.bits).unwrap_or(16);
            // U32 storage packs 32/bits logical elements per physical element
            let pack = 32 / bits;
            let packed_in = arch.hidden_size / pack;
            let n_groups = arch.hidden_size / gs;
            Some(PackedLinearShapes {
                weight: vec![arch.vocab_size, packed_in],
                scales: vec![arch.vocab_size, n_groups],
                biases: vec![arch.vocab_size, n_groups],
                bits,
                group_size: gs,
                groups: n_groups,
            })
        } else {
            None
        },
    });

    // Final norm
    spec.global_tensors.push(TensorBinding {
        name: format!("{}.norm.weight", root),
        role: TensorRole::FinalNorm,
        logical_shape: vec![arch.hidden_size],
        packed_shape: None,
    });

    // LM head
    if !arch.tie_word_embeddings {
        spec.global_tensors.push(TensorBinding {
            name: format!("{}.lm_head.weight", root),
            role: TensorRole::LmHead,
            logical_shape: vec![arch.vocab_size, arch.hidden_size],
            packed_shape: None,
        });
    }

    // Per-layer compilation
    for (i, kind) in arch.layer_types.iter().enumerate() {
        let layer = i as u32;
        let is_full = *kind == AttentionKind::FullAttention;

        let rope = if is_full {
            arch.rope_global.as_ref().unwrap_or(&arch.rope_local)
        } else {
            &arch.rope_local
        };

        let mut tensors = Vec::new();

        // Attention norms
        tensors.push(norm_binding(
            root,
            layer,
            "input_layernorm",
            TensorRole::AttentionNorm,
            arch.hidden_size,
        ));
        tensors.push(norm_binding(
            root,
            layer,
            "post_attention_layernorm",
            TensorRole::FfnNorm,
            arch.hidden_size,
        ));

        // QK norms
        let norm_dim = if is_full {
            arch.global_head_dim.unwrap_or(arch.head_dim)
        } else {
            arch.head_dim
        };
        tensors.push(TensorBinding {
            name: format!("{}.layers.{}.self_attn.q_norm.weight", root, layer),
            role: TensorRole::QNorm,
            logical_shape: vec![norm_dim],
            packed_shape: None,
        });
        tensors.push(TensorBinding {
            name: format!("{}.layers.{}.self_attn.k_norm.weight", root, layer),
            role: TensorRole::KNorm,
            logical_shape: vec![norm_dim],
            packed_shape: None,
        });

        // QKV projections
        // Full-attention: k_proj uses global dims (1×512), no separate v_proj
        let actual_kv_out = if is_full {
            arch.num_global_key_value_heads.unwrap_or(1)
                * arch.global_head_dim.unwrap_or(arch.head_dim)
        } else {
            arch.num_key_value_heads * arch.head_dim
        };
        tensors.push(quantized_linear(
            root,
            layer,
            "self_attn.q_proj",
            TensorRole::QProj,
            if is_full {
                arch.num_attention_heads * arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.num_attention_heads * arch.head_dim
            },
            arch.hidden_size,
            gs,
            bits,
        ));
        tensors.push(quantized_linear(
            root,
            layer,
            "self_attn.k_proj",
            TensorRole::KProj,
            actual_kv_out,
            arch.hidden_size,
            gs,
            bits,
        ));
        if !is_full {
            tensors.push(quantized_linear(
                root,
                layer,
                "self_attn.v_proj",
                TensorRole::VProj,
                arch.num_key_value_heads * arch.head_dim,
                arch.hidden_size,
                gs,
                bits,
            ));
        }
        tensors.push(quantized_linear(
            root,
            layer,
            "self_attn.o_proj",
            TensorRole::OProj,
            arch.hidden_size,
            if is_full {
                arch.num_attention_heads * arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.num_attention_heads * arch.head_dim
            },
            gs,
            bits,
        ));

        // MLP
        tensors.push(quantized_linear(
            root,
            layer,
            "mlp.gate_proj",
            TensorRole::GateProj,
            arch.intermediate_size,
            arch.hidden_size,
            gs,
            bits,
        ));
        tensors.push(quantized_linear(
            root,
            layer,
            "mlp.up_proj",
            TensorRole::UpProj,
            arch.intermediate_size,
            arch.hidden_size,
            gs,
            bits,
        ));
        tensors.push(quantized_linear(
            root,
            layer,
            "mlp.down_proj",
            TensorRole::DownProj,
            arch.hidden_size,
            arch.intermediate_size,
            gs,
            bits,
        ));

        let sliding_window = if is_full {
            None
        } else {
            Some(arch.sliding_window)
        };

        spec.layers.push(LayerSpec {
            index: layer,
            attention_kind: kind.clone(),
            q_out: if is_full {
                arch.num_attention_heads * arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.num_attention_heads * arch.head_dim
            },
            kv_out: if is_full {
                arch.num_global_key_value_heads.unwrap_or(1)
                    * arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.num_key_value_heads * arch.head_dim
            },
            n_heads: arch.num_attention_heads,
            n_kv_heads: arch.num_key_value_heads,
            head_dim: if is_full {
                arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.head_dim
            },
            global_kv_out: if is_full {
                Some(
                    arch.num_global_key_value_heads.unwrap_or(1)
                        * arch.global_head_dim.unwrap_or(arch.head_dim),
                )
            } else {
                None
            },
            n_global_kv_heads: arch.num_global_key_value_heads,
            global_head_dim: arch.global_head_dim,
            rope_theta: rope.theta,
            rope_type: rope.rope_type.clone(),
            partial_rotary_factor: rope.partial_rotary_factor,
            sliding_window,
            tensors,
        });
    }

    spec
}

/// Filter the compiled spec to only include bindings for tensors that exist
/// in the source model's tensor map. This makes the compiler dynamic — it
/// adapts to model architectures that omit optional tensors (e.g., Q/K norms
/// in Qwen2.5, biases in newer architectures).
pub fn filter_spec_to_existing(spec: &mut ExecutionSpec, existing_tensor_names: &HashSet<String>) {
    // Global tensors: remove bindings for tensors that don't exist
    spec.global_tensors.retain(|b| {
        if existing_tensor_names.contains(&b.name) {
            true
        } else {
            eprintln!(
                "[dynamic-compile] skipping missing global tensor: {}",
                b.name
            );
            false
        }
    });

    // Layer tensors: check per-layer bindings
    for layer in spec.layers.iter_mut() {
        layer.tensors.retain(|b| {
            if existing_tensor_names.contains(&b.name) {
                true
            } else {
                eprintln!(
                    "[dynamic-compile] skipping missing layer tensor: {}",
                    b.name
                );
                false
            }
        });
    }
}

fn norm_binding(root: &str, layer: u32, name: &str, role: TensorRole, dim: u32) -> TensorBinding {
    TensorBinding {
        name: format!("{}.layers.{}.{}.weight", root, layer, name),
        role,
        logical_shape: vec![dim],
        packed_shape: None,
    }
}

fn quantized_linear(
    root: &str,
    layer: u32,
    proj_name: &str,
    role: TensorRole,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    bits: u32,
) -> TensorBinding {
    // U32 storage packs `32 / bits` logical elements per physical element.
    let packed_shape = if bits > 0 && bits <= 16 {
        // U32 storage packs `32 / bits` logical elements per physical element.
        let pack = 32 / bits;
        let packed_in = in_dim / pack;
        let n_groups = in_dim / group_size;

        Some(PackedLinearShapes {
            weight: vec![out_dim, packed_in],
            scales: vec![out_dim, n_groups],
            biases: vec![out_dim, n_groups],
            bits,
            group_size,
            groups: n_groups,
        })
    } else {
        None
    };

    TensorBinding {
        name: format!("{}.layers.{}.{}.weight", root, layer, proj_name),
        role,
        logical_shape: vec![out_dim, in_dim],
        packed_shape,
    }
}

// Re-exported from config_namespace module.
pub use crate::config_namespace::*;

// ── Raw JSON parsing to normalized types ───────────────────────────────────

#[derive(Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_type: Option<String>,
    // Fallback fields for flat configs (no nested text_config)
    #[serde(default)]
    hidden_size: Option<u32>,
    #[serde(default)]
    intermediate_size: Option<u32>,
    #[serde(default)]
    num_attention_heads: Option<u32>,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    #[serde(default)]
    head_dim: Option<u32>,
    #[serde(default)]
    global_head_dim: Option<u32>,
    #[serde(default)]
    num_global_key_value_heads: Option<u32>,
    #[serde(default)]
    num_hidden_layers: Option<u32>,
    #[serde(default)]
    vocab_size: Option<u32>,
    #[serde(default)]
    sliding_window: Option<u32>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    attention_k_eq_v: Option<bool>,
    #[serde(default)]
    final_logit_softcapping: Option<f64>,
    #[serde(default)]
    hidden_size_per_layer_input: Option<u32>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    enable_moe_block: Option<bool>,
    #[serde(default)]
    moe_intermediate_size: Option<u32>,
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    top_k_experts: Option<u32>,
    #[serde(default)]
    num_kv_shared_layers: Option<u32>,
    text_config: Option<RawTextConfig>,
    #[serde(default)]
    vision_config: Option<VisionArchitecture>,
    #[serde(default)]
    audio_config: Option<AudioArchitecture>,
    #[serde(default)]
    #[serde(alias = "quantization_config")]
    quantization: Option<RawQuantization>,
    #[serde(default)]
    max_position_embeddings: Option<u32>,
    #[serde(default)]
    dtype: Option<String>,
}
impl RawConfig {
    fn to_text_config_fallback(&self) -> RawTextConfig {
        RawTextConfig {
            hidden_size: self.hidden_size.unwrap_or(2048),
            intermediate_size: self.intermediate_size.unwrap_or(8192),
            num_attention_heads: self.num_attention_heads.unwrap_or(16),
            num_key_value_heads: self.num_key_value_heads.unwrap_or(4),
            head_dim: self.head_dim.unwrap_or_else(|| {
                self.hidden_size.unwrap_or(2048) / self.num_attention_heads.unwrap_or(16)
            }),
            global_head_dim: self.global_head_dim,
            num_global_key_value_heads: self.num_global_key_value_heads,
            num_hidden_layers: self.num_hidden_layers.unwrap_or(24),
            vocab_size: self.vocab_size.unwrap_or(32768),
            sliding_window: self.sliding_window.unwrap_or(32768),
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-6),
            tie_word_embeddings: self.tie_word_embeddings,
            attention_k_eq_v: self.attention_k_eq_v,
            final_logit_softcapping: self.final_logit_softcapping,
            hidden_size_per_layer_input: self.hidden_size_per_layer_input,
            layer_types: self.layer_types.clone().unwrap_or_default(),
            rope_parameters: None,
            model_type: self.model_type.clone(),
        }
    }
}

#[derive(Deserialize, Clone)]
struct RawTextConfig {
    hidden_size: u32,
    intermediate_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    global_head_dim: Option<u32>,
    num_global_key_value_heads: Option<u32>,
    num_hidden_layers: u32,
    vocab_size: u32,
    sliding_window: u32,
    max_position_embeddings: Option<u32>,
    rms_norm_eps: f64,
    tie_word_embeddings: Option<bool>,
    attention_k_eq_v: Option<bool>,
    final_logit_softcapping: Option<f64>,
    hidden_size_per_layer_input: Option<u32>,
    layer_types: Vec<String>,
    rope_parameters: Option<RawRopeParams>,
    model_type: Option<String>,
}

#[derive(Deserialize, Clone)]
struct RawRopeParams {
    sliding_attention: Option<RawRopeSpec>,
    full_attention: Option<RawRopeSpec>,
}

#[derive(Deserialize, Clone)]
struct RawRopeSpec {
    rope_theta: f64,
    rope_type: Option<String>,
    partial_rotary_factor: Option<f64>,
}

#[derive(Deserialize, Clone)]
struct RawQuantization {
    group_size: Option<u32>,
    bits: Option<u32>,
    mode: Option<String>,
}

/// Parse config.json and produce a normalized TextArchitecture + QuantizationMeta.
pub fn parse_config(
    config_path: &str,
) -> crate::Result<(TextArchitecture, Option<QuantizationMeta>, ModelManifest)> {
    let config_json = std::fs::read_to_string(config_path)
        .map_err(|e| crate::Error::from_reason(format!("Cannot read config: {}", e)))?;

    // Hash the raw config for provenance
    let mut hasher = Sha256::new();
    hasher.update(config_json.as_bytes());
    let config_hash = format!("{:x}", hasher.finalize());

    let raw: RawConfig = serde_json::from_str(&config_json)
        .map_err(|e| crate::Error::from_reason(format!("Invalid config JSON: {}", e)))?;

    let text = raw
        .text_config
        .clone()
        .unwrap_or_else(|| raw.to_text_config_fallback());

    let max_pos = text
        .max_position_embeddings
        .or(raw.max_position_embeddings)
        .unwrap_or(131072);

    let mut layer_types: Vec<AttentionKind> = text
        .layer_types
        .iter()
        .map(|s| match s.as_str() {
            "full_attention" => AttentionKind::FullAttention,
            _ => AttentionKind::SlidingAttention,
        })
        .collect();

    // If layer_types is empty (flat configs like Qwen, Llama), default to all sliding.
    if layer_types.is_empty() {
        for _ in 0..text.num_hidden_layers {
            layer_types.push(AttentionKind::SlidingAttention);
        }
    } else if layer_types.len() != text.num_hidden_layers as usize {
        return Err(crate::Error::from_reason(format!(
            "layer_types count ({}) != num_hidden_layers ({})",
            layer_types.len(),
            text.num_hidden_layers
        )));
    }

    let rope_local = {
        let raw_rope = text
            .rope_parameters
            .as_ref()
            .and_then(|r| r.sliding_attention.as_ref())
            .map(|s| RopeSpec {
                theta: s.rope_theta,
                rope_type: s.rope_type.clone().unwrap_or_else(|| "default".into()),
                partial_rotary_factor: s.partial_rotary_factor,
            })
            .unwrap_or_else(|| RopeSpec {
                theta: 10000.0,
                rope_type: "default".into(),
                partial_rotary_factor: None,
            });
        raw_rope
    };

    let rope_global = text
        .rope_parameters
        .as_ref()
        .and_then(|r| r.full_attention.as_ref())
        .map(|s| RopeSpec {
            theta: s.rope_theta,
            rope_type: s.rope_type.clone().unwrap_or_else(|| "proportional".into()),
            partial_rotary_factor: s.partial_rotary_factor,
        });

    let moe_config = if raw.enable_moe_block.unwrap_or(false) {
        let num_experts = raw.num_experts.unwrap_or(0);
        let top_k = raw.top_k_experts.unwrap_or(1);
        let inter_size = raw
            .moe_intermediate_size
            .or_else(|| Some(text.intermediate_size))
            .unwrap_or(0);
        if num_experts > 0 && top_k > 0 {
            Some(MoEConfig {
                num_experts,
                top_k_experts: top_k,
                intermediate_size: inter_size,
                shared_experts: false,
            })
        } else {
            None
        }
    } else {
        None
    };

    let arch = TextArchitecture {
        diffusion_config: None,
        hidden_size: text.hidden_size,
        intermediate_size: text.intermediate_size,
        num_attention_heads: text.num_attention_heads,
        num_key_value_heads: text.num_key_value_heads,
        head_dim: text.head_dim,
        global_head_dim: text.global_head_dim,
        num_global_key_value_heads: text.num_global_key_value_heads,
        num_hidden_layers: text.num_hidden_layers,
        vocab_size: text.vocab_size,
        sliding_window: text.sliding_window,
        max_position_embeddings: max_pos,
        rms_norm_eps: text.rms_norm_eps,
        tie_word_embeddings: text.tie_word_embeddings.unwrap_or(true),
        attention_k_eq_v: text.attention_k_eq_v.unwrap_or(true),
        final_logit_softcapping: text.final_logit_softcapping,
        hidden_size_per_layer_input: text.hidden_size_per_layer_input.unwrap_or(0),
        layer_types,
        rope_local,
        rope_global,
        model_type: text
            .model_type
            .clone()
            .unwrap_or_else(|| "gemma4_unified_text".into()),
        moe_config,
    };

    let q_bits = raw.quantization.as_ref().and_then(|q| q.bits);
    let q_group_size = raw.quantization.as_ref().and_then(|q| q.group_size);
    let has_explicit_quant = raw.quantization.is_some();
    let explicit_quant = raw.quantization.map(|q| QuantizationMeta {
        bits: q.bits.unwrap_or(16),
        group_size: q.group_size.unwrap_or(64),
        mode: match q.mode.as_deref() {
            Some("affine") => QuantizationMode::Affine,
            _ => QuantizationMode::None,
        },
        overrides: HashMap::new(),
    });

    // For models with a nested text_config (e.g. Gemma4 Unified), the
    // conversion process may not have written an explicit quantization
    // section into config.json.  Detect this case by checking whether
    // the top-level model_type contains known unified/conversion patterns
    // and default to 8-bit block quantization if no explicit metadata.
    let quant = explicit_quant.or_else(|| {
        if raw.text_config.is_some() {
            let mt = raw.model_type.as_deref().unwrap_or("");
            if mt.contains("unified") || mt.starts_with("gemma4") {
                Some(QuantizationMeta {
                    bits: 8,
                    group_size: 64,
                    mode: QuantizationMode::Affine,
                    overrides: HashMap::new(),
                })
            } else {
                None
            }
        } else {
            None
        }
    });

    let manifest = ModelManifest {
        config_path: config_path.into(),
        config_hash,
        model_type: raw.model_type.unwrap_or_default(),
        has_text_config: true, // we already checked text_config exists
        has_vision_config: raw.vision_config.is_some(),
        has_audio_config: raw.audio_config.is_some(),
        has_quantization_metadata: has_explicit_quant,
        quantization_bits: q_bits,
        quantization_group_size: q_group_size,
        quantization_mode: quant.as_ref().map(|q| format!("{:?}", q.mode)),
        vision_config: raw.vision_config.clone(),
        audio_config: raw.audio_config.clone(),
        safetensors_shards: Vec::new(),
    };

    Ok((arch, quant, manifest))
}

// ── Compilation Planning ───────────────────────────────────────────────────

/// Disposition of a tensor in the compiled image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorDisposition {
    /// No physical payload; another tensor is the canonical storage.
    AliasOnly { canonical_tensor_id: u32 },
    /// Bytes copied unchanged into destination segment.
    RelocateAndAlign,
    /// Source bytes can be directly referenced (external-source profile).
    PreserveInPlace,
    /// Small metadata tensor that should be transformed on CPU.
    CpuTransform { recipe: String },
    /// Large data-parallel tensor that should be transformed on GPU.
    GpuTransform { recipe: String },
    /// Tensor participates in Core ML backend island.
    CoreMlLoweringInput,
    /// Not emitted (e.g., unused multimodal wrapper in text-only profile).
    DiscardWithReason { reason: String },
}

/// A single tensor's identity and placement in the compiled image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedTensor {
    pub id: u32,
    pub name: String,
    pub disposition: TensorDisposition,
    pub source_shard: String,
    pub source_offset: u64,
    pub source_byte_length: u64,
    pub destination_segment: String,
    pub destination_offset: u64,
    pub destination_byte_length: u64,
    pub logical_dtype: String,
    pub logical_shape: Vec<u32>,
}

/// A planned binary segment containing tensors in execution order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedSegment {
    pub id: String,
    pub filename: String,
    pub byte_size: u64,
    pub kind: String,
    pub tensor_count: usize,
}

/// A complete, validated, immutable compilation plan.
/// Produced by the planning phase before any payload emission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationPlan {
    pub model_identity: String,
    pub source_config_hash: String,
    pub source_shard_hashes: Vec<String>,
    pub tensor_table: Vec<PlannedTensor>,
    pub segments: Vec<PlannedSegment>,
    pub total_source_bytes: u64,
    pub total_image_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_layer(index: u32) -> LayerPlan {
        LayerPlan {
            layer_index: index,
            attention_kind: "sliding_attention".into(),
            segment_id: format!("layer_{}", index),
            hidden_size: 64,
            n_heads: 4,
            n_kv_heads: 1,
            head_dim: 16,
            global_head_dim: None,
            n_global_kv_heads: None,
            sliding_window: 4096,
            rope_theta: 10000.0,
            partial_rotary_factor: None,
            attention_k_eq_v: false,
            q_norm_enabled: false,
            k_norm_enabled: false,
            q_proj_tensor_id: 1,
            k_proj_tensor_id: 2,
            v_proj_tensor_id: 3,
            o_proj_tensor_id: 4,
            q_norm_tensor_id: None,
            k_norm_tensor_id: None,
            gate_proj_tensor_id: 5,
            up_proj_tensor_id: 6,
            down_proj_tensor_id: 7,
            input_layernorm_tensor_id: 8,
            post_attention_layernorm_tensor_id: 9,
            pre_ffw_layernorm_tensor_id: None,
            post_ffw_layernorm_tensor_id: None,
            layer_scalar_ids: Vec::new(),
            quantization_ids: Vec::new(),
            route: Default::default(),
            fused_operations: Default::default(),
        }
    }

    fn base_plan() -> ModelExecutionPlan {
        ModelExecutionPlan {
            prologue: ProloguePlan {
                segment_id: "persistent".into(),
                embedding_tensor_id: 10,
                embedding_name: "model.embed_tokens.weight".into(),
                embedding_shape: vec![64, 64],
                embedding_dtype: "U8".into(),
            },
            layers: vec![valid_layer(0)],
            epilogue: EpiloguePlan {
                segment_id: "persistent".into(),
                final_norm_tensor_id: 11,
                final_norm_name: "model.norm.weight".into(),
                output_projection_tensor_id: None,
                output_projection_name: None,
                final_logit_softcapping: None,
                vocab_size: 64,
            },
            hidden_size: 64,
            vocab_size: 64,
            sliding_window: 4096,
            final_logit_softcapping: None,
            tie_word_embeddings: true,
            rms_norm_eps: 1e-6,
            fused_ane_islands: vec![],
            speculative_config: None,
            generation_regime: Default::default(),
            diffusion_config: Default::default(),
            diffusion_execution_plan: Default::default(),
            kv_cache_mode: Default::default(),
        }
    }

    #[test]
    fn validate_rejects_malformed_plans() {
        // 1. Zero layers
        {
            let mut plan = base_plan();
            plan.layers.clear();
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter()
                    .any(|e| e.contains("execution plan has zero layers")),
                "expected zero-layers error, got: {:?}",
                errs
            );
        }

        // 2. Layer index mismatch (layer at index 1 has layer_index=0)
        {
            let mut plan = base_plan();
            let mut l1 = valid_layer(1);
            l1.layer_index = 0;
            plan.layers.push(l1);
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter().any(|e| e.contains("layer 1 has index 0")),
                "expected index mismatch error, got: {:?}",
                errs
            );
        }

        // 3. Layer hidden_size != model hidden_size
        {
            let mut plan = base_plan();
            plan.layers[0].hidden_size = 128;
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter()
                    .any(|e| e.contains("hidden_size") && e.contains("128") && e.contains("64")),
                "expected hidden_size mismatch error, got: {:?}",
                errs
            );
        }

        // 4. q_proj_tensor_id = 0
        {
            let mut plan = base_plan();
            plan.layers[0].q_proj_tensor_id = 0;
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter().any(|e| e.contains("zero q_proj_tensor_id")),
                "expected zero q_proj_tensor_id error, got: {:?}",
                errs
            );
        }

        // 5. full_attention layer missing global_head_dim
        {
            let mut plan = base_plan();
            plan.layers[0].attention_kind = "full_attention".into();
            plan.layers[0].global_head_dim = None;
            // full_attention branch checks global_head_dim, not v_proj
            plan.layers[0].v_proj_tensor_id = 99;
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter().any(|e| e.contains("missing global_head_dim")),
                "expected missing global_head_dim error, got: {:?}",
                errs
            );
        }

        // 6. Unknown attention_kind
        {
            let mut plan = base_plan();
            plan.layers[0].attention_kind = "bogus".into();
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter()
                    .any(|e| e.contains("unknown attention_kind: bogus")),
                "expected unknown attention_kind error, got: {:?}",
                errs
            );
        }

        // 7. Prologue with zero embedding_tensor_id
        {
            let mut plan = base_plan();
            plan.prologue.embedding_tensor_id = 0;
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter().any(|e| e.contains("zero embedding_tensor_id")),
                "expected zero embedding_tensor_id error, got: {:?}",
                errs
            );
        }

        // 8. Epilogue with zero final_norm_tensor_id
        {
            let mut plan = base_plan();
            plan.epilogue.final_norm_tensor_id = 0;
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter().any(|e| e.contains("zero final_norm_tensor_id")),
                "expected zero final_norm_tensor_id error, got: {:?}",
                errs
            );
        }
    }
}

/// Unified server configuration loaded from config.toml, environment
/// variables, and CLI arguments (in ascending priority order).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub server: ServerConfigSection,
    pub model: ModelConfigSection,
    pub cache: CacheConfigSection,
    pub speculation: SpecConfigSection,
    pub cluster: ClusterConfigSection,
}

/// Server networking and runtime settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfigSection {
    pub port: u16,
    pub host: String,
    pub max_concurrent: u32,
    pub rate_limit_per_min: u32,
    pub rate_limit_tokens_per_sec: f64,
    pub rate_limit_burst: u64,
    pub log_level: String,
    pub runtime_mode: String,
}

/// Model loading and download policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfigSection {
    pub model_path: Option<String>,
    pub auto_download: bool,
    pub max_model_cache_gb: f64,
}

/// KV cache topology and compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfigSection {
    pub kv_cache_tiers: u32,
    pub compression_ratio: f64,
    pub evolkv_enabled: bool,
}

/// Speculative decoding parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpecConfigSection {
    pub draft_count: u32,
    pub draft_length: u32,
    pub spechub_enabled: bool,
}

/// EXO cluster membership and autoscaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClusterConfigSection {
    pub exo_enabled: bool,
    pub exo_port: u16,
    pub autoscale_min: u32,
    pub autoscale_max: u32,
}

impl Default for ServerConfigSection {
    fn default() -> Self {
        Self {
            port: 11434,
            host: "0.0.0.0".into(),
            max_concurrent: 64,
            rate_limit_per_min: 60,
            rate_limit_tokens_per_sec: 100.0,
            rate_limit_burst: 1000,
            log_level: "info".into(),
            runtime_mode: "safe".into(),
        }
    }
}

impl Default for ModelConfigSection {
    fn default() -> Self {
        Self {
            model_path: None,
            auto_download: false,
            max_model_cache_gb: 16.0,
        }
    }
}

impl Default for CacheConfigSection {
    fn default() -> Self {
        Self {
            kv_cache_tiers: 3,
            compression_ratio: 0.5,
            evolkv_enabled: true,
        }
    }
}

impl Default for SpecConfigSection {
    fn default() -> Self {
        Self {
            draft_count: 4,
            draft_length: 16,
            spechub_enabled: true,
        }
    }
}

impl Default for ClusterConfigSection {
    fn default() -> Self {
        Self {
            exo_enabled: false,
            exo_port: 52415,
            autoscale_min: 1,
            autoscale_max: 8,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerConfigSection::default(),
            model: ModelConfigSection::default(),
            cache: CacheConfigSection::default(),
            speculation: SpecConfigSection::default(),
            cluster: ClusterConfigSection::default(),
        }
    }
}

impl ServerConfig {
    /// Load from config file, then environment variables.
    /// Config file path: $HOME/.tribunus/config.toml
    /// (override with TRIBUNUS_CONFIG_PATH env var).
    pub fn load() -> Self {
        let mut config = Self::default();
        let config_path = std::env::var("TRIBUNUS_CONFIG_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            format!("{}/.tribunus/config.toml", home)
        });
        if let Ok(file_config) = Self::load_config_toml(&config_path) {
            config.merge(file_config);
        }
        config.load_env_overrides();
        config
    }

    /// Parse a TOML config file into a ServerConfig.
    pub fn load_config_toml(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Cannot read config file '{}': {}", path, e))?;
        toml::from_str(&content).map_err(|e| format!("Invalid config file '{}': {}", path, e))
    }

    /// Override fields from environment variables.
    pub fn load_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("TRIBUNUS_PORT") {
            if let Ok(n) = v.parse::<u16>() {
                self.server.port = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_HOST") {
            self.server.host = v;
        }
        if let Ok(v) = std::env::var("TRIBUNUS_MAX_CONCURRENT") {
            if let Ok(n) = v.parse::<u32>() {
                self.server.max_concurrent = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_RATE_LIMIT") {
            if let Ok(n) = v.parse::<u32>() {
                self.server.rate_limit_per_min = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_RATE_LIMIT_TOKENS_PER_SEC") {
            if let Ok(f) = v.parse::<f64>() {
                self.server.rate_limit_tokens_per_sec = f;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_RATE_LIMIT_BURST") {
            if let Ok(n) = v.parse::<u64>() {
                self.server.rate_limit_burst = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_LOG_LEVEL") {
            self.server.log_level = v;
        }
        if let Ok(v) = std::env::var("TRIBUNUS_RUNTIME_MODE") {
            self.server.runtime_mode = v.to_lowercase();
        }
        if let Ok(v) = std::env::var("TRIBUNUS_MODEL_PATH") {
            self.model.model_path = Some(v);
        }
        if let Ok(v) = std::env::var("TRIBUNUS_AUTO_DOWNLOAD") {
            self.model.auto_download = v.eq_ignore_ascii_case("true") || v == "1";
        }
        if let Ok(v) = std::env::var("TRIBUNUS_MAX_MODEL_CACHE_GB") {
            if let Ok(f) = v.parse::<f64>() {
                self.model.max_model_cache_gb = f;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_KV_CACHE_TIERS") {
            if let Ok(n) = v.parse::<u32>() {
                if n >= 2 && n <= 4 {
                    self.cache.kv_cache_tiers = n;
                }
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_COMPRESSION_RATIO") {
            if let Ok(f) = v.parse::<f64>() {
                self.cache.compression_ratio = f;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_EVOLKV_ENABLED") {
            self.cache.evolkv_enabled = v.eq_ignore_ascii_case("true") || v == "1";
        }
        if let Ok(v) = std::env::var("TRIBUNUS_DRAFT_COUNT") {
            if let Ok(n) = v.parse::<u32>() {
                self.speculation.draft_count = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_DRAFT_LENGTH") {
            if let Ok(n) = v.parse::<u32>() {
                self.speculation.draft_length = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_SPECHUB_ENABLED") {
            self.speculation.spechub_enabled = v.eq_ignore_ascii_case("true") || v == "1";
        }
        if let Ok(v) = std::env::var("TRIBUNUS_EXO_ENABLED") {
            self.cluster.exo_enabled = v.eq_ignore_ascii_case("true") || v == "1";
        }
        if let Ok(v) = std::env::var("TRIBUNUS_EXO_PORT") {
            if let Ok(n) = v.parse::<u16>() {
                self.cluster.exo_port = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_AUTOSCALE_MIN") {
            if let Ok(n) = v.parse::<u32>() {
                self.cluster.autoscale_min = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_AUTOSCALE_MAX") {
            if let Ok(n) = v.parse::<u32>() {
                self.cluster.autoscale_max = n;
            }
        }
    }

    /// Override fields from CLI arguments.
    /// Must be called after load() so CLI args take highest priority.
    pub fn apply_cli_args(&mut self, args: &[String]) {
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--port" => {
                    i += 1;
                    if i < args.len() {
                        if let Ok(n) = args[i].parse::<u16>() {
                            self.server.port = n;
                        }
                    }
                }
                "--host" => {
                    i += 1;
                    if i < args.len() {
                        self.server.host = args[i].clone();
                    }
                }
                "--model" | "--model-path" => {
                    i += 1;
                    if i < args.len() {
                        self.model.model_path = Some(args[i].clone());
                    }
                }
                "--exo" => {
                    self.cluster.exo_enabled = true;
                }
                "--exo-port" => {
                    i += 1;
                    if i < args.len() {
                        if let Ok(n) = args[i].parse::<u16>() {
                            self.cluster.exo_port = n;
                        }
                    }
                }
                "--runtime-mode" => {
                    i += 1;
                    if i < args.len() {
                        self.server.runtime_mode = args[i].to_lowercase();
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    /// Merge another config's non-default fields into self.
    fn merge(&mut self, other: ServerConfig) {
        self.server.port = other.server.port;
        self.server.host = other.server.host;
        self.server.max_concurrent = other.server.max_concurrent;
        self.server.rate_limit_per_min = other.server.rate_limit_per_min;
        self.server.log_level = other.server.log_level;
        self.server.runtime_mode = other.server.runtime_mode;

        if other.model.model_path.is_some() {
            self.model.model_path = other.model.model_path;
        }
        self.model.auto_download = other.model.auto_download;
        self.model.max_model_cache_gb = other.model.max_model_cache_gb;

        self.cache.kv_cache_tiers = other.cache.kv_cache_tiers;
        self.cache.compression_ratio = other.cache.compression_ratio;
        self.cache.evolkv_enabled = other.cache.evolkv_enabled;

        self.speculation.draft_count = other.speculation.draft_count;
        self.speculation.draft_length = other.speculation.draft_length;
        self.speculation.spechub_enabled = other.speculation.spechub_enabled;

        self.cluster.exo_enabled = other.cluster.exo_enabled;
        self.cluster.exo_port = other.cluster.exo_port;
        self.cluster.autoscale_min = other.cluster.autoscale_min;
        self.cluster.autoscale_max = other.cluster.autoscale_max;
    }
}

/// Generate per-backend fusion plans.
pub fn generate_backend_plans(
    plan: &ModelExecutionPlan,
    backends: &[&str],
) -> HashMap<String, std::collections::HashMap<String, Vec<FusedOperation>>> {
    let mut result = std::collections::HashMap::new();
    for backend in backends {
        let layer_ops: std::collections::HashMap<String, Vec<FusedOperation>> = plan
            .layers
            .iter()
            .map(|layer| {
                (
                    layer.layer_index.to_string(),
                    layer.fused_operations.clone(),
                )
            })
            .collect();
        result.insert(backend.to_string(), layer_ops);
    }
    result
}
