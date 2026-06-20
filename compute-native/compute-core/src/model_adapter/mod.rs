//! Model-family adapter layer: normalises diverse model sources (safetensors +
//! HuggingFace config.json) into a canonical representation that the ComputeImage
//! compiler can lower to backend-specific artifacts.
//!
//! # Architecture
//!
//! ```text
//! SourceModel (raw config.json + safetensor shards)
//!   → AdapterRegistry::select(config, tensor_names) → ModelFamilyAdapter
//!   → adapter.normalize(source) → CanonicalModel
//!   → compile.rs bridge → LoadedSource
//! ```
//!
//! # Adding a new family
//!
//! 1. Write a struct implementing `ModelFamilyAdapter`
//! 2. Register it in `AdapterRegistry::new()`
//! 3. Add a synthetic fixture in `fixtures.rs`
//! 4. Add a conformance case in `conformance.rs`

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ═══════════════════════════════════════════════════════════════════════════
// Canonical roles
// ═══════════════════════════════════════════════════════════════════════════

/// Canonical tensor roles that every adapter maps source names to.
///
/// Layers are indexed from 0. Every layer index that the architecture declares
/// MUST have a complete set of required roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CanonicalRole {
    Embedding,
    FinalNorm,
    LmHead,
    AttnNorm(u32),
    Q(u32),
    K(u32),
    V(u32),
    O(u32),
    MlpNorm(u32),
    Gate(u32),
    Up(u32),
    Down(u32),
    QNorm(u32),
    KNorm(u32),
}

impl std::fmt::Display for CanonicalRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanonicalRole::Embedding => write!(f, "Embedding"),
            CanonicalRole::FinalNorm => write!(f, "FinalNorm"),
            CanonicalRole::LmHead => write!(f, "LmHead"),
            CanonicalRole::AttnNorm(i) => write!(f, "AttnNorm({})", i),
            CanonicalRole::Q(i) => write!(f, "Q({})", i),
            CanonicalRole::K(i) => write!(f, "K({})", i),
            CanonicalRole::V(i) => write!(f, "V({})", i),
            CanonicalRole::O(i) => write!(f, "O({})", i),
            CanonicalRole::MlpNorm(i) => write!(f, "MlpNorm({})", i),
            CanonicalRole::Gate(i) => write!(f, "Gate({})", i),
            CanonicalRole::Up(i) => write!(f, "Up({})", i),
            CanonicalRole::Down(i) => write!(f, "Down({})", i),
            CanonicalRole::QNorm(i) => write!(f, "QNorm({})", i),
            CanonicalRole::KNorm(i) => write!(f, "KNorm({})", i),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Data types
// ═══════════════════════════════════════════════════════════════════════════

/// A single canonical tensor with raw bytes and metadata.
#[derive(Clone, Debug)]
pub struct TensorData {
    pub dtype: String,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
}

/// Raw source model before normalisation.
#[derive(Clone, Debug)]
pub struct SourceModel {
    pub config: Value,
    pub config_path: PathBuf,
    pub model_type: String,
    pub tensor_names: Vec<String>,
    /// Raw tensor data keyed by source tensor name.
    /// Each entry: (dtype, shape, raw_bytes).
    pub tensors: HashMap<String, (String, Vec<u32>, Vec<u8>)>,
}

/// Fully normalised model consumed by the compiler pipeline.
#[derive(Clone, Debug)]
pub struct CanonicalModel {
    /// Architecture parameters extracted and validated from config.
    pub architecture: super::config::TextArchitecture,
    /// Canonical role → actual tensor data.
    pub tensors: HashMap<CanonicalRole, TensorData>,
}

/// Human-readable normalisation failure.
#[derive(Clone, Debug)]
pub struct NormalizationReport {
    pub family: String,
    pub errors: Vec<String>,
    pub missing_roles: Vec<CanonicalRole>,
    pub shape_mismatches: Vec<String>,
}

impl std::fmt::Display for NormalizationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "normalization failed for family '{}':", self.family)?;
        for e in &self.errors {
            writeln!(f, "  {}", e)?;
        }
        if !self.missing_roles.is_empty() {
            let roles: Vec<String> = self.missing_roles.iter().map(|r| r.to_string()).collect();
            writeln!(f, "  missing roles: {}", roles.join(", "))?;
        }
        if !self.shape_mismatches.is_empty() {
            for m in &self.shape_mismatches {
                writeln!(f, "  shape mismatch: {}", m)?;
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Adapter trait
// ═══════════════════════════════════════════════════════════════════════════

/// Adapter that normalises one model family's source into canonical form.
pub trait ModelFamilyAdapter: Send + Sync {
    /// Short stable identifier (e.g. "qwen2", "llama").
    fn family_name(&self) -> &'static str;

    /// Config `model_type` values this adapter claims.
    fn claimed_config_types(&self) -> &'static [&'static str];

    /// Evidence-based check: does this adapter match the given source?
    /// Inspects actual tensor_names for structure; not just config model_type.
    fn detect(&self, config: &Value, tensor_names: &[String]) -> bool;

    /// Normalise the source into canonical form.
    /// Returns Err with a `NormalizationReport` when roles are missing,
    /// shapes mismatch, or architecture parameters are invalid.
    fn normalize(&self, source: &SourceModel) -> Result<CanonicalModel, NormalizationReport>;
}

// ═══════════════════════════════════════════════════════════════════════════
// Adapter registry
// ═══════════════════════════════════════════════════════════════════════════

/// Registry of all available model-family adapters.
pub struct AdapterRegistry {
    adapters: Vec<Box<dyn ModelFamilyAdapter>>,
}

impl AdapterRegistry {
    /// Create registry with all built-in adapters.
    pub fn new() -> Self {
        Self {
            adapters: vec![
                Box::new(super::model_adapter::qwen2::Qwen2Adapter),
                Box::new(super::model_adapter::llama::LlamaAdapter),
                Box::new(super::model_adapter::mistral::MistralAdapter),
                Box::new(super::model_adapter::gemma::GemmaAdapter),
                Box::new(super::model_adapter::phi::PhiAdapter),
            ],
        }
    }

    /// Select the best matching adapter using evidence-based matching.
    ///
    /// Selection order:
    /// 1. Filter by claimed `model_type` from config.
    /// 2. For candidates, call `detect()` for tensor-name validation.
    /// 3. If multiple match, pick the first (priority: Qwen2, Llama, Mistral, Gemma, Phi).
    /// 4. If none match, return error listing all candidates and why they failed.
    pub fn select<'a>(
        &'a self,
        config: &Value,
        tensor_names: &[String],
    ) -> Result<&'a dyn ModelFamilyAdapter, String> {
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let mut matches: Vec<&dyn ModelFamilyAdapter> = Vec::new();
        let mut candidates: Vec<String> = Vec::new();

        for adapter in &self.adapters {
            let claimed = adapter.claimed_config_types();
            let matches_type = claimed.iter().any(|t| *t == model_type);

            candidates.push(format!(
                "{} (type_match={}, detect={})",
                adapter.family_name(),
                matches_type,
                adapter.detect(config, tensor_names),
            ));

            if matches_type && adapter.detect(config, tensor_names) {
                matches.push(adapter.as_ref());
            }
        }

        if matches.is_empty() {
            return Err(format!(
                "unsupported model_type '{}': no adapter matched. Candidates:\n  {}",
                model_type,
                candidates.join("\n  ")
            ));
        }

        Ok(matches[0])
    }

    /// Register a custom adapter (for extensibility / testing).
    pub fn register(&mut self, adapter: Box<dyn ModelFamilyAdapter>) {
        self.adapters.push(adapter);
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Adapter submodules
// ═══════════════════════════════════════════════════════════════════════════

pub mod qwen2;
pub mod llama;
pub mod mistral;
pub mod gemma;
pub mod phi;
pub mod fixtures;

#[cfg(test)]
mod tests;
