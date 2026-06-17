//! oQ dynamic quantization — load-time mixed-precision quantization.
//!
//! Reference: `ref/omlx/oq.py`, design: `docs/omlx-oq-quantization.md`
//!
//! Combines GGUF K-quant layer strategy, Unsloth selective non-quantization,
//! and BnB MSE-optimal clipping. Levels: oQ2 through oQ8 with fractional
//! expert down_proj boost.

/// oQ quantization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OqLevel {
    Oq2,
    Oq2_5,
    Oq2_7,
    Oq3,
    Oq3_5,
    Oq4,
    Oq5,
    Oq6,
    Oq8,
}

impl OqLevel {
    /// Base bits for this level (2, 3, 4, 5, 6, or 8)
    pub fn base_bits(&self) -> u32 {
        match self {
            Self::Oq2 | Self::Oq2_5 | Self::Oq2_7 => 2,
            Self::Oq3 | Self::Oq3_5 => 3,
            Self::Oq4 => 4,
            Self::Oq5 => 5,
            Self::Oq6 => 6,
            Self::Oq8 => 8,
        }
    }

    /// Whether this level applies expert down_proj boost
    pub fn expert_down_boost(&self) -> bool {
        matches!(self, Self::Oq2_5 | Self::Oq2_7 | Self::Oq3_5)
    }

    /// Protection strategy label (maps to K-quant strategy)
    pub fn protection_label(&self) -> &'static str {
        match self.base_bits() {
            2 => "KQUANT_2", // Most aggressive quant
            3 => "KQUANT_3",
            4 => "KQUANT_4",
            5 => "KQUANT_5",
            6 => "KQUANT_6",
            _ => "NO_QUANT", // oQ8 keeps full precision
        }
    }
}

/// Quantization dtype target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OqDtype {
    Bf16,
    F16,
}

/// Configuration for oQ quantization
#[derive(Debug, Clone)]
pub struct OqConfig {
    pub level: OqLevel,
    pub group_size: usize,
    pub dtype: OqDtype,
    /// Optional path to a proxy/sensitivity model for adaptive quantization
    pub sensitivity_model: Option<String>,
}

impl Default for OqConfig {
    fn default() -> Self {
        Self {
            level: OqLevel::Oq4,
            group_size: 64,
            dtype: OqDtype::Bf16,
            sensitivity_model: None,
        }
    }
}

/// A quantized weight tensor
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// The quantized data (packed bits)
    pub data: Vec<u8>,
    /// Per-group scale factors
    pub scale: Vec<f32>,
    /// Per-group zero points (None for symmetric quantization)
    pub zero_point: Option<Vec<f32>>,
    /// Number of elements per quantization group
    pub group_size: usize,
    /// Original tensor shape before quantization
    pub orig_shape: Vec<usize>,
}

/// Errors during oQ quantization
#[derive(Debug, thiserror::Error)]
pub enum OqError {
    #[error("Unsupported quantization level: {0:?}")]
    UnsupportedLevel(OqLevel),
    #[error("Tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("Sensitivity model not found: {0}")]
    SensitivityModelNotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Apply oQ quantization to model weights.
///
/// Follows the Python reference in ref/omlx/oq.py:
/// 1. Build layer quant plan from level + K-quant strategy
/// 2. Apply MSE-optimal clipping per group
/// 3. Quantize weights, store scale + zero-point
/// 4. Apply expert down_proj boost for fractional levels
pub fn apply_oq(_weights: &[u8], _config: &OqConfig) -> Result<Vec<QuantizedTensor>, OqError> {
    // TODO: Implement per oq.py reference
    // - Build layer quant plan
    // - Compute MSE-optimal clip thresholds
    // - Apply K-quant style quantization per group
    // - Return quantized tensors
    todo!("oQ quantization not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_bits() {
        assert_eq!(OqLevel::Oq4.base_bits(), 4);
        assert_eq!(OqLevel::Oq8.base_bits(), 8);
        assert_eq!(OqLevel::Oq2_5.base_bits(), 2);
    }

    #[test]
    fn test_expert_down_boost() {
        assert!(OqLevel::Oq2_5.expert_down_boost());
        assert!(!OqLevel::Oq4.expert_down_boost());
    }
}
