//! Compile-time quantization transforms for ComputeImage weights.
//! NF4 (NormalFloat 4-bit) and 8-bit affine quantization.

pub(crate) use super::compile::{
    apply_af8_quantize, apply_nf4_quantize, apply_quantize_to_loaded, half_to_f32,
    quantize_af8_group, quantize_nf4_group, quantize_nf4_value, NF4_CODEBOOK,
};
