//! Compile-time quantization transforms for ComputeImage weights.
//! NF4 (NormalFloat 4-bit) and 8-bit affine quantization.

pub(crate) use super::compile::{
    apply_quantize_to_loaded, quantize_nf4_value, quantize_nf4_group,
    quantize_af8_group, apply_nf4_quantize, apply_af8_quantize,
    half_to_f32, NF4_CODEBOOK,
};
