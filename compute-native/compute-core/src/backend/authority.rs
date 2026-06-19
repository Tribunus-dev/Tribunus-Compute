//! Shared authority dequantize logic for quantized int4 matmul.
//!
//! Extracts 4-bit nibbles from packed U32 weight words and dequantizes
//! them to f32 using per-group scale and bias.  This algorithm is shared
//! by both the MLX authority path (dequantize + matmul as a fallback for
//! fused-kernel crashes) and the Candle CPU backend.

/// Dequantize packed U32 int4 weights to f32 using correct nibble extraction.
///
/// # Parameters
/// - `w_u32` — packed weight words (one row per `n_out` rows × `packed_cols`
///   columns; each word holds 8 nibbles).
/// - `scales` — `[n_out * n_groups]` f32 scale factors.
/// - `biases` — `[n_out * n_groups]` f32 biases.
/// - `n_out` — number of output rows (logical N dimension).
/// - `k` — number of weight columns (logical K dimension).
/// - `n_groups` — number of quantization groups along K.
/// - `packed_cols` — physical packed columns (K / 8 rounded up).
/// - `group_size` — number of elements per quantization group.
///
/// # Returns
/// A `[n_out × k]` f32 buffer of dequantized weights.
pub fn dequantize_int4_weights(
    w_u32: &[u32],
    scales: &[f32],
    biases: &[f32],
    n_out: usize,
    k: usize,
    n_groups: usize,
    packed_cols: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut w_f32 = vec![0.0f32; n_out * k];
    for row in 0..n_out {
        for g in 0..n_groups {
            let scale = scales[row * n_groups + g];
            let bias = biases[row * n_groups + g];
            let start = g * group_size;
            let end = (start + group_size).min(k);
            for elem_idx in start..end {
                let word_idx = row * packed_cols + elem_idx / 8;
                let lane = elem_idx % 8;
                let qval = (w_u32[word_idx] >> (lane * 4)) & 0xF;
                w_f32[row * k + elem_idx] = (qval as f32) * scale + bias;
            }
        }
    }
    w_f32
}
