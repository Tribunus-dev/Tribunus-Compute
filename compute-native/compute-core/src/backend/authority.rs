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
// ── GGML/Q4_0 quantization format ───────────────────────────────────────────

/// Convert IEEE 754 half-precision (fp16) bits to f32.
fn f16_from_bits(bits: u16) -> f32 {
    let sign = if (bits >> 15) & 1 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x03FF;
    if exp == 0 {
        // Denormalized: (-1)^s * 2^-14 * (mant / 2^10) = (-1)^s * mant * 2^-24
        sign * (mant as f32) * (2.0f32).powi(-24)
    } else if exp == 31 {
        // Infinity or NaN — saturate to large value
        sign * f32::MAX
    } else {
        // Normalized: (-1)^s * 2^(e-15) * (1 + mant / 2^10)
        sign * (2.0f32).powi(exp as i32 - 15) * (1.0 + (mant as f32) * (2.0f32).powi(-10))
    }
}

/// GGML Q4_0 block quantization format.
///
/// Block size 32, 4 bits per value, scale is fp16.
/// Each block is 18 bytes: 2 bytes fp16 scale + 16 bytes int4 nibbles = 18 bytes for 32 values.
/// Effective bits per weight: 4.5 bpw.
#[derive(Clone, Debug)]
pub struct BlockQ4_0 {
    pub scale: f32,   // dequantized from fp16
    pub qs: [u8; 16], // 32 int4 nibbles packed little-endian
}

impl BlockQ4_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const BYTES_PER_BLOCK: usize = 18;

    /// Dequantize this block to a `[f32; 32]` output array.
    pub fn dequantize(&self, out: &mut [f32; 32]) {
        for (i, nibble) in self.qs.iter().enumerate() {
            // Each byte holds two signed int4 values: low nibble first.
            let lo = (nibble & 0x0F) as i8 - 8;
            let hi = ((nibble >> 4) & 0x0F) as i8 - 8;
            out[i * 2] = (lo as f32) * self.scale;
            out[i * 2 + 1] = (hi as f32) * self.scale;
        }
    }

    /// Read a Q4_0 block from packed GGML bytes (18 bytes).
    pub fn from_bytes(data: &[u8; 18]) -> Self {
        let scale = f16_from_bits(u16::from_le_bytes([data[0], data[1]]));
        let mut qs = [0u8; 16];
        qs.copy_from_slice(&data[2..18]);
        BlockQ4_0 { scale, qs }
    }
}

/// Dequantize a full weight matrix stored in GGML Q4_0 block format.
///
/// # Parameters
/// - `packed` — raw packed bytes from the GGUF tensor data section.
/// - `n_rows` — number of logical rows (output dimension).
/// - `n_cols` — number of logical columns (input dimension).
///
/// # Returns
/// A flat `[n_rows × n_cols]` f32 buffer in row-major order.
pub fn dequantize_ggml_q4_0(packed: &[u8], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let block_count = (n_cols + BlockQ4_0::BLOCK_SIZE - 1) / BlockQ4_0::BLOCK_SIZE;
    let mut out = vec![0.0f32; n_rows * n_cols];
    for row in 0..n_rows {
        for block_idx in 0..block_count {
            let byte_offset = row * block_count * BlockQ4_0::BYTES_PER_BLOCK
                + block_idx * BlockQ4_0::BYTES_PER_BLOCK;
            let mut block_bytes = [0u8; 18];
            block_bytes.copy_from_slice(&packed[byte_offset..byte_offset + 18]);
            let block = BlockQ4_0::from_bytes(&block_bytes);
            let mut buf = [0.0f32; 32];
            block.dequantize(&mut buf);
            let elem_offset = row * n_cols + block_idx * 32;
            let copy_len = 32.min(n_cols - block_idx * 32);
            out[elem_offset..elem_offset + copy_len].copy_from_slice(&buf[..copy_len]);
        }
    }
    out
}

// ── GGUF header parsing ────────────────────────────────────────────────────

/// Minimal GGUF header parser for model weight discovery.
#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    /// Parse a GGUF header from raw bytes.
    ///
    /// Returns `None` if the magic is missing or the data slice is too short.
    pub fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 20 {
            return None;
        }
        let magic = &data[0..4];
        if magic != b"GGUF" {
            return None;
        }
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let tensor_count = if version >= 2 {
            u64::from_le_bytes(data[8..16].try_into().ok()?)
        } else {
            // v1 used uint32 at offset 8
            u32::from_le_bytes(data[8..12].try_into().ok()?) as u64
        };
        let meta_offset = if version >= 2 { 16 } else { 12 };
        let metadata_kv_count = if version >= 2 {
            u64::from_le_bytes(data[meta_offset..meta_offset + 8].try_into().ok()?)
        } else {
            // v1 used uint32
            u32::from_le_bytes(data[meta_offset..meta_offset + 4].try_into().ok()?) as u64
        };
        Some(GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_from_bits() {
        // 0x3C00 = 1.0 in fp16
        let one = f16_from_bits(0x3C00);
        assert!((one - 1.0).abs() < 1e-3, "expected 1.0, got {one}");

        // 0xBC00 = -1.0 in fp16
        let neg_one = f16_from_bits(0xBC00);
        assert!((neg_one + 1.0).abs() < 1e-3, "expected -1.0, got {neg_one}");

        // 0x0000 = 0.0
        let zero = f16_from_bits(0x0000);
        assert_eq!(zero, 0.0);

        // 0x3C80 = 1.0 + (128/1024) = 1.125 in fp16
        let one_pt_125 = f16_from_bits(0x3C80);
        assert!(
            (one_pt_125 - 1.125).abs() < 1e-3,
            "expected 1.125, got {one_pt_125}"
        );

        // Denormalized: smallest positive subnormal
        let tiny = f16_from_bits(0x0001);
        assert!(
            tiny > 0.0 && tiny < 1e-6,
            "expected tiny positive, got {tiny}"
        );
    }

    #[test]
    fn test_block_q4_0_dequantize() {
        // Scale = 1.0 (0x3C00 in fp16), all nibbles = 0x01 (low) / 0x10 (high) = value 1-8 = -7
        let mut block_bytes = [0u8; 18];
        block_bytes[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for i in 2..18 {
            block_bytes[i] = 0x11; // both nibbles = 1 => 1 - 8 = -7
        }
        let block = BlockQ4_0::from_bytes(&block_bytes);
        assert!((block.scale - 1.0).abs() < 1e-3);

        let mut out = [0.0f32; 32];
        block.dequantize(&mut out);
        for (i, &v) in out.iter().enumerate() {
            let expected = -7.0; // (1 - 8) * 1.0
            assert!(
                (v - expected).abs() < 1e-5,
                "out[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_block_q4_0_different_nibbles() {
        // Scale = 2.0 (0x4000 in fp16)
        let mut block_bytes = [0u8; 18];
        block_bytes[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
        // Each byte: low nibble = 0xF (= 15-8 = 7), high nibble = 0x0 (= 0-8 = -8)
        for i in 2..18 {
            block_bytes[i] = 0x0F;
        }
        let block = BlockQ4_0::from_bytes(&block_bytes);
        assert!((block.scale - 2.0).abs() < 1e-3);

        let mut out = [0.0f32; 32];
        block.dequantize(&mut out);
        for i in 0..16 {
            assert!(
                (out[i * 2] - 14.0).abs() < 1e-5,
                "out[{}] = {}, expected 14.0",
                i * 2,
                out[i * 2]
            );
            assert!(
                (out[i * 2 + 1] + 16.0).abs() < 1e-5,
                "out[{}] = {}, expected -16.0",
                i * 2 + 1,
                out[i * 2 + 1]
            );
        }
    }

    #[test]
    fn test_dequantize_ggml_q4_0_roundtrip() {
        // 2 rows, 64 cols = 4 blocks (each block = 32 cols)
        let n_rows = 2;
        let n_cols: usize = 64;
        let block_count = (n_cols + 31) / 32;

        // Build packed bytes: each block has scale=1.0 and known nibble pattern
        let total_blocks = n_rows * block_count;
        let packed_len = total_blocks * BlockQ4_0::BYTES_PER_BLOCK;
        let mut packed = vec![0u8; packed_len];

        for row in 0..n_rows {
            for b in 0..block_count {
                let offset =
                    row * block_count * BlockQ4_0::BYTES_PER_BLOCK + b * BlockQ4_0::BYTES_PER_BLOCK;
                // Scale = row+1 (in fp16)
                let scale_f16 = if row == 0 { 0x3C00u16 } else { 0x4000u16 }; // 1.0 or 2.0
                packed[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
                // Nibbles: low=nibble_idx+1, high=nibble_idx+2 (mod 16)
                for nib in 0..16 {
                    let lo = ((nib + 1) & 0x0F) as u8;
                    let hi = ((nib + 2) & 0x0F) as u8;
                    packed[offset + 2 + nib] = (hi << 4) | lo;
                }
            }
        }

        let result = dequantize_ggml_q4_0(&packed, n_rows, n_cols);
        assert_eq!(result.len(), n_rows * n_cols);

        // Spot-check first block, row 0
        let scale = 1.0;
        for nib in 0..16 {
            let lo = ((nib + 1) & 0x0F) as i8 - 8;
            let hi = ((nib + 2) & 0x0F) as i8 - 8;
            assert!(
                (result[nib * 2] - (lo as f32) * scale).abs() < 1e-5,
                "row=0 elem={} got {} expected {}",
                nib * 2,
                result[nib * 2],
                (lo as f32) * scale
            );
            assert!(
                (result[nib * 2 + 1] - (hi as f32) * scale).abs() < 1e-5,
                "row=0 elem={} got {} expected {}",
                nib * 2 + 1,
                result[nib * 2 + 1],
                (hi as f32) * scale
            );
        }
    }

    #[test]
    fn test_dequantize_ggml_q4_0_non_multiple() {
        // 1 row, 40 cols (not a multiple of 32)
        let n_rows = 1;
        let n_cols = 40;
        let block_count = (n_cols + 31) / 32; // 2

        let packed_len = n_rows * block_count * BlockQ4_0::BYTES_PER_BLOCK;
        let mut packed = vec![0u8; packed_len];
        // block 0: scale=1.0, all zeros
        packed[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // block 1: scale=0.5, all ones
        let offset_1 = BlockQ4_0::BYTES_PER_BLOCK;
        packed[offset_1..offset_1 + 2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5 in fp16
        for i in offset_1 + 2..offset_1 + 18 {
            packed[i] = 0x11;
        }

        let result = dequantize_ggml_q4_0(&packed, n_rows, n_cols);
        assert_eq!(result.len(), 40);

        // First 32: scale=1.0, nibbles=0 => -8.0
        for i in 0..32 {
            assert!(
                (result[i] + 8.0).abs() < 1e-5,
                "result[{i}] = {}, expected -8.0",
                result[i]
            );
        }
        // Remaining 8: scale=0.5, nibbles=1 => (1-8)*0.5 = -3.5
        for i in 32..40 {
            assert!(
                (result[i] + 3.5).abs() < 1e-5,
                "result[{i}] = {}, expected -3.5",
                result[i]
            );
        }
    }

    #[test]
    fn test_gguf_header_parse() {
        // Build a valid GGUF v3 header
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF"); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&7u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&42u64.to_le_bytes()); // metadata_kv_count

        let header = GgufHeader::parse(&buf).expect("should parse");
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 7);
        assert_eq!(header.metadata_kv_count, 42);
    }

    #[test]
    fn test_gguf_header_parse_bad_magic() {
        let mut buf = vec![0u8; 20];
        buf[0..4].copy_from_slice(b"NOTG");
        assert!(GgufHeader::parse(&buf).is_none());
    }

    #[test]
    fn test_gguf_header_parse_too_short() {
        let buf = [0u8; 4];
        assert!(GgufHeader::parse(&buf).is_none());
    }

    #[test]
    fn test_gguf_header_parse_v1() {
        // v1 used uint32 for counts
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF"); // magic
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&5u32.to_le_bytes()); // tensor_count (uint32 in v1)
        buf.extend_from_slice(&3u32.to_le_bytes()); // metadata_kv_count (uint32 in v1)

        let header = GgufHeader::parse(&buf).expect("should parse v1");
        assert_eq!(header.version, 1);
        assert_eq!(header.tensor_count, 5);
        assert_eq!(header.metadata_kv_count, 3);
    }
}
