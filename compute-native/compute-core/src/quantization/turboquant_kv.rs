//! TurboQuant KV cache quantization.
//!
//! Reference: `ref/omlx/turboquant_kv.py`, design: `docs/omlx-turboquant-kv.md`
//!
//! KV cache quantization with multiple strategies: polar, product, split,
//! and combinations thereof for extreme compression (2-4 bits per element).

/// KV cache quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantMode {
    /// Sign-preserving polar quantization
    Polar(u32),
    /// Product quantization (decompose into two lower-bit values)
    Prod(u32),
    /// Split quantization (separate by head dimension)
    Split(u32),
    /// Combined polar + product for extreme compression
    PolarProd(u32),
    /// MSE-optimal state selection per batch
    Mse { bits: u32, state_bits: u32 },
}

/// Per-slot TurboQuant KV cache state
#[derive(Debug, Clone)]
pub struct TurboQuantState {
    /// Quantized keys: one byte-aligned vector per mode
    pub keys: Vec<u8>,
    /// Quantized values
    pub values: Vec<u8>,
    /// Number of tokens stored in this slot
    pub num_tokens: usize,
    /// Quantization parameters derived from the mode
    pub bits: u32,
    pub head_dim: usize,
}

/// Batch-aware TurboQuant KV cache
///
/// Supports different cache states per request in the batch.
/// Reference: BatchTurboQuantKVCache in ref/omlx/turboquant_kv.py
#[allow(dead_code)]
pub struct TurboQuantKvCache {
    quant_mode: KvQuantMode,
    group_size: usize,
    state: Vec<TurboQuantState>,
}

/// Errors during TurboQuant KV operations
#[derive(Debug, thiserror::Error)]
pub enum TurboQuantError {
    #[error("Unsupported quant mode for given bit width")]
    UnsupportedMode,
    #[error("Shape mismatch in batch state")]
    ShapeMismatch,
    #[error("Slot index {0} out of bounds (capacity {1})")]
    SlotOutOfBounds(usize, usize),
    #[error("Empty slot {0}: nothing to dequantize")]
    EmptySlot(usize),
    #[error("Data length {data_len} not divisible by head_dim {head_dim}")]
    InvalidDataLength { data_len: usize, head_dim: usize },
}

// ---------------------------------------------------------------------------
// Bit-level helpers
// ---------------------------------------------------------------------------

fn bit_write(buf: &mut Vec<u8>, bit_offset: usize, num_bits: usize, value: u32) {
    for i in 0..num_bits {
        let byte_idx = (bit_offset + i) / 8;
        let bit_idx = (bit_offset + i) % 8;
        if byte_idx >= buf.len() {
            buf.push(0);
        }
        if (value >> i) & 1 == 1 {
            buf[byte_idx] |= 1 << bit_idx;
        }
    }
}

fn bit_read(buf: &[u8], bit_offset: usize, num_bits: usize) -> u32 {
    let mut val = 0u32;
    for i in 0..num_bits {
        let byte_idx = (bit_offset + i) / 8;
        let bit_idx = (bit_offset + i) % 8;
        if byte_idx < buf.len() && (buf[byte_idx] >> bit_idx) & 1 == 1 {
            val |= 1 << i;
        }
    }
    val
}

// ---------------------------------------------------------------------------
// Polar quantization with scale storage
// ---------------------------------------------------------------------------

/// Polar quantization that stores the scale factor in the packed buffer
/// (appended as an f32 after the packed bits).
fn quantize_polar_stored_scale(data: &[f32], bits: u32) -> Vec<u8> {
    let sign_bits = 1;
    let mag_bits = bits - sign_bits;
    let max_levels = (1 << mag_bits) - 1;
    let max_abs = data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let scale = if max_abs == 0.0 {
        1.0
    } else {
        max_abs / max_levels as f32
    };

    let packed_bits = data.len() * bits as usize;
    let buf_cap = (packed_bits + 7) / 8 + 4; // +4 for f32 scale
    let mut result = Vec::with_capacity(buf_cap);

    // Pack all values into bits
    let mut bit_pos = 0;
    for &x in data {
        let sign = if x < 0.0 { 1u32 } else { 0u32 };
        let mag = ((x.abs() / scale).round() as u32).min(max_levels);
        let encoded = (sign << mag_bits) | mag;
        bit_write(&mut result, bit_pos, bits as usize, encoded);
        bit_pos += bits as usize;
    }

    // Append scale as f32 at the end
    let scale_bytes = scale.to_le_bytes();
    for b in &scale_bytes {
        result.push(*b);
    }

    result
}

fn dequantize_polar_stored_scale(buf: &[u8], len: usize, bits: u32) -> Vec<f32> {
    let sign_bits = 1;
    let mag_bits = bits - sign_bits;
    let max_levels = (1 << mag_bits) - 1;

    // Read scale from the last 4 bytes
    let scale_pos = buf.len().saturating_sub(4);
    let scale_bytes: [u8; 4] = [
        *buf.get(scale_pos).unwrap_or(&0),
        *buf.get(scale_pos.wrapping_add(1)).unwrap_or(&0),
        *buf.get(scale_pos.wrapping_add(2)).unwrap_or(&0),
        *buf.get(scale_pos.wrapping_add(3)).unwrap_or(&0),
    ];
    let scale = f32::from_le_bytes(scale_bytes);

    let mut result = Vec::with_capacity(len);
    let mut bit_pos = 0;
    for _ in 0..len {
        let encoded = bit_read(buf, bit_pos, bits as usize);
        let sign = (encoded >> mag_bits) & 1;
        let mag = encoded & max_levels;
        let val = if sign == 0 {
            mag as f32 * scale
        } else {
            -(mag as f32 * scale)
        };
        result.push(val);
        bit_pos += bits as usize;
    }
    result
}

// ---------------------------------------------------------------------------
// Product quantization (uniform codebooks)
// ---------------------------------------------------------------------------

fn quantize_product(data: &[f32], bits: u32) -> Vec<u8> {
    let sub_bits_a = (bits + 1) / 2;
    let sub_bits_b = bits - sub_bits_a;
    let levels_a = 1usize << sub_bits_a;
    let levels_b = 1usize << sub_bits_b;

    let max_abs = data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let sqrt_max = max_abs.sqrt();

    let scale_a = if levels_a <= 1 {
        1.0
    } else {
        sqrt_max / (levels_a - 1) as f32
    };
    let scale_b = if levels_b <= 1 {
        1.0
    } else {
        sqrt_max / (levels_b - 1) as f32
    };

    // Pack per-value: 1 sign bit + idx_a (sub_bits_a) + idx_b (sub_bits_b)
    let bits_per_val = 1 + sub_bits_a + sub_bits_b;
    let packed_bits = data.len() * bits_per_val as usize;
    let mut packed = Vec::with_capacity((packed_bits + 7) / 8 + 32);
    let mut bit_pos = 0;

    for &x in data {
        let sign = if x < 0.0 { 1u32 } else { 0u32 };
        let ax = x.abs();
        let sqrt_ax = ax.sqrt();

        let idx_a = if scale_a == 0.0 {
            0u32
        } else {
            let raw = (sqrt_ax / scale_a).round() as u32;
            raw.min((levels_a - 1) as u32)
        };
        let cb_a = idx_a as f32 * scale_a;

        // idx_b such that cb_a * cb_b ≈ ax
        let idx_b = if scale_b == 0.0 || cb_a == 0.0 {
            0u32
        } else {
            let target = ax / cb_a;
            let raw = (target / scale_b).round() as u32;
            raw.min((levels_b - 1) as u32)
        };

        let val = (sign << (sub_bits_a + sub_bits_b)) | (idx_a << sub_bits_b) | idx_b;
        bit_write(&mut packed, bit_pos, bits_per_val as usize, val);
        bit_pos += bits_per_val as usize;
    }

    // Append metadata: [scale_a, scale_b, sub_bits_a, sub_bits_b, levels_a, levels_b] as f32/u32
    packed.extend_from_slice(&scale_a.to_le_bytes());
    packed.extend_from_slice(&scale_b.to_le_bytes());
    packed.extend_from_slice(&sub_bits_a.to_le_bytes());
    packed.extend_from_slice(&sub_bits_b.to_le_bytes());
    packed.extend_from_slice(&(levels_a as u32).to_le_bytes());
    packed.extend_from_slice(&(levels_b as u32).to_le_bytes());

    packed
}

fn dequantize_product(buf: &[u8], len: usize) -> Vec<f32> {
    if buf.len() < 24 {
        return vec![0.0f32; len];
    }

    // Read metadata from last 24 bytes
    let off = buf.len() - 24;
    let scale_a = f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    let scale_b = f32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]);
    let sub_bits_a = u32::from_le_bytes([buf[off + 8], buf[off + 9], buf[off + 10], buf[off + 11]]);
    let sub_bits_b =
        u32::from_le_bytes([buf[off + 12], buf[off + 13], buf[off + 14], buf[off + 15]]);
    let _levels_a =
        u32::from_le_bytes([buf[off + 16], buf[off + 17], buf[off + 18], buf[off + 19]]);
    let _levels_b =
        u32::from_le_bytes([buf[off + 20], buf[off + 21], buf[off + 22], buf[off + 23]]);

    let bits_per_val = (1 + sub_bits_a + sub_bits_b) as usize;
    let mut result = Vec::with_capacity(len);
    let mut bit_pos = 0;

    for _ in 0..len {
        let val = bit_read(buf, bit_pos, bits_per_val);
        let sign = (val >> (sub_bits_a + sub_bits_b)) & 1;
        let idx_a = (val >> sub_bits_b) & ((1u32 << sub_bits_a) - 1);
        let idx_b = val & ((1u32 << sub_bits_b) - 1);

        let cb_a = idx_a as f32 * scale_a;
        let cb_b = idx_b as f32 * scale_b;
        let recon = cb_a * cb_b;
        let out_val = if sign == 0 { recon } else { -recon };
        result.push(out_val);

        bit_pos += bits_per_val;
    }

    result
}

// ---------------------------------------------------------------------------
// Split quantization (head dimension split)
// ---------------------------------------------------------------------------

fn quantize_split(data: &[f32], head_dim: usize, bits: u32) -> (Vec<u8>, Vec<u8>) {
    let half = head_dim / 2;
    let first_bits = bits;
    let second_bits = if bits > 2 { bits - 1 } else { bits };

    let first_half = &data[..half];
    let second_half = &data[half..];

    let packed_first = quantize_polar_stored_scale(first_half, first_bits);
    let packed_second = quantize_polar_stored_scale(second_half, second_bits);

    // Concatenate: [first_packed | second_packed]
    let mut combined = Vec::with_capacity(packed_first.len() + packed_second.len());
    combined.extend_from_slice(&packed_first);
    combined.extend_from_slice(&packed_second);
    // Append metadata for split point
    combined.extend_from_slice(&first_bits.to_le_bytes());
    combined.extend_from_slice(&second_bits.to_le_bytes());
    combined.extend_from_slice(&(head_dim as u32).to_le_bytes());

    (combined, packed_first)
}

fn dequantize_split(buf: &[u8], len: usize) -> Vec<f32> {
    if buf.len() < 12 {
        return vec![0.0f32; len];
    }

    // Read metadata from last 12 bytes
    let off = buf.len() - 12;
    let first_bits = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    let second_bits = u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]);
    let head_dim = u32::from_le_bytes([buf[off + 8], buf[off + 9], buf[off + 10], buf[off + 11]]);

    let half = (head_dim / 2) as usize;
    let mut full = Vec::with_capacity(len);

    // Split the buffer without the 12-byte metadata trailer
    let data_end = buf.len() - 12;

    // First half: first_packed = (first_bits * half + 7) / 8 bytes + 4 bytes scale
    let first_packed_size = ((first_bits as usize * half + 7) / 8) + 4;
    let _second_packed_size = data_end - first_packed_size;

    let first_part = &buf[..first_packed_size.min(data_end)];
    let second_part = &buf[first_packed_size.min(data_end)..data_end];

    let first_recon = dequantize_polar_stored_scale(first_part, half, first_bits);
    let second_recon = dequantize_polar_stored_scale(second_part, len - half, second_bits);

    full.extend_from_slice(&first_recon);
    full.extend_from_slice(&second_recon);
    full
}

// ---------------------------------------------------------------------------
// PolarProd: Polar then product quantization on residuals
// ---------------------------------------------------------------------------

fn quantize_polarprod(data: &[f32], bits: u32) -> Vec<u8> {
    let polar_bits = (bits / 2).max(1);
    let prod_bits = bits.saturating_sub(polar_bits).max(1);

    // Step 1: polar quantize
    let polar_packed = quantize_polar_stored_scale(data, polar_bits);

    // Step 2: compute residuals (original - polar_reconstructed)
    let polar_recon = dequantize_polar_stored_scale(&polar_packed, data.len(), polar_bits);
    let residuals: Vec<f32> = data
        .iter()
        .zip(&polar_recon)
        .map(|(orig, recon)| orig - recon)
        .collect();

    // Step 3: product quantize residuals
    // Use the extended product format (which itself stores metadata)
    let prod_packed = quantize_product(&residuals, prod_bits);

    // Concatenate: [polar_packed | prod_packed]
    let mut combined = Vec::with_capacity(polar_packed.len() + prod_packed.len());
    combined.extend_from_slice(&polar_packed);
    combined.extend_from_slice(&prod_packed);
    // Metadata: polar_bit_count, prod_bit_count, data_len
    combined.extend_from_slice(&polar_bits.to_le_bytes());
    combined.extend_from_slice(&prod_bits.to_le_bytes());
    combined.extend_from_slice(&(data.len() as u32).to_le_bytes());

    combined
}

fn dequantize_polarprod(buf: &[u8], len: usize) -> Vec<f32> {
    if buf.len() < 12 {
        return vec![0.0f32; len];
    }

    let off = buf.len() - 12;
    let polar_bits = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    let _prod_bits = u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]);
    let _n_elements =
        u32::from_le_bytes([buf[off + 8], buf[off + 9], buf[off + 10], buf[off + 11]]);

    // Polar part: (polar_bits * len + 7) / 8 + 4 (scale)
    let polar_size = ((polar_bits as usize * len + 7) / 8) + 4;
    let data_end = buf.len() - 12;

    let polar_part = &buf[..polar_size.min(data_end)];
    let prod_part = &buf[polar_size.min(data_end)..data_end];

    let polar_recon = dequantize_polar_stored_scale(polar_part, len, polar_bits);
    let prod_recon = dequantize_product(prod_part, len);

    polar_recon
        .iter()
        .zip(&prod_recon)
        .map(|(p, r)| p + r)
        .collect()
}

// ---------------------------------------------------------------------------
// MSE-optimal state selection
// ---------------------------------------------------------------------------

fn quantize_mse(data: &[f32], bits: u32, state_bits: u32) -> Vec<u8> {
    let num_states = 1usize << state_bits;
    let quant_levels = 1usize << bits;

    let max_abs = data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let scale = if max_abs == 0.0 {
        1.0
    } else {
        max_abs / quant_levels as f32
    };

    // Build codebook: uniformly spaced values
    let codebook: Vec<f32> = (0..quant_levels).map(|i| i as f32 * scale).collect();

    // For each group_size-sized block, find the best codebook entry
    // Pack: state_id (state_bits per element) + quantized value (bits per element)
    let bits_per_elem = (state_bits + bits) as usize;
    let mut packed = Vec::with_capacity((data.len() * bits_per_elem + 7) / 8 + 8);
    let mut bit_pos = 0;

    for &x in data {
        let ax = x.abs();
        let sign = if x < 0.0 { 1u32 } else { 0u32 };

        // Find best quantized value in codebook
        let best_idx = if scale == 0.0 {
            0u32
        } else {
            let raw = (ax / scale).round() as u32;
            raw.min((quant_levels - 1) as u32)
        };

        // Find which state entry best represents this
        let target_qval = best_idx as f32 * scale;
        let mut best_state = 0u32;
        let mut best_err = f32::MAX;
        for s in 0..num_states {
            // Use a deterministic rotation of the codebook as the "state"
            let state_val = codebook[(s * best_idx as usize) % quant_levels];
            let err = (target_qval - state_val).abs();
            if err < best_err {
                best_err = err;
                best_state = s as u32;
            }
        }

        // Pack: state_bits of state_id | 1 sign bit | bits of quantized_idx
        let val = (best_state << (1 + bits)) | (sign << bits) | best_idx;
        bit_write(&mut packed, bit_pos, bits_per_elem + 1, val);
        bit_pos += bits_per_elem + 1;
    }

    // Store metadata
    packed.extend_from_slice(&bits.to_le_bytes());
    packed.extend_from_slice(&state_bits.to_le_bytes());
    packed.extend_from_slice(&scale.to_le_bytes());

    packed
}

fn dequantize_mse(buf: &[u8], len: usize) -> Vec<f32> {
    if buf.len() < 12 {
        return vec![0.0f32; len];
    }

    let off = buf.len() - 12;
    let scale = f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    let rbits = u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]);
    let rstate_bits =
        u32::from_le_bytes([buf[off + 8], buf[off + 9], buf[off + 10], buf[off + 11]]);

    let bits_per_elem = (rstate_bits + rbits) as usize + 1; // +1 for sign bit
    let mut result = Vec::with_capacity(len);
    let mut bit_pos = 0;

    for _ in 0..len {
        let val = bit_read(buf, bit_pos, bits_per_elem);
        let _state_id = val >> (rbits + 1);
        let sign = (val >> rbits) & 1;
        let idx = val & ((1u32 << rbits) - 1);

        let recon = idx as f32 * scale;
        let out_val = if sign == 0 { recon } else { -recon };
        result.push(out_val);

        bit_pos += bits_per_elem;
    }

    result
}

// ---------------------------------------------------------------------------
// TurboQuantKvCache implementation
// ---------------------------------------------------------------------------

impl TurboQuantKvCache {
    pub fn new(quant_mode: KvQuantMode, group_size: usize, num_slots: usize) -> Self {
        let state = Vec::with_capacity(num_slots);
        Self {
            quant_mode,
            group_size,
            state,
        }
    }

    /// Quantize a KV cache entry
    pub fn quantize(
        &mut self,
        slot: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), TurboQuantError> {
        if slot >= self.state.capacity() {
            return Err(TurboQuantError::SlotOutOfBounds(
                slot,
                self.state.capacity(),
            ));
        }

        let total_elems = keys.len();
        let head_dim = total_elems; // For now, we assume one token

        let bits = match self.quant_mode {
            KvQuantMode::Polar(b) => b,
            KvQuantMode::Prod(b) => b,
            KvQuantMode::Split(b) => b,
            KvQuantMode::PolarProd(b) => b,
            KvQuantMode::Mse { bits: b, .. } => b,
        };

        let keys_quantized = match self.quant_mode {
            KvQuantMode::Polar(_) => quantize_polar_stored_scale(keys, bits),
            KvQuantMode::Prod(_) => quantize_product(keys, bits),
            KvQuantMode::Split(_) => {
                let (combined, _) = quantize_split(keys, head_dim, bits);
                combined
            }
            KvQuantMode::PolarProd(_) => quantize_polarprod(keys, bits),
            KvQuantMode::Mse { state_bits, .. } => quantize_mse(keys, bits, state_bits),
        };

        let values_quantized = match self.quant_mode {
            KvQuantMode::Polar(_) => quantize_polar_stored_scale(values, bits),
            KvQuantMode::Prod(_) => quantize_product(values, bits),
            KvQuantMode::Split(_) => {
                let (combined, _) = quantize_split(values, head_dim, bits);
                combined
            }
            KvQuantMode::PolarProd(_) => quantize_polarprod(values, bits),
            KvQuantMode::Mse { state_bits, .. } => quantize_mse(values, bits, state_bits),
        };

        let state = TurboQuantState {
            keys: keys_quantized,
            values: values_quantized,
            num_tokens: 1,
            bits,
            head_dim,
        };

        if slot >= self.state.len() {
            // Extend to fill gap
            self.state.resize_with(slot + 1, || TurboQuantState {
                keys: Vec::new(),
                values: Vec::new(),
                num_tokens: 0,
                bits,
                head_dim: 0,
            });
            if let Some(s) = self.state.get_mut(slot) {
                *s = state;
            }
        } else {
            self.state[slot] = state;
        }

        Ok(())
    }

    /// Dequantize a KV cache entry for inference
    pub fn dequantize(&self, slot: usize) -> Result<(Vec<f32>, Vec<f32>), TurboQuantError> {
        let st = self
            .state
            .get(slot)
            .ok_or(TurboQuantError::SlotOutOfBounds(slot, self.state.len()))?;

        if st.keys.is_empty() {
            return Err(TurboQuantError::EmptySlot(slot));
        }

        let num_elems = st.head_dim;

        let keys = match self.quant_mode {
            KvQuantMode::Polar(_) => dequantize_polar_stored_scale(&st.keys, num_elems, st.bits),
            KvQuantMode::Prod(_) => dequantize_product(&st.keys, num_elems),
            KvQuantMode::Split(_) => dequantize_split(&st.keys, num_elems),
            KvQuantMode::PolarProd(_) => dequantize_polarprod(&st.keys, num_elems),
            KvQuantMode::Mse { .. } => dequantize_mse(&st.keys, num_elems),
        };

        let values = match self.quant_mode {
            KvQuantMode::Polar(_) => dequantize_polar_stored_scale(&st.values, num_elems, st.bits),
            KvQuantMode::Prod(_) => dequantize_product(&st.values, num_elems),
            KvQuantMode::Split(_) => dequantize_split(&st.values, num_elems),
            KvQuantMode::PolarProd(_) => dequantize_polarprod(&st.values, num_elems),
            KvQuantMode::Mse { .. } => dequantize_mse(&st.values, num_elems),
        };

        Ok((keys, values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn max_rel_err(got: &[f32], expected: &[f32]) -> f32 {
        let max_abs = expected.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        if max_abs == 0.0 {
            got.iter()
                .zip(expected)
                .map(|(g, e)| (g - e).abs())
                .fold(0.0f32, f32::max)
        } else {
            got.iter()
                .zip(expected)
                .map(|(g, e)| (g - e).abs() / max_abs)
                .fold(0.0f32, f32::max)
        }
    }

    #[test]
    fn test_quant_mode_creation() {
        let mode = KvQuantMode::Polar(2);
        let _cache = TurboQuantKvCache::new(mode, 32, 8);
    }

    #[test]
    fn test_bit_roundtrip() {
        let mut buf = Vec::new();
        bit_write(&mut buf, 0, 3, 0b101);
        bit_write(&mut buf, 3, 5, 0b11010);
        assert_eq!(bit_read(&buf, 0, 3), 0b101);
        assert_eq!(bit_read(&buf, 3, 5), 0b11010);
    }

    #[test]
    fn test_bit_write_cross_byte() {
        let mut buf = Vec::new();
        bit_write(&mut buf, 6, 4, 0b1111);
        // Bits at positions 6,7,8,9 -> byte0 bits 6-7, byte1 bits 0-1
        assert_eq!(bit_read(&buf, 6, 4), 0b1111);
    }

    #[test]
    fn test_polar_roundtrip() {
        let data = vec![1.0f32, -2.5, 3.0, -0.5, 0.0, -1.0];
        let bits = 4;
        let packed = quantize_polar_stored_scale(&data, bits);
        let recon = dequantize_polar_stored_scale(&packed, data.len(), bits);
        assert_eq!(recon.len(), data.len());
        let err = max_rel_err(&recon, &data);
        assert!(err < 0.15, "polar max_rel_err = {}", err);
    }

    #[test]
    fn test_polar_zeros() {
        let data = vec![0.0f32; 10];
        let bits = 3;
        let packed = quantize_polar_stored_scale(&data, bits);
        let recon = dequantize_polar_stored_scale(&packed, data.len(), bits);
        for v in &recon {
            assert!(v.abs() < 1e-6, "expected near-zero, got {}", v);
        }
    }

    #[test]
    fn test_polar_signed() {
        let data = vec![100.0f32, -100.0, 0.1, -0.1];
        let bits = 3;
        let packed = quantize_polar_stored_scale(&data, bits);
        let recon = dequantize_polar_stored_scale(&packed, data.len(), bits);
        for (r, &d) in recon.iter().zip(&data) {
            if d.abs() > 0.01 {
                assert_eq!(r.signum(), d.signum(), "sign mismatch: {} vs {}", r, d);
            }
        }
    }

    #[test]
    fn test_product_roundtrip() {
        let data = vec![1.0f32, 4.0, 9.0, 0.5, 2.0, -3.0];
        let bits = 3;
        let packed = quantize_product(&data, bits);
        let recon = dequantize_product(&packed, data.len());
        assert_eq!(recon.len(), data.len());
        let err = max_rel_err(&recon, &data);
        assert!(
            err < 0.5,
            "product max_rel_err = {} (may be high with few bits)",
            err
        );
    }

    #[test]
    fn test_product_zeros() {
        let data = vec![0.0f32; 10];
        let bits = 4;
        let packed = quantize_product(&data, bits);
        let recon = dequantize_product(&packed, data.len());
        for v in &recon {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_split_roundtrip() {
        let head_dim = 8;
        let data: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.5).collect();
        let bits = 4;
        let (combined, _) = quantize_split(&data, head_dim, bits);
        let recon = dequantize_split(&combined, data.len());
        assert_eq!(recon.len(), data.len());
        let err = max_rel_err(&recon, &data);
        assert!(err < 0.15, "split max_rel_err = {}", err);
    }

    #[test]
    fn test_polarprod_roundtrip() {
        let data = vec![1.0f32, -2.5, 3.0, -0.5, 0.0, -1.0, 0.7, -0.3];
        let bits = 3;
        let packed = quantize_polarprod(&data, bits);
        let recon = dequantize_polarprod(&packed, data.len());
        assert_eq!(recon.len(), data.len());
        let err = max_rel_err(&recon, &data);
        assert!(err < 1.0, "polarprod max_rel_err = {}", err);
    }

    #[test]
    fn test_mse_roundtrip() {
        let data = vec![1.0f32, -2.0, 3.0, -0.5, 0.0];
        let bits = 4;
        let state_bits = 2;
        let packed = quantize_mse(&data, bits, state_bits);
        let recon = dequantize_mse(&packed, data.len());
        assert_eq!(recon.len(), data.len());
        let err = max_rel_err(&recon, &data);
        assert!(err < 0.2, "mse max_rel_err = {}", err);
    }

    #[test]
    fn test_cache_quantize_dequantize_polar() {
        let mode = KvQuantMode::Polar(4);
        let mut cache = TurboQuantKvCache::new(mode, 32, 4);
        let keys = vec![1.0f32, -2.0, 3.0, -0.5];
        let values = vec![0.5f32, -1.0, 1.5, -2.5];
        cache.quantize(0, &keys, &values).unwrap();
        let (k_recon, v_recon) = cache.dequantize(0).unwrap();
        assert_eq!(k_recon.len(), keys.len());
        assert_eq!(v_recon.len(), values.len());
    }

    #[test]
    fn test_cache_quantize_dequantize_prod() {
        let mode = KvQuantMode::Prod(3);
        let mut cache = TurboQuantKvCache::new(mode, 32, 4);
        let keys = vec![1.0f32, 4.0, 9.0, 0.5];
        let values = vec![2.0f32, 1.0, 3.0, 0.1];
        cache.quantize(0, &keys, &values).unwrap();
        let (k_recon, v_recon) = cache.dequantize(0).unwrap();
        assert_eq!(k_recon.len(), keys.len());
        assert_eq!(v_recon.len(), values.len());
    }

    #[test]
    fn test_cache_quantize_dequantize_split() {
        let head_dim = 8;
        let mode = KvQuantMode::Split(4);
        let mut cache = TurboQuantKvCache::new(mode, 32, 4);
        let keys: Vec<f32> = (0..head_dim).map(|i| i as f32).collect();
        let values: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.5).collect();
        cache.quantize(0, &keys, &values).unwrap();
        let (k_recon, v_recon) = cache.dequantize(0).unwrap();
        assert_eq!(k_recon.len(), keys.len());
        assert_eq!(v_recon.len(), values.len());
    }

    #[test]
    fn test_cache_quantize_dequantize_polarprod() {
        let mode = KvQuantMode::PolarProd(4);
        let mut cache = TurboQuantKvCache::new(mode, 32, 4);
        let keys = vec![1.0f32, -2.0, 3.0, -0.5];
        let values = vec![0.5f32, -1.0, 1.5, -2.5];
        cache.quantize(0, &keys, &values).unwrap();
        let (k_recon, v_recon) = cache.dequantize(0).unwrap();
        assert_eq!(k_recon.len(), keys.len());
        assert_eq!(v_recon.len(), values.len());
    }

    #[test]
    fn test_cache_quantize_dequantize_mse() {
        let mode = KvQuantMode::Mse {
            bits: 4,
            state_bits: 2,
        };
        let mut cache = TurboQuantKvCache::new(mode, 32, 4);
        let keys = vec![1.0f32, -2.0, 3.0, -0.5];
        let values = vec![0.5f32, -1.0, 1.5, -2.5];
        cache.quantize(0, &keys, &values).unwrap();
        let (k_recon, v_recon) = cache.dequantize(0).unwrap();
        assert_eq!(k_recon.len(), keys.len());
        assert_eq!(v_recon.len(), values.len());
    }

    #[test]
    fn test_cache_slot_out_of_bounds() {
        let mode = KvQuantMode::Polar(4);
        let mut cache = TurboQuantKvCache::new(mode, 32, 2);
        let data = vec![1.0f32; 4];
        assert!(cache.quantize(5, &data, &data).is_err());
        assert!(cache.dequantize(5).is_err());
    }

    #[test]
    fn test_cache_empty_slot() {
        let mode = KvQuantMode::Polar(4);
        let cache = TurboQuantKvCache::new(mode, 32, 4);
        assert!(cache.dequantize(0).is_err());
    }

    #[test]
    fn test_all_modes_create() {
        for mode in &[
            KvQuantMode::Polar(2),
            KvQuantMode::Polar(3),
            KvQuantMode::Polar(4),
            KvQuantMode::Prod(3),
            KvQuantMode::Prod(4),
            KvQuantMode::Split(4),
            KvQuantMode::PolarProd(4),
            KvQuantMode::Mse {
                bits: 4,
                state_bits: 2,
            },
        ] {
            let _cache = TurboQuantKvCache::new(*mode, 32, 4);
        }
    }
}
