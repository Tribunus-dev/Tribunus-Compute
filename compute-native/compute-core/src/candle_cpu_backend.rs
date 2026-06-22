//! Candle CPU backend — [`TensorBackend`] implementation using candle-core.
//!
//! This backend runs on CPU via [`candle_core::Tensor`] on
//! `candle_core::Device::Cpu`. It supports f32, quantized int4 matmul
//! (dequantize + matmul), RoPE, RMS norm, and common element-wise ops.
//!
//! # Feature gate
//!
//! This module is compiled only when `feature = "candle-cpu"` is enabled
//! (see `Cargo.toml`).

use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use crate::backend::DType as BackendDType;
use crate::backend::{
    authority, BackendCapabilities, EvaluationReceipt, MatmulOp, QuantizedMatmulOp,
    QuantizedWeightHandle, ReadbackReceipt, RmsNormOp, RoPEOp, TensorBackend, TensorHandle,
};

// ── DType mapping ──────────────────────────────────────────────────────────

/// Convert a backend [`BackendDType`] to a candle [`DType`].
///
/// Returns `None` for dtypes that have no candle equivalent.
/// candle-core 0.8.4 exposes: U8, U32, I64, BF16, F16, F32, F64.
fn backend_dtype_to_candle(dt: BackendDType) -> Option<DType> {
    match dt {
        BackendDType::F32 => Some(DType::F32),
        BackendDType::F16 => Some(DType::F16),
        BackendDType::BF16 => Some(DType::BF16),
        BackendDType::I8 | BackendDType::I32 => Some(DType::I64),
        BackendDType::U8 => Some(DType::U8),
        BackendDType::U32 => Some(DType::U32),
    }
}

// ── Candle CPU backend ─────────────────────────────────────────────────────

/// Candle-core backed implementation of [`TensorBackend`].
///
/// Uses generational slot-maps identical to [`MlxBackend`]. Tensors live in
/// CPU memory; all operations are eager (candle has no lazy evaluation).
pub struct CandleCpuBackend {
    tensors: Vec<Option<Tensor>>,
    generations: Vec<u32>,
    free_list: Vec<usize>,
    weight_tensors: Vec<Option<Tensor>>,
    weight_generations: Vec<u32>,
    name: String,
}

impl CandleCpuBackend {
    /// Create a new empty backend with default name `"candle-cpu"`.
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            weight_tensors: Vec::new(),
            weight_generations: Vec::new(),
            name: "candle-cpu".to_string(),
        }
    }

    /// Create a new backend with a custom name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            tensors: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            weight_tensors: Vec::new(),
            weight_generations: Vec::new(),
            name: name.into(),
        }
    }

    /// Allocate a slot for a regular tensor and return the handle.
    fn alloc_tensor(&mut self, t: Tensor) -> TensorHandle {
        if let Some(idx) = self.free_list.pop() {
            self.generations[idx] += 1;
            self.tensors[idx] = Some(t);
            TensorHandle {
                slot: idx as u32,
                generation: self.generations[idx],
            }
        } else {
            let idx = self.tensors.len();
            self.tensors.push(Some(t));
            self.generations.push(1);
            TensorHandle {
                slot: idx as u32,
                generation: 1,
            }
        }
    }

    /// Allocate a slot for a quantized weight tensor and return the handle.
    pub(crate) fn alloc_weight(&mut self, t: Tensor) -> QuantizedWeightHandle {
        let idx = self.weight_tensors.len();
        self.weight_tensors.push(Some(t));
        self.weight_generations.push(1);
        QuantizedWeightHandle {
            slot: idx as u32,
            generation: 1,
        }
    }

    /// Get an immutable reference to the tensor at `handle`, validating slot
    /// and generation.
    fn get_tensor(&self, handle: TensorHandle) -> Result<&Tensor, String> {
        let slot = handle.slot as usize;
        let generation = handle.generation;
        match self.tensors.get(slot) {
            Some(Some(t)) if generation == self.generations[slot] => Ok(t),
            _ => Err(format!(
                "CandleCpuBackend: invalid tensor handle (slot={}, gen={})",
                slot, generation,
            )),
        }
    }

    /// Get an immutable reference to the weight tensor at `handle`.
    fn get_weight(&self, handle: QuantizedWeightHandle) -> Result<&Tensor, String> {
        let slot = handle.slot as usize;
        let generation = handle.generation;
        match self.weight_tensors.get(slot) {
            Some(Some(t)) if generation == self.weight_generations[slot] => Ok(t),
            _ => Err(format!(
                "CandleCpuBackend: invalid weight handle (slot={}, gen={})",
                slot, generation,
            )),
        }
    }
}

impl Default for CandleCpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorBackend for CandleCpuBackend {
    // ── Creation ───────────────────────────────────────────────────────

    fn create_f32(&mut self, data: &[f32], shape: &[i32]) -> Result<TensorHandle, String> {
        let candle_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let t = Tensor::from_slice(data, candle_shape.as_slice(), &Device::Cpu)
            .map_err(|e| format!("create_f32: {e}"))?;
        Ok(self.alloc_tensor(t))
    }

    fn create_u32(&mut self, data: &[u32], shape: &[i32]) -> Result<TensorHandle, String> {
        let candle_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let t = Tensor::from_slice(data, candle_shape.as_slice(), &Device::Cpu)
            .map_err(|e| format!("create_u32: {e}"))?;
        Ok(self.alloc_tensor(t))
    }

    fn create_f32_from_bf16_bits(
        &mut self,
        data: &[u16],
        shape: &[i32],
    ) -> Result<TensorHandle, String> {
        // Convert BF16 bits (stored as u16) to f32.
        let f32_vec: Vec<f32> = data
            .iter()
            .map(|&v| {
                let bits = (v as u32) << 16;
                f32::from_bits(bits)
            })
            .collect();
        let candle_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let t = Tensor::from_slice(&f32_vec, candle_shape.as_slice(), &Device::Cpu)
            .map_err(|e| format!("create_f32_from_bf16_bits: {e}"))?;
        Ok(self.alloc_tensor(t))
    }

    fn create_owned_from_bytes(
        &mut self,
        data: &[u8],
        shape: &[i32],
        dtype: BackendDType,
    ) -> Result<TensorHandle, String> {
        let candle_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let t = match dtype {
            BackendDType::F32 => {
                let (prefix, aligned, suffix) = unsafe { data.align_to::<f32>() };
                if !prefix.is_empty() || !suffix.is_empty() {
                    return Err("create_owned_from_bytes: f32 data not aligned to 4 bytes".into());
                }
                Tensor::from_slice(aligned, candle_shape.as_slice(), &Device::Cpu)
                    .map_err(|e| format!("create_owned_from_bytes(f32): {e}"))?
            }
            BackendDType::U32 => {
                let (prefix, aligned, suffix) = unsafe { data.align_to::<u32>() };
                if !prefix.is_empty() || !suffix.is_empty() {
                    return Err("create_owned_from_bytes: u32 data not aligned to 4 bytes".into());
                }
                Tensor::from_slice(aligned, candle_shape.as_slice(), &Device::Cpu)
                    .map_err(|e| format!("create_owned_from_bytes(u32): {e}"))?
            }
            _ => {
                return Err(format!(
                    "create_owned_from_bytes: dtype {dtype:?} is not physically supported; \
                     use create_f32_from_bf16_bits for BF16 data",
                ));
            }
        };
        Ok(self.alloc_tensor(t))
    }

    // ── Core compute ───────────────────────────────────────────────────

    fn quantized_matmul(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        scales: TensorHandle,
        biases: TensorHandle,
    ) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;
        let x_dims = x_t.shape().dims();
        if x_dims.len() < 2 {
            return Err("quantized_matmul: input must have at least 2 dimensions".into());
        }
        let x_m = x_dims[x_dims.len() - 2] as u32;
        let x_k = x_dims[x_dims.len() - 1] as u32;
        if x_m != op.m {
            return Err(format!(
                "quantized_matmul: input M={} != op.m={}",
                x_m, op.m
            ));
        }
        if x_k != op.k {
            return Err(format!(
                "quantized_matmul: input K={} != op.k={}",
                x_k, op.k
            ));
        }

        let w_t = self.get_weight(w)?;
        let w_dims = w_t.shape().dims();
        if w_dims.len() < 2 {
            return Err("quantized_matmul: weight must have at least 2 dimensions".into());
        }
        let w_k = w_dims[w_dims.len() - 2] as u32;
        let packed_cols = w_dims[w_dims.len() - 1];
        if w_k != op.k {
            return Err(format!(
                "quantized_matmul: weight K={} != op.k={}",
                w_k, op.k
            ));
        }

        let s_t = self.get_tensor(scales)?;
        let b_t = self.get_tensor(biases)?;

        let x_dims = x_t.shape().dims();
        let _m = x_dims[x_dims.len() - 2];
        let k = x_dims[x_dims.len() - 1];
        let n_out = w_dims[0];
        let n_groups = s_t.shape().dims()[s_t.shape().dims().len() - 1];
        let gs = op.group_size as usize;

        // Read packed weights as u32
        let w_flat = w_t.flatten_all().map_err(|e| format!("w flatten: {e}"))?;
        let w_u32: Vec<u32> = w_flat
            .to_vec1::<u32>()
            .map_err(|e| format!("w to_vec1: {e}"))?;

        // Read scales and biases
        let s_flat = s_t.flatten_all().map_err(|e| format!("s flatten: {e}"))?;
        let scales_f32: Vec<f32> = s_flat
            .to_vec1::<f32>()
            .map_err(|e| format!("s to_vec1: {e}"))?;
        let b_flat = b_t.flatten_all().map_err(|e| format!("b flatten: {e}"))?;
        let biases_f32: Vec<f32> = b_flat
            .to_vec1::<f32>()
            .map_err(|e| format!("b to_vec1: {e}"))?;

        // Dequantize: packed int4 (8 nibbles per u32) -> f32 [n_out, k]
        // Use shared authority dequantize utility.
        let w_f32 = authority::dequantize_int4_weights(
            &w_u32,
            &scales_f32,
            &biases_f32,
            n_out,
            k,
            n_groups,
            packed_cols,
            gs,
        );

        // Build dequantized weight tensor [n_out, k] then transpose to [k, n_out]
        let w_deq = Tensor::from_slice(&w_f32, (n_out, k), &Device::Cpu)
            .map_err(|e| format!("deq weight from_slice: {e}"))?;
        let wt = w_deq
            .transpose(0, 1)
            .map_err(|e| format!("deq weight transpose: {e}"))?;

        // Regular matmul: x [m, k] @ wt [k, n_out] -> [m, n_out]
        let result = x_t
            .matmul(&wt)
            .map_err(|e| format!("quantized_matmul matmul: {e}"))?;

        Ok(self.alloc_tensor(result))
    }

    fn matmul(
        &mut self,
        op: &MatmulOp,
        a: TensorHandle,
        b: TensorHandle,
    ) -> Result<TensorHandle, String> {
        let a_t = self.get_tensor(a)?;
        let b_t = self.get_tensor(b)?;

        let a_dims = a_t.shape().dims();
        let b_dims = b_t.shape().dims();
        let a_m = if a_dims.len() >= 2 {
            a_dims[a_dims.len() - 2] as u32
        } else {
            1
        };
        let a_k = *a_dims.last().unwrap_or(&0) as u32;
        let b_k = if b_dims.len() >= 2 {
            b_dims[b_dims.len() - 2] as u32
        } else {
            b_dims[0] as u32
        };
        let b_n = *b_dims.last().unwrap_or(&0) as u32;

        if a_m != op.m {
            return Err(format!("matmul: A.M={} != op.m={}", a_m, op.m));
        }
        if a_k != op.k || b_k != op.k {
            return Err(format!(
                "matmul: K mismatch (A.K={}, B.K={}, op.k={})",
                a_k, b_k, op.k
            ));
        }
        if b_n != op.n {
            return Err(format!("matmul: B.N={} != op.n={}", b_n, op.n));
        }

        let out = a_t.matmul(b_t).map_err(|e| format!("matmul failed: {e}"))?;

        // Validate output shape
        let out_dims = out.shape().dims();
        let out_m = if out_dims.len() >= 2 {
            out_dims[out_dims.len() - 2] as u32
        } else {
            1
        };
        let out_n = *out_dims.last().unwrap_or(&0) as u32;
        if out_m != op.m || out_n != op.n {
            return Err(format!(
                "matmul: output ({},{}) != op ({},{})",
                out_m, out_n, op.m, op.n
            ));
        }

        Ok(self.alloc_tensor(out))
    }

    fn rms_norm(
        &mut self,
        op: &RmsNormOp,
        x: TensorHandle,
        weight: TensorHandle,
    ) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;
        let w_t = self.get_tensor(weight)?;

        let dim = op.dim as usize;
        let eps = op.eps;

        let x_dtype = x_t.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        // Cast to internal dtype if needed for numeric stability.
        let x_internal = x_t
            .to_dtype(internal_dtype)
            .map_err(|e| format!("rms_norm to_dtype: {e}"))?;

        let sq = x_internal.sqr().map_err(|e| format!("rms_norm sqr: {e}"))?;
        let mean_sq = sq.mean(dim).map_err(|e| format!("rms_norm mean: {e}"))?;
        let eps_t = Tensor::new(eps as f64, &Device::Cpu)
            .map_err(|e| format!("rms_norm eps const: {e}"))?;
        let var_eps = mean_sq
            .broadcast_add(&eps_t)
            .map_err(|e| format!("rms_norm add eps: {e}"))?;
        let rms = var_eps.sqrt().map_err(|e| format!("rms_norm sqrt: {e}"))?;
        let normalized = x_internal
            .broadcast_div(&rms)
            .map_err(|e| format!("rms_norm div: {e}"))?;

        // Cast back to original dtype and multiply by weight.
        let normalized = normalized
            .to_dtype(x_dtype)
            .map_err(|e| format!("rms_norm to_dtype back: {e}"))?;

        let out = normalized
            .broadcast_mul(w_t)
            .map_err(|e| format!("rms_norm mul weight: {e}"))?;

        Ok(self.alloc_tensor(out))
    }

    fn rope(&mut self, op: &RoPEOp, x: TensorHandle) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;
        let x_rank = x_t.shape().rank();

        // Expected shape: [batch, seq_len, n_heads, head_dim]
        // or [seq_len, head_dim]
        if x_rank < 2 {
            return Err("rope: input must have at least 2 dimensions".into());
        }

        let head_dim = op.head_dim as usize;
        let x_dims = x_t.shape().dims();

        // Determine sequence length from shape.
        let seq_len = x_dims[x_rank - 2];

        // Inv_freq for each pair in head_dim.
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| {
                let exponent = i as f64 / head_dim as f64;
                1.0 / (10_000.0f64.powf(exponent)) as f32
            })
            .collect();

        let half = head_dim / 2;

        // Build cos/sin for all positions.
        let mut cos_vals = vec![0.0f32; seq_len * half];
        let mut sin_vals = vec![0.0f32; seq_len * half];

        for (pos_idx, &pos) in op.positions.iter().enumerate() {
            if pos_idx >= seq_len {
                break;
            }
            for (i, &inv) in inv_freq.iter().enumerate() {
                let theta = pos as f32 * inv;
                cos_vals[pos_idx * half + i] = theta.cos();
                sin_vals[pos_idx * half + i] = theta.sin();
            }
        }

        let cos_t = Tensor::from_slice(&cos_vals, (seq_len, half), &Device::Cpu)
            .map_err(|e| format!("rope cos from_slice: {e}"))?;
        let sin_t = Tensor::from_slice(&sin_vals, (seq_len, half), &Device::Cpu)
            .map_err(|e| format!("rope sin from_slice: {e}"))?;

        // Split x into two halves along head_dim (last dim for 2D tensors,
        // x_rank-1 for higher-rank).
        let split_dim = x_rank - 1;
        let x1 = x_t
            .narrow(split_dim, 0, half)
            .map_err(|e| format!("rope narrow x1: {e}"))?;
        let x2 = x_t
            .narrow(split_dim, half, head_dim - half)
            .map_err(|e| format!("rope narrow x2: {e}"))?;

        // Apply rotation: x_rotated = x * cos + rotate_half(x) * sin
        // x1_rot = x1 * cos - x2 * sin
        // x2_rot = x2 * cos + x1 * sin
        let x1_cos = x1
            .broadcast_mul(&cos_t)
            .map_err(|e| format!("rope x1*cos: {e}"))?;
        let x2_sin = x2
            .broadcast_mul(&sin_t)
            .map_err(|e| format!("rope x2*sin: {e}"))?;
        let x1_rot = (x1_cos - x2_sin).map_err(|e| format!("rope x1_rot sub: {e}"))?;

        let x2_cos = x2
            .broadcast_mul(&cos_t)
            .map_err(|e| format!("rope x2*cos: {e}"))?;
        let x1_sin = x1
            .broadcast_mul(&sin_t)
            .map_err(|e| format!("rope x1*sin: {e}"))?;
        let x2_rot = (x2_cos + x1_sin).map_err(|e| format!("rope x2_rot add: {e}"))?;

        let out =
            Tensor::cat(&[&x1_rot, &x2_rot], split_dim).map_err(|e| format!("rope cat: {e}"))?;

        Ok(self.alloc_tensor(out))
    }

    fn add(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String> {
        let a_t = self.get_tensor(a)?;
        let b_t = self.get_tensor(b)?;
        let out = a_t.add(b_t).map_err(|e| format!("add failed: {e}"))?;
        Ok(self.alloc_tensor(out))
    }

    fn multiply(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String> {
        let a_t = self.get_tensor(a)?;
        let b_t = self.get_tensor(b)?;
        let out = a_t.mul(b_t).map_err(|e| format!("multiply failed: {e}"))?;
        Ok(self.alloc_tensor(out))
    }

    fn silu(&mut self, x: TensorHandle) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;
        // Candle has silu as a unary op directly: x * sigmoid(x).
        let out = x_t.silu().map_err(|e| format!("silu failed: {e}"))?;
        Ok(self.alloc_tensor(out))
    }

    fn transpose(&mut self, x: TensorHandle, dims: &[i32]) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;
        if dims.len() != 2 {
            return Err(format!("transpose: expected 2 dims, got {}", dims.len()));
        }
        let out = x_t
            .transpose(dims[0] as usize, dims[1] as usize)
            .map_err(|e| format!("transpose failed: {e}"))?;
        Ok(self.alloc_tensor(out))
    }

    fn reshape(&mut self, x: TensorHandle, shape: &[i32]) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;
        // Compute the target shape handling -1 (infer) dimensions.
        let neg_count = shape.iter().filter(|&&d| d < 0).count();
        let out = if neg_count > 0 {
            if neg_count > 1 {
                return Err("reshape: only one -1 dimension is allowed".into());
            }
            let elem_count = x_t.elem_count();
            let known: usize = shape
                .iter()
                .filter(|&&d| d >= 0)
                .map(|&d| d as usize)
                .product();
            if known == 0 || elem_count % known != 0 {
                return Err(format!(
                    "reshape: cannot infer dimension (elem_count={}, known_product={})",
                    elem_count, known
                ));
            }
            let inferred = elem_count / known;
            let final_shape: Vec<usize> = shape
                .iter()
                .map(|&d| if d < 0 { inferred } else { d as usize })
                .collect();
            x_t.reshape(final_shape.as_slice())
                .map_err(|e| format!("reshape failed: {e}"))?
        } else {
            let candle_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            x_t.reshape(candle_shape.as_slice())
                .map_err(|e| format!("reshape failed: {e}"))?
        };
        Ok(self.alloc_tensor(out))
    }

    fn softmax(&mut self, x: TensorHandle, axis: i32) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;

        // Manual softmax: exp(x - max) / sum(exp(x - max)) along axis.
        let max = x_t
            .max_keepdim(axis as usize)
            .map_err(|e| format!("softmax max: {e}"))?;
        let diff = x_t
            .broadcast_sub(&max)
            .map_err(|e| format!("softmax sub: {e}"))?;
        let num = diff.exp().map_err(|e| format!("softmax exp: {e}"))?;
        let den = num
            .sum_keepdim(axis as usize)
            .map_err(|e| format!("softmax sum: {e}"))?;
        let out = num
            .broadcast_div(&den)
            .map_err(|e| format!("softmax div: {e}"))?;

        Ok(self.alloc_tensor(out))
    }

    fn index_select(
        &mut self,
        x: TensorHandle,
        indices: &[u32],
        axis: i32,
    ) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;

        // Build indices tensor as i64 (candle's index_select uses signed integer for indices).
        let indices_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
        let idx_t = Tensor::from_slice(&indices_i64, indices.len(), &Device::Cpu)
            .map_err(|e| format!("index_select indices from_slice: {e}"))?;

        let out = x_t
            .index_select(&idx_t, axis as usize)
            .map_err(|e| format!("index_select failed: {e}"))?;

        Ok(self.alloc_tensor(out))
    }

    // ── Missing ops (stubs) ────────────────────────────────────────────

    fn concatenate(
        &mut self,
        _tensors: &[TensorHandle],
        _axis: i32,
    ) -> Result<TensorHandle, String> {
        Err("concatenate: not implemented for CandleCpuBackend".into())
    }

    fn slice(
        &mut self,
        x: TensorHandle,
        start: &[i32],
        stop: &[i32],
        _step: &[i32],
    ) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;
        let x_dims = x_t.shape().dims();
        let rank = x_dims.len();

        // For each dimension, narrow from start to stop (step is assumed 1).
        let mut result = x_t.clone();
        for dim in 0..rank {
            let s = if dim < start.len() { start[dim] } else { 0 } as usize;
            let e = if dim < stop.len() {
                stop[dim] as usize
            } else {
                x_dims[dim]
            };
            if s >= e || s >= x_dims[dim] || e > x_dims[dim] {
                return Err(format!(
                    "slice: invalid range dim={dim} start={s} stop={e} shape={}",
                    x_dims[dim]
                ));
            }
            let len = e - s;
            result = result
                .narrow(dim, s, len)
                .map_err(|e| format!("slice narrow dim={dim}: {e}"))?;
        }
        Ok(self.alloc_tensor(result))
    }

    fn cast(&mut self, x: TensorHandle, dtype: BackendDType) -> Result<TensorHandle, String> {
        let x_t = self.get_tensor(x)?;
        let cdtype = backend_dtype_to_candle(dtype)
            .ok_or_else(|| format!("cast: no candle equivalent for dtype {dtype:?}"))?;
        let out = x_t.to_dtype(cdtype).map_err(|e| format!("cast: {e}"))?;
        Ok(self.alloc_tensor(out))
    }

    // ── Lifecycle / inspection ─────────────────────────────────────────

    fn evaluate(
        &mut self,
        group_id: u64,
        outputs: &[TensorHandle],
    ) -> Result<EvaluationReceipt, String> {
        // Candle is eager — evaluation is a no-op.
        let start = Instant::now();

        // Validate handles exist.
        for &h in outputs {
            self.get_tensor(h)?;
        }

        let elapsed = start.elapsed();

        Ok(EvaluationReceipt {
            group_id,
            graph_build_ns: 0,
            submit_ns: 0,
            sync_ns: elapsed.as_nanos() as u64,
            output_count: outputs.len(),
            active_memory_after: 0,
            cache_memory_after: 0,
            observed_substrate: Some("candle-cpu".to_string()),
            eval_calls: 1,
        })
    }

    fn read_f32(&mut self, handle: TensorHandle) -> Result<ReadbackReceipt, String> {
        let start = Instant::now();
        let t = self.get_tensor(handle)?;
        // Flatten then read as f32 vec1.
        let data = if t.shape().rank() == 1 {
            t.to_vec1::<f32>()
                .map_err(|e| format!("read_f32 to_vec1: {e}"))?
        } else {
            t.flatten_all()
                .map_err(|e| format!("read_f32 flatten: {e}"))?
                .to_vec1::<f32>()
                .map_err(|e| format!("read_f32 to_vec1: {e}"))?
        };
        let elapsed = start.elapsed();
        Ok(ReadbackReceipt {
            data,
            forced_eval: false, // candle is eager; already computed
            sync_ns: elapsed.as_nanos() as u64,
            observed_substrate: Some("candle-cpu".to_string()),
        })
    }

    fn shape(&self, handle: TensorHandle) -> Result<Vec<i32>, String> {
        let t = self.get_tensor(handle)?;
        Ok(t.shape().dims().iter().map(|&d| d as i32).collect())
    }

    fn release(&mut self, handle: TensorHandle) -> Result<(), String> {
        let slot = handle.slot as usize;
        let generation = handle.generation;

        if slot >= self.tensors.len() {
            return Err(format!(
                "release: invalid handle (slot={}, gen={})",
                slot, generation
            ));
        }
        let current_gen = self.generations[slot];
        if generation != current_gen {
            return Err(format!(
                "release: stale handle (slot={}, gen={}, current={})",
                slot, generation, current_gen,
            ));
        }
        if self.tensors[slot].is_none() {
            return Err(format!(
                "release: handle already released (slot={}, gen={})",
                slot, generation
            ));
        }

        self.tensors[slot] = None;
        self.generations[slot] += 1;
        self.free_list.push(slot);
        Ok(())
    }

    fn active_memory(&self) -> (u64, u64) {
        // Candle doesn't provide active memory tracking directly.
        (0, 0)
    }

    fn backend_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            can_gpu: false,
            can_cpu: true,
            supports_quantized: true,
            supports_bf16_native: true,
            backend_name: self.name.clone(),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        DType as BackendDType, MatmulOp, QuantizedMatmulOp, RmsNormOp, RoPEOp, TensorBackend,
        TensorHandle,
    };

    /// Helper to create an f32 TensorHandle with a given shape.
    fn make_f32(backend: &mut CandleCpuBackend, data: &[f32], shape: &[i32]) -> TensorHandle {
        backend.create_f32(data, shape).unwrap()
    }

    /// Helper to read back an f32 tensor handle.
    fn read_f32(backend: &mut CandleCpuBackend, h: TensorHandle) -> Vec<f32> {
        backend.read_f32(h).unwrap().data
    }

    #[test]
    fn test_create_and_read_f32() {
        let mut backend = CandleCpuBackend::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let h = backend.create_f32(&data, &[2, 2]).unwrap();
        let output = read_f32(&mut backend, h);
        assert_eq!(output, data);
    }

    #[test]
    fn test_matmul() {
        let mut backend = CandleCpuBackend::new();
        // A = [[1.0, 2.0], [3.0, 4.0]], B = [[5.0, 6.0], [7.0, 8.0]]
        // Result: [[19, 22], [43, 50]]
        let a = make_f32(&mut backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_f32(&mut backend, &[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let op = MatmulOp { m: 2, n: 2, k: 2 };
        let result = backend.matmul(&op, a, b).unwrap();
        let output = read_f32(&mut backend, result);
        assert_eq!(output, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_add_multiply() {
        let mut backend = CandleCpuBackend::new();
        let a = make_f32(&mut backend, &[1.0, 2.0, 3.0], &[3]);
        let b = make_f32(&mut backend, &[4.0, 5.0, 6.0], &[3]);

        let sum = backend.add(a, b).unwrap();
        let sum_data = read_f32(&mut backend, sum);
        assert_eq!(sum_data, vec![5.0, 7.0, 9.0]);

        let c = make_f32(&mut backend, &[2.0, 3.0, 4.0], &[3]);
        let d = make_f32(&mut backend, &[5.0, 6.0, 7.0], &[3]);
        let prod = backend.multiply(c, d).unwrap();
        let prod_data = read_f32(&mut backend, prod);
        assert_eq!(prod_data, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_rms_norm() {
        let mut backend = CandleCpuBackend::new();
        // x = [1.0, 2.0, 3.0, 4.0], weight = [1.0, 1.0, 1.0, 1.0], eps = 1e-6
        // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // normalized = [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386] ≈ [0.365, 0.730, 1.095, 1.461]
        let x = make_f32(&mut backend, &[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let w = make_f32(&mut backend, &[1.0, 1.0, 1.0, 1.0], &[4]);
        let op = RmsNormOp { dim: 1, eps: 1e-6 };
        let result = backend.rms_norm(&op, x, w).unwrap();
        let output = read_f32(&mut backend, result);
        let rms = ((1.0f64 + 4.0 + 9.0 + 16.0) / 4.0).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| (v as f64 / rms) as f32)
            .collect();
        for (got, exp) in output.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_quantized_projection() {
        let mut backend = CandleCpuBackend::new();

        let m = 2usize;
        let k = 8usize;
        let n_out = 4usize;
        let group_size = 4usize;
        let n_groups = k / group_size; // 2
        let bits = 4u8;
        let packed_cols = k / 8; // 1 u32 per row (8 nibbles)

        // Input f32: [m, k]
        let x_data: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let x = backend.create_f32(&x_data, &[m as i32, k as i32]).unwrap();

        // Packed int4 weights: [n_out, packed_cols] as U32.
        // Each u32 stores 8 int4 values. Set all to 1 (value 1).
        let w_u32: Vec<u32> = vec![0x11111111u32; n_out * packed_cols];

        // Weight handle: need to store in weight_tensors slot.
        let w_t = Tensor::from_slice(&w_u32, (n_out, packed_cols), &Device::Cpu).unwrap();
        let w_handle = backend.alloc_weight(w_t);

        // Scales: [n_out, n_groups], all 1.0
        let scales_data: Vec<f32> = vec![1.0; n_out * n_groups];
        let scales = backend
            .create_f32(&scales_data, &[n_out as i32, n_groups as i32])
            .unwrap();

        // Biases: [n_out, n_groups], all 0.0
        let biases_data: Vec<f32> = vec![0.0; n_out * n_groups];
        let biases = backend
            .create_f32(&biases_data, &[n_out as i32, n_groups as i32])
            .unwrap();

        let op = QuantizedMatmulOp {
            m: m as u32,
            n: n_out as u32,
            k: k as u32,
            input_dtype: BackendDType::F32,
            weight_dtype: BackendDType::U32,
            scale_dtype: BackendDType::F32,
            bias_dtype: BackendDType::F32,
            output_dtype: BackendDType::F32,
            group_size: group_size as u32,
            bits,
            transpose: true,
        };

        let result = backend
            .quantized_matmul(&op, x, w_handle, scales, biases)
            .unwrap();
        let output = read_f32(&mut backend, result);

        // With all nibbles = 1, scale=1, bias=0: dequantized weight = all 1.0
        // w_deq = [[1; k]; n_out], so matmul x @ w_deq.T:
        // result[i,j] = sum over k of x[i,k] * 1 = sum of row i
        assert_eq!(output.len(), m * n_out);
        // For x = [[1..8]]: row sums = 36
        for j in 0..n_out {
            let expected_sum: f32 = (1..=8).map(|v| v as f32).sum();
            assert!(
                (output[j] - expected_sum).abs() < 1e-4,
                "output[0][{j}] = {}, expected {expected_sum}",
                output[j]
            );
        }
        for row in 0..m {
            for j in 0..n_out {
                let expected: f32 = (1..=8).map(|v| v as f32).sum();
                let idx = row * n_out + j;
                assert!(
                    (output[idx] - expected).abs() < 1e-4,
                    "output[{row}][{j}] = {}, expected {expected}",
                    output[idx]
                );
            }
        }
    }

    #[test]
    fn test_backend_capabilities() {
        let backend = CandleCpuBackend::new();
        let caps = backend.backend_capabilities();
        assert_eq!(caps.can_cpu, true);
        assert_eq!(caps.can_gpu, false);
        assert_eq!(caps.supports_quantized, true);
        assert_eq!(caps.supports_bf16_native, true);
        assert_eq!(caps.backend_name, "candle-cpu");
    }

    #[test]
    fn test_silu() {
        let mut backend = CandleCpuBackend::new();
        let x = make_f32(&mut backend, &[0.0, 1.0, -1.0, 2.0], &[4]);
        let result = backend.silu(x).unwrap();
        let output = read_f32(&mut backend, result);
        // SiLU(x) = x * sigmoid(x)
        let expected: Vec<f32> = [0.0, 1.0, -1.0, 2.0]
            .iter()
            .map(|&v| v * (1.0 / (1.0 + f32::exp(-v))))
            .collect();
        for (got, exp) in output.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_softmax() {
        let mut backend = CandleCpuBackend::new();
        let x = make_f32(&mut backend, &[0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let result = backend.softmax(x, 1).unwrap();
        let output = read_f32(&mut backend, result);
        // Row 0: softmax([0, 1])
        let e0 = 1.0f64.exp();
        let e1 = 1.0f64.exp();
        let s0 = e0 + e1;
        let e2 = 2.0f64.exp();
        let e3 = 3.0f64.exp();
        let s1 = e2 + e3;
        let expected = vec![
            (e0 / s0) as f32,
            (e1 / s0) as f32,
            (e2 / s1) as f32,
            (e3 / s1) as f32,
        ];
        for (got, exp) in output.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_transpose_reshape() {
        let mut backend = CandleCpuBackend::new();
        let x = make_f32(&mut backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t = backend.transpose(x, &[0, 1]).unwrap();
        let t_data = read_f32(&mut backend, t);
        assert_eq!(t_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let x2 = make_f32(&mut backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = backend.reshape(x2, &[3, 2]).unwrap();
        let r_data = read_f32(&mut backend, r);
        assert_eq!(r_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_index_select() {
        let mut backend = CandleCpuBackend::new();
        // x = [[1,2,3], [4,5,6], [7,8,9]]
        let x = make_f32(
            &mut backend,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
        );
        let result = backend.index_select(x, &[0, 2], 1).unwrap();
        let output = read_f32(&mut backend, result);
        // Result: [[1, 3], [4, 6], [7, 9]]
        assert_eq!(output, vec![1.0, 3.0, 4.0, 6.0, 7.0, 9.0]);
    }

    #[test]
    fn test_release() {
        let mut backend = CandleCpuBackend::new();
        let h = make_f32(&mut backend, &[1.0, 2.0], &[2]);
        backend.release(h).unwrap();
        // Releasing again should fail.
        assert!(backend.release(h).is_err());
        // Using the released handle should fail.
        assert!(backend.read_f32(h).is_err());
    }

    #[test]
    fn test_slice() {
        let mut backend = CandleCpuBackend::new();
        // x = [[1,2,3,4], [5,6,7,8]]
        let x = make_f32(
            &mut backend,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        );
        let result = backend.slice(x, &[0, 1], &[1, 3], &[1, 1]).unwrap();
        let output = read_f32(&mut backend, result);
        assert_eq!(output, vec![2.0, 3.0]);
    }

    #[test]
    fn test_cast() {
        let mut backend = CandleCpuBackend::new();
        let x = make_f32(&mut backend, &[1.5, 2.5, 3.5], &[3]);
        // Cast to U32 (via I64 in candle).
        let result = backend.cast(x, BackendDType::U32).unwrap();
        let output = read_f32(&mut backend, result);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_create_f32_from_bf16_bits() {
        let mut backend = CandleCpuBackend::new();
        // BF16(1.0) = 0x3F80 as u16 bits
        // BF16(2.0) = 0x4000 as u16 bits
        let bf16_bits: Vec<u16> = vec![0x3F80, 0x4000];
        let h = backend.create_f32_from_bf16_bits(&bf16_bits, &[2]).unwrap();
        let output = read_f32(&mut backend, h);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_rope() {
        let mut backend = CandleCpuBackend::new();
        // x = [[1,2,3,4], [5,6,7,8]]  shape [2, 4] -> [seq_len=2, head_dim=4]
        let x = make_f32(
            &mut backend,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        );
        let op = RoPEOp {
            head_dim: 4,
            positions: vec![0, 1],
        };
        let result = backend.rope(&op, x).unwrap();
        let output = read_f32(&mut backend, result);
        // Sanity: output should have same size as input (2, 4)
        assert_eq!(output.len(), 8);
        // At position 0, cos=1, sin=0 -> identity rotation
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
    }
}
