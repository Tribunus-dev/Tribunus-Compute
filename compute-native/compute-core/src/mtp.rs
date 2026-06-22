//! Model-agnostic Multi-Token Prediction (MTP).
//!
//! Predicts N future tokens from a single hidden state using lightweight
//! MLP prediction heads.  Works with ANY transformer model — no architecture
//! changes needed.
//!
//! Each head is a two-layer MLP: hidden -> SiLU -> logits.
//! The heads run in parallel and are verified by SpecHub.

use mlx_rs::ops;
use mlx_rs::{Array, Dtype};
use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Pseudo-RNG (XorShift32) — deterministic, no external deps
// ---------------------------------------------------------------------------

struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn new() -> Self {
        let seed = 0xdead_beeu32.wrapping_add(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u32)
                .unwrap_or(0x7a3b_c9d1),
        );
        Self {
            state: seed.max(1),
        }
    }

    fn gen_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        (self.state >> 9) as f32 * (1.0 / 8388608.0)
    }
}

// ---------------------------------------------------------------------------
// MtpHead
// ---------------------------------------------------------------------------

/// A single MTP prediction head.
///
/// Architecture: `hidden -> Linear(hidden_dim, hidden_dim) -> SiLU ->
///               Linear(hidden_dim, vocab_size) -> logits`
pub struct MtpHead {
    /// [hidden_dim, hidden_dim]
    pub layer1_weight: Array,
    /// [hidden_dim]
    pub layer1_bias: Array,
    /// [hidden_dim, vocab_size]
    pub layer2_weight: Array,
    /// [vocab_size]
    pub layer2_bias: Array,
}

impl MtpHead {
    /// Create a new head with random (Kaiming-uniform) initialization.
    pub fn new(hidden_dim: u32, vocab_size: u32) -> Self {
        let scale1 = (6.0 / (hidden_dim as f64 + hidden_dim as f64)).sqrt() as f32;
        let scale2 = (6.0 / (hidden_dim as f64 + vocab_size as f64)).sqrt() as f32;

        let mut rng = XorShift32::new();
        let w1_data: Vec<f32> = (0..hidden_dim as usize * hidden_dim as usize)
            .map(|_| rng.gen_f32() * 2.0 * scale1 - scale1)
            .collect();
        let w2_data: Vec<f32> = (0..hidden_dim as usize * vocab_size as usize)
            .map(|_| rng.gen_f32() * 2.0 * scale2 - scale2)
            .collect();
        let b1_data: Vec<f32> = (0..hidden_dim as usize)
            .map(|_| rng.gen_f32() * 2.0 * scale1 - scale1)
            .collect();
        let b2_data: Vec<f32> = vec![0.0f32; vocab_size as usize];

        Self {
            layer1_weight: Array::from_slice(&w1_data, &[hidden_dim as i32, hidden_dim as i32]),
            layer1_bias: Array::from_slice(&b1_data, &[hidden_dim as i32]),
            layer2_weight: Array::from_slice(&w2_data, &[hidden_dim as i32, vocab_size as i32]),
            layer2_bias: Array::from_slice(&b2_data, &[vocab_size as i32]),
        }
    }

    /// Predict the logit distribution for this future position.
    ///
    /// `hidden`: [batch, hidden_dim]
    /// Returns:  [batch, vocab_size]
    pub fn forward(&self, hidden: &Array) -> Result<Array, String> {
        // layer1: hidden @ W1 + b1  →  [batch, hidden_dim]
        let h = ops::matmul(hidden, &self.layer1_weight)
            .map_err(|e| format!("mtp head matmul l1: {:?}", e))?;
        let h = h
            .add(&self.layer1_bias)
            .map_err(|e| format!("mtp head add bias l1: {:?}", e))?;
        // SiLU activation
        let h = mlx_rs::nn::silu(&h).map_err(|e| format!("mtp silu: {:?}", e))?;
        // layer2: h @ W2 + b2  →  [batch, vocab_size]
        let logits = ops::matmul(&h, &self.layer2_weight)
            .map_err(|e| format!("mtp head matmul l2: {:?}", e))?;
        logits
            .add(&self.layer2_bias)
            .map_err(|e| format!("mtp head add bias l2: {:?}", e))
    }
}

// ---------------------------------------------------------------------------
// MtpModule
// ---------------------------------------------------------------------------

/// Multi-Token Prediction module — model-agnostic.
///
/// Takes the last hidden state from any transformer model and predicts
/// N future tokens through lightweight MLP heads.  Predictions are
/// verified by the target model via SpecHub.
pub struct MtpModule {
    /// Number of future tokens to predict (default: 3).
    pub n_future: u32,
    /// Hidden dimension of the base model.
    pub hidden_dim: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Prediction heads: [hidden_dim -> hidden_dim -> vocab_size] each.
    pub heads: Vec<MtpHead>,
}

impl MtpModule {
    /// Create a new MTP module with Kaiming-uniform initialized heads.
    pub fn new(n_future: u32, hidden_dim: u32, vocab_size: u32) -> Self {
        let heads: Vec<MtpHead> = (0..n_future)
            .map(|_| MtpHead::new(hidden_dim, vocab_size))
            .collect();
        Self {
            n_future,
            hidden_dim,
            vocab_size,
            heads,
        }
    }

    /// Run all N heads in parallel on the given hidden state.
    ///
    /// `hidden`: [1, hidden_dim] — single hidden state vector.
    /// Returns N logit vectors, one per future position, each [1, vocab_size].
    pub fn predict_all(&self, hidden: &Array) -> Result<Vec<Array>, String> {
        self.heads
            .iter()
            .map(|head| head.forward(hidden))
            .collect()
    }

    /// Sample N tokens from the predicted distributions.
    ///
    /// `hidden`: [1, hidden_dim] — single step's hidden state.
    /// `temp`: sampling temperature (> 0.0; 1.0 = standard softmax).
    /// Returns N sampled token IDs.
    pub fn sample_tokens(&self, hidden: &Array, temp: f64) -> Result<Vec<u32>, String> {
        let predictions = self.predict_all(hidden)?;
        let mut rng = XorShift32::new();
        let mut tokens = Vec::with_capacity(self.n_future as usize);

        for logits in &predictions {
            let token = sample_from_logits(logits, temp, &mut rng)?;
            tokens.push(token);
        }

        Ok(tokens)
    }

    /// Run a training step: predict future tokens and compute MTP loss.
    ///
    /// `hidden`: [1, hidden_dim] — the hidden state at position t.
    /// `targets`: slice of N token IDs `[t+1, t+2, ..., t+N]` (N = n_future).
    /// `lr`: learning rate for SGD update.
    /// Returns the average cross-entropy loss across all N heads.
    pub fn train_step(
        &mut self,
        hidden: &Array,
        targets: &[u32],
        lr: f64,
    ) -> Result<f64, String> {
        if targets.len() < self.n_future as usize {
            return Err(format!(
                "train_step: need {} targets, got {}",
                self.n_future,
                targets.len()
            ));
        }

        let lr_f32 = lr as f32;
        let mut total_loss = 0.0f64;
        let mut new_heads: Vec<MtpHead> = Vec::with_capacity(self.heads.len());

        for i in 0..self.heads.len() {
            let head = &self.heads[i];
            // Forward: get logits for this head
            let logits = head.forward(hidden)?; // [1, vocab_size]

            // Cross-entropy loss for a single target token
            let target = targets[i];
            if target >= self.vocab_size {
                return Err(format!(
                    "train_step: target {} >= vocab_size {}",
                    target, self.vocab_size
                ));
            }

            // log_softmax: logits - logsumexp(logits)
            let lse = ops::logsumexp_axis(&logits, -1, Some(true))
                .map_err(|e| format!("logsumexp: {:?}", e))?;
            let log_softmax = ops::subtract(&logits, &lse)
                .map_err(|e| format!("subtract log_softmax: {:?}", e))?;

            // Extract the log-probability of the target token
            let target_arr = Array::from_slice(&[target as i32], &[1, 1]);
            let nll = log_softmax
                .take_along_axis(&target_arr, -1)
                .map_err(|e| format!("take_along_axis: {:?}", e))?;
            // nll is [1, 1]; extract scalar
            nll.eval().map_err(|e| format!("nll eval: {:?}", e))?;
            let nll_val = -nll.as_slice::<f32>()[0] as f64;
            total_loss += nll_val;

            // --- Manual gradient computation (SGD) ---
            // We need: dL/dW2, dL/db2, dL/dW1, dL/db1
            //
            // Let h = SiLU(x @ W1 + b1) where x = hidden
            // Let z = h @ W2 + b2
            // L = -log_softmax(z)[target]
            //
            // dL/dz = softmax(z) - one_hot(target)  (gradient of cross-entropy w.r.t. logits)
            //
            // We'll compute this directly.

            // softmax(z)
            let softmax_z = ops::softmax_axes(&logits, &[-1], None::<bool>)
                .map_err(|e| format!("softmax: {:?}", e))?;

            // Create one-hot for target
            let mut one_hot_data = vec![0.0f32; self.vocab_size as usize];
            one_hot_data[target as usize] = 1.0;
            let one_hot = Array::from_slice(&one_hot_data, &[1, self.vocab_size as i32]);

            // dz = softmax - one_hot  (gradient of CE w.r.t. logits)  [1, vocab_size]
            let dz = ops::subtract(&softmax_z, &one_hot)
                .map_err(|e| format!("dz subtract: {:?}", e))?;

            // dL/dW2 = h^T @ dz  →  [hidden_dim, vocab_size]
            let h = ops::matmul(hidden, &head.layer1_weight)
                .map_err(|e| format!("grad matmul l1: {:?}", e))?;
            let h = h
                .add(&head.layer1_bias)
                .map_err(|e| format!("grad add bias l1: {:?}", e))?;
            let h = mlx_rs::nn::silu(&h).map_err(|e| format!("grad silu: {:?}", e))?;

            // Need h^T: [hidden_dim, 1]
            let h_t = ops::transpose_axes(&h, &[1, 0])
                .map_err(|e| format!("h transpose: {:?}", e))?;
            let d_w2 = ops::matmul(&h_t, &dz)
                .map_err(|e| format!("grad dW2 matmul: {:?}", e))?;

            // dL/db2 = dz summed over batch dim (sum over axis 0) → [vocab_size]
            let d_b2 = ops::sum_axis(&dz, 0, None::<bool>)
                .map_err(|e| format!("grad db2 sum: {:?}", e))?;

            // Backprop through layer2: dh = dz @ W2^T  →  [1, hidden_dim]
            let w2_t = ops::transpose_axes(&head.layer2_weight, &[1, 0])
                .map_err(|e| format!("w2 transpose: {:?}", e))?;
            let dh = ops::matmul(&dz, &w2_t)
                .map_err(|e| format!("grad dh matmul: {:?}", e))?;

            // Backprop through SiLU: ds = dh * sigmoid(h) * (1 + h * (1 - sigmoid(h)))
            //   where sigmoid = 1/(1 + exp(-h))
            // SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            // Or more simply: ds = dh * (h > 0 ? 1/(1+exp(-h)) * (1 + h * (1 - 1/(1+exp(-h)))) : ...)
            // Let's compute: sig = sigmoid(h); silu_deriv = sig * (1 + h * (1 - sig))
            let neg_h = ops::multiply(&h, &Array::from_f32(-1.0))
                .map_err(|e| format!("neg h: {:?}", e))?;
            let exp_neg_h = ops::exp(&neg_h)
                .map_err(|e| format!("exp neg h: {:?}", e))?;
            let one_arr = Array::from_f32(1.0);
            let denom = ops::add(&exp_neg_h, &one_arr)
                .map_err(|e| format!("denom add: {:?}", e))?;
            let sig = ops::divide(&one_arr, &denom)
                .map_err(|e| format!("sigmoid div: {:?}", e))?;
            // silu_deriv = sig * (1 + h * (1 - sig))
            let one_minus_sig = ops::subtract(&one_arr, &sig)
                .map_err(|e| format!("1-sig: {:?}", e))?;
            let h_times_one_minus_sig = ops::multiply(&h, &one_minus_sig)
                .map_err(|e| format!("h*(1-sig): {:?}", e))?;
            let inner = ops::add(&one_arr, &h_times_one_minus_sig)
                .map_err(|e| format!("inner add: {:?}", e))?;
            let silu_deriv = ops::multiply(&sig, &inner)
                .map_err(|e| format!("silu deriv: {:?}", e))?;
            let ds = ops::multiply(&dh, &silu_deriv)
                .map_err(|e| format!("ds multiply: {:?}", e))?;

            // dL/dW1 = hidden^T @ ds  →  [hidden_dim, hidden_dim]
            let hidden_t = ops::transpose_axes(hidden, &[1, 0])
                .map_err(|e| format!("hidden transpose: {:?}", e))?;
            let d_w1 = ops::matmul(&hidden_t, &ds)
                .map_err(|e| format!("grad dW1 matmul: {:?}", e))?;

            // dL/db1 = ds summed over batch dim → [hidden_dim]
            let d_b1 = ops::sum_axis(&ds, 0, None::<bool>)
                .map_err(|e| format!("grad db1 sum: {:?}", e))?;

            // --- SGD update ---
            let lr_arr = Array::from_f32(lr_f32);

            // W2 -= lr * dW2,  b2 -= lr * db2
            let w2_step = ops::multiply(&d_w2, &lr_arr)
                .map_err(|e| format!("w2 step: {:?}", e))?;
            let new_w2 = ops::subtract(&head.layer2_weight, &w2_step)
                .map_err(|e| format!("w2 update: {:?}", e))?;
            let b2_step = ops::multiply(&d_b2, &lr_arr)
                .map_err(|e| format!("b2 step: {:?}", e))?;
            let new_b2 = ops::subtract(&head.layer2_bias, &b2_step)
                .map_err(|e| format!("b2 update: {:?}", e))?;

            // W1 -= lr * dW1,  b1 -= lr * db1
            let w1_step = ops::multiply(&d_w1, &lr_arr)
                .map_err(|e| format!("w1 step: {:?}", e))?;
            let new_w1 = ops::subtract(&head.layer1_weight, &w1_step)
                .map_err(|e| format!("w1 update: {:?}", e))?;
            let b1_step = ops::multiply(&d_b1, &lr_arr)
                .map_err(|e| format!("b1 step: {:?}", e))?;
            let new_b1 = ops::subtract(&head.layer1_bias, &b1_step)
                .map_err(|e| format!("b1 update: {:?}", e))?;

            // Write back — we cannot mutate in the borrow, so reassign the head.
            // Collect into a temporary Vec then swap the field.
            let new_head = MtpHead {
                layer1_weight: new_w1,
                layer1_bias: new_b1,
                layer2_weight: new_w2,
                layer2_bias: new_b2,
            };
            new_heads.push(new_head);
        }

        // Write all updated heads back.
        self.heads = new_heads;

        let avg_loss = total_loss / self.n_future as f64;
        Ok(avg_loss)
    }

/// Reinterpret f32 bytes as u8 bytes (same underlying memory).
fn f32_slice_to_u8_slice(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

    /// Save weights to safetensors.
    pub fn save(&self, path: &str) -> Result<(), String> {
        use safetensors::tensor::{serialize_to_file, TensorView};

        let mut tensors: Vec<(String, TensorView)> = Vec::new();
        for (i, head) in self.heads.iter().enumerate() {
            let prefix = format!("head_{}", i);

            let w1_slice = head
                .layer1_weight
                .as_slice::<f32>();
            let w1_bytes = f32_slice_to_u8_slice(w1_slice);
            let w1_shape: Vec<usize> = head
                .layer1_weight
                .shape()
                .iter()
                .map(|&d| d as usize)
                .collect();
            tensors.push((
                format!("{}.layer1_weight", prefix),
                TensorView::new(safetensors::Dtype::F32, w1_shape.clone(), w1_bytes)
                    .map_err(|e| format!("w1 tensor view: {:?}", e))?,
            ));

            let b1_slice = head
                .layer1_bias
                .as_slice::<f32>();
            let b1_bytes = f32_slice_to_u8_slice(b1_slice);
            let b1_shape: Vec<usize> = head
                .layer1_bias
                .shape()
                .iter()
                .map(|&d| d as usize)
                .collect();
            tensors.push((
                format!("{}.layer1_bias", prefix),
                TensorView::new(safetensors::Dtype::F32, b1_shape.clone(), b1_bytes)
                    .map_err(|e| format!("b1 tensor view: {:?}", e))?,
            ));

            let w2_slice = head
                .layer2_weight
                .as_slice::<f32>();
            let w2_bytes = f32_slice_to_u8_slice(w2_slice);
            let w2_shape: Vec<usize> = head
                .layer2_weight
                .shape()
                .iter()
                .map(|&d| d as usize)
                .collect();
            tensors.push((
                format!("{}.layer2_weight", prefix),
                TensorView::new(safetensors::Dtype::F32, w2_shape.clone(), w2_bytes)
                    .map_err(|e| format!("w2 tensor view: {:?}", e))?,
            ));

            let b2_slice = head
                .layer2_bias
                .as_slice::<f32>();
            let b2_bytes = f32_slice_to_u8_slice(b2_slice);
            let b2_shape: Vec<usize> = head
                .layer2_bias
                .shape()
                .iter()
                .map(|&d| d as usize)
                .collect();
            tensors.push((
                format!("{}.layer2_bias", prefix),
                TensorView::new(safetensors::Dtype::F32, b2_shape.clone(), b2_bytes)
                    .map_err(|e| format!("b2 tensor view: {:?}", e))?,
            ));
        }

        serialize_to_file(tensors, &None::<std::collections::HashMap<String, String>>, path.as_ref())
            .map_err(|e| format!("save safetensors: {:?}", e))
    }

    /// Load weights from safetensors.
    pub fn load(path: &str) -> Result<Self, String> {
        use safetensors::tensor::SafeTensors;

        let data = std::fs::read(path).map_err(|e| format!("read safetensors: {:?}", e))?;
        let st = SafeTensors::deserialize(&data)
            .map_err(|e| format!("deserialize safetensors: {:?}", e))?;

        // Determine n_future by counting head_ prefixes in the tensor names
        let mut max_head_idx: i32 = -1;
        for name in st.names() {
            if let Some(rest) = name.strip_prefix("head_") {
                if let Some(idx_str) = rest.split('.').next() {
                    if let Ok(idx) = idx_str.parse::<i32>() {
                        max_head_idx = max_head_idx.max(idx);
                    }
                }
            }
        }

        if max_head_idx < 0 {
            return Err("no MTP heads found in safetensors".to_string());
        }

        let n_future = (max_head_idx + 1) as u32;
        let mut heads = Vec::with_capacity(n_future as usize);

        for i in 0..n_future {
            let prefix = format!("head_{}", i);

            let w1_tv = st
                .tensor(&format!("{}.layer1_weight", prefix))
                .map_err(|e| format!("missing {}.layer1_weight: {:?}", prefix, e))?;
            let b1_tv = st
                .tensor(&format!("{}.layer1_bias", prefix))
                .map_err(|e| format!("missing {}.layer1_bias: {:?}", prefix, e))?;
            let w2_tv = st
                .tensor(&format!("{}.layer2_weight", prefix))
                .map_err(|e| format!("missing {}.layer2_weight: {:?}", prefix, e))?;
            let b2_tv = st
                .tensor(&format!("{}.layer2_bias", prefix))
                .map_err(|e| format!("missing {}.layer2_bias: {:?}", prefix, e))?;

            let layer1_weight = tensor_view_to_array(&w1_tv);
            let layer1_bias = tensor_view_to_array(&b1_tv);
            let layer2_weight = tensor_view_to_array(&w2_tv);
            let layer2_bias = tensor_view_to_array(&b2_tv);

            let hidden_dim = layer1_weight.shape()[0] as u32;
            let vocab_size = layer2_weight.shape()[1] as u32;

            heads.push(MtpHead {
                layer1_weight,
                layer1_bias,
                layer2_weight,
                layer2_bias,
            });

            if i == 0 {
                // Store metadata from first head
                let _ = hidden_dim;
                let _ = vocab_size;
            }
        }

        // Infer hidden_dim and vocab_size from the first head's weights
        let hidden_dim = heads[0].layer1_weight.shape()[0] as u32;
        let vocab_size = heads[0].layer2_weight.shape()[1] as u32;

        Ok(Self {
            n_future,
            hidden_dim,
            vocab_size,
            heads,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sample a token from logits with temperature scaling.
fn sample_from_logits(logits: &Array, temp: f64, rng: &mut XorShift32) -> Result<u32, String> {
    let temp_inv = 1.0 / temp.max(1e-8);

    // Compute softmax probabilities
    let scaled = if (temp - 1.0).abs() > 1e-8 {
        ops::multiply(logits, &Array::from_f32(temp_inv as f32))
            .map_err(|e| format!("scale logits: {:?}", e))?
    } else {
        logits.clone()
    };

    let probs = ops::softmax_axes(&scaled, &[-1], None::<bool>)
        .map_err(|e| format!("softmax: {:?}", e))?;

    // Materialize and sample
    probs.eval().map_err(|e| format!("probs eval: {:?}", e))?;
    let probs_slice = probs
        .as_slice::<f32>();

    let vocab_size = probs_slice.len();
    let r = rng.gen_f32();
    let mut cum = 0.0f32;
    for (i, &p) in probs_slice.iter().enumerate() {
        cum += p;
        if r < cum {
            return Ok(i as u32);
        }
    }
    // Fallback (shouldn't happen if probabilities sum to 1)
    Ok((vocab_size - 1) as u32)
}

/// Convert a safetensors TensorView to an mlx_rs Array.
///
/// Convert a &[f32] to &[u8] for safetensors serialization.
fn f32_slice_to_u8_slice(data: &[f32]) -> &[u8] {
    let byte_len = data.len() * 4;
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) }
}

/// Convert a safetensors TensorView to an mlx_rs Array.
fn tensor_view_to_array(tv: &safetensors::tensor::TensorView) -> Array {
    let data = tv.data();
    let shape: Vec<i32> = tv.shape().iter().map(|&d| d as i32).collect();
    unsafe { Array::from_raw_data(data.as_ptr() as *const c_void, &shape, Dtype::Float32) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_head_forward() {
        let hidden_dim = 64;
        let vocab_size = 256;
        let head = MtpHead::new(hidden_dim, vocab_size);
        let hidden = Array::from_slice(&vec![0.5f32; hidden_dim as usize], &[1, hidden_dim as i32]);
        let logits = head.forward(&hidden).unwrap();
        assert_eq!(logits.shape(), &[1, vocab_size as i32]);
    }

    #[test]
    fn test_mtp_predict_all() {
        let hidden_dim = 64;
        let vocab_size = 256;
        let n_future = 3;
        let mtp = MtpModule::new(n_future, hidden_dim, vocab_size);
        let hidden = Array::from_slice(&vec![0.5f32; hidden_dim as usize], &[1, hidden_dim as i32]);
        let predictions = mtp.predict_all(&hidden).unwrap();
        assert_eq!(predictions.len(), n_future as usize);
        for p in &predictions {
            assert_eq!(p.shape(), &[1, vocab_size as i32]);
        }
    }

    #[test]
    fn test_mtp_sample_tokens() {
        let hidden_dim = 64;
        let vocab_size = 256;
        let n_future = 3;
        let mtp = MtpModule::new(n_future, hidden_dim, vocab_size);
        let hidden = Array::from_slice(&vec![0.5f32; hidden_dim as usize], &[1, hidden_dim as i32]);
        let tokens = mtp.sample_tokens(&hidden, 1.0).unwrap();
        assert_eq!(tokens.len(), n_future as usize);
        for &t in &tokens {
            assert!(t < vocab_size);
        }
    }

    #[test]
    fn test_mtp_train_step() {
        let hidden_dim = 32;
        let vocab_size = 128;
        let n_future = 2;
        let mut mtp = MtpModule::new(n_future, hidden_dim, vocab_size);
        let hidden = Array::from_slice(&vec![0.5f32; hidden_dim as usize], &[1, hidden_dim as i32]);
        let targets = vec![10u32, 42u32];

        let loss = mtp.train_step(&hidden, &targets, 1e-3).unwrap();
        assert!(loss.is_finite());
        assert!(loss > 0.0);

        // Second step should reduce loss
        let loss2 = mtp.train_step(&hidden, &targets, 1e-3).unwrap();
        assert!(loss2.is_finite());
    }

    #[test]
    fn test_mtp_save_load() {
        let hidden_dim = 32;
        let vocab_size = 128;
        let n_future = 2;
        let mtp = MtpModule::new(n_future, hidden_dim, vocab_size);
        let dir = std::env::temp_dir();
        let path = dir.join("test_mtp.safetensors");
        let path_str = path.to_str().unwrap().to_string();

        mtp.save(&path_str).unwrap();
        assert!(path.exists());

        let loaded = MtpModule::load(&path_str).unwrap();
        assert_eq!(loaded.n_future, n_future);
        assert_eq!(loaded.hidden_dim, hidden_dim);
        assert_eq!(loaded.vocab_size, vocab_size);
        assert_eq!(loaded.heads.len(), n_future as usize);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }
}
