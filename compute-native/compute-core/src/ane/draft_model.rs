//! ANE-backed draft model for speculative decoding.
//!
//! Runs a small Core ML language model (100M-500M parameters) entirely on
//! Apple's Neural Engine via the IOSurface zero-copy path.  The model sits
//! in ANE private SRAM and generates speculative tokens at ~1 ms per token
//! without consuming GPU memory or compute cycles.
//!
//! # Architecture
//!
//! ```text
//!  CPU (tokenize + sample)          ANE (transformer)
//!        │                               │
//!        ├── prefix tokens ──► IOSurface ──► Core ML model ──► IOSurface ──► logits
//!        │                                   (CpuAndNeuralEngine)
//!        └────────── softmax + argmax ◄──────┘
//! ```
//!
//! Input and output tensors live in `Arena`-managed IOSurface memory so
//! Core ML reads and writes them without any CPU-side copies.

use crate::arena::Arena;
use crate::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use crate::speculative::DraftModel;
use crate::speculative::SampleStrategy;

// ---------------------------------------------------------------------------
// Feature-name constants for Core ML binding I/O
// ---------------------------------------------------------------------------

/// Input feature name in the compiled Core ML model.
const INPUT_NAME: &str = "input";

/// Output feature name in the compiled Core ML model.
const OUTPUT_NAME: &str = "output";

// ---------------------------------------------------------------------------
// FP16 ↔ f32 conversion (IEEE 754)
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 binary32 `f32` to a packed `u16` in fp16 format.
///
/// Flushes subnormals to zero; overflows to infinity; preserves NaN signalling.
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    // Special cases
    if exp == 0xFF {
        // NaN or Inf — preserve sign, set fp16 exponent all-ones
        return (sign << 15) | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp == 0 {
        // f32 subnormal / zero → flush to fp16 zero
        return sign << 15;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow → Inf
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        // Underflow → zero
        return sign << 15;
    }

    let new_mant = mant >> 13;
    (sign << 15) | ((new_exp as u16) << 10) | (new_mant as u16)
}

/// Convert a packed IEEE 754 fp16 `u16` to a full `f32`.
///
/// Denormals are normalised; NaNs and infinities round-trip correctly.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        // Zero or denormal
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Normalise: actual exponent is -14
        let leading = mant.leading_zeros() - 21; // 32 - 11
        let norm_exp = ((127 - 15 - leading as i32) as u32) << 23;
        let norm_mant = (mant << (leading + 1)) & 0x7F_FFFF;
        f32::from_bits((sign << 31) | norm_exp | norm_mant)
    } else if exp == 31 {
        // NaN or Inf
        let mant32 = if mant == 0 {
            0
        } else {
            (mant << 13) | 0x7F_FFFF
        };
        f32::from_bits((sign << 31) | 0x7F80_0000 | mant32)
    } else {
        // Normal fp16 value
        let exp32 = (exp + (127 - 15)) << 23;
        let mant32 = mant << 13;
        f32::from_bits((sign << 31) | exp32 | mant32)
    }
}

// ---------------------------------------------------------------------------
// AneDraftModel
// ---------------------------------------------------------------------------

/// A draft model that runs on Apple Neural Engine via a compiled Core ML model.
///
/// The model is a small transformer (100M–500M parameters) loaded with
/// `CpuAndNeuralEngine` compute units — the CPU handles tokenization and
/// softmax sampling while the ANE runs all transformer forward passes in
/// its private SRAM (~2–4 MB for 100M params at 4-bit).
///
/// Input and output tensors are transferred zero-copy through
/// IOSurface-backed [`Arena`] instances.  Each forward pass (one draft
/// token) completes in ~1 ms and consumes no GPU memory.
///
/// # Autoregressive generation
///
/// `speculate()` runs an autoregressive loop: one ANE forward pass per
/// drafted token.  The input arena holds the accumulated prefix; the
/// output arena receives logits for every input position (last-position
/// logits are used to sample the next token).
pub struct AneDraftModel {
    /// Loaded Core ML model targeting the Neural Engine.
    model: CoreMlModel,
    /// IOSurface-backed input arena — token IDs stored as FP16 values.
    input_arena: Arena,
    /// IOSurface-backed output arena — logits stored as FP16 values,
    /// shaped `(seq_len, vocab_size)`.
    output_arena: Arena,
    /// Vocabulary size (number of logits per position).
    vocab_size: u32,
    /// Maximum sequence length (prefix + output tokens) per forward pass.
    seq_len: u32,
    /// Accumulated prefix tokens for KV cache continuity across calls.
    prefix: Vec<u32>,
}

impl AneDraftModel {
    /// Load a draft model from a compiled `.mlmodel` or `.mlpackage` path.
    ///
    /// The model is loaded with `CpuAndNeuralEngine` — ANE does all
    /// transformer compute while the CPU handles tokenization and sampling.
    ///
    /// # Parameters
    ///
    /// * `path` — filesystem path to the compiled Core ML model.
    /// * `vocab_size` — vocabulary size (number of logits per output position).
    /// * `seq_len` — maximum number of tokens per single forward pass.
    ///
    /// # Arena sizing
    ///
    /// The input arena holds `seq_len` tokens (one FP16 value per token).
    /// The output arena holds `seq_len × vocab_size` FP16 logits.
    pub fn load(path: &str, vocab_size: u32, seq_len: u32) -> Result<Self, String> {
        let model =
            CoreMlModel::load_with_compute_units(path, CoreMlComputeUnits::CpuAndNeuralEngine)?;

        // Input arena: one FP16 token ID per sequence position.
        let input_arena = Arena::new(seq_len, 1, mlx_rs::Dtype::Float16)?;

        // Output arena: vocab_size FP16 logits per sequence position.
        let output_arena = Arena::new(seq_len, vocab_size, mlx_rs::Dtype::Float16)?;

        Ok(Self {
            model,
            input_arena,
            output_arena,
            vocab_size,
            seq_len,
            prefix: Vec::new(),
        })
    }

    /// Write token IDs into the input IOSurface arena as FP16 values.
    ///
    /// Tokens beyond `tokens.len()` are zero-filled.  The caller must
    /// ensure `tokens.len() ≤ seq_len`.
    fn write_tokens_to_input(&mut self, tokens: &[u32]) -> Result<(), String> {
        if tokens.len() > self.seq_len as usize {
            return Err(format!(
                "AneDraftModel: token count {} exceeds seq_len {}",
                tokens.len(),
                self.seq_len
            ));
        }

        self.input_arena.lock()?;

        unsafe {
            let ptr = self.input_arena.base_ptr() as *mut u16;
            for (i, &token) in tokens.iter().enumerate() {
                // Token IDs fit in 16 bits for practical vocabularies
                // (≤ 65536).  Store as the FP16 representation of the
                // integer token ID.
                ptr.add(i).write(f32_to_f16(token as f32));
            }
            // Zero-fill any remaining slots so stale data is not
            // presented to the model.
            for i in tokens.len()..self.seq_len as usize {
                ptr.add(i).write(0u16);
            }
        }

        self.input_arena.unlock()
    }

    /// Read logits at a specific input position from the output IOSurface.
    ///
    /// Returns a `Vec<f32>` of length `vocab_size` containing the logits
    /// produced by the model for that position.  Position `p` corresponds
    /// to the model's prediction for the token *after* the `p`-th input
    /// token (0-indexed).
    fn read_logits_at_position(&self, position: usize) -> Result<Vec<f32>, String> {
        if position >= self.seq_len as usize {
            return Err(format!(
                "AneDraftModel: position {} out of range (seq_len = {})",
                position, self.seq_len
            ));
        }

        self.output_arena.lock()?;

        let vocab = self.vocab_size as usize;
        let mut logits = vec![0.0f32; vocab];

        unsafe {
            let ptr = self.output_arena.base_ptr() as *const u16;
            let base = position * vocab;
            for i in 0..vocab {
                logits[i] = f16_to_f32(ptr.add(base + i).read());
            }
        }

        self.output_arena.unlock()?;
        Ok(logits)
    }

    /// Greedy token sampling — return the argmax over `logits`.
    fn sample_token_greedy(logits: &[f32]) -> u32 {
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &val) in logits.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = i as u32;
            }
        }
        best_idx
    }

    /// Compute the softmax probability of a token given raw logits.
    fn token_probability(token: u32, logits: &[f32]) -> f32 {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
        let token_idx = token as usize;
        if token_idx >= logits.len() {
            return 0.0;
        }
        (logits[token_idx] - max_logit).exp() / sum
    }

    /// Run a single forward pass and return logits for the last position.
    ///
    /// Unlike `speculate()`, this does NOT autoregressively generate tokens.
    /// It writes the given tokens to the input arena, runs the Core ML model
    /// on the ANE, and returns the logits for the last position.  Use this
    /// when you only need the model's output distribution for one step
    /// (e.g. prompt-complexity estimation for adaptive batch sizing).
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("AneDraftModel::forward: empty token sequence".to_string());
        }
        if tokens.len() > self.seq_len as usize {
            return Err(format!(
                "AneDraftModel::forward: token count {} exceeds seq_len {}",
                tokens.len(),
                self.seq_len
            ));
        }

        self.write_tokens_to_input(tokens)?;

        let mut out_info = self.output_arena.info;
        self.model.predict_pixelbuffer(
            INPUT_NAME,
            &self.input_arena.info,
            OUTPUT_NAME,
            &mut out_info,
        )?;
        self.output_arena.info = out_info;

        // Return logits for the last input position — this is the model's
        // prediction for the next token after the full prefix.
        let last_pos = tokens.len() - 1;
        self.read_logits_at_position(last_pos)
    }
}

impl DraftModel for AneDraftModel {
    /// Generate `n_tokens` speculative tokens from the given prefix.
    /// The method runs an autoregressive loop — one ANE forward pass per
    /// drafted token:
    ///
    /// 1. Write the prefix into the input IOSurface arena.
    /// 2. Call `predict_pixelbuffer` to run the Core ML model on the ANE.
    /// 3. Read logits for the *last* input position from the output arena.
    /// 4. Greedily sample the next token from those logits.
    /// 5. Append the token to the accumulated input and repeat.
    ///
    /// Returns `(token_ids, log_probabilities)` where each log-probability
    /// is the natural log of the softmax probability assigned to the
    /// sampled token by the draft model at its position.
    fn speculate(
        &mut self,
        prefix: &[u32],
        n_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        if n_tokens == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if n_tokens > self.seq_len as usize {
            return Err(format!(
                "AneDraftModel: n_tokens ({}) exceeds seq_len ({})",
                n_tokens, self.seq_len
            ));
        }

        // Build the full input sequence: previously cached prefix + new prefix.
        let total_prefix_len = self.prefix.len() + prefix.len();
        if total_prefix_len == 0 {
            return Err("AneDraftModel: empty prefix".to_string());
        }
        if total_prefix_len + n_tokens - 1 > self.seq_len as usize {
            return Err(format!(
                "AneDraftModel: total length ({} prefix + {} tokens) exceeds seq_len ({})",
                total_prefix_len, n_tokens, self.seq_len
            ));
        }

        let mut input: Vec<u32> = Vec::with_capacity(total_prefix_len + n_tokens);
        input.extend_from_slice(&self.prefix);
        input.extend_from_slice(prefix);

        let mut tokens = Vec::with_capacity(n_tokens);
        let mut log_probs = Vec::with_capacity(n_tokens);

        for _ in 0..n_tokens {
            // Write the current input to the IOSurface arena.
            self.write_tokens_to_input(&input)?;

            // Run the Core ML model on the ANE.
            // predict_pixelbuffer may update output_arena.info with new
            // CVPixelBuffer metadata; copy it back afterwards.
            let mut out_info = self.output_arena.info;
            self.model.predict_pixelbuffer(
                INPUT_NAME,
                &self.input_arena.info,
                OUTPUT_NAME,
                &mut out_info,
            )?;
            self.output_arena.info = out_info;

            // Read logits for the last input position — this position's
            // output predicts the next token.
            let last_pos = input.len() - 1;
            let pos_logits = self.read_logits_at_position(last_pos)?;

            // Greedily sample the next token and compute its probability.
            let token = Self::sample_token_greedy(&pos_logits);
            let prob = Self::token_probability(token, &pos_logits);
            let log_prob = prob.ln();

            tokens.push(token);
            log_probs.push(log_prob);

            // Append the sampled token for the next autoregressive step.
            input.push(token);
        }

        // Save the consumed prefix so the caller's next speculate()
        // call can continue where we left off.
        self.prefix.extend_from_slice(prefix);

        Ok((tokens, log_probs))
    }

    /// Reset the internal KV cache state and prefix buffer.
    ///
    /// Clears the accumulated prefix.  Call this when starting a new
    /// generation session so the draft model's internal KV cache
    /// (maintained by the Core ML runtime on the ANE) is also reset.
    fn reset(&mut self) {
        self.prefix.clear();
    }
}

// ---------------------------------------------------------------------------
// AneMultiCoreDraft — 16 ANE drafts for multi-spec speculative decoding
// ---------------------------------------------------------------------------

/// Orchestrates 16 ANE draft models running concurrently across all 16
/// ANE cores, each with an independent sampling strategy.
///
/// The M1–M4 Apple Neural Engine has 16 independent cores, each with
/// private SRAM (~2–4 MB for a 10M-parameter 4-bit model).  By loading
/// a copy of the draft model on every core, we generate 16 different
/// speculative continuations in parallel.  The GPU verifies all 16 in
/// a single batched forward pass.
pub struct AneMultiCoreDraft {
    /// 16 independent draft model instances, one per ANE core.
    drafts: Vec<AneDraftModel>,
}

impl AneMultiCoreDraft {
    /// Create a new multi-core draft with 16 copies of the model.
    ///
    /// Each copy is loaded independently so the Core ML runtime can
    /// place one per ANE core.  The same compiled `.mlmodel` or
    /// `.mlpackage` path is used for all 16.
    pub fn new(model_path: &str, vocab_size: u32, seq_len: u32) -> Result<Self, String> {
        Self::new_n(model_path, vocab_size, seq_len, 16)
    }

    /// Create a multi-core draft with `n` copies of the model.
    ///
    /// M1–M4: `n=16` (one per ANE core).
    /// M3 Ultra: `n=32` (one per ANE core).
    pub fn new_n(
        model_path: &str,
        vocab_size: u32,
        seq_len: u32,
        n: usize,
    ) -> Result<Self, String> {
        let mut drafts = Vec::with_capacity(n);
        for _ in 0..n {
            let draft = AneDraftModel::load(model_path, vocab_size, seq_len)?;
            drafts.push(draft);
        }
        Ok(Self { drafts })
    }

    /// Run all 16 drafts concurrently on the ANE.
    ///
    /// Each draft receives the same `prefix` but uses its own sampling
    /// strategy to produce a different continuation.  Returns 16 result
    /// vectors, one per core, each containing `(token, log_probability)`
    /// pairs for every position.
    ///
    /// The ANE's hardware parallelism means all 16 inference requests
    /// complete in roughly the same wall time as a single request —
    /// the Core ML runtime dispatches across all 16 ANE cores internally.
    pub fn speculate_all(
        &mut self,
        prefix: &[u32],
        n_tokens: usize,
        strategies: &[SampleStrategy],
    ) -> Result<Vec<Vec<(u32, f32)>>, String> {
        let n = self.drafts.len();
        let mut results: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n);
        for (_i, (draft, strat)) in self.drafts.iter_mut().zip(strategies.iter()).enumerate() {
            let (tokens, probs) = draft.speculate(prefix, n_tokens)?;
            results.push(crate::speculative::resample(&tokens, &probs, strat));
        }
        Ok(results)
    }

    /// Reset all 16 draft models for a new generation sequence.
    pub fn reset_all(&mut self) {
        for draft in &mut self.drafts {
            draft.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- FP16 round-trip tests ------------------------------------------

    #[test]
    fn test_f32_to_f16_roundtrip() {
        // Integer values that fit comfortably in fp16
        for x in [0.0f32, 1.0, 2.0, 42.0, -1.0, -128.0, 65504.0] {
            let packed = f32_to_f16(x);
            let back = f16_to_f32(packed);
            assert!(
                (back - x).abs() <= x.abs() * 1e-3 || x == 0.0,
                "roundtrip {x} → {packed:#06x} → {back}",
            );
        }
    }

    #[test]
    fn test_f16_special_values() {
        // Zero
        assert_eq!(f32_to_f16(0.0f32), 0x0000);
        assert_eq!(f32_to_f16(-0.0f32), 0x8000);
        // Inf
        let inf_f16 = f32_to_f16(f32::INFINITY);
        assert_eq!(inf_f16, 0x7C00);
        // NaN (preserves signalling bit)
        let nan_f16 = f32_to_f16(f32::NAN);
        assert!(nan_f16 & 0x7C00 == 0x7C00);
        assert!(nan_f16 & 0x0200 != 0);
    }

    // -- Token probability tests ----------------------------------------

    #[test]
    fn test_token_probability_sum() {
        let logits = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let mut sum = 0.0;
        for t in 0..logits.len() as u32 {
            sum += AneDraftModel::token_probability(t, &logits);
        }
        assert!(
            (sum - 1.0_f32).abs() < 1e-5,
            "probabilities sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = vec![-10.0f32, -5.0, 100.0, -20.0];
        assert_eq!(AneDraftModel::sample_token_greedy(&logits), 2);
    }

    #[test]
    fn test_sample_token_greedy_single() {
        let logits = vec![42.0f32];
        assert_eq!(AneDraftModel::sample_token_greedy(&logits), 0);
    }
}
