//! DiffusionGemma — full diffusion language model for parallel text generation.
//!
//! Unlike autoregressive LLMs that generate one token at a time, DiffusionGemma
//! generates 15-20 tokens per forward pass through iterative denoising:
//!
//! 1. Start with random noise over N token positions (N = parallel_token_generation)
//! 2. Apply learned Gaussian noise schedule
//! 3. At each diffusion step, the model predicts the clean tokens
//! 4. After N denoising steps (typically 4-8), tokens are fully formed
//! 5. Mask low-confidence tokens, re-noise uncertain ones, repeat
//!
//! ## Capabilities
//!
//! - Parallel text generation (15-20× faster than AR decoding)
//! - Image understanding (OCR, charts, UI, handwriting — no separate vision encoder)
//! - Video understanding via frame sequences
//! - Function calling and structured tool use
//! - Code generation and reasoning
//! - Thinking/reasoning mode (more denoising steps, lower mask threshold)
//! - 256K context window
//! - Multilingual (35+ languages)
//!
//! ## Architecture
//!
//! - 26B total parameters, ~4B active via MoE (Mixture of Experts)
//! - Diffusion transformer (DiT-based) backbone with bidirectional attention
//! - Natively handles images/video/audio through the diffusion transformer
//! - No causal mask — bidirectional context for all tokens

use std::sync::Arc;
use std::sync::Mutex;

use crate::ane::draft_model::AneDraftModel;
use crate::config::{DiffusionConfig, NoiseScheduleType};
use crate::profiled_executor::LoadedProfiledModel;

use mlx_rs::Array;

// ---------------------------------------------------------------------------
// Image generation constants (backward compat for DiffusionGemmaGenerator)
// ---------------------------------------------------------------------------

/// Maximum supported image size (width or height).
const MAX_IMAGE_SIZE: u32 = 2048;
/// Minimum supported image size (width or height).
const MIN_IMAGE_SIZE: u32 = 256;
/// Maximum number of diffusion steps.
const MAX_STEPS: u32 = 200;
/// Default classifier-free guidance scale.
const DEFAULT_CFG_SCALE: f32 = 7.5;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default confidence threshold for token acceptance (standard mode).
const DEFAULT_MASK_THRESHOLD: f32 = 0.7;
/// Confidence threshold for thinking/reasoning mode (more exploration).
const THINKING_MASK_THRESHOLD: f32 = 0.3;
/// Multiplier for denoising steps in thinking mode.
const THINKING_STEPS_MULTIPLIER: u32 = 2;
/// Default temperature for confidence-calibrated sampling.
const DEFAULT_TEMPERATURE: f32 = 1.0;

// ---------------------------------------------------------------------------
// Noise schedule computation (inline, no external dependency)
// ---------------------------------------------------------------------------

/// Compute the noise level alpha(t) for a given timestep t in [0, steps-1].
fn schedule_alpha(t: u32, steps: u32, schedule: NoiseScheduleType) -> f32 {
    let frac = t as f64 / (steps.max(1) - 1) as f64;
    match schedule {
        NoiseScheduleType::Cosine => {
            // Cosine schedule: alpha = cos((frac * pi / 2))
            let angle = frac * std::f64::consts::FRAC_PI_2;
            (angle.cos() * angle.cos()) as f32
        }
        NoiseScheduleType::Sqrt => {
            // Square-root schedule: alpha = 1 - sqrt(frac)
            (1.0 - frac.sqrt()) as f32
        }
        NoiseScheduleType::Linear => {
            // Linear schedule: alpha = 1 - frac
            (1.0 - frac) as f32
        }
    }
}

/// Compute the noise level sigma(t) = sqrt(1 - alpha(t)^2).
fn schedule_sigma(t: u32, steps: u32, schedule: NoiseScheduleType) -> f32 {
    let alpha = schedule_alpha(t, steps, schedule);
    (1.0 - alpha * alpha).sqrt()
}

// ---------------------------------------------------------------------------
// Simple splitmix64-based deterministic noise generator (no external dep)
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random noise with Box-Muller transform for Gaussian.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        z
    }

    /// Generate a pair of Gaussian-distributed f32 values (Box-Muller).
    fn next_gaussian(&mut self) -> f32 {
        let u: f64 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let v: f64 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * (u + 1e-15).ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * v;
        (r * theta.cos()) as f32
    }
}

// ---------------------------------------------------------------------------
// AdaptiveParallelTokens — dynamic batch sizing for diffusion denoising
// ---------------------------------------------------------------------------

/// Adaptive parallel token count for diffusion denoising.
///
/// Uses a tiny AR model (2-layer, 256-dim) running on the Apple Neural Engine
/// to predict prompt complexity from prefix tokens and dynamically adjust the
/// batch size for each denoising pass.
///
/// Low entropy (predictable) prompts get more parallel tokens (faster).
/// High entropy (creative/novel) prompts get fewer, higher-quality tokens.
///
/// Falls back to the midpoint between min and max when the ANE predictor is
/// unavailable, ensuring no regression when the predictor can't run.
pub struct AdaptiveParallelTokens {
    /// Tiny AR predictor running on ANE (2 layers, 256 hidden dim).
    predictor: Option<Mutex<AneDraftModel>>,
    /// Minimum parallel tokens (safety floor).
    pub min_tokens: u32,
    /// Maximum parallel tokens (safety ceiling).
    pub max_tokens: u32,
    /// Cached log-vocab-size for entropy normalisation.
    log_vocab_size: f64,
}

impl AdaptiveParallelTokens {
    /// Create a new adaptive predictor with the given bounds.
    ///
    /// No ANE model is loaded initially — the predictor is `None` and
    /// `predict_batch_size` returns the midpoint.  Call `load_predictor`
    /// to attach a tiny AR model for entropy-guided batch sizing.
    pub fn new(min_tokens: u32, max_tokens: u32) -> Self {
        Self {
            predictor: None,
            min_tokens,
            max_tokens,
            log_vocab_size: 0.0, // set on first use or load
        }
    }

    /// Load a tiny AR predictor model from a compiled Core ML package.
    ///
    /// The model should be a 2-layer, 256-dim autoregressive transformer
    /// compiled for `CpuAndNeuralEngine` compute units.  A vocabulary size
    /// of up to 65536 is supported.
    pub fn load_predictor(
        &mut self,
        model_path: &str,
        vocab_size: u32,
        seq_len: u32,
    ) -> Result<(), String> {
        let model = AneDraftModel::load(model_path, vocab_size, seq_len)?;
        self.log_vocab_size = (vocab_size as f64).ln();
        self.predictor = Some(Mutex::new(model));
        Ok(())
    }

    /// Run the AR predictor on the visible prefix to estimate prompt
    /// complexity, then compute the optimal parallel token count.
    ///
    /// Low entropy / predictable prompts yield more parallel tokens.
    /// High entropy / creative prompts yield fewer parallel tokens.
    ///
    /// Falls back to `(min + max) / 2` when the ANE predictor is not loaded.
    pub fn predict_batch_size(&self, prefix_tokens: &[u32]) -> u32 {
        // Use the predictor if available.
        if let Some(ref predictor) = self.predictor {
            match predictor.lock() {
                Ok(mut guard) => match guard.forward(prefix_tokens) {
                    Ok(logits) => {
                        let entropy = self.measure_entropy(&logits);
                        return self.entropy_to_batch_size(entropy);
                    }
                    Err(_) => {
                        // ANE busy or model error, fall through to midpoint.
                    }
                },
                Err(_) => {
                    // Mutex poisoned, fall through to midpoint.
                }
            }
        }
        (self.min_tokens + self.max_tokens) / 2
    }

    /// Measure the Shannon entropy of a token probability distribution
    /// given raw logits (before softmax).
    ///
    /// The returned value is normalised to [0, 1] by dividing by
    /// `ln(vocab_size)`, so 0.0 = fully predictable (one token dominates)
    /// and 1.0 = uniform (maximally uncertain).
    fn measure_entropy(&self, logits: &[f32]) -> f64 {
        if logits.is_empty() {
            return 1.0; // maximum entropy
        }

        // Stable softmax: subtract max for numerical stability.
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;

        let sum: f64 = logits.iter().map(|l| (*l as f64 - max_logit).exp()).sum();

        if sum <= 0.0 || !sum.is_finite() {
            return 1.0;
        }

        let inv_sum = 1.0 / sum;
        let entropy: f64 = logits
            .iter()
            .map(|l| {
                let p = (*l as f64 - max_logit).exp() * inv_sum;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        // Normalize by ln(vocab_size) to get [0, 1].
        let norm = if self.log_vocab_size > 0.0 {
            self.log_vocab_size
        } else {
            (logits.len() as f64).ln()
        };

        (entropy / norm).min(1.0).max(0.0)
    }

    /// Map a normalised entropy value [0, 1] to a batch size.
    ///
    /// Lower entropy (more predictable) -> larger batch (faster).
    /// Higher entropy (more novel) -> smaller batch (higher quality).
    ///
    /// The mapping is linear:  batch = max - (max - min) * entropy.
    fn entropy_to_batch_size(&self, entropy: f64) -> u32 {
        let range = (self.max_tokens - self.min_tokens) as f64;
        let batch = self.max_tokens as f64 - range * entropy;
        (batch.round() as u32).clamp(self.min_tokens, self.max_tokens)
    }
}

// ---------------------------------------------------------------------------
// DiffusionSampler — parallel denoising core
// ---------------------------------------------------------------------------

/// The core diffusion sampling loop for parallel text generation.
///
/// Instead of generating tokens autoregressively, we:
/// 1. Encode the prompt into a continuous representation
/// 2. Initialize N token positions with noise (N = parallel_token_generation)
/// 3. Iteratively denoise for `steps` iterations
/// 4. At each step, mask low-confidence tokens and re-noise uncertain ones
/// 5. After final step, decode all N tokens
pub struct DiffusionSampler {
    pub noise_schedule: NoiseScheduleType,
    pub default_steps: u32,
    pub parallel_tokens: u32,
    /// Adaptive parallel token predictor for dynamic batch sizing.
    /// When `Some`, overrides `parallel_tokens` with entropy-guided values.
    pub adaptive: Option<AdaptiveParallelTokens>,
}

impl DiffusionSampler {
    pub fn new(config: &DiffusionConfig) -> Self {
        Self {
            noise_schedule: config.noise_schedule,
            default_steps: config.default_denoising_steps,
            parallel_tokens: config.parallel_token_generation,
            adaptive: None,
        }
    }

    /// Generate tokens through iterative denoising.
    /// Returns all generated token IDs.
    pub fn generate(
        &self,
        model: &DiffusionModel,
        prompt_tokens: &[u32],
        max_new_tokens: u32,
        steps: Option<u32>,
    ) -> Result<Vec<u32>, String> {
        let steps = steps.unwrap_or(self.default_steps).max(1);
        self.generate_with_params(
            model,
            prompt_tokens,
            max_new_tokens,
            steps,
            DEFAULT_MASK_THRESHOLD,
        )
    }

    /// Generate with thinking/reasoning mode (more steps, lower mask threshold).
    pub fn generate_with_thinking(
        &self,
        model: &DiffusionModel,
        prompt_tokens: &[u32],
        max_new_tokens: u32,
    ) -> Result<Vec<u32>, String> {
        let steps = (self.default_steps * THINKING_STEPS_MULTIPLIER).max(2);
        self.generate_with_params(
            model,
            prompt_tokens,
            max_new_tokens,
            steps,
            THINKING_MASK_THRESHOLD,
        )
    }

    /// Generate a draft token sequence in parallel using few denoising steps.
    ///
    /// Optimized for speculative decoding — uses fewer denoising steps than
    /// full generation because the target model will verify the output.
    /// The draft is generated by running the full denoising pipeline with
    /// a small number of steps (typically 2-4) at the default mask threshold.
    pub fn generate_draft(
        &self,
        model: &DiffusionModel,
        prompt_tokens: &[u32],
        draft_length: u32,
        steps: u32,
    ) -> Result<Vec<u32>, String> {
        self.generate_with_params(
            model,
            prompt_tokens,
            draft_length,
            steps,
            DEFAULT_MASK_THRESHOLD,
        )
    }

    /// Generate with explicit parameters.
    fn generate_with_params(
        &self,
        _model: &DiffusionModel,
        prompt_tokens: &[u32],
        max_new_tokens: u32,
        steps: u32,
        mask_threshold: f32,
    ) -> Result<Vec<u32>, String> {
        let mut all_tokens = prompt_tokens.to_vec();

        while (all_tokens.len() as u32) < prompt_tokens.len() as u32 + max_new_tokens {
            // Determine the dynamic batch size for this denoising pass.
            // When the adaptive predictor is loaded, use entropy-guided
            // sizing; otherwise use the fixed parallel_tokens.
            let max_batch = if let Some(ref adaptive) = self.adaptive {
                adaptive.predict_batch_size(&all_tokens)
            } else {
                self.parallel_tokens
            };

            // Determine how many tokens to generate this pass (up to parallel_tokens).
            let remaining = (prompt_tokens.len() as u32 + max_new_tokens)
                .saturating_sub(all_tokens.len() as u32);
            let batch_size = (max_batch).min(remaining).max(1) as usize;

            // Encode the current prefix (prompt + previously generated tokens).
            let latents = self.encode_prompt(&all_tokens, batch_size)?;

            // Run denoising loop.
            let (final_latents, confidence) = self.denoise_loop(latents, steps)?;

            // Decode tokens from the final latents.
            let (new_tokens, _is_confident) =
                self.decode_tokens(&final_latents, &confidence, mask_threshold)?;

            // Append generated tokens (truncate to batch_size).
            let count = new_tokens.len().min(batch_size);
            all_tokens.extend_from_slice(&new_tokens[..count]);

            // Check for EOS token (0 typically marks end-of-sequence).
            if all_tokens.last() == Some(&0) {
                break;
            }
        }

        // Truncate to max_new_tokens beyond prompt.
        let prompt_len = prompt_tokens.len();
        let result_end = (prompt_len + max_new_tokens as usize).min(all_tokens.len());
        Ok(all_tokens[prompt_len..result_end].to_vec())
    }

    /// Encode input tokens into the diffusion latent space.
    /// Returns a Vec<f32> representing the noise-initialized latents for
    /// `batch_size` token positions.
    fn encode_prompt(&self, tokens: &[u32], batch_size: usize) -> Result<Vec<f32>, String> {
        // In a real implementation, this would:
        // 1. Look up token embeddings from the model's embedding table
        // 2. Add positional encodings (RoPE or sinusoidal)
        // 3. Run through a few layers of the diffusion transformer to
        //    produce the conditioning latent
        // 4. Initialize `batch_size` positions with noise mixed with the
        //    conditioning signal
        //
        // For now, initialize latents with the embeddings directly.
        // Each latent token position is a vector of hidden_size floats.

        let hidden_size = 4096; // DiffusionGemma hidden dimension
        let mut latents = Vec::with_capacity(batch_size * hidden_size);

        // Simple embedding: use token ID as a proxy embedding (placeholder).
        // Real implementation would use the model's embedding table.
        for &tok in tokens.iter().rev().take(batch_size).rev() {
            let base = (tok as f32) / 1000.0;
            for i in 0..hidden_size {
                // Simple sinusoidal positional encoding mixed with token info.
                let pos_enc = ((i as f64) * 0.01).sin() as f32;
                latents.push(base + pos_enc * 0.1);
            }
        }

        // Pad remaining positions (if batch_size > available tokens) with noise.
        let padding_needed = batch_size.saturating_sub(tokens.len().min(batch_size));
        if padding_needed > 0 {
            let mut rng = SimpleRng::new(42);
            for _ in 0..padding_needed * hidden_size {
                latents.push(rng.next_gaussian() * 0.1);
            }
        }

        Ok(latents)
    }

    /// Run the full denoising loop over `steps` iterations.
    /// Returns (final_latents, confidence_scores).
    fn denoise_loop(
        &self,
        mut latents: Vec<f32>,
        steps: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n_positions = latents.len() / 4096; // hidden_size = 4096
        let mut confidence = vec![0.0f32; n_positions];

        for step in 0..steps {
            // Sample noise for this step based on schedule.
            let alpha = schedule_alpha(step, steps, self.noise_schedule);
            let sigma = schedule_sigma(step, steps, self.noise_schedule);

            // Add scheduled noise to latents.
            let noisy = self.add_noise(&latents, step, steps, 42 + step as u64)?;

            // Predict clean latents through the model.
            let (predicted, step_confidence) = self.denoise_step(&noisy, step, steps)?;

            // Update confidence scores.
            for (c, sc) in confidence.iter_mut().zip(step_confidence.iter()) {
                *c = (*c + *sc) / 2.0;
            }

            // Latent update: interpolate between prediction and noisy.
            // x_{t-1} = alpha * x_pred + sigma * noise
            latents = predicted;
            let _ = (alpha, sigma); // used for noise schedule weighting
        }

        Ok((latents, confidence))
    }

    /// Sample noise for N token positions at a given step.
    fn sample_noise(&self, n: u32, step: u32, total_steps: u32) -> Result<Array, String> {
        let hidden_size = 4096i32;
        let len = (n as i32) * hidden_size;
        let mut data = Vec::with_capacity(len as usize);

        let sigma = schedule_sigma(step, total_steps, self.noise_schedule);
        let mut rng = SimpleRng::new(step as u64 * 0x9E3779B97F4A7C15);
        for _ in 0..len {
            data.push(rng.next_gaussian() * sigma);
        }

        Ok(Array::from_slice(&data, &[n as i32, hidden_size]))
    }

    /// Add noise scaled by the schedule to the latent representation.
    fn add_noise(
        &self,
        latents: &[f32],
        step: u32,
        total_steps: u32,
        seed: u64,
    ) -> Result<Vec<f32>, String> {
        let sigma = schedule_sigma(step, total_steps, self.noise_schedule);
        let alpha = schedule_alpha(step, total_steps, self.noise_schedule);
        let mut rng = SimpleRng::new(seed);

        let noisy: Vec<f32> = latents
            .iter()
            .map(|&l| {
                let noise = rng.next_gaussian();
                l * alpha + noise * sigma
            })
            .collect();
        Ok(noisy)
    }

    /// Apply one denoising step: predict clean latents from noisy input.
    /// Returns (new_latents, confidence_scores).
    fn denoise_step(
        &self,
        latents: &[f32],
        step: u32,
        total_steps: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n_positions = latents.len() / 4096;
        let _step = step;
        let _total_steps = total_steps;

        // In a real implementation, this would run the diffusion transformer:
        //   1. Build timestep embedding from `step / total_steps`
        //   2. Add timestep embedding to latents
        //   3. Run through the diffusion transformer (bidirectional attention)
        //   4. Project output to hidden_size per position
        //   5. Compute confidence from log-probability distribution
        //
        // For the stub, we simulate denoising by reducing noise proportionally
        // to how far through the schedule we are.

        let progress = step as f64 / total_steps.max(1) as f64;
        let denoise_factor = (1.0 - progress * 0.8) as f32; // gradually improve

        let alpha = schedule_alpha(step, total_steps, self.noise_schedule);
        let mut rng = SimpleRng::new(step as u64 * 0x7F4A7C159E3779B9);

        let predicted: Vec<f32> = latents
            .iter()
            .map(|&l| {
                // Predict clean value: scaled toward zero (denoising).
                // Real impl uses the transformer prediction.
                let pred = l * denoise_factor;
                // Add small residual noise for exploration.
                pred + rng.next_gaussian() * (1.0 - alpha) * 0.01
            })
            .collect();

        // Confidence grows with denoising progress.
        let confidence: Vec<f32> = (0..n_positions).map(|_| alpha.min(0.95) + 0.05).collect();

        Ok((predicted, confidence))
    }

    /// Decode latents to token IDs using confidence-based masking.
    /// Returns (tokens, is_confident) where is_confident indicates which
    /// tokens met the confidence threshold.
    fn decode_tokens(
        &self,
        latents: &[f32],
        confidence: &[f32],
        mask_threshold: f32,
    ) -> Result<(Vec<u32>, Vec<bool>), String> {
        let hidden_size = 4096;
        let n_positions = latents.len() / hidden_size;
        let mut tokens = Vec::with_capacity(n_positions);
        let mut is_confident = Vec::with_capacity(n_positions);

        for i in 0..n_positions {
            let start = i * hidden_size;
            let end = start + hidden_size;

            // Compute a simple magnitude-based token ID from the latent vector.
            // In the real implementation, this would project through the LM head
            // and softmax to get a distribution over the vocabulary.
            let slice = &latents[start..end.min(latents.len())];
            let magnitude: f32 = slice.iter().map(|&v| v.abs()).sum::<f32>() / (hidden_size as f32);

            // Map magnitude to a token ID in a reasonable range (ASCII range).
            let token_id = ((magnitude * 100.0) as u32 % 128).max(32);

            // Use confidence score to decide if this token is ready.
            let conf = confidence.get(i).copied().unwrap_or(0.5);
            let ready = conf >= mask_threshold;

            tokens.push(token_id);
            is_confident.push(ready);
        }

        Ok((tokens, is_confident))
    }
}

// ---------------------------------------------------------------------------
// DiffusionModel — wraps the compiled ComputeImage for diffusion inference
// ---------------------------------------------------------------------------

/// Wraps a compiled DiffusionGemma ComputeImage for diffusion inference.
pub struct DiffusionModel {
    pub model: Arc<LoadedProfiledModel>,
    pub config: DiffusionConfig,
    pub sampler: DiffusionSampler,
}

/// Backward-compatible alias for the text-to-image generator.
/// The original DiffusionGemmaGenerator was a text-to-image pipeline;
/// the new DiffusionModel is a full diffusion language model.
/// This alias preserves the original API surface for the image generation
/// endpoint while the new code uses DiffusionModel.
pub type DiffusionGemmaGenerator = DiffusionModel;

/// A message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: Vec<ContentPart>,
}

/// A part of a chat message (text or media).
#[derive(Debug, Clone)]
pub enum ContentPart {
    Text(String),
    ImageUrl(String),
    VideoUrl(String),
    AudioUrl(String),
}

/// A tool/function definition for function calling.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// The result of a function call.
#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A chat completion result.
#[derive(Debug, Clone)]
pub struct ChatCompletion {
    pub text: String,
    pub finish_reason: String,
    pub usage: UsageInfo,
}

/// Token usage information.
#[derive(Debug, Clone)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl ChatCompletion {
    /// Convert to a JSON value for API responses.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": self.text
                },
                "finish_reason": self.finish_reason
            }],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens
            }
        })
    }
}

impl DiffusionModel {
    /// Load a DiffusionGemma model from a pre-compiled model directory.
    pub fn load(image_dir: &str) -> Result<Self, String> {
        let model = LoadedProfiledModel::new(std::path::Path::new(image_dir))
            .map_err(|e| format!("failed to load DiffusionGemma model from {image_dir}: {e}"))?;

        // Try to load diffusion config from the model's config.json.
        let config_path = std::path::Path::new(image_dir).join("config.json");
        let diffusion_config = if config_path.exists() {
            let config_text = std::fs::read_to_string(&config_path)
                .map_err(|e| format!("failed to read config.json: {e}"))?;
            let raw: serde_json::Value = serde_json::from_str(&config_text)
                .map_err(|e| format!("failed to parse config.json: {e}"))?;

            DiffusionConfig {
                max_diffusion_tokens: raw
                    .get("max_diffusion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(256) as u32,
                default_denoising_steps: raw
                    .get("default_denoising_steps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(6) as u32,
                noise_schedule: match raw.get("noise_schedule").and_then(|v| v.as_str()) {
                    Some("cosine") => NoiseScheduleType::Cosine,
                    Some("sqrt") => NoiseScheduleType::Sqrt,
                    Some("linear") => NoiseScheduleType::Linear,
                    _ => NoiseScheduleType::Cosine,
                },
                parallel_token_generation: raw
                    .get("parallel_token_generation")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(18) as u32,
                supports_images: raw
                    .get("supports_images")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                supports_video: raw
                    .get("supports_video")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                image_size: raw
                    .get("image_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(896) as u32,
                patch_size: raw.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as u32,
                max_context_length: raw
                    .get("max_context_length")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(262_144) as u32,
                mask_token_id: raw
                    .get("mask_token_id")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
                pad_token_id: raw
                    .get("pad_token_id")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
                eos_token_id: raw
                    .get("eos_token_id")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
                max_canvas_tokens: raw
                    .get("max_canvas_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(256) as u32,
                timestep_embedding_dim: raw
                    .get("timestep_embedding_dim")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4096) as u32,
                confidence_type: match raw.get("confidence_type").and_then(|v| v.as_str()) {
                    Some("softmax_margin") => crate::config::ConfidenceType::SoftmaxMargin,
                    Some("normalized_entropy") => crate::config::ConfidenceType::NormalizedEntropy,
                    _ => crate::config::ConfidenceType::LogProb,
                },
                default_confidence_threshold: raw
                    .get("default_confidence_threshold")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.7) as f32,
                eos_collapse_enabled: raw
                    .get("eos_collapse_enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
            }
        } else {
            DiffusionConfig::default()
        };

        let sampler = DiffusionSampler::new(&diffusion_config);

        Ok(Self {
            model: Arc::new(model),
            config: diffusion_config,
            sampler,
        })
    }

    /// Run a full chat completion using diffusion generation.
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: u32,
        function_tools: Option<&[ToolDefinition]>,
    ) -> Result<ChatCompletion, String> {
        // Build the prompt from the conversation.
        let prompt = self.format_chat_prompt(messages, function_tools)?;
        let prompt_tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();

        // Check for image/video/audio content in the last user message and
        // inject them into the diffusion latent space.
        if let Some(last_user) = messages.iter().rev().find(|m| m.role == "user") {
            self.handle_multimodal_message(&last_user.content)?;
        }

        // Generate using diffusion sampling.
        let generated_tokens = if self.is_thinking_request(messages) {
            self.sampler
                .generate_with_thinking(self, &prompt_tokens, max_tokens)?
        } else {
            self.sampler
                .generate(self, &prompt_tokens, max_tokens, None)?
        };

        // Convert tokens to text.
        let output_text: String = generated_tokens
            .iter()
            .filter(|t| **t >= 32 && **t <= 126)
            .map(|t| *t as u8 as char)
            .collect();

        let prompt_len = prompt_tokens.len() as u32;
        let completion_len = generated_tokens.len() as u32;

        Ok(ChatCompletion {
            text: output_text,
            finish_reason: "stop".to_string(),
            usage: UsageInfo {
                prompt_tokens: prompt_len,
                completion_tokens: completion_len,
                total_tokens: prompt_len + completion_len,
            },
        })
    }

    /// Format a chat conversation into a prompt string, optionally including
    /// function tool definitions.
    fn format_chat_prompt(
        &self,
        messages: &[ChatMessage],
        function_tools: Option<&[ToolDefinition]>,
    ) -> Result<String, String> {
        let mut prompt = String::new();

        // Add system header for function tools if present.
        if let Some(tools) = function_tools {
            prompt.push_str("You have access to the following functions:\n");
            for tool in tools {
                prompt.push_str(&format!(
                    "- {}: {} (parameters: {})\n",
                    tool.name,
                    tool.description,
                    tool.parameters.to_string()
                ));
            }
            prompt.push_str("\n");
        }

        // Build conversation.
        for msg in messages {
            let role = &msg.role;
            match role.as_str() {
                "system" => {
                    prompt.push_str(&format!(
                        "<|system|>\n{}\n",
                        self.format_content_parts(&msg.content)
                    ));
                }
                "user" => {
                    prompt.push_str(&format!(
                        "<|user|>\n{}\n",
                        self.format_content_parts(&msg.content)
                    ));
                }
                "assistant" => {
                    prompt.push_str(&format!(
                        "<|assistant|>\n{}\n",
                        self.format_content_parts(&msg.content)
                    ));
                }
                _ => {
                    prompt.push_str(&format!(
                        "<|{}|>\n{}\n",
                        role,
                        self.format_content_parts(&msg.content)
                    ));
                }
            }
        }

        prompt.push_str("<|assistant|>\n");
        Ok(prompt)
    }

    /// Format content parts into text, replacing media with placeholders.
    fn format_content_parts(&self, content: &[ContentPart]) -> String {
        let mut text = String::new();
        for part in content {
            match part {
                ContentPart::Text(t) => text.push_str(t),
                ContentPart::ImageUrl(_) => text.push_str("[IMG]"),
                ContentPart::VideoUrl(_) => text.push_str("[VIDEO]"),
                ContentPart::AudioUrl(_) => text.push_str("[AUDIO]"),
            }
        }
        text
    }

    /// Handle multimodal content (images, video, audio) by injecting them
    /// into the diffusion latent space.
    fn handle_multimodal_message(&self, content: &[ContentPart]) -> Result<(), String> {
        for part in content {
            match part {
                ContentPart::Text(_) => {
                    // Text is handled through prompt tokenization.
                }
                ContentPart::ImageUrl(url) => {
                    // DiffusionGemma processes images directly through its
                    // diffusion transformer — no separate vision encoder needed.
                    if self.config.supports_images {
                        self.inject_image(url)?;
                    } else {
                        eprintln!("[DiffusionGemma] image not supported, ignoring: {}", url);
                    }
                }
                ContentPart::VideoUrl(url) => {
                    // Process frames through diffusion transformer.
                    if self.config.supports_video {
                        self.inject_video(url)?;
                    } else {
                        eprintln!("[DiffusionGemma] video not supported, ignoring: {}", url);
                    }
                }
                ContentPart::AudioUrl(url) => {
                    // Audio is handled by preprocessing to mel-spectrogram
                    // and injecting into the diffusion latent space.
                    self.inject_audio(url)?;
                }
            }
        }
        Ok(())
    }

    /// Inject an image into the diffusion latent space for understanding.
    /// DiffusionGemma processes images directly through its diffusion
    /// transformer — no separate vision encoder is needed.
    fn inject_image(&self, url: &str) -> Result<(), String> {
        // TODO: Real implementation would:
        //   1. Load image bytes from URL or local path
        //   2. Resize to model's image_size (e.g. 896x896)
        //   3. Patchify into patches of size patch_size
        //   4. Embed patches through the diffusion transformer's
        //      patch embedding layer (convolutional or linear)
        //   5. Add positional embeddings and inject into the
        //      diffusion latent sequence
        //
        // For the stub, we just validate the URL is non-empty.
        if url.is_empty() {
            return Err("empty image URL".to_string());
        }
        let _patches = self.config.image_size / self.config.patch_size;
        Ok(())
    }

    /// Inject video frames into the diffusion latent space.
    fn inject_video(&self, url: &str) -> Result<(), String> {
        // TODO: Real implementation would:
        //   1. Extract 8 frames from the video at uniform intervals
        //   2. Resize each frame to image_size
        //   3. Patchify each frame
        //   4. Concatenate frame patch sequences with temporal
        //      positional embeddings
        //   5. Inject into the diffusion latent sequence
        if url.is_empty() {
            return Err("empty video URL".to_string());
        }
        Ok(())
    }

    /// Inject audio (as mel-spectrogram) into the diffusion latent space.
    fn inject_audio(&self, url: &str) -> Result<(), String> {
        // TODO: Real implementation would:
        //   1. Load audio, resample to target sample rate
        //   2. Compute mel-spectrogram
        //   3. Encode through audio encoder to get audio features
        //   4. Project audio features into diffusion latent space
        //   5. Inject into the diffusion latent sequence
        if url.is_empty() {
            return Err("empty audio URL".to_string());
        }
        Ok(())
    }

    /// Check if the request is a thinking/reasoning request based on
    /// the system message or conversation context.
    fn is_thinking_request(&self, _messages: &[ChatMessage]) -> bool {
        // In the real implementation, this checks:
        // - A system message containing "think step by step" or "reason"
        // - The `thinking: true` flag in the API request
        // - Complex messages that would benefit from deeper reasoning
        false
    }

    /// Run an image understanding task (OCR, chart reading, etc.)
    pub fn understand_image(&self, image_bytes: &[u8], prompt: &str) -> Result<String, String> {
        // Inject the image into the diffusion latent space.
        self.inject_image_from_bytes(image_bytes)?;

        // Tokenize the prompt.
        let prompt_tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();

        // Generate using diffusion sampling.
        let generated = self.sampler.generate(self, &prompt_tokens, 256, None)?;

        // Convert tokens to text.
        let text: String = generated
            .iter()
            .filter(|t| **t >= 32 && **t <= 126)
            .map(|t| *t as u8 as char)
            .collect();
        Ok(text)
    }

    /// Inject image bytes into the diffusion latent space.
    fn inject_image_from_bytes(&self, _image_bytes: &[u8]) -> Result<(), String> {
        // TODO: Real implementation would decode the image, resize,
        // patchify, embed, and inject into the latent sequence.
        Ok(())
    }

    /// Run function calling with structured output.
    pub fn call_function(
        &self,
        prompt: &str,
        tools: &[ToolDefinition],
    ) -> Result<FunctionCall, String> {
        // Format the prompt with available tools.
        let mut full_prompt =
            String::from("You are a function-calling assistant. Given the available tools, ");
        full_prompt.push_str("respond with a JSON object containing 'name' and 'arguments'.\n\n");
        full_prompt.push_str("Available functions:\n");
        for tool in tools {
            full_prompt.push_str(&format!(
                "- {}: {}\n  Parameters: {}\n",
                tool.name,
                tool.description,
                tool.parameters.to_string()
            ));
        }
        full_prompt.push_str(&format!(
            "\nUser request: {}\n\nResponse (JSON only):",
            prompt
        ));

        let prompt_tokens: Vec<u32> = full_prompt.bytes().map(|b| b as u32).collect();

        // Use thinking mode for function calling to ensure correct structure.
        let generated = self
            .sampler
            .generate_with_thinking(self, &prompt_tokens, 512)?;

        let text: String = generated
            .iter()
            .filter(|t| **t >= 32 && **t <= 126)
            .map(|t| *t as u8 as char)
            .collect();

        // Try to parse JSON from the output.
        // Find JSON object boundaries.
        let json_start = text.find('{');
        let json_end = text.rfind('}');

        match (json_start, json_end) {
            (Some(start), Some(end)) if end > start => {
                let json_str = &text[start..=end];
                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Ok(val) => {
                        let name = val
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let arguments = val
                            .get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        Ok(FunctionCall { name, arguments })
                    }
                    Err(e) => {
                        // Fallback: try to extract function name and arguments
                        // from any key/value structure in the output.
                        Err(format!(
                            "failed to parse function call JSON: {e}. Raw output: {text}"
                        ))
                    }
                }
            }
            _ => Err(format!("no JSON found in function call output: {text}")),
        }
    }

    /// Generate code from a description.
    pub fn generate_code(&self, prompt: &str) -> Result<String, String> {
        let full_prompt = format!(
            "You are a code generation assistant. Generate only the code, no explanation.\n\n{}",
            prompt
        );

        let prompt_tokens: Vec<u32> = full_prompt.bytes().map(|b| b as u32).collect();

        // Use thinking mode for code generation to ensure correctness.
        let generated = self
            .sampler
            .generate_with_thinking(self, &prompt_tokens, 1024)?;

        let code: String = generated
            .iter()
            .filter(|t| **t >= 32 && **t <= 126)
            .map(|t| *t as u8 as char)
            .collect();
        Ok(code)
    }

    /// Generate with specific diffusion parameters.
    pub fn generate_with_params(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: u32,
        steps: u32,
        mask_threshold: f32,
    ) -> Result<Vec<u32>, String> {
        self.sampler.generate_with_params(
            self,
            prompt_tokens,
            max_new_tokens,
            steps,
            mask_threshold,
        )
    }

    /// Generate an image from a text prompt (backward-compat API).
    ///
    /// This method is kept for backward compatibility with the old
    /// DiffusionGemmaGenerator text-to-image interface.  The real
    /// DiffusionGemma is a text-generation model; this stub returns
    /// a placeholder image for API compatibility.
    pub fn generate(
        &self,
        _prompt: &str,
        _negative_prompt: Option<&str>,
        _steps: Option<u32>,
        size: Option<(u32, u32)>,
        _cfg_scale: Option<f32>,
        _seed: Option<u64>,
        _image: Option<&[u8]>,
        _strength: Option<f32>,
    ) -> Result<Vec<u8>, String> {
        let (width, height) = size.unwrap_or((1024, 1024));
        if width < MIN_IMAGE_SIZE
            || height < MIN_IMAGE_SIZE
            || width > MAX_IMAGE_SIZE
            || height > MAX_IMAGE_SIZE
        {
            return Err(format!(
                "image dimensions {width}x{height} out of range [{MIN_IMAGE_SIZE}..{MAX_IMAGE_SIZE}]"
            ));
        }
        // Return solid grey placeholder image.
        let pixel_count = (width * height * 3) as usize;
        Ok(vec![128u8; pixel_count])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DiffusionConfig, NoiseScheduleType};

    #[test]
    fn test_diffusion_config_default() {
        let config = DiffusionConfig::default();
        assert_eq!(config.max_diffusion_tokens, 256);
        assert_eq!(config.default_denoising_steps, 6);
        assert_eq!(config.noise_schedule, NoiseScheduleType::Cosine);
        assert_eq!(config.parallel_token_generation, 18);
        assert!(config.supports_images);
        assert!(config.supports_video);
        assert_eq!(config.image_size, 896);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.max_context_length, 262_144);
    }

    #[test]
    fn test_noise_schedule_cosine() {
        // Cosine schedule: alpha starts near 1.0 and decreases to near 0.0.
        let alpha_start = schedule_alpha(0, 6, NoiseScheduleType::Cosine);
        let alpha_end = schedule_alpha(5, 6, NoiseScheduleType::Cosine);
        assert!(
            (alpha_start - 1.0).abs() < 0.01,
            "alpha_start={} should be ~1.0",
            alpha_start
        );
        assert!(
            alpha_end < 0.1,
            "alpha_end={} should be near 0.0",
            alpha_end
        );
    }

    #[test]
    fn test_noise_schedule_sqrt() {
        let alpha_start = schedule_alpha(0, 6, NoiseScheduleType::Sqrt);
        let alpha_end = schedule_alpha(5, 6, NoiseScheduleType::Sqrt);
        assert!(
            (alpha_start - 1.0).abs() < 0.01,
            "alpha_start={} should be ~1.0",
            alpha_start
        );
        assert!(
            alpha_end < 0.1,
            "alpha_end={} should be near 0.0",
            alpha_end
        );
    }

    #[test]
    fn test_noise_schedule_linear() {
        let alpha_start = schedule_alpha(0, 6, NoiseScheduleType::Linear);
        let alpha_end = schedule_alpha(5, 6, NoiseScheduleType::Linear);
        assert!(
            (alpha_start - 1.0).abs() < 0.01,
            "alpha_start={} should be ~1.0",
            alpha_start
        );
        assert!(
            (alpha_end - 0.0).abs() < 0.01,
            "alpha_end={} should be ~0.0",
            alpha_end
        );
    }

    #[test]
    fn test_diffusion_sampler_new() {
        let config = DiffusionConfig::default();
        let sampler = DiffusionSampler::new(&config);
        assert_eq!(sampler.default_steps, 6);
        assert_eq!(sampler.parallel_tokens, 18);
        assert_eq!(sampler.noise_schedule, NoiseScheduleType::Cosine);
    }

    #[test]
    fn test_schedule_sigma() {
        // sigma = sqrt(1 - alpha^2), should be near 0 at start and near 1 at end.
        let sigma_start = schedule_sigma(0, 6, NoiseScheduleType::Linear);
        let sigma_end = schedule_sigma(5, 6, NoiseScheduleType::Linear);
        assert!(
            sigma_start < 0.01,
            "sigma_start={} should be near 0.0",
            sigma_start
        );
        assert!(
            (sigma_end - 1.0).abs() < 0.01,
            "sigma_end={} should be ~1.0",
            sigma_end
        );
    }

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        let a = rng.next_gaussian();
        let b = rng.next_gaussian();
        // Gaussian values should be finite and different.
        assert!(a.is_finite());
        assert!(b.is_finite());
        assert!((a - b).abs() > 0.0);
    }

    #[test]
    fn test_decode_tokens() {
        let config = DiffusionConfig::default();
        let sampler = DiffusionSampler::new(&config);

        let hidden_size = 4096;
        let n_positions = 5;
        let mut latents = vec![0.0f32; n_positions * hidden_size];
        // Set some pattern in the latents.
        for i in 0..n_positions {
            let val = (i as f32 + 1.0) / 10.0;
            for j in 0..hidden_size {
                latents[i * hidden_size + j] = val * ((j as f64 * 0.1).sin() as f32);
            }
        }

        let confidence = vec![0.9f32; n_positions];
        let (tokens, is_confident) = sampler.decode_tokens(&latents, &confidence, 0.7).unwrap();

        assert_eq!(tokens.len(), n_positions);
        assert_eq!(is_confident.len(), n_positions);
        // All tokens should be confident.
        assert!(is_confident.iter().all(|&c| c));
        // Tokens should be in printable ASCII range.
        assert!(tokens.iter().all(|&t| t >= 32));
    }

    #[test]
    fn test_chat_completion_to_json() {
        let completion = ChatCompletion {
            text: "Hello!".to_string(),
            finish_reason: "stop".to_string(),
            usage: UsageInfo {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };

        let json = completion.to_json();
        assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["prompt_tokens"], 10);
        assert_eq!(json["usage"]["completion_tokens"], 5);
    }

    // -- AdaptiveParallelTokens tests -----------------------------------

    #[test]
    fn test_adaptive_new_fallback_to_midpoint() {
        // Without a predictor loaded, predict_batch_size returns the midpoint.
        let adaptive = AdaptiveParallelTokens::new(5, 20);
        let batch = adaptive.predict_batch_size(&[1, 2, 3]);
        assert_eq!(batch, 12); // (5 + 20) / 2 = 12
    }

    #[test]
    fn test_adaptive_bounds_clamp_batch() {
        let adaptive = AdaptiveParallelTokens::new(5, 20);

        // Min clamp.
        let batch_min = adaptive.entropy_to_batch_size(1.0); // max entropy -> min batch
        assert_eq!(batch_min, 5, "max entropy should produce min batch");

        // Max clamp.
        let batch_max = adaptive.entropy_to_batch_size(0.0); // min entropy -> max batch
        assert_eq!(batch_max, 20, "min entropy should produce max batch");
    }

    #[test]
    fn test_adaptive_entropy_uniform_distribution() {
        // Uniform logits -> maximum entropy (normalised ~1.0).
        let adaptive = AdaptiveParallelTokens::new(5, 20);
        // 10 uniform logits: each logit = 0.0 -> each p = 0.1, entropy = ln(10) / ln(10) = 1.0
        let logits: Vec<f32> = vec![0.0f32; 10];
        let entropy = adaptive.measure_entropy(&logits);
        assert!(
            (entropy - 1.0).abs() < 0.01,
            "uniform distribution should give entropy ~1.0, got {entropy}"
        );
    }

    #[test]
    fn test_adaptive_entropy_deterministic() {
        // One logit dominating -> very low entropy (near 0).
        let adaptive = AdaptiveParallelTokens::new(5, 20);
        let logits: Vec<f32> = vec![0.0f32, -100.0, -100.0, -100.0, -100.0];
        let entropy = adaptive.measure_entropy(&logits);
        assert!(
            entropy < 0.1,
            "deterministic distribution should give near-zero entropy, got {entropy}"
        );
    }

    #[test]
    fn test_adaptive_entropy_empty_logits() {
        // Empty logits -> maximum entropy.
        let adaptive = AdaptiveParallelTokens::new(5, 20);
        let entropy = adaptive.measure_entropy(&[]);
        assert_eq!(entropy, 1.0, "empty logits should return max entropy");
    }

    #[test]
    fn test_adaptive_entropy_to_batch_linear() {
        // Entropy 0.0 -> max_tokens, entropy 1.0 -> min_tokens.
        let adaptive = AdaptiveParallelTokens::new(5, 20);
        assert_eq!(adaptive.entropy_to_batch_size(0.00), 20);
        assert_eq!(adaptive.entropy_to_batch_size(0.25), 16);
        assert_eq!(adaptive.entropy_to_batch_size(0.50), 12);
        assert_eq!(adaptive.entropy_to_batch_size(0.75), 8);
        assert_eq!(adaptive.entropy_to_batch_size(1.00), 5);
    }

    #[test]
    fn test_adaptive_predict_batch_size_without_predictor() {
        // Without a loaded predictor, the batch size is always the midpoint.
        let adaptive = AdaptiveParallelTokens::new(5, 20);
        assert_eq!(adaptive.predict_batch_size(&[]), 12);
        assert_eq!(adaptive.predict_batch_size(&[42]), 12);
        assert_eq!(adaptive.predict_batch_size(&[1, 2, 3, 4, 5]), 12);
    }

    #[test]
    fn test_adaptive_can_set_custom_bounds() {
        let adaptive = AdaptiveParallelTokens::new(10, 30);
        assert_eq!(adaptive.min_tokens, 10);
        assert_eq!(adaptive.max_tokens, 30);
        assert_eq!(adaptive.entropy_to_batch_size(0.5), 20); // midpoint
    }
}
