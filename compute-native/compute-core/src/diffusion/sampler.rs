//! Pure-CPU deterministic diffusion sampler.
//!
//! Receives logits from a transformer forward pass and decides which tokens
//! to commit and which to remask. All operations are on CPU-side `Vec<f32>`
//! and `Vec<u32>` — no MLX Array dependency.

use crate::config::MaskSelection;

use super::canvas::TokenCanvas;

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64 + Box-Muller)
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random number generator (splitmix64 + Box-Muller).
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Compute the next `u64` via splitmix64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        z
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    /// Gaussian-distributed `f32` via Box-Muller transform.
    pub fn next_gaussian(&mut self) -> f32 {
        let u: f64 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let v: f64 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * (u + 1e-15).ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * v;
        (r * theta.cos()) as f32
    }
}

// ---------------------------------------------------------------------------
// DiffusionSampler
// ---------------------------------------------------------------------------

/// Output of one `sampler.sample()` call.
pub struct SamplerOutput {
    /// Token IDs selected for each position (unmasked positions keep their previous token).
    pub token_ids: Vec<u32>,
    /// Confidence scores for each position, in `[0, 1]`.
    pub confidence_scores: Vec<f32>,
    /// Which positions were committed this step.
    pub commit_mask: Vec<bool>,
    /// Which positions are scheduled for remasking next step.
    pub remask_mask: Vec<bool>,
    /// Whether an EOS token was committed anywhere.
    pub eos_triggered: bool,
}

/// Convergence status from `check_convergence`.
pub enum ConvergenceResult {
    /// Not yet converged — continue sampling.
    NotConverged,
    /// Converged: confidence on all unresolved positions is stable.
    Converged { reason: String, patience_steps: u32 },
    /// Every position is committed.
    AllCommitted,
    /// An EOS token collapsed the remaining positions.
    EosCollapse,
    /// The maximum number of steps was reached.
    MaxStepsReached,
}

/// Pure-CPU deterministic diffusion sampler.
///
/// Receives logits from a transformer forward pass and decides which tokens
/// to commit and which to remask.
pub struct DiffusionSampler {
    /// Seed for the PRNG.
    pub seed: u64,
    /// Temperature for logit scaling before softmax.
    pub temperature: f32,
    /// If set, restrict sampling to the top-K logits.
    pub top_k: Option<u32>,
    /// If set, restrict sampling to the top-P (nucleus) probability mass.
    pub top_p: Option<f32>,
    /// Minimum confidence to commit a token.
    pub confidence_threshold: f32,
    /// Strategy for selecting which tokens to remask.
    pub mask_selection: MaskSelection,
    /// End-of-sequence token ID.
    pub eos_token_id: u32,
    /// Mask token ID used for positions awaiting generation.
    pub mask_token_id: u32,
    /// Internal RNG.
    pub rng: SimpleRng,
}

impl DiffusionSampler {
    /// Create a new `DiffusionSampler` from configuration.
    pub fn new(config: &crate::config::DiffusionConfig, seed: u64) -> Self {
        Self {
            seed,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            confidence_threshold: config.default_confidence_threshold,
            mask_selection: MaskSelection::LowestConfidence,
            eos_token_id: config.eos_token_id,
            mask_token_id: config.mask_token_id,
            rng: SimpleRng::new(seed),
        }
    }

    /// Sample tokens from logits.
    ///
    /// `logits` is flattened `[num_positions * vocab_size]`.
    /// Returns token IDs, confidence scores, and commit/remask decisions.
    pub fn sample(
        &mut self,
        logits: &[f32],
        num_positions: usize,
        vocab_size: usize,
        canvas: &TokenCanvas,
    ) -> SamplerOutput {
        let expected_len = num_positions * vocab_size;
        assert_eq!(
            logits.len(),
            expected_len,
            "logits.len() {} != num_positions {} * vocab_size {}",
            logits.len(),
            num_positions,
            vocab_size
        );

        let mut token_ids = Vec::with_capacity(num_positions);
        let mut confidence_scores = Vec::with_capacity(num_positions);
        let mut commit_mask = Vec::with_capacity(num_positions);
        let mut remask_mask = Vec::with_capacity(num_positions);
        let mut eos_triggered = false;

        for pos in 0..num_positions {
            let base = pos * vocab_size;

            // If this position is already committed, keep it.
            if pos < canvas.committed.len() && canvas.committed[pos] {
                token_ids.push(canvas.tokens[pos].unwrap_or(self.mask_token_id));
                confidence_scores.push(canvas.confidence[pos]);
                commit_mask.push(false);
                remask_mask.push(false);
                continue;
            }

            let logits_slice = &logits[base..base + vocab_size];

            // Apply softmax to get probabilities.
            let (probs, confidence) = self.softmax_with_confidence(logits_slice);

            // Sample a token from the probability distribution.
            let sampled_token = self.sample_from_probs(&probs);

            // Decide whether to commit based on confidence threshold.
            let do_commit = confidence >= self.confidence_threshold;

            // Decide whether to remask (remask if NOT committed).
            let do_remask = !do_commit;

            token_ids.push(sampled_token);
            confidence_scores.push(confidence);
            commit_mask.push(do_commit);
            remask_mask.push(do_remask);

            if do_commit && sampled_token == self.eos_token_id {
                eos_triggered = true;
            }
        }

        SamplerOutput {
            token_ids,
            confidence_scores,
            commit_mask,
            remask_mask,
            eos_triggered,
        }
    }

    /// Check convergence conditions and return a result.
    pub fn check_convergence(
        &self,
        canvas: &TokenCanvas,
        step: u32,
        max_steps: u32,
        patience: u32,
        unchanged_steps: u32,
    ) -> ConvergenceResult {
        if canvas.all_committed() {
            return ConvergenceResult::AllCommitted;
        }

        if step >= max_steps {
            return ConvergenceResult::MaxStepsReached;
        }

        if unchanged_steps >= patience {
            let unresolved = canvas.num_unresolved();
            return ConvergenceResult::Converged {
                reason: format!(
                    "no change for {} steps ({} unresolved positions)",
                    unchanged_steps, unresolved
                ),
                patience_steps: unchanged_steps,
            };
        }

        ConvergenceResult::NotConverged
    }

    /// Apply EOS collapse: positions after the first committed EOS get forced
    /// to EOS and marked committed. Returns `true` if any collapse happened.
    pub fn apply_eos_collapse(&self, canvas: &mut TokenCanvas) -> bool {
        // Find the first committed EOS position.
        let eos_pos = canvas
            .tokens
            .iter()
            .position(|t| t.as_ref().copied() == Some(self.eos_token_id));

        let Some(eos_idx) = eos_pos else {
            return false;
        };

        // Everything after the first committed EOS gets collapsed to EOS.
        let mut collapsed = false;
        for i in (eos_idx + 1)..canvas.tokens.len() {
            if !canvas.committed[i] {
                canvas.tokens[i] = Some(self.eos_token_id);
                canvas.committed[i] = true;
                canvas.confidence[i] = 1.0;
                collapsed = true;
            }
        }

        if collapsed {
            canvas.total_committed = canvas.committed.iter().map(|&c| c as u32).sum();
            canvas.total_unresolved = canvas.num_unresolved();
        }

        collapsed
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Stable softmax over a logit slice. Returns (probabilities, confidence).
    /// Confidence = max probability for the sampled/greedy token.
    fn softmax_with_confidence(&self, logits: &[f32]) -> (Vec<f32>, f32) {
        let temp = self.temperature.max(1e-6);

        // Find max for numerical stability.
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp((logit - max) / temperature).
        let mut sum = 0.0f64;
        let mut scaled = Vec::with_capacity(logits.len());
        for &l in logits {
            let v = ((l - max_logit) / temp).exp() as f64;
            sum += v;
            scaled.push(v);
        }

        // Normalize and pick confidence = max probability.
        let inv_sum = 1.0 / sum;
        let mut max_prob = 0.0f32;
        let probs: Vec<f32> = scaled
            .iter()
            .map(|&v| {
                let p = (v * inv_sum) as f32;
                if p > max_prob {
                    max_prob = p;
                }
                p
            })
            .collect();

        (probs, max_prob)
    }

    /// Sample a token index from a probability distribution.
    fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        // top-K filtering.
        let filtered = if let Some(k) = self.top_k {
            self.top_k_filter(probs, k)
        } else {
            probs.to_vec()
        };

        // top-P (nucleus) filtering.
        let filtered = if let Some(p) = self.top_p {
            self.top_p_filter(&filtered, p)
        } else {
            filtered
        };

        // Greedy: pick argmax.
        let max_idx = filtered
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // With temperature > 0, do stochastic sampling; otherwise argmax.
        if self.temperature > 0.0 {
            self.stochastic_sample(&filtered)
        } else {
            max_idx as u32
        }
    }

    /// Greedy-only: restrict to top-K probabilities; zero out the rest.
    fn top_k_filter(&self, probs: &[f32], k: u32) -> Vec<f32> {
        let k = (k as usize).min(probs.len());
        if k == probs.len() {
            return probs.to_vec();
        }

        // Find the k-th largest probability.
        let mut sorted: Vec<f32> = probs.to_vec();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k.saturating_sub(1)];

        // Zero out values below threshold.
        let mut filtered = Vec::with_capacity(probs.len());
        for &p in probs {
            if p >= threshold {
                filtered.push(p);
            } else {
                filtered.push(0.0);
            }
        }
        filtered
    }

    /// Nucleus (top-P) filtering: keep the smallest set of tokens whose
    /// cumulative probability exceeds P; zero out the rest.
    fn top_p_filter(&self, probs: &[f32], p: f32) -> Vec<f32> {
        // Create index-value pairs sorted descending by probability.
        let mut pairs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        pairs.sort_unstable_by(|(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cum = 0.0f32;
        let mut threshold = 0.0f32;
        for (_, prob) in &pairs {
            cum += prob;
            if cum >= p {
                threshold = *prob;
                break;
            }
        }

        let mut filtered = Vec::with_capacity(probs.len());
        for &prob in probs {
            if prob >= threshold && prob > 0.0 {
                filtered.push(prob);
            } else {
                filtered.push(0.0);
            }
        }
        filtered
    }

    /// Stochastic sample according to the probability distribution.
    fn stochastic_sample(&mut self, probs: &[f32]) -> u32 {
        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            return 0;
        }

        let r = self.rng.next_f32() * total;
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                return i as u32;
            }
        }

        // Fallback (shouldn't normally reach here).
        (probs.len() - 1) as u32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rng_deterministic() {
        let mut a = SimpleRng::new(42);
        let mut b = SimpleRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_simple_rng_gaussian_bounds() {
        let mut rng = SimpleRng::new(12345);
        for _ in 0..1000 {
            let g = rng.next_gaussian();
            assert!(g.is_finite(), "gaussian returned non-finite {}", g);
        }
    }

    #[test]
    fn test_simple_rng_f32_range() {
        let mut rng = SimpleRng::new(99);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "f32 out of range: {}", v);
        }
    }

    #[test]
    fn test_diffusion_sampler_new() {
        let config = crate::config::DiffusionConfig::default();
        let sampler = DiffusionSampler::new(&config, 42);
        assert_eq!(sampler.seed, 42);
        assert!((sampler.confidence_threshold - 0.7).abs() < 1e-6);
        assert_eq!(sampler.eos_token_id, 0);
        assert_eq!(sampler.mask_token_id, 0);
    }

    #[test]
    fn test_sample_basic() {
        let mut sampler = DiffusionSampler {
            seed: 1,
            temperature: 0.0, // greedy
            top_k: None,
            top_p: None,
            confidence_threshold: 0.5,
            mask_selection: MaskSelection::LowestConfidence,
            eos_token_id: 0,
            mask_token_id: 0,
            rng: SimpleRng::new(1),
        };

        let canvas = TokenCanvas::new(4, 0, 0);

        // Logits: 4 positions, 6 vocab; each position has a clear winner.
        let logits: Vec<f32> = vec![
            10.0, 1.0, 1.0, 1.0, 1.0, 1.0, // pos0: token 0
            1.0, 10.0, 1.0, 1.0, 1.0, 1.0, // pos1: token 1
            1.0, 1.0, 10.0, 1.0, 1.0, 1.0, // pos2: token 2
            1.0, 1.0, 1.0, 10.0, 1.0, 1.0, // pos3: token 3
        ];

        let output = sampler.sample(&logits, 4, 6, &canvas);
        assert_eq!(output.token_ids.len(), 4);
        assert_eq!(output.token_ids, vec![0, 1, 2, 3]);
        assert_eq!(output.commit_mask.len(), 4);
        assert_eq!(output.remask_mask.len(), 4);
        // With greedy temp=0 and high logit separation, all should commit.
        for &cm in &output.commit_mask {
            assert!(
                cm,
                "all positions should commit with high-confidence logits"
            );
        }
        assert!(!output.eos_triggered);
    }

    #[test]
    fn test_eos_triggered() {
        let mut sampler = DiffusionSampler {
            seed: 2,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            confidence_threshold: 0.5,
            mask_selection: MaskSelection::LowestConfidence,
            eos_token_id: 0,
            mask_token_id: 0,
            rng: SimpleRng::new(2),
        };

        let canvas = TokenCanvas::new(2, 0, 0);

        // First position has eos_token_id=0 as the argmax.
        let logits = vec![
            20.0, 1.0, // pos0: token 0 (EOS)
            1.0, 20.0, // pos1: token 1
        ];

        let output = sampler.sample(&logits, 2, 2, &canvas);
        assert!(output.eos_triggered, "EOS should be triggered");
        assert_eq!(output.token_ids[0], 0);
    }

    #[test]
    fn test_convergence_all_committed() {
        let sampler = DiffusionSampler {
            seed: 0,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            confidence_threshold: 0.5,
            mask_selection: MaskSelection::LowestConfidence,
            eos_token_id: 0,
            mask_token_id: 0,
            rng: SimpleRng::new(0),
        };

        let mut canvas = TokenCanvas::new(3, 0, 0);
        canvas.committed = vec![true, true, true];
        canvas.total_committed = 3;

        let result = sampler.check_convergence(&canvas, 5, 10, 3, 1);
        assert!(matches!(result, ConvergenceResult::AllCommitted));
    }

    #[test]
    fn test_convergence_max_steps() {
        let sampler = DiffusionSampler {
            seed: 0,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            confidence_threshold: 0.5,
            mask_selection: MaskSelection::LowestConfidence,
            eos_token_id: 0,
            mask_token_id: 0,
            rng: SimpleRng::new(0),
        };

        let canvas = TokenCanvas::new(3, 0, 0);
        let result = sampler.check_convergence(&canvas, 10, 10, 3, 2);
        assert!(matches!(result, ConvergenceResult::MaxStepsReached));
    }

    #[test]
    fn test_convergence_not_converged() {
        let sampler = DiffusionSampler {
            seed: 0,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            confidence_threshold: 0.5,
            mask_selection: MaskSelection::LowestConfidence,
            eos_token_id: 0,
            mask_token_id: 0,
            rng: SimpleRng::new(0),
        };

        let canvas = TokenCanvas::new(5, 0, 0);
        let result = sampler.check_convergence(&canvas, 2, 10, 3, 1);
        assert!(matches!(result, ConvergenceResult::NotConverged));
    }

    #[test]
    fn test_apply_eos_collapse() {
        let sampler = DiffusionSampler {
            seed: 0,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            confidence_threshold: 0.5,
            mask_selection: MaskSelection::LowestConfidence,
            eos_token_id: 0,
            mask_token_id: 0,
            rng: SimpleRng::new(0),
        };

        let mut canvas = TokenCanvas::new(5, 0, 0);
        canvas.tokens[1] = Some(0); // EOS at pos 1
        canvas.committed[1] = true;
        canvas.total_committed = 1;

        let collapsed = sampler.apply_eos_collapse(&mut canvas);
        assert!(collapsed, "should have collapsed trailing positions");
        assert!(canvas.committed[2]);
        assert!(canvas.committed[3]);
        assert!(canvas.committed[4]);
        assert_eq!(canvas.tokens[2], Some(0));
        assert_eq!(canvas.tokens[3], Some(0));
        assert_eq!(canvas.tokens[4], Some(0));
        assert_eq!(canvas.total_committed, 5);
    }
}
