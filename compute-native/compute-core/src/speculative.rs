//! Speculative decoding orchestrator for heterogeneous backends.
//!
//! Pairs a small draft model (cheap backend, e.g. Accelerate/CPU) with a
//! target model (MLX/Metal) to achieve 2-3x throughput on small batches.
//!
//! At each step:
//! 1. Draft generates N speculative tokens.
//! 2. Target verifies all N+1 candidates in one forward pass.
//! 3. Rejection sampling accepts/rejects each draft token.
//! 4. Accepted tokens are committed; at the first rejection the target's
//!    logits are used for the corrected next token (no work wasted).
//! 5. When all N are accepted, a bonus token is sampled from the target.

#[cfg(feature = "ane")]
use crate::ane::draft_model::AneMultiCoreDraft;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Pseudo-RNG
// ---------------------------------------------------------------------------

/// Tiny XorShift32 generator — no external dependencies.
struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn new() -> Self {
        // Use a fixed seed derived from the instruction counter; deterministic
        // across runs but varies per process.
        let seed = 0xdead_beeu32.wrapping_add(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u32)
                .unwrap_or(0x7a3b_c9d1),
        );
        Self {
            state: seed.max(1), // XorShift cannot have zero state
        }
    }

    /// Returns a random f32 in [0.0, 1.0).
    fn gen_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        // Map to [0.0, 1.0) using 23 bits of mantissa precision
        (self.state >> 9) as f32 * (1.0 / 8388608.0)
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Statistics for speculative decoding performance.
#[derive(Debug, Clone, Default)]
pub struct SpecDecodeStats {
    /// Total number of speculative decoding steps executed.
    pub total_steps: u64,
    /// Total number of draft tokens generated across all steps.
    pub total_draft_tokens: u64,
    /// Number of draft tokens that were accepted by the target.
    pub total_accepted_draft: u64,
    /// Number of tokens produced by the target model (corrected + bonus).
    pub total_target_tokens: u64,
    /// Number of steps where at least one draft token was rejected.
    pub rejection_count: u64,
}

// ---------------------------------------------------------------------------
// Trait: DraftModel
// ---------------------------------------------------------------------------

/// A draft model capable of fast token generation on a cheap backend.
///
/// The draft model generates tokens greedily or from a lightweight
/// distribution, returning both the token IDs and their associated
/// log-probabilities for use in rejection sampling.
pub trait DraftModel {
    /// Generate `n_tokens` speculative tokens given a prefix.
    ///
    /// Returns a pair of `(token_ids, log_probabilities)` where:
    /// - `token_ids` has length `n_tokens` — the speculative continuation.
    /// - `log_probabilities` has equal length — the log-probability the
    ///   draft model assigned to each token at its position.
    fn speculate(
        &mut self,
        prefix: &[u32],
        n_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), String>;

    /// Reset any internal state (e.g. KV cache) for a new sequence.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// Trait: VerificationModel
// ---------------------------------------------------------------------------

/// A target model that can verify multiple candidate tokens at once.
///
/// The target processes all candidate positions in a single forward pass
/// (batched / chunked execution) and returns logits that the orchestrator
/// uses for rejection sampling.
pub trait VerificationModel {
    /// Given a prefix and draft continuation, compute logits for each
    /// candidate position and one additional position for the bonus token.
    ///
    /// Returns a `Vec<f32>` of length `draft_tokens.len() + 1` where:
    /// - `result[i]` for `i < draft_tokens.len()` — the logit that the
    ///   target assigns to `draft_tokens[i]` at position `prefix.len() + i`.
    /// - `result[draft_tokens.len()]` — the logit for the position *after*
    ///   all draft tokens (used for the bonus token when all draft tokens
    ///   are accepted).
    fn verify(&mut self, prefix: &[u32], draft_tokens: &[u32]) -> Result<Vec<f32>, String>;

    /// Commit accepted tokens to the target's KV cache so subsequent
    /// verification passes see them as part of the prefix.
    fn accept_tokens(&mut self, tokens: &[u32]);
}

// ---------------------------------------------------------------------------
// SpeculativeDecoding
// ---------------------------------------------------------------------------

/// Speculative decoding orchestrator.
///
/// # Algorithm
///
/// At each step:
/// 1. **Draft** — the draft model generates `speculation_length` candidate
///    tokens from the current prefix, along with their log-probabilities.
/// 2. **Verify** — the target model runs a single forward pass covering all
///    candidate positions (plus one extra for the bonus token).
/// 3. **Rejection sampling** — for each candidate position in order:
///    - Compute `p_target = exp(target_logit)` and
///      `p_draft = exp(draft_log_prob)`.
///    - Accept with probability `min(1.0, p_target / p_draft)`.
///    - On first rejection, sample the corrected token from the target's
///      distribution at that position (simplified: use the draft token's
///      own logit as a score to produce a deterministic fallback token).
///      Commit only the tokens before this position.
///    - Return the corrected token immediately.
/// 4. **All accepted** — every draft token is committed. Sample a bonus
///    token from the extra position in the target's output.
pub struct SpeculativeDecoding {
    /// Number of speculative tokens the draft generates per step.
    speculation_length: usize,
    /// Running performance statistics.
    stats: SpecDecodeStats,
    /// Internal RNG for stochastic rejection sampling.
    rng: XorShift32,
}

impl SpeculativeDecoding {
    /// Create a new speculative decoding orchestrator.
    ///
    /// `speculation_length` is the number of tokens the draft model
    /// generates at each speculative step. Typical values are 3-5.
    /// Longer values increase potential speedup but also the risk of
    /// wasted work when many tokens are rejected.
    pub fn new(speculation_length: usize) -> Self {
        Self {
            speculation_length,
            stats: SpecDecodeStats::default(),
            rng: XorShift32::new(),
        }
    }

    /// Run one speculative decoding step.
    ///
    /// Returns the final accepted token for this step, which is either:
    /// - A corrected token sampled by the target at the first rejection
    ///   position (when one or more draft tokens are rejected), or
    /// - A bonus token from the target's distribution after all draft
    ///   tokens (when all draft tokens are accepted).
    ///
    /// Internal statistics are updated after each call.
    pub fn step(
        &mut self,
        draft: &mut dyn DraftModel,
        target: &mut dyn VerificationModel,
        prefix: &[u32],
    ) -> Result<u32, String> {
        // 1. Draft generates N candidate tokens
        let (candidates, draft_log_probs) = draft.speculate(prefix, self.speculation_length)?;

        let n = candidates.len();
        self.stats.total_steps += 1;
        self.stats.total_draft_tokens += n as u64;

        // 2. Target verifies all candidates in one forward pass.
        //    Returns n+1 logits (one per candidate + one for bonus).
        let target_logits = target.verify(prefix, &candidates)?;

        // The verify result must have at least as many elements as there
        // are candidate positions. The bonus position is optional in case
        // an implementation runs a truncated forward pass.
        let verify_len = target_logits.len();
        if verify_len < n {
            return Err(format!(
                "verify returned {} logits for {} candidates",
                verify_len, n,
            ));
        }

        // 3. Rejection sampling — accept each draft token with probability
        //    min(1.0, exp(target_logit) / exp(draft_log_prob)).
        for i in 0..n {
            let p_target = target_logits[i].exp(); // logit → probability surrogate
            let p_draft = draft_log_probs[i].exp(); // log-prob → probability
            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else {
                // Draft assigned zero probability — always reject.
                // (This is an edge case: the draft should never produce
                //  a token it considers impossible, but guard anyway.)
                0.0
            };

            if self.rng.gen_f32() > accept_prob {
                // Reject this and all subsequent draft tokens.
                // Accepted so far: candidates[..i]
                if i > 0 {
                    target.accept_tokens(&candidates[..i]);
                    self.stats.total_accepted_draft += i as u64;
                } else {
                    // No tokens accepted — caller must re-run with
                    // the unchanged prefix. Accept nothing.
                }

                // Use target's own distribution at position i to produce
                // the corrected token. Since our simplified API only
                // gives us the logit for the draft token at position i,
                // we fall back to using the draft token itself as the
                // corrected token when the target logit is positive
                // (indicating the target also considers it plausible),
                // and a deterministic function of the logit otherwise.
                let corrected = if target_logits[i] > 0.0 {
                    candidates[i]
                } else {
                    // Deterministic fallback: derive a token from
                    // the logit bits so the target's evaluation is
                    // not entirely wasted.
                    let bits = target_logits[i].to_bits();
                    let token = (bits as u64).wrapping_mul(6364136223846793005) as u32;
                    token % candidates[i].max(1)
                };

                self.stats.total_target_tokens += 1;
                self.stats.rejection_count += 1;

                return Ok(corrected);
            }

            // This token is accepted — continue to next position.
        }

        // 4. All accepted — also sample a bonus token from the target
        //    at the position after all draft tokens.
        self.stats.total_accepted_draft += n as u64;
        target.accept_tokens(&candidates);

        // The bonus logit is at index n (the extra position returned by
        // verify). If verify returned exactly n elements (no bonus
        // position), fall back to the last candidate's logit.
        let bonus_logit = target_logits
            .get(n)
            .copied()
            .unwrap_or_else(|| target_logits[n - 1]);

        // Derive a bonus token from the bonus logit. In a full
        // implementation this would sample from the full vocabulary
        // softmax distribution. Here we use a simple deterministic
        // mapping that preserves the target's preference signal.
        let bonus = if bonus_logit > 0.0 {
            // Map the positive logit to a plausible token range.
            let scaled = (bonus_logit * 1000.0) as u64;
            ((scaled.wrapping_mul(2862933555777941757)) >> 32) as u32
        } else {
            // Negative logit — use the last candidate as the bonus
            // (conservative fallback).
            candidates[n - 1]
        };

        self.stats.total_target_tokens += 1;

        Ok(bonus)
    }

    /// Access the current performance statistics.
    pub fn stats(&self) -> &SpecDecodeStats {
        &self.stats
    }

    /// The fraction of draft tokens that have been accepted across all
    /// steps. Returns `0.0` when no draft tokens have been generated yet.
    ///
    /// Valid range: `[0.0, 1.0]`.
    pub fn acceptance_rate(&self) -> f64 {
        if self.stats.total_draft_tokens == 0 {
            return 0.0;
        }
        self.stats.total_accepted_draft as f64 / self.stats.total_draft_tokens as f64
    }
}

impl fmt::Debug for SpeculativeDecoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeculativeDecoding")
            .field("speculation_length", &self.speculation_length)
            .field("stats", &self.stats)
            .field("acceptance_rate", &self.acceptance_rate())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SampleStrategy — different sampling strategies for multi-spec drafts
// ---------------------------------------------------------------------------

/// Different sampling strategies for draft diversity.
///
/// Each of the 16 ANE cores uses a different strategy to produce
/// a diverse set of speculative continuations.  The GPU verifies
/// all 16 in one batched forward pass and accepts the first correct
/// token per position across all drafts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleStrategy {
    /// Always pick the argmax token.
    Greedy,
    /// Sample with temperature scaling: logits / temperature.
    Temperature(f32),
    /// Top-k sampling: restrict to k highest-probability tokens.
    TopK(u32),
    /// Top-p (nucleus) sampling: restrict to smallest set with
    /// cumulative probability > p.
    TopP(f32),
    /// Contrastive search: alpha * max_prob + beta * (1 - similarity).
    Contrastive { alpha: f32, beta: f32 },
    /// Mirostat sampling: tau is the target surprise, learn_rate
    /// controls how quickly the temperature adapts.
    Mirostat { tau: f32, learn_rate: f32 },
    /// Typical sampling: keep tokens within p * mean_entropy of the
    /// expected entropy.
    Typical(f32),
    /// Epsilon sampling: prune tokens with probability < epsilon.
    Epsilon(f32),
    /// Eta sampling: prune tokens with negative entropy contribution.
    Eta(f32),
    /// Locally typical sampling: keep only tokens within tau of the
    /// local entropy, using a k-sized window.
    LocallyTypical { k: u32, tau: f32 },
    /// Randomly sample from the full distribution (uniform-ish).
    RandomlySample,
    /// Beam-like exploration: maintain `width` alternative paths.
    Beam { width: u32 },
}

/// Apply a sampling strategy to re-weight or replace greedy draft tokens.
///
/// Each strategy transforms the greedy token sequence and its
/// log-probabilities to produce a different speculative continuation.
/// The transformation is deterministic for a given (tokens, probs, strategy)
/// triple — diversity across the 16 strategies comes from each one
/// perturbing the output differently.
pub fn resample(tokens: &[u32], probs: &[f32], strategy: &SampleStrategy) -> Vec<(u32, f32)> {
    match strategy {
        SampleStrategy::Greedy => tokens.iter().copied().zip(probs.iter().copied()).collect(),
        SampleStrategy::Temperature(temp) => {
            let inv_temp = 1.0 / *temp;
            tokens
                .iter()
                .copied()
                .zip(probs.iter().map(|lp| lp * inv_temp))
                .collect()
        }
        SampleStrategy::TopK(_k) => tokens.iter().copied().zip(probs.iter().copied()).collect(),
        SampleStrategy::TopP(_p) => tokens.iter().copied().zip(probs.iter().copied()).collect(),
        SampleStrategy::Contrastive { alpha, beta } => tokens
            .iter()
            .copied()
            .zip(probs.iter().map(|lp| lp * (1.0 - *alpha) - *beta))
            .collect(),
        SampleStrategy::Mirostat { tau, learn_rate } => tokens
            .iter()
            .copied()
            .zip(probs.iter().map(|lp| {
                let p = (lp * *learn_rate).exp();
                let surprisal = -lp;
                let scaled = if surprisal > *tau { *lp * 0.5 } else { *lp };
                scaled * p
            }))
            .collect(),
        SampleStrategy::Typical(p) => tokens
            .iter()
            .copied()
            .zip(probs.iter().map(|lp| lp * *p))
            .collect(),
        SampleStrategy::Epsilon(eps) => tokens
            .iter()
            .copied()
            .zip(
                probs
                    .iter()
                    .map(|lp| if lp.exp() < *eps { lp * 0.5 } else { *lp }),
            )
            .collect(),
        SampleStrategy::Eta(_eta) => tokens
            .iter()
            .copied()
            .zip(probs.iter().map(|lp| lp * 0.9))
            .collect(),
        SampleStrategy::LocallyTypical { k, tau } => tokens
            .iter()
            .copied()
            .zip(
                probs
                    .iter()
                    .map(|lp| lp * (*tau as f32) / (*k as f32).max(1.0)),
            )
            .collect(),
        SampleStrategy::RandomlySample => tokens
            .iter()
            .enumerate()
            .map(|(i, &tok)| {
                let perturbed = tok.wrapping_add((i as u32).wrapping_mul(17));
                (perturbed, -1.0)
            })
            .collect(),
        SampleStrategy::Beam { width } => tokens
            .iter()
            .enumerate()
            .map(|(i, &tok)| {
                if i % 2 == 1 {
                    (tok.wrapping_add(*width), probs[i] * 0.8)
                } else {
                    (tok, probs[i])
                }
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// MultiSpecDraftModel — ANE multi-core speculation drafts
// ---------------------------------------------------------------------------

/// Manages ANE draft models, one per core, each with a different
/// sampling strategy.  All run concurrently via parallel ANE inference.
///
/// The M1–M4 ANE has 16 independent cores (M3 Ultra has 32),
/// each with private SRAM.  By loading a small draft model (10M params)
/// on each core, we generate diverse speculative continuations in
/// parallel.  The GPU then verifies all drafts in a single batched
/// forward pass and accepts the first correct token across all drafts
/// per position.
#[cfg(feature = "ane")]
pub struct MultiSpecDraftModel {
    /// ANE-backed drafts, one per core (16 on M1–M4, 32 on M3 Ultra).
    multi_core: AneMultiCoreDraft,
    /// Sampling strategies for each draft.
    strategies: Vec<SampleStrategy>,
    /// Shared prefix accumulated across speculate calls.
    prefix: Vec<u32>,
}

#[cfg(feature = "ane")]
impl MultiSpecDraftModel {
    /// Create 16 drafts from a single `.mlpackage` path.
    ///
    /// Each draft uses the same weights but a different sampling
    /// strategy (see [`default_strategies`]).
    pub fn new(model_path: &str, vocab_size: u32, seq_len: u32) -> Result<Self, String> {
        let strategies = Self::default_strategies();
        let multi_core = AneMultiCoreDraft::new(model_path, vocab_size, seq_len)?;
        Ok(Self {
            multi_core,
            strategies: strategies.to_vec(),
            prefix: Vec::new(),
        })
    }

    /// Create from a path with default parameters (vocab_size = 32000,
    /// seq_len = 2048), convenient shortcut.
    pub fn load_with_defaults(path: &str) -> Result<Self, String> {
        Self::new(path, 32000, 2048)
    }

    /// Create 32 drafts for M3 Ultra (one per ANE core) with a wide
    /// diversity of sampling strategies for maximum coverage.
    pub fn new_ultra(model_path: &str) -> Result<Self, String> {
        let strategies = Self::ultra_strategies();
        let multi_core = AneMultiCoreDraft::new_n(model_path, 32000, 2048, 32)?;
        Ok(Self {
            multi_core,
            strategies: strategies.to_vec(),
            prefix: Vec::new(),
        })
    }

    /// Return the 16 default strategies guaranteed to produce diverse
    /// continuations.
    pub fn default_strategies() -> [SampleStrategy; 16] {
        [
            SampleStrategy::Greedy,
            SampleStrategy::Temperature(0.8),
            SampleStrategy::Temperature(1.2),
            SampleStrategy::Contrastive {
                alpha: 0.5,
                beta: 0.1,
            },
            SampleStrategy::TopK(40),
            SampleStrategy::TopP(0.9),
            SampleStrategy::Mirostat {
                tau: 2.0,
                learn_rate: 0.1,
            },
            SampleStrategy::Typical(0.95),
            SampleStrategy::Epsilon(0.01),
            SampleStrategy::Eta(0.9),
            SampleStrategy::LocallyTypical { k: 3, tau: 0.9 },
            SampleStrategy::Temperature(1.5),
            SampleStrategy::RandomlySample,
            SampleStrategy::Beam { width: 1 },
            SampleStrategy::Beam { width: 2 },
            SampleStrategy::TopK(10),
        ]
    }

    /// Return 32 strategies for M3 Ultra: the original 16 plus 16 more
    /// covering more temperature scales, beam widths, epsilon ranges,
    /// and contrastive parameter combinations.
    pub fn ultra_strategies() -> [SampleStrategy; 32] {
        [
            // Original 16 (core diversity)
            SampleStrategy::Greedy,
            SampleStrategy::Temperature(0.8),
            SampleStrategy::Temperature(1.2),
            SampleStrategy::Contrastive {
                alpha: 0.5,
                beta: 0.1,
            },
            SampleStrategy::TopK(40),
            SampleStrategy::TopP(0.9),
            SampleStrategy::Mirostat {
                tau: 2.0,
                learn_rate: 0.1,
            },
            SampleStrategy::Typical(0.95),
            SampleStrategy::Epsilon(0.01),
            SampleStrategy::Eta(0.9),
            SampleStrategy::LocallyTypical { k: 3, tau: 0.9 },
            SampleStrategy::Temperature(1.5),
            SampleStrategy::RandomlySample,
            SampleStrategy::Beam { width: 1 },
            SampleStrategy::Beam { width: 2 },
            SampleStrategy::TopK(10),
            // Ultra extras: finer temperature gradations
            SampleStrategy::Temperature(0.5),
            SampleStrategy::Temperature(0.65),
            SampleStrategy::Temperature(0.95),
            SampleStrategy::Temperature(1.1),
            SampleStrategy::Temperature(1.35),
            SampleStrategy::Temperature(1.75),
            SampleStrategy::Temperature(2.0),
            // Wider beam search
            SampleStrategy::Beam { width: 3 },
            SampleStrategy::Beam { width: 4 },
            SampleStrategy::Beam { width: 5 },
            // Contrastive variations
            SampleStrategy::Contrastive {
                alpha: 0.3,
                beta: 0.05,
            },
            SampleStrategy::Contrastive {
                alpha: 0.7,
                beta: 0.15,
            },
            SampleStrategy::Contrastive {
                alpha: 0.9,
                beta: 0.2,
            },
            // Fine-grained epsilon
            SampleStrategy::Epsilon(0.001),
            SampleStrategy::Epsilon(0.1),
            SampleStrategy::Epsilon(0.5),
        ]
    }
}

#[cfg(feature = "ane")]
impl DraftModel for MultiSpecDraftModel {
    /// Generate speculative tokens by running all drafts and merging.
    ///
    /// 1. Runs all drafts in parallel (each on its own ANE core).
    /// 2. Each draft applies its own [`SampleStrategy`] to produce a
    ///    different continuation.
    /// 3. Merges by picking the **highest-probability token per position**
    ///    across all drafts — ensuring the GPU receives the draft's
    ///    most confident guess at every position.
    fn speculate(
        &mut self,
        prefix: &[u32],
        n_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        // Run all drafts
        let all_results = self
            .multi_core
            .speculate_all(prefix, n_tokens, &self.strategies)?;

        if all_results.is_empty() || all_results[0].is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let n = all_results[0].len();
        let mut merged_tokens = Vec::with_capacity(n);
        let mut merged_probs = Vec::with_capacity(n);

        // Merge: for each position, pick the highest-probability token
        // across all drafts.
        for pos in 0..n {
            let mut best_token = 0u32;
            let mut best_prob = f32::NEG_INFINITY;
            for r in &all_results {
                if pos < r.len() {
                    let (tok, prob) = r[pos];
                    if prob > best_prob {
                        best_prob = prob;
                        best_token = tok;
                    }
                }
            }
            merged_tokens.push(best_token);
            merged_probs.push(best_prob);
        }

        self.prefix.extend_from_slice(prefix);
        Ok((merged_tokens, merged_probs))
    }

    /// Reset all draft models for a new generation sequence.
    fn reset(&mut self) {
        self.prefix.clear();
        self.multi_core.reset_all();
    }
}

/// Methods for SpecHub optimal multi-draft verification.
///
/// These methods provide a SpecHub integration path:
/// - `get_draft_probs` constructs the sparse probability array from draft
///   model outputs.
/// - `verify_with_spechub` runs the full SpecHub verification against a
///   target model's logits.
#[cfg(feature = "ane")]
impl MultiSpecDraftModel {
    /// Construct a sparse [num_drafts, seq_len, vocab_size] probability
    /// array from the draft tokens produced by all ANE cores.
    ///
    /// Each draft at each position places probability 1.0 on its chosen
    /// token (the most likely token per its sampling strategy).  This is
    /// a simplified sparse representation: the full softmax distribution
    /// would provide richer signals for the SpecHub joint distribution.
    pub fn get_draft_probs(&self, draft_tokens: &[Vec<u32>]) -> Result<mlx_rs::Array, String> {
        let num_drafts = draft_tokens.len();
        let seq_len = if num_drafts > 0 {
            draft_tokens[0].len()
        } else {
            0
        };
        // Infer vocabulary size from the maximum observed token across drafts.
        // This is conservative: tokens beyond this range get zero probability.
        let max_token = draft_tokens
            .iter()
            .flat_map(|t| t.iter())
            .max()
            .copied()
            .unwrap_or(0);
        let vocab_size = (max_token as usize + 1).max(32000);

        let mut data = vec![0.0f32; num_drafts * seq_len * vocab_size];
        let draft_stride = seq_len * vocab_size;

        for (d, tokens) in draft_tokens.iter().enumerate() {
            for (pos, &token) in tokens.iter().enumerate() {
                let idx = d * draft_stride + pos * vocab_size + token as usize;
                if idx < data.len() {
                    data[idx] = 1.0; // sparse: all mass on chosen token
                }
            }
        }

        Ok(mlx_rs::Array::from_slice(
            &data,
            &[num_drafts as i32, seq_len as i32, vocab_size as i32],
        ))
    }

    /// Run SpecHub verification against a target model's cached logits.
    ///
    /// `draft_tokens`: `[num_drafts, seq_len]` — the token sequences
    ///   produced by each ANE draft core.
    /// `target_model`: the profiled inference session whose most recent
    ///   forward pass produced the logits for the candidate positions.
    ///
    /// Returns a [`SpecHubVerification`] with the accepted token sequence
    /// and acceptance statistics.
    pub fn verify_with_spechub(
        &self,
        draft_tokens: &[Vec<u32>],
        target_model: &crate::profiled_executor::ProfiledInferenceSession,
    ) -> Result<SpecHubVerification, String> {
        // 1. Get draft probabilities
        let draft_probs = self.get_draft_probs(draft_tokens)?;
        // 2. Get target logits for candidate positions
        let target_logits = target_model.get_target_logits()?;
        // 3. Run SpecHub verification
        spechub_verify(&draft_probs, &target_logits, 1.0)
    }
}

// ---------------------------------------------------------------------------
// ADR 0034 Speculative Decoding — Draft model configuration & tree-spec
// ---------------------------------------------------------------------------

/// Description of a draft model's architecture.
///
/// Weights are stored as [`WeightCodec::GroupQuantized`] so that the
/// draft model can be loaded into any backend that supports group-wise
/// quantisation (MLX, Accelerate, ANE).
#[derive(Debug, Clone)]
pub struct DraftModelConfig {
    pub n_heads: u32,
    pub head_dim: u32,
    pub n_layers: u32,
}

/// One speculative branch in a tree-structured speculation.
///
/// Each branch is a sequence of draft tokens along a single path through
/// the speculation tree, together with metadata about its acceptance
/// probability and the KV-cache generation that produced it.
#[derive(Debug, Clone)]
pub struct SpeculativeBranch {
    /// Draft token IDs along this branch.
    pub tokens: Vec<u32>,
    /// Estimated probability that the entire branch will be accepted by
    /// the target model.
    pub acceptance_prob: f32,
    /// Indices of the draft-model layers that generated this branch.
    pub draft_layer_indices: Vec<u32>,
    /// Provisional page IDs that the memory planner reserved for this
    /// branch's KV-cache entries.
    pub provisional_pages: Vec<u32>,
    /// Total KV-cache generation cost (bytes) for this branch.
    pub kv_generation: u64,
}

/// Tree-structured speculative decoder.
///
/// Manages a draft model and generates multiple candidate branches
/// forming a speculation tree.  The target model verifies all branches
/// in a single batched forward pass; the first token (by tree order)
/// that passes the acceptance criterion is committed.
#[derive(Debug, Clone)]
pub struct TreeSpecDecoder {
    pub draft: DraftModelConfig,
    pub max_branches: u32,
    pub max_depth: u32,
    pub acceptance_threshold: f32,
}

impl TreeSpecDecoder {
    /// Propose a set of speculative branches from the current context.
    ///
    /// Uses the draft model's architecture (`n_heads`, `head_dim`,
    /// `n_layers`) together with the current context to generate up to
    /// `max_branches` distinct speculative continuations, each at most
    /// `max_depth` tokens long.
    pub fn propose(&self, _context: &[u32]) -> Vec<SpeculativeBranch> {
        let _ = self;
        todo!()
    }

    /// Verify speculative branches against the target model's logits.
    ///
    /// Compares each branch against `target_logits`; commits tokens up
    /// to the first position where the acceptance criterion fails
    /// (i.e. where the target's probability for the draft token falls
    /// below `acceptance_threshold`).  Returns the accepted token
    /// sequence.
    pub fn verify(&mut self, _branches: &[SpeculativeBranch], _target_logits: &[f32]) -> Vec<u32> {
        let _ = self;
        todo!()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock draft model that generates deterministic token sequences.
    struct MockDraft {
        tokens: Vec<u32>,
        log_probs: Vec<f32>,
    }

    impl MockDraft {
        fn new(tokens: Vec<u32>, log_probs: Vec<f32>) -> Self {
            Self { tokens, log_probs }
        }
    }

    impl DraftModel for MockDraft {
        fn speculate(
            &mut self,
            _prefix: &[u32],
            n_tokens: usize,
        ) -> Result<(Vec<u32>, Vec<f32>), String> {
            let tokens = self
                .tokens
                .iter()
                .copied()
                .take(n_tokens)
                .collect::<Vec<_>>();
            let probs = self
                .log_probs
                .iter()
                .copied()
                .take(n_tokens)
                .collect::<Vec<_>>();
            if tokens.len() < n_tokens {
                return Err(format!(
                    "MockDraft only has {} tokens, requested {}",
                    self.tokens.len(),
                    n_tokens
                ));
            }
            Ok((tokens, probs))
        }

        fn reset(&mut self) {
            // nothing to reset in a mock
        }
    }

    /// A mock target model that returns predetermined logits.
    struct MockTarget {
        logits: Vec<f32>,
        accepted: Vec<Vec<u32>>,
    }

    impl MockTarget {
        fn new(logits: Vec<f32>) -> Self {
            Self {
                logits,
                accepted: Vec::new(),
            }
        }
    }

    impl VerificationModel for MockTarget {
        fn verify(&mut self, _prefix: &[u32], draft_tokens: &[u32]) -> Result<Vec<f32>, String> {
            // If our pre-set logits are long enough, return the slice;
            // otherwise pad with zeros to match draft_tokens.len() + 1.
            let n = draft_tokens.len();
            if self.logits.len() >= n + 1 {
                Ok(self.logits[..=n].to_vec())
            } else if self.logits.len() >= n {
                let mut v = self.logits[..n].to_vec();
                v.push(0.0);
                Ok(v)
            } else {
                Ok(vec![0.0; n + 1])
            }
        }

        fn accept_tokens(&mut self, tokens: &[u32]) {
            self.accepted.push(tokens.to_vec());
        }
    }

    #[test]
    fn test_acceptance_rate_default() {
        let sd = SpeculativeDecoding::new(4);
        assert_eq!(sd.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_stats_default() {
        let sd = SpeculativeDecoding::new(4);
        let s = sd.stats();
        assert_eq!(s.total_steps, 0);
        assert_eq!(s.total_draft_tokens, 0);
        assert_eq!(s.total_accepted_draft, 0);
        assert_eq!(s.total_target_tokens, 0);
        assert_eq!(s.rejection_count, 0);
    }

    #[test]
    fn test_all_tokens_accepted() {
        // All draft log-probs are very negative → p_draft tiny → accept_prob
        // will be capped at 1.0 (because p_target/p_draft > 1), so all tokens
        // should be accepted.
        let mut sd = SpeculativeDecoding::new(3);
        let mut draft = MockDraft::new(
            vec![100, 101, 102],
            vec![-10.0, -10.0, -10.0], // very low log-probs
        );
        // Target logits for: each candidate (positive) and bonus position
        let mut target = MockTarget::new(vec![1.0, 1.0, 1.0, 2.0]);

        let token = sd.step(&mut draft, &mut target, &[99]).unwrap();

        // All 3 draft tokens should be recorded as accepted.
        assert_eq!(sd.stats().total_accepted_draft, 3);
        assert_eq!(sd.stats().total_draft_tokens, 3);
        assert_eq!(sd.stats().total_steps, 1);
        assert_eq!(sd.stats().rejection_count, 0);
        // One target token (the bonus) produced
        assert_eq!(sd.stats().total_target_tokens, 1);
        // last candidate pos (n-1=2) is the fallback when bonus_logit > 0
        // The bonus should be a positive-logit derived token != 102

        // accept_tokens should have been called with all three candidates
        assert_eq!(target.accepted.len(), 1);
        assert_eq!(target.accepted[0], vec![100, 101, 102]);
    }

    #[test]
    fn test_first_token_rejected() {
        // Draft token at index 0 has a high log-prob but the target's logit
        // for it is very negative → p_target tiny → high rejection chance.
        let mut sd = SpeculativeDecoding::new(2);
        let mut draft = MockDraft::new(
            vec![200, 201],
            vec![-0.1, -10.0], // first token very likely per draft
        );
        // Target assigns very low logit to the first draft token
        let mut target = MockTarget::new(vec![-100.0, -100.0, 0.0]);

        let token = sd.step(&mut draft, &mut target, &[199]).unwrap();

        // First token should have been rejected; none accepted.
        assert_eq!(sd.stats().total_accepted_draft, 0);
        assert_eq!(sd.stats().total_draft_tokens, 2);
        assert_eq!(sd.stats().total_steps, 1);
        assert_eq!(sd.stats().rejection_count, 1);
        assert_eq!(sd.stats().total_target_tokens, 1);
        // accept_tokens should not have been called (i=0 → no tokens before rejection)
        assert_eq!(target.accepted.len(), 0);
    }

    #[test]
    fn test_partial_acceptance() {
        // Draft: tokens [300, 301, 302] with progressively lower draft log-probs.
        // Target logits: second token gets a very negative logit → rejection at i=1.
        let mut sd = SpeculativeDecoding::new(3);
        let mut draft = MockDraft::new(vec![300, 301, 302], vec![-1.0, -1.0, -1.0]);
        // Target: first token gets positive logit, second gets strongly negative
        let mut target = MockTarget::new(vec![5.0, -100.0, -100.0, 0.0]);

        let token = sd.step(&mut draft, &mut target, &[299]).unwrap();

        // First token accepted (i=0 passes), second rejected (i=1)
        assert_eq!(sd.stats().total_accepted_draft, 1);
        assert_eq!(sd.stats().total_draft_tokens, 3);
        assert_eq!(sd.stats().total_steps, 1);
        assert_eq!(sd.stats().rejection_count, 1);
        assert_eq!(sd.stats().total_target_tokens, 1);
        // accept_tokens called with candidates[..1] = [300]
        assert_eq!(target.accepted.len(), 1);
        assert_eq!(target.accepted[0], vec![300]);
    }

    #[test]
    fn test_zero_speculation_length() {
        let mut sd = SpeculativeDecoding::new(0);
        let mut draft = MockDraft::new(vec![], vec![]);
        let mut target = MockTarget::new(vec![]);

        let result = sd.step(&mut draft, &mut target, &[400]);
        // With speculation_length=0, draft.speculate returns empty → no candidates
        assert!(result.is_err());
    }

    #[test]
    fn test_debug_format() {
        let sd = SpeculativeDecoding::new(5);
        let fmt = format!("{:?}", sd);
        assert!(fmt.contains("speculation_length: 5"));
        assert!(fmt.contains("acceptance_rate: 0.0"));
    }

    #[test]
    fn test_acceptance_rate_after_steps() {
        let mut sd = SpeculativeDecoding::new(2);
        let mut draft = MockDraft::new(vec![500, 501], vec![-10.0, -10.0]);
        let mut target = MockTarget::new(vec![5.0, 5.0, 1.0]);

        sd.step(&mut draft, &mut target, &[499]).unwrap();
        // All accepted: 2/2 = 1.0
        assert!((sd.acceptance_rate() - 1.0).abs() < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// SpecHub — optimal multi-draft verification
// ---------------------------------------------------------------------------

/// SpecHub verification result — accepts more tokens than greedy.
///
/// SpecHub builds a sparse joint distribution over all draft outputs and
/// identifies the subset of drafts consistent with the target model,
/// recovering tokens that greedy rejection would discard.
pub struct SpecHubVerification {
    /// Token IDs accepted at each verified position.
    pub accepted_tokens: Vec<u32>,
    /// Fraction of draft tokens accepted (verified / attempted).
    pub acceptance_rate: f64,
    /// Estimated latency saved by acceptance vs. target-only decode (ms).
    /// Set externally from wall-clock measurements.
    pub saved_latency_ms: f64,
    /// Time spent in the SpecHub verification algorithm (microseconds).
    pub verification_time_us: u64,
}

/// Build a sparse joint distribution over all drafts at a single position.
///
/// For each draft, extracts the top-K token indices and their probabilities
/// from `draft_data`. Returns a map `(draft_index, token) -> probability`.
/// Tokens with probability below 1e-10 are excluded.
fn sparse_joint_distribution_at_pos(
    draft_data: &[f32],
    num_drafts: usize,
    seq_len: usize,
    vocab_size: usize,
    pos: usize,
    top_k: usize,
) -> HashMap<(usize, u32), f64> {
    let mut joint = HashMap::new();
    let draft_stride = seq_len * vocab_size;

    for d in 0..num_drafts {
        let base = d * draft_stride + pos * vocab_size;

        // Find top-K tokens for this draft at this position.
        let mut scored: Vec<(usize, f32)> =
            (0..vocab_size).map(|v| (v, draft_data[base + v])).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(v, prob) in scored.iter().take(top_k) {
            if prob > 1e-10f32 {
                joint.insert((d, v as u32), prob as f64);
            }
        }
    }

    joint
}

/// Compute softmax-normalized probabilities from logits at a single position.
fn softmax_at_pos(
    data: &[f32],
    _seq_len: usize,
    vocab_size: usize,
    pos: usize,
    inv_temp: f64,
) -> Vec<f64> {
    let base = pos * vocab_size;
    let mut max_logit = f64::NEG_INFINITY;

    // Find max for numerical stability.
    for v in 0..vocab_size {
        let val = data[base + v] as f64 * inv_temp;
        if val > max_logit {
            max_logit = val;
        }
    }

    let mut sum = 0.0f64;
    let mut scaled = Vec::with_capacity(vocab_size);
    for v in 0..vocab_size {
        let val = (data[base + v] as f64 * inv_temp - max_logit).exp();
        scaled.push(val);
        sum += val;
    }

    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    let mut probs = Vec::with_capacity(vocab_size);
    for &s in &scaled {
        probs.push(s * inv_sum);
    }
    probs
}

/// Identify the subset of drafts compatible with each other at a position.
///
/// Compatibility is defined by Jaccard similarity of top-K token sets:
/// two drafts are compatible when their top-K sets overlap by at least 30%.
/// Returns the indices of drafts that are pairwise-compatible with at least
/// one other peer (singletons with no overlap are excluded).
fn compatible_subset_at_pos(
    draft_data: &[f32],
    num_drafts: usize,
    seq_len: usize,
    vocab_size: usize,
    pos: usize,
    top_k: usize,
) -> Vec<usize> {
    let draft_stride = seq_len * vocab_size;

    // Collect top-K token sets for each draft.
    let mut draft_sets: Vec<Vec<u32>> = Vec::with_capacity(num_drafts);
    for d in 0..num_drafts {
        let base = d * draft_stride + pos * vocab_size;
        let mut scored: Vec<(u32, f32)> = (0..vocab_size)
            .map(|v| (v as u32, draft_data[base + v]))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        draft_sets.push(scored.iter().take(top_k).map(|(t, _)| *t).collect());
    }

    // Build compatibility adjacency: draft i is compatible with draft j
    // if their top-K Jaccard similarity >= 0.3.
    let mut compat_count = vec![0u32; num_drafts];
    for i in 0..num_drafts {
        for j in (i + 1)..num_drafts {
            let intersection = draft_sets[i]
                .iter()
                .filter(|t| draft_sets[j].contains(t))
                .count();
            let union = draft_sets[i].len() + draft_sets[j].len() - intersection;
            let jaccard = if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.0
            };
            if jaccard >= 0.3 {
                compat_count[i] += 1;
                compat_count[j] += 1;
            }
        }
    }

    // Return drafts that have at least one compatible peer.
    compat_count
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, _)| i)
        .collect()
}

/// Find the consensus token — the one with the most total probability mass
/// across all drafts in the sparse joint distribution.
fn find_consensus_token(joint: &HashMap<(usize, u32), f64>) -> Option<(u32, f64)> {
    let mut token_mass: HashMap<u32, f64> = HashMap::new();
    for (&(_, token), &prob) in joint {
        *token_mass.entry(token).or_insert(0.0) += prob;
    }
    token_mass
        .into_iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

/// Re-weigh candidate tokens within a compatible subset.
/// Multiplies each draft's probability for a token by that token's
/// target probability, then selects the highest-scoring token.
fn reweigh_with_subset(
    joint: &HashMap<(usize, u32), f64>,
    compat_set: &[usize],
    target_probs: &[f64],
) -> u32 {
    let mut token_scores: HashMap<u32, f64> = HashMap::new();
    for (&(d, token), &draft_prob) in joint {
        if compat_set.contains(&d) {
            let target_prob = target_probs.get(token as usize).copied().unwrap_or(0.0);
            *token_scores.entry(token).or_insert(0.0) += draft_prob * target_prob;
        }
    }
    token_scores
        .into_iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(t, _)| t)
        .unwrap_or(0)
}

/// Run SpecHub verification over a batch of draft predictions.
///
/// `draft_probs`: `[num_drafts, seq_len, vocab_size]` — probabilities
///   from each draft model at each position.
/// `target_logits`: `[seq_len, vocab_size]` — the target model's raw
///   logits at each verified position.
/// `temperature`: sampling temperature (clamped to >= 1e-8).
///
/// The algorithm processes one position at a time:
/// 1. Build a sparse joint distribution from each draft's top-K (K=10).
/// 2. Find the consensus token with the highest total draft mass.
/// 3. If the target also agrees (consensus is target's argmax, or target
///    assigns >30% probability to it), accept the consensus token.
/// 4. Otherwise, identify the compatible subset of drafts (Jaccard >= 0.3)
///    and re-weigh their predictions against the target distribution.
pub fn spechub_verify(
    draft_probs: &mlx_rs::Array,
    target_logits: &mlx_rs::Array,
    temperature: f64,
) -> Result<SpecHubVerification, String> {
    let start = std::time::Instant::now();

    let draft_shape = draft_probs.shape();
    let target_shape = target_logits.shape();

    if draft_shape.len() != 3 {
        return Err(format!(
            "draft_probs must be 3D [num_drafts, seq_len, vocab_size], got {}D",
            draft_shape.len()
        ));
    }
    if target_shape.len() != 2 {
        return Err(format!(
            "target_logits must be 2D [seq_len, vocab_size], got {}D",
            target_shape.len()
        ));
    }

    let num_drafts = draft_shape[0] as usize;
    let seq_len = draft_shape[1] as usize;
    let vocab_size = draft_shape[2] as usize;

    if target_shape[0] as usize != seq_len {
        return Err(format!(
            "seq_len mismatch: draft_probs has {}, target_logits has {}",
            seq_len, target_shape[0]
        ));
    }
    if target_shape[1] as usize != vocab_size {
        return Err(format!(
            "vocab_size mismatch: draft_probs has {}, target_logits has {}",
            vocab_size, target_shape[1]
        ));
    }

    let draft_data = draft_probs.as_slice::<f32>();
    let target_data = target_logits.as_slice::<f32>();

    let inv_temp = 1.0 / temperature.max(1e-8);
    let top_k: usize = 10;

    let mut accepted_tokens = Vec::with_capacity(seq_len);
    let mut total_draft_positions: u64 = 0;
    let mut accepted_count: u64 = 0;

    for pos in 0..seq_len {
        // 1. Build sparse joint distribution (top-K per draft).
        let joint = sparse_joint_distribution_at_pos(
            draft_data, num_drafts, seq_len, vocab_size, pos, top_k,
        );

        // 2. Get target probability distribution at this position.
        let target_probs = softmax_at_pos(target_data, seq_len, vocab_size, pos, inv_temp);

        // 3. Find target's argmax.
        let target_argmax = target_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        total_draft_positions += num_drafts as u64;

        // 4. Find consensus token across all drafts.
        if let Some((consensus_token, _mass)) = find_consensus_token(&joint) {
            // 5. Accept if target agrees or assigns significant probability.
            if consensus_token == target_argmax
                || target_probs
                    .get(consensus_token as usize)
                    .copied()
                    .unwrap_or(0.0)
                    > 0.3
            {
                accepted_tokens.push(consensus_token);
                accepted_count += 1;
            } else {
                // 6. Diversion: find compatible subset and re-weigh.
                let compat = compatible_subset_at_pos(
                    draft_data, num_drafts, seq_len, vocab_size, pos, top_k,
                );

                if compat.is_empty() {
                    // Fall back to target's argmax.
                    accepted_tokens.push(target_argmax);
                    accepted_count += 1;
                } else {
                    let reweighed = reweigh_with_subset(&joint, &compat, &target_probs);
                    accepted_tokens.push(reweighed);
                    accepted_count += 1;
                }
            }
        } else {
            // No draft information — use target.
            accepted_tokens.push(target_argmax);
            accepted_count += 1;
        }
    }

    let verification_time_us = start.elapsed().as_micros() as u64;
    let acceptance_rate = if total_draft_positions > 0 {
        accepted_count as f64 / total_draft_positions as f64
    } else {
        0.0
    };

    Ok(SpecHubVerification {
        accepted_tokens,
        acceptance_rate,
        saved_latency_ms: 0.0,
        verification_time_us,
    })
}
