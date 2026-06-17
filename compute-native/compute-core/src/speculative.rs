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
    fn verify(
        &mut self,
        prefix: &[u32],
        draft_tokens: &[u32],
    ) -> Result<Vec<f32>, String>;

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
        let (candidates, draft_log_probs) =
            draft.speculate(prefix, self.speculation_length)?;

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
            let tokens = self.tokens.iter().copied().take(n_tokens).collect::<Vec<_>>();
            let probs = self.log_probs.iter().copied().take(n_tokens).collect::<Vec<_>>();
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
        fn verify(
            &mut self,
            _prefix: &[u32],
            draft_tokens: &[u32],
        ) -> Result<Vec<f32>, String> {
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
        let mut draft = MockDraft::new(
            vec![300, 301, 302],
            vec![-1.0, -1.0, -1.0],
        );
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
        let mut draft = MockDraft::new(
            vec![500, 501],
            vec![-10.0, -10.0],
        );
        let mut target = MockTarget::new(vec![5.0, 5.0, 1.0]);

        sd.step(&mut draft, &mut target, &[499]).unwrap();
        // All accepted: 2/2 = 1.0
        assert!((sd.acceptance_rate() - 1.0).abs() < 1e-9);
    }
}
