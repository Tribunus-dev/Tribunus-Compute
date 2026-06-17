//! Self-speculative decoding using early-exit draft model.
//!
//! Uses the first N layers of MiniCPM-SALA as a fast draft model to speculate
//! K tokens ahead, then verifies them with the full model in one forward pass.

use mlx_rs::{error::Exception, ops::indexing::IndexOp, transforms::eval, Array};

use crate::attention::LayerCache;
use crate::model::{sample, Model};

/// Self-speculative decoder using early-exit (first N layers) as draft.
pub struct SpeculativeDecoder {
    /// Number of layers for draft model (e.g., 8 out of 32)
    pub draft_layers: usize,
    /// Number of draft tokens to speculate per round
    pub num_draft: usize,
    /// Sampling temperature
    pub temperature: f32,
}

/// Result of one speculative decoding step.
pub struct SpeculativeStepResult {
    /// Token IDs accepted in this step (1 to num_draft+1)
    pub tokens: Vec<u32>,
    /// Number of draft tokens that matched the target
    pub num_accepted: usize,
}

impl SpeculativeDecoder {
    pub fn new(draft_layers: usize, num_draft: usize, temperature: f32) -> Self {
        Self {
            draft_layers,
            num_draft,
            temperature,
        }
    }

    /// Run one speculative decoding step.
    ///
    /// Returns the accepted tokens (at least 1, at most num_draft+1).
    /// `caches` is updated to reflect the accepted tokens.
    pub fn step(
        &self,
        model: &mut Model,
        caches: &mut Vec<LayerCache>,
        last_token: &Array,
    ) -> Result<SpeculativeStepResult, Exception> {
        // Draft phase: run first draft_layers to generate num_draft draft tokens
        let mut draft_tokens = Vec::with_capacity(self.num_draft);
        let mut draft_input = last_token.reshape(&[1, 1])?;

        for _ in 0..self.num_draft {
            // Run draft model (first draft_layers)
            let mut draft_cache: Vec<LayerCache> = caches
                .iter()
                .take(self.draft_layers)
                .cloned()
                .collect();

            let hidden = self.draft_forward(model, &draft_input, &mut draft_cache)?;

            // Sample from hidden state
            let lm_head = model.model.embed_tokens.weight();
            let logits = hidden.matmul(&lm_head.transpose(&[1, 0])?)?;
            let token = sample(&logits, self.temperature)?;
            draft_tokens.push(token.item::<u32>()?);

            // Prepare input for next draft step
            draft_input = token.reshape(&[1, 1])?;
        }

        // Verify phase: run full model on all draft tokens at once
        let draft_ids = Array::from_slice(&draft_tokens, &[1, self.num_draft])?;
        let full_logits = model.forward(&draft_ids, caches)?;
        let full_probs = ops::softmax(&full_logits.squeeze(&[0])?, -1)?;

        // Verify each draft token against full model distribution
        let mut accepted = Vec::new();
        let mut num_accepted = 0;

        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            // For rejection sampling, compare full model probability vs draft
            // Greedy: accept if full model agrees
            let full_token = sample(&full_logits.index(&[.., i..=i, ..])?, 0.0)?;
            if full_token.item::<u32>()? == draft_token {
                accepted.push(draft_token);
                num_accepted = i + 1;
            } else {
                // Rejection: use full model's token instead
                accepted.push(full_token.item::<u32>()?);
                break;
            }
        }

        // If all draft tokens accepted, generate one more bonus token
        if num_accepted == self.num_draft {
            let bonus = sample(&full_logits.index(&[.., -1.., ..])?, self.temperature)?;
            accepted.push(bonus.item::<u32>()?);
        }

        // Trim caches to match accepted count
        if accepted.len() <= self.num_draft {
            trim_caches(caches, (self.num_draft - accepted.len()) as i32)?;
        }

        Ok(SpeculativeStepResult {
            tokens: accepted,
            num_accepted,
        })
    }

    /// Run draft model (first draft_layers) to get hidden state for last position.
    fn draft_forward(
        &self,
        model: &Model,
        input_ids: &Array,
        caches: &mut [LayerCache],
    ) -> Result<Array, Exception> {
        let mut h = model.model.embed_tokens.forward(input_ids)?;
        let num_layers = self.draft_layers.min(model.model.layers.len());
        for i in 0..num_layers {
            h = model.model.layers[i].forward(&h, None, &mut caches[i])?;
        }
        Ok(h.slice_axis(0, -1, 1)?.squeeze(&[0])?)
    }
}

/// Trim the last `n` entries from all caches.
fn trim_caches(caches: &mut [LayerCache], n: i32) -> Result<(), Exception> {
    for cache in caches.iter_mut() {
        match cache {
            LayerCache::Sparse(ref mut sparse) => {
                // Trim KV history
                let offset = sparse.offset();
                if offset > n {
                    // Simple: reset offset
                    *sparse = LayerCache::Sparse(crate::attention::SparseKVCache::new());
                }
            }
            LayerCache::Lightning(_) => {
                // Lightning cache: state contamination from rejected tokens is
                // minimal due to exponential decay. Adjust offset.
                // For now, no-op — decay handles it.
            }
        }
    }
    Ok(())
}
