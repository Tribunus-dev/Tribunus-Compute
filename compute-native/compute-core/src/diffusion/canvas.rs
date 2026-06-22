//! Token canvas for discrete masked diffusion.
//!
//! Manages the evolving state of token positions across diffusion steps:
//! which tokens are committed, their confidence scores, and unresolved
//! positions awaiting generation.

use super::sampler::SamplerOutput;

/// Manages the token canvas state for discrete masked diffusion.
pub struct TokenCanvas {
    /// Maximum number of token positions.
    pub capacity: u32,
    /// Prompt tokens used to seed the canvas (prefix).
    pub prompt_tokens: Vec<u32>,
    /// Current tokens at each position (`None` = still masked/unset).
    pub tokens: Vec<Option<u32>>,
    /// Whether each position has been committed (locked in).
    pub committed: Vec<bool>,
    /// Confidence score for each position, in `[0, 1]`.
    pub confidence: Vec<f32>,
    /// Mask token ID used to fill unresolved positions.
    pub mask_token_id: u32,
    /// Padding token ID.
    pub pad_token_id: u32,
    /// How many steps since any position last changed.
    pub unchanged_steps: u32,
    /// Commit mask from the most recent sampling step.
    pub last_commit_mask: Vec<bool>,
    /// Running count of committed positions.
    pub total_committed: u32,
    /// Running count of positions not yet committed.
    pub total_unresolved: u32,
}

impl TokenCanvas {
    /// Create a new empty canvas with the given capacity.
    pub fn new(capacity: u32, mask_token_id: u32, pad_token_id: u32) -> Self {
        let capacity_usize = capacity as usize;
        Self {
            capacity,
            prompt_tokens: Vec::new(),
            tokens: vec![Some(mask_token_id); capacity_usize],
            committed: vec![false; capacity_usize],
            confidence: vec![0.0f32; capacity_usize],
            mask_token_id,
            pad_token_id,
            unchanged_steps: 0,
            last_commit_mask: vec![false; capacity_usize],
            total_committed: 0,
            total_unresolved: capacity,
        }
    }

    /// Seed the canvas from a prompt sequence.
    ///
    /// Prompt tokens occupy the first `prompt.len()` positions and are
    /// marked committed with full confidence. Remaining positions are
    /// initialized to the mask token.
    pub fn initialize_from_prompt(&mut self, prompt: &[u32]) {
        let prompt_len = prompt.len().min(self.capacity as usize);

        for i in 0..prompt_len {
            self.tokens[i] = Some(prompt[i]);
            self.committed[i] = true;
            self.confidence[i] = 1.0;
        }

        // Fill remaining positions with the mask token (already the default).
        for i in prompt_len..self.capacity as usize {
            self.tokens[i] = Some(self.mask_token_id);
            self.committed[i] = false;
            self.confidence[i] = 0.0;
        }

        self.prompt_tokens = prompt.to_vec();
        self.last_commit_mask = self.committed.clone();
        self.total_committed = prompt_len as u32;
        self.total_unresolved = self.capacity - self.total_committed;
        self.unchanged_steps = 0;
    }

    /// Update the canvas state from a sampler output.
    ///
    /// For each position:
    /// - If `commit_mask[pos]` is true, the token is committed and locked.
    /// - If `remask_mask[pos]` is true, the token is reset to mask (unresolved).
    /// - Otherwise, the token remains as-is.
    ///
    /// Returns the number of positions that changed state.
    pub fn update(&mut self, output: &SamplerOutput) -> u32 {
        let mut changed = 0u32;

        for i in 0..self.capacity as usize {
            if i >= output.token_ids.len() {
                break;
            }

            if i < self.committed.len() && self.committed[i] {
                // Already committed — no change.
                continue;
            }

            if output.commit_mask[i] {
                // Commit this token.
                self.tokens[i] = Some(output.token_ids[i]);
                self.committed[i] = true;
                self.confidence[i] = output.confidence_scores[i];
                changed += 1;
            } else if output.remask_mask[i] {
                // Remask: reset token and confidence.
                self.tokens[i] = Some(self.mask_token_id);
                self.confidence[i] = 0.0;
                changed += 1;
            } else {
                // Keep the current token but update its confidence.
                self.confidence[i] = output.confidence_scores[i];
            }
        }

        if changed == 0 {
            self.unchanged_steps += 1;
        } else {
            self.unchanged_steps = 0;
        }

        self.last_commit_mask = output.commit_mask.clone();
        self.total_committed = self.committed.iter().map(|&c| c as u32).sum();
        self.total_unresolved = self.num_unresolved();

        changed
    }

    /// Return the committed portion as a text string.
    ///
    /// Tokens are laid out as: prompt tokens followed by committed generated
    /// tokens. Uncommitted positions are skipped.
    pub fn committed_text(&self) -> String {
        let mut s = String::new();
        for i in 0..self.capacity as usize {
            if self.committed[i] {
                if let Some(tok) = self.tokens[i] {
                    if tok != self.pad_token_id && tok != self.mask_token_id {
                        // Append token as a character if it's in printable range.
                        if tok >= 32 && tok <= 126 {
                            s.push(char::from_u32(tok).unwrap_or('?'));
                        } else {
                            // Non-printable: emit as U+FFFD replacement char.
                            s.push('\u{FFFD}');
                        }
                    }
                }
            }
        }
        s
    }

    /// Fraction of positions (excluding the prompt) that are committed.
    ///
    /// Returns 0.0 when the canvas has no capacity or all prompt.
    /// Returns 1.0 when every non-prompt position is committed.
    pub fn resolution_ratio(&self) -> f32 {
        let prompt_len = self.prompt_tokens.len() as u32;
        if self.capacity <= prompt_len {
            return 1.0;
        }
        let generative = self.capacity - prompt_len;
        let committed_generative = self
            .committed
            .iter()
            .skip(prompt_len as usize)
            .filter(|&&c| c)
            .count() as u32;
        committed_generative as f32 / generative as f32
    }

    /// Whether every position is committed.
    pub fn all_committed(&self) -> bool {
        self.committed.iter().all(|&c| c)
    }

    /// Number of positions not yet committed.
    pub fn num_unresolved(&self) -> u32 {
        self.committed.iter().map(|&c| if c { 0 } else { 1 }).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion::sampler::SamplerOutput;

    #[test]
    fn test_canvas_new() {
        let canvas = TokenCanvas::new(10, 1, 0);
        assert_eq!(canvas.capacity, 10);
        assert_eq!(canvas.tokens.len(), 10);
        assert_eq!(canvas.committed.len(), 10);
        assert_eq!(canvas.confidence.len(), 10);
        assert_eq!(canvas.total_committed, 0);
        assert_eq!(canvas.total_unresolved, 10);
        assert!(canvas.tokens.iter().all(|t| *t == Some(1)));
    }

    #[test]
    fn test_initialize_from_prompt() {
        let mut canvas = TokenCanvas::new(8, 0, 0);
        let prompt = vec![10, 20, 30];
        canvas.initialize_from_prompt(&prompt);

        assert_eq!(canvas.total_committed, 3);
        assert_eq!(canvas.total_unresolved, 5);
        assert!(canvas.committed[0]);
        assert!(canvas.committed[1]);
        assert!(canvas.committed[2]);
        assert_eq!(canvas.tokens[0], Some(10));
        assert_eq!(canvas.tokens[1], Some(20));
        assert_eq!(canvas.tokens[2], Some(30));
        assert_eq!(canvas.tokens[3], Some(0));
        assert!(!canvas.committed[3]);
    }

    #[test]
    fn test_update_commits_tokens() {
        let mut canvas = TokenCanvas::new(4, 0, 0);
        canvas.initialize_from_prompt(&[]);

        let output = SamplerOutput {
            token_ids: vec![5, 6, 7, 8],
            confidence_scores: vec![0.9, 0.8, 0.3, 0.4],
            commit_mask: vec![true, true, false, false],
            remask_mask: vec![false, false, true, false],
            eos_triggered: false,
        };

        let changed = canvas.update(&output);
        assert_eq!(changed, 2, "two positions should change");

        assert!(canvas.committed[0]);
        assert!(canvas.committed[1]);
        assert!(!canvas.committed[2]); // not committed, remasked
        assert!(!canvas.committed[3]); // not committed, confidence updated

        assert_eq!(canvas.tokens[0], Some(5));
        assert_eq!(canvas.tokens[1], Some(6));
        // Remasked -> mask token.
        assert_eq!(canvas.tokens[2], Some(0));
        // Unchanged but confidence updated.
        assert_eq!(canvas.confidence[3], 0.4);

        assert_eq!(canvas.total_committed, 2);
        assert_eq!(canvas.total_unresolved, 2);
    }

    #[test]
    fn test_unchanged_steps_increment() {
        let mut canvas = TokenCanvas::new(3, 0, 0);

        // First update with no commits.
        let output1 = SamplerOutput {
            token_ids: vec![1, 2, 3],
            confidence_scores: vec![0.1, 0.2, 0.3],
            commit_mask: vec![false, false, false],
            remask_mask: vec![false, false, false],
            eos_triggered: false,
        };
        canvas.update(&output1);
        assert_eq!(canvas.unchanged_steps, 1);

        // Second update with no commits.
        let output2 = SamplerOutput {
            token_ids: vec![1, 2, 3],
            confidence_scores: vec![0.2, 0.3, 0.4],
            commit_mask: vec![false, false, false],
            remask_mask: vec![false, false, false],
            eos_triggered: false,
        };
        canvas.update(&output2);
        assert_eq!(canvas.unchanged_steps, 2);

        // Now commit something — should reset unchanged_steps.
        let output3 = SamplerOutput {
            token_ids: vec![1, 2, 3],
            confidence_scores: vec![0.9, 0.3, 0.4],
            commit_mask: vec![true, false, false],
            remask_mask: vec![false, false, false],
            eos_triggered: false,
        };
        canvas.update(&output3);
        assert_eq!(canvas.unchanged_steps, 0);
    }

    #[test]
    fn test_all_committed() {
        let mut canvas = TokenCanvas::new(3, 0, 0);
        assert!(!canvas.all_committed());

        canvas.committed = vec![true, true, true];
        canvas.total_committed = 3;
        assert!(canvas.all_committed());
    }

    #[test]
    fn test_resolution_ratio() {
        let mut canvas = TokenCanvas::new(10, 0, 0);
        canvas.initialize_from_prompt(&[1, 2, 3]);

        // Initially no generative positions committed.
        assert!((canvas.resolution_ratio() - 0.0).abs() < 1e-6);

        // Mark all as committed.
        for i in 3..10 {
            canvas.committed[i] = true;
        }
        canvas.total_committed = 10;

        // Now all 7 generative positions are committed.
        assert!((canvas.resolution_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_num_unresolved() {
        let mut canvas = TokenCanvas::new(5, 0, 0);
        assert_eq!(canvas.num_unresolved(), 5);

        canvas.committed = vec![true, false, true, false, false];
        assert_eq!(canvas.num_unresolved(), 3);
    }
}
