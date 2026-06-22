//! Batch construction utilities for the continuous batching scheduler.
//!
//! Reference: `ref/omlx/scheduler.py`

use super::{Batch, HardwareConfig, Request, Slot};

/// Build a prefill batch from queued requests
pub fn build_prefill_batch(requests: &[Request], max_size: usize) -> Batch {
    let slots: Vec<Slot> = requests
        .iter()
        .take(max_size)
        .enumerate()
        .map(|(i, req)| Slot {
            id: i,
            request_id: Some(req.id),
            tokens_generated: 0,
            kv_cache_start: 0,
            kv_cache_length: req.prompt.len(),
            backend_id: 0,
            kv_cache_pages: vec![],
        })
        .collect();

    Batch {
        slots: slots.clone(),
        batch_size: slots.len(),
        max_batch_size: max_size,
    }
}

/// Build a decode batch from active requests
pub fn build_decode_batch(active: &[Request], max_size: usize) -> Batch {
    let slots: Vec<Slot> = active
        .iter()
        .take(max_size)
        .enumerate()
        .map(|(i, req)| Slot {
            id: i,
            request_id: Some(req.id),
            tokens_generated: req.max_tokens,
            kv_cache_start: 0,
            kv_cache_length: req.max_tokens,
            backend_id: 0,
            kv_cache_pages: vec![],
        })
        .collect();

    Batch {
        slots: slots.clone(),
        batch_size: slots.len(),
        max_batch_size: max_size,
    }
}

// ---------------------------------------------------------------------------
// BatchedPrefill — concatenate multiple prompts for batched forward pass
// ---------------------------------------------------------------------------

/// Concatenate multiple prompts into a single batched forward pass.
///
/// MLX handles batched inputs naturally by stacking sequences along the
/// batch dimension.  This is the key throughput optimization for memory-rich
/// hardware (e.g. M3 Ultra with 512 GB): instead of serial prefills, we
/// merge all queued prompts into one forward pass.
///
/// # Usage
///
/// ```ignore
/// let batched = BatchedPrefill::new(&queued_prompts, 4096);
/// let first_tokens = batched.execute(&model)?;
/// ```
pub struct BatchedPrefill {
    /// The prompts to process, each as a vector of token IDs.
    pub prompts: Vec<Vec<u32>>,
    /// Maximum sequence length in tokens (pads shorter prompts).
    pub max_seq_len: u32,
}

impl BatchedPrefill {
    /// Create a new batched prefill from the given prompts.
    pub fn new(prompts: Vec<Vec<u32>>, max_seq_len: u32) -> Self {
        Self {
            prompts,
            max_seq_len,
        }
    }

    /// Create a batched prefill configured for the detected hardware.
    pub fn new_for_hardware(hw: &HardwareConfig, model_prompt: Vec<u32>) -> Self {
        // Duplicate the prompt across the batch dimension.  In real usage
        // each slot has its own prompt; this is a convenience for single
        // model-server startup or warmup.
        let count = hw.recommended_batch_size as usize;
        let prompts = vec![model_prompt; count];
        Self {
            prompts,
            max_seq_len: 262_144,
        }
    }

    /// Run a single batched prefill for all queued prompts.
    ///
    /// Returns the first sampled token for each sequence in the batch,
    /// one per prompt in order.
    pub fn execute(
        &self,
        _model: &crate::profiled_executor::LoadedProfiledModel,
    ) -> Result<Vec<u32>, String> {
        let batch_size = self.prompts.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // In a production deployment, this unpacks into the model runtime's
        // batched prefill path (MLX / Metal / CoreML).  The exact mechanism
        // depends on the backend, but the principle is the same: stack
        // all prompt sequences into one tensor and run a single forward pass.
        //
        // Placeholder — the batched prefill kernel is upstream of this crate
        // in the model runtime's `batch_prefill()` entry point.
        let _seq_lens: Vec<usize> = self.prompts.iter().map(|p| p.len()).collect();

        // Verify the model can hold the longest sequence.
        let max_prompt_len = _seq_lens.iter().max().copied().unwrap_or(0);
        if max_prompt_len as u32 > self.max_seq_len {
            return Err(format!(
                "Prompt length {} exceeds max sequence length {}",
                max_prompt_len, self.max_seq_len
            ));
        }

        // Return placeholder tokens (one per batch slot).
        // In production this calls `model.batch_prefill(&self.prompts)`.
        Ok(vec![0u32; batch_size])
    }
}

/// Batch multiple decode steps into a single forward pass.
///
/// When decode batching is enabled, the scheduler collects pending decode
/// slots and runs them as a single tensor operation rather than N serial
/// steps.  The MLX backend handles this automatically when multiple
/// sequences share the same model weights.
pub fn batch_decode_steps(slots: &[Slot]) -> Vec<Vec<u32>> {
    slots
        .iter()
        .map(|s| {
            // Each slot emits its next token.  In production this calls
            // into the model runtime's batched decode.
            vec![s.id as u32]
        })
        .collect()
}
