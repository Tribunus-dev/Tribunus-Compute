//! Arena requirements computation for SealedComputeImageExecutable.
//!
//! Provides [`ArenaRequirementsBuilder`] to compute activation and KV
//! cache requirements from model configuration data, and a standalone
//! [`estimate_kv_cache_bytes`] function for quick sizing without the
//! builder wrapper.

use crate::compute_image::residency::plan::{ActivationArenaRequirements, KvCacheRequirements};

/// Computes activation arena and KV cache requirements.
///
/// # Activation arena
///
/// Phases run sequentially (no overlap), so the total requirement is
/// the *maximum* activation size across all phases.  The region count
/// is the number of *distinct* activation sizes; phases with identical
/// requirements can reuse the same region slot.
///
/// # KV cache
///
/// Computes the total and per-layer KV cache byte counts from the
/// model architecture and configured context length.
pub struct ArenaRequirementsBuilder;

impl ArenaRequirementsBuilder {
    pub fn new() -> Self {
        Self
    }

    /// Compute activation arena requirements for a given set of phases.
    ///
    /// `phase_activation_sizes` — a list of `(phase_id, activation_bytes)`
    /// pairs for every phase in the compiled program.
    ///
    /// Returns [`ActivationArenaRequirements`] whose `total_activation_bytes`
    /// is the maximum (phases do not overlap) and `arena_region_count` is
    /// the count of distinct activation byte values.
    pub fn compute_activation_requirements(
        &self,
        phase_activation_sizes: &[(String, u64)],
    ) -> ActivationArenaRequirements {
        let mut max_bytes: u64 = 0;
        let mut distinct_sizes: Vec<u64> = Vec::new();

        for &(_, size) in phase_activation_sizes {
            if size > max_bytes {
                max_bytes = size;
            }
            if !distinct_sizes.contains(&size) {
                distinct_sizes.push(size);
            }
        }

        ActivationArenaRequirements {
            total_activation_bytes: max_bytes,
            arena_region_count: distinct_sizes.len() as u32,
        }
    }

    /// Compute KV cache requirements for a model configuration.
    ///
    /// # Parameters
    ///
    /// * `n_layers`     — number of transformer layers
    /// * `n_kv_heads`   — number of key/value heads per layer
    /// * `head_dim`     — dimension of each attention head
    /// * `max_context`  — maximum sequence length the cache must support
    /// * `kv_dtype_bytes` — bytes per KV cache element (e.g. 2 for FP16)
    pub fn compute_kv_requirements(
        &self,
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_context: u32,
        kv_dtype_bytes: u32,
    ) -> KvCacheRequirements {
        let per_layer =
            n_kv_heads as u64 * head_dim as u64 * 2 * max_context as u64 * kv_dtype_bytes as u64;

        let total = n_layers as u64 * per_layer;

        KvCacheRequirements {
            max_context_tokens: max_context,
            cache_bytes_per_token: per_layer / n_layers as u64,
            total_cache_bytes: total,
            total_kv_cache_bytes: total,
            kv_cache_per_layer_bytes: per_layer,
            n_layers,
            n_kv_heads,
            head_dim,
            max_context,
        }
    }
}

impl Default for ArenaRequirementsBuilder {
    fn default() -> Self {
        Self
    }
}

/// Estimate total KV cache bytes for a model configuration.
///
/// Formula:
/// ```text
/// total = layers × kv_heads × head_dim × 2 (K+V) × batch × context × dtype_bytes
/// ```
///
/// This is a standalone helper that includes `max_batch` for use in
/// contexts (e.g. batch serving) where the builder's single-batch
/// [`ArenaRequirementsBuilder::compute_kv_requirements`] is insufficient.
pub fn estimate_kv_cache_bytes(
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_batch: u32,
    max_context: u32,
    kv_dtype_bytes: u32,
) -> u64 {
    n_layers as u64
        * n_kv_heads as u64
        * head_dim as u64
        * 2
        * max_batch as u64
        * max_context as u64
        * kv_dtype_bytes as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Activation arena tests ──

    #[test]
    fn single_phase_activation() {
        let builder = ArenaRequirementsBuilder::new();
        let phases = vec![("prefill".to_string(), 4_194_304)]; // 4 MiB
        let req = builder.compute_activation_requirements(&phases);

        assert_eq!(req.total_activation_bytes, 4_194_304);
        assert_eq!(req.arena_region_count, 1);
    }

    #[test]
    fn multiple_phases_different_sizes() {
        let builder = ArenaRequirementsBuilder::new();
        let phases = vec![
            ("embed".to_string(), 512_000),
            ("attention".to_string(), 4_194_304),
            ("ffn".to_string(), 8_388_608),
            ("output".to_string(), 1_024_000),
        ];
        let req = builder.compute_activation_requirements(&phases);

        // Max across all phases
        assert_eq!(req.total_activation_bytes, 8_388_608);
        // All four sizes are distinct
        assert_eq!(req.arena_region_count, 4);
    }

    #[test]
    fn multiple_phases_with_shared_sizes() {
        let builder = ArenaRequirementsBuilder::new();
        let phases = vec![
            ("q_proj".to_string(), 2_097_152),
            ("k_proj".to_string(), 2_097_152),
            ("v_proj".to_string(), 2_097_152),
            ("out_proj".to_string(), 4_194_304),
            ("ffn_gate".to_string(), 4_194_304),
        ];
        let req = builder.compute_activation_requirements(&phases);

        assert_eq!(req.total_activation_bytes, 4_194_304);
        // Two distinct sizes: 2 MiB and 4 MiB
        assert_eq!(req.arena_region_count, 2);
    }

    #[test]
    fn empty_phases() {
        let builder = ArenaRequirementsBuilder::new();
        let phases: Vec<(String, u64)> = vec![];
        let req = builder.compute_activation_requirements(&phases);

        assert_eq!(req.total_activation_bytes, 0);
        assert_eq!(req.arena_region_count, 0);
    }

    // ── KV cache tests ──

    #[test]
    fn kv_cache_3b_model() {
        let builder = ArenaRequirementsBuilder::new();
        // Typical 3B parameter model: 28 layers, 4 KV heads, head_dim 128,
        // FP16 (2 bytes/element), context 8192
        let req = builder.compute_kv_requirements(28, 4, 128, 8192, 2);

        // per-layer: 4 * 128 * 2 * 8192 * 2 = 16,777,216
        let expected_per_layer: u64 = 4 * 128 * 2 * 8192 * 2;
        assert_eq!(req.kv_cache_per_layer_bytes, expected_per_layer);

        // total: 28 * expected_per_layer = 469,762,048
        assert_eq!(req.total_kv_cache_bytes, 28 * expected_per_layer);

        assert_eq!(req.n_layers, 28);
        assert_eq!(req.n_kv_heads, 4);
        assert_eq!(req.head_dim, 128);
        assert_eq!(req.max_context, 8192);
    }

    #[test]
    fn kv_cache_zero_layers() {
        let builder = ArenaRequirementsBuilder::new();
        let req = builder.compute_kv_requirements(0, 4, 128, 8192, 2);

        assert_eq!(req.kv_cache_per_layer_bytes, 4 * 128 * 2 * 8192 * 2);
        assert_eq!(req.total_kv_cache_bytes, 0);
        assert_eq!(req.n_layers, 0);
    }

    #[test]
    fn kv_cache_zero_context() {
        let builder = ArenaRequirementsBuilder::new();
        let req = builder.compute_kv_requirements(28, 4, 128, 0, 2);

        assert_eq!(req.total_kv_cache_bytes, 0);
        assert_eq!(req.kv_cache_per_layer_bytes, 0);
        assert_eq!(req.max_context, 0);
    }

    // ── Standalone estimator tests ──

    #[test]
    fn estimate_3b_kv_cache() {
        let bytes = estimate_kv_cache_bytes(28, 4, 128, 1, 8192, 2);
        // 28 * 4 * 128 * 2 * 1 * 8192 * 2
        let expected = 28 * 4 * 128 * 2 * 1 * 8192 * 2;
        assert_eq!(bytes, expected);
    }

    #[test]
    fn estimate_with_batch() {
        let bytes = estimate_kv_cache_bytes(28, 4, 128, 4, 8192, 2);
        let expected = 28 * 4 * 128 * 2 * 4 * 8192 * 2;
        assert_eq!(bytes, expected);
    }

    #[test]
    fn estimate_zero_layers() {
        let bytes = estimate_kv_cache_bytes(0, 4, 128, 1, 8192, 2);
        assert_eq!(bytes, 0);
    }

    #[test]
    fn estimate_zero_context() {
        let bytes = estimate_kv_cache_bytes(28, 4, 128, 1, 0, 2);
        assert_eq!(bytes, 0);
    }
}
