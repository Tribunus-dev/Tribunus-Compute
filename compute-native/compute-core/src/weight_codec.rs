//! ADR 0035 Pillar 1 — weight encoding and page residency.
//!
//! Defines the codec taxonomy for quantised weight storage and the page-level
//! metadata that ties a codec variant to a backend-compatibility signature and
//! a residency tier.  The layer above (the weight allocator / memory planner)
//! uses [`WeightPage`] to track which pages are resident in which memory tier
//! and with which codec they are encoded.

use serde::{Deserialize, Serialize};

/// Weight encoding scheme.
///
/// Each variant describes a different quantisation approach with a
/// different accuracy / throughput / memory trade-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightCodec {
    /// No compression — weights stored as fp16 or bf16.
    Identity,

    /// Group-wise uniform quantisation with optional AWQ scale reordering.
    GroupQuantized {
        /// Number of elements per quantisation group (typically 32, 64, 128).
        group_size: u32,
        /// Number of bits per element (4 for NF4/int4 or 2 for FP2).
        bits: u32,
        /// Whether AWQ-permuted scale/zero-point reordering is applied.
        awq_scaled: bool,
    },

    /// Rotation-based quantisation (QuaRot / SpinQuant) — future.
    RotationQuantized,

    /// Codebook-based quantisation (AQLM) — future.
    CodebookQuantized,
}

/// Memory residency tier for a weight page.
///
/// Tiers form a hierarchy from always-present (Mandatory) to evictable
/// (Cold).  The runtime's memory planner decides which tier each page
/// occupies based on access frequency and reuse distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidencyTier {
    /// Always resident; never evicted (e.g. embedding tables, lm_head).
    Mandatory,
    /// Likely to be reused soon; kept in fast device memory.
    Hot,
    /// May be reused; allowed in a cheaper tier (e.g. unified memory or
    /// compressed swap).
    Warm,
    /// Rarely accessed; may be offloaded to host memory or storage.
    Cold,
}

/// Metadata for a single weight page.
///
/// A page is the unit of weight encoding and memory management: all weights
/// within one page share the same codec, backend-compatibility marker, and
/// residency tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightPage {
    /// Unique page identifier.
    pub page_id: u32,
    /// Codec used to encode the weights on this page.
    pub codec: WeightCodec,
    /// Opaque backend-compatibility bitfield.
    ///
    /// Bit 0: MLX
    /// Bit 1: Accelerate
    /// Bit 2: ANE
    /// Bit 3: CoreML
    /// Bits 4-7: reserved for future backends
    pub backend_compat: u8,
    /// Residency tier assigned by the memory planner.
    pub residency_tier: ResidencyTier,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_codec_identity() {
        let codec = WeightCodec::Identity;
        let json = serde_json::to_string(&codec).unwrap();
        let back: WeightCodec = serde_json::from_str(&json).unwrap();
        assert_eq!(codec, back);
    }

    #[test]
    fn test_weight_codec_group_quantized() {
        let codec = WeightCodec::GroupQuantized {
            group_size: 64,
            bits: 4,
            awq_scaled: true,
        };
        let json = serde_json::to_string(&codec).unwrap();
        let back: WeightCodec = serde_json::from_str(&json).unwrap();
        assert_eq!(codec, back);
    }

    #[test]
    fn test_residency_tier_order() {
        // Verify serialisation round-trip for all tiers.
        let tiers = [
            ResidencyTier::Mandatory,
            ResidencyTier::Hot,
            ResidencyTier::Warm,
            ResidencyTier::Cold,
        ];
        for tier in &tiers {
            let json = serde_json::to_string(tier).unwrap();
            let back: ResidencyTier = serde_json::from_str(&json).unwrap();
            assert_eq!(*tier, back);
        }
    }

    #[test]
    fn test_weight_page_roundtrip() {
        let page = WeightPage {
            page_id: 42,
            codec: WeightCodec::Identity,
            backend_compat: 0b0001,
            residency_tier: ResidencyTier::Hot,
        };
        let json = serde_json::to_string(&page).unwrap();
        let back: WeightPage = serde_json::from_str(&json).unwrap();
        assert_eq!(page.page_id, back.page_id);
        assert_eq!(page.codec, back.codec);
        assert_eq!(page.backend_compat, back.backend_compat);
        assert_eq!(page.residency_tier, back.residency_tier);
    }
}
