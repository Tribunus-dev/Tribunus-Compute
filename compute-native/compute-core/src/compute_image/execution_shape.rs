use serde::{Deserialize, Serialize};

/// Shape-specialized execution variants.
///
/// Each variant identifies a distinct execution shape class — the runtime
/// selects a compiled phase program whose `ExecutionShapeClass` best matches
/// the incoming request shape.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum ExecutionShapeClass {
    /// Single-token decode (autoregressive generation, one step).
    #[default]
    Decode1,
    /// Batched decode with up to `max_batch` concurrent sequences.
    DecodeBatch { max_batch: u32 },
    /// Prefix prefill with up to `tokens` KV entries.
    PrefillBucket { tokens: u32 },
    /// Chunked prefill — processes `chunk_tokens` per micro-batch.
    ChunkedPrefill { chunk_tokens: u32 },
    /// Mixed batch — interleaved decode/prefill within the same invocation.
    MixedBatch,
    /// Diffusion forward — processes image/video canvas tokens.
    DiffusionForward { max_canvas_tokens: u32 },
}

impl ExecutionShapeClass {
    /// Return a human-readable label for this shape class.
    pub fn variant_name(&self) -> &'static str {
        match self {
            ExecutionShapeClass::Decode1 => "Decode1",
            ExecutionShapeClass::DecodeBatch { .. } => "DecodeBatch",
            ExecutionShapeClass::PrefillBucket { .. } => "PrefillBucket",
            ExecutionShapeClass::ChunkedPrefill { .. } => "ChunkedPrefill",
            ExecutionShapeClass::MixedBatch => "MixedBatch",
            ExecutionShapeClass::DiffusionForward { .. } => "DiffusionForward",
        }
    }
}


