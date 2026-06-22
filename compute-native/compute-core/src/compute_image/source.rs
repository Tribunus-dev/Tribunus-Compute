//! Source loading — tensor loading, hashing, identity building.
//!
//! Extracted from compile.rs in a phased refactoring. Currently re-exports from
//! compile.rs; actual definitions will migrate here in a later phase.

/// Re-export compile.rs items through this module.
/// When refactoring progresses, actual definitions will move here and the
/// following types/functions will be imported directly:
///   use crate::compute_image::manifest::ShardHash;
///   use crate::compute_image::compile::LoadedSource;
pub(crate) use super::compile::{
    build_source_identity, diff_tensors, load_source, load_source_tensor_table, LoadedSource,
    SourceTensor, SourceTensorInfo,
};
