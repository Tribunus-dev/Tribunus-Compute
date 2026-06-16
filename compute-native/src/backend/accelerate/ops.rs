//! Canonical operation definitions for Accelerate backend.
//!
//! This module defines the canonical operations that the Accelerate backend
//! needs to support for v0, including their categories and properties.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Canonical operation kinds for Accelerate backend.
///
/// These are the operations that matter to the existing lattice/conformance gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalOp {
    // Memory/identity operations
    /// Constant roundtrip (serialize/deserialize)
    ConstantRoundtrip,
    /// Identity operation (no-op)
    Identity,

    // Elementwise arithmetic operations
    /// Elementwise addition
    Add,
    /// Elementwise multiplication
    Multiply,

    // Activation functions
    /// Sigmoid activation: 1 / (1 + exp(-x))
    Sigmoid,
    /// SiLU activation: sigmoid(x) * x (composite operation)
    Silu,

    // Linear algebra operations
    /// Matrix multiplication
    Matmul,

    // Layout operations
    /// Reshape operation
    Reshape,
    /// Transpose operation
    Transpose,

    // Softmax operations
    /// Softmax operation
    Softmax,

    // KV-cache operations (for future expansion)
    /// KV-cache view operation
    KvCacheView,
    /// KV-cache write operation
    KvCacheWrite,
    /// KV-cache append operation
    KvCacheAppend,
}

impl fmt::Display for CanonicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CanonicalOp::ConstantRoundtrip => write!(f, "constant_roundtrip"),
            CanonicalOp::Identity => write!(f, "identity"),
            CanonicalOp::Add => write!(f, "add"),
            CanonicalOp::Multiply => write!(f, "multiply"),
            CanonicalOp::Sigmoid => write!(f, "sigmoid"),
            CanonicalOp::Silu => write!(f, "silu"),
            CanonicalOp::Matmul => write!(f, "matmul"),
            CanonicalOp::Reshape => write!(f, "reshape"),
            CanonicalOp::Transpose => write!(f, "transpose"),
            CanonicalOp::Softmax => write!(f, "softmax"),
            CanonicalOp::KvCacheView => write!(f, "kv_cache_view"),
            CanonicalOp::KvCacheWrite => write!(f, "kv_cache_write"),
            CanonicalOp::KvCacheAppend => write!(f, "kv_cache_append"),
        }
    }
}

impl CanonicalOp {
    /// Returns the operation category.
    pub fn category(&self) -> &'static str {
        match self {
            CanonicalOp::ConstantRoundtrip | CanonicalOp::Identity => "memory",
            CanonicalOp::Add | CanonicalOp::Multiply => "elementwise",
            CanonicalOp::Sigmoid | CanonicalOp::Silu => "activation",
            CanonicalOp::Matmul => "matmul",
            CanonicalOp::Reshape | CanonicalOp::Transpose => "layout",
            CanonicalOp::Softmax => "reduction",
            CanonicalOp::KvCacheView | CanonicalOp::KvCacheWrite | CanonicalOp::KvCacheAppend => {
                "kv_cache"
            }
        }
    }

    /// Returns true if this is a memory/identity operation.
    pub fn is_memory(&self) -> bool {
        matches!(self, CanonicalOp::ConstantRoundtrip | CanonicalOp::Identity)
    }

    /// Returns true if this is an elementwise operation.
    pub fn is_elementwise(&self) -> bool {
        matches!(self, CanonicalOp::Add | CanonicalOp::Multiply)
    }

    /// Returns true if this is an activation function.
    pub fn is_activation(&self) -> bool {
        matches!(self, CanonicalOp::Sigmoid | CanonicalOp::Silu)
    }

    /// Returns true if this is a matrix multiplication operation.
    pub fn is_matmul(&self) -> bool {
        matches!(self, CanonicalOp::Matmul)
    }

    /// Returns true if this is a layout operation.
    pub fn is_layout(&self) -> bool {
        matches!(self, CanonicalOp::Reshape | CanonicalOp::Transpose)
    }

    /// Returns true if this is a reduction operation.
    pub fn is_reduction(&self) -> bool {
        matches!(self, CanonicalOp::Softmax)
    }

    /// Returns true if this is a KV-cache operation.
    pub fn is_kv_cache(&self) -> bool {
        matches!(
            self,
            CanonicalOp::KvCacheView | CanonicalOp::KvCacheWrite | CanonicalOp::KvCacheAppend
        )
    }

    /// Returns the operation as a string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            CanonicalOp::ConstantRoundtrip => "constant_roundtrip",
            CanonicalOp::Identity => "identity",
            CanonicalOp::Add => "add",
            CanonicalOp::Multiply => "multiply",
            CanonicalOp::Sigmoid => "sigmoid",
            CanonicalOp::Silu => "silu",
            CanonicalOp::Matmul => "matmul",
            CanonicalOp::Reshape => "reshape",
            CanonicalOp::Transpose => "transpose",
            CanonicalOp::Softmax => "softmax",
            CanonicalOp::KvCacheView => "kv_cache_view",
            CanonicalOp::KvCacheWrite => "kv_cache_write",
            CanonicalOp::KvCacheAppend => "kv_cache_append",
        }
    }

    /// Returns all canonical operations.
    pub fn all() -> &'static [CanonicalOp] {
        &[
            CanonicalOp::ConstantRoundtrip,
            CanonicalOp::Identity,
            CanonicalOp::Add,
            CanonicalOp::Multiply,
            CanonicalOp::Sigmoid,
            CanonicalOp::Silu,
            CanonicalOp::Matmul,
            CanonicalOp::Reshape,
            CanonicalOp::Transpose,
            CanonicalOp::Softmax,
            CanonicalOp::KvCacheView,
            CanonicalOp::KvCacheWrite,
            CanonicalOp::KvCacheAppend,
        ]
    }

    /// Returns the v0 canonical operations (the core set for initial implementation).
    pub fn v0_core() -> &'static [CanonicalOp] {
        &[
            CanonicalOp::ConstantRoundtrip,
            CanonicalOp::Identity,
            CanonicalOp::Add,
            CanonicalOp::Multiply,
            CanonicalOp::Matmul,
        ]
    }

    /// Returns the v0.5 operations (adding activations).
    pub fn v0_activations() -> &'static [CanonicalOp] {
        &[CanonicalOp::Sigmoid, CanonicalOp::Silu]
    }

    /// Returns the v0.75 operations (adding layout).
    pub fn v0_layout() -> &'static [CanonicalOp] {
        &[CanonicalOp::Reshape, CanonicalOp::Transpose]
    }

    /// Returns the v1.0 operations (adding softmax).
    pub fn v1_softmax() -> &'static [CanonicalOp] {
        &[CanonicalOp::Softmax]
    }

    /// Returns the v1.5 operations (adding KV-cache).
    pub fn v1_kv_cache() -> &'static [CanonicalOp] {
        &[
            CanonicalOp::KvCacheView,
            CanonicalOp::KvCacheWrite,
            CanonicalOp::KvCacheAppend,
        ]
    }
}

impl Default for CanonicalOp {
    fn default() -> Self {
        CanonicalOp::Identity
    }
}

/// Operation category for grouping operations by their computational nature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpCategory {
    Memory,
    Elementwise,
    Activation,
    Matmul,
    Layout,
    Reduction,
    KvCache,
}

impl fmt::Display for OpCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpCategory::Memory => write!(f, "memory"),
            OpCategory::Elementwise => write!(f, "elementwise"),
            OpCategory::Activation => write!(f, "activation"),
            OpCategory::Matmul => write!(f, "matmul"),
            OpCategory::Layout => write!(f, "layout"),
            OpCategory::Reduction => write!(f, "reduction"),
            OpCategory::KvCache => write!(f, "kv_cache"),
        }
    }
}

impl OpCategory {
    pub fn all() -> &'static [OpCategory] {
        &[
            OpCategory::Memory,
            OpCategory::Elementwise,
            OpCategory::Activation,
            OpCategory::Matmul,
            OpCategory::Layout,
            OpCategory::Reduction,
            OpCategory::KvCache,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_op_display() {
        assert_eq!(CanonicalOp::Identity.to_string(), "identity");
        assert_eq!(CanonicalOp::Add.to_string(), "add");
        assert_eq!(CanonicalOp::Multiply.to_string(), "multiply");
        assert_eq!(CanonicalOp::Sigmoid.to_string(), "sigmoid");
        assert_eq!(CanonicalOp::Silu.to_string(), "silu");
        assert_eq!(CanonicalOp::Matmul.to_string(), "matmul");
        assert_eq!(CanonicalOp::Reshape.to_string(), "reshape");
        assert_eq!(CanonicalOp::Transpose.to_string(), "transpose");
        assert_eq!(CanonicalOp::Softmax.to_string(), "softmax");
    }

    #[test]
    fn test_canonical_op_categories() {
        assert_eq!(CanonicalOp::Identity.category(), "memory");
        assert_eq!(CanonicalOp::Add.category(), "elementwise");
        assert_eq!(CanonicalOp::Sigmoid.category(), "activation");
        assert_eq!(CanonicalOp::Matmul.category(), "matmul");
        assert_eq!(CanonicalOp::Reshape.category(), "layout");
        assert_eq!(CanonicalOp::Softmax.category(), "reduction");
        assert_eq!(CanonicalOp::KvCacheView.category(), "kv_cache");
    }

    #[test]
    fn test_canonical_op_predicates() {
        assert!(CanonicalOp::Identity.is_memory());
        assert!(CanonicalOp::Add.is_elementwise());
        assert!(CanonicalOp::Sigmoid.is_activation());
        assert!(CanonicalOp::Matmul.is_matmul());
        assert!(CanonicalOp::Reshape.is_layout());
        assert!(CanonicalOp::Softmax.is_reduction());
        assert!(CanonicalOp::KvCacheView.is_kv_cache());

        assert!(!CanonicalOp::Add.is_memory());
        assert!(!CanonicalOp::Identity.is_elementwise());
    }

    #[test]
    fn test_canonical_op_as_str() {
        assert_eq!(CanonicalOp::Identity.as_str(), "identity");
        assert_eq!(CanonicalOp::Matmul.as_str(), "matmul");
    }

    #[test]
    fn test_all_canonical_ops() {
        let all = CanonicalOp::all();
        assert_eq!(all.len(), 13);
        assert!(all.contains(&CanonicalOp::Identity));
        assert!(all.contains(&CanonicalOp::KvCacheAppend));
    }

    #[test]
    fn test_v0_core_ops() {
        let core = CanonicalOp::v0_core();
        assert_eq!(core.len(), 5);
        assert!(core.contains(&CanonicalOp::Identity));
        assert!(core.contains(&CanonicalOp::Add));
        assert!(core.contains(&CanonicalOp::Multiply));
        assert!(core.contains(&CanonicalOp::Matmul));
    }

    #[test]
    fn test_v0_activations() {
        let activations = CanonicalOp::v0_activations();
        assert_eq!(activations.len(), 2);
        assert!(activations.contains(&CanonicalOp::Sigmoid));
        assert!(activations.contains(&CanonicalOp::Silu));
    }

    #[test]
    fn test_v0_layout() {
        let layout = CanonicalOp::v0_layout();
        assert_eq!(layout.len(), 2);
        assert!(layout.contains(&CanonicalOp::Reshape));
        assert!(layout.contains(&CanonicalOp::Transpose));
    }

    #[test]
    fn test_canonical_op_default() {
        assert_eq!(CanonicalOp::default(), CanonicalOp::Identity);
    }

    #[test]
    fn test_op_category_display() {
        assert_eq!(OpCategory::Memory.to_string(), "memory");
        assert_eq!(OpCategory::Elementwise.to_string(), "elementwise");
        assert_eq!(OpCategory::Activation.to_string(), "activation");
        assert_eq!(OpCategory::Matmul.to_string(), "matmul");
        assert_eq!(OpCategory::Layout.to_string(), "layout");
        assert_eq!(OpCategory::Reduction.to_string(), "reduction");
        assert_eq!(OpCategory::KvCache.to_string(), "kv_cache");
    }

    #[test]
    fn test_all_op_categories() {
        let all = OpCategory::all();
        assert_eq!(all.len(), 7);
    }
}
