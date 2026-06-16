//! Accelerate layout definitions.
//!
//! This module defines the memory layout types supported by the Accelerate backend,
//! including row-major, column-major, contiguous, strided, and transposed views.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Memory layout types for Accelerate operations.
///
/// Layout is critical for Accelerate/BLAS operations as bugs often come from
/// leading dimension and matrix order mistakes rather than math mistakes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerateLayout {
    /// Row-major layout (C-style): consecutive elements in a row
    RowMajor,
    /// Column-major layout (Fortran-style): consecutive elements in a column
    ColumnMajor,
    /// Contiguous layout with default strides (same as RowMajor for most cases)
    Contiguous,
    /// Strided layout with explicit strides
    Strided,
    /// Transposed view: metadata-only, no data movement
    TransposedView,
    /// Materialized transpose: data has been physically transposed
    MaterializedTranspose,
}

impl fmt::Display for AccelerateLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateLayout::RowMajor => write!(f, "row_major"),
            AccelerateLayout::ColumnMajor => write!(f, "column_major"),
            AccelerateLayout::Contiguous => write!(f, "contiguous"),
            AccelerateLayout::Strided => write!(f, "strided"),
            AccelerateLayout::TransposedView => write!(f, "transposed_view"),
            AccelerateLayout::MaterializedTranspose => write!(f, "materialized_transpose"),
        }
    }
}

impl AccelerateLayout {
    /// Returns true if this layout is row-major or contiguous (which is typically row-major).
    pub fn is_row_major(&self) -> bool {
        matches!(self, AccelerateLayout::RowMajor | AccelerateLayout::Contiguous)
    }

    /// Returns true if this layout is column-major.
    pub fn is_column_major(&self) -> bool {
        matches!(self, AccelerateLayout::ColumnMajor)
    }

    /// Returns true if this layout is contiguous.
    pub fn is_contiguous(&self) -> bool {
        matches!(self, AccelerateLayout::Contiguous | AccelerateLayout::RowMajor)
    }

    /// Returns true if this layout involves transposition.
    pub fn is_transposed(&self) -> bool {
        matches!(
            self,
            AccelerateLayout::TransposedView | AccelerateLayout::MaterializedTranspose
        )
    }

    /// Returns true if this is a view-only layout (no data movement).
    pub fn is_view(&self) -> bool {
        matches!(self, AccelerateLayout::TransposedView)
    }

    /// Returns true if this layout requires materialization for Accelerate operations.
    /// Some Accelerate operations require contiguous storage.
    pub fn requires_materialization(&self) -> bool {
        matches!(
            self,
            AccelerateLayout::ColumnMajor | AccelerateLayout::Strided | AccelerateLayout::TransposedView
        )
    }

    /// Returns the layout as a string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            AccelerateLayout::RowMajor => "row_major",
            AccelerateLayout::ColumnMajor => "column_major",
            AccelerateLayout::Contiguous => "contiguous",
            AccelerateLayout::Strided => "strided",
            AccelerateLayout::TransposedView => "transposed_view",
            AccelerateLayout::MaterializedTranspose => "materialized_transpose",
        }
    }

    /// Returns all layout types.
    pub fn all() -> &'static [AccelerateLayout] {
        &[
            AccelerateLayout::RowMajor,
            AccelerateLayout::ColumnMajor,
            AccelerateLayout::Contiguous,
            AccelerateLayout::Strided,
            AccelerateLayout::TransposedView,
            AccelerateLayout::MaterializedTranspose,
        ]
    }

    /// Returns the default layout for BLAS operations.
    /// BLAS typically expects column-major for matrices.
    pub fn blas_default() -> AccelerateLayout {
        AccelerateLayout::ColumnMajor
    }

    /// Returns the default layout for vDSP operations.
    /// vDSP typically works with contiguous arrays.
    pub fn vdsp_default() -> AccelerateLayout {
        AccelerateLayout::Contiguous
    }

    /// Returns the default layout for BNNS operations.
    /// BNNS typically expects row-major/NHWC-style layouts.
    pub fn bnns_default() -> AccelerateLayout {
        AccelerateLayout::RowMajor
    }
}

impl Default for AccelerateLayout {
    fn default() -> Self {
        AccelerateLayout::RowMajor
    }
}

/// Layout policy for operation support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutPolicy {
    /// Operation supports this layout natively.
    Native,
    /// Operation supports this layout via conversion/materialization.
    Convert,
    /// Operation does not support this layout.
    Unsupported,
}

impl fmt::Display for LayoutPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutPolicy::Native => write!(f, "native"),
            LayoutPolicy::Convert => write!(f, "convert"),
            LayoutPolicy::Unsupported => write!(f, "unsupported"),
        }
    }
}

impl Default for LayoutPolicy {
    fn default() -> Self {
        LayoutPolicy::Native
    }
}

/// Shape constraints for operation support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeConstraints {
    /// Minimum number of dimensions.
    pub min_dims: Option<usize>,
    /// Maximum number of dimensions.
    pub max_dims: Option<usize>,
    /// Required dimension sizes (if any).
    pub required_dims: Vec<Option<usize>>,
    /// Whether all dimensions must be equal.
    pub must_be_square: bool,
    /// Whether the last two dimensions must be equal (for matrix operations).
    pub must_be_matrix: bool,
}

impl ShapeConstraints {
    pub fn new() -> Self {
        Self {
            min_dims: None,
            max_dims: None,
            required_dims: Vec::new(),
            must_be_square: false,
            must_be_matrix: false,
        }
    }

    /// Returns true if this constraint is satisfied by the given shape.
    pub fn is_satisfied_by(&self, shape: &[usize]) -> bool {
        // Check min dimensions
        if let Some(min) = self.min_dims {
            if shape.len() < min {
                return false;
            }
        }

        // Check max dimensions
        if let Some(max) = self.max_dims {
            if shape.len() > max {
                return false;
            }
        }

        // Check required dimensions
        if !self.required_dims.is_empty() {
            if shape.len() != self.required_dims.len() {
                return false;
            }
            for (i, &req) in self.required_dims.iter().enumerate() {
                if let Some(req_size) = req {
                    if shape[i] != req_size {
                        return false;
                    }
                }
            }
        }

        // Check square constraint
        if self.must_be_square {
            if shape.len() < 2 {
                return false;
            }
            for i in 1..shape.len() {
                if shape[i] != shape[0] {
                    return false;
                }
            }
        }

        // Check matrix constraint (at least 2 dimensions)
        if self.must_be_matrix {
            if shape.len() < 2 {
                return false;
            }
        }

        true
    }
}

impl Default for ShapeConstraints {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocation policy for operation support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocationPolicy {
    /// No additional allocation required.
    NoAllocation,
    /// Temporary allocation required for intermediate results.
    Temporary,
    /// Output allocation required.
    Output,
    /// Both temporary and output allocation required.
    TemporaryAndOutput,
}

impl fmt::Display for AllocationPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocationPolicy::NoAllocation => write!(f, "no_allocation"),
            AllocationPolicy::Temporary => write!(f, "temporary"),
            AllocationPolicy::Output => write!(f, "output"),
            AllocationPolicy::TemporaryAndOutput => write!(f, "temporary_and_output"),
        }
    }
}

impl Default for AllocationPolicy {
    fn default() -> Self {
        AllocationPolicy::NoAllocation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_display() {
        assert_eq!(AccelerateLayout::RowMajor.to_string(), "row_major");
        assert_eq!(AccelerateLayout::ColumnMajor.to_string(), "column_major");
        assert_eq!(AccelerateLayout::Contiguous.to_string(), "contiguous");
        assert_eq!(AccelerateLayout::Strided.to_string(), "strided");
        assert_eq!(AccelerateLayout::TransposedView.to_string(), "transposed_view");
        assert_eq!(
            AccelerateLayout::MaterializedTranspose.to_string(),
            "materialized_transpose"
        );
    }

    #[test]
    fn test_layout_categories() {
        assert!(AccelerateLayout::RowMajor.is_row_major());
        assert!(AccelerateLayout::Contiguous.is_row_major());
        assert!(!AccelerateLayout::ColumnMajor.is_row_major());

        assert!(AccelerateLayout::ColumnMajor.is_column_major());
        assert!(!AccelerateLayout::RowMajor.is_column_major());

        assert!(AccelerateLayout::Contiguous.is_contiguous());
        assert!(AccelerateLayout::RowMajor.is_contiguous());
        assert!(!AccelerateLayout::ColumnMajor.is_contiguous());

        assert!(AccelerateLayout::TransposedView.is_transposed());
        assert!(AccelerateLayout::MaterializedTranspose.is_transposed());
        assert!(!AccelerateLayout::RowMajor.is_transposed());

        assert!(AccelerateLayout::TransposedView.is_view());
        assert!(!AccelerateLayout::MaterializedTranspose.is_view());
    }

    #[test]
    fn test_layout_materialization() {
        assert!(AccelerateLayout::ColumnMajor.requires_materialization());
        assert!(AccelerateLayout::Strided.requires_materialization());
        assert!(AccelerateLayout::TransposedView.requires_materialization());
        assert!(!AccelerateLayout::RowMajor.requires_materialization());
        assert!(!AccelerateLayout::Contiguous.requires_materialization());
    }

    #[test]
    fn test_layout_as_str() {
        assert_eq!(AccelerateLayout::RowMajor.as_str(), "row_major");
        assert_eq!(AccelerateLayout::MaterializedTranspose.as_str(), "materialized_transpose");
    }

    #[test]
    fn test_all_layouts() {
        let all = AccelerateLayout::all();
        assert_eq!(all.len(), 6);
        assert!(all.contains(&AccelerateLayout::RowMajor));
        assert!(all.contains(&AccelerateLayout::MaterializedTranspose));
    }

    #[test]
    fn test_layout_defaults() {
        assert_eq!(AccelerateLayout::default(), AccelerateLayout::RowMajor);
        assert_eq!(AccelerateLayout::blas_default(), AccelerateLayout::ColumnMajor);
        assert_eq!(AccelerateLayout::vdsp_default(), AccelerateLayout::Contiguous);
        assert_eq!(AccelerateLayout::bnns_default(), AccelerateLayout::RowMajor);
    }

    #[test]
    fn test_layout_policy_display() {
        assert_eq!(LayoutPolicy::Native.to_string(), "native");
        assert_eq!(LayoutPolicy::Convert.to_string(), "convert");
        assert_eq!(LayoutPolicy::Unsupported.to_string(), "unsupported");
    }

    #[test]
    fn test_layout_policy_default() {
        assert_eq!(LayoutPolicy::default(), LayoutPolicy::Native);
    }

    #[test]
    fn test_allocation_policy_display() {
        assert_eq!(AllocationPolicy::NoAllocation.to_string(), "no_allocation");
        assert_eq!(AllocationPolicy::Temporary.to_string(), "temporary");
        assert_eq!(AllocationPolicy::Output.to_string(), "output");
        assert_eq!(
            AllocationPolicy::TemporaryAndOutput.to_string(),
            "temporary_and_output"
        );
    }

    #[test]
    fn test_allocation_policy_default() {
        assert_eq!(AllocationPolicy::default(), AllocationPolicy::NoAllocation);
    }

    #[test]
    fn test_shape_constraints_basic() {
        let constraints = ShapeConstraints::new();
        assert!(constraints.is_satisfied_by(&[1, 2, 3]));
        assert!(constraints.is_satisfied_by(&[]));
    }

    #[test]
    fn test_shape_constraints_min_dims() {
        let mut constraints = ShapeConstraints::new();
        constraints.min_dims = Some(2);
        assert!(constraints.is_satisfied_by(&[1, 2]));
        assert!(!constraints.is_satisfied_by(&[1]));
    }

    #[test]
    fn test_shape_constraints_max_dims() {
        let mut constraints = ShapeConstraints::new();
        constraints.max_dims = Some(2);
        assert!(constraints.is_satisfied_by(&[1, 2]));
        assert!(!constraints.is_satisfied_by(&[1, 2, 3]));
    }

    #[test]
    fn test_shape_constraints_square() {
        let mut constraints = ShapeConstraints::new();
        constraints.must_be_square = true;
        assert!(constraints.is_satisfied_by(&[3, 3]));
        assert!(!constraints.is_satisfied_by(&[2, 3]));
    }

    #[test]
    fn test_shape_constraints_matrix() {
        let mut constraints = ShapeConstraints::new();
        constraints.must_be_matrix = true;
        assert!(constraints.is_satisfied_by(&[2, 3]));
        assert!(!constraints.is_satisfied_by(&[3]));
    }
}
