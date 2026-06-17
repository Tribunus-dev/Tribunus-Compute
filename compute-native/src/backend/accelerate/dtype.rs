//! Accelerate data type definitions.
//!
//! This module defines the data types supported by the Accelerate backend,
//! including their properties and conversion utilities.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Data types supported by the Accelerate backend.
///
/// The v0 implementation focuses on f32 as the primary dtype, with future
/// support for f16, bf16, and other types as the Accelerate APIs stabilize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerateDType {
    /// 32-bit floating point (primary dtype for v0)
    F32,
    /// 64-bit floating point
    F64,
    /// 16-bit floating point (IEEE half-precision)
    F16,
    /// Brain floating point (16-bit)
    Bf16,
    /// 32-bit signed integer
    I32,
    /// 32-bit unsigned integer
    U32,
    /// 16-bit signed integer
    I16,
    /// 16-bit unsigned integer
    U16,
    /// 8-bit signed integer
    I8,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
}

impl fmt::Display for AccelerateDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateDType::F32 => write!(f, "f32"),
            AccelerateDType::F64 => write!(f, "f64"),
            AccelerateDType::F16 => write!(f, "f16"),
            AccelerateDType::Bf16 => write!(f, "bf16"),
            AccelerateDType::I32 => write!(f, "i32"),
            AccelerateDType::U32 => write!(f, "u32"),
            AccelerateDType::I16 => write!(f, "i16"),
            AccelerateDType::U16 => write!(f, "u16"),
            AccelerateDType::I8 => write!(f, "i8"),
            AccelerateDType::U8 => write!(f, "u8"),
            AccelerateDType::Bool => write!(f, "bool"),
        }
    }
}

impl AccelerateDType {
    /// Returns the size in bytes of this dtype.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            AccelerateDType::F32 | AccelerateDType::I32 | AccelerateDType::U32 => 4,
            AccelerateDType::F64 => 8,
            AccelerateDType::F16 | AccelerateDType::Bf16 | AccelerateDType::I16 | AccelerateDType::U16 => 2,
            AccelerateDType::I8 | AccelerateDType::U8 | AccelerateDType::Bool => 1,
        }
    }

    /// Returns true if this is a floating-point type.
    pub fn is_floating_point(&self) -> bool {
        matches!(
            self,
            AccelerateDType::F32 | AccelerateDType::F64 | AccelerateDType::F16 | AccelerateDType::Bf16
        )
    }

    /// Returns true if this is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            AccelerateDType::I32 | AccelerateDType::U32 | AccelerateDType::I16 | AccelerateDType::U16
                | AccelerateDType::I8 | AccelerateDType::U8
        )
    }

    /// Returns true if this is a boolean type.
    pub fn is_boolean(&self) -> bool {
        matches!(self, AccelerateDType::Bool)
    }

    /// Returns true if this dtype is supported by BLAS operations.
    /// BLAS typically supports f32, f64, and sometimes f16/bf16 on newer systems.
    pub fn is_blas_supported(&self) -> bool {
        matches!(self, AccelerateDType::F32 | AccelerateDType::F64)
    }

    /// Returns true if this dtype is supported by vDSP operations.
    /// vDSP typically supports f32 and f64 for most operations.
    pub fn is_vdsp_supported(&self) -> bool {
        matches!(self, AccelerateDType::F32 | AccelerateDType::F64)
    }

    /// Returns true if this dtype is supported by BNNS operations.
    /// BNNS typically supports f32 and f16/bf16 on Apple Silicon.
    pub fn is_bnns_supported(&self) -> bool {
        matches!(
            self,
            AccelerateDType::F32 | AccelerateDType::F16 | AccelerateDType::Bf16
        )
    }

    /// Returns the dtype as a string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            AccelerateDType::F32 => "f32",
            AccelerateDType::F64 => "f64",
            AccelerateDType::F16 => "f16",
            AccelerateDType::Bf16 => "bf16",
            AccelerateDType::I32 => "i32",
            AccelerateDType::U32 => "u32",
            AccelerateDType::I16 => "i16",
            AccelerateDType::U16 => "u16",
            AccelerateDType::I8 => "i8",
            AccelerateDType::U8 => "u8",
            AccelerateDType::Bool => "bool",
        }
    }

    /// Attempts to parse a dtype from a string.
    pub fn from_str(s: &str) -> Option<AccelerateDType> {
        match s.to_lowercase().as_str() {
            "f32" | "float32" | "float" => Some(AccelerateDType::F32),
            "f64" | "float64" | "double" => Some(AccelerateDType::F64),
            "f16" | "float16" | "half" => Some(AccelerateDType::F16),
            "bf16" | "bfloat16" | "bfloat" => Some(AccelerateDType::Bf16),
            "i32" | "int32" | "int" => Some(AccelerateDType::I32),
            "u32" | "uint32" | "uint" => Some(AccelerateDType::U32),
            "i16" | "int16" | "short" => Some(AccelerateDType::I16),
            "u16" | "uint16" | "ushort" => Some(AccelerateDType::U16),
            "i8" | "int8" | "char" => Some(AccelerateDType::I8),
            "u8" | "uint8" | "uchar" | "byte" => Some(AccelerateDType::U8),
            "bool" | "boolean" => Some(AccelerateDType::Bool),
            _ => None,
        }
    }

    /// Returns all supported dtypes.
    pub fn all() -> &'static [AccelerateDType] {
        &[
            AccelerateDType::F32,
            AccelerateDType::F64,
            AccelerateDType::F16,
            AccelerateDType::Bf16,
            AccelerateDType::I32,
            AccelerateDType::U32,
            AccelerateDType::I16,
            AccelerateDType::U16,
            AccelerateDType::I8,
            AccelerateDType::U8,
            AccelerateDType::Bool,
        ]
    }

    /// Returns the v0 supported dtypes (primarily f32).
    pub fn v0_supported() -> &'static [AccelerateDType] {
        &[AccelerateDType::F32]
    }
}

impl Default for AccelerateDType {
    fn default() -> Self {
        AccelerateDType::F32
    }
}

/// DType policy for operation support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DTypePolicy {
    /// Operation supports this dtype natively.
    Native,
    /// Operation supports this dtype via conversion.
    Convert,
    /// Operation does not support this dtype.
    Unsupported,
}

impl fmt::Display for DTypePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DTypePolicy::Native => write!(f, "native"),
            DTypePolicy::Convert => write!(f, "convert"),
            DTypePolicy::Unsupported => write!(f, "unsupported"),
        }
    }
}

impl Default for DTypePolicy {
    fn default() -> Self {
        DTypePolicy::Native
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_display() {
        assert_eq!(AccelerateDType::F32.to_string(), "f32");
        assert_eq!(AccelerateDType::F64.to_string(), "f64");
        assert_eq!(AccelerateDType::F16.to_string(), "f16");
        assert_eq!(AccelerateDType::Bf16.to_string(), "bf16");
        assert_eq!(AccelerateDType::I32.to_string(), "i32");
        assert_eq!(AccelerateDType::Bool.to_string(), "bool");
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(AccelerateDType::F32.size_in_bytes(), 4);
        assert_eq!(AccelerateDType::F64.size_in_bytes(), 8);
        assert_eq!(AccelerateDType::F16.size_in_bytes(), 2);
        assert_eq!(AccelerateDType::Bf16.size_in_bytes(), 2);
        assert_eq!(AccelerateDType::I32.size_in_bytes(), 4);
        assert_eq!(AccelerateDType::Bool.size_in_bytes(), 1);
    }

    #[test]
    fn test_dtype_categories() {
        assert!(AccelerateDType::F32.is_floating_point());
        assert!(AccelerateDType::F64.is_floating_point());
        assert!(AccelerateDType::F16.is_floating_point());
        assert!(AccelerateDType::Bf16.is_floating_point());
        assert!(!AccelerateDType::I32.is_floating_point());

        assert!(AccelerateDType::I32.is_integer());
        assert!(AccelerateDType::U32.is_integer());
        assert!(!AccelerateDType::F32.is_integer());

        assert!(AccelerateDType::Bool.is_boolean());
        assert!(!AccelerateDType::I32.is_boolean());
    }

    #[test]
    fn test_dtype_subsystem_support() {
        assert!(AccelerateDType::F32.is_blas_supported());
        assert!(AccelerateDType::F64.is_blas_supported());
        assert!(!AccelerateDType::F16.is_blas_supported());

        assert!(AccelerateDType::F32.is_vdsp_supported());
        assert!(AccelerateDType::F64.is_vdsp_supported());
        assert!(!AccelerateDType::I32.is_vdsp_supported());

        assert!(AccelerateDType::F32.is_bnns_supported());
        assert!(AccelerateDType::F16.is_bnns_supported());
        assert!(AccelerateDType::Bf16.is_bnns_supported());
        assert!(!AccelerateDType::I32.is_bnns_supported());
    }

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(AccelerateDType::from_str("f32"), Some(AccelerateDType::F32));
        assert_eq!(AccelerateDType::from_str("float32"), Some(AccelerateDType::F32));
        assert_eq!(AccelerateDType::from_str("float"), Some(AccelerateDType::F32));
        assert_eq!(AccelerateDType::from_str("f64"), Some(AccelerateDType::F64));
        assert_eq!(AccelerateDType::from_str("bool"), Some(AccelerateDType::Bool));
        assert_eq!(AccelerateDType::from_str("unknown"), None);
    }

    #[test]
    fn test_dtype_as_str() {
        assert_eq!(AccelerateDType::F32.as_str(), "f32");
        assert_eq!(AccelerateDType::Bool.as_str(), "bool");
    }

    #[test]
    fn test_all_dtypes() {
        let all = AccelerateDType::all();
        assert_eq!(all.len(), 11);
        assert!(all.contains(&AccelerateDType::F32));
        assert!(all.contains(&AccelerateDType::Bool));
    }

    #[test]
    fn test_v0_supported_dtypes() {
        let v0 = AccelerateDType::v0_supported();
        assert_eq!(v0.len(), 1);
        assert_eq!(v0[0], AccelerateDType::F32);
    }

    #[test]
    fn test_dtype_default() {
        assert_eq!(AccelerateDType::default(), AccelerateDType::F32);
    }

    #[test]
    fn test_dtype_policy_display() {
        assert_eq!(DTypePolicy::Native.to_string(), "native");
        assert_eq!(DTypePolicy::Convert.to_string(), "convert");
        assert_eq!(DTypePolicy::Unsupported.to_string(), "unsupported");
    }

    #[test]
    fn test_dtype_policy_default() {
        assert_eq!(DTypePolicy::default(), DTypePolicy::Native);
    }
}
