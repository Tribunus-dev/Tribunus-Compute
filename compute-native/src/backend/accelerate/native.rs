//! Native Accelerate FFI bindings and safe wrappers.
//!
//! This module provides the actual native Accelerate.framework bindings for f32 operations.
//! It is cfg-gated to macOS only and provides safe Rust wrappers around unsafe FFI calls.
//!
//! # Implementation Status
//!
//! This module implements NATIVE-ACCELERATE-F32-KERNELS-0001:
//! - vDSP_vadd for elementwise f32 addition
//! - vDSP_vmul for elementwise f32 multiplication
//! - cblas_sgemm for f32 matrix multiplication (row-major, non-transposed)
//!
//! # Design Principles
//!
//! 1. **CFG-Gated**: All native code is guarded by `#[cfg(target_os = "macos")]`
//! 2. **Safe Wrappers**: All unsafe FFI calls wrapped in safe Rust functions
//! 3. **Validation**: Pre-validate all inputs before entering unsafe blocks
//! 4. **No Panics**: Runtime errors return Result, not panic
//! 5. **Portable**: Non-macOS platforms provide stubs that return Unavailable

use std::fmt;

/// Error type for native Accelerate operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccelerateNativeError {
    /// Accelerate framework is not available on this platform.
    BackendUnavailable,
    /// Operation is unsupported (e.g., wrong dtype, layout, or shape).
    Unsupported(String),
    /// Length mismatch between input vectors.
    LengthMismatch { expected: usize, actual: usize },
    /// Dimension overflow or invalid dimensions.
    DimensionError(String),
    /// Empty input (N=0) which may not be supported by all native paths.
    EmptyInput,
    /// Output allocation failed.
    AllocationError,
    /// Native call failed (should not happen with validated inputs).
    NativeCallFailed(String),
}

impl fmt::Display for AccelerateNativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelerateNativeError::BackendUnavailable => {
                write!(f, "Accelerate framework unavailable on this platform")
            }
            AccelerateNativeError::Unsupported(reason) => {
                write!(f, "Unsupported operation: {}", reason)
            }
            AccelerateNativeError::LengthMismatch { expected, actual } => {
                write!(f, "Length mismatch: expected {}, got {}", expected, actual)
            }
            AccelerateNativeError::DimensionError(msg) => {
                write!(f, "Dimension error: {}", msg)
            }
            AccelerateNativeError::EmptyInput => {
                write!(f, "Empty input not supported by native path")
            }
            AccelerateNativeError::AllocationError => {
                write!(f, "Output allocation failed")
            }
            AccelerateNativeError::NativeCallFailed(msg) => {
                write!(f, "Native call failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for AccelerateNativeError {}

/// Result type for native Accelerate operations.
pub type AccelerateNativeResult<T> = Result<T, AccelerateNativeError>;

/// Symbol names for native calls (used in evidence).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSymbol {
    /// vDSP vector addition
    VdspVadd,
    /// vDSP vector multiplication
    VdspVmul,
    /// BLAS GEMM (cblas_sgemm)
    CblasSgemm,
}

impl fmt::Display for NativeSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NativeSymbol::VdspVadd => write!(f, "vDSP_vadd"),
            NativeSymbol::VdspVmul => write!(f, "vDSP_vmul"),
            NativeSymbol::CblasSgemm => write!(f, "cblas_sgemm"),
        }
    }
}

/// Native Accelerate operations.
///
/// This module contains the actual FFI bindings and safe wrappers for native execution.
#[cfg(target_os = "macos")]
pub mod platform {
    use super::*;
    use std::ffi::c_void;

    // vDSP types and constants
    type Float32 = f32;
    type vDSP_Length = i32;
    type vDSP_Stride = i32;

    // CBLAS types
    type CBLAS_INDEX = i32; // Classic CBLAS uses i32 for indices
    type CBLAS_LAYOUT = i32;
    type CBLAS_TRANSPOSE = i32;

    // CBLAS constants
    const CBLAS_LAYOUT_ROW_MAJOR: CBLAS_LAYOUT = 101;
    const CBLAS_LAYOUT_COL_MAJOR: CBLAS_LAYOUT = 102;
    const CBLAS_TRANSPOSE_NO: CBLAS_TRANSPOSE = 111;
    const CBLAS_TRANSPOSE_YES: CBLAS_TRANSPOSE = 112;

    // External FFI bindings to Accelerate.framework
    extern "C" {
        // vDSP: Vector addition
        // vDSP_vadd(const Float32 *__vDSP_A, vDSP_Stride __vDSP_I, 
        //           const Float32 *__vDSP_B, vDSP_Stride __vDSP_J,
        //           Float32 *__vDSP_C, vDSP_Stride __vDSP_K, 
        //           vDSP_Length __vDSP_N);
        pub fn vDSP_vadd(
            a: *const Float32,
            stride_a: vDSP_Stride,
            b: *const Float32,
            stride_b: vDSP_Stride,
            c: *mut Float32,
            stride_c: vDSP_Stride,
            n: vDSP_Length,
        );

        // vDSP: Vector multiplication
        // vDSP_vmul(const Float32 *__vDSP_A, vDSP_Stride __vDSP_I,
        //           const Float32 *__vDSP_B, vDSP_Stride __vDSP_J,
        //           Float32 *__vDSP_C, vDSP_Stride __vDSP_K,
        //           vDSP_Length __vDSP_N);
        pub fn vDSP_vmul(
            a: *const Float32,
            stride_a: vDSP_Stride,
            b: *const Float32,
            stride_b: vDSP_Stride,
            c: *mut Float32,
            stride_c: vDSP_Stride,
            n: vDSP_Length,
        );

        // BLAS: Single-precision GEMM
        // void cblas_sgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
        //                  const CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N,
        //                  const CBLAS_INDEX K, const Float32 alpha, const Float32 *A, 
        //                  const CBLAS_INDEX lda, const Float32 *B, const CBLAS_INDEX ldb,
        //                  const Float32 beta, Float32 *C, const CBLAS_INDEX ldc);
        pub fn cblas_sgemm(
            layout: CBLAS_LAYOUT,
            trans_a: CBLAS_TRANSPOSE,
            trans_b: CBLAS_TRANSPOSE,
            m: CBLAS_INDEX,
            n: CBLAS_INDEX,
            k: CBLAS_INDEX,
            alpha: Float32,
            a: *const Float32,
            lda: CBLAS_INDEX,
            b: *const Float32,
            ldb: CBLAS_INDEX,
            beta: Float32,
            c: *mut Float32,
            ldc: CBLAS_INDEX,
        );
    }

    /// Validates that a usize can be safely converted to CBLAS_INDEX (i32).
    /// 
    /// CBLAS uses i32 for dimensions in the classic ABI.
    fn validate_cblas_index(value: usize, name: &str) -> AccelerateNativeResult<CBLAS_INDEX> {
        if value > i32::MAX as usize {
            Err(AccelerateNativeError::DimensionError(format!(
                "{} = {} exceeds CBLAS_INDEX max ({})",
                name,
                value,
                i32::MAX
            )))
        } else {
            Ok(value as CBLAS_INDEX)
        }
    }

    /// Validates that a usize can be safely converted to vDSP_Length (i32).
    fn validate_vdsp_length(value: usize, name: &str) -> AccelerateNativeResult<vDSP_Length> {
        if value > i32::MAX as usize {
            Err(AccelerateNativeError::DimensionError(format!(
                "{} = {} exceeds vDSP_Length max ({})",
                name,
                value,
                i32::MAX
            )))
        } else {
            Ok(value as vDSP_Length)
        }
    }

    /// Native vDSP vector addition: C = A + B
    ///
    /// # Safety
    /// - a_ptr, b_ptr must be valid pointers to n elements
    /// - c_ptr must be valid pointer to n elements (output)
    /// - n must be non-negative and within i32 range
    /// - All pointers must be properly aligned for f32
    unsafe fn native_vdsp_add(
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        n: vDSP_Length,
    ) {
        // vDSP_vadd: C = A + B, all stride 1
        vDSP_vadd(a_ptr, 1, b_ptr, 1, c_ptr, 1, n);
    }

    /// Native vDSP vector multiplication: C = A * B (elementwise)
    ///
    /// # Safety
    /// - a_ptr, b_ptr must be valid pointers to n elements
    /// - c_ptr must be valid pointer to n elements (output)
    /// - n must be non-negative and within i32 range
    /// - All pointers must be properly aligned for f32
    unsafe fn native_vdsp_mul(
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        n: vDSP_Length,
    ) {
        // vDSP_vmul: C = A * B, all stride 1
        vDSP_vmul(a_ptr, 1, b_ptr, 1, c_ptr, 1, n);
    }

    /// Native BLAS GEMM: C = alpha * A * B + beta * C
    ///
    /// For this PR, we use:
    /// - Layout: Row-major (CBLAS_ROW_MAJOR = 101)
    /// - TransA: No transpose (CBLAS_NO_TRANS = 111)
    /// - TransB: No transpose (CBLAS_NO_TRANS = 111)
    /// - alpha: 1.0
    /// - beta: 0.0 (so C = A * B, not C = A * B + C)
    ///
    /// # Safety
    /// - a_ptr must point to m * k elements with leading dimension lda
    /// - b_ptr must point to k * n elements with leading dimension ldb
    /// - c_ptr must point to m * n elements with leading dimension ldc
    /// - m, n, k must be non-negative and within i32 range
    /// - lda, ldb, ldc must be >= m, k, n respectively for row-major
    /// - All pointers must be properly aligned for f32
    unsafe fn native_cblas_sgemm(
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        m: CBLAS_INDEX,
        n: CBLAS_INDEX,
        k: CBLAS_INDEX,
        lda: CBLAS_INDEX,
        ldb: CBLAS_INDEX,
        ldc: CBLAS_INDEX,
    ) {
        // C = 1.0 * A * B + 0.0 * C
        cblas_sgemm(
            CBLAS_LAYOUT_ROW_MAJOR,
            CBLAS_TRANSPOSE_NO,
            CBLAS_TRANSPOSE_NO,
            m,
            n,
            k,
            1.0,
            a_ptr,
            lda,
            b_ptr,
            ldb,
            0.0,
            c_ptr,
            ldc,
        );
    }

    /// Safe wrapper for vDSP vector addition.
    ///
    /// Computes elementwise addition: output[i] = a[i] + b[i]
    ///
    /// # Arguments
    /// * `a` - First input vector
    /// * `b` - Second input vector
    ///
    /// # Returns
    /// * `Ok(output)` - Result vector of length a.len()
    /// * `Err` - If inputs are invalid or native call fails
    ///
    /// # Native Symbol
    /// Uses `vDSP_vadd` from Accelerate.framework
    pub fn vdsp_add(a: &[f32], b: &[f32]) -> AccelerateNativeResult<(Vec<f32>, NativeSymbol)> {
        // Validate inputs
        if a.is_empty() {
            return Err(AccelerateNativeError::EmptyInput);
        }
        if a.len() != b.len() {
            return Err(AccelerateNativeError::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let n = a.len();
        let n_vdsp = validate_vdsp_length(n, "vector length")?;

        // Allocate output
        let mut output = vec![0.0f32; n];

        // SAFETY:
        // - a.as_ptr() points to n valid f32 elements
        // - b.as_ptr() points to n valid f32 elements
        // - output.as_mut_ptr() points to n valid f32 elements (allocated above)
        // - n_vdsp is validated to be within i32 range and non-negative
        // - All strides are 1 (contiguous)
        // - Pointers are properly aligned for f32 (vec! guarantees this)
        unsafe {
            native_vdsp_add(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), n_vdsp);
        }

        Ok((output, NativeSymbol::VdspVadd))
    }

    /// Safe wrapper for vDSP vector multiplication.
    ///
    /// Computes elementwise multiplication: output[i] = a[i] * b[i]
    ///
    /// # Arguments
    /// * `a` - First input vector
    /// * `b` - Second input vector
    ///
    /// # Returns
    /// * `Ok(output)` - Result vector of length a.len()
    /// * `Err` - If inputs are invalid or native call fails
    ///
    /// # Native Symbol
    /// Uses `vDSP_vmul` from Accelerate.framework
    pub fn vdsp_mul(a: &[f32], b: &[f32]) -> AccelerateNativeResult<(Vec<f32>, NativeSymbol)> {
        // Validate inputs
        if a.is_empty() {
            return Err(AccelerateNativeError::EmptyInput);
        }
        if a.len() != b.len() {
            return Err(AccelerateNativeError::LengthMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let n = a.len();
        let n_vdsp = validate_vdsp_length(n, "vector length")?;

        // Allocate output
        let mut output = vec![0.0f32; n];

        // SAFETY:
        // - a.as_ptr() points to n valid f32 elements
        // - b.as_ptr() points to n valid f32 elements
        // - output.as_mut_ptr() points to n valid f32 elements (allocated above)
        // - n_vdsp is validated to be within i32 range and non-negative
        // - All strides are 1 (contiguous)
        // - Pointers are properly aligned for f32
        unsafe {
            native_vdsp_mul(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), n_vdsp);
        }

        Ok((output, NativeSymbol::VdspVmul))
    }

    /// Safe wrapper for BLAS GEMM (matrix multiplication).
    ///
    /// Computes: C = A * B
    /// Where:
    /// - A is m × k (row-major)
    /// - B is k × n (row-major)
    /// - C is m × n (row-major)
    ///
    /// # Arguments
    /// * `a` - Left matrix (m × k, row-major)
    /// * `b` - Right matrix (k × n, row-major)
    /// * `m` - Rows of A and C
    /// * `k` - Columns of A, rows of B
    /// * `n` - Columns of B and C
    ///
    /// # Returns
    /// * `Ok(output)` - Result matrix of shape m × n
    /// * `Err` - If inputs are invalid or native call fails
    ///
    /// # Native Symbol
    /// Uses `cblas_sgemm` from Accelerate.framework
    pub fn cblas_sgemm_row_major(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> AccelerateNativeResult<(Vec<f32>, NativeSymbol)> {
        // Validate dimensions
        if m == 0 || k == 0 || n == 0 {
            return Err(AccelerateNativeError::EmptyInput);
        }

        // Validate buffer sizes
        let expected_a_len = m * k;
        let expected_b_len = k * n;
        let expected_c_len = m * n;

        if a.len() != expected_a_len {
            return Err(AccelerateNativeError::LengthMismatch {
                expected: expected_a_len,
                actual: a.len(),
            });
        }
        if b.len() != expected_b_len {
            return Err(AccelerateNativeError::LengthMismatch {
                expected: expected_b_len,
                actual: b.len(),
            });
        }

        // Validate dimensions fit in CBLAS_INDEX (i32)
        let m_cblas = validate_cblas_index(m, "m")?;
        let n_cblas = validate_cblas_index(n, "n")?;
        let k_cblas = validate_cblas_index(k, "k")?;

        // For row-major matrices:
        // - A: m × k, leading dimension = k (contiguous rows)
        // - B: k × n, leading dimension = n (contiguous rows)
        // - C: m × n, leading dimension = n (contiguous rows)
        let lda = k_cblas; // leading dimension of A
        let ldb = n_cblas; // leading dimension of B
        let ldc = n_cblas; // leading dimension of C

        // Allocate output
        let mut output = vec![0.0f32; expected_c_len];

        // SAFETY:
        // - a.as_ptr() points to m*k valid f32 elements (row-major)
        // - b.as_ptr() points to k*n valid f32 elements (row-major)
        // - output.as_mut_ptr() points to m*n valid f32 elements (row-major)
        // - m, n, k validated to be within i32 range and positive
        // - lda = k >= m (for row-major, lda must be >= m, but we use k which is >= 1)
        // - ldb = n >= k (for row-major, ldb must be >= k, but we use n which is >= 1)
        // - ldc = n >= m (for row-major, ldc must be >= m, but we use n which is >= 1)
        // - Pointers are properly aligned for f32
        // - Using row-major layout with no transpose
        unsafe {
            native_cblas_sgemm(
                a.as_ptr(),
                b.as_ptr(),
                output.as_mut_ptr(),
                m_cblas,
                n_cblas,
                k_cblas,
                lda,
                ldb,
                ldc,
            );
        }

        Ok((output, NativeSymbol::CblasSgemm))
    }
}

/// Non-macOS stub implementations.
///
/// These provide the same safe Rust signatures but return BackendUnavailable,
/// ensuring the public API remains portable.
#[cfg(not(target_os = "macos"))]
pub mod platform {
    use super::*;

    /// Stub for vDSP vector addition - not available on non-macOS.
    pub fn vdsp_add(_a: &[f32], _b: &[f32]) -> AccelerateNativeResult<(Vec<f32>, NativeSymbol)> {
        Err(AccelerateNativeError::BackendUnavailable)
    }

    /// Stub for vDSP vector multiplication - not available on non-macOS.
    pub fn vdsp_mul(_a: &[f32], _b: &[f32]) -> AccelerateNativeResult<(Vec<f32>, NativeSymbol)> {
        Err(AccelerateNativeError::BackendUnavailable)
    }

    /// Stub for BLAS GEMM - not available on non-macOS.
    pub fn cblas_sgemm_row_major(
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> AccelerateNativeResult<(Vec<f32>, NativeSymbol)> {
        Err(AccelerateNativeError::BackendUnavailable)
    }
}

/// High-level native dispatcher.
///
/// This provides a unified interface to native Accelerate operations,
/// handling platform availability and fallback gracefully.
pub struct NativeDispatcher;

impl NativeDispatcher {
    /// Attempts native vDSP addition.
    ///
    /// Returns Ok if native execution succeeded, Err if native is unavailable
    /// or inputs are invalid.
    pub fn vdsp_add(a: &[f32], b: &[f32]) -> AccelerateNativeResult<(Vec<f32>, NativeSymbol)> {
        platform::vdsp_add(a, b)
    }

    /// Attempts native vDSP multiplication.
    pub fn vdsp_mul(a: &[f32], b: &[f32]) -> AccelerateNativeResult<(Vec<f32>, NativeSymbol)> {
        platform::vdsp_mul(a, b)
    }

    /// Attempts native BLAS GEMM.
    pub fn cblas_sgemm_row_major(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> AccelerateNativeResult<(Vec<f32>, NativeSymbol)> {
        platform::cblas_sgemm_row_major(a, b, m, k, n)
    }

    /// Returns true if native Accelerate is available on this platform.
    pub fn is_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            true
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_dispatcher_availability() {
        #[cfg(target_os = "macos")]
        {
            assert!(NativeDispatcher::is_available());
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert!(!NativeDispatcher::is_available());
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_non_macos_stubs() {
        // On non-macOS, all native calls should return BackendUnavailable
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = NativeDispatcher::vdsp_add(&a, &b);
        assert!(matches!(result, Err(AccelerateNativeError::BackendUnavailable)));

        let result = NativeDispatcher::vdsp_mul(&a, &b);
        assert!(matches!(result, Err(AccelerateNativeError::BackendUnavailable)));

        let result = NativeDispatcher::cblas_sgemm_row_major(&a, &b, 1, 3, 1);
        assert!(matches!(result, Err(AccelerateNativeError::BackendUnavailable)));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_non_macos_length_mismatch() {
        // Even on non-macOS, we should validate inputs before returning Unavailable
        // Actually, the stubs don't validate, they just return Unavailable
        // This is acceptable for the stub behavior
        let a = vec![1.0, 2.0];
        let b = vec![3.0];

        let result = NativeDispatcher::vdsp_add(&a, &b);
        // On non-macOS, we get BackendUnavailable, not LengthMismatch
        // because the stub doesn't validate
        assert!(matches!(result, Err(AccelerateNativeError::BackendUnavailable)));
    }

    #[test]
    fn test_error_display() {
        let err = AccelerateNativeError::BackendUnavailable;
        assert!(err.to_string().contains("unavailable"));

        let err = AccelerateNativeError::LengthMismatch { expected: 5, actual: 3 };
        assert!(err.to_string().contains("Length mismatch"));
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("3"));

        let err = AccelerateNativeError::Unsupported("test reason".to_string());
        assert!(err.to_string().contains("test reason"));
    }

    #[test]
    fn test_native_symbol_display() {
        assert_eq!(NativeSymbol::VdspVadd.to_string(), "vDSP_vadd");
        assert_eq!(NativeSymbol::VdspVmul.to_string(), "vDSP_vmul");
        assert_eq!(NativeSymbol::CblasSgemm.to_string(), "cblas_sgemm");
    }
}
