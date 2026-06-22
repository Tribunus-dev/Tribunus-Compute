//! Zero-copy data bridge between candle Tensors and mlx-rs Arrays.
//!
//! On Apple Silicon with `StorageModeShared` Metal buffers, memory allocated
//! by either framework resides in the same physical pool (Unified Memory
//! Architecture / UMA).  This module lets both frameworks reference the same
//! bytes without copying.
//!
//! # Architecture
//!
//! ```text
//! UnifiedMemoryBlock   (owns Vec<u8>)
//!   ├── data_ptr()   → *const u8      ──→ candle CpuStorage
//!   ├── as_mlx_array() → Array        ──→ zero-copy via StaticStorage
//!   └── byte_len()   → usize
//!
//! bytes_to_mlx_array(vec, shape, dtype) → Array   (zero-copy via OwnedBuffer)
//! mlx_array_to_bytes(arr)               → Vec<u8> (eval + copy out)
//! ```
//!
//! # Safety
//!
//! - `UnifiedMemoryBlock` **must** outlive any `Array` returned by
//!   [`as_mlx_array`].  The Array's data pointer references the block's
//!   internal `Vec<u8>` without shared ownership or a lifetime bound.

use mlx_rs::{Array, Dtype};
use std::alloc::Layout;
use std::sync::Arc;

use crate::external_array::{new_external_array, ExternalStorage, OwnedBuffer, StaticStorage};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the element size in bytes for a given MLX dtype.
fn dtype_byte_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Bool => 1,
        Dtype::Uint8 => 1,
        Dtype::Uint16 => 2,
        Dtype::Uint32 => 4,
        Dtype::Uint64 => 8,
        Dtype::Int8 => 1,
        Dtype::Int16 => 2,
        Dtype::Int32 => 4,
        Dtype::Int64 => 8,
        Dtype::Float16 => 2,
        Dtype::Float32 => 4,
        Dtype::Float64 => 8,
        Dtype::Bfloat16 => 2,
        Dtype::Complex64 => 8,
    }
}

/// Compute the byte length of an MLX array given its shape and dtype.
fn shape_byte_len(shape: &[i32], dtype: Dtype) -> usize {
    let n: usize = shape.iter().copied().map(|d| d as usize).product();
    n * dtype_byte_size(dtype)
}

// ---------------------------------------------------------------------------
// UnifiedMemoryBlock
// ---------------------------------------------------------------------------

/// A raw memory region that can be viewed as either a candle Tensor
/// (via [`data_ptr`]) or an mlx-rs `Array` (via [`as_mlx_array`]).
///
/// On Apple Silicon the bytes are allocated on the unified heap and are
/// directly accessible to both CPU and GPU without copies.
pub struct UnifiedMemoryBlock {
    data: Vec<u8>,
    mlx_array: Option<Arc<Array>>,
}

impl UnifiedMemoryBlock {
    /// Allocate a new unified memory block of `size` bytes (zero-filled).
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            mlx_array: None,
        }
    }

    /// Create an mlx-rs `Array` that shares this block's memory **without
    /// copying**.
    ///
    /// The returned `Array` references the block's internal `Vec<u8>` data
    /// through an `Arc<StaticStorage>`; the deleter does **not** free the
    /// underlying memory.  This is a true zero-copy path — no bytes are moved.
    ///
    /// # Safety
    ///
    /// The caller **must** ensure that this `UnifiedMemoryBlock` outlives any
    /// `Array` returned by this method.  Dropping the block while an Array
    /// still references its data produces a dangling pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if `shape` and `dtype` require a different number of
    /// bytes than this block's capacity.
    pub fn as_mlx_array(&self, shape: &[i32], dtype: Dtype) -> Result<Array, String> {
        let expected = shape_byte_len(shape, dtype);
        if expected != self.data.len() {
            return Err(format!(
                "UnifiedMemoryBlock::as_mlx_array: size mismatch: \
                 shape {shape:?}/{dtype:?} needs {expected} bytes, block has {} bytes",
                self.data.len()
            ));
        }

        // If we already created an external array, return a new handle to the
        // same underlying MLX object (cheap Arc clone of the Array).
        if let Some(arc_arr) = &self.mlx_array {
            return Ok((**arc_arr).clone());
        }

        // Zero-copy: wrap the Vec's data pointer in a StaticStorage.
        // The deleter callback drops the Arc<StaticStorage>, which is a no-op
        // (StaticStorage does not deallocate).
        let storage = Arc::new(unsafe { StaticStorage::new(self.data.as_ptr(), self.data.len()) })
            as Arc<dyn ExternalStorage + Send + Sync>;

        let arr = unsafe { new_external_array(storage, shape, dtype) }
            .map_err(|e| format!("UnifiedMemoryBlock::as_mlx_array: {:?}", e))?;

        Ok(arr)
    }

    /// Get a raw pointer to the data (for candle `Tensor` construction via
    /// `CpuStorage`, or for direct CPU access).
    ///
    /// The pointer is valid for at least [`byte_len`] bytes and remains
    /// valid for the lifetime of this block.
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Return the block's byte length.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

// `Vec<u8>` is already `Send + Sync`.
unsafe impl Send for UnifiedMemoryBlock {}
unsafe impl Sync for UnifiedMemoryBlock {}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Convert an mlx-rs `Array`'s internal data to a new `Vec<u8>`.
///
/// This **evaluates** the array first (forcing any lazy Metal operations to
/// complete) and then copies the result into a freshly allocated `Vec<u8>`.
/// On Apple Silicon the read-back is from the same physical memory, so there
/// is no GPU→CPU transfer overhead.
///
/// # Errors
///
/// Returns an error if evaluation fails, if the array's data is not
/// contiguous, or if the dtype has no Rust-native representation.
pub fn mlx_array_to_bytes(arr: &Array) -> Result<Vec<u8>, String> {
    // Force any lazy operations to complete.
    arr.eval()
        .map_err(|e| format!("mlx_array_to_bytes: eval failed: {:?}", e))?;

    let nbytes = arr.nbytes();
    if nbytes == 0 {
        return Ok(Vec::new());
    }

    // Try to read as the most common dtypes.  We use try_as_slice which
    // returns a borrowed slice of the array's internal buffer (available
    // after eval for contiguous arrays).
    let bytes: Vec<u8> = match arr.dtype() {
        Dtype::Uint8 => {
            let slice = arr
                .try_as_slice::<u8>()
                .map_err(|e| format!("mlx_array_to_bytes (u8): {:?}", e))?;
            slice.to_vec()
        }
        Dtype::Float32 => {
            // Read as f32 then transmute the bits to bytes.
            let slice: &[f32] = arr
                .try_as_slice::<f32>()
                .map_err(|e| format!("mlx_array_to_bytes (f32): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Float16 | Dtype::Bfloat16 => {
            // Read as u16 (2-byte elements) then reinterpret.
            let slice: &[u16] = arr
                .try_as_slice::<u16>()
                .map_err(|e| format!("mlx_array_to_bytes (f16/bf16): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Int32 => {
            let slice: &[i32] = arr
                .try_as_slice::<i32>()
                .map_err(|e| format!("mlx_array_to_bytes (i32): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Uint32 => {
            let slice: &[u32] = arr
                .try_as_slice::<u32>()
                .map_err(|e| format!("mlx_array_to_bytes (u32): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Int64 => {
            let slice: &[i64] = arr
                .try_as_slice::<i64>()
                .map_err(|e| format!("mlx_array_to_bytes (i64): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Uint64 => {
            let slice: &[u64] = arr
                .try_as_slice::<u64>()
                .map_err(|e| format!("mlx_array_to_bytes (u64): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Int16 | Dtype::Uint16 => {
            let slice: &[u16] = arr
                .try_as_slice::<u16>()
                .map_err(|e| format!("mlx_array_to_bytes (i16/u16): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Int8 => {
            let slice: &[i8] = arr
                .try_as_slice::<i8>()
                .map_err(|e| format!("mlx_array_to_bytes (i8): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Bool => {
            let slice: &[bool] = arr
                .try_as_slice::<bool>()
                .map_err(|e| format!("mlx_array_to_bytes (bool): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Float64 => {
            let slice: &[f64] = arr
                .try_as_slice::<f64>()
                .map_err(|e| format!("mlx_array_to_bytes (f64): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
        Dtype::Complex64 => {
            // Complex64 is two f32s per element.
            let slice: &[f32] = arr
                .try_as_slice::<f32>()
                .map_err(|e| format!("mlx_array_to_bytes (complex64): {:?}", e))?;
            let ptr = slice.as_ptr() as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec()
        }
    };

    Ok(bytes)
}

/// Convert a `Vec<u8>` into an mlx-rs `Array` without copying.
///
/// `Vec`'s backing allocation is transferred to an [`OwnedBuffer`] and handed
/// to MLX's zero-copy external-array constructor.  The caller's `Vec` is
/// consumed (its memory is now managed by the MLX deleter).
///
/// This is a true zero-copy operation on Apple Silicon — the data stays in
/// place and is referenced directly by the MLX array.
///
/// # Errors
///
/// Returns an error if `shape` and `dtype` require a different number of
/// bytes than `data.len()`, or if the MLX external-array constructor fails.
pub fn bytes_to_mlx_array(data: Vec<u8>, shape: &[i32], dtype: Dtype) -> Result<Array, String> {
    let expected = shape_byte_len(shape, dtype);
    if expected != data.len() {
        return Err(format!(
            "bytes_to_mlx_array: size mismatch: shape {shape:?}/{dtype:?} \
             needs {expected} bytes, got {} bytes",
            data.len()
        ));
    }

    if data.is_empty() {
        let buf = OwnedBuffer::new(0);
        let storage: Arc<dyn ExternalStorage + Send + Sync> = Arc::new(buf);
        return unsafe { new_external_array(storage, shape, dtype) }
            .map_err(|e| format!("bytes_to_mlx_array (empty): {:?}", e));
    }

    let len = data.len();
    let cap = data.capacity();

    // Vec<u8> uses the global allocator with alignment 1.
    // We need the original allocation layout so OwnedBuffer::drop can
    // deallocate correctly.  `Vec`'s `RawVec` uses
    // `Layout::array::<u8>(capacity)`, which is equivalent to
    // `Layout::from_size_align(cap, 1)`.
    let layout = Layout::from_size_align(cap, 1)
        .map_err(|_| "bytes_to_mlx_array: invalid layout from Vec capacity".to_string())?;

    let ptr = data.as_ptr() as *mut u8;
    // Prevent Vec::drop from freeing the allocation (OwnedBuffer takes over).
    std::mem::forget(data);

    // SAFETY: `ptr` was allocated with `layout` by the global allocator, `len`
    // bytes are initialized, and `OwnedBuffer::drop` will dealloc with the same
    // layout.  Capacity equals layout.size() so the `capacity ≤ layout.size()`
    // contract of `from_raw` is satisfied.
    let buf = unsafe { OwnedBuffer::from_raw(ptr, len, cap, layout) };
    let storage: Arc<dyn ExternalStorage + Send + Sync> = Arc::new(buf);

    unsafe { new_external_array(storage, shape, dtype) }
        .map_err(|e| format!("bytes_to_mlx_array: {:?}", e))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // UnifiedMemoryBlock round-trip: write → MLX multiply → read back
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_block_round_trip() {
        let shape = &[2i32, 4i32];
        let n: usize = (shape[0] * shape[1]) as usize;
        let byte_len = n * 4; // Float32

        let block = UnifiedMemoryBlock::new(byte_len);

        // Write known f32 values into the block via the raw pointer.
        let ptr = block.data_ptr() as *mut f32;
        for i in 0..n {
            unsafe {
                *ptr.add(i) = i as f32;
            }
        }

        // Create a zero-copy MLX Array that shares the block's memory.
        let arr = block
            .as_mlx_array(shape, Dtype::Float32)
            .expect("as_mlx_array");

        // Run a simple Metal op: multiply every element by 2.
        let two = Array::from_slice(&[2.0f32], &[1]);
        let result = arr.multiply(&two).expect("multiply");

        // Eval and read back.
        let bytes = mlx_array_to_bytes(&result).expect("mlx_array_to_bytes");
        let values: &[f32] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n) };

        for i in 0..n {
            let expected = (i as f32) * 2.0;
            assert!(
                (values[i] - expected).abs() < 1e-5,
                "result[{i}]: expected {expected}, got {}",
                values[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // bytes_to_mlx_array round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_bytes_to_mlx_array_round_trip() {
        let shape = &[3i32, 2i32];
        let n: usize = (shape[0] * shape[1]) as usize;
        let values: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();

        let bytes: Vec<u8> =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, n * 4).to_vec() };

        let arr = bytes_to_mlx_array(bytes, shape, Dtype::Float32).expect("bytes_to_mlx_array");

        // Read back.
        let readback = mlx_array_to_bytes(&arr).expect("mlx_array_to_bytes");
        let readback_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(readback.as_ptr() as *const f32, n) };

        for i in 0..n {
            let expected = i as f32 * 1.5;
            assert!(
                (readback_f32[i] - expected).abs() < 1e-5,
                "readback[{i}]: expected {expected}, got {}",
                readback_f32[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // UnifiedMemoryBlock size-mismatch error
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_block_size_mismatch() {
        let block = UnifiedMemoryBlock::new(16); // 4 × float32, but shape claims 8 elements
        let err = block
            .as_mlx_array(&[8i32], Dtype::Float32)
            .expect_err("should fail on size mismatch");
        assert!(
            err.contains("size mismatch"),
            "error should mention size mismatch: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // mlx_array_to_bytes on an empty array
    // -----------------------------------------------------------------------

    #[test]
    fn test_mlx_array_to_bytes_empty() {
        let arr = Array::from_slice::<f32>(&[], &[0]);
        let bytes = mlx_array_to_bytes(&arr).expect("empty array to bytes");
        assert!(bytes.is_empty(), "empty array should produce empty bytes");
    }

    // -----------------------------------------------------------------------
    // bytes_to_mlx_array size-mismatch error
    // -----------------------------------------------------------------------

    #[test]
    fn test_bytes_to_mlx_array_size_mismatch() {
        let data = vec![0u8; 8]; // 8 bytes, but shape + dtype need 16
        let err = bytes_to_mlx_array(data, &[4i32], Dtype::Float32)
            .expect_err("should fail on size mismatch");
        assert!(
            err.contains("size mismatch"),
            "error should mention size mismatch: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // data_ptr and byte_len accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_block_accessors() {
        let block = UnifiedMemoryBlock::new(64);
        assert_eq!(block.byte_len(), 64);
        assert!(!block.data_ptr().is_null(), "data pointer must not be null");
    }
}
