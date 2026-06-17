//! IOSurface-backed [`ExternalStorage`] for zero-copy MLX array construction.
//!
//! Bridges [`Arena`] (IOSurface + CVPixelBuffer) into the external array
//! subsystem so that MLX `Array` values can be built directly over GPU-shared
//! IOSurface memory without copying.
//!
//! # Safety
//!
//! The [`Arena`] must outlive every MLX `Array` constructed from it (the
//! `Arc` ownership inside [`IosurfaceStorage`] enforces this automatically).
//! The caller is responsible for ensuring that locking/unlocking of the
//! underlying `CVPixelBuffer` respects the access pattern of the consumer
//! (CPU reads require a lock; Metal GPU access does not).

use std::sync::Arc;

use mlx_rs::{Array, Dtype};

use crate::arena::Arena;
use crate::external_array::{new_external_array, ExternalStorage};

// ---------------------------------------------------------------------------
// IosurfaceStorage
// ---------------------------------------------------------------------------

/// [`ExternalStorage`] implementation backed by an IOSurface [`Arena`].
///
/// Enables zero-copy construction of `mlx_rs::Array` values from IOSurface
/// memory via [`new_external_array`].
///
/// The underlying [`Arena`] is kept alive through its `Arc` reference — when
/// MLX releases the array, the deleter callback drops its copy of the `Arc`
/// (which may be the last reference).
pub struct IosurfaceStorage {
    arena: Arc<Arena>,
}

impl IosurfaceStorage {
    /// Wrap an already-allocated [`Arc<Arena>`] into external storage.
    pub fn new(arena: Arc<Arena>) -> Self {
        IosurfaceStorage { arena }
    }

    /// Borrow the wrapped [`Arena`].
    pub fn arena(&self) -> &Arena {
        &self.arena
    }
}

impl ExternalStorage for IosurfaceStorage {
    fn data_ptr(&self) -> *const u8 {
        // Safety: base_ptr is valid as long as the Arena is alive (which the
        // Arc guarantees).  CVPixelBuffer base addresses are fixed after the
        // first lock, so no additional locking is needed to read the pointer.
        unsafe { self.arena.base_ptr() as *const u8 }
    }

    fn byte_len(&self) -> usize {
        self.arena.byte_len()
    }
}

// Safety: Arena is Send + Sync (its fields are Send + Sync or carry explicit
// unsafe impls), so Arc<Arena> and IosurfaceStorage inherit Send + Sync.
//
// The trait bound `ExternalStorage: Send + Sync` is satisfied automatically.
unsafe impl Send for IosurfaceStorage {}
unsafe impl Sync for IosurfaceStorage {}

// ---------------------------------------------------------------------------
// Convenience: arena_to_mlx_array
// ---------------------------------------------------------------------------

/// Convert an IOSurface-backed [`Arena`] into a no-copy `mlx_rs::Array`.
///
/// The returned `Array` shares the IOSurface memory directly — no element data
/// is copied.  The [`Arena`] is kept alive through the `Arc` chain; when MLX
/// releases the array the deleter callback drops the final reference.
///
/// # Errors
///
/// Returns an error string if the dtype is incompatible with the Arena (the
/// Arena is FP16-only) or if the external array construction fails.
///
/// # Safety
///
/// - The Arena's IOSurface memory must be valid for the lifetime of the
///   returned Array (the Arc ownership ensures this).
/// - The caller must ensure any CPU-side access is gated through
///   [`Arena::lock`] / [`Arena::unlock`].  Metal GPU access does not require
///   a CVPixelBuffer lock.
pub fn arena_to_mlx_array(arena: Arc<Arena>, shape: &[i32], dtype: Dtype) -> Result<Array, String> {
    let storage: Arc<dyn ExternalStorage + Send + Sync> = Arc::new(IosurfaceStorage::new(arena));

    // Safety: the Arc keeps the Arena alive until the deleter fires; the
    // IOSurface memory is page-aligned and valid for the Arena's lifetime.
    unsafe { new_external_array(storage, shape, dtype).map_err(|e| e.to_string()) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use mlx_rs::Dtype;

    use crate::arena::Arena;

    use super::*;

    #[test]
    fn test_iosurface_storage_round_trip() {
        // Allocate a small FP16 arena.
        let arena = Arena::new(1, 16, Dtype::Float16).expect("arena allocation failed");
        let arena = Arc::new(arena);

        let shape = &[1i32, 16i32];
        let array = arena_to_mlx_array(arena.clone(), shape, Dtype::Float16)
            .expect("arena_to_mlx_array failed");

        // The array should have the expected shape.
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.shape(), &[1, 16]);
        assert_eq!(array.dtype(), Dtype::Float16);
    }

    #[test]
    fn test_iosurface_storage_data_ptr() {
        let arena = Arena::new(1, 8, Dtype::Float16).expect("arena allocation failed");
        let arena = Arc::new(arena);

        let storage = IosurfaceStorage::new(arena.clone());
        assert!(!storage.data_ptr().is_null());
        assert_eq!(storage.byte_len(), arena.byte_len());

        // arena() returns a reference to the wrapped Arc.
        assert_eq!(storage.arena().byte_len(), arena.byte_len());
    }

    #[test]
    fn test_iosurface_storage_send_sync() {
        fn assert_send<T: Send>(_: &T) {}
        fn assert_sync<T: Sync>(_: &T) {}

        let arena = Arena::new(1, 4, Dtype::Float16).expect("arena allocation failed");
        let storage = IosurfaceStorage::new(Arc::new(arena));

        assert_send(&storage);
        assert_sync(&storage);
    }

    #[test]
    fn test_arena_to_mlx_array_wrong_dtype() {
        let arena = Arena::new(1, 4, Dtype::Float16).expect("arena allocation failed");
        let arena = Arc::new(arena);

        // Float32 is not supported — the Arena should reject it, but here we
        // test that arena_to_mlx_array propagates the error rather than panic.
        let result = arena_to_mlx_array(arena, &[1i32, 4i32], Dtype::Float32);
        assert!(result.is_err(), "expected error for unsupported dtype");
    }

    #[test]
    fn test_iosurface_storage_trait_object() {
        let arena = Arena::new(1, 8, Dtype::Float16).expect("arena allocation failed");
        let arena = Arc::new(arena);

        let storage: Arc<dyn ExternalStorage + Send + Sync> =
            Arc::new(IosurfaceStorage::new(arena));

        assert!(!storage.data_ptr().is_null());
        assert_eq!(storage.byte_len(), 1 * 8 * 2); // 1x8 FP16 = 16 bytes
    }
}
