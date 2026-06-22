//! Bridge between ComputeImage segments and tensor types (mlx-rs, candle).
//!
//! The canonical pipeline for ALL tensor loading:
//!
//! 1. ComputeImage is compiled (segments + manifest.json)
//! 2. Segments are mmap'd via MappedSegment (zero-copy, MAP_PRIVATE)
//! 3. Tensors are constructed from segment pointers (zero-copy, no allocation)
//! 4. Inference reads weights directly from mmap'd pages
//!
//! See `docs/compute-image-memory-architecture.md` for the full architecture.

use std::sync::Arc;

use mlx_rs::{Array, Dtype as MlxDtype};

use crate::compute_image::TensorEntry;
use crate::external_array::{new_external_array, StaticStorage};
use crate::mapped_image::MappedSegment;

/// Errors during ComputeImage tensor loading.
#[derive(Debug)]
pub enum ComputeImageLoadError {
    /// The tensor entry references a segment index that doesn't exist.
    SegmentNotFound(String),
    /// The storage dtype string is not recognized.
    UnsupportedDtype(String),
    /// The pointer calculation produced an invalid address.
    InvalidPointer(String),
    /// An MLX operation failed.
    Mlx(String),
}

impl std::fmt::Display for ComputeImageLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SegmentNotFound(s) => write!(f, "segment not found: {s}"),
            Self::UnsupportedDtype(s) => write!(f, "unsupported dtype: {s}"),
            Self::InvalidPointer(s) => write!(f, "invalid pointer: {s}"),
            Self::Mlx(s) => write!(f, "mlx error: {s}"),
        }
    }
}

impl std::error::Error for ComputeImageLoadError {}

/// Convert a ComputeImage storage dtype string to an mlx-rs Dtype.
fn to_mlx_dtype(storage_dtype: &str) -> Result<MlxDtype, ComputeImageLoadError> {
    match storage_dtype {
        "f32" | "F32" | "Float32" => Ok(MlxDtype::Float32),
        "bf16" | "BF16" | "BFloat16" => Ok(MlxDtype::Bfloat16),
        "f16" | "F16" | "Float16" => Ok(MlxDtype::Float16),
        "u8" | "U8" | "Uint8" => Ok(MlxDtype::Uint8),
        "i8" | "I8" | "Int8" => Ok(MlxDtype::Int8),
        "u32" | "U32" | "Uint32" => Ok(MlxDtype::Uint32),
        other => Err(ComputeImageLoadError::UnsupportedDtype(other.to_string())),
    }
}

// ---------------------------------------------------------------------------
// load_mlx_tensor — zero-copy mlx-rs Array from ComputeImage
// ---------------------------------------------------------------------------

/// Create a no-copy `mlx_rs::Array` from a ComputeImage tensor entry.
///
/// # Zero-copy guarantee
/// The returned Array reads directly from the MappedSegment's mmap'd pages.
/// No memory is allocated for the tensor data — only the small Array handle
/// is created on the heap.
///
/// # Safety
/// The returned `Array` borrows the segment's backing memory through
/// the `Arc<MappedSegment>` reference counted in the `StaticStorage`.
/// The segment must remain alive (the Arc must not drop) for the lifetime
/// of the returned Array.
pub fn load_mlx_tensor(
    segment: &Arc<MappedSegment>,
    entry: &TensorEntry,
) -> Result<Array, ComputeImageLoadError> {
    let offset = entry.offset as usize;
    let byte_len = entry.byte_length as usize;

    // Bounds check: offset + byte_len must fit within the segment
    if offset + byte_len > segment.len() {
        return Err(ComputeImageLoadError::InvalidPointer(format!(
            "tensor {} offset {} + byte_len {} exceeds segment length {}",
            entry.name,
            offset,
            byte_len,
            segment.len()
        )));
    }

    // Compute the pointer into the mmap'd segment
    let ptr = unsafe { segment.data_ptr().add(offset) };

    // Wrap in StaticStorage (non-owning, no deallocation)
    let storage: Arc<StaticStorage> = Arc::new(unsafe { StaticStorage::new(ptr, byte_len) });

    // Convert the logical shape to i32 for MLX
    let shape: Vec<i32> = entry.logical_shape.iter().map(|&d| d as i32).collect();

    // Convert storage dtype
    let dtype = to_mlx_dtype(&entry.storage_dtype)?;

    // Create the no-copy Array
    unsafe {
        new_external_array(
            storage as Arc<dyn crate::external_array::ExternalStorage + Send + Sync>,
            &shape,
            dtype,
        )
        .map_err(|e| ComputeImageLoadError::Mlx(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// load_mlx_tensor_by_name — lookup by logical name
// ---------------------------------------------------------------------------

/// Look up a tensor by logical name in the manifest and load it.
///
/// This is the high-level API: given a ComputeImage's loaded segments and
/// manifest, find the tensor by name and construct a no-copy Array.
pub fn load_mlx_tensor_by_name(
    name: &str,
    segments: &[Arc<MappedSegment>],
    tensor_table: &[TensorEntry],
) -> Result<Array, ComputeImageLoadError> {
    let entry = tensor_table
        .iter()
        .find(|e| e.name == name)
        .ok_or_else(|| {
            ComputeImageLoadError::SegmentNotFound(format!("tensor {name} not found"))
        })?;

    // Find the segment by ID (position in the segments vec)
    let segment = segments
        .get(entry.segment.parse::<usize>().unwrap_or(0) as usize)
        .ok_or_else(|| {
            ComputeImageLoadError::SegmentNotFound(format!(
                "segment {} for tensor {name} not loaded",
                entry.segment
            ))
        })?;

    load_mlx_tensor(segment, entry)
}

// ---------------------------------------------------------------------------
// load_all_weight_tensors — batch loader for weight tensors
// ---------------------------------------------------------------------------

/// Load all weight tensors from a ComputeImage into a name→Array map.
///
/// Only loads tensors with `crate::compute_image::TensorRole::Weight` — skips KV cache
/// initializers and scratch buffers.
///
/// Returns a HashMap of tensor_name → no-copy Array.
pub fn load_all_weight_tensors(
    segments: &[Arc<MappedSegment>],
    tensor_table: &[TensorEntry],
) -> Result<std::collections::HashMap<String, Array>, ComputeImageLoadError> {
    let mut result = std::collections::HashMap::new();

    for entry in tensor_table {
        if entry.role != "weight" && entry.role != "Weight" {
            continue;
        }

        let segment = segments
            .get(entry.segment.parse::<usize>().unwrap_or(0) as usize)
            .ok_or_else(|| {
                ComputeImageLoadError::SegmentNotFound(format!(
                    "segment {} for tensor {} not loaded",
                    entry.segment, entry.name
                ))
            })?;

        let arr = load_mlx_tensor(segment, entry)?;
        result.insert(entry.name.clone(), arr);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapped_image::MappedSegment;
    use std::path::Path;
    use std::sync::Arc;

    /// Create a minimal test segment file on disk and load a tensor from it.
    #[test]
    fn test_load_mlx_tensor_round_trip() {
        let dir = std::env::temp_dir().join("compute_image_bridge_test");
        let _ = std::fs::create_dir_all(&dir);

        let segment_path = dir.join("segment_000.bin");
        // Write 64 bytes of known data: [0u8, 1u8, ..., 63u8]
        let data: Vec<u8> = (0u8..64).collect();
        std::fs::write(&segment_path, &data).expect("write segment");

        let segment = MappedSegment::new(&segment_path, None).expect("mmap segment");

        // Create a tensor entry pointing at offset 0, byte_len 32, shape [8] of U8
        let entry = TensorEntry {
            id: 0,
            name: "test.tensor".into(),
            role: String::from("Weight"),
            layer: Some(0),
            segment: "0".into(),
            source_filename: "test".into(),
            source_sha256: String::new(),
            source_offset: 0,
            offset: 0,
            byte_length: 32,
            logical_dtype: "Uint8".into(),
            storage_dtype: "U8".into(),
            logical_shape: vec![32],
            physical_shape: vec![32],
            mutability: "read_only".into(),
            quantization: None,
            tensor_alignment_bytes: 16,
            layout_version: 1,
            artifact_bindings: Default::default(),
        };

        let arr = load_mlx_tensor(&segment, &entry).expect("load mlx tensor");
        assert_eq!(arr.size(), 32, "array has 32 elements");
        // The array should read directly from the mmap'd data — first 32 bytes are 0..31
        // We verify by checking shape only, as actual data reading requires eval
    }

    /// Test that out-of-bounds offset is caught.
    #[test]
    fn test_load_mlx_tensor_oob() {
        let dir = std::env::temp_dir().join("compute_image_bridge_oob_test");
        let _ = std::fs::create_dir_all(&dir);

        let segment_path = dir.join("segment_000.bin");
        std::fs::write(&segment_path, &[0u8; 32]).expect("write segment");

        let segment = MappedSegment::new(&segment_path, None).expect("mmap segment");

        let entry = TensorEntry {
            id: 0,
            name: "test.oob".into(),
            role: String::from("Weight"),
            layer: Some(0),
            segment: "0".into(),
            source_filename: "test".into(),
            source_sha256: String::new(),
            source_offset: 0,
            offset: 16,      // start at 16
            byte_length: 32, // but only 16 bytes remain
            logical_dtype: "Uint8".into(),
            storage_dtype: "U8".into(),
            logical_shape: vec![32],
            physical_shape: vec![32],
            mutability: "read_only".into(),
            quantization: None,
            tensor_alignment_bytes: 16,
            layout_version: 1,
            artifact_bindings: Default::default(),
        };

        let result = load_mlx_tensor(&segment, &entry);
        assert!(result.is_err(), "OOB access should fail");
    }

    /// Test load_mlx_tensor_by_name with a valid name.
    #[test]
    fn test_load_by_name() {
        let dir = std::env::temp_dir().join("compute_image_bridge_name_test");
        let _ = std::fs::create_dir_all(&dir);

        let segment_path = dir.join("segment_000.bin");
        std::fs::write(&segment_path, &[1u8; 64]).expect("write segment");

        let segment = MappedSegment::new(&segment_path, None).expect("mmap segment");

        let entry = TensorEntry {
            id: 0,
            name: "model.layers.0.weight".into(),
            role: String::from("Weight"),
            layer: Some(0),
            segment: "0".into(),
            source_filename: "test".into(),
            source_sha256: String::new(),
            source_offset: 0,
            offset: 0,
            byte_length: 64,
            logical_dtype: "Uint8".into(),
            storage_dtype: "U8".into(),
            logical_shape: vec![64],
            physical_shape: vec![64],
            mutability: "read_only".into(),
            quantization: None,
            tensor_alignment_bytes: 16,
            layout_version: 1,
            artifact_bindings: Default::default(),
        };

        let tensor_table = vec![entry];
        let segments = vec![segment];

        let arr = load_mlx_tensor_by_name("model.layers.0.weight", &segments, &tensor_table)
            .expect("load by name");

        assert_eq!(arr.size(), 64, "loaded tensor has 64 elements");
    }
}
