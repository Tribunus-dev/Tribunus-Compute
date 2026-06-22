//! Build `weight_dict` for `orion_compile_mil()` from ComputeImage segments.
//!
//! The weight dict maps BLOBFILE paths (referenced in MIL text via
//! `@model_path/weights/<name>.bin`) to NSData objects wrapping mmap'd
//! segment memory (no copy).  The ANE compiler reads weights directly
//! from the same MappedSegment pages as MLX and Accelerate.

use std::ffi::CString;
use std::sync::Arc;

use crate::compute_image::TensorEntry;
use crate::mapped_image::MappedSegment;

/// Opaque NSDictionary pointer returned by the ObjC bridge.
/// Must be CFReleased when no longer needed.
#[repr(C)]
struct OpaqueDict(*mut std::ffi::c_void);

unsafe impl Send for OpaqueDict {}
unsafe impl Sync for OpaqueDict {}

/// A single weight blob entry for the FFI bridge.
#[repr(C)]
struct OrionWeightBlobEntry {
    blob_path: *const std::os::raw::c_char,
    data: *const std::ffi::c_void,
    length: usize,
    offset: u64,
}

extern "C" {
    fn build_ane_weight_dict(
        blobs: *const OrionWeightBlobEntry,
        count: usize,
    ) -> *mut std::ffi::c_void;
}

/// Build the ANE weight_dict NSDictionary from ComputeImage segments.
///
/// For each weight tensor in `tensor_table` belonging to `layer_idx`,
/// creates an NSData wrapping the mmap'd segment pointer (no copy)
/// and associates it with a BLOBFILE path used by the MIL text.
///
/// The returned pointer is a retained NSDictionary* — must CFRelease
/// after the ANE program is compiled.
///
/// Returns null if no weight blobs could be built.
pub unsafe fn build_weight_dict(
    segments: &[Arc<MappedSegment>],
    tensor_table: &[TensorEntry],
    layer_idx: usize,
) -> *mut std::ffi::c_void {
    // Build the entry list
    let mut ffi_entries: Vec<OrionWeightBlobEntry> = Vec::new();
    let mut cstrings: Vec<CString> = Vec::new();

    for entry in tensor_table {
        if entry.layer != Some(layer_idx as u32) {
            continue;
        }
        if entry.role != "weight" {
            continue;
        }

        let seg_idx: usize = entry.segment.parse().unwrap_or(0);
        let segment = match segments.get(seg_idx) {
            Some(s) => s,
            None => continue,
        };

        let ptr = segment.data_ptr().add(entry.offset as usize);
        let blob_key = name_to_blob_key(&entry.name);
        let blob_path = format!("@model_path/weights/layer_{}_{}.bin", layer_idx, blob_key);
        let cstr = CString::new(blob_path).unwrap();

        ffi_entries.push(OrionWeightBlobEntry {
            blob_path: cstr.as_ptr(),
            data: ptr as *const std::ffi::c_void,
            length: entry.byte_length as usize,
            offset: 0,
        });
        cstrings.push(cstr);
    }

    if ffi_entries.is_empty() {
        return std::ptr::null_mut();
    }

    build_ane_weight_dict(ffi_entries.as_ptr(), ffi_entries.len())
}

/// Convert a tensor name to a compact blob key.
fn name_to_blob_key(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// Compile an Orion ANE program with weights from ComputeImage segments.
///
/// This is the high-level API: given a ComputeImage, a layer index,
/// and MIL text for that layer's attention kernel:
///   1. Build the weight_dict pointing into MappedSegment memory
///   2. Call orion_compile_mil(mil_text, weight_dict, tag)
///   3. Release the weight_dict
///   4. Return the compiled program
///
/// The resulting ANE program reads weights from the same mmap'd pages
/// as MLX and Accelerate — zero-copy across all three backends.
pub unsafe fn compile_ane_layer(
    segments: &[Arc<MappedSegment>],
    tensor_table: &[TensorEntry],
    layer_idx: usize,
    mil_text: &str,
    tag: &str,
) -> Result<*const std::ffi::c_void, String> {
    // Declare orion_compile_mil once at function scope
    extern "C" {
        fn orion_compile_mil(
            mil_text: *const std::os::raw::c_char,
            wdict: *mut std::ffi::c_void,
            program_tag: *const std::os::raw::c_char,
        ) -> *mut std::ffi::c_void;
        fn CFRelease(obj: *const std::ffi::c_void);
    }

    let wdict = build_weight_dict(segments, tensor_table, layer_idx);
    let tag_cstr = CString::new(tag).map_err(|e| format!("tag cstring: {e}"))?;
    let mil_cstr = CString::new(mil_text).map_err(|e| format!("mil cstring: {e}"))?;
    let prog = orion_compile_mil(mil_cstr.as_ptr(), wdict, tag_cstr.as_ptr());
    // Release weight_dict — the ANE compiler wrote the weight data to
    // its temp directory during compilation, so the NSData refs are done.
    CFRelease(wdict as *const std::ffi::c_void);

    match prog.is_null() {
        false => Ok(prog as *const std::ffi::c_void),
        true => Err("orion_compile_mil returned null".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_to_blob_key() {
        assert_eq!(
            name_to_blob_key("model.layers.0.self_attn.q_proj.weight"),
            "weight"
        );
        assert_eq!(name_to_blob_key("model.embed_tokens.weight"), "weight");
        assert_eq!(name_to_blob_key("w"), "w");
    }

    #[test]
    fn test_build_weight_dict_empty() {
        let result = unsafe { build_weight_dict(&[], &[], 0) };
        assert!(result.is_null(), "empty segments should produce null dict");
    }
}
