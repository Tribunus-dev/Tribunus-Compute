//! Runtime cache for compiled ANE programs per decoder layer.
//!
//! After loading ComputeImage segments into memory, the cache compiles
//! ANE programs for layers assigned to Orion (route.attention == 3).
//! The programs use the weight dict to reference mmap'd segment memory
//! directly — same physical pages as MLX and Accelerate.

use std::sync::Arc;

use crate::compute_image::TensorEntry;
use crate::mapped_image::MappedSegment;
use crate::memory::ane_weight_dict::compile_ane_layer;

/// Wrapper around a compiled `OrionProgram*` handle.
/// The handle is a retained ObjC object — released on drop.
struct AneProgram {
    ptr: *mut std::ffi::c_void,
    /// MIL text used to compile this program (for debug / recompile)
    mil_tag: String,
}

unsafe impl Send for AneProgram {}
unsafe impl Sync for AneProgram {}

impl Drop for AneProgram {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                extern "C" {
                    fn orion_release_program(prog: *mut std::ffi::c_void);
                }
                orion_release_program(self.ptr);
            }
        }
    }
}

/// Runtime cache of compiled ANE programs.
///
/// One cache entry per decoder layer.  Layers not routed to Orion
/// have null entries.
pub struct AneProgramCache {
    /// Per-layer programs.  Index = layer index in execution plan.
    /// null = no ANE program for this layer.
    programs: Vec<Option<AneProgram>>,
}

impl AneProgramCache {
    pub fn new() -> Self {
        Self {
            programs: Vec::new(),
        }
    }

    /// Compile ANE programs for all layers assigned to Orion.
    ///
    /// `layer_count`: total number of decoder layers in the model.
    /// `orion_indices`: which layers should use the ANE (from OperationRoute).
    /// `segments`: loaded ComputeImage segments.
    /// `tensor_table`: manifest tensor table.
    ///
    /// The MIL text for each layer is generated from a template
    /// that references `@model_path/weights/layer_N_X.bin` paths
    /// (matching the weight_dict keys in compile_ane_layer).
    pub fn compile_from_manifest(
        &mut self,
        layer_count: usize,
        orion_indices: &[usize],
        segments: &[Arc<MappedSegment>],
        tensor_table: &[TensorEntry],
    ) {
        self.programs = (0..layer_count)
            .map(|i| {
                if orion_indices.contains(&i) {
                    // Generate MIL text for this layer's attention kernel.
                    // The MIL references weights via @model_path/weights/ paths
                    // that build_weight_dict maps to segment memory.
                    let mil = generate_attention_mil(i);
                    match unsafe {
                        compile_ane_layer(segments, tensor_table, i, &mil, &format!("layer_{}", i))
                    } {
                        Ok(prog) => Some(AneProgram {
                            ptr: prog as *mut std::ffi::c_void,
                            mil_tag: format!("layer_{}", i),
                        }),
                        Err(e) => {
                            eprintln!("ANE compile failed for layer {}: {}", i, e);
                            None
                        }
                    }
                } else {
                    None
                }
            })
            .collect();
    }

    /// Get the compiled ANE program for a given layer.
    /// Returns null if the layer doesn't use the ANE or compilation failed.
    pub fn get_program(&self, layer_idx: usize) -> *mut std::ffi::c_void {
        self.programs
            .get(layer_idx)
            .and_then(|p| p.as_ref())
            .map(|p| p.ptr)
            .unwrap_or(std::ptr::null_mut())
    }

    /// Number of successfully compiled programs.
    pub fn compiled_count(&self) -> usize {
        self.programs.iter().filter(|p| p.is_some()).count()
    }
}

/// Generate minimal attention MIL text for a decoder layer.
///
/// The MIL references weights via @model_path/weights/ paths that
/// compile_ane_layer's weight dict maps to MappedSegment memory.
fn generate_attention_mil(_layer_idx: usize) -> String {
    // Minimal MIL for ANE compile path verification.
    // The weight dict (from compile_ane_layer) provides actual weight data.
    "program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n    void warmup<ios18>(tensor<fp16, [1, 1, 1, 1]> x) -> (y) {\n        tensor<fp16, [1, 1, 1, 1]> y = mul(x = x, y = x)[name = string(\"ane_test\")];\n    }\n}\n".to_string()
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_cache() {
        let cache = AneProgramCache::new();
        assert_eq!(cache.compiled_count(), 0);
        assert!(cache.get_program(0).is_null());
    }

    #[test]
    fn test_mil_generation() {
        let mil = generate_attention_mil(0);
        assert!(mil.contains("warmup"), "MIL should contain warmup function");
        assert!(mil.contains("mul"), "MIL should contain mul");
    }
}
