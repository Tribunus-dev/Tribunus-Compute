//! ComputeImage runtime — segment activation and layer execution.

pub use super::manifest::{ImageRuntime, LayerLease};

use super::manifest::{
    mlx_active_memory_bytes, mlx_cache_memory_bytes, QuantizedLinearBinding, TensorEntry,
};
use crate::backend::MlxBackend;
use crate::projection_executor::{
    MaterializationClass, ProjectionExecutor, QuantizedProjectionDescriptor, RuntimeMode,
    StorageDtype,
};
use crate::projection_identity;
use crate::session::SamplerConfig;
use mlx_rs::Array;
use std::collections::HashMap;
use std::time::Instant;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Convert raw bytes + dtype + shape into an `mlx_rs::Array`.
fn dtype_to_array(bytes: &[u8], dtype: &str, shape: &[u32]) -> crate::Result<Array> {
    let dims = shape.iter().map(|&dim| dim as i32).collect::<Vec<_>>();
    match dtype {
        "U8" | "Uint8" => Ok(Array::from_slice(bytes, &dims)),
        "U32" | "Uint32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "u32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "I8" | "Int8" => {
            let data = bytes.iter().map(|&byte| byte as i8).collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "I32" | "Int32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "i32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "F32" | "Float32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "f32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "BF16" | "BFloat16" => {
            if bytes.len() % 2 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "bf16 payload length is not a multiple of 2: {}",
                    bytes.len()
                )));
            }
            // Convert BF16 to F32 for MLX compute compatibility
            let data = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bf = u16::from_le_bytes([chunk[0], chunk[1]]);
                    // BF16 to F32: shift left 16, reinterpret as f32
                    f32::from_bits((bf as u32) << 16)
                })
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        other => Err(crate::Error::from_reason(format!(
            "unsupported tensor storage dtype: {}",
            other
        ))),
    }
}

/// Returns the process resident set size in bytes, or 0 if unavailable.
fn process_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut u32,
                task_info_count: *mut u32,
            ) -> i32;
            fn mach_task_self() -> u32;
        }
        // TASK_VM_INFO = 22, mach_vm_size_t phys_footprint is at offset 4 (u64).
        // We use TASK_BASIC_INFO (flavor=5) which has resident_size at word 1.
        const TASK_BASIC_INFO: u32 = 5;
        const TASK_BASIC_INFO_COUNT: u32 = 10; // words
        let mut info = [0u32; 10];
        let mut count = TASK_BASIC_INFO_COUNT;
        let ret = unsafe {
            task_info(
                mach_task_self(),
                TASK_BASIC_INFO,
                info.as_mut_ptr(),
                &mut count,
            )
        };
        if ret == 0 && count >= 2 {
            // resident_size is the second field (u32 words on 32-bit, but mach
            // struct is actually two natural_t for virtual/resident on 64-bit).
            // Read as little-endian u64 from words 1..3.
            let lo = info[1] as u64;
            let hi = info[2] as u64;
            return (hi << 32) | lo;
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Linux: parse /proc/self/status VmRSS line.
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Ok(kb) = rest.trim().trim_end_matches(" kB").parse::<u64>() {
                        return kb * 1024;
                    }
                }
            }
        }
        0
    }
}

// ── ImageRuntime implementation ───────────────────────────────────────────

impl ImageRuntime {
    /// Load persistent segment tensors (embeddings, final norm) into the
    /// ARRAY_REGISTRY so that every layer forward pass can reference them.
    pub(crate) fn activate_persistent(&mut self) -> crate::Result<()> {
        let persistent_segment_ids = &self.manifest.residency_plan.persistent_segments;
        for seg_id in persistent_segment_ids {
            let segment = self
                .manifest
                .segments
                .iter()
                .find(|s| &s.id == seg_id)
                .ok_or_else(|| {
                    crate::Error::from_reason(format!("persistent segment not found: {}", seg_id))
                })?;

            let bytes = std::fs::read(self.image_dir.join(&segment.filename)).map_err(|e| {
                crate::Error::from_reason(format!(
                    "read persistent segment {}: {}",
                    segment.filename, e
                ))
            })?;
            self.total_bytes_activated += bytes.len() as u64;

            for &tensor_id in &segment.tensor_ids {
                let entry = self
                    .manifest
                    .tensor_table
                    .iter()
                    .find(|e| e.id == tensor_id)
                    .ok_or_else(|| {
                        crate::Error::from_reason(format!("tensor {} not in table", tensor_id))
                    })?;

                let slice = Self::slice_tensor_bytes(&bytes, entry)?;
                let array = dtype_to_array(slice, &entry.storage_dtype, &entry.physical_shape)?;
                let handle = crate::bridge::ARRAY_REGISTRY.write().insert(array, None);
                self.persistent_handles.insert(entry.name.clone(), handle);
            }
        }

        // Build quantized bindings for persistent tensors (embeddings).
        self.rebuild_quantized_bindings_from_persistent()?;
        Ok(())
    }

    /// Activate the tensors for a single layer by reading its segment from disk.
    /// Returns a `LayerLease` whose `Drop` impl releases the tensors from
    /// `ARRAY_REGISTRY`.
    ///
    /// **IMPORTANT**: the caller MUST call `hidden.eval()` before dropping the
    /// lease, otherwise the lazy MLX graph may hold dangling references to
    /// released arrays.
    pub fn activate_layer(&self, layer_index: u32) -> crate::Result<LayerLease> {
        let seg_id = format!("layer_{}", layer_index);
        let segment = self
            .manifest
            .segments
            .iter()
            .find(|s| s.id == seg_id)
            .ok_or_else(|| {
                crate::Error::from_reason(format!("layer segment not found: {}", seg_id))
            })?;

        let bytes = std::fs::read(self.image_dir.join(&segment.filename)).map_err(|e| {
            crate::Error::from_reason(format!("read layer segment {}: {}", segment.filename, e))
        })?;
        let bytes_read = bytes.len() as u64;

        let mut handles = Vec::new();
        for &tensor_id in &segment.tensor_ids {
            let entry = self
                .manifest
                .tensor_table
                .iter()
                .find(|e| e.id == tensor_id)
                .ok_or_else(|| {
                    crate::Error::from_reason(format!("tensor {} not in table", tensor_id))
                })?;

            let slice = Self::slice_tensor_bytes(&bytes, entry)?;
            let array = dtype_to_array(slice, &entry.storage_dtype, &entry.physical_shape)?;
            let handle = crate::bridge::ARRAY_REGISTRY.write().insert(array, None);
            handles.push(handle);
        }

        Ok(LayerLease {
            layer_index,
            segment_id: seg_id,
            bytes_read,
            handles,
        })
    }

    /// Slice the raw bytes for a specific tensor entry out of a segment
    /// payload.
    fn slice_tensor_bytes<'a>(
        segment_bytes: &'a [u8],
        entry: &TensorEntry,
    ) -> crate::Result<&'a [u8]> {
        let start = entry.offset as usize;
        let end = start + entry.byte_length as usize;
        if end > segment_bytes.len() {
            return Err(crate::Error::from_reason(format!(
                "tensor {} offset {}..{} exceeds segment length {}",
                entry.name,
                start,
                end,
                segment_bytes.len()
            )));
        }
        Ok(&segment_bytes[start..end])
    }

    /// Build a `LayerArrays`-equivalent lookup by reading active tensor
    /// handles from `ARRAY_REGISTRY` for the given layer. Both persistent
    /// handles (for embeddings needed during the layer forward pass) and the
    /// just-activated layer handles (currently in `ARRAY_REGISTRY` under the
    /// handles owned by the lease) are accessible via `self.lookup_handle()`.
    #[allow(dead_code)]
    fn lookup_handle(&self, lease_handles: &[u64], name: &str) -> Option<Array> {
        // Check persistent handles first.
        if let Some(&h) = self.persistent_handles.get(name) {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            return reg.get(h).cloned();
        }
        // Check the active layer handles by matching tensor names.
        // We match by name through the registry since LayerLease stores
        // handles in tensor_id order; we need to map name -> handle.
        // Build a temporary name->handle map from the last segment's
        // tensor_ids. (This is only called from run_six_layer_prefix, which
        // holds a lease.)
        let _ = lease_handles; // not needed; name lookup goes through ARRAY_REGISTRY scan
        None
    }

    /// Build a per-layer tensor lookup from the lease's handles and the
    /// manifest tensor table. Returns a `HashMap<name, Array>` for the layer.
    pub(crate) fn build_layer_arrays_from_lease(
        &self,
        layer_index: u32,
        lease: &LayerLease,
    ) -> crate::Result<HashMap<String, Array>> {
        let seg_id = format!("layer_{}", layer_index);
        let segment = self
            .manifest
            .segments
            .iter()
            .find(|s| s.id == seg_id)
            .ok_or_else(|| crate::Error::from_reason(format!("segment {} not found", seg_id)))?;

        if segment.tensor_ids.len() != lease.handles.len() {
            return Err(crate::Error::from_reason(format!(
                "layer {} segment has {} tensors but lease has {} handles",
                layer_index,
                segment.tensor_ids.len(),
                lease.handles.len()
            )));
        }

        let reg = crate::bridge::ARRAY_REGISTRY.read();
        let mut map = HashMap::new();
        for (&tensor_id, &handle) in segment.tensor_ids.iter().zip(lease.handles.iter()) {
            let entry = self
                .manifest
                .tensor_table
                .iter()
                .find(|e| e.id == tensor_id)
                .ok_or_else(|| {
                    crate::Error::from_reason(format!("tensor {} not in table", tensor_id))
                })?;
            let array = reg.get(handle).cloned().ok_or_else(|| {
                crate::Error::from_reason(format!(
                    "handle {} not in registry for {}",
                    handle, entry.name
                ))
            })?;
            map.insert(entry.name.clone(), array);
        }
        Ok(map)
    }

    /// Rebuild quantized bindings from the currently active persistent
    /// handles.
    pub(crate) fn rebuild_quantized_bindings_from_persistent(&mut self) -> crate::Result<()> {
        self.quantized_bindings.clear();
        for entry in &self.manifest.tensor_table {
            // Only build bindings for tensors in persistent segments that
            // have quantization.
            if !self
                .manifest
                .residency_plan
                .persistent_segments
                .iter()
                .any(|pid| *pid == entry.segment)
            {
                continue;
            }
            if let Some(quantization) = &entry.quantization {
                let scales_entry = self
                    .manifest
                    .tensor_table
                    .iter()
                    .find(|e| e.id == quantization.scale_tensor_id)
                    .ok_or_else(|| {
                        crate::Error::from_reason(format!(
                            "missing scale tensor for {}",
                            entry.name
                        ))
                    })?;
                let biases_entry = self
                    .manifest
                    .tensor_table
                    .iter()
                    .find(|e| e.id == quantization.bias_tensor_id)
                    .ok_or_else(|| {
                        crate::Error::from_reason(format!("missing bias tensor for {}", entry.name))
                    })?;

                let w_handle = *self.persistent_handles.get(&entry.name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing persistent handle: {}", entry.name))
                })?;
                let s_handle =
                    *self
                        .persistent_handles
                        .get(&scales_entry.name)
                        .ok_or_else(|| {
                            crate::Error::from_reason(format!(
                                "missing persistent scale handle: {}",
                                scales_entry.name
                            ))
                        })?;
                let b_handle =
                    *self
                        .persistent_handles
                        .get(&biases_entry.name)
                        .ok_or_else(|| {
                            crate::Error::from_reason(format!(
                                "missing persistent bias handle: {}",
                                biases_entry.name
                            ))
                        })?;

                let binding = QuantizedLinearBinding::new(
                    w_handle,
                    s_handle,
                    b_handle,
                    entry.logical_shape[0],
                    entry.logical_shape[1],
                    quantization.group_size,
                    quantization.bits,
                    true,
                );
                self.quantized_bindings.insert(entry.name.clone(), binding);
            }
        }
        Ok(())
    }

    /// Number of quantized bindings for persistent tensors (fixture test
    /// assertion).
    pub fn quantized_binding_count(&self) -> usize {
        self.quantized_bindings.len()
    }

    /// Number of persistent tensor handles currently active.
    pub fn persistent_handle_count(&self) -> usize {
        self.persistent_handles.len()
    }

    /// Total bytes activated across all segment reads (persistent + layer
    /// activations).
    pub fn total_bytes_activated(&self) -> u64 {
        self.total_bytes_activated
    }

    /// Execute the six-layer prefix using segment-scoped residency.
    ///
    /// For each layer:
    ///   1. Activate the layer segment (reads from disk, registers arrays).
    ///   2. Build the layer forward pass using persistent + layer arrays.
    ///   3. Force evaluation of the hidden state (eval before retire).
    ///   4. Drop the `LayerLease`, releasing that layer's arrays.
    ///
    /// Per-layer telemetry is emitted to stderr for residency verification.
    pub fn run_six_layer_prefix(&mut self) -> crate::Result<Array> {
        if self.released {
            return Err(crate::Error::from_reason("image runtime already released"));
        }

        let arch = self.manifest.architecture.clone();
        // Detect namespace root from compiled tensor handles.
        let root = if self.persistent_handles.contains_key("language_model.model.embed_tokens.weight") {
            "language_model.model"
        } else if self.persistent_handles.contains_key("model.embed_tokens.weight") {
            "model"
        } else {
            "model"
        };
        let layer_count = usize::min(
            6,
            usize::min(arch.layer_types.len(), arch.num_hidden_layers as usize),
        );

        // Embed using persistent tensors.
        let emb_w_name = format!("{}.embed_tokens.weight", root);
        let emb_s_name = format!("{}.embed_tokens.scales", root);
        let emb_b_name = format!("{}.embed_tokens.biases", root);

        let (emb_w, emb_s, emb_b) = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            let emb_w = reg
                .get(*self.persistent_handles.get(&emb_w_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing persistent tensor: {}", emb_w_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed weight handle invalid"))?;
            let emb_s = reg
                .get(*self.persistent_handles.get(&emb_s_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing persistent tensor: {}", emb_s_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed scales handle invalid"))?;
            let emb_b = reg
                .get(*self.persistent_handles.get(&emb_b_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing persistent tensor: {}", emb_b_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed biases handle invalid"))?;
            (emb_w, emb_s, emb_b)
        };

        let tok = Array::from_slice(&[2i32], &[1]);
        let mut hidden =
            crate::primitives::quantized_embedding_lookup(&tok, &emb_w, &emb_s, &emb_b)
                .map_err(|e| crate::Error::from_reason(format!("embed lookup: {:?}", e)))?
                .multiply(&Array::from_f32((arch.hidden_size as f32).sqrt()))
                .map_err(|e| crate::Error::from_reason(format!("embed scale: {:?}", e)))?;

        let (rope_cos, rope_sin) = crate::primitives::rope_freqs(
            arch.head_dim,
            arch.max_position_embeddings,
            arch.rope_local.theta as f32,
        )
        .map_err(|e| crate::Error::from_reason(format!("rope local: {:?}", e)))?;
        let full_rope = arch.rope_global.as_ref().unwrap_or(&arch.rope_local);
        let (full_cos, full_sin) = crate::primitives::rope_freqs(
            arch.global_head_dim.unwrap_or(arch.head_dim),
            arch.max_position_embeddings,
            full_rope.theta as f32,
        )
        .map_err(|e| crate::Error::from_reason(format!("rope global: {:?}", e)))?;

        for layer in 0..layer_count {
            let t0 = Instant::now();
            let rss_before = process_rss_bytes();
            let _active_before = mlx_active_memory_bytes();
            let _cached_before = mlx_cache_memory_bytes();
            let handles_before = crate::bridge::handle_count();

            // Activate this layer's segment (reads from disk).
            let lease = self.activate_layer(layer as u32).map_err(|e| {
                crate::Error::from_reason(format!("activate layer {}: {}", layer, e))
            })?;
            let bytes_read = lease.bytes_read;

            // Build the layer tensor map from the lease.
            let layer_map = self.build_layer_arrays_from_lease(layer as u32, &lease)?;

            // Helper closure to look up a tensor by name.
            let get_tensor = |name: &str| -> crate::Result<Array> {
                if let Some(arr) = layer_map.get(name) {
                    return Ok(arr.clone());
                }
                if let Some(&h) = self.persistent_handles.get(name) {
                    let reg = crate::bridge::ARRAY_REGISTRY.read();
                    return reg.get(h).cloned().ok_or_else(|| {
                        crate::Error::from_reason(format!("persistent handle invalid for {}", name))
                    });
                }
                Err(crate::Error::from_reason(format!(
                    "tensor not found for layer {}: {}",
                    layer, name
                )))
            };

            let base = format!("{}.layers.{}", root, layer);
            let is_full = matches!(
                arch.layer_types[layer],
                crate::config::AttentionKind::FullAttention
            );

            let attn_norm = get_tensor(&format!("{}.input_layernorm.weight", base))?;
            let ffn_norm = get_tensor(&format!("{}.post_attention_layernorm.weight", base))?;
            let (qw, qs, qb) = (
                get_tensor(&format!("{}.self_attn.q_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.q_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.q_proj.biases", base))?,
            );
            let (kw, ks, kb) = (
                get_tensor(&format!("{}.self_attn.k_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.k_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.k_proj.biases", base))?,
            );
            let (vw, vs, vb) = if !is_full {
                (
                    get_tensor(&format!("{}.self_attn.v_proj.weight", base))?,
                    get_tensor(&format!("{}.self_attn.v_proj.scales", base))?,
                    get_tensor(&format!("{}.self_attn.v_proj.biases", base))?,
                )
            } else {
                (
                    Array::from_slice(&[0.0f32], &[1]),
                    Array::from_slice(&[0.0f32], &[1]),
                    Array::from_slice(&[0.0f32], &[1]),
                )
            };
            let (ow, os, ob) = (
                get_tensor(&format!("{}.self_attn.o_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.o_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.o_proj.biases", base))?,
            );
            let (gw, gs, gb) = (
                get_tensor(&format!("{}.mlp.gate_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.gate_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.gate_proj.biases", base))?,
            );
            let (uw, us, ub) = (
                get_tensor(&format!("{}.mlp.up_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.up_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.up_proj.biases", base))?,
            );
            let (dw, ds, db) = (
                get_tensor(&format!("{}.mlp.down_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.down_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.down_proj.biases", base))?,
            );

            // Run the layer forward pass.
            let layer_arrays = crate::model::LayerArraysRef {
                attn_norm: &attn_norm,
                ffn_norm: &ffn_norm,
                qw: &qw,
                qs: &qs,
                qb: &qb,
                kw: &kw,
                ks: &ks,
                kb: &kb,
                vw: &vw,
                vs: &vs,
                vb: &vb,
                ow: &ow,
                os: &os,
                ob: &ob,
                gw: &gw,
                gs: &gs,
                gb: &gb,
                uw: &uw,
                us: &us,
                ub: &ub,
                dw: &dw,
                ds: &ds,
                db: &db,
            };

            hidden = if is_full {
                crate::model::run_full_layer_arrays(
                    &hidden,
                    &layer_arrays,
                    &arch,
                    &full_cos,
                    &full_sin,
                    0,
                )
                .map_err(|e| crate::Error::from_reason(format!("layer {} full: {:?}", layer, e)))?
            } else {
                crate::model::run_sliding_layer_arrays(
                    &hidden,
                    &layer_arrays,
                    &arch,
                    &rope_cos,
                    &rope_sin,
                    0,
                )
                .map_err(|e| {
                    crate::Error::from_reason(format!("layer {} sliding: {:?}", layer, e))
                })?
            };

            // *** CRITICAL: eval BEFORE dropping lease ***
            // MLX is lazy -- the graph still references the layer arrays
            // until eval() forces the computation. Dropping the lease before
            // eval leaves the graph with dead backing storage.
            hidden
                .eval()
                .map_err(|e| crate::Error::from_reason(format!("eval layer {}: {:?}", layer, e)))?;

            let elapsed_ms = t0.elapsed().as_millis();
            let rss_after = process_rss_bytes();
            let active_after = mlx_active_memory_bytes();
            let cached_after = mlx_cache_memory_bytes();
            let handles_after = crate::bridge::handle_count();

            // Emit per-layer residency receipt.
            let _rss_evaluated = rss_after;
            let active_evaluated = active_after;
            let cached_evaluated = cached_after;
            let handles_evaluated = handles_after;
            let seg_id = lease.segment_id.clone();

            // *** Retire the layer segment. ***
            // hidden.eval() has already forced the kernel to consume the
            // weights.
            drop(lease);

            // Capture telemetry AFTER retirement to prove logical release.
            let rss_retired = process_rss_bytes();
            let active_retired = mlx_active_memory_bytes();
            let cached_retired = mlx_cache_memory_bytes();
            let handles_retired = crate::bridge::handle_count();

            eprintln!(
                "[image-runtime] layer={} segment={} bytes_read={} elapsed_ms={} \
                 rss_delta={} mlx_active={}->{} mlx_cached={}->{} handles={}->{}->{}",
                layer,
                seg_id,
                bytes_read,
                elapsed_ms,
                rss_retired as i64 - rss_before as i64,
                active_evaluated,
                active_retired,
                cached_evaluated,
                cached_retired,
                handles_before,
                handles_evaluated,
                handles_retired,
            );
        }

        // Final norm + LM head projection using persistent embed tensors.
        let fn_w_name = format!("{}.norm.weight", root);
        let fn_w = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            reg.get(*self.persistent_handles.get(&fn_w_name).ok_or_else(|| {
                crate::Error::from_reason(format!("missing persistent tensor: {}", fn_w_name))
            })?)
            .cloned()
            .ok_or_else(|| crate::Error::from_reason("norm weight handle invalid"))?
        };
        let final_hidden = crate::primitives::rms_norm(&hidden, &fn_w, 1e-6)
            .map_err(|e| crate::Error::from_reason(format!("final norm: {:?}", e)))?;

        // LM head aliases embed_tokens (tie_word_embeddings); reuse emb_w.
        let out = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            let ew =
                reg.get(*self.persistent_handles.get(&emb_w_name).ok_or_else(|| {
                    crate::Error::from_reason("embed weight gone before lm_head")
                })?)
                .cloned()
                .ok_or_else(|| {
                    crate::Error::from_reason("embed weight handle invalid at lm_head")
                })?;
            let es =
                reg.get(*self.persistent_handles.get(&emb_s_name).ok_or_else(|| {
                    crate::Error::from_reason("embed scales gone before lm_head")
                })?)
                .cloned()
                .ok_or_else(|| {
                    crate::Error::from_reason("embed scales handle invalid at lm_head")
                })?;
            let eb =
                reg.get(*self.persistent_handles.get(&emb_b_name).ok_or_else(|| {
                    crate::Error::from_reason("embed biases gone before lm_head")
                })?)
                .cloned()
                .ok_or_else(|| {
                    crate::Error::from_reason("embed biases handle invalid at lm_head")
                })?;
            // Look up quantization metadata from the manifest tensor table.
            let (_qbits, _qgroup_size, _storage_dtype_str) = {
                let table = &self.manifest.tensor_table;
                let entry = table.iter().find(|t| t.name == emb_w_name).ok_or_else(|| {
                    crate::Error::from_reason("embed_tokens.weight not found in tensor table")
                })?;
                let q = entry.quantization.as_ref().ok_or_else(|| {
                    crate::Error::from_reason("embed_tokens.weight missing quantization descriptor")
                })?;
                (q.bits as u8, q.group_size, entry.storage_dtype.clone())
            };
            let storage_dtype = match _storage_dtype_str.as_str() {
                "U8" => StorageDtype::U8,
                "I8" => StorageDtype::I8,
                _ => StorageDtype::U32,
            };
            let w_shape: Vec<u32> = ew.shape().iter().map(|&d| d as u32).collect();
            let desc = QuantizedProjectionDescriptor {
                family: projection_identity::ProjectionFamily::LmHead,
                logical_in_features: arch.hidden_size,
                logical_out_features: arch.vocab_size,
                bits: _qbits,
                group_size: _qgroup_size,
                storage_dtype,
                physical_weight_shape: w_shape,
                layer_index: 0,
                weight_materialization: MaterializationClass::MlxOwned,
            };
            let mut backend = MlxBackend::new();
            let x_h = backend.alloc(final_hidden);
            let w_h = backend.alloc_weight(ew);
            let s_h = backend.alloc(es);
            let b_h = backend.alloc(eb);
            let result_h = {
                let mut executor = ProjectionExecutor {
                    backend: &mut backend,
                    mode: RuntimeMode::Safe,
                };
                executor
                    .run_projection(x_h, w_h, s_h, b_h, &desc)
                    .map_err(|e| {
                        crate::Error::from_reason(format!("lm_head projection: {:?}", e))
                    })?
            };
            backend
                .get(result_h)
                .map_err(|e| crate::Error::from_reason(format!("lm_head result: {:?}", e)))?
                .clone()
        };
        out.eval()
            .map_err(|e| crate::Error::from_reason(format!("final eval: {:?}", e)))?;

        self.release();
        Ok(out)
    }

    /// Execute the complete 48-layer model from the compiled execution plan.
    ///
    /// This is the canonical forward path:
    ///   1. Run the prologue (embedding -> hidden state)
    ///   2. For each layer in the execution plan:
    ///      a. Activate the layer segment
    ///      b. Run the layer executor from the compiled plan
    ///      c. eval() before dropping the lease
    ///      d. Record per-layer telemetry
    ///   3. Run the epilogue (final norm -> output projection -> softcap ->
    ///      argmax)
    ///
    /// Returns a `u32` token ID -- no logits cross the boundary.
    /// Per-layer receipts are emitted to stderr.
    pub fn run_full_model(&mut self, token_ids: &[i32]) -> crate::Result<u32> {
        if self.released {
            return Err(crate::Error::from_reason("image runtime already released"));
        }

        let plan = &self.manifest.execution_plan;
        plan.validate().map_err(|errors| {
            crate::Error::from_reason(format!(
                "execution plan validation failed: {}",
                errors.join("; ")
            ))
        })?;

        let arch = &self.manifest.architecture;
        // Detect namespace root from compiled tensor handles.
        let root = if self.persistent_handles.contains_key("language_model.model.embed_tokens.weight") {
            "language_model.model"
        } else if self.persistent_handles.contains_key("model.embed_tokens.weight") {
            "model"
        } else {
            // Probe the first persistent handle's prefix as fallback.
            let first_key = self.persistent_handles.keys().next().cloned().unwrap_or_default();
            if first_key.starts_with("language_model.") { "language_model.model" }
            else { "model" }
        };
        let seq_len = token_ids.len() as i32;

        // --- Prologue: embedding lookup ---
        let emb_w_name = format!("{}.embed_tokens.weight", root);
        let emb_s_name = format!("{}.embed_tokens.scales", root);
        let emb_b_name = format!("{}.embed_tokens.biases", root);

        let (emb_w, emb_s, emb_b) = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            let w =
                reg.get(*self.persistent_handles.get(&emb_w_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing: {}", emb_w_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed weight invalid"))?;
            let s =
                reg.get(*self.persistent_handles.get(&emb_s_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing: {}", emb_s_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed scales invalid"))?;
            let b =
                reg.get(*self.persistent_handles.get(&emb_b_name).ok_or_else(|| {
                    crate::Error::from_reason(format!("missing: {}", emb_b_name))
                })?)
                .cloned()
                .ok_or_else(|| crate::Error::from_reason("embed biases invalid"))?;
            (w, s, b)
        };

        let tok = Array::from_slice(token_ids, &[1, seq_len]);
        let mut hidden = crate::executor::run_prologue(
            &tok,
            &emb_w,
            &emb_s,
            &emb_b,
            &plan.prologue,
            (arch.hidden_size as f32).sqrt(),
        )
        .map_err(|e| crate::Error::from_reason(format!("prologue: {:?}", e)))?;

        hidden
            .eval()
            .map_err(|e| crate::Error::from_reason(format!("prologue eval: {:?}", e)))?;

        // Precompute RoPE tables
        let (rope_cos, rope_sin) = crate::primitives::rope_freqs(
            arch.head_dim,
            arch.max_position_embeddings,
            arch.rope_local.theta as f32,
        )
        .map_err(|e| crate::Error::from_reason(format!("rope local: {:?}", e)))?;
        let full_rope = arch.rope_global.as_ref().unwrap_or(&arch.rope_local);
        let (full_cos, full_sin) = crate::primitives::rope_freqs(
            arch.global_head_dim.unwrap_or(arch.head_dim),
            arch.max_position_embeddings,
            full_rope.theta as f32,
        )
        .map_err(|e| crate::Error::from_reason(format!("rope global: {:?}", e)))?;

        // Build per-layer KV caches for single-pass validation
        let max_seq_len = arch.max_position_embeddings.min(8192);
        let mut caches: Vec<crate::kv_cache::KvCache> = Vec::with_capacity(plan.layers.len());
        for layer_plan in &plan.layers {
            let is_sliding = layer_plan.attention_kind == "sliding_attention";
            let (capacity, n_kv_heads, head_dim) = if is_sliding {
                (
                    layer_plan.sliding_window,
                    layer_plan.n_kv_heads,
                    layer_plan.head_dim,
                )
            } else {
                let g_kv = layer_plan.n_global_kv_heads.unwrap_or(1);
                let g_hd = layer_plan.global_head_dim.unwrap_or(layer_plan.head_dim);
                (max_seq_len, g_kv, g_hd)
            };
            caches.push(crate::kv_cache::KvCache::new(
                capacity, n_kv_heads, head_dim, is_sliding,
            ));
        }

        let idle_handles = crate::bridge::handle_count();
        eprintln!(
            "[full-model] idle_handles={} layer_count={}",
            idle_handles,
            plan.layers.len(),
        );
        // --- Decoder layers ---
        for layer_plan in &plan.layers {
            let l = layer_plan.layer_index;
            let t0 = Instant::now();
            let handles_before = crate::bridge::handle_count();
            let active_before = mlx_active_memory_bytes();

            // Activate the layer segment
            let lease = self
                .activate_layer(l)
                .map_err(|e| crate::Error::from_reason(format!("activate layer {}: {}", l, e)))?;
            let bytes_read = lease.bytes_read;

            // Build layer tensor map from the lease
            let layer_map = self.build_layer_arrays_from_lease(l, &lease)?;

            // Helper to look up a tensor
            let get_tensor = |name: &str| -> crate::Result<Array> {
                if let Some(arr) = layer_map.get(name) {
                    return Ok(arr.clone());
                }
                if let Some(&h) = self.persistent_handles.get(name) {
                    let reg = crate::bridge::ARRAY_REGISTRY.read();
                    return reg.get(h).cloned().ok_or_else(|| {
                        crate::Error::from_reason(format!("persistent handle invalid for {}", name))
                    });
                }
                Err(crate::Error::from_reason(format!(
                    "tensor not found for layer {}: {}",
                    l, name
                )))
            };

            let base = format!("{}.layers.{}", root, l);
            let is_full = layer_plan.attention_kind == "full_attention";

            let attn_norm = get_tensor(&format!("{}.input_layernorm.weight", base))?;
            let ffn_norm = get_tensor(&format!("{}.post_attention_layernorm.weight", base))?;
            let (qw, qs, qb) = (
                get_tensor(&format!("{}.self_attn.q_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.q_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.q_proj.biases", base))?,
            );
            let (kw, ks, kb) = (
                get_tensor(&format!("{}.self_attn.k_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.k_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.k_proj.biases", base))?,
            );
            let (vw, vs, vb) = if is_full {
                // K-equals-V: reuse k_proj
                (kw.clone(), ks.clone(), kb.clone())
            } else {
                (
                    get_tensor(&format!("{}.self_attn.v_proj.weight", base))?,
                    get_tensor(&format!("{}.self_attn.v_proj.scales", base))?,
                    get_tensor(&format!("{}.self_attn.v_proj.biases", base))?,
                )
            };
            let (ow, os, ob) = (
                get_tensor(&format!("{}.self_attn.o_proj.weight", base))?,
                get_tensor(&format!("{}.self_attn.o_proj.scales", base))?,
                get_tensor(&format!("{}.self_attn.o_proj.biases", base))?,
            );
            let (gw, gs, gb) = (
                get_tensor(&format!("{}.mlp.gate_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.gate_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.gate_proj.biases", base))?,
            );
            let (uw, us, ub) = (
                get_tensor(&format!("{}.mlp.up_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.up_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.up_proj.biases", base))?,
            );
            let (dw, ds, db) = (
                get_tensor(&format!("{}.mlp.down_proj.weight", base))?,
                get_tensor(&format!("{}.mlp.down_proj.scales", base))?,
                get_tensor(&format!("{}.mlp.down_proj.biases", base))?,
            );

            // Q/K norm weights
            let q_norm = get_tensor(&format!("{}.self_attn.q_norm.weight", base)).ok();
            let k_norm = get_tensor(&format!("{}.self_attn.k_norm.weight", base)).ok();

            // Select RoPE tables
            let (rcos, rsin) = if is_full {
                (&full_cos, &full_sin)
            } else {
                (&rope_cos, &rope_sin)
            };

            // Run the layer executor
            hidden = crate::executor::run_layer(
                &hidden,
                layer_plan,
                &crate::config::operation_route::OperationRoute::default(),
                None,
                &[],
                &attn_norm,
                &ffn_norm,
                &qw,
                &qs,
                &qb,
                &kw,
                &ks,
                &kb,
                &vw,
                &vs,
                &vb,
                &ow,
                &os,
                &ob,
                q_norm.as_ref(),
                k_norm.as_ref(),
                &gw,
                &gs,
                &gb,
                &uw,
                &us,
                &ub,
                &dw,
                &ds,
                &db,
                rcos,
                rsin,
                &mut caches[l as usize],
                0, // kv_offset = 0 for single-pass
                arch.rms_norm_eps as f32,
                &projection_identity::ProjectionContext {
                    run_id: "test".into(),
                    phase: projection_identity::Phase::Prefill,
                    forward_pass_index: 1,
                    token_step: None,
                    layer_index: l as usize,
                    attention_kind: projection_identity::AttentionKind::Sliding,
                },
            )
            .map_err(|e| crate::Error::from_reason(format!("layer {}: {:?}", l, e)))?;

            // *** CRITICAL: eval BEFORE dropping lease ***
            hidden
                .eval()
                .map_err(|e| crate::Error::from_reason(format!("eval layer {}: {:?}", l, e)))?;

            let elapsed_ms = t0.elapsed().as_millis();
            let handles_after = crate::bridge::handle_count();
            let _active_evaluated = mlx_active_memory_bytes();
            let seg_id = lease.segment_id.clone();

            // Retire the layer segment
            drop(lease);

            let handles_retired = crate::bridge::handle_count();
            let active_retired = mlx_active_memory_bytes();

            let output_shape = hidden.shape();
            let is_finite = hidden
                .try_as_slice::<f32>()
                .map(|v| v.iter().all(|x| x.is_finite()))
                .unwrap_or(false);

            eprintln!(
                "[full-model] layer={} kind={} segment={} bytes={} elapsed_ms={} \
                 handles={}->{}->{} active_mem={}->{} shape={:?} finite={}",
                l,
                layer_plan.attention_kind,
                seg_id,
                bytes_read,
                elapsed_ms,
                handles_before,
                handles_after,
                handles_retired,
                active_before,
                active_retired,
                output_shape,
                is_finite,
            );
        }

        // Verify return to idle
        let final_handles = crate::bridge::handle_count();
        eprintln!(
            "[full-model] all_layers_done final_handles={} idle_handles={}",
            final_handles, idle_handles,
        );

        // --- Epilogue: final norm + output projection + softcapping +
        //    argmax ---
        let fn_w_name = format!("{}.norm.weight", root);
        let fn_w = {
            let reg = crate::bridge::ARRAY_REGISTRY.read();
            reg.get(
                *self
                    .persistent_handles
                    .get(&fn_w_name)
                    .ok_or_else(|| crate::Error::from_reason(format!("missing: {}", fn_w_name)))?,
            )
            .cloned()
            .ok_or_else(|| crate::Error::from_reason("norm weight invalid"))?
        };

        let epi = crate::executor::run_epilogue(
            &hidden,
            &fn_w,
            &emb_w,
            &emb_s,
            &emb_b,
            &plan.epilogue,
            arch.rms_norm_eps as f32,
            arch.tie_word_embeddings,
            &SamplerConfig::default(),
        )
        .map_err(|e| crate::Error::from_reason(format!("epilogue: {:?}", e)))?;

        epi.selected_token
            .eval()
            .map_err(|e| crate::Error::from_reason(format!("epilogue eval: {:?}", e)))?;
        let token_id = epi
            .selected_token
            .try_as_slice::<u32>()
            .map_err(|e| crate::Error::from_reason(format!("epilogue token: {:?}", e)))?
            .first()
            .copied()
            .unwrap_or(0);

        self.release();
        Ok(token_id)
    }

    /// Release all persistent tensor handles.
    pub fn release(&mut self) {
        if self.released {
            return;
        }
        for handle in self
            .persistent_handles
            .values()
            .copied()
            .collect::<Vec<_>>()
        {
            let _ = crate::bridge::free_array(handle);
        }
        self.persistent_handles.clear();
        self.quantized_bindings.clear();
        self.released = true;
    }
}
