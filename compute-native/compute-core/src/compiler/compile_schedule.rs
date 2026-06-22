//! Model-to-schedule compiler — translates a model manifest into a
//! [`ScheduledModule`] with populated regions, memory plan, transfer plan,
//! and evaluation boundaries.
//!
//! This is the bridge between the static model description (weights, layers,
//! shapes) and the runtime execution infrastructure that allocates IOSurface
//! slices, assigns backends, and sequences evaluation.

use crate::backend::routing::{
    BackendId, EvidenceDigest, OperationId, PhysicalLayout, TensorId, TensorShape,
};
use crate::backend::DType;
use crate::compiler::scheduled::{
    BufferReuse, DependencyKind, MemoryPlan, RegionDependency, RegionId, ScheduledModule,
    ScheduledRegion, SealedEvaluationBoundary, StorageClass, TransferPlan,
};
use crate::config::{LayerPlan, ModelExecutionPlan, TextArchitecture};
use std::collections::HashMap;

/// Compile a model manifest to a [`ScheduledModule`].
///
/// Each transformer layer becomes one region.  The regions are ordered
/// sequentially.  Evaluation boundaries are placed every 6 layers
/// (matching the OPT-0005 synchronization fence).  The memory plan
/// estimates per-layer peak memory and total budget.
pub fn compile_model_to_scheduled_module(
    plan: &ModelExecutionPlan,
    arch: &TextArchitecture,
    source_semantic_digest: EvidenceDigest,
) -> ScheduledModule {
    let mut module = ScheduledModule::new(source_semantic_digest);

    let n_layers = plan.layers.len() as u64;
    let hidden_size = arch.hidden_size as u64;
    let intermediate_size = arch.intermediate_size as u64;
    let seq_len = arch.max_position_embeddings.min(8192) as u64;
    let _vocab_size = arch.vocab_size as u64;

    // ── Compute per-layer memory ──
    //
    // Each layer needs:
    //   - Hidden state buffer: seq_len × hidden_size × 4 (FP32, worst case)
    //   - Attention QKV workspace: seq_len × (n_heads + 2×n_kv_heads) × head_dim × 4
    //   - Attention score matrix: n_heads × seq_len × seq_len × 4 (only at peak)
    //   - O projection: seq_len × hidden_size × 4
    //   - FFN gate/up: 2 × seq_len × intermediate_size × 4
    //   - FFN down: seq_len × hidden_size × 4
    //
    // In practice MLX fuses many of these, but we estimate the peak budget.

    let hidden_state_bytes = seq_len * hidden_size * 4;

    let per_layer_bytes: Vec<u64> = plan
        .layers
        .iter()
        .map(|l| {
            let hd = l.head_dim as u64;
            let nh = l.n_heads as u64;
            let nkv = l.n_kv_heads as u64;

            // Flash attention: no score matrix materialized.
            // QKV workspace (concurrent Q/K/V intermediates)
            let qkv_workspace = seq_len * (nh + 2 * nkv) * hd * 4;
            // FFN gate + up intermediates
            let ffn_inter = 2 * seq_len * intermediate_size * 4;

            // Peak: concurrent QKV workspace + FFN intermediates + hidden state I/O
            let peak = qkv_workspace.max(ffn_inter).max(hidden_state_bytes);

            peak
        })
        .collect();

    // ── Create regions: one per layer ──
    for (i, layer) in plan.layers.iter().enumerate() {
        let layer_id = i as u64;
        let is_full = layer.attention_kind == "full_attention";
        let hd = layer.head_dim as u64;
        let nh = layer.n_heads as u64;
        let nkv = layer.n_kv_heads as u64;

        let backend_id = BackendId(layer.route.dominant_backend() as u32);

        let mut physical_tensors = Vec::new();

        // Hidden state input/output tensor
        physical_tensors.push(physical_tensor(
            TensorId(layer_id * 100 + 1),
            format!("layer_{}_hidden_input", i),
            vec![seq_len as u32, hidden_size as u32],
            DType::F32,
            StorageClass::IoSurface,
            backend_id,
        ));
        physical_tensors.push(physical_tensor(
            TensorId(layer_id * 100 + 2),
            format!("layer_{}_hidden_output", i),
            vec![seq_len as u32, hidden_size as u32],
            DType::F32,
            StorageClass::IoSurface,
            backend_id,
        ));

        // KV cache K
        let cache_capacity = if is_full {
            seq_len
        } else {
            layer.sliding_window as u64
        };
        let kv_cache_size = if is_full {
            layer.n_global_kv_heads.unwrap_or(nkv as u32) as u64
        } else {
            nkv
        };
        let kv_hd = if is_full {
            layer.global_head_dim.unwrap_or(layer.head_dim) as u64
        } else {
            hd
        };

        physical_tensors.push(physical_tensor(
            TensorId(layer_id * 100 + 3),
            format!("layer_{}_k_cache", i),
            vec![cache_capacity as u32, kv_cache_size as u32, kv_hd as u32],
            DType::F32,
            StorageClass::IoSurface,
            backend_id,
        ));
        physical_tensors.push(physical_tensor(
            TensorId(layer_id * 100 + 4),
            format!("layer_{}_v_cache", i),
            vec![cache_capacity as u32, kv_cache_size as u32, kv_hd as u32],
            DType::F32,
            StorageClass::IoSurface,
            backend_id,
        ));

        // Attention Q/K/V intermediates
        for (suffix, n, h) in &[("q", nh, hd), ("k", nkv, hd), ("v", nkv, hd)] {
            physical_tensors.push(physical_tensor(
                TensorId(layer_id * 100 + 10 + suffix.as_bytes()[0] as u64),
                format!("layer_{}_attn_{}", i, suffix),
                vec![seq_len as u32, *n as u32, *h as u32],
                DType::F32,
                StorageClass::IoSurface,
                backend_id,
            ));
        }

        // FFN intermediates
        physical_tensors.push(physical_tensor(
            TensorId(layer_id * 100 + 20),
            format!("layer_{}_gate_proj", i),
            vec![seq_len as u32, intermediate_size as u32],
            DType::F32,
            StorageClass::IoSurface,
            backend_id,
        ));
        physical_tensors.push(physical_tensor(
            TensorId(layer_id * 100 + 21),
            format!("layer_{}_up_proj", i),
            vec![seq_len as u32, intermediate_size as u32],
            DType::F32,
            StorageClass::IoSurface,
            backend_id,
        ));
        physical_tensors.push(physical_tensor(
            TensorId(layer_id * 100 + 22),
            format!("layer_{}_down_proj", i),
            vec![seq_len as u32, hidden_size as u32],
            DType::F32,
            StorageClass::IoSurface,
            backend_id,
        ));

        // Dependencies: previous layer
        let dependencies = if i > 0 {
            vec![RegionDependency {
                predecessor: RegionId(i as u64 - 1),
                tensors: vec![TensorId((i as u64 - 1) * 100 + 2)],
                kind: DependencyKind::Data,
            }]
        } else {
            vec![]
        };

        // State effects: KV cache write
        let state_effects = vec![crate::compiler::scheduled::StateEffect::KvCacheWrite];

        module.regions.push(ScheduledRegion {
            region_id: RegionId(layer_id),
            name: format!("layer_{}_{}", i, layer.attention_kind),
            operations: vec![
                OperationId(layer_id * 10 + 1), // attention
                OperationId(layer_id * 10 + 2), // ffn
            ],
            selected_backend: backend_id,
            physical_tensors,
            inputs: vec![TensorId(layer_id * 100 + 1)],
            outputs: vec![TensorId(layer_id * 100 + 2)],
            dependencies,
            fusions: vec![],
            fusion_regions: vec![],
            state_effects,
            temp_memory_bytes: per_layer_bytes[i as usize],
            is_fence: false,
        });
    }

    // ── Memory plan ──
    let peak_layer_bytes: u64 = per_layer_bytes
        .iter()
        .max()
        .copied()
        .unwrap_or(hidden_state_bytes);
    let peak_bytes = per_layer_bytes
        .iter()
        .max()
        .copied()
        .unwrap_or(hidden_state_bytes);

    // Buffer reuse: adjacent layers' hidden inputs/outputs share storage
    let mut buffer_reuse = Vec::new();
    for i in 1..n_layers {
        let prev_out = TensorId((i - 1) * 100 + 2);
        let curr_in = TensorId(i * 100 + 1);
        buffer_reuse.push(BufferReuse {
            tensor_a: prev_out,
            tensor_b: curr_in,
            size_bytes: hidden_state_bytes,
        });
    }

    module.memory_plan = MemoryPlan {
        // Total runtime memory = one layer's peak + hidden state I/O (+ vocab embedding is a weight, not runtime temp)
        total_bytes: peak_layer_bytes + hidden_state_bytes,
        peak_bytes: peak_bytes.max(hidden_state_bytes),
        per_backend: {
            let mut m = HashMap::new();
            m.insert(BackendId(0), peak_layer_bytes); // MLX
            m.insert(BackendId(1), 0); // Accelerate
            m.insert(BackendId(2), 0); // Core ML
            m
        },
        aliases: vec![],
        buffer_reuse,
    };

    // ── Transfers: IOSurface ↔ backend buffers ──
    for (i, _layer) in plan.layers.iter().enumerate() {
        let _layer_id = i as u64;
        let in_tensor_id = TensorId(_layer_id * 100 + 1);
        let out_tensor_id = TensorId(_layer_id * 100 + 2);
        module.transfers.push(TransferPlan {
            source: in_tensor_id,
            destination: in_tensor_id,
            source_backend: BackendId(0),
            dest_backend: BackendId(0),
            source_storage: StorageClass::IoSurface,
            dest_storage: StorageClass::IoSurface,
            size_bytes: hidden_state_bytes,
            zero_copy_capable: true,
            zero_copy_verified: false,
            conversion_needed: false,
        });
        module.transfers.push(TransferPlan {
            source: out_tensor_id,
            destination: out_tensor_id,
            source_backend: BackendId(0),
            dest_backend: BackendId(0),
            source_storage: StorageClass::IoSurface,
            dest_storage: StorageClass::IoSurface,
            size_bytes: hidden_state_bytes,
            zero_copy_capable: true,
            zero_copy_verified: false,
            conversion_needed: false,
        });
    }

    // ── Evaluation boundaries (OPT-0005: every 6 layers) ──
    let boundary_interval = 6u64;
    for start in (0..n_layers).step_by(boundary_interval as usize) {
        let end = (start + boundary_interval).min(n_layers);
        let regions: Vec<RegionId> = (start..end).map(|i| RegionId(i)).collect();
        module.evaluation_boundaries.push(SealedEvaluationBoundary {
            regions,
            requires_fence: start > 0,
            same_backend: true,
        });
    }

    module.seal();
    module
}

/// Convenience: build memory plan from a layer plan and architecture.
/// Returns the estimated peak memory in bytes for a single layer.
pub fn estimate_layer_peak_memory(layer: &LayerPlan, arch: &TextArchitecture) -> u64 {
    let seq_len = arch.max_position_embeddings.min(8192) as u64;
    let hidden_size = arch.hidden_size as u64;
    let intermediate_size = arch.intermediate_size as u64;
    let hd = layer.head_dim as u64;
    let nh = layer.n_heads as u64;
    let nkv = layer.n_kv_heads as u64;

    let qkv_workspace = seq_len * (nh + 2 * nkv) * hd * 4;
    let ffn_inter = 2 * seq_len * intermediate_size * 4;

    qkv_workspace.max(ffn_inter).max(seq_len * hidden_size * 4)
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn physical_tensor(
    id: TensorId,
    name: String,
    shape: Vec<u32>,
    dtype: DType,
    storage: StorageClass,
    backend: BackendId,
) -> crate::compiler::scheduled::PhysicalTensor {
    crate::compiler::scheduled::PhysicalTensor {
        semantic_id: id,
        name,
        shape: TensorShape { dims: shape },
        dtype,
        layout: PhysicalLayout::RowMajor,
        storage_class: storage,
        backend,
        materialized: true,
        alignment: 4096,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_arch() -> TextArchitecture {
        TextArchitecture {
            diffusion_config: None,
            moe_config: Default::default(),
            hidden_size: 3840,
            intermediate_size: 15360,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 120,
            global_head_dim: Some(512),
            num_global_key_value_heads: Some(1),
            num_hidden_layers: 48,
            vocab_size: 256128,
            sliding_window: 8192,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            attention_k_eq_v: false,
            final_logit_softcapping: None,
            hidden_size_per_layer_input: 3840,
            layer_types: vec![crate::config::AttentionKind::SlidingAttention; 48],
            rope_local: crate::config::RopeSpec {
                theta: 500_000.0,
                rope_type: "default".into(),
                partial_rotary_factor: None,
            },
            rope_global: None,
            model_type: "gemma".into(),
        }
    }

    fn test_execution_plan(arch: &TextArchitecture) -> ModelExecutionPlan {
        let mut layers = Vec::new();
        for i in 0..48 {
            layers.push(LayerPlan {
                layer_index: i,
                attention_kind: "sliding_attention".into(),
                segment_id: "weights".into(),
                hidden_size: arch.hidden_size,
                n_heads: arch.num_attention_heads,
                n_kv_heads: arch.num_key_value_heads,
                head_dim: arch.head_dim,
                global_head_dim: if i % 6 == 5 {
                    arch.global_head_dim
                } else {
                    None
                },
                n_global_kv_heads: if i % 6 == 5 {
                    arch.num_global_key_value_heads
                } else {
                    None
                },
                sliding_window: arch.sliding_window,
                rope_theta: 500_000.0,
                partial_rotary_factor: None,
                attention_k_eq_v: false,
                q_norm_enabled: false,
                k_norm_enabled: false,
                q_proj_tensor_id: 100 + i * 10,
                k_proj_tensor_id: 101 + i * 10,
                v_proj_tensor_id: 102 + i * 10,
                o_proj_tensor_id: 103 + i * 10,
                q_norm_tensor_id: None,
                k_norm_tensor_id: None,
                gate_proj_tensor_id: 104 + i * 10,
                up_proj_tensor_id: 105 + i * 10,
                down_proj_tensor_id: 106 + i * 10,
                input_layernorm_tensor_id: 107 + i * 10,
                post_attention_layernorm_tensor_id: 108 + i * 10,
                pre_ffw_layernorm_tensor_id: None,
                post_ffw_layernorm_tensor_id: None,
                layer_scalar_ids: vec![],
                quantization_ids: vec![],
                route: crate::config::operation_route::OperationRoute::default(),
                fused_operations: Default::default(),
            });
        }

        ModelExecutionPlan {
            prologue: crate::config::ProloguePlan::default(),
            layers,
            epilogue: crate::config::EpiloguePlan::default(),
            fused_ane_islands: vec![],
            hidden_size: arch.hidden_size,
            vocab_size: arch.vocab_size,
            sliding_window: arch.sliding_window,
            final_logit_softcapping: None,
            tie_word_embeddings: false,
            rms_norm_eps: arch.rms_norm_eps,
            speculative_config: None,
            generation_regime: Default::default(),
            diffusion_config: Default::default(),
            diffusion_execution_plan: Default::default(),
            kv_cache_mode: Default::default(),
        }
    }

    #[test]
    fn compile_gemma4_schedule() {
        let arch = test_arch();
        let plan = test_execution_plan(&arch);
        let digest = EvidenceDigest("test_gemma4".into());
        let module = compile_model_to_scheduled_module(&plan, &arch, digest);

        assert_eq!(module.regions.len(), 48);
        assert_eq!(module.evaluation_boundaries.len(), 8); // 48 / 6
        for region in &module.regions {
            assert!(
                region.temp_memory_bytes > 0,
                "region {} has zero temp memory",
                region.region_id.0
            );
        }
        assert!(
            module.memory_plan.total_bytes > 0,
            "memory plan should have non-zero total"
        );
        assert!(
            module.memory_plan.peak_bytes > 0,
            "memory plan should have non-zero peak"
        );
        assert!(
            !module.memory_plan.buffer_reuse.is_empty(),
            "should have buffer reuse entries"
        );
        // Must not include O(n²) attention scores or 48× sum of peaks.
        assert!(
            module.memory_plan.total_bytes < 10_000_000_000, // < 10 GB, not 50+
            "total_bytes {} should be realistic (< 10GB)",
            module.memory_plan.total_bytes
        );
        assert!(
            module.memory_plan.total_bytes > 100_000_000, // > 100 MB
            "total_bytes {} should be non-trivial (> 100MB)",
            module.memory_plan.total_bytes
        );
    }

    #[test]
    fn single_layer_peak_estimate() {
        let arch = test_arch();
        let plan = test_execution_plan(&arch);
        let peak = estimate_layer_peak_memory(&plan.layers[0], &arch);
        assert!(peak > 0);
        // For Gemma 4 with seq=8192, peak should be substantial
        assert!(peak > 1_000_000); // > 1MB
    }

    #[test]
    fn memory_plan_has_all_backends() {
        let arch = test_arch();
        let plan = test_execution_plan(&arch);
        let digest = EvidenceDigest("test_backends".into());
        let module = compile_model_to_scheduled_module(&plan, &arch, digest);

        assert!(module.memory_plan.per_backend.contains_key(&BackendId(0)));
        assert!(module.memory_plan.per_backend.contains_key(&BackendId(1)));
        assert!(module.memory_plan.per_backend.contains_key(&BackendId(2)));
    }
}
