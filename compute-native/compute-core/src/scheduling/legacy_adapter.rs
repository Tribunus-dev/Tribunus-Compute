//! Compatibility bridge between [`CurrentActivation`] and the existing
//! executor functions (`executor::run_layer_with_sinks`).
//!
//! The [`LegacyMlxLayerInvocation`] struct extracts MLX-compatible views
//! from the scheduler's activation types and delegates to the stateless
//! executor API with full weight, cache, and RoPE arguments.

use crate::config::operation_route::OperationRoute;
use crate::config::LayerPlan;
use crate::executor;
use crate::executor::SinkState;
use crate::inference::execution_image_state::RopeTables;
use crate::kv_cache::KvCache;
#[cfg(feature = "mlx-backend")]
use crate::profiled_model::LayerWeights;
use crate::scheduling::activation_binding::CurrentActivation;
use mlx_rs::Array;

/// Invocation parameters for the legacy MLX layer runner.
///
/// This adapter extracts MLX-compatible views from the current activation
/// and passes them to `executor::run_layer_with_sinks()`.
pub struct LegacyMlxLayerInvocation<'a> {
    pub hidden: Option<&'a CurrentActivation>,
    pub layer_plan: &'a LayerPlan,
    pub layer_weights: &'a LayerWeights,
    pub kv_cache: Option<&'a mut KvCache>,
    pub sink_state: Option<&'a mut SinkState>,
    pub rope_tables: Option<&'a RopeTables>,
    pub memory_island: Option<&'a crate::heterogeneous::SharedMemoryIsland>,
    pub route: &'a OperationRoute,
}

impl<'a> LegacyMlxLayerInvocation<'a> {
    /// Execute a single layer through the existing MLX path.
    ///
    /// Returns the output hidden state array or an error string.
    pub fn run_layer(&mut self) -> Result<Array, String> {
        let hidden = match self
            .hidden
            .and_then(|a| a.mlx_compatibility_view.as_ref())
        {
            Some(arr) => arr.clone(),
            None => {
                return Err(
                    "no MLX compatibility view available for legacy layer execution"
                        .to_string(),
                );
            }
        };

        let mut kv_cache = self
            .kv_cache
            .take()
            .ok_or_else(|| "KV cache required for legacy layer execution".to_string())?;
        let sink_state = self
            .sink_state
            .take()
            .ok_or_else(|| "sink state required for legacy layer execution".to_string())?;
        let rope_tables = self
            .rope_tables
            .ok_or_else(|| "RoPE tables required for legacy layer execution".to_string())?;

        // RoPE tables are stored as `Arc<[f32]>`; convert to `Array` for the
        // executor API.
        let rope_cos = Array::from_slice(
            &rope_tables.cos,
            &[rope_tables.cos.len() as i32, 1],
        );
        let rope_sin = Array::from_slice(
            &rope_tables.sin,
            &[rope_tables.sin.len() as i32, 1],
        );

        let lw = self.layer_weights;

        // Determine decode vs prefill from the KV cache logical position.
        let is_decode = kv_cache.total_appended > 0;

        // Build a minimal projection context.
        let ctx = crate::projection_identity::ProjectionContext {
            run_id: "legacy-adapter".to_string(),
            phase: if is_decode {
                crate::projection_identity::Phase::Decode
            } else {
                crate::projection_identity::Phase::Prefill
            },
            forward_pass_index: 0,
            token_step: Some(kv_cache.total_appended),
            layer_index: self.layer_plan.layer_index as usize,
            attention_kind: if self.layer_plan.attention_kind == "full_attention" {
                crate::projection_identity::AttentionKind::Full
            } else {
                crate::projection_identity::AttentionKind::Sliding
            },
        };

        // Decompose `Arc<Array>` weight references.
        let total_appended = kv_cache.total_appended;
        let result = executor::run_layer_with_sinks(
            &hidden,
            self.layer_plan,
            self.route,
            self.memory_island,
            &[], // ane_coreml_models — empty; callers configure externally
            &lw.input_layernorm,
            &lw.post_attention_layernorm,
            &lw.q_proj_w,
            &lw.q_proj_s,
            &lw.q_proj_b,
            &lw.k_proj_w,
            &lw.k_proj_s,
            &lw.k_proj_b,
            &lw.v_proj_w,
            &lw.v_proj_s,
            &lw.v_proj_b,
            &lw.o_proj_w,
            &lw.o_proj_s,
            &lw.o_proj_b,
            lw.q_norm.as_deref(),
            lw.k_norm.as_deref(),
            &lw.gate_proj_w,
            &lw.gate_proj_s,
            &lw.gate_proj_b,
            &lw.up_proj_w,
            &lw.up_proj_s,
            &lw.up_proj_b,
            &lw.down_proj_w,
            &lw.down_proj_s,
            &lw.down_proj_b,
            &rope_cos,
            &rope_sin,
            &mut kv_cache,
            total_appended,
            1e-6f32, // rms_norm_eps — sourced from TextArchitecture
            &ctx,
            sink_state,
            is_decode,
        )
        .map_err(|e| format!("run_layer_with_sinks failed: {}", e))?;

        Ok(result)
    }

    /// Check whether an MLX compatibility view is available without
    /// materialization.
    pub fn has_direct_mlx_view(&self) -> bool {
        self.hidden
            .and_then(|a| a.mlx_compatibility_view.as_ref())
            .is_some()
    }

    /// If no direct MLX view exists, report the required materialization
    /// step.
    pub fn required_materialization(&self) -> Option<String> {
        if self.has_direct_mlx_view() {
            None
        } else {
            Some("MlxCompatibilityViewCreation".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::operation_route::OperationRoute;

    #[test]
    fn test_no_activation_errors() {
        let route = OperationRoute::default();
        let plan = LayerPlan {
            layer_index: 0,
            attention_kind: "sliding_attention".to_string(),
            segment_id: "test".to_string(),
            hidden_size: 64,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 16,
            global_head_dim: None,
            n_global_kv_heads: None,
            sliding_window: 128,
            rope_theta: 10000.0,
            partial_rotary_factor: None,
            attention_k_eq_v: false,
            q_norm_enabled: false,
            k_norm_enabled: false,
            q_proj_tensor_id: 0,
            k_proj_tensor_id: 0,
            v_proj_tensor_id: 0,
            o_proj_tensor_id: 0,
            q_norm_tensor_id: None,
            k_norm_tensor_id: None,
            gate_proj_tensor_id: 0,
            up_proj_tensor_id: 0,
            down_proj_tensor_id: 0,
            input_layernorm_tensor_id: 0,
            post_attention_layernorm_tensor_id: 0,
            pre_ffw_layernorm_tensor_id: None,
            post_ffw_layernorm_tensor_id: None,
            layer_scalar_ids: vec![],
            quantization_ids: vec![],
            route: route.clone(),
            fused_operations: vec![],
        };
        // Build a minimal LayerWeights — all fields default to zero-size
        // MLX arrays (caller must replace with real data before execution).
        let zero = Array::from_slice(&[0.0f32], &[1]);
        let weights = LayerWeights {
            input_layernorm: std::sync::Arc::new(zero.clone()),
            post_attention_layernorm: std::sync::Arc::new(zero.clone()),
            q_proj_w: std::sync::Arc::new(zero.clone()),
            q_proj_s: std::sync::Arc::new(zero.clone()),
            q_proj_b: std::sync::Arc::new(zero.clone()),
            k_proj_w: std::sync::Arc::new(zero.clone()),
            k_proj_s: std::sync::Arc::new(zero.clone()),
            k_proj_b: std::sync::Arc::new(zero.clone()),
            v_proj_w: std::sync::Arc::new(zero.clone()),
            v_proj_s: std::sync::Arc::new(zero.clone()),
            v_proj_b: std::sync::Arc::new(zero.clone()),
            o_proj_w: std::sync::Arc::new(zero.clone()),
            o_proj_s: std::sync::Arc::new(zero.clone()),
            o_proj_b: std::sync::Arc::new(zero.clone()),
            gate_proj_w: std::sync::Arc::new(zero.clone()),
            gate_proj_s: std::sync::Arc::new(zero.clone()),
            gate_proj_b: std::sync::Arc::new(zero.clone()),
            up_proj_w: std::sync::Arc::new(zero.clone()),
            up_proj_s: std::sync::Arc::new(zero.clone()),
            up_proj_b: std::sync::Arc::new(zero.clone()),
            down_proj_w: std::sync::Arc::new(zero.clone()),
            down_proj_s: std::sync::Arc::new(zero.clone()),
            down_proj_b: std::sync::Arc::new(zero.clone()),
            q_norm: None,
            k_norm: None,
        };

        let mut invocation = LegacyMlxLayerInvocation {
            hidden: None,
            layer_plan: &plan,
            layer_weights: &weights,
            kv_cache: None,
            sink_state: None,
            rope_tables: None,
            memory_island: None,
            route: &route,
        };

        assert!(invocation.required_materialization().is_some());
        assert!(!invocation.has_direct_mlx_view());
    }
}
