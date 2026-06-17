//! Executor: storage-neutral Gemma 4 decoder execution from compiled plans.
//!
//! Three executors (prologue, layer, epilogue) that consume Plan + resolved
//! MLX Array references. They do not know whether tensors came from copied
//! segments, mapped storage, or test fixtures. The caller is responsible for
//! calling `eval()` on the result before dropping the weight leases.

use crate::config::{EpiloguePlan, LayerPlan, ProloguePlan};
use crate::kv_cache::KvCache;
use crate::config::operation_route::OperationRoute;
use crate::primitives;
use crate::backend::routing::BackendId;
use crate::projection_identity::{dtype_to_storage, ProjectionContext, ProjectionFamily};
use crate::session::SamplerConfig;
use crate::grammar::{GrammarFSM, GrammarTokenizer};
use crate::ane::hot_row_predictor::HotRowPredictor;
use crate::ane::moe_scheduler::{AneMoEScheduler, ExpertWeights};
use crate::ane::weight_row_cache::WeightRowCache;
use mlx_rs::error::Result as MlxResult;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

// ── Attention Sink Reuse ────────────────────────────────────────────────────

/// Precomputed attention sink contributions.
///
/// During prefill, the first N tokens' K/V projections are captured. During
/// decode, attention is computed only over these precomputed sink vectors plus
/// a sliding window of recent tokens, avoiding O(seq_len) KV attention.
#[derive(Debug, Clone)]
pub struct SinkState {
    /// Number of initial tokens treated as permanent sinks.
    pub num_permanent_sinks: u32,
    /// Precomputed K vectors for sink positions, shape [n_sinks, n_kv_heads, head_dim].
    pub sink_k: Option<Array>,
    /// Precomputed V vectors for sink positions, shape [n_sinks, n_kv_heads, head_dim].
    pub sink_v: Option<Array>,
    /// Emergent sink token positions discovered during generation.
    pub emergent_sinks: Vec<u32>,
    /// Base number of recent tokens to attend to (alongside sinks).
    pub window_size: u32,
    /// Adaptive window — grows when attention uncertainty is high.
    pub adaptive_window: u32,
    /// Entropy of the last step's attention distribution.
    pub last_entropy: f32,
}

impl SinkState {
    pub fn new(num_sinks: u32, window_size: u32) -> Self {
        Self {
            num_permanent_sinks: num_sinks,
            sink_k: None,
            sink_v: None,
            emergent_sinks: Vec::new(),
            window_size,
            adaptive_window: window_size,
            last_entropy: 0.0,
        }
    }

    pub fn capture_sinks(
        &mut self,
        cache: &crate::kv_cache::KvCache,
    ) -> MlxResult<()> {
        let n_sinks = self.num_permanent_sinks as usize;
        let (k_cached, v_cached) = cache.read_window().ok_or_else(|| {
            mlx_rs::error::Exception::custom("capture_sinks: cache is empty")
        })?;
        let cached_len = k_cached.shape()[0] as usize;
        let n_sinks_actual = n_sinks.min(cached_len);
        if n_sinks_actual == 0 {
            return Ok(());
        }
        let sink_k = k_cached.index((..n_sinks_actual as i32, .., ..))?;
        let sink_v = v_cached.index((..n_sinks_actual as i32, .., ..))?;
        self.sink_k = Some(sink_k);
        self.sink_v = Some(sink_v);
        Ok(())
    }

    pub fn sink_attention(
        &self,
        q: &Array,
        cache: &crate::kv_cache::KvCache,
        n_rep: u32,
        head_dim: u32,
    ) -> MlxResult<Array> {
        // Q shape is [n_heads, 1, head_dim] — single query token per head.
        let n_heads = q.shape()[0];
        let (k_cached, v_cached) = cache.read_window().ok_or_else(|| {
            mlx_rs::error::Exception::custom("sink_attention: cache is empty")
        })?;
        let cached_len = k_cached.shape()[0] as usize;
        // Window starts after sink positions to avoid overlap.
        let sink_end = self.num_permanent_sinks as usize;
        let window_end = cached_len;
        let window_start = sink_end.max(window_end.saturating_sub(self.adaptive_window as usize));
        debug_assert!(window_start >= sink_end, "window must start after sinks");
        let (window_k, window_v) = if window_start < cached_len {
            let wk = k_cached.index((window_start as i32.., .., ..))?;
            let wv = v_cached.index((window_start as i32.., .., ..))?;
            (wk, wv)
        } else {
            let wk = k_cached.index((0..0, .., ..))?;
            let wv = v_cached.index((0..0, .., ..))?;
            (wk, wv)
        };
        let k_full = match (&self.sink_k, window_k.shape()[0] > 0) {
            (Some(sk), true) => mlx_rs::ops::concatenate(&[sk, &window_k], 0)?,
            (Some(sk), false) => sk.clone(),
            (None, true) => window_k,
            (None, false) => return Err(mlx_rs::error::Exception::custom("sink_attention: no sinks and no window tokens")),
        };
        let v_full = match (&self.sink_v, window_v.shape()[0] > 0) {
            (Some(sv), true) => mlx_rs::ops::concatenate(&[sv, &window_v], 0)?,
            (Some(sv), false) => sv.clone(),
            (None, true) => window_v,
            (None, false) => return Err(mlx_rs::error::Exception::custom("sink_attention: no sinks and no window tokens")),
        };
        let full_len = k_full.shape()[0];
        let k_exp = repeat_kv(&k_full, n_rep)?;
        let v_exp = repeat_kv(&v_full, n_rep)?;
        let kt = k_exp.reshape(&[n_heads, full_len as i32, head_dim as i32])?;
        let kt_t = mlx_rs::ops::transpose_axes(&kt, &[0, 2, 1])?;
        let vt = v_exp.reshape(&[n_heads, full_len as i32, head_dim as i32])?;
        // Q is already [n_heads, 1, head_dim] from the caller.
        let qt = q.reshape(&[n_heads, 1, head_dim as i32])?;
        let scale = (head_dim as f32).sqrt();
        let scores = qt.matmul(&kt_t)?.divide(&Array::from_f32(scale))?;  // [n_heads, 1, full_len]
        let attn = mlx_rs::ops::softmax_axes(&scores, &[-1], None)?;
        let out = attn.matmul(&vt)?.reshape(&[1, -1])?;
        Ok(out)
    }

    pub fn update_adaptive_window(&mut self, attention_weights: &Array) {
        let entropy = if attention_weights.ndim() >= 2 {
            let seq = attention_weights.shape().last().copied().unwrap_or(1);
            let head_0 = attention_weights.index((0..1, 0..seq as i32)).ok();
            if let Some(h) = &head_0 {
                if let Ok(flat) = h.reshape(&[-1]) {
                    if let Ok(slice) = flat.try_as_slice::<f32>() {
                        let mut e = 0.0f32;
                        for &p in slice.iter() {
                            if p > 0.0 { e -= p * p.log(std::f32::consts::E); }
                        }
                        self.last_entropy = e;
                        e
                    } else { self.last_entropy }
                } else { self.last_entropy }
            } else { self.last_entropy }
        } else { self.last_entropy };
        let base = (self.adaptive_window as f32).ln().max(1.0);
        let threshold = base * 0.8;
        if entropy > threshold && self.adaptive_window < self.window_size * 4 {
            self.adaptive_window = (self.adaptive_window * 3 / 2).min(self.window_size * 4);
        } else if entropy < threshold * 0.3 && self.adaptive_window > self.window_size {
            self.adaptive_window = (self.adaptive_window * 2 / 3).max(self.window_size);
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Hidden scale for the prologue embedding: sqrt(hidden_size).
pub fn prologue_hidden_scale(plan: &ProloguePlan) -> f32 {
    (plan.embedding_shape[1] as f32).sqrt()
}

/// Determine which backend handles a given layer based on attention kind.
///
/// - sliding_attention → Core ML / ANE (backend 2)
/// - full_attention → MLX / GPU (backend 0)
pub fn resolve_attention_backend(layer_plan: &LayerPlan) -> BackendId {
    if layer_plan.attention_kind == "full_attention" {
        BackendId(0) // MLX (GPU)
    } else {
        BackendId(2) // Core ML (ANE)
    }
}

// ── Prologue ───────────────────────────────────────────────────────────────

/// Embedding lookup: token_ids → initial hidden state.
/// Uses dequantized embedding weights (Gather → dequantize → scale).
pub fn run_prologue(
    token_ids: &Array,
    emb_weight: &Array,
    emb_scales: &Array,
    emb_biases: &Array,
    _plan: &ProloguePlan,
    hidden_scale: f32,
) -> MlxResult<Array> {
    // Shape contract: token_ids rank 1 (flat tokens) or 2 (batchless [1, tokens]).
    debug_assert!(
        token_ids.ndim() == 1 || token_ids.ndim() == 2,
        "token_ids must be rank 1 or 2, got rank {}",
        token_ids.ndim()
    );
    // Flatten to 1D if a singleton batch dim is present.
    let flat_ids = if token_ids.ndim() == 2 {
        token_ids.reshape(&[-1])?
    } else {
        token_ids.clone()
    };

    let emb =
        primitives::quantized_embedding_lookup(&flat_ids, emb_weight, emb_scales, emb_biases)?;
    // Hidden state must be rank 2 (no batch dim): [tokens, hidden_size]
    debug_assert_eq!(
        emb.ndim(),
        2,
        "hidden state must be rank 2 (batchless), got rank {}",
        emb.ndim()
    );
    emb.multiply(&Array::from_f32(hidden_scale))
}

/// Token offset for vision tokens in the embedding space.
/// Token IDs >= this value are considered vision feature tokens.
pub const VISION_TOKEN_OFFSET: u32 = 250_000;

/// Apply vision feature embeddings to the hidden state after prologue.
///
/// Vision token IDs (>= VISION_TOKEN_OFFSET) in `token_ids` are replaced
/// with pre-computed vision features from the vision encoder.  The modified
/// hidden state is returned with vision features spliced in at the correct
/// token positions.
///
/// # Arguments
/// * `hidden` — hidden state from `run_prologue`, shape `[seq_len, hidden_size]`
/// * `token_ids` — the input token IDs (flat, 1D)
/// * `vision_features` — features from the vision encoder, shape `[num_patches, projection_dim]`
///
/// # Returns
/// Updated hidden state with vision features at vision token positions.
pub fn apply_vision_embeddings(
    hidden: &Array,
    token_ids: &Array,
    vision_features: &Array,
) -> MlxResult<Array> {
    // Get the flat token IDs.
    let flat_ids = if token_ids.ndim() == 2 {
        token_ids.reshape(&[-1])?
    } else {
        token_ids.clone()
    };

    let ids: Vec<u32> = flat_ids
        .try_as_slice::<u32>()
        .map_err(|e| mlx_rs::error::Exception::format(&format!(
            "apply_vision_embeddings: token_ids as_slice: {:?}", e
        )))?
        .to_vec();

    let seq_len = ids.len();
    let hidden_size = hidden.shape().get(1).copied().unwrap_or(1) as usize;

    // Count how many vision tokens there are.
    let vision_count = ids.iter().filter(|&&id| id >= VISION_TOKEN_OFFSET).count();
    if vision_count == 0 {
        return Ok(hidden.clone());
}

    // Get hidden data as mutable slice.
    let mut h_data: Vec<f32> = hidden
        .as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::format(&format!(
            "apply_vision_embeddings: hidden as_slice: {:?}", e
        )))?
        .to_vec();

    // Get vision features data.
    let vf_data: Vec<f32> = vision_features
        .as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::format(&format!(
            "apply_vision_embeddings: vision_features as_slice: {:?}", e
        )))?
        .to_vec();

    let vf_dim = vision_features.shape().get(1).copied().unwrap_or(1) as usize;
    let num_patches = vision_features.shape().get(0).copied().unwrap_or(1) as usize;

    // Iterate through token IDs and replace vision token positions.
    let mut vf_offset = 0;
    for pos in 0..seq_len {
        if ids[pos] >= VISION_TOKEN_OFFSET {
            // This position should use a vision feature.
            let vf_idx = vf_offset.min(num_patches - 1);
            let start = pos * hidden_size;
            let end = (start + hidden_size.min(vf_dim)).min(h_data.len());
            for k in start..end {
                let vf_src = vf_idx * vf_dim + (k - pos * hidden_size);
                h_data[k] = if vf_src < vf_data.len() {
                    vf_data[vf_src]
                } else {
                    0.0
                };
}
            vf_offset += 1;
}
}

    let dims: Vec<i32> = vec![seq_len as i32, hidden_size as i32];
    Ok(Array::from_slice(&h_data, &dims))
}

// ── Decoder Layer ─────────────────────────────────────────────────────────

/// Execute one decoder layer from a compiled LayerPlan and resolved tensors.
///
/// The plan determines whether sliding or global attention is used — no
/// branching on layer index. All weights are passed as resolved MLX Arrays.
/// The caller MUST eval the result before dropping weight leases.
pub fn run_layer(
    hidden: &Array,
    plan: &LayerPlan,
    route: &OperationRoute,
    island: Option<&crate::heterogeneous::SharedMemoryIsland>,
    ane_coreml_models: &[Option<std::sync::Arc<crate::coreml_bridge::CoreMlModel>>],
    // Attention norm weights
    attn_norm: &Array,
    ffn_norm: &Array,
    // QKV projections (weight, scales, biases triplets)
    qw: &Array,
    qs: &Array,
    qb: &Array,
    kw: &Array,
    ks: &Array,
    kb: &Array,
    vw: &Array,
    vs: &Array,
    vb: &Array,
    ow: &Array,
    os: &Array,
    ob: &Array,
    // Q/K norm weights
    q_norm_weight: Option<&Array>,
    k_norm_weight: Option<&Array>,
    // MLP projections
    gw: &Array,
    gs: &Array,
    gb: &Array,
    uw: &Array,
    us: &Array,
    ub: &Array,
    dw: &Array,
    ds: &Array,
    db: &Array,
    // RoPE tables
    rope_cos: &Array,
    rope_sin: &Array,
    // KV cache for this layer
    cache: &mut KvCache,
    kv_offset: u32,
    rms_norm_eps: f32,
    ctx: &ProjectionContext,
) -> MlxResult<Array> {
    // Shape contract: hidden state is batchless [tokens, hidden_size].
    debug_assert_eq!(
        hidden.ndim(),
        2,
        "hidden state must be rank 2 (batchless), got rank {}",
        hidden.ndim()
    );
    let _n_tokens = hidden.shape()[0];

    // --- Attention norm ---
    let residual = hidden;
    let normed = crate::heterogeneous::dispatch_rms_norm(hidden, attn_norm, rms_norm_eps, route, island)?;

    // --- Attention ---
    let attn_out = match plan.attention_kind.as_str() {
        "sliding_attention" => sliding_attention_layer(
            &normed,
            plan,
            route,
            ane_coreml_models,
            plan.layer_index as usize,
            island,
            qw,
            qs,
            qb,
            kw,
            ks,
            kb,
            vw,
            vs,
            vb,
            ow,
            os,
            ob,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            kv_offset,
            cache,
            ctx,
        )?,
        "full_attention" => full_attention_layer(
            &normed,
            plan,
            route,
            ane_coreml_models,
            plan.layer_index as usize,
            island,
            qw,
            qs,
            qb,
            kw,
            ks,
            kb,
            vw,
            vs,
            vb,
            ow,
            os,
            ob,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            kv_offset,
            cache,
            ctx,
        )?,
        other => {
            return Err(mlx_rs::error::Exception::custom(format!(
                "unknown attention_kind: {}",
                other
            )));
        }
    };

    let hidden = residual.add(&attn_out)?;
    let hidden = crate::heterogeneous::dispatch_add(residual, &attn_out, route, island)?;

    // --- FFN norm ---
    let residual = &hidden;
    let normed = primitives::rms_norm(&hidden, ffn_norm, rms_norm_eps)?;

    // --- SwiGLU MLP ---
    let gate = qmatmul_attributed(
        &normed,
        gw,
        gs,
        gb,
        true,
        64,
        8,
        ctx,
        ProjectionFamily::GateProj,
        4,
    )?;
    let up = qmatmul_attributed(
        &normed,
        uw,
        us,
        ub,
        true,
        64,
        8,
        ctx,
        ProjectionFamily::UpProj,
        5,
    )?;
    let gated = mlx_rs::nn::silu(&gate)?.multiply(&up)?;
    let gated = crate::heterogeneous::dispatch_multiply(&mlx_rs::nn::silu(&gate)?, &up, route, island)?;
    let ffn_out = qmatmul_attributed(
        &gated,
        dw,
        ds,
        db,
        true,
        64,
        8,
        ctx,
        ProjectionFamily::DownProj,
        6,
    )?;

    let result = crate::heterogeneous::dispatch_add(residual, &ffn_out, route, island)?;
    result.eval()?;
    Ok(result)
}

/// Execute one decoder layer with attention sink reuse.
///
/// During prefill: full attention as normal, then capture sinks for future decode.
/// During decode: use sink attention over precomputed sink K/V + adaptive window
/// instead of attending to the full KV cache.
pub fn run_layer_with_sinks(
    hidden: &Array,
    plan: &LayerPlan,
    route: &OperationRoute,
    island: Option<&crate::heterogeneous::SharedMemoryIsland>,
    ane_coreml_models: &[Option<std::sync::Arc<crate::coreml_bridge::CoreMlModel>>],
    attn_norm: &Array,
    ffn_norm: &Array,
    qw: &Array, qs: &Array, qb: &Array,
    kw: &Array, ks: &Array, kb: &Array,
    vw: &Array, vs: &Array, vb: &Array,
    ow: &Array, os: &Array, ob: &Array,
    q_norm_weight: Option<&Array>,
    k_norm_weight: Option<&Array>,
    gw: &Array, gs: &Array, gb: &Array,
    uw: &Array, us: &Array, ub: &Array,
    dw: &Array, ds: &Array, db: &Array,
    rope_cos: &Array,
    rope_sin: &Array,
    cache: &mut KvCache,
    kv_offset: u32,
    rms_norm_eps: f32,
    ctx: &ProjectionContext,
    sink_state: &mut SinkState,
    is_decode: bool,
) -> MlxResult<Array> {
    if is_decode {
        // --- Decode path: compute Q, then attend only to sinks + window ---
        let n_heads = plan.n_heads;
        let head_dim = if plan.attention_kind == "full_attention" {
            plan.global_head_dim.unwrap_or(plan.head_dim)
        } else {
            plan.head_dim
        };
        let n_kv_heads = if plan.attention_kind == "full_attention" {
            plan.n_global_kv_heads.unwrap_or(plan.n_kv_heads)
        } else {
            plan.n_kv_heads
        };
        let n_rep = n_heads / n_kv_heads;

        let residual = hidden;
        let normed = crate::heterogeneous::dispatch_rms_norm(hidden, attn_norm, rms_norm_eps, route, island)?;

        // Compute Q, K, V projections (same as sliding_attention_layer / full_attention_layer).
        let q = qmatmul_attributed(&normed, qw, qs, qb, true, 64, 8, ctx, ProjectionFamily::QProj, 0)?
            .reshape(&[1, n_heads as i32, head_dim as i32])?;
        let k = qmatmul_attributed(&normed, kw, ks, kb, true, 64, 8, ctx, ProjectionFamily::KProj, 1)?
            .reshape(&[1, n_kv_heads as i32, head_dim as i32])?;
        let v = if plan.attention_k_eq_v {
            k.clone()
        } else {
            qmatmul_attributed(&normed, vw, vs, vb, true, 64, 8, ctx, ProjectionFamily::VProj, 2)?
                .reshape(&[1, n_kv_heads as i32, head_dim as i32])?
        };

        // Q norm and K norm.
        let q = if let Some(wn) = q_norm_weight {
            primitives::rms_norm(&q.reshape(&[-1, head_dim as i32])?, wn, 1e-6)?
        } else {
            primitives::rms_norm_scale_free(&q.reshape(&[-1, head_dim as i32])?, 1e-6)?
        }.reshape(&[1, n_heads as i32, head_dim as i32])?;

        let k = if let Some(wn) = k_norm_weight {
            primitives::rms_norm(&k.reshape(&[-1, head_dim as i32])?, wn, 1e-6)?
        } else {
            primitives::rms_norm_scale_free(&k.reshape(&[-1, head_dim as i32])?, 1e-6)?
        }.reshape(&[1, n_kv_heads as i32, head_dim as i32])?;

        // Apply RoPE.
        let q4d = q.reshape(&[1, n_heads as i32, 1, head_dim as i32])?;
        let q4d = primitives::rope_apply(
            &q4d, rope_cos, rope_sin, kv_offset, plan.partial_rotary_factor,
        )?;
        // Reshape to [n_heads, 1, head_dim] for batched matmul over heads.
        let q_rope = q4d.reshape(&[-1, 1, head_dim as i32])?;

        let k4d = k.reshape(&[1, n_kv_heads as i32, 1, head_dim as i32])?;
        let k4d = primitives::rope_apply(
            &k4d, rope_cos, rope_sin, kv_offset, plan.partial_rotary_factor,
        )?;
        let k_rope = k4d.reshape(&[1, n_kv_heads as i32, head_dim as i32])?;

        q_rope.eval()?;
        k_rope.eval()?;
        v.eval()?;

        // Store K, V in cache for future window reads.
        cache.append(k_rope, v)?;

        // Use sink attention: only attend to precomputed sinks + window.
        let attn_out = sink_state.sink_attention(&q_rope, cache, n_rep, head_dim)?;

        let attn_proj = qmatmul_attributed(
            &attn_out, ow, os, ob, true, 64, 8, ctx, ProjectionFamily::OProj, 3,
        )?.reshape(&[1, -1])?;

        // Residual + FFN (same as run_layer).
        let hidden_after_attn = crate::heterogeneous::dispatch_add(residual, &attn_proj, route, island)?;

        let residual_ffn = &hidden_after_attn;
        let normed_ffn = primitives::rms_norm(&hidden_after_attn, ffn_norm, rms_norm_eps)?;

        let gate = qmatmul_attributed(&normed_ffn, gw, gs, gb, true, 64, 8, ctx, ProjectionFamily::GateProj, 4)?;
        let up = qmatmul_attributed(&normed_ffn, uw, us, ub, true, 64, 8, ctx, ProjectionFamily::UpProj, 5)?;
        let gated = crate::heterogeneous::dispatch_multiply(&mlx_rs::nn::silu(&gate)?, &up, route, island)?;
        let ffn_out = qmatmul_attributed(&gated, dw, ds, db, true, 64, 8, ctx, ProjectionFamily::DownProj, 6)?;

        let result = crate::heterogeneous::dispatch_add(residual_ffn, &ffn_out, route, island)?;
        result.eval()?;
        Ok(result)
    } else {
        // --- Prefill path: full attention, then capture sinks ---
        let result = run_layer(
            hidden, plan, route, island, ane_coreml_models,
            attn_norm, ffn_norm,
            qw, qs, qb, kw, ks, kb, vw, vs, vb, ow, os, ob,
            q_norm_weight, k_norm_weight,
            gw, gs, gb, uw, us, ub, dw, ds, db,
            rope_cos, rope_sin,
            cache, kv_offset, rms_norm_eps, ctx,
        )?;
        // Capture sink K/V from cache for future decode steps.
        if let Err(e) = sink_state.capture_sinks(cache) {
            eprintln!("[sink] capture_sinks layer {}: {}", plan.layer_index, e);
        }
        Ok(result)
    }
}

// ── Attention implementations ──────────────────────────────────────────────

fn qmatmul(x: &Array, w: &Array, s: &Array, b: &Array) -> MlxResult<Array> {
    let group_size = if s.shape().len() >= 1 {
        (w.shape()[1] as i32 * 4) / s.shape()[s.shape().len() - 1]
    } else {
        64
    };

    // OPT-0006-T2 diagnostic: first-call stride/contiguity probe.
    // Answers: do external (mmap-backed) arrays trigger hidden copies?
    use std::sync::atomic::{AtomicBool, Ordering};
    static T2_PROBE: AtomicBool = AtomicBool::new(true);
    if T2_PROBE.swap(false, Ordering::Relaxed) {
        let ws = w.shape();
        w.eval()?;
        let t_ext_start = std::time::Instant::now();
        let r1 = mlx_rs::ops::quantized_matmul(x, w, s, b, true, group_size, 8)?;
        let _ = r1.eval()?;
        let t_ext = t_ext_start.elapsed();
        let ws_str: Vec<String> = ws.iter().map(|d| d.to_string()).collect();
        // Try contiguous copy comparison — may fail for external arrays
        // with dtype mismatch; fall back to external-only timing.
        let w_read: Option<Vec<u8>> = w.try_as_slice::<u8>().ok().map(|s| s.to_vec());
        if let Some(w_vec) = w_read {
            let wc = Array::from_slice(&w_vec, &ws);
            let s_vec: Vec<f32> = s
                .try_as_slice::<f32>()
                .map_err(|e| mlx_rs::error::Exception::custom(format!("s as_slice: {:?}", e)))?
                .to_vec();
            let sc = Array::from_slice(&s_vec, &s.shape());
            let b_vec: Vec<f32> = b
                .try_as_slice::<f32>()
                .map_err(|e| mlx_rs::error::Exception::custom(format!("b as_slice: {:?}", e)))?
                .to_vec();
            let bc = Array::from_slice(&b_vec, &b.shape());
            let t_copy_start = std::time::Instant::now();
            let r2 = mlx_rs::ops::quantized_matmul(x, &wc, &sc, &bc, true, group_size, 8)?;
            let _ = r2.eval()?;
            let t_copy = t_copy_start.elapsed();
            let wss: Vec<String> = ws.iter().map(|d| d.to_string()).collect();
            eprintln!(
                "[OPT-0006-T2] first-qmatmul shape=[{}] strides=[{}] ext={:.1}ms copy={:.1}ms ratio={:.2}x",
                ws_str.join(","), wss.join(","),
                t_ext.as_secs_f64() * 1000.0,
                t_copy.as_secs_f64() * 1000.0,
                t_copy.as_secs_f64() / t_ext.as_secs_f64(),
            );
        } else {
            eprintln!(
                "[OPT-0006-T2] first-qmatmul shape=[{}] ext={:.1}ms (copy comparison unavailable — external array dtype mismatch)",
                ws_str.join(","),
                t_ext.as_secs_f64() * 1000.0,
            );
        }
        return Ok(r1);
    }

    mlx_rs::ops::quantized_matmul(x, w, s, b, true, group_size, 8)
}

// ── Projection attribution wrapper ────────────────────────────────────

/// Wrapper around qmatmul that conditionally emits a projection attribution
/// event when `TRIBUNUS_PROJECTION_ATTRIBUTION=1`.
///
/// When the env var is unset (the common case), this function degenerates
/// to a direct call to `qmatmul` — zero overhead beyond an AtomicBool load.
/// No eval, synchronization, or host readback occurs inside the timer.
fn qmatmul_attributed(
    x: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    _transpose: bool,
    _group_size: i32,
    _bits: i32,
    ctx: &ProjectionContext,
    family: ProjectionFamily,
    invocation: usize,
) -> MlxResult<Array> {
    // One-time gate check: cache in a static AtomicBool so the fast path
    // is a single relaxed load (no env var string allocation).
    static GATE_INIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static GATE_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    if !GATE_INIT.load(std::sync::atomic::Ordering::Relaxed) {
        let enabled = std::env::var("TRIBUNUS_PROJECTION_ATTRIBUTION")
            .ok()
            .as_deref()
            == Some("1");
        GATE_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
        GATE_INIT.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    if !GATE_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return qmatmul(x, w, s, b);
    }

    // ---- Instrumented path ----
    let start = std::time::Instant::now();
    let result = qmatmul(x, w, s, b)?;
    let delta_ns = start.elapsed().as_nanos() as u64;

    let input_shape = x.shape();
    let w_shape = w.shape();

    // Compute group_size from scales shape (same logic as qmatmul).
    let gs = if s.shape().len() >= 1 {
        (w.shape()[1] as i32 * 4) / s.shape()[s.shape().len() - 1]
    } else {
        64
    };

    let runtime_dtype_str = format!("{:?}", w.dtype());
    let storage_dtype_str = dtype_to_storage(&w.dtype());

    // Build weight_physical = weight_logical for now; the physical shape
    // is identical because we operate on the canonical MLX representation.
    let w_d0 = w_shape.first().copied().unwrap_or(0);
    let w_d1 = w_shape.get(1).copied().unwrap_or(0);

    let token_step_str = match ctx.token_step {
        Some(ts) => format!("{}", ts),
        None => String::new(),
    };

    eprintln!(
        "[proj] run_id={} phase={} forward_pass={} token_step={} layer={} kind={} family={} invocation={} graph_build_ns={} input=[{},{}] weight_logical=[{},{}] weight_physical=[{},{}] storage_dtype={} runtime_dtype={} group_size={} bits={} transpose={}",
        ctx.run_id,
        ctx.phase.as_str(),
        ctx.forward_pass_index,
        token_step_str,
        ctx.layer_index,
        ctx.attention_kind.as_str(),
        family.as_str(),
        invocation,
        delta_ns,
        input_shape.first().copied().unwrap_or(0),
        input_shape.get(1).copied().unwrap_or(0),
        w_d0,
        w_d1,
        w_d0,
        w_d1,
        storage_dtype_str,
        runtime_dtype_str,
        gs,
        8,   // bits — always 8 for our quantized matmul
        true, // transpose — always true for qmatmul
    );

    Ok(result)
}

fn sliding_attention_layer(
    x: &Array,
    plan: &LayerPlan,
    route: &OperationRoute,
    ane_coreml_models: &[Option<std::sync::Arc<crate::coreml_bridge::CoreMlModel>>],
    layer_idx: usize,
    island: Option<&crate::heterogeneous::SharedMemoryIsland>,
    qw: &Array,
    qs: &Array,
    qb: &Array,
    kw: &Array,
    ks: &Array,
    kb: &Array,
    vw: &Array,
    vs: &Array,
    vb: &Array,
    ow: &Array,
    os: &Array,
    ob: &Array,
    q_norm_weight: Option<&Array>,
    k_norm_weight: Option<&Array>,
    rope_cos: &Array,
    rope_sin: &Array,
    kv_offset: u32,
    cache: &mut KvCache,
    ctx: &ProjectionContext,
) -> MlxResult<Array> {
    let n_tokens = x.shape()[0];
    let n_heads = plan.n_heads;
    let n_kv_heads = plan.n_kv_heads;
    let head_dim = plan.head_dim;
    let n_rep = n_heads / n_kv_heads;

    let q = qmatmul_attributed(x, qw, qs, qb, true, 64, 8, ctx, ProjectionFamily::QProj, 0)?
        .reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;
    let k = qmatmul_attributed(x, kw, ks, kb, true, 64, 8, ctx, ProjectionFamily::KProj, 1)?
        .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;
    let v = qmatmul_attributed(x, vw, vs, vb, true, 64, 8, ctx, ProjectionFamily::VProj, 2)?
        .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    let q = if let Some(wn) = q_norm_weight {
        primitives::rms_norm(&q.reshape(&[-1, head_dim as i32])?, wn, 1e-6)?
    } else {
        primitives::rms_norm_scale_free(&q.reshape(&[-1, head_dim as i32])?, 1e-6)?
    }
    .reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;

    let k = if let Some(wn) = k_norm_weight {
        primitives::rms_norm(&k.reshape(&[-1, head_dim as i32])?, wn, 1e-6)?
    } else {
        primitives::rms_norm_scale_free(&k.reshape(&[-1, head_dim as i32])?, 1e-6)?
    }
    .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    let q4d = q.reshape(&[1, n_heads as i32, n_tokens, head_dim as i32])?;
    let q4d = primitives::rope_apply(
        &q4d,
        rope_cos,
        rope_sin,
        kv_offset,
        plan.partial_rotary_factor,
    )?;
    let q = q4d.reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;

    let k4d = k.reshape(&[1, n_kv_heads as i32, n_tokens, head_dim as i32])?;
    let k4d = primitives::rope_apply(
        &k4d,
        rope_cos,
        rope_sin,
        kv_offset,
        plan.partial_rotary_factor,
    )?;
    let k = k4d.reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    // Materialize the current token batch before appending so the cache holds
    // stable KV tensors rather than a larger lazy graph.
    // Per-step commit: construct candidate K/V updates, complete layer evaluation,
    // then commit the cache position. A failed layer must not partially advance the cache.
    q.eval()?;
    k.eval()?;
    v.eval()?;

    // ANE dispatch: if this layer is routed to ANE, send Q/K/V to CoreML
    if route.attention == crate::heterogeneous::ANE {
        return crate::heterogeneous::dispatch_attention_coreml(
            &q, &k, &v,
            ane_coreml_models,
            layer_idx,
            island.ok_or_else(|| {
                mlx_rs::error::Exception::from(
                    "ANE sliding attention requires SharedMemoryIsland",
                )
            })?,
        );
    }

    cache.append(k, v)?;
    let (k_cached, v_cached) = cache
        .read_window()
        .expect("cache must be non-empty after append");
    let cached_seq = k_cached.shape()[0];

    // GQA repeat KV
    let k_exp = repeat_kv(&k_cached, n_rep)?;
    let v_exp = repeat_kv(&v_cached, n_rep)?;

    // Attention scores: Q [heads, n_tokens, hd] @ K^T [heads, hd, cached_seq]
    let qt = q.reshape(&[n_heads as i32, n_tokens, head_dim as i32])?;
    let kt = k_exp.reshape(&[n_heads as i32, cached_seq as i32, head_dim as i32])?;
    let vt = v_exp.reshape(&[n_heads as i32, cached_seq as i32, head_dim as i32])?;

    let scale = (head_dim as f32).sqrt();
    let scores = qt
        .matmul(&mlx_rs::ops::transpose_axes(&kt, &[0, 2, 1])?)?
        .divide(&Array::from_f32(scale))?;

    // Causal + sliding mask sized [n_tokens, cached_seq].
    let mask = causal_mask(n_tokens as u32, cached_seq as u32, kv_offset)?.add(&sliding_mask(
        n_tokens as u32,
        cached_seq as u32,
        plan.sliding_window,
        kv_offset,
    )?)?;
    eprintln!(
        "[mask] cached_seq={} n_tokens={} n_heads={}",
        cached_seq, n_tokens, n_heads
    );
    let scores = scores.add(&mask)?;

    let attn = mlx_rs::ops::softmax_axes(&scores, &[-1], None)?;
    let out = attn
        .matmul(&vt)?
        .reshape(&[n_tokens, (n_heads * head_dim) as i32])?;
    qmatmul_attributed(
        &out,
        ow,
        os,
        ob,
        true,
        64,
        8,
        ctx,
        ProjectionFamily::OProj,
        3,
    )?
    .reshape(&[n_tokens, -1])
}

fn full_attention_layer(
    x: &Array,
    plan: &LayerPlan,
    route: &OperationRoute,
    ane_coreml_models: &[Option<std::sync::Arc<crate::coreml_bridge::CoreMlModel>>],
    layer_idx: usize,
    island: Option<&crate::heterogeneous::SharedMemoryIsland>,
    qw: &Array,
    qs: &Array,
    qb: &Array,
    kw: &Array,
    ks: &Array,
    kb: &Array,
    vw: &Array,
    vs: &Array,
    vb: &Array,
    ow: &Array,
    os: &Array,
    ob: &Array,
    q_norm_weight: Option<&Array>,
    k_norm_weight: Option<&Array>,
    rope_cos: &Array,
    rope_sin: &Array,
    kv_offset: u32,
    cache: &mut KvCache,
    ctx: &ProjectionContext,
) -> MlxResult<Array> {
    let n_tokens = x.shape()[0];
    let n_heads = plan.n_heads;
    let head_dim = plan.global_head_dim.unwrap_or(plan.head_dim);
    let n_kv_heads = plan.n_global_kv_heads.unwrap_or(plan.n_kv_heads);
    let n_rep = n_heads / n_kv_heads;

    let q = qmatmul_attributed(x, qw, qs, qb, true, 64, 8, ctx, ProjectionFamily::QProj, 0)?
        .reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;
    let k = qmatmul_attributed(x, kw, ks, kb, true, 64, 8, ctx, ProjectionFamily::KProj, 1)?
        .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    // Plan-driven V semantics: when attention_k_eq_v is true, K and V share
    // weights so we alias K as V rather than computing a separate projection.
    let v: Array = if plan.attention_k_eq_v {
        k.clone()
    } else {
        qmatmul_attributed(x, vw, vs, vb, true, 64, 8, ctx, ProjectionFamily::VProj, 2)?
            .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?
    };

    let q = if let Some(wn) = q_norm_weight {
        primitives::rms_norm(&q.reshape(&[-1, head_dim as i32])?, wn, 1e-6)?
    } else {
        primitives::rms_norm_scale_free(&q.reshape(&[-1, head_dim as i32])?, 1e-6)?
    }
    .reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;

    let k = if let Some(wn) = k_norm_weight {
        primitives::rms_norm(&k.reshape(&[-1, head_dim as i32])?, wn, 1e-6)?
    } else {
        primitives::rms_norm_scale_free(&k.reshape(&[-1, head_dim as i32])?, 1e-6)?
    }
    .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    let q4d = q.reshape(&[1, n_heads as i32, n_tokens, head_dim as i32])?;
    let q4d = primitives::rope_apply(
        &q4d,
        rope_cos,
        rope_sin,
        kv_offset,
        plan.partial_rotary_factor,
    )?;
    let q = q4d.reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;

    let k4d = k.reshape(&[1, n_kv_heads as i32, n_tokens, head_dim as i32])?;
    let k4d = primitives::rope_apply(
        &k4d,
        rope_cos,
        rope_sin,
        kv_offset,
        plan.partial_rotary_factor,
    )?;
    let k = k4d.reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    // Per-step commit: construct candidate K/V updates, complete layer evaluation,
    // then commit the cache position. A failed layer must not partially advance the cache.
    q.eval()?;
    k.eval()?;
    v.eval()?;
    // ANE dispatch: if this layer is routed to ANE, send Q/K/V to CoreML
    if route.attention == crate::heterogeneous::ANE {
        return crate::heterogeneous::dispatch_attention_coreml(
            &q, &k, &v,
            ane_coreml_models,
            layer_idx,
            island.ok_or_else(|| {
                mlx_rs::error::Exception::from(
                    "ANE full attention requires SharedMemoryIsland",
                )
            })?,
        );
    }

    cache.append(k, v)?;
    let (k_cached, v_cached) = cache
        .read_window()
        .expect("cache must be non-empty after append");
    let cached_seq = k_cached.shape()[0];

    // GQA repeat KV
    let k_exp = repeat_kv(&k_cached, n_rep)?;
    let v_exp = repeat_kv(&v_cached, n_rep)?;

    let qt = q.reshape(&[n_heads as i32, n_tokens, head_dim as i32])?;
    let kt = k_exp.reshape(&[n_heads as i32, cached_seq as i32, head_dim as i32])?;
    let vt = v_exp.reshape(&[n_heads as i32, cached_seq as i32, head_dim as i32])?;

    let scale = (head_dim as f32).sqrt();
    let scores = qt
        .matmul(&mlx_rs::ops::transpose_axes(&kt, &[0, 2, 1])?)?
        .divide(&Array::from_f32(scale))?;

    // Full causal mask sized [n_tokens, cached_seq].
    let mask = causal_mask(n_tokens as u32, cached_seq as u32, kv_offset)?;
    let scores = scores.add(&mask)?;

    let attn = mlx_rs::ops::softmax_axes(&scores, &[-1], None)?;
    let out = attn
        .matmul(&vt)?
        .reshape(&[n_tokens, (n_heads * head_dim) as i32])?;
    qmatmul_attributed(
        &out,
        ow,
        os,
        ob,
        true,
        64,
        8,
        ctx,
        ProjectionFamily::OProj,
        3,
    )?
    .reshape(&[n_tokens, -1])
}

/// Fused MoE forward pass: route each token to its top-K experts,
/// compute only those experts' outputs via gated FFN, and combine
/// results weighted by the router probabilities.
///
/// # Arguments
/// * `hidden` — input hidden states, shape `[seq_len, hidden_size]`
/// * `gate_proj` — per-expert gate projection weights, length=num_experts
/// * `up_proj` — per-expert up projection weights, length=num_experts
/// * `down_proj` — per-expert down projection weights, length=num_experts
/// * `router` — learned routing weight matrix, shape [hidden_size, num_experts]
/// * `top_k` — number of experts to activate per token
///
/// # Returns
/// Output array of the same shape as `hidden`: `[seq_len, hidden_size]`
///
/// Only `top_k` experts are computed per token (not all N), preserving the
/// computational benefit of sparse MoE routing.
pub fn moe_forward(
    hidden: &Array,
    gate_proj: &[Array],
    up_proj: &[Array],
    down_proj: &[Array],
    router: &Array,
    top_k: u32,
) -> Result<Array, String> {
    let num_experts = gate_proj.len() as usize;
    if num_experts == 0 {
        return Err("moe_forward: zero experts provided".into());
    }
    if top_k == 0 || (top_k as usize) > num_experts {
        return Err(format!(
            "moe_forward: top_k={} must be in 1..={}",
            top_k, num_experts
        ));
    }
    let top_k_usize = top_k as usize;

    let seq_len = hidden.shape()[0];
    let hidden_size = hidden.shape()[1];

    // 1. Router: hidden @ router -> softmax -> routing probs
    let router_logits = hidden
        .matmul(router)
        .map_err(|e| format!("moe router matmul: {:?}", e))?;
    let routing_probs = mlx_rs::ops::softmax_axes(&router_logits, &[-1], None)
        .map_err(|e| format!("moe softmax: {:?}", e))?;

    // 2. Top-K indices: argsort descending (negate), take first K
    let neg_probs = routing_probs
        .neg()
    ;
    let sorted_idx = ops::argsort_axis(&neg_probs, -1)
        .map_err(|e| format!("moe argsort: {:?}", e))?;
    let top_k_indices = sorted_idx
        .index((.., ..(top_k as i32)))
        .map_err(|e| format!("moe slice top-k indices: {:?}", e))?;

    // 3. Gather top-K routing weights via take_along_axis
    let top_k_weights = ops::indexing::take_along_axis(
        &routing_probs,
        &top_k_indices,
        Some(-1),
    )
    .map_err(|e| format!("moe gather weights: {:?}", e))?;

    // Eval to read indices/weights on host for expert grouping
    top_k_indices
        .eval()
        .map_err(|e| format!("moe indices eval: {:?}", e))?;
    top_k_weights
        .eval()
        .map_err(|e| format!("moe weights eval: {:?}", e))?;

    let flat_indices: &[u32] = top_k_indices
        .try_as_slice::<u32>()
        .map_err(|e| format!("moe indices slice: {:?}", e))?;
    let flat_weights: &[f32] = top_k_weights
        .try_as_slice::<f32>()
        .map_err(|e| format!("moe weights slice: {:?}", e))?;

    // 4. Group tokens by expert
    let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
    for t in 0..seq_len {
        for p in 0..top_k_usize {
            let e = flat_indices[t * top_k_usize + p] as usize;
            let w = flat_weights[t * top_k_usize + p];
            if e < num_experts && w > 0.0f32 {
                expert_tokens[e].push((t, w));
            }
        }
    }

    // 5. Per-expert FFN computation and weighted accumulation
    let seq_len_i32 = seq_len as i32;
    let hidden_size_i32 = hidden_size as i32;
    let mut output = Array::zeros::<f32>(&[seq_len_i32, hidden_size_i32])
        .map_err(|e| format!("moe output zeros: {:?}", e))?;

    for (e_idx, tokens) in expert_tokens.iter().enumerate() {
        if tokens.is_empty() {
            continue;
        }

        let n_assign = tokens.len() as i32;
        let token_positions: Vec<u32> = tokens.iter().map(|(t, _)| *t as u32).collect();
        let token_weights: Vec<f32> = tokens.iter().map(|(_, w)| *w).collect();

        let idx_arr = Array::from_slice(&token_positions, &[n_assign]);

        // Gather this expert's tokens
        let expert_input = hidden
            .take_axis(&idx_arr, 0)
            .map_err(|e| format!("moe expert {} gather: {:?}", e_idx, e))?;

        // Gated FFN: SiLU(gate @ x) * (up @ x) -> down
        let gate_out = expert_input
            .matmul(&gate_proj[e_idx])
            .map_err(|e| format!("moe expert {} gate: {:?}", e_idx, e))?;
        let up_out = expert_input
            .matmul(&up_proj[e_idx])
            .map_err(|e| format!("moe expert {} up: {:?}", e_idx, e))?;
        let gated = mlx_rs::nn::silu(&gate_out)
            .map_err(|e| format!("moe expert {} silu: {:?}", e_idx, e))?
            .multiply(&up_out)
            .map_err(|e| format!("moe expert {} mul: {:?}", e_idx, e))?;
        let expert_out = gated
            .matmul(&down_proj[e_idx])
            .map_err(|e| format!("moe expert {} down: {:?}", e_idx, e))?;

        // Scale by routing weight
        let weight_arr = Array::from_slice(&token_weights, &[n_assign, 1]);
        let weighted = expert_out
            .multiply(&weight_arr)
            .map_err(|e| format!("moe expert {} weight: {:?}", e_idx, e))?;

        // Accumulate contribution via slice-based add
        // Scatter each assigned token's weighted row back to its original
        // position using slice-based row updates.
        for (i, &t) in token_positions.iter().enumerate() {
            let row_idx = i as i32;
            let t_idx = t as i32;
            let row = weighted
                .index((row_idx, ..))
                .map_err(|e| format!("moe expert {} row index: {:?}", e_idx, e))?;
            let existing_row = output
                .index((t_idx, ..))
                .map_err(|e| format!("moe expert {} existing: {:?}", e_idx, e))?;
            let combined_row = existing_row
                .add(&row)
                .map_err(|e| format!("moe expert {} add row: {:?}", e_idx, e))?;
            let row_starts: Vec<i32> = vec![t_idx, 0];
            let row_ends: Vec<i32> = vec![t_idx + 1, hidden_size_i32];
            let reshaped = combined_row
                .reshape(&[1, hidden_size_i32])
                .map_err(|e| format!("moe expert {} reshape row: {:?}", e_idx, e))?;
            output = output
                .slice_update(&reshaped, &row_starts, &row_ends)
                .map_err(|e| format!("moe expert {} scatter row: {:?}", e_idx, e))?;
        }
    }

    Ok(output)
}

// ── Epilogue ───────────────────────────────────────────────────────────────

/// Result of an epilogue execution.
///
/// The caller MUST `eval()` the `selected_token` before reading the scalar
/// value. The `logits` field (when `Some`) holds the full logits tensor
/// (shape `[1, seq_len, vocab_size]`) for optional inspection.
pub struct EpilogueResult {
    /// Scalar token array — caller MUST eval() before reading.
    pub selected_token: Array,
    /// Full logits tensor [1, seq_len, vocab_size] before last-token slicing.
    pub logits: Option<Array>,
}

/// Final normalization, tied output projection, softcapping, and token selection.
///
/// Returns an `EpilogueResult` so the caller can explicitly `eval()` the
/// selected token before reading it. Logits are returned as an `Option` for
/// optional inspection — the caller can `eval()` and inspect them as needed.
///
/// This function does NOT force `eval()` on the returned arrays.
pub fn run_epilogue(
    hidden: &Array,
    final_norm: &Array,
    output_weight: &Array,
    output_scales: &Array,
    output_biases: &Array,
    plan: &EpiloguePlan,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    sampler: &SamplerConfig,
) -> MlxResult<EpilogueResult> {
    // Shape contract: hidden state is batchless [tokens, hidden_size].
    debug_assert_eq!(
        hidden.ndim(),
        2,
        "hidden state must be rank 2 (batchless), got rank {}",
        hidden.ndim()
    );

    // Final RMSNorm
    let normed = primitives::rms_norm(hidden, final_norm, rms_norm_eps)?;

    // Tied output projection: quantized matmul with embedding weights
    let group_size = if output_scales.shape().len() >= 1 {
        (output_weight.shape()[1] as i32 * 4)
            / output_scales.shape()[output_scales.shape().len() - 1]
    } else {
        64
    };
    let logits = if tie_word_embeddings {
        mlx_rs::ops::quantized_matmul(
            &normed,
            output_weight,
            output_scales,
            output_biases,
            true, // transpose weight
            group_size,
            8,
        )?
    } else {
        // Non-tied: use dedicated lm_head tensor if available
        mlx_rs::ops::quantized_matmul(
            &normed,
            output_weight,
            output_scales,
            output_biases,
            true,
            group_size,
            8,
        )?
    };

    // Final logit softcapping
    let logits = if let Some(cap) = plan.final_logit_softcapping {
        let cap_f32 = cap as f32;
        let scaled = logits.divide(&Array::from_f32(cap_f32))?;
        let tanh = mlx_rs::ops::tanh(&scaled)?;
        tanh.multiply(&Array::from_f32(cap_f32))?
    } else {
        logits
    };

    // logits is rank 2 [tokens, vocab]. Extract last token row, then restore
    // batch+seq dims for argmax compat: [1, 1, vocab_size].
    let last_row = logits.index(((logits.shape()[0] - 1)..logits.shape()[0], ..));
    let last_logits = last_row.reshape(&[1, 1, -1])?;

    // Check if grammar masking is required.
    let has_grammar = sampler.grammar.is_some() && sampler.grammar_tokenizer.is_some();

    // Greedy path without grammar: fast argmax (no f32 extraction overhead)
    if sampler.is_greedy() && !has_grammar {
        let token_arr = ops::indexing::argmax_axis(&last_logits, -1, false)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("argmax: {:?}", e)))?;
        return Ok(EpilogueResult {
            selected_token: token_arr,
            logits: Some(logits),
        });
    }

    // Must extract f32 logits for grammar masking and/or non-greedy sampling.
    last_logits.eval()?;

    // Flatten to 1D for contiguous extraction
    let flat = last_logits.reshape(&[-1])?;
    let vocab_size = flat.shape()[0] as usize;
    let mut logits_vec: Vec<f32> = flat
        .try_as_slice::<f32>()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("read logits: {:?}", e)))?
        .to_vec();

    // 0. Grammar mask: set invalid token logits to -inf
    //    Applied before temperature/top-k/top-p so invalid tokens are
    //    never candidates regardless of other parameters.
    if has_grammar {
        sampler.apply_grammar_mask(&mut logits_vec);
    }

    // Greedy path with grammar: argmax on the masked logits
    if sampler.is_greedy() {
        let token = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(EpilogueResult {
            selected_token: Array::from_slice(&[token], &[1]),
            logits: Some(logits),
        });
    }

    // 1. Temperature scaling
    if let Some(temp) = sampler.temperature {
        if temp > 0.0 && (temp - 1.0).abs() > f32::EPSILON {
            let scale = 1.0 / temp;
            for v in &mut logits_vec {
                *v *= scale;
            }
        }
    }

    // 2. Top-k filtering
    if let Some(k) = sampler.top_k {
        let k = (k as usize).min(vocab_size);
        if k > 0 && k < vocab_size {
            let mut sorted = logits_vec.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = sorted[k - 1];
            for v in &mut logits_vec {
                if *v < threshold {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    // 3. Top-p (nucleus) filtering
    if let Some(p) = sampler.top_p {
        if p > 0.0 && p < 1.0 {
            // Compute softmax probabilities for sorting
            let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; vocab_size];
            let mut prob_sum = 0.0f32;
            for (i, &v) in logits_vec.iter().enumerate() {
                let e = (v - max_l).exp();
                probs[i] = e;
                prob_sum += e;
            }
            if prob_sum > 0.0 {
                for v in &mut probs {
                    *v /= prob_sum;
                }
            }

            // Sort indices by probability descending
            let mut indices: Vec<usize> = (0..vocab_size).collect();
            indices.sort_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Find cumulative cutoff; zero out logits beyond it
            let mut cumsum = 0.0f32;
            for (rank, &idx) in indices.iter().enumerate() {
                cumsum += probs[idx];
                if cumsum > p {
                    for &i in &indices[rank..] {
                        logits_vec[i] = f32::NEG_INFINITY;
                    }
                    break;
                }
            }
        }
    }

    // 4. Check if everything was filtered — fall back to argmax
    let all_inf = logits_vec.iter().all(|v| !v.is_finite() || v.is_nan());
    if all_inf {
        let token = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(EpilogueResult {
            selected_token: Array::from_slice(&[token], &[1]),
            logits: Some(logits),
        });
    }

    // 5. Categorical sample via MLX
    let shape = [1i32, 1, vocab_size as i32];
    let filtered_arr = Array::from_slice(&logits_vec, &shape);
    let key = match sampler.seed {
        Some(s) => Some(mlx_rs::random::key(s)?),
        None => None,
    };
    let token_arr = mlx_rs::random::categorical(&filtered_arr, None, None, key.as_ref())?;
    Ok(EpilogueResult {
        selected_token: token_arr,
        logits: Some(logits),
    })
}

/// Accelerated epilogue path with ANE weight prefetch.
///
/// Same interface as [`run_epilogue`] but additionally predicts the next
/// token candidate(s) via the ANE predictor, pre-fetches their LM head weight
/// rows into ANE SRAM, and uses the hybrid LM head path that reads cached
/// rows from IOSurface-backed memory for zero-latency access.
///
/// When the predictor is unavailable (`None`), falls back to the standard
/// `run_epilogue` path unchanged.
pub fn run_epilogue_prefetch(
    hidden: &Array,
    final_norm: &Array,
    output_weight: &Array,
    output_scales: &Array,
    output_biases: &Array,
    plan: &EpiloguePlan,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    sampler: &SamplerConfig,
    predictor: Option<&mut HotRowPredictor>,
    row_cache: Option<&mut WeightRowCache>,
) -> MlxResult<EpilogueResult> {
    // Fall back to standard epilogue if prefetch is not configured.
    let (predictor, row_cache) = match (predictor, row_cache) {
        (Some(p), Some(c)) => (p, c),
        _ => {
            return run_epilogue(
                hidden,
                final_norm,
                output_weight,
                output_scales,
                output_biases,
                plan,
                rms_norm_eps,
                tie_word_embeddings,
                sampler,
            );
        }
    };

    // Shape contract: hidden state is [1, hidden_size] for decode.
    debug_assert_eq!(
        hidden.ndim(),
        2,
        "hidden state must be rank 2 (batchless), got rank {}",
        hidden.ndim()
    );

    // Final RMSNorm (same as run_epilogue)
    let normed = primitives::rms_norm(hidden, final_norm, rms_norm_eps)?;

    // 1. Read hidden state as f32 slice for the predictor.
    let hidden_slice: Vec<f32> = normed
        .try_as_slice::<f32>()
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!(
                "run_epilogue_prefetch: read normed hidden: {:?}",
                e
            ))
        })?
        .to_vec();

    // 2. Predict next token candidates on ANE.
    let candidates = predictor.predict(&hidden_slice).map_err(|e| {
        mlx_rs::error::Exception::custom(format!(
            "run_epilogue_prefetch: predictor error: {}",
            e
        ))
    })?;

    // 3. Pre-fetch those rows into ANE SRAM.
    row_cache.prefetch_rows(&candidates, output_weight).map_err(|e| {
        mlx_rs::error::Exception::custom(format!(
            "run_epilogue_prefetch: prefetch error: {}",
            e
        ))
    })?;

    // 4. Run hybrid LM head (cached rows are fast, rest are normal).
    let logits = row_cache.hybrid_lm_head(&normed, output_weight).map_err(|e| {
        mlx_rs::error::Exception::custom(format!(
            "run_epilogue_prefetch: hybrid_lm_head error: {}",
            e
        ))
    })?;

    // 5. Final logit softcapping (same as run_epilogue)
    let logits = if let Some(cap) = plan.final_logit_softcapping {
        let cap_f32 = cap as f32;
        let scaled = logits.divide(&Array::from_f32(cap_f32))?;
        let tanh = mlx_rs::ops::tanh(&scaled)?;
        tanh.multiply(&Array::from_f32(cap_f32))?
    } else {
        logits
    };

    // 6. Restore batch+seq dims: [1, 1, vocab_size] for argmax compat.
    let last_logits = logits.reshape(&[1, 1, -1])?;

    // Check if grammar masking is required.
    let has_grammar = sampler.grammar.is_some() && sampler.grammar_tokenizer.is_some();

    // 7. Sample.
    let selected_token = if sampler.is_greedy() && !has_grammar {
        // Fast path: argmax directly (no f32 extraction)
        let token_arr = ops::indexing::argmax_axis(&last_logits, -1, false)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("argmax: {:?}", e)))?;
        token_arr
    } else {
        // Must extract f32 for grammar masking and/or non-greedy sampling.
        last_logits.eval().map_err(|e| {
            mlx_rs::error::Exception::custom(format!("last_logits eval: {:?}", e))
        })?;

        let flat = last_logits.reshape(&[-1])?;
        let vocab_size = flat.shape()[0] as usize;
        let mut logits_vec: Vec<f32> = flat
            .try_as_slice::<f32>()
            .map_err(|e| {
                mlx_rs::error::Exception::custom(format!("read logits: {:?}", e))
            })?
            .to_vec();

        // 0. Grammar mask: set invalid token logits to -inf
        if has_grammar {
            sampler.apply_grammar_mask(&mut logits_vec);
        }

        // Greedy path with grammar: argmax on the masked logits
        if sampler.is_greedy() {
            // With grammar mask applied, pick highest-scoring valid token
            let token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            Array::from_slice(&[token], &[1])
        } else {
        // Non-greedy sampling: temperature, top-k, top-p
        // Temperature
        if let Some(temp) = sampler.temperature {
            if temp > 0.0 && (temp - 1.0).abs() > f32::EPSILON {
                let scale = 1.0 / temp;
                for v in &mut logits_vec {
                    *v *= scale;
                }
            }
        }

        // Top-k
        if let Some(k) = sampler.top_k {
            let k = (k as usize).min(vocab_size);
            if k > 0 && k < vocab_size {
                let mut sorted = logits_vec.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let threshold = sorted[k - 1];
                for v in &mut logits_vec {
                    if *v < threshold {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Top-p
        if let Some(p) = sampler.top_p {
            if p > 0.0 && p < 1.0 {
                let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs = vec![0.0f32; vocab_size];
                let mut prob_sum = 0.0f32;
                for (i, &v) in logits_vec.iter().enumerate() {
                    let e = (v - max_l).exp();
                    probs[i] = e;
                    prob_sum += e;
                }
                if prob_sum > 0.0 {
                    for v in &mut probs {
                        *v /= prob_sum;
                    }
                }
                let mut indices: Vec<usize> = (0..vocab_size).collect();
                indices.sort_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut cumsum = 0.0f32;
                for (rank, &idx) in indices.iter().enumerate() {
                    cumsum += probs[idx];
                    if cumsum > p {
                        for &i in &indices[rank..] {
                            logits_vec[i] = f32::NEG_INFINITY;
                        }
                        break;
                    }
                }
            }
        }

        let all_inf = logits_vec.iter().all(|v| !v.is_finite() || v.is_nan());
        let token = if all_inf {
            logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        } else {
            let shape = [1i32, 1, vocab_size as i32];
            let filtered_arr = Array::from_slice(&logits_vec, &shape);
            let key = match sampler.seed {
                Some(s) => Some(mlx_rs::random::key(s)?),
                None => None,
            };
            let token_arr = mlx_rs::random::categorical(&filtered_arr, None, None, key.as_ref())?;
            let t: Vec<u32> = token_arr
                .try_as_slice::<u32>()
                .map_err(|e| mlx_rs::error::Exception::custom(format!("categorical: {:?}", e)))?
                .to_vec();
            t.first().copied().unwrap_or(0)
        };
        Array::from_slice(&[token], &[1])
        }
    };

    // 8. Update prediction statistics.
    let token_slice: Vec<u32> = selected_token
        .try_as_slice::<u32>()
        .map_err(|e| {
            mlx_rs::error::Exception::custom(format!(
                "run_epilogue_prefetch: read token: {:?}",
                e
            ))
        })?
        .to_vec();
    let selected_token_id = token_slice.first().copied().unwrap_or(0);
    predictor.record_outcome(selected_token_id);

    Ok(EpilogueResult {
        selected_token,
        logits: Some(logits),
    })
}

// ── Mask helpers ───────────────────────────────────────────────────────────

/// Build a causal attention mask sized [query_len, kv_len].
///
/// Position `i` of the query attends to key positions `j <= offset + i`.
/// For single-token decode against a cache (query_len=1), the mask is [1, kv_len].
fn causal_mask(query_len: u32, kv_len: u32, offset: u32) -> MlxResult<Array> {
    let rows_usize = query_len as usize;
    let cols_usize = kv_len as usize;
    let mut d = vec![0.0f32; rows_usize * cols_usize];
    for i in 0..rows_usize {
        let max_key = offset as usize + i;
        for j in 0..cols_usize {
            if j > max_key {
                d[i * cols_usize + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Array::from_slice(
        &d,
        &[1, 1, query_len as i32, kv_len as i32],
    ))
}

/// Build a sliding-window attention mask sized [query_len, kv_len].
///
/// Each query position attends only to keys within the sliding window.
/// For single-token decode against a cache, the mask is [1, kv_len].
fn sliding_mask(query_len: u32, kv_len: u32, window: u32, offset: u32) -> MlxResult<Array> {
    let rows_usize = query_len as usize;
    let cols_usize = kv_len as usize;
    let mut d = vec![0.0f32; rows_usize * cols_usize];
    for i in 0..rows_usize {
        let query_pos = offset as usize + i;
        let min_key = query_pos.saturating_add(1).saturating_sub(window as usize);
        for j in 0..cols_usize {
            if j < min_key || j > query_pos {
                d[i * cols_usize + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Array::from_slice(
        &d,
        &[1, 1, query_len as i32, kv_len as i32],
    ))
}

fn repeat_kv(x: &Array, n_rep: u32) -> MlxResult<Array> {
    if n_rep <= 1 {
        return Ok(x.clone());
    }
    // x: [N, n_kv, hd] -> insert dim at axis 1 -> [N, 1, n_kv, hd] -> tile -> [N, n_rep, n_kv, hd] -> [N, n_rep*n_kv, hd]
    let s = x.shape();
    let r = x.reshape(&[s[0], 1, s[1], s[2]])?;
    let r = mlx_rs::ops::tile(&r, &[1, n_rep as i32, 1, 1])?;
    r.reshape(&[s[0], s[1] * n_rep as i32, s[2]])
}

// ── MoE Layer ──────────────────────────────────────────────────────────────

/// Execute one MoE FFN layer on ANE using NPUMoE expert scheduling.
pub fn run_moe_layer(
    hidden: &Array,
    router_weight: &Array,
    router_bias: Option<&Array>,
    expert_weights: &[ExpertWeights],
    moe_config: &crate::config::MoEConfig,
    scheduler: &AneMoEScheduler,
) -> MlxResult<Array> {
    let logits = hidden.matmul(router_weight)?;
    let logits = if let Some(bias) = router_bias {
        logits.add(bias)?
    } else {
        logits
    };
    scheduler
        .forward_moe(hidden, expert_weights, &logits, moe_config.top_k_experts)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("run_moe_layer: {}", e)))
}
