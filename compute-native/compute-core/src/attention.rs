//! Gemma 4 attention blocks — 2D inputs (no batch/seq reshape needed).

use crate::backend::{MlxBackend, QuantizedWeightHandle, TensorHandle};
use crate::primitives;
use crate::projection_executor::{
    MaterializationClass, ProjectionExecutor, QuantizedProjectionDescriptor, RuntimeMode,
    StorageDtype,
};
use crate::projection_identity::ProjectionFamily;
use mlx_rs::error::Result as MlxResult;
use mlx_rs::Array;

/// Helper: run one attention projection through [`ProjectionExecutor`]
/// with the correct descriptor derived from tensors.
fn run_attention_projection(
    backend: &mut MlxBackend,
    x: TensorHandle,
    w: QuantizedWeightHandle,
    s: TensorHandle,
    b: TensorHandle,
    family: ProjectionFamily,
    hidden_size: u32,
    out_features: u32,
) -> MlxResult<Array> {
    let (group_size, physical_weight_shape) = {
        let w_arr = backend
            .get_weight(w)
            .map_err(|e| mlx_rs::error::Exception::custom(e))?;
        let s_arr = backend
            .get(s)
            .map_err(|e| mlx_rs::error::Exception::custom(e))?;
        let gs = (w_arr.shape()[1] as i32 * 4) / s_arr.shape()[1];
        let pws = vec![w_arr.shape()[0] as u32, w_arr.shape()[1] as u32];
        (gs as u32, pws)
    };
    let desc = QuantizedProjectionDescriptor {
        family,
        logical_in_features: hidden_size,
        logical_out_features: out_features,
        bits: 8,
        group_size,
        storage_dtype: StorageDtype::U32,
        physical_weight_shape,
        layer_index: 0,
        weight_materialization: MaterializationClass::MlxOwned,
    };
    let result_h = {
        let mut executor = ProjectionExecutor {
            backend: &mut *backend,
            mode: RuntimeMode::Safe,
        };
        executor
            .run_projection(x, w, s, b, &desc)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("{e}")))?
    };
    let result = backend
        .get(result_h)
        .map_err(|e| mlx_rs::error::Exception::custom(e))?
        .clone();
    Ok(result)
}

pub fn sliding_attention(
    x: &Array, // [N, 3840]
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
    rope_cos: &Array,
    rope_sin: &Array,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    sliding_window: u32,
    kv_offset: u32,
) -> MlxResult<Array> {
    let n_tokens = x.shape()[0];
    let n_rep = n_heads / n_kv_heads;
    let hidden_size = x.shape()[1] as u32;

    let mut backend = MlxBackend::new();
    let x_h = backend.alloc(x.clone());
    let qw_h = backend.alloc_weight(qw.clone());
    let qs_h = backend.alloc(qs.clone());
    let qb_h = backend.alloc(qb.clone());
    let kw_h = backend.alloc_weight(kw.clone());
    let ks_h = backend.alloc(ks.clone());
    let kb_h = backend.alloc(kb.clone());
    let vw_h = backend.alloc_weight(vw.clone());
    let vs_h = backend.alloc(vs.clone());
    let vb_h = backend.alloc(vb.clone());
    let ow_h = backend.alloc_weight(ow.clone());
    let os_h = backend.alloc(os.clone());
    let ob_h = backend.alloc(ob.clone());

    let q = run_attention_projection(
        &mut backend,
        x_h,
        qw_h,
        qs_h,
        qb_h,
        ProjectionFamily::QProj,
        hidden_size,
        n_heads * head_dim,
    )?
    .reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;
    let k = run_attention_projection(
        &mut backend,
        x_h,
        kw_h,
        ks_h,
        kb_h,
        ProjectionFamily::KProj,
        hidden_size,
        n_kv_heads * head_dim,
    )?
    .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;
    let v = run_attention_projection(
        &mut backend,
        x_h,
        vw_h,
        vs_h,
        vb_h,
        ProjectionFamily::VProj,
        hidden_size,
        n_kv_heads * head_dim,
    )?
    .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    let q = primitives::rms_norm_scale_free(&q.reshape(&[-1, head_dim as i32])?, 1e-6)?
        .reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;
    let k = primitives::rms_norm_scale_free(&k.reshape(&[-1, head_dim as i32])?, 1e-6)?
        .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    // RoPE: need [1, heads, seq, hd] format
    let q4d = q.reshape(&[1, n_heads as i32, n_tokens, head_dim as i32])?;
    let k4d = k.reshape(&[1, n_kv_heads as i32, n_tokens, head_dim as i32])?;
    let q4d = primitives::rope_apply(&q4d, rope_cos, rope_sin, kv_offset, None)?;
    let k4d = primitives::rope_apply(&k4d, rope_cos, rope_sin, kv_offset, None)?;
    let q = q4d.reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;
    let k = k4d.reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    let k_exp = if n_rep > 1 { repeat_kv(&k, n_rep)? } else { k };
    let v_exp = if n_rep > 1 { repeat_kv(&v, n_rep)? } else { v };

    // Scores: [heads, seq, hd] @ [heads, hd, seq] → [heads, seq, seq]
    let qt = q.reshape(&[n_heads as i32, n_tokens, head_dim as i32])?;
    let kt = k_exp.reshape(&[n_heads as i32, n_tokens, head_dim as i32])?;
    let vt = v_exp.reshape(&[n_heads as i32, n_tokens, head_dim as i32])?;
    let scale = (head_dim as f32).sqrt();
    let scores = qt
        .matmul(&mlx_rs::ops::transpose_axes(&kt, &[0, 2, 1])?)?
        .divide(&Array::from_f32(scale))?;

    let mask = causal_mask(n_tokens as u32)?.add(&sliding_mask(
        n_tokens as u32,
        sliding_window,
        n_heads,
    )?)?;
    let scores = scores.add(&mask)?;
    let attn = mlx_rs::ops::softmax_axes(&scores, &[-1], None)?;
    let out = attn
        .matmul(&vt)?
        .reshape(&[n_tokens, (n_heads * head_dim) as i32])?;
    let out_h = backend.alloc(out);
    run_attention_projection(
        &mut backend,
        out_h,
        ow_h,
        os_h,
        ob_h,
        ProjectionFamily::OProj,
        hidden_size,
        hidden_size,
    )?
    .reshape(&[n_tokens, -1])
}

pub fn full_attention(
    x: &Array,
    qw: &Array,
    qs: &Array,
    qb: &Array,
    kw: &Array,
    ks: &Array,
    kb: &Array,
    ow: &Array,
    os: &Array,
    ob: &Array,
    rope_cos: &Array,
    rope_sin: &Array,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_offset: u32,
) -> MlxResult<Array> {
    let n_tokens = x.shape()[0];
    let n_rep = n_heads / n_kv_heads;
    let hidden_size = x.shape()[1] as u32;

    let mut backend = MlxBackend::new();
    let x_h = backend.alloc(x.clone());
    let qw_h = backend.alloc_weight(qw.clone());
    let qs_h = backend.alloc(qs.clone());
    let qb_h = backend.alloc(qb.clone());
    let kw_h = backend.alloc_weight(kw.clone());
    let ks_h = backend.alloc(ks.clone());
    let kb_h = backend.alloc(kb.clone());
    let ow_h = backend.alloc_weight(ow.clone());
    let os_h = backend.alloc(os.clone());
    let ob_h = backend.alloc(ob.clone());

    let q = run_attention_projection(
        &mut backend,
        x_h,
        qw_h,
        qs_h,
        qb_h,
        ProjectionFamily::QProj,
        hidden_size,
        n_heads * head_dim,
    )?
    .reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;
    let k = run_attention_projection(
        &mut backend,
        x_h,
        kw_h,
        ks_h,
        kb_h,
        ProjectionFamily::KProj,
        hidden_size,
        n_kv_heads * head_dim,
    )?
    .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    let q = primitives::rms_norm_scale_free(&q.reshape(&[-1, head_dim as i32])?, 1e-6)?
        .reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;
    let k = primitives::rms_norm_scale_free(&k.reshape(&[-1, head_dim as i32])?, 1e-6)?
        .reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;

    let q4d = q.reshape(&[1, n_heads as i32, n_tokens, head_dim as i32])?;
    let k4d = k.reshape(&[1, n_kv_heads as i32, n_tokens, head_dim as i32])?;
    let q4d = primitives::rope_apply(&q4d, rope_cos, rope_sin, kv_offset, None)?;
    let k4d = primitives::rope_apply(&k4d, rope_cos, rope_sin, kv_offset, None)?;
    let q = q4d.reshape(&[n_tokens, n_heads as i32, head_dim as i32])?;
    let k = k4d.reshape(&[n_tokens, n_kv_heads as i32, head_dim as i32])?;
    let v = k.clone();

    let k_exp = if n_rep > 1 { repeat_kv(&k, n_rep)? } else { k };
    let v_exp = if n_rep > 1 { repeat_kv(&v, n_rep)? } else { v };

    let qt = q.reshape(&[n_heads as i32, n_tokens, head_dim as i32])?;
    let kt = k_exp.reshape(&[n_heads as i32, n_tokens, head_dim as i32])?;
    let vt = v_exp.reshape(&[n_heads as i32, n_tokens, head_dim as i32])?;
    let scale = (head_dim as f32).sqrt();
    let scores = qt
        .matmul(&mlx_rs::ops::transpose_axes(&kt, &[0, 2, 1])?)?
        .divide(&Array::from_f32(scale))?;

    let scores = scores.add(&causal_mask(n_tokens as u32)?)?;
    let attn = mlx_rs::ops::softmax_axes(&scores, &[-1], None)?;
    let out = attn
        .matmul(&vt)?
        .reshape(&[n_tokens, (n_heads * head_dim) as i32])?;
    let out_h = backend.alloc(out);
    run_attention_projection(
        &mut backend,
        out_h,
        ow_h,
        os_h,
        ob_h,
        ProjectionFamily::OProj,
        hidden_size,
        hidden_size,
    )?
    .reshape(&[n_tokens, -1])
}

fn causal_mask(seq_len: u32) -> MlxResult<Array> {
    let n = seq_len as usize;
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if j > i {
                d[i * n + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Array::from_slice(
        &d,
        &[1, 1, seq_len as i32, seq_len as i32],
    ))
}

fn sliding_mask(seq_len: u32, window: u32, _n_heads: u32) -> MlxResult<Array> {
    if seq_len <= window {
        return Ok(Array::from_slice(&[0.0f32], &[1, 1, 1, 1]));
    }
    let n = seq_len as usize;
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if (j as i32) < (i as i32 - window as i32) {
                d[i * n + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Array::from_slice(
        &d,
        &[1, 1, seq_len as i32, seq_len as i32],
    ))
}

fn repeat_kv(x: &Array, n_rep: u32) -> MlxResult<Array> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    // x: [N, n_kv, hd] → insert dim at axis 1 → [N, 1, n_kv, hd] → tile → [N, n_rep, n_kv, hd] → [N, n_rep*n_kv, hd]
    let s = x.shape();
    let r = x.reshape(&[s[0], 1, s[1], s[2]])?;
    let r = mlx_rs::ops::tile(&r, &[1, n_rep as i32, 1, 1])?;
    r.reshape(&[s[0], s[1] * n_rep as i32, s[2]])
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Shard {
        _buf: Vec<u8>,
        tensors: safetensors::SafeTensors<'static>,
    }
    impl Shard {
        fn load(p: &str) -> Self {
            // Leak-safe for test-framework lifetime (reclaimed on process exit)
            let buf: &'static [u8] = std::fs::read(p).unwrap().leak();
            Self {
                _buf: Vec::new(),
                tensors: safetensors::SafeTensors::deserialize(buf).unwrap(),
            }
        }

        fn a(&self, n: &str) -> Array {
            let tv = self.tensors.tensor(n).unwrap();
            crate::model::tensor_view_to_array(&tv)
        }
    }

    fn test_input() -> Array {
        let mut d = vec![0.0f32; 3840];
        let mut s = 42u64;
        for i in 0..3840 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            d[i] = (s as f32) / (u64::MAX as f32) * 0.02;
        }
        Array::from_slice(&d, &[1, 3840])
    }

    #[test]
    fn sliding_attn_l0() {
        let s = Shard::load("models/gemma4-12b-8bit/model-00001-of-00003.safetensors");
        let r = "language_model.model";
        let l = "layers.0";
        let (cw, sw) = primitives::rope_freqs(256, 1024, 10000.0).unwrap();
        let out = sliding_attention(
            &test_input(),
            &s.a(&format!("{}.{}.self_attn.q_proj.weight", r, l)),
            &s.a(&format!("{}.{}.self_attn.q_proj.scales", r, l)),
            &s.a(&format!("{}.{}.self_attn.q_proj.biases", r, l)),
            &s.a(&format!("{}.{}.self_attn.k_proj.weight", r, l)),
            &s.a(&format!("{}.{}.self_attn.k_proj.scales", r, l)),
            &s.a(&format!("{}.{}.self_attn.k_proj.biases", r, l)),
            &s.a(&format!("{}.{}.self_attn.v_proj.weight", r, l)),
            &s.a(&format!("{}.{}.self_attn.v_proj.scales", r, l)),
            &s.a(&format!("{}.{}.self_attn.v_proj.biases", r, l)),
            &s.a(&format!("{}.{}.self_attn.o_proj.weight", r, l)),
            &s.a(&format!("{}.{}.self_attn.o_proj.scales", r, l)),
            &s.a(&format!("{}.{}.self_attn.o_proj.biases", r, l)),
            &cw,
            &sw,
            16,
            8,
            256,
            1024,
            0,
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 3840]);
        let v: Vec<f32> = out.try_as_slice::<f32>().unwrap().to_vec();
        assert!(v.iter().all(|x| x.is_finite()));
        println!(
            "sliding-L0 PASS: shape=[1,3840] first={:.4} last={:.4}",
            v[0], v[3839]
        );
    }

    #[test]
    fn full_attn_l5() {
        let s = Shard::load("models/gemma4-12b-8bit/model-00001-of-00003.safetensors");
        let r = "language_model.model";
        let l = "layers.5";
        let (cw, sw) = primitives::rope_freqs(512, 131072, 1_000_000.0).unwrap();
        let out = full_attention(
            &test_input(),
            &s.a(&format!("{}.{}.self_attn.q_proj.weight", r, l)),
            &s.a(&format!("{}.{}.self_attn.q_proj.scales", r, l)),
            &s.a(&format!("{}.{}.self_attn.q_proj.biases", r, l)),
            &s.a(&format!("{}.{}.self_attn.k_proj.weight", r, l)),
            &s.a(&format!("{}.{}.self_attn.k_proj.scales", r, l)),
            &s.a(&format!("{}.{}.self_attn.k_proj.biases", r, l)),
            &s.a(&format!("{}.{}.self_attn.o_proj.weight", r, l)),
            &s.a(&format!("{}.{}.self_attn.o_proj.scales", r, l)),
            &s.a(&format!("{}.{}.self_attn.o_proj.biases", r, l)),
            &cw,
            &sw,
            16,
            1,
            512,
            0,
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 3840]);
        let v: Vec<f32> = out.try_as_slice::<f32>().unwrap().to_vec();
        assert!(v.iter().all(|x| x.is_finite()));
        println!(
            "full-L5 PASS: shape=[1,3840] first={:.4} last={:.4}",
            v[0], v[3839]
        );
    }
}
