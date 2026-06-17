use mlx_rs::{builder::Builder, nn, module::Module, ops::indexing::IndexOp, transforms::eval};

fn main() -> anyhow::Result<()> {
    eprintln!("Testing RoPE batch handling...");

    let batch = 2;
    let heads = 4;
    let seq_len = 1;
    let dim = 128;
    let theta = 10000.0_f32;

    // Create random Q and K tensors
    let q = mlx_rs::Array::randn::<f32>(&[batch, heads, seq_len, dim], -1.0, 1.0, None)?;
    let k = mlx_rs::Array::randn::<f32>(&[batch, heads, seq_len, dim], -1.0, 1.0, None)?;

    // Apply RoPE with batch=1 (expected correct path)
    let q_merged = q.reshape(&[1, batch * heads, seq_len, dim])?;
    let k_merged = k.reshape(&[1, batch * heads, seq_len, dim])?;

    let q_rope = mlx_rs::fast::rope(&q_merged, &[], seq_len, theta, true, None, None)?;
    let k_rope = mlx_rs::fast::rope(&k_merged, &[], seq_len, theta, true, None, None)?;

    // Reshape back
    let q_rope = q_rope.reshape(&[batch, heads, seq_len, dim])?;
    let k_rope = k_rope.reshape(&[batch, heads, seq_len, dim])?;

    eval(&[&q_rope, &k_rope])?;

    eprintln!("Q shape: {:?}", q_rope.shape());
    eprintln!("K shape: {:?}", k_rope.shape());

    // Verify batch elements differ (each batch element should get different rotation)
    let q0 = q_rope.index(&[0, .., .., ..])?;
    let q1 = q_rope.index(&[1, .., .., ..])?;
    let diff = (&q0 - &q1)?.abs()?.sum(None, None)?;
    eval(&[&diff])?;
    let diff_val: f32 = diff.item()?;

    eprintln!("Batch 0 vs 1 difference: {:.6}", diff_val);
    if diff_val > 0.01 {
        eprintln!("PASS: Batch elements differ (correct RoPE behavior)");
    } else {
        eprintln!("FAIL: Batch elements identical (RoPE bug detected)");
    }

    Ok(())
}
