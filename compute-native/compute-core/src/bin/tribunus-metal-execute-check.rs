//! Metal kernel execution proof.
//! Loads a .metallib artifact, dispatches on GPU, reads back output,
//! compares against MLX JIT quantized_matmul and CPU reference.
//!
//! Usage: cargo run --features mlx-backend,metal-dispatch \
//!   --bin tribunus-metal-execute-check
//!
//! ABI from mlx_kernel_abi.c:
//!   buffer(0): weights (u32 packed NF4)
//!   buffer(1): scales (f32 per-group)
//!   buffer(2): input (f32 activation)
//!   buffer(3): output (f32 result)
//!   constant buffer(4): K
//!   constant buffer(5): N
//!   constant buffer(6): M
//!   entry_point: affine_qmm_t
//!   threads_per_threadgroup: [32, 1, 1]

use tribunus_compute_core::compute_image::manifest::MetalKernelArtifact;

fn main() {
    // Which artifact to test — pick the q_proj shape (k=4864, n=896)
    let target_shape = [4864u32, 896u32]; // [k, n]
    let image_dir = std::path::Path::new("./models/qwen2.5-hw-bench");

    // 1. Load manifest and find target artifact
    let manifest_json =
        std::fs::read_to_string(image_dir.join("manifest.json")).expect("manifest.json not found");
    let manifest: tribunus_compute_core::compute_image::manifest::Manifest =
        serde_json::from_str(&manifest_json).expect("invalid manifest");

    let art = manifest
        .metal_kernel_artifacts
        .iter()
        .find(|a| a.logical_shape == target_shape)
        .expect("no artifact with [4864, 896] shape found");

    let k = art.dispatch.k as usize;
    let n = art.dispatch.n as usize;
    let gs = art.dispatch.group_size as usize;
    let n_groups = k / gs;
    let packed_words_per_col = k / 8; // NF4: 8 values per u32
    let storage_n = n;
    let storage_k_words = packed_words_per_col;

    println!("=== Metal Execution Proof ===");
    println!("artifact:   {}", art.artifact_id);
    println!("entry_point: {}", art.dispatch.entry_point);
    println!("k={} n={}  gs={}  bits={}", k, n, gs, art.dispatch.bits);
    println!("threadgroup: {:?}", art.dispatch.threads_per_threadgroup);
    println!("grid:       {:?}", art.dispatch.threadgroups_per_grid);
    println!("slot_map:   {:?}", art.dispatch.buffer_slot_map);
    println!("scalars:    {:?}", art.dispatch.scalar_index_map);

    // 2. Load .metallib
    let metallib_bytes =
        std::fs::read(image_dir.join(&art.metallib_relpath)).expect("failed to read .metallib");
    println!("metallib:    {} bytes", metallib_bytes.len());

    // 3. Create Metal device, library, pipeline
    let device = metal::Device::system_default().expect("no Metal device");
    println!("device:     {}", device.name());

    let library = device
        .new_library_with_data(&metallib_bytes)
        .expect("failed to create Metal library");
    let function = library
        .get_function(&art.dispatch.entry_point, None)
        .expect("entry_point not found in library");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("failed to create pipeline state");
    println!("pipeline:   OK  (ABI v{})", art.dispatch.kernel_abi_version);

    // 4. Build deterministic test data (same for Metal, MLX, CPU)
    //    Input: M=1 decode step, K-length float vector
    let m = 1usize;
    let mut input_vals = vec![0.0f32; m * k];
    for i in 0..k {
        input_vals[i] = ((i as f32) * 0.1).sin();
    }

    //    Weights: NF4-packed u32 (each u32 holds 8 four-bit signed values)
    let weight_len = storage_n * packed_words_per_col;
    let mut weight_vals = vec![0u32; weight_len];
    for col in 0..storage_n {
        for w in 0..packed_words_per_col {
            // Each word holds 8 nibbles: value at position v = (col*w*8 + v) % 15, sign-extended
            let mut word = 0u32;
            for nibble in 0..8 {
                let raw = ((col * packed_words_per_col * 8 + w * 8 + nibble) % 15) as u32;
                // Sign-extend 4-bit
                let signed = if raw >= 8 { raw | 0xFFFFFFF0 } else { raw };
                word |= (signed & 0xF) << (nibble * 4);
            }
            weight_vals[col * packed_words_per_col + w] = word;
        }
    }

    //    Scales: one f32 per group per column
    let scale_len = storage_n * n_groups;
    let mut scale_vals = vec![0.0f32; scale_len];
    for i in 0..scale_len {
        scale_vals[i] = (i as f32).fract() * 0.1 + 0.01;
    }

    let input_bytes = (m * k * 4) as u64;
    let weight_bytes = (weight_len * 4) as u64;
    let scale_bytes = (scale_len * 4) as u64;
    let output_bytes = (m * n * 4) as u64;
    println!(
        "buffers: input={}B weight={}B scale={}B output={}B",
        input_bytes, weight_bytes, scale_bytes, output_bytes
    );

    // 5. Create shared MTLBuffers
    let input_buf = device.new_buffer_with_data(
        input_vals.as_ptr() as *const std::ffi::c_void,
        input_bytes,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let weight_buf = device.new_buffer_with_data(
        weight_vals.as_ptr() as *const std::ffi::c_void,
        weight_bytes,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let scale_buf = device.new_buffer_with_data(
        scale_vals.as_ptr() as *const std::ffi::c_void,
        scale_bytes,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(output_bytes, metal::MTLResourceOptions::StorageModeShared);

    // 6. Dispatch
    let queue = device.new_command_queue();
    let cmd_buf = queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);

    // Bind buffers at known ABI slots
    enc.set_buffer(0, Some(&weight_buf), 0); // weights
    enc.set_buffer(1, Some(&scale_buf), 0); // scales
    enc.set_buffer(2, Some(&input_buf), 0); // input
    enc.set_buffer(3, Some(&output_buf), 0); // output
                                             // Scalars at constant buffer slots 4-6
    let k_val: u32 = k as u32;
    let n_val: u32 = n as u32;
    let m_val: u32 = m as u32;
    enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &m_val as *const u32 as *const std::ffi::c_void);

    // Dispatch geometry
    let grid = metal::MTLSize::new(
        art.dispatch.threadgroups_per_grid[0] as u64,
        art.dispatch.threadgroups_per_grid[1] as u64,
        art.dispatch.threadgroups_per_grid[2] as u64,
    );
    let group = metal::MTLSize::new(
        art.dispatch.threads_per_threadgroup[0] as u64,
        art.dispatch.threads_per_threadgroup[1] as u64,
        art.dispatch.threads_per_threadgroup[2] as u64,
    );
    enc.dispatch_thread_groups(grid, group);
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // 7. Read back Metal output
    let out_ptr = output_buf.contents() as *const f32;
    let metal_out = unsafe { std::slice::from_raw_parts(out_ptr, m * n) };
    println!("\nMetal dispatch (first 8):");
    for (i, &v) in metal_out.iter().enumerate().take(8) {
        println!("  [{}] {:.6}", i, v);
    }

    // 8. MLX JIT reference
    println!("\nMLX JIT reference...");
    let x_arr = mlx_rs::Array::from_slice(&input_vals, &[m as i32, k as i32]);
    let w_arr = mlx_rs::Array::from_slice(
        &weight_vals,
        &[storage_n as i32, packed_words_per_col as i32],
    );
    let s_arr = mlx_rs::Array::from_slice(&scale_vals, &[storage_n as i32, n_groups as i32]);
    let bias_vals = vec![0.0f32; storage_n * n_groups];
    let b_arr = mlx_rs::Array::from_slice(&bias_vals, &[storage_n as i32, n_groups as i32]);
    let result = mlx_rs::ops::quantized_matmul(
        &x_arr,
        &w_arr,
        &s_arr,
        Some(&b_arr) as Option<&mlx_rs::Array>,
        true,
        gs as i32,
        art.bits as i32,
    )
    .expect("MLX quantized_matmul failed");
    let _ = result.eval();
    let mlx_out: Vec<f32> = result.as_slice::<f32>().to_vec();

    println!("MLX JIT (first 8):");
    for (i, &v) in mlx_out.iter().enumerate().take(8) {
        println!("  [{}] {:.6}", i, v);
    }

    // 9. CPU reference
    println!("\nCPU reference...");
    let cpu_out = cpu_quantized_matmul(
        &input_vals,
        &weight_vals,
        &scale_vals,
        k,
        n,
        gs,
        art.bits as usize,
    );
    println!("CPU reference (first 8):");
    for (i, &v) in cpu_out.iter().enumerate().take(8) {
        println!("  [{}] {:.6}", i, v);
    }

    // 10. Comparison
    println!("\n--- Comparison ---");
    let diff_mm: Vec<f64> = metal_out
        .iter()
        .zip(mlx_out.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .collect();
    let diff_mc: Vec<f64> = metal_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .collect();
    let diff_mx_cpu: Vec<f64> = mlx_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .collect();

    let max_mm = diff_mm.iter().cloned().fold(0.0f64, f64::max);
    let max_mc = diff_mc.iter().cloned().fold(0.0f64, f64::max);
    let max_mlx_cpu = diff_mx_cpu.iter().cloned().fold(0.0f64, f64::max);

    println!("  Metal vs MLX JIT:   max diff = {:.8e}", max_mm);
    println!("  Metal vs CPU ref:   max diff = {:.8e}", max_mc);
    println!("  MLX JIT vs CPU ref: max diff = {:.8e}", max_mlx_cpu);

    let threshold = 1e-3;
    let ok_mm = max_mm < threshold;
    let ok_mlx = max_mlx_cpu < threshold;
    if ok_mm && ok_mlx {
        println!("\n  ✓ PARITY OK: direct Metal dispatch matches MLX and CPU");
    } else {
        println!("\n  ⚠  PARITY ISSUE:");
        if !ok_mm {
            println!(
                "     Metal vs MLX JIT exceed threshold ({:.1e} vs {:.1e})",
                max_mm, threshold
            );
        }
        if !ok_mlx {
            println!(
                "     MLX JIT vs CPU exceed threshold ({:.1e} vs {:.1e})",
                max_mlx_cpu, threshold
            );
        }
    }
}

/// CPU dequantize + matmul reference.
/// weights: [n, k/8] u32 (NF4 packed)
/// scales: [n, k/gs] f32
fn cpu_quantized_matmul(
    input: &[f32],
    weights: &[u32],
    scales: &[f32],
    k: usize,
    n: usize,
    gs: usize,
    _bits: usize,
) -> Vec<f32> {
    let n_groups = k / gs;
    let packed_per_col = k / 8;
    let mut output = vec![0.0f32; n];
    for col in 0..n {
        let mut sum = 0.0f32;
        for g in 0..n_groups {
            let scale = scales[col * n_groups + g];
            for v in 0..gs / 8 {
                let packed = weights[col * packed_per_col + g * (gs / 8) + v];
                for nibble in 0..8 {
                    let idx = (packed >> (nibble * 4)) & 0xF;
                    let signed = (idx as i32) << 28 >> 28; // sign extend
                    let deq = scale * signed as f32;
                    let in_idx = g * gs + v * 8 + nibble;
                    if in_idx < k {
                        sum += input[in_idx] * deq;
                    }
                }
            }
        }
        output[col] = sum;
    }
    output
}
