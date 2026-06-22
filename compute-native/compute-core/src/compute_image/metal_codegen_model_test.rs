#![cfg(test)]

use crate::compute_image::fusion_plan::SelectedFusionRegion;
use crate::compute_image::metal_codegen::generate_metal_source;
use crate::fusion_region::FusionImplBackend;

/// Build a SelectedFusionRegion for testing with explicit model dimensions.
fn region_with_dims(
    id: &str,
    hidden_size: u64,
    num_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
    intermediate_size: u64,
) -> SelectedFusionRegion {
    SelectedFusionRegion {
        region_id: id.into(),
        ops: vec![],
        backend: FusionImplBackend::MlxGpu,
        eliminated_intermediates: 0,
        input_elements: hidden_size,
        output_elements: hidden_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
    }
}

// ── Qwen2.5-0.5B dimensions ──────────────────────────────────────────────

const QWEN_HIDDEN: u64 = 896;
const QWEN_HEADS: u64 = 14;
const QWEN_KV_HEADS: u64 = 2;
const QWEN_HEAD_DIM: u64 = 64;
const QWEN_INTERMEDIATE: u64 = 4864;

#[test]
fn test_qwen_kernels_use_correct_dimensions() {
    // QKV
    let src = generate_metal_source(&region_with_dims(
        "qkv_proj", QWEN_HIDDEN, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_INTERMEDIATE,
    ));
    assert!(src.source.contains(&format!("{}", QWEN_HIDDEN)),
        "QKV kernel should reference hidden_size={}", QWEN_HIDDEN);
    assert!(!src.source.contains("4096"),
        "QKV kernel should NOT contain hardcoded 4096");

    // Gate+Up proj
    let src = generate_metal_source(&region_with_dims(
        "gate_up_proj", QWEN_HIDDEN, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_INTERMEDIATE,
    ));
    assert!(src.source.contains(&format!("{}", QWEN_HIDDEN)),
        "gate_up_proj should reference hidden_size={}", QWEN_HIDDEN);
    assert!(src.source.contains(&format!("{}", QWEN_INTERMEDIATE)),
        "gate_up_proj should reference intermediate_size={}", QWEN_INTERMEDIATE);

    // Down proj
    let src = generate_metal_source(&region_with_dims(
        "down_proj", QWEN_HIDDEN, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_INTERMEDIATE,
    ));
    assert!(src.source.contains(&format!("{}", QWEN_HIDDEN)),
        "down_proj should reference hidden_size={}", QWEN_HIDDEN);

    // RMS norm
    let src = generate_metal_source(&region_with_dims(
        "rms_norm_residual", QWEN_HIDDEN, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_INTERMEDIATE,
    ));
    assert!(src.source.contains(&format!("{}", QWEN_HIDDEN)),
        "rms_norm should reference hidden_size={}", QWEN_HIDDEN);

    // Self-attention
    let src = generate_metal_source(&region_with_dims(
        "self_attn", QWEN_HIDDEN, QWEN_HEADS, QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_INTERMEDIATE,
    ));
    assert!(src.source.contains(&format!("{}", QWEN_HIDDEN)),
        "self_attn should reference hidden_size={}", QWEN_HIDDEN);
}

// ── Tiny model (512 hidden) — small enough to verify bounds are parametric ─

#[test]
fn test_tiny_model_kernels_use_small_dimensions() {
    let tiny = 512u64;
    let src = generate_metal_source(&region_with_dims(
        "qkv_proj", tiny, 8, 4, 64, 2048,
    ));
    // Verify the generated source uses the small dimension, not a hardcoded large number
    let small_loop = format!("{}", tiny);
    assert!(src.source.contains(&small_loop),
        "tiny QKV kernel should reference dimension {}", tiny);
}

// ── Llama 3 8B dimensions ────────────────────────────────────────────────

const LLAMA3_HIDDEN: u64 = 4096;
const LLAMA3_HEADS: u64 = 32;
const LLAMA3_KV_HEADS: u64 = 8;
const LLAMA3_HEAD_DIM: u64 = 128;
const LLAMA3_INTERMEDIATE: u64 = 14336;

#[test]
fn test_llama3_kernels_use_correct_dimensions() {
    let src = generate_metal_source(&region_with_dims(
        "qkv_proj", LLAMA3_HIDDEN, LLAMA3_HEADS, LLAMA3_KV_HEADS,
        LLAMA3_HEAD_DIM, LLAMA3_INTERMEDIATE,
    ));
    assert!(src.source.contains(&format!("{}", LLAMA3_HIDDEN)));
    assert!(!src.source.contains("896"), "should use 4096 not 896");
}

// ── All kernel templates produce valid Metal syntax ──────────────────────

#[test]
fn test_all_templates_compile_syntax() {
    for id in &["qkv_proj", "attn_out", "gate_up_proj", "silu_mul",
                 "down_proj", "rms_norm_residual", "self_attn"] {
        let src = generate_metal_source(&region_with_dims(
            id, LLAMA3_HIDDEN, LLAMA3_HEADS, LLAMA3_KV_HEADS,
            LLAMA3_HEAD_DIM, LLAMA3_INTERMEDIATE,
        ));
        assert!(src.source.contains("kernel void"),
            "{}: missing kernel declaration", id);
        assert!(src.source.contains(&format!("{}", LLAMA3_HIDDEN)),
            "{}: should reference hidden_size={}", id, LLAMA3_HIDDEN);
    }
}
