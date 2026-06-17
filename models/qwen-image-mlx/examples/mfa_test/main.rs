//! Minimal Metal Flash Attention test
//!
//! Run with:
//!   DYLD_LIBRARY_PATH=~/home/OminiX-MLX/universal-metal-flash-attention/.build/release \
//!   cargo run --example mfa_test

use std::ffi::c_void;
use std::ptr;

// FFI bindings
#[repr(C)]
#[derive(Clone, Copy)]
struct MfaContext(*mut c_void);

#[repr(C)]
#[derive(Clone, Copy)]
struct MfaBuffer(*mut c_void);

const MFA_SUCCESS: i32 = 0;
const MFA_PRECISION_FP32: i32 = 2;
const MFA_MASK_TYPE_NONE: i32 = 0;
const MFA_MASK_SCALAR_BYTE: i32 = 0;

#[link(name = "MFAFFI")]
extern "C" {
    fn mfa_is_device_supported() -> bool;
    fn mfa_get_version(major: *mut i32, minor: *mut i32, patch: *mut i32);
    fn mfa_create_context(context: *mut MfaContext) -> i32;
    fn mfa_destroy_context(context: MfaContext);
    fn mfa_create_buffer(context: MfaContext, size_bytes: usize, buffer: *mut MfaBuffer) -> i32;
    fn mfa_buffer_contents(buffer: MfaBuffer) -> *mut c_void;
    fn mfa_destroy_buffer(buffer: MfaBuffer);
    fn mfa_attention_forward(
        context: MfaContext,
        q: MfaBuffer, k: MfaBuffer, v: MfaBuffer, out: MfaBuffer,
        batch_size: u32, seq_len_q: u32, seq_len_kv: u32,
        num_heads: u32, head_dim: u16,
        softmax_scale: f32, causal: bool,
        input_precision: i32,
        intermediate_precision: i32,
        output_precision: i32,
        transpose_q: bool, transpose_k: bool, transpose_v: bool, transpose_o: bool,
        mask_ptr: *const c_void, mask_size_bytes: usize,
        mask_shape: *const i64, mask_strides: *const i64, mask_ndim: u32,
        mask_type: i32, mask_scalar_type: i32,
    ) -> i32;
}

fn main() {
    println!("Metal Flash Attention Test");
    println!("=========================");

    // Check if device is supported
    let supported = unsafe { mfa_is_device_supported() };
    println!("Device supported: {}", supported);

    if !supported {
        println!("Device does not support Metal Flash Attention");
        return;
    }

    // Get version
    let mut major = 0i32;
    let mut minor = 0i32;
    let mut patch = 0i32;
    unsafe {
        mfa_get_version(&mut major, &mut minor, &mut patch);
    }
    println!("MFA version: {}.{}.{}", major, minor, patch);

    // Create context
    let mut context = MfaContext(ptr::null_mut());
    let err = unsafe { mfa_create_context(&mut context) };
    if err != MFA_SUCCESS {
        eprintln!("Failed to create MFA context: error {}", err);
        return;
    }
    println!("Context created successfully");

    // Create test buffers (small Q, K, V for a single attention head)
    let batch_size = 1u32;
    let seq_len = 16u32;
    let num_heads = 1u32;
    let head_dim = 64u16;
    let elem_size = 4u32; // fp32 = 4 bytes

    let q_size = (batch_size * num_heads as u32 * seq_len * head_dim as u32 * elem_size) as usize;
    let k_size = q_size;
    let v_size = q_size;
    let out_size = q_size;

    let mut q_buf = MfaBuffer(ptr::null_mut());
    let mut k_buf = MfaBuffer(ptr::null_mut());
    let mut v_buf = MfaBuffer(ptr::null_mut());
    let mut out_buf = MfaBuffer(ptr::null_mut());

    unsafe {
        mfa_create_buffer(context, q_size, &mut q_buf);
        mfa_create_buffer(context, k_size, &mut k_buf);
        mfa_create_buffer(context, v_size, &mut v_buf);
        mfa_create_buffer(context, out_size, &mut out_buf);
    }

    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // Run attention forward
    let err = unsafe {
        mfa_attention_forward(
            context,
            q_buf, k_buf, v_buf, out_buf,
            batch_size, seq_len, seq_len,
            num_heads, head_dim,
            softmax_scale, false, // causal = false
            MFA_PRECISION_FP32, // input precision
            MFA_PRECISION_FP32, // intermediate precision
            MFA_PRECISION_FP32, // output precision
            false, false, false, false, // no transpose
            ptr::null(), 0, // no mask
            ptr::null(), ptr::null(), 0, // no mask shape/strides
            MFA_MASK_TYPE_NONE, MFA_MASK_SCALAR_BYTE,
        )
    };

    if err == MFA_SUCCESS {
        println!("Attention forward pass: SUCCESS");
    } else {
        eprintln!("Attention forward pass failed: error {}", err);
    }

    // Cleanup
    unsafe {
        mfa_destroy_buffer(q_buf);
        mfa_destroy_buffer(k_buf);
        mfa_destroy_buffer(v_buf);
        mfa_destroy_buffer(out_buf);
        mfa_destroy_context(context);
    }

    println!("Test complete");
}
