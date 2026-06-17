# Metal FlashAttention Integration for Qwen-Image

## Overview

This document outlines the feasibility of integrating [universal-metal-flash-attention](https://github.com/bghira/universal-metal-flash-attention) to accelerate the DiT transformer in Qwen-Image.

## API Summary

The library provides a C FFI with these key functions:

```c
// Context management
mfa_error_t mfa_create_context(mfa_context_t* context);
void mfa_destroy_context(mfa_context_t context);

// Buffer management
mfa_error_t mfa_create_buffer(mfa_context_t context, size_t size_bytes, mfa_buffer_t* buffer);
mfa_error_t mfa_buffer_from_ptr(mfa_context_t context, void* data_ptr, size_t size_bytes, mfa_buffer_t* buffer);
void* mfa_buffer_contents(mfa_buffer_t buffer);
void mfa_destroy_buffer(mfa_buffer_t buffer);

// Forward attention
mfa_error_t mfa_attention_forward(
    mfa_context_t context,
    mfa_buffer_t q, mfa_buffer_t k, mfa_buffer_t v, mfa_buffer_t out,
    uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
    uint32_t num_heads, uint16_t head_dim,
    float softmax_scale, bool causal,
    mfa_precision_t input_precision,
    mfa_precision_t intermediate_precision,
    mfa_precision_t output_precision,
    bool transpose_q, bool transpose_k, bool transpose_v, bool transpose_o,
    const void* mask_ptr, size_t mask_size_bytes,
    const int64_t* mask_shape, const int64_t* mask_strides, uint32_t mask_ndim,
    mfa_mask_type_t mask_type, mfa_mask_scalar_t mask_scalar_type
);
```

## Precision Types

```c
MFA_PRECISION_FP16  // 0
MFA_PRECISION_BF16  // 1
MFA_PRECISION_FP32  // 2
MFA_PRECISION_INT8  // 3
MFA_PRECISION_INT4  // 4
```

## Requirements

- macOS 15+ (Sequoia)
- Xcode 15+ with Swift 5.10+
- Metal-capable device (M1+)

## Integration Architecture

```
+-----------------------------------------------------------+
|                    Qwen-Image Pipeline                     |
+-----------------------------------------------------------+
|  Text Encoder (MLX)  ->  Transformer (MFA)  ->  VAE (MLX) |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
|               Metal Flash Attention Bridge                 |
+-----------------------------------------------------------+
|  1. Convert MLX Array -> MFA Buffer (zero-copy if possible)|
|  2. Call mfa_attention_forward()                           |
|  3. Convert MFA Buffer -> MLX Array                       |
+-----------------------------------------------------------+
```

## Qwen-Image Attention Parameters

| Parameter | Value |
|-----------|-------|
| batch_size | 1 (typically) |
| seq_len_q | 1024 (32x32 patches) + txt_seq |
| seq_len_kv | Same as seq_len_q |
| num_heads | 24 |
| head_dim | 128 |
| precision | BF16 |
| causal | false (joint attention) |

## Implementation Steps

### Step 1: Build universal-metal-flash-attention

```bash
git clone https://github.com/bghira/universal-metal-flash-attention
cd universal-metal-flash-attention
git submodule update --init --recursive
swift build -c release
```

### Step 2: Create Rust FFI Bindings

```rust
// src/mfa_sys.rs
#[repr(C)]
pub struct MfaContext(*mut std::ffi::c_void);

#[repr(C)]
pub struct MfaBuffer(*mut std::ffi::c_void);

pub const MFA_PRECISION_BF16: i32 = 1;
pub const MFA_SUCCESS: i32 = 0;

#[link(name = "MFAFFI")]
extern "C" {
    pub fn mfa_create_context(context: *mut MfaContext) -> i32;
    pub fn mfa_destroy_context(context: MfaContext);
    pub fn mfa_create_buffer(...) -> i32;
    pub fn mfa_buffer_from_ptr(...) -> i32;
    pub fn mfa_buffer_contents(buffer: MfaBuffer) -> *mut std::ffi::c_void;
    pub fn mfa_destroy_buffer(buffer: MfaBuffer);
    pub fn mfa_attention_forward(...) -> i32;
    pub fn mfa_is_device_supported() -> bool;
}
```

### Step 3: Safe Rust Wrapper

```rust
pub struct FlashAttention {
    context: MfaContext,
}

impl FlashAttention {
    pub fn new() -> Result<Self, MfaError> {
        if !unsafe { mfa_is_device_supported() } {
            return Err(MfaError::DeviceNotSupported);
        }
        let mut context = MfaContext(std::ptr::null_mut());
        let err = unsafe { mfa_create_context(&mut context) };
        if err != MFA_SUCCESS { return Err(MfaError::from_code(err)); }
        Ok(Self { context })
    }

    pub fn forward(&self, q: &Array, k: &Array, v: &Array,
        num_heads: u32, head_dim: u16, scale: f32) -> Result<Array, MfaError> {
        // Implementation: convert MLX arrays to MFA buffers,
        // call mfa_attention_forward, convert back
        todo!()
    }
}

impl Drop for FlashAttention {
    fn drop(&mut self) {
        unsafe { mfa_destroy_context(self.context) };
    }
}
```

## Challenges

### 1. MLX Metal Buffer Access
MLX doesn't expose the underlying MTLBuffer directly.

### 2. Memory Layout
MFA expects specific tensor layouts.

### 3. Synchronization
MLX uses lazy evaluation. Must ensure arrays are evaluated before MFA access.

### 4. Build Complexity
Need to build Swift package and link against MFAFFI library.

## Expected Performance

Based on Draw Things benchmarks:
- 20-25% faster attention on M3/M4
- 43-120% faster for full image generation

For Qwen-Image:
- Current: ~3.7s per step
- Expected: ~3.0s per step (20% improvement)
