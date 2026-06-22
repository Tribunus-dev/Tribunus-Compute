//! Tensix FFI bindings — Rust interface to the Tenstorrent C API.
//! Compiled against bridge.h via build.rs -> bridge.cpp.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::CStr;
use std::fmt;

// ============================================================================
// Opaque handle types matching bridge.h
// ============================================================================

#[repr(C)]
pub struct TensixDevice {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TensixBuffer {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TensixProgram {
    _private: [u8; 0],
}

// ============================================================================
// Error type
// ============================================================================

#[repr(C)]
pub struct TensixError {
    pub message: [u8; 256],
    pub code: i32,
}

impl fmt::Debug for TensixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = CStr::from_bytes_until_nul(&self.message)
            .map(|c| c.to_string_lossy())
            .unwrap_or_else(|_| "(invalid utf8)".into());
        write!(f, "TensixError(code={}, msg={})", self.code, msg)
    }
}

impl TensixError {
    pub fn ok() -> Self {
        TensixError { message: [0u8; 256], code: 0 }
    }

    pub fn is_ok(&self) -> bool {
        self.code == 0
    }

    pub fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}

// ============================================================================
// Memory type
// ============================================================================

#[repr(C)]
pub enum TensixMemoryType {
    DRAM = 0,
    Host = 1,
    Trace = 2,
}

// ============================================================================
// Op parameter structs (matching bridge.h)
// ============================================================================

#[repr(C)]
pub struct TensixMatmulParams {
    pub M: u32,
    pub N: u32,
    pub K: u32,
    pub transpose_a: u8,
    pub transpose_b: u8,
    pub dtype: u8, // 0=f32, 1=f16, 2=bf16
}

#[repr(C)]
pub struct TensixSdpaParams {
    pub batch: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub seq_len_q: u32,
    pub seq_len_kv: u32,
    pub scale: f32,
    pub dtype: u8,
}

#[repr(C)]
pub struct TensixNormParams {
    pub dim: u32,
    pub eps: f32,
    pub dtype: u8,
}

#[repr(C)]
pub struct TensixRopeParams {
    pub dim: u32,
    pub theta: f32,
    pub max_seq_len: i32,
}

#[repr(C)]
pub struct TensixProfileEvent {
    pub kernel_cycles: u64,
    pub sync_ns: u64,
    pub dram_bytes_read: u64,
    pub dram_bytes_written: u64,
    pub core_count: u32,
    pub cb_occupancy: f32,
    pub noc_utilization: f32,
}

// ============================================================================
// extern "C" declarations
// ============================================================================

extern "C" {
    // Device
    pub fn tensix_open_device(device_id: i32, err: *mut TensixError) -> *mut TensixDevice;
    pub fn tensix_close_device(dev: *mut TensixDevice);
    pub fn tensix_device_core_count(dev: *mut TensixDevice) -> i32;
    pub fn tensix_device_arch(dev: *mut TensixDevice, out: *mut u8, out_len: usize);

    // Buffer
    pub fn tensix_allocate_buffer(
        dev: *mut TensixDevice, bytes: u64, mem_type: TensixMemoryType,
        err: *mut TensixError) -> *mut TensixBuffer;
    pub fn tensix_deallocate_buffer(buf: *mut TensixBuffer);
    pub fn tensix_write_to_buffer(
        buf: *mut TensixBuffer, data: *const f32, count: u64,
        offset: u64, err: *mut TensixError);
    pub fn tensix_read_from_buffer(
        data: *mut f32, buf: *mut TensixBuffer, count: u64,
        offset: u64, err: *mut TensixError);

    // Compute ops
    pub fn tensix_matmul(
        dev: *mut TensixDevice, a: *mut TensixBuffer, b: *mut TensixBuffer,
        c: *mut TensixBuffer, params: *const TensixMatmulParams, err: *mut TensixError);
    pub fn tensix_sdpa(
        dev: *mut TensixDevice, q: *mut TensixBuffer, k: *mut TensixBuffer,
        v: *mut TensixBuffer, out: *mut TensixBuffer,
        params: *const TensixSdpaParams, err: *mut TensixError);
    pub fn tensix_rms_norm(
        dev: *mut TensixDevice, x: *mut TensixBuffer, weight: *mut TensixBuffer,
        out: *mut TensixBuffer, params: *const TensixNormParams, err: *mut TensixError);
    pub fn tensix_rope(
        dev: *mut TensixDevice, x: *mut TensixBuffer, cos: *mut TensixBuffer,
        sin: *mut TensixBuffer, out: *mut TensixBuffer,
        params: *const TensixRopeParams, err: *mut TensixError);
    pub fn tensix_silu(
        dev: *mut TensixDevice, x: *mut TensixBuffer,
        out: *mut TensixBuffer, err: *mut TensixError);
    pub fn tensix_add(
        dev: *mut TensixDevice, a: *mut TensixBuffer, b: *mut TensixBuffer,
        out: *mut TensixBuffer, err: *mut TensixError);

    // Program compilation
    pub fn tensix_compile_program(
        dev: *mut TensixDevice, program_spec_json: *const u8,
        json_len: usize, err: *mut TensixError) -> *mut TensixProgram;
    pub fn tensix_execute_program(
        dev: *mut TensixDevice, prog: *mut TensixProgram, err: *mut TensixError);
    pub fn tensix_free_program(prog: *mut TensixProgram);

    // Sync
    pub fn tensix_synchronize_device(dev: *mut TensixDevice);

    // Profiling
    pub fn tensix_read_profiler(
        dev: *mut TensixDevice, event: *mut TensixProfileEvent, err: *mut TensixError);
    pub fn tensix_reset_profiler(dev: *mut TensixDevice);
}
