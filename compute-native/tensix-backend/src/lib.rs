//! TensixBackend — Tenstorrent Tensix execution backend.
//!
//! Implements [`TensorBackend`] via the [`tensix-ffi`] FFI bridge to
//! Metalium/TTNN.
//!
//! Feature gate: feature = `"tensix-backend"` set at the compute-core level.
//! On systems without tt-metal, the mock mode (feature = "mock") provides
//! host-memory-only stubs.

#![allow(dead_code)]

use std::collections::HashMap;

use tensix_ffi::*;
use tribunus_compute_core::backend::{
    BackendCapabilities, DType, EvaluationReceipt, MatmulOp, QuantizedMatmulOp,
    QuantizedWeightHandle, ReadbackReceipt, RmsNormOp, RoPEOp, TensorBackend, TensorHandle,
};

// ============================================================================
// Helpers
// ============================================================================

/// Number of bytes per element for a DType.
fn dtype_size(dt: DType) -> usize {
    match dt {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::U32 | DType::I32 => 4,
        DType::U8 | DType::I8 => 1,
    }
}

// ============================================================================
// Slot types
// ============================================================================

/// A generational slot-map entry for a Tensix device (or host-mock) buffer.
struct TensixSlot {
    /// Opaque device buffer handle, or null for mock-mode fallback.
    buffer: *mut TensixBuffer,
    shape: Vec<i32>,
    dtype: DType,
    generation: u32,
    /// Host-side copy used when `buffer` is null (mock mode).
    host_data: Option<Vec<u8>>,
}

// TensixSlot is thread-safe because the FFI handles are only used from
// single-threaded callers, and `*mut TensixBuffer` is Send+Sync despite
// the raw pointer.
unsafe impl Send for TensixSlot {}
unsafe impl Sync for TensixSlot {}

/// A generational slot-map entry for a quantized weight stored in host memory
/// (Tensix does not yet support native quantized matmul).
struct TensixWeightSlot {
    data: Vec<u8>,
    shape: Vec<i32>,
    dtype: DType,
    generation: u32,
}

// ============================================================================
// Backend
// ============================================================================

/// Tenstorrent Tensix backend.
///
/// Holds an open Tensix device.  When `device` is null (constructed via
/// [`new_mock`](TensixBackend::new_mock)) the backend stores data in host
/// memory and returns errors for device-only compute operations.
pub struct TensixBackend {
    device: *mut TensixDevice,
    device_id: i32,
    slots: Vec<Option<TensixSlot>>,
    generations: Vec<u32>,
    free_list: Vec<usize>,
    weight_slots: Vec<Option<TensixWeightSlot>>,
    weight_generations: Vec<u32>,
    weight_free_list: Vec<usize>,
    programs: HashMap<u64, *mut TensixProgram>,
    name: String,
}

// Safety: all unsafe FFI calls are serialised through `&mut self`.
unsafe impl Send for TensixBackend {}
unsafe impl Sync for TensixBackend {}

impl TensixBackend {
    /// Open the Tensix device at `device_id` and return a real backend.
    ///
    /// Returns `Err` when the device cannot be opened (hardware unavailable
    /// or built in stub mode).
    pub fn new(device_id: i32) -> Result<Self, String> {
        let mut err = TensixError::ok();
        let device = unsafe { tensix_open_device(device_id, &mut err) };
        if device.is_null() || !err.is_ok() {
            return Err(format!(
                "TensixBackend::new({}): device open failed — {}",
                device_id,
                err.to_string(),
            ));
        }
        Ok(Self {
            device,
            device_id,
            slots: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            weight_slots: Vec::new(),
            weight_generations: Vec::new(),
            weight_free_list: Vec::new(),
            programs: HashMap::new(),
            name: format!("tensix-{}", device_id),
        })
    }

    /// Build a mock backend with no real device — host memory only.
    pub fn new_mock() -> Self {
        Self {
            device: std::ptr::null_mut(),
            device_id: -1,
            slots: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            weight_slots: Vec::new(),
            weight_generations: Vec::new(),
            weight_free_list: Vec::new(),
            programs: HashMap::new(),
            name: "tensix-mock".into(),
        }
    }

    // ── Slot-map helpers ──────────────────────────────────────────────

    fn alloc_handle(
        &mut self,
        buffer: *mut TensixBuffer,
        shape: Vec<i32>,
        dtype: DType,
        host_data: Option<Vec<u8>>,
    ) -> TensorHandle {
        if let Some(idx) = self.free_list.pop() {
            let gen = self.generations[idx];
            self.slots[idx] = Some(TensixSlot { buffer, shape, dtype, generation: gen, host_data });
            TensorHandle { slot: idx as u32, generation: gen }
        } else {
            let gen = 1u32;
            let idx = self.slots.len();
            self.slots.push(Some(TensixSlot {
                buffer,
                shape,
                dtype,
                generation: gen,
                host_data,
            }));
            self.generations.push(gen);
            TensorHandle { slot: idx as u32, generation: gen }
        }
    }

    fn get_slot(&self, handle: TensorHandle) -> Result<&TensixSlot, String> {
        let slot = handle.slot as usize;
        let gen = handle.generation;
        match self.slots.get(slot) {
            Some(Some(s)) if s.generation == gen => Ok(s),
            _ => Err(format!(
                "TensixBackend: invalid handle (slot={}, gen={})",
                slot, gen,
            )),
        }
    }

    fn get_slot_mut(&mut self, handle: TensorHandle) -> Result<&mut TensixSlot, String> {
        let slot = handle.slot as usize;
        let gen = handle.generation;
        match self.slots.get_mut(slot) {
            Some(Some(s)) if s.generation == gen => Ok(s),
            _ => Err(format!(
                "TensixBackend: invalid handle (slot={}, gen={})",
                slot, gen,
            )),
        }
    }

    fn release_slot(&mut self, handle: TensorHandle) -> Result<(), String> {
        let slot = handle.slot as usize;
        let gen = handle.generation;

        if slot >= self.slots.len() {
            return Err(format!(
                "release: invalid handle (slot={}, gen={})",
                slot, gen,
            ));
        }
        let current = self.generations[slot];
        if gen != current {
            return Err(format!(
                "release: stale handle (slot={}, gen={}, current={})",
                slot, gen, current,
            ));
        }
        if self.slots[slot].is_none() {
            return Err(format!(
                "release: handle already released (slot={}, gen={})",
                slot, gen,
            ));
        }

        // Deallocate device buffer if present.
        if let Some(s) = &self.slots[slot] {
            if !s.buffer.is_null() {
                unsafe { tensix_deallocate_buffer(s.buffer); }
            }
        }
        self.slots[slot] = None;
        self.generations[slot] += 1;
        self.free_list.push(slot);
        Ok(())
    }

    // ── Weight slot-map helpers ───────────────────────────────────────

    fn alloc_weight_handle(&mut self, data: Vec<u8>, shape: Vec<i32>, dtype: DType) -> QuantizedWeightHandle {
        if let Some(idx) = self.weight_free_list.pop() {
            let gen = self.weight_generations[idx];
            self.weight_slots[idx] = Some(TensixWeightSlot { data, shape, dtype, generation: gen });
            QuantizedWeightHandle { slot: idx as u32, generation: gen }
        } else {
            let gen = 1u32;
            let idx = self.weight_slots.len();
            self.weight_slots.push(Some(TensixWeightSlot { data, shape, dtype, generation: gen }));
            self.weight_generations.push(gen);
            QuantizedWeightHandle { slot: idx as u32, generation: gen }
        }
    }

    fn get_weight_slot(&self, handle: QuantizedWeightHandle) -> Result<&TensixWeightSlot, String> {
        let slot = handle.slot as usize;
        let gen = handle.generation;
        match self.weight_slots.get(slot) {
            Some(Some(s)) if s.generation == gen => Ok(s),
            _ => Err(format!(
                "TensixBackend: invalid weight handle (slot={}, gen={})",
                slot, gen,
            )),
        }
    }

    // ── FFI helpers ───────────────────────────────────────────────────

    /// Call `tensix_allocate_buffer` and return the buffer pointer.
    fn allocate_device_buffer(&self, bytes: u64, mem_type: TensixMemoryType) -> Result<*mut TensixBuffer, String> {
        if self.device.is_null() {
            return Err("TensixBackend: device is null (mock mode)".into());
        }
        let mut err = TensixError::ok();
        let buf = unsafe { tensix_allocate_buffer(self.device, bytes, mem_type, &mut err) };
        if buf.is_null() || !err.is_ok() {
            return Err(format!("tensix_allocate_buffer failed: {}", err.to_string()));
        }
        Ok(buf)
    }

    /// Write f32 data to a device buffer.
    fn write_to_device_buffer(&self, buf: *mut TensixBuffer, data: &[f32]) -> Result<(), String> {
        if self.device.is_null() {
            return Err("TensixBackend: device is null (mock mode)".into());
        }
        let mut err = TensixError::ok();
        unsafe {
            tensix_write_to_buffer(buf, data.as_ptr(), data.len() as u64, 0, &mut err);
        }
        if !err.is_ok() {
            return Err(format!("tensix_write_to_buffer failed: {}", err.to_string()));
        }
        Ok(())
    }

    /// Write raw bytes wrapped as f32 slices (matching the FFI signature that
    /// takes `*const f32` even for non-f32 buffers).
    fn write_bytes_to_device_buffer(&self, buf: *mut TensixBuffer, bytes: &[u8]) -> Result<(), String> {
        if self.device.is_null() {
            return Err("TensixBackend: device is null (mock mode)".into());
        }
        // The FFI function takes *const f32; we reinterpret the byte slice.
        let elem_count = bytes.len() / 4; // number of f32 "slots"
        let data_ptr = bytes.as_ptr() as *const f32;
        let mut err = TensixError::ok();
        unsafe {
            tensix_write_to_buffer(buf, data_ptr, elem_count as u64, 0, &mut err);
        }
        if !err.is_ok() {
            return Err(format!("write_bytes_to_device_buffer failed: {}", err.to_string()));
        }
        Ok(())
    }

    /// Read f32 data from a device buffer.
    fn read_from_device_buffer(&self, buf: *mut TensixBuffer, count: usize) -> Result<Vec<f32>, String> {
        if self.device.is_null() {
            return Err("TensixBackend: device is null (mock mode)".into());
        }
        let mut data = vec![0.0f32; count];
        let mut err = TensixError::ok();
        unsafe {
            tensix_read_from_buffer(data.as_mut_ptr(), buf, count as u64, 0, &mut err);
        }
        if !err.is_ok() {
            return Err(format!("tensix_read_from_buffer failed: {}", err.to_string()));
        }
        Ok(data)
    }

    /// True when this backend is backed by a real device.
    fn is_real(&self) -> bool {
        !self.device.is_null()
    }

    /// Return a conservative estimate of active bytes by summing slot
    /// element counts x dtype size.
    fn estimate_active_bytes(&self) -> u64 {
        self.slots
            .iter()
            .flatten()
            .map(|s| {
                let elems: usize = s.shape.iter().map(|&d| d as usize).product();
                (elems * dtype_size(s.dtype)) as u64
            })
            .sum()
    }
}

// ============================================================================
// TensorBackend trait
// ============================================================================

impl TensorBackend for TensixBackend {
    // ── Creation ───────────────────────────────────────────────────────

    fn create_f32(&mut self, data: &[f32], shape: &[i32]) -> Result<TensorHandle, String> {
        let bytes = data.len() as u64 * 4;
        if self.is_real() {
            let buf = self.allocate_device_buffer(bytes, TensixMemoryType::Host)?;
            self.write_to_device_buffer(buf, data)?;
            Ok(self.alloc_handle(buf, shape.to_vec(), DType::F32, None))
        } else {
            // Mock mode: store raw bytes in host_data.
            let bytes: Vec<u8> = data
                .iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();
            Ok(self.alloc_handle(
                std::ptr::null_mut(),
                shape.to_vec(),
                DType::F32,
                Some(bytes),
            ))
        }
    }

    fn create_u32(&mut self, data: &[u32], shape: &[i32]) -> Result<TensorHandle, String> {
        let bytes = data.len() as u64 * 4;
        if self.is_real() {
            let buf = self.allocate_device_buffer(bytes, TensixMemoryType::Host)?;
            // Reinterpret u32 slice as f32 for the FFI write call.
            let f32_ptr = data.as_ptr() as *const f32;
            let f32_slice = unsafe { std::slice::from_raw_parts(f32_ptr, data.len()) };
            self.write_to_device_buffer(buf, f32_slice)?;
            Ok(self.alloc_handle(buf, shape.to_vec(), DType::U32, None))
        } else {
            let bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();
            Ok(self.alloc_handle(
                std::ptr::null_mut(),
                shape.to_vec(),
                DType::U32,
                Some(bytes),
            ))
        }
    }

    fn create_f32_from_bf16_bits(
        &mut self,
        data: &[u16],
        shape: &[i32],
    ) -> Result<TensorHandle, String> {
        // Convert BF16 bits to F32, store as F32 on device.
        let f32_vec: Vec<f32> = data
            .iter()
            .map(|&v| {
                let bits = (v as u32) << 16;
                f32::from_bits(bits)
            })
            .collect();
        self.create_f32(&f32_vec, shape)
    }

    fn create_owned_from_bytes(
        &mut self,
        data: &[u8],
        shape: &[i32],
        dtype: DType,
    ) -> Result<TensorHandle, String> {
        match dtype {
            DType::F32 => {
                let (prefix, aligned, suffix) = unsafe { data.align_to::<f32>() };
                if !prefix.is_empty() || !suffix.is_empty() {
                    return Err(
                        "create_owned_from_bytes: F32 data not aligned to 4 bytes".into(),
                    );
                }
                self.create_f32(aligned, shape)
            }
            DType::U32 => {
                let (prefix, aligned, suffix) = unsafe { data.align_to::<u32>() };
                if !prefix.is_empty() || !suffix.is_empty() {
                    return Err(
                        "create_owned_from_bytes: U32 data not aligned to 4 bytes".into(),
                    );
                }
                self.create_u32(aligned, shape)
            }
            _ => Err(format!(
                "create_owned_from_bytes: dtype {dtype:?} is not physically supported; \
                 use create_f32_from_bf16_bits for BF16 data",
            )),
        }
    }

    fn bind_external(
        &mut self,
        _owner_token: u64,
        _data: &[u8],
        _shape: &[i32],
        _dtype: DType,
    ) -> Result<TensorHandle, String> {
        Err("bind_external: not implemented for TensixBackend".into())
    }

    // ── Core compute ───────────────────────────────────────────────────

    fn quantized_matmul(
        &mut self,
        _op: &QuantizedMatmulOp,
        _x: TensorHandle,
        _w: QuantizedWeightHandle,
        _scales: TensorHandle,
        _biases: TensorHandle,
    ) -> Result<TensorHandle, String> {
        Err("quantized_matmul: not yet implemented for TensixBackend".into())
    }

    fn matmul(
        &mut self,
        op: &MatmulOp,
        a: TensorHandle,
        b: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if !self.is_real() {
            return Err("matmul: mock backend (no device)".into());
        }
        let slot_a = self.get_slot(a)?;
        let slot_b = self.get_slot(b)?;

        // Validate shapes match the op descriptor.
        let a_shape = &slot_a.shape;
        let b_shape = &slot_b.shape;
        let a_m = if a_shape.len() >= 2 { a_shape[a_shape.len() - 2] as u32 } else { 1 };
        let a_k = *a_shape.last().unwrap_or(&0) as u32;
        let b_k = if b_shape.len() >= 2 { b_shape[b_shape.len() - 2] as u32 } else { b_shape[0] as u32 };
        let b_n = *b_shape.last().unwrap_or(&0) as u32;

        if a_m != op.m {
            return Err(format!("matmul: A.M={} != op.m={}", a_m, op.m));
        }
        if a_k != op.k || b_k != op.k {
            return Err(format!(
                "matmul: K mismatch (A.K={}, B.K={}, op.k={})",
                a_k, b_k, op.k,
            ));
        }
        if b_n != op.n {
            return Err(format!("matmul: B.N={} != op.n={}", b_n, op.n));
        }

        // Allocate output buffer.
        let out_bytes = (op.m as u64) * (op.n as u64) * 4; // f32
        let c = self.allocate_device_buffer(out_bytes, TensixMemoryType::DRAM)?;

        let params = TensixMatmulParams {
            M: op.m,
            N: op.n,
            K: op.k,
            transpose_a: 0,
            transpose_b: 0,
            dtype: 0, // f32
        };
        let mut err = TensixError::ok();
        unsafe {
            tensix_matmul(
                self.device,
                slot_a.buffer,
                slot_b.buffer,
                c,
                &params,
                &mut err,
            );
        }
        if !err.is_ok() {
            return Err(format!("matmul failed: {}", err.to_string()));
        }

        let out_shape = vec![op.m as i32, op.n as i32];
        Ok(self.alloc_handle(c, out_shape, DType::F32, None))
    }

    fn rms_norm(
        &mut self,
        op: &RmsNormOp,
        x: TensorHandle,
        weight: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if !self.is_real() {
            return Err("rms_norm: mock backend (no device)".into());
        }
        let slot_x = self.get_slot(x)?;
        let slot_w = self.get_slot(weight)?;

        // Allocate output buffer: same shape as x, dtype f32.
        let elems: usize = slot_x.shape.iter().map(|&d| d as usize).product();
        let out_bytes = (elems as u64) * 4;
        let out = self.allocate_device_buffer(out_bytes, TensixMemoryType::DRAM)?;

        let params = TensixNormParams {
            dim: op.dim,
            eps: op.eps,
            dtype: 0, // f32
        };
        let mut err = TensixError::ok();
        unsafe {
            tensix_rms_norm(
                self.device,
                slot_x.buffer,
                slot_w.buffer,
                out,
                &params,
                &mut err,
            );
        }
        if !err.is_ok() {
            return Err(format!("rms_norm failed: {}", err.to_string()));
        }

        Ok(self.alloc_handle(out, slot_x.shape.clone(), DType::F32, None))
    }

    fn rope(&mut self, op: &RoPEOp, x: TensorHandle) -> Result<TensorHandle, String> {
        if !self.is_real() {
            return Err("rope: mock backend (no device)".into());
        }
        let slot_x = self.get_slot(x)?;

        // Allocate output buffer: same shape as x.
        let elems: usize = slot_x.shape.iter().map(|&d| d as usize).product();
        let out_bytes = (elems as u64) * 4;
        let out = self.allocate_device_buffer(out_bytes, TensixMemoryType::DRAM)?;

        let params = TensixRopeParams {
            dim: op.head_dim,
            theta: 10_000.0f32,
            max_seq_len: op.positions.len() as i32,
        };
        let mut err = TensixError::ok();
        unsafe {
            // The rope FFI takes cos and sin buffers. For now we pass null
            // and let the kernel compute them internally.
            tensix_rope(
                self.device,
                slot_x.buffer,
                std::ptr::null_mut(), // cos (computed internally)
                std::ptr::null_mut(), // sin (computed internally)
                out,
                &params,
                &mut err,
            );
        }
        if !err.is_ok() {
            return Err(format!("rope failed: {}", err.to_string()));
        }

        Ok(self.alloc_handle(out, slot_x.shape.clone(), DType::F32, None))
    }

    fn add(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String> {
        if !self.is_real() {
            return Err("add: mock backend (no device)".into());
        }
        let slot_a = self.get_slot(a)?;
        let slot_b = self.get_slot(b)?;

        // Determine output shape (broadcast).
        let out_shape = broadcast_shape(&slot_a.shape, &slot_b.shape)?;
        let elems: usize = out_shape.iter().map(|&d| d as usize).product();
        let out_bytes = (elems as u64) * 4;
        let out = self.allocate_device_buffer(out_bytes, TensixMemoryType::DRAM)?;

        let mut err = TensixError::ok();
        unsafe {
            tensix_add(
                self.device,
                slot_a.buffer,
                slot_b.buffer,
                out,
                &mut err,
            );
        }
        if !err.is_ok() {
            return Err(format!("add failed: {}", err.to_string()));
        }

        Ok(self.alloc_handle(out, out_shape, DType::F32, None))
    }

    fn multiply(&mut self, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle, String> {
        Err("multiply: not yet implemented for TensixBackend".into())
    }

    fn silu(&mut self, x: TensorHandle) -> Result<TensorHandle, String> {
        if !self.is_real() {
            return Err("silu: mock backend (no device)".into());
        }
        let slot_x = self.get_slot(x)?;

        let elems: usize = slot_x.shape.iter().map(|&d| d as usize).product();
        let out_bytes = (elems as u64) * 4;
        let out = self.allocate_device_buffer(out_bytes, TensixMemoryType::DRAM)?;

        let mut err = TensixError::ok();
        unsafe {
            tensix_silu(self.device, slot_x.buffer, out, &mut err);
        }
        if !err.is_ok() {
            return Err(format!("silu failed: {}", err.to_string()));
        }

        Ok(self.alloc_handle(out, slot_x.shape.clone(), DType::F32, None))
    }

    fn transpose(&mut self, _x: TensorHandle, _dims: &[i32]) -> Result<TensorHandle, String> {
        Err("transpose: not yet implemented for TensixBackend".into())
    }

    fn reshape(&mut self, _x: TensorHandle, _shape: &[i32]) -> Result<TensorHandle, String> {
        Err("reshape: not yet implemented for TensixBackend".into())
    }

    fn softmax(&mut self, _x: TensorHandle, _axis: i32) -> Result<TensorHandle, String> {
        Err("softmax: not yet implemented for TensixBackend".into())
    }

    fn index_select(
        &mut self,
        _x: TensorHandle,
        _indices: &[u32],
        _axis: i32,
    ) -> Result<TensorHandle, String> {
        Err("index_select: not yet implemented for TensixBackend".into())
    }

    // ── Missing ops (stubs) ────────────────────────────────────────────

    fn concatenate(
        &mut self,
        _tensors: &[TensorHandle],
        _axis: i32,
    ) -> Result<TensorHandle, String> {
        Err("concatenate: not yet implemented for TensixBackend".into())
    }

    fn slice(
        &mut self,
        _x: TensorHandle,
        _start: &[i32],
        _stop: &[i32],
        _step: &[i32],
    ) -> Result<TensorHandle, String> {
        Err("slice: not yet implemented for TensixBackend".into())
    }

    fn cast(&mut self, _x: TensorHandle, _dtype: DType) -> Result<TensorHandle, String> {
        Err("cast: not yet implemented for TensixBackend".into())
    }

    // ── Lifecycle / inspection ─────────────────────────────────────────

    fn evaluate(
        &mut self,
        group_id: u64,
        outputs: &[TensorHandle],
    ) -> Result<EvaluationReceipt, String> {
        let start = std::time::Instant::now();

        // Validate all handles exist.
        for &h in outputs {
            self.get_slot(h)?;
        }

        // Synchronise the device to ensure all queued operations complete.
        if self.is_real() {
            unsafe { tensix_synchronize_device(self.device); }
        }

        let elapsed = start.elapsed();
        let active = self.estimate_active_bytes();

        Ok(EvaluationReceipt {
            group_id,
            graph_build_ns: 0,
            submit_ns: 0,
            sync_ns: elapsed.as_nanos() as u64,
            output_count: outputs.len(),
            active_memory_after: active,
            cache_memory_after: 0,
            observed_substrate: Some(self.name.clone()),
            eval_calls: 1,
        })
    }

    fn read_f32(&mut self, handle: TensorHandle) -> Result<ReadbackReceipt, String> {
        let start = std::time::Instant::now();
        let slot = self.get_slot(handle)?;

        let data = if self.is_real() {
            // Synchronise first, then read back.
            unsafe { tensix_synchronize_device(self.device); }
            let elems: usize = slot.shape.iter().map(|&d| d as usize).product();
            self.read_from_device_buffer(slot.buffer, elems)?
        } else {
            // Mock mode: decode from stored bytes.
            match slot.host_data.as_ref() {
                Some(bytes) => {
                    bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                }
                None => return Err("read_f32: no host data in mock mode".into()),
            }
        };

        let elapsed = start.elapsed();
        Ok(ReadbackReceipt {
            data,
            forced_eval: self.is_real(), // needed sync for device reads
            sync_ns: elapsed.as_nanos() as u64,
            observed_substrate: Some(self.name.clone()),
        })
    }

    fn shape(&self, handle: TensorHandle) -> Result<Vec<i32>, String> {
        let slot = self.get_slot(handle)?;
        Ok(slot.shape.clone())
    }

    fn release(&mut self, handle: TensorHandle) -> Result<(), String> {
        self.release_slot(handle)
    }

    fn active_memory(&self) -> (u64, u64) {
        (self.estimate_active_bytes(), 0)
    }

    fn backend_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            can_gpu: true,
            can_cpu: false,
            supports_quantized: false,
            supports_bf16_native: true,
            backend_name: self.name.clone(),
        }
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Compute the broadcast shape of two input shapes.
fn broadcast_shape(a: &[i32], b: &[i32]) -> Result<Vec<i32>, String> {
    let max_rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(max_rank);
    for i in 0..max_rank {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da != db && da != 1 && db != 1 {
            return Err(format!(
                "broadcast_shape: incompatible dimensions at axis {}: {} vs {}",
                i, da, db,
            ));
        }
        out.push(da.max(db));
    }
    out.reverse();
    Ok(out)
}
