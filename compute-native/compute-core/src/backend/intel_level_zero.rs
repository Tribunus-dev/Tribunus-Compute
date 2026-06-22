//! Intel Level Zero backend — [`TensorBackend`] implementation for Intel iGPU.
//!
//! This backend targets Intel integrated GPUs on Linux via the oneAPI
//! Level Zero API.  In the initial stub implementation (compiled on any
//! platform with `feature = "intel"`), all compute operations return
//! errors.  Tensor creation, shape inspection, and reference-counted
//! lifecycle use a generational slot-map identical to
//! [`CandleCpuBackend`](crate::candle_cpu_backend::CandleCpuBackend).
//!
//! # Feature gate
//!
//! This module is compiled only when `feature = "intel"` is enabled
//! (see `Cargo.toml`).

use std::time::Instant;

use crate::backend::DType as BackendDType;
use crate::backend::{
    BackendCapabilities, EvaluationReceipt, MatmulOp, QuantizedMatmulOp, QuantizedWeightHandle,
    ReadbackReceipt, RmsNormOp, RoPEOp, TensorBackend, TensorHandle,
};

// ── Intel Level Zero backend ───────────────────────────────────────────────

/// Intel iGPU compute backend using oneAPI Level Zero.
///
/// Uses generational slot-maps identical to [`CandleCpuBackend`] and
/// [`MlxBackend`].  In stub mode, tensor storage and lifecycle are backed
/// by `Vec<f32>` arrays; compute operations return errors.
///
/// When Level Zero FFI is wired in (Phase 2), tensors will live in
/// USM (Unified Shared Memory) allocations accessible from both CPU and
/// the Intel iGPU, enabling zero-copy dispatch.
pub struct IntelLevelZeroBackend {
    /// Slot-map storage — each live tensor is a `Vec<f32>`.
    storages: Vec<Option<Vec<f32>>>,
    /// Per-slot shapes stored as flat dims.
    shapes: Vec<Option<Vec<i32>>>,
    /// Per-slot dtypes.
    dtypes: Vec<Option<BackendDType>>,
    /// Per-slot generation counter; bumped on release.
    generations: Vec<u32>,
    /// Recycled slot indices.
    free_list: Vec<usize>,
    /// Backend display name.
    name: String,
}

impl IntelLevelZeroBackend {
    /// Create a new empty backend with default name.
    pub fn new() -> Self {
        Self {
            storages: Vec::new(),
            shapes: Vec::new(),
            dtypes: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            name: "intel-level-zero".to_string(),
        }
    }

    /// Create a new backend with a custom name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            storages: Vec::new(),
            shapes: Vec::new(),
            dtypes: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            name: name.into(),
        }
    }

    /// Allocate a slot for tensor data and return the generational handle.
    fn alloc_slot(&mut self, data: Vec<f32>, shape: Vec<i32>, dtype: BackendDType) -> TensorHandle {
        if let Some(idx) = self.free_list.pop() {
            self.generations[idx] += 1;
            self.storages[idx] = Some(data);
            self.shapes[idx] = Some(shape);
            self.dtypes[idx] = Some(dtype);
            TensorHandle {
                slot: idx as u32,
                generation: self.generations[idx],
            }
        } else {
            let idx = self.storages.len();
            self.storages.push(Some(data));
            self.shapes.push(Some(shape));
            self.dtypes.push(Some(dtype));
            self.generations.push(1);
            TensorHandle {
                slot: idx as u32,
                generation: 1,
            }
        }
    }

    /// Get an immutable reference to the tensor data at `handle`, validating
    /// slot and generation.
    fn get_storage(&self, handle: TensorHandle) -> Result<&[f32], String> {
        let slot = handle.slot as usize;
        let generation = handle.generation;
        match self.storages.get(slot) {
            Some(Some(data)) if generation == self.generations[slot] => Ok(data.as_slice()),
            _ => Err(format!(
                "IntelLevelZero: invalid tensor handle (slot={}, gen={})",
                slot, generation,
            )),
        }
    }

    /// Get an immutable reference to the shape at `handle`.
    fn get_shape(&self, handle: TensorHandle) -> Result<&[i32], String> {
        let slot = handle.slot as usize;
        let generation = handle.generation;
        match self.shapes.get(slot) {
            Some(Some(shape)) if generation == self.generations[slot] => Ok(shape.as_slice()),
            _ => Err(format!(
                "IntelLevelZero: invalid tensor handle (slot={}, gen={})",
                slot, generation,
            )),
        }
    }
}

impl Default for IntelLevelZeroBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorBackend for IntelLevelZeroBackend {
    // ── Creation ───────────────────────────────────────────────────────

    fn create_f32(&mut self, data: &[f32], shape: &[i32]) -> Result<TensorHandle, String> {
        Ok(self.alloc_slot(data.to_vec(), shape.to_vec(), BackendDType::F32))
    }

    fn create_u32(&mut self, data: &[u32], shape: &[i32]) -> Result<TensorHandle, String> {
        // Store u32 bits as f32 for uniform storage (same approach as
        // create_f32_from_bf16_bits in the reference backends).
        let f32_data: Vec<f32> = data.iter().map(|&v| f32::from_bits(v)).collect();
        Ok(self.alloc_slot(f32_data, shape.to_vec(), BackendDType::U32))
    }

    fn create_f32_from_bf16_bits(
        &mut self,
        data: &[u16],
        shape: &[i32],
    ) -> Result<TensorHandle, String> {
        // Convert BF16 bits (stored as u16) to f32.
        let f32_vec: Vec<f32> = data
            .iter()
            .map(|&v| {
                let bits = (v as u32) << 16;
                f32::from_bits(bits)
            })
            .collect();
        Ok(self.alloc_slot(f32_vec, shape.to_vec(), BackendDType::F32))
    }

    fn create_owned_from_bytes(
        &mut self,
        data: &[u8],
        shape: &[i32],
        dtype: BackendDType,
    ) -> Result<TensorHandle, String> {
        match dtype {
            BackendDType::F32 => {
                let (prefix, aligned, suffix) = unsafe { data.align_to::<f32>() };
                if !prefix.is_empty() || !suffix.is_empty() {
                    return Err("create_owned_from_bytes: f32 data not aligned to 4 bytes".into());
                }
                Ok(self.alloc_slot(aligned.to_vec(), shape.to_vec(), dtype))
            }
            BackendDType::U32 => {
                let (prefix, aligned, suffix) = unsafe { data.align_to::<u32>() };
                if !prefix.is_empty() || !suffix.is_empty() {
                    return Err("create_owned_from_bytes: u32 data not aligned to 4 bytes".into());
                }
                let f32_data: Vec<f32> = aligned.iter().map(|&v| f32::from_bits(v)).collect();
                Ok(self.alloc_slot(f32_data, shape.to_vec(), dtype))
            }
            _ => {
                return Err(format!(
                    "create_owned_from_bytes: dtype {dtype:?} is not physically supported; \
                     use create_f32_from_bf16_bits for BF16 data",
                ));
            }
        }
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
        Err("IntelLevelZero: quantized_matmul not implemented in stub mode".into())
    }

    fn matmul(
        &mut self,
        _op: &MatmulOp,
        _a: TensorHandle,
        _b: TensorHandle,
    ) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: matmul not implemented in stub mode".into())
    }

    fn rms_norm(
        &mut self,
        _op: &RmsNormOp,
        _x: TensorHandle,
        _weight: TensorHandle,
    ) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: rms_norm not implemented in stub mode".into())
    }

    fn rope(&mut self, _op: &RoPEOp, _x: TensorHandle) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: rope not implemented in stub mode".into())
    }

    fn add(&mut self, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: add not implemented in stub mode".into())
    }

    fn multiply(&mut self, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: multiply not implemented in stub mode".into())
    }

    fn silu(&mut self, _x: TensorHandle) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: silu not implemented in stub mode".into())
    }

    fn transpose(&mut self, _x: TensorHandle, _dims: &[i32]) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: transpose not implemented in stub mode".into())
    }

    fn reshape(&mut self, _x: TensorHandle, _shape: &[i32]) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: reshape not implemented in stub mode".into())
    }

    fn softmax(&mut self, _x: TensorHandle, _axis: i32) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: softmax not implemented in stub mode".into())
    }

    fn index_select(
        &mut self,
        _x: TensorHandle,
        _indices: &[u32],
        _axis: i32,
    ) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: index_select not implemented in stub mode".into())
    }

    // ── Missing ops (stubs) ────────────────────────────────────────────

    fn concatenate(
        &mut self,
        _tensors: &[TensorHandle],
        _axis: i32,
    ) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: concatenate not implemented in stub mode".into())
    }

    fn slice(
        &mut self,
        _x: TensorHandle,
        _start: &[i32],
        _stop: &[i32],
        _step: &[i32],
    ) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: slice not implemented in stub mode".into())
    }

    fn cast(&mut self, _x: TensorHandle, _dtype: BackendDType) -> Result<TensorHandle, String> {
        Err("IntelLevelZero: cast not implemented in stub mode".into())
    }

    // ── Lifecycle / inspection ─────────────────────────────────────────

    fn evaluate(
        &mut self,
        group_id: u64,
        outputs: &[TensorHandle],
    ) -> Result<EvaluationReceipt, String> {
        let start = Instant::now();

        // Validate handles exist.
        for &h in outputs {
            let _ = self.get_storage(h)?;
        }

        let elapsed = start.elapsed();

        Ok(EvaluationReceipt {
            group_id,
            graph_build_ns: 0,
            submit_ns: 0,
            sync_ns: elapsed.as_nanos() as u64,
            output_count: outputs.len(),
            active_memory_after: 0,
            cache_memory_after: 0,
            observed_substrate: Some("intel-level-zero-stub".to_string()),
            eval_calls: outputs.len().max(1),
        })
    }

    fn read_f32(&mut self, handle: TensorHandle) -> Result<ReadbackReceipt, String> {
        let start = Instant::now();
        let data = self.get_storage(handle)?.to_vec();
        let elapsed = start.elapsed();
        Ok(ReadbackReceipt {
            data,
            forced_eval: false,
            sync_ns: elapsed.as_nanos() as u64,
            observed_substrate: Some("intel-level-zero-stub".to_string()),
        })
    }

    fn shape(&self, handle: TensorHandle) -> Result<Vec<i32>, String> {
        Ok(self.get_shape(handle)?.to_vec())
    }

    fn release(&mut self, handle: TensorHandle) -> Result<(), String> {
        let slot = handle.slot as usize;
        let generation = handle.generation;

        if slot >= self.storages.len() {
            return Err(format!(
                "release: invalid handle (slot={}, gen={})",
                slot, generation
            ));
        }
        let current_gen = self.generations[slot];
        if generation != current_gen {
            return Err(format!(
                "release: stale handle (slot={}, gen={}, current={})",
                slot, generation, current_gen,
            ));
        }
        if self.storages[slot].is_none() {
            return Err(format!(
                "release: handle already released (slot={}, gen={})",
                slot, generation
            ));
        }

        self.storages[slot] = None;
        self.shapes[slot] = None;
        self.dtypes[slot] = None;
        self.generations[slot] += 1;
        self.free_list.push(slot);
        Ok(())
    }

    fn active_memory(&self) -> (u64, u64) {
        let active: u64 = self
            .storages
            .iter()
            .flatten()
            .map(|d| (d.len() * 4) as u64)
            .sum();
        (active, 0)
    }

    fn backend_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            can_gpu: true,
            can_cpu: true,
            supports_quantized: false,
            supports_bf16_native: false,
            backend_name: self.name.clone(),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        DType as BackendDType, MatmulOp, QuantizedMatmulOp, RmsNormOp, RoPEOp, TensorBackend,
        TensorHandle,
    };

    /// Helper to create an f32 TensorHandle with a given shape.
    fn make_f32(backend: &mut IntelLevelZeroBackend, data: &[f32], shape: &[i32]) -> TensorHandle {
        backend.create_f32(data, shape).unwrap()
    }

    /// Helper to read back an f32 tensor handle.
    fn read_f32(backend: &mut IntelLevelZeroBackend, h: TensorHandle) -> Vec<f32> {
        backend.read_f32(h).unwrap().data
    }

    #[test]
    fn test_create_and_read_f32() {
        let mut backend = IntelLevelZeroBackend::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let h = backend.create_f32(&data, &[2, 2]).unwrap();
        let output = read_f32(&mut backend, h);
        assert_eq!(output, data);
    }

    #[test]
    fn test_create_u32() {
        let mut backend = IntelLevelZeroBackend::new();
        let data: Vec<u32> = vec![1, 2, 3];
        let h = backend.create_u32(&data, &[3]).unwrap();
        let output = read_f32(&mut backend, h);
        // u32 bits stored as f32::from_bits
        let expected: Vec<f32> = data.iter().map(|&v| f32::from_bits(v)).collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_create_f32_from_bf16_bits() {
        let mut backend = IntelLevelZeroBackend::new();
        // BF16(1.0) = 0x3F80 as u16 bits
        // BF16(2.0) = 0x4000 as u16 bits
        let bf16_bits: Vec<u16> = vec![0x3F80, 0x4000];
        let h = backend.create_f32_from_bf16_bits(&bf16_bits, &[2]).unwrap();
        let output = read_f32(&mut backend, h);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_create_owned_from_bytes_f32() {
        let mut backend = IntelLevelZeroBackend::new();
        let input: Vec<f32> = vec![1.5, 2.5, 3.5];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        let h = backend
            .create_owned_from_bytes(bytes, &[3], BackendDType::F32)
            .unwrap();
        let output = read_f32(&mut backend, h);
        assert_eq!(output, input);
    }

    #[test]
    fn test_shape() {
        let mut backend = IntelLevelZeroBackend::new();
        let h = make_f32(&mut backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(backend.shape(h).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_backend_capabilities() {
        let backend = IntelLevelZeroBackend::new();
        let caps = backend.backend_capabilities();
        assert_eq!(caps.can_gpu, true);
        assert_eq!(caps.can_cpu, true);
        assert_eq!(caps.supports_quantized, false);
        assert_eq!(caps.supports_bf16_native, false);
        assert_eq!(caps.backend_name, "intel-level-zero");
    }

    #[test]
    fn test_release() {
        let mut backend = IntelLevelZeroBackend::new();
        let h = make_f32(&mut backend, &[1.0, 2.0], &[2]);
        backend.release(h).unwrap();
        // Releasing again should fail.
        assert!(backend.release(h).is_err());
        // Using the released handle should fail.
        assert!(backend.read_f32(h).is_err());
    }

    #[test]
    fn test_evaluate() {
        let mut backend = IntelLevelZeroBackend::new();
        let h = make_f32(&mut backend, &[10.0, 20.0], &[2]);
        let receipt = backend.evaluate(42, &[h]).unwrap();
        assert_eq!(receipt.group_id, 42);
        assert_eq!(receipt.output_count, 1);
        assert_eq!(
            receipt.observed_substrate.as_deref(),
            Some("intel-level-zero-stub")
        );
    }

    #[test]
    fn test_active_memory() {
        let mut backend = IntelLevelZeroBackend::new();
        assert_eq!(backend.active_memory(), (0, 0));

        let _h = make_f32(&mut backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // 4 elements * 4 bytes = 16
        assert_eq!(backend.active_memory(), (16, 0));
    }

    #[test]
    fn test_compute_ops_return_error_in_stub_mode() {
        let mut backend = IntelLevelZeroBackend::new();
        let a = make_f32(&mut backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_f32(&mut backend, &[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let op = MatmulOp { m: 2, n: 2, k: 2 };
        assert!(backend.matmul(&op, a, b).is_err());
        assert!(backend.add(a, b).is_err());
        assert!(backend.multiply(a, b).is_err());
    }
}
