//! Accelerate CPU backend — vDSP, BLAS, BNNS, vForce.
//!
//! F32 matmul wired via cblas_sgemm.  All other primitives return
//! "not yet implemented" until native bindings are added.

use super::accelerate_ffi;
use super::routing::*;
use super::*;
use crate::memory::allocator::IosurfaceAllocator;
use parking_lot::Mutex;
use std::sync::Arc;

/// Maps an operation family to the appropriate Accelerate sublibrary.
pub fn sublibrary_for(family: OperationFamily) -> Option<AccelerateSubLibrary> {
    match family {
        OperationFamily::Matmul => Some(AccelerateSubLibrary::Blas),
        OperationFamily::QuantizedMatmul => Some(AccelerateSubLibrary::Bnns),
        OperationFamily::RmsNorm => Some(AccelerateSubLibrary::Bnns),
        OperationFamily::RoPE => Some(AccelerateSubLibrary::VDsp),
        OperationFamily::Silu => Some(AccelerateSubLibrary::VForce),
        OperationFamily::Add => Some(AccelerateSubLibrary::VDsp),
        OperationFamily::Multiply => Some(AccelerateSubLibrary::VDsp),
        OperationFamily::Softmax => Some(AccelerateSubLibrary::Bnns),
        OperationFamily::Transpose => Some(AccelerateSubLibrary::VDsp),
        OperationFamily::Reshape => Some(AccelerateSubLibrary::VDsp),
        OperationFamily::Reduction => Some(AccelerateSubLibrary::VDsp),
        OperationFamily::Sampling => Some(AccelerateSubLibrary::VDsp),
        OperationFamily::LayoutTransform => Some(AccelerateSubLibrary::VDsp),
        OperationFamily::Checksum => Some(AccelerateSubLibrary::VDsp),
        OperationFamily::MlpBlock
        | OperationFamily::AttentionBlock
        | OperationFamily::DecoderLayer
        | OperationFamily::PrefillFragment
        | OperationFamily::IndexSelect => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerateSubLibrary {
    Blas,
    VDsp,
    Bnns,
    VForce,
}

/// Backing storage for an Accelerate tensor.
/// - Owned: internal Vec<f32> (created via create_f32)
/// - External: pointer to externally-managed memory (e.g. IOSurface from Arena)
///   The external memory is NOT freed on release — ownership remains with the caller.
#[derive(Clone)]
enum TensorStorage {
    Owned(Vec<f32>),
    External { ptr: *const f32, len: usize },
}

/// Metadata for a quantized weight group
#[derive(Debug, Clone)]
struct WeightMeta {
    group_size: usize,
    bits: u8,
    shape: Vec<i32>,
    /// Number of quantization groups: k * n / group_size / bits
    num_groups: usize,
}

/// A quantized weight with lazy dequantization cache
struct DequantizedWeight {
    raw_data: Vec<u8>,
    meta: WeightMeta,
    /// Cached dequantized f32 version (populated on first access)
    dequantized: Option<Vec<f32>>,
}

pub struct AccelerateBackend {
    name: String,
    tensors: Vec<Option<TensorStorage>>,
    generations: Vec<u32>,
    shapes: Vec<Option<Vec<i32>>>,
    free_list: Vec<usize>,
    /// Quantized weight storage
    weight_storage: Vec<Option<DequantizedWeight>>,
    weight_generations: Vec<u32>,
    weight_free_list: Vec<usize>,
    /// Optional shared IOSurface allocator for scratch buffers.
    allocator: Option<Arc<Mutex<IosurfaceAllocator>>>,
}

impl AccelerateBackend {
    pub fn new() -> Self {
        Self {
            name: "accelerate".into(),
            tensors: Vec::new(),
            generations: Vec::new(),
            shapes: Vec::new(),
            free_list: Vec::new(),
            weight_storage: Vec::new(),
            weight_generations: Vec::new(),
            weight_free_list: Vec::new(),
            allocator: None,
        }
    }

    /// Configure the backend to use a shared IOSurface allocator.
    /// Subsequent scratch buffer allocations will come from the IOSurface pool.
    pub fn set_allocator(&mut self, allocator: Arc<Mutex<IosurfaceAllocator>>) {
        self.allocator = Some(allocator);
    }

    /// Allocate a new tensor slot and return its index.
    /// Handles free-list reuse and generation bumping.
    fn allocate_slot(&mut self, storage: TensorStorage, shape: &[i32]) -> Result<usize, String> {
        if let Some(idx) = self.free_list.pop() {
            let generation = self.generations[idx]
                .checked_add(1)
                .ok_or_else(|| "generation overflow".to_string())?;
            self.generations[idx] = generation;
            self.tensors[idx] = Some(storage);
            self.shapes[idx] = Some(shape.to_vec());
            Ok(idx)
        } else {
            let idx = self.tensors.len();
            self.tensors.push(Some(storage));
            self.generations.push(1);
            self.shapes.push(Some(shape.to_vec()));
            Ok(idx)
        }
    }

    /// Create an output tensor. Uses IOSurface when a shared allocator is
    /// available, otherwise falls back to Owned(Vec<f32>).
    fn make_output<F>(&mut self, shape: &[i32], fill: F) -> Result<TensorHandle, String>
    where
        F: FnOnce(&mut [f32]) -> Result<(), String>,
    {
        let n = shape.iter().product::<i32>() as usize;
        // Clone the allocator Arc to avoid holding a reference through
        // self while calling self.allocate_slot later.
        let alloc_opt = self.allocator.clone();
        if let Some(allocator) = alloc_opt {
            let alloc = allocator.lock();
            let arena_id = alloc.allocate(1, n as u32, mlx_rs::Dtype::Float16)?;
            let arena = alloc
                .get_arena(arena_id)
                .ok_or_else(|| "arena not found".to_string())?;
            let ptr = unsafe { arena.base_ptr() as *mut f32 };
            let len = n;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            fill(slice)?;

            drop(alloc);
            let idx = self.allocate_slot(TensorStorage::External { ptr, len }, shape)?;
            Ok(TensorHandle {
                slot: idx as u32,
                generation: self.generations[idx],
            })
        } else {
            let mut buf = vec![0.0f32; n];
            fill(&mut buf)?;
            self.create_f32(&buf, shape)
        }
    }

    fn data(&self, handle: TensorHandle) -> Result<&[f32], String> {
        let slot = handle.slot as usize;
        match self.tensors.get(slot) {
            Some(Some(storage)) if handle.generation == self.generations[slot] => {
                match storage {
                    TensorStorage::Owned(v) => Ok(v.as_slice()),
                    TensorStorage::External { ptr, len } => {
                        // Safety: the caller guarantees the external pointer is valid
                        // for the lifetime of the handle
                        unsafe { Ok(std::slice::from_raw_parts(*ptr, *len)) }
                    }
                }
            }
            _ => Err(format!(
                "AccelerateBackend: invalid handle slot={slot} gen={}",
                handle.generation
            )),
        }
    }

    /// Shape of a stored tensor (immutable access).
    fn stored_shape(&self, handle: TensorHandle) -> Result<&[i32], String> {
        let slot = handle.slot as usize;
        match self.shapes.get(slot) {
            Some(Some(shape)) if handle.generation == self.generations[slot] => Ok(shape),
            _ => Err(format!("AccelerateBackend: invalid handle slot={slot}")),
        }
    }

    pub fn execute(
        &mut self,
        operation: &OperationDescriptor,
        _inputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        let mapping = sublibrary_for(operation.family);
        Err(format!(
            "AccelerateBackend: {:?} maps to {:?} but native implementation not yet available",
            operation.family, mapping
        ))
    }

    /// Register a quantized weight and return a handle for it.
    /// The weight data is stored as raw u8 bytes; dequantization happens lazily
    /// on the first quantized_matmul call.
    pub fn register_quantized_weight(
        &mut self,
        data: &[u8],
        group_size: usize,
        bits: u8,
        shape: &[i32],
    ) -> QuantizedWeightHandle {
        let num_elements: usize = shape.iter().map(|&d| d as usize).product();
        let num_groups = num_elements / group_size;

        let weight = DequantizedWeight {
            raw_data: data.to_vec(),
            meta: WeightMeta {
                group_size,
                bits,
                shape: shape.to_vec(),
                num_groups,
            },
            dequantized: None,
        };

        let idx = if let Some(idx) = self.weight_free_list.pop() {
            self.weight_generations[idx] = self.weight_generations[idx].checked_add(1).unwrap_or(1);
            self.weight_storage[idx] = Some(weight);
            idx
        } else {
            let idx = self.weight_storage.len();
            self.weight_storage.push(Some(weight));
            self.weight_generations.push(1);
            idx
        };
        QuantizedWeightHandle {
            slot: idx as u32,
            generation: self.weight_generations[idx],
        }
    }

    fn get_weight_mut(
        &mut self,
        handle: QuantizedWeightHandle,
    ) -> Result<&mut DequantizedWeight, String> {
        let slot = handle.slot as usize;
        let generation = handle.generation;
        match self.weight_storage.get_mut(slot) {
            Some(Some(w)) if generation == self.weight_generations[slot] => Ok(w),
            _ => Err(format!(
                "invalid quantized weight slot={slot} gen={generation}"
            )),
        }
    }

    /// Release a quantized weight, freeing its slot.
    pub fn release_weight(&mut self, handle: QuantizedWeightHandle) -> Result<(), String> {
        let slot = handle.slot as usize;
        let generation = handle.generation;
        if slot >= self.weight_storage.len() || self.weight_generations[slot] != generation {
            return Err(format!(
                "invalid weight handle slot={slot} gen={generation}"
            ));
        }
        if self.weight_storage[slot].is_none() {
            return Err(format!("weight slot {slot} already released"));
        }
        self.weight_storage[slot] = None;
        self.weight_free_list.push(slot);
        Ok(())
    }

    /// Dequantize a weight from packed u8 to f32.
    /// For group_size=N, the data is laid out as [group0_values..., group1_values..., ...]
    /// where each value is stored in `bits` bits (packed into u8).
    ///
    /// For simplicity with bits=8, each u8 is one value:
    ///   w_f32[group * group_size + i] = raw[group * group_size + i] * scales[group] + biases[group]
    fn dequantize_weight(
        weight: &DequantizedWeight,
        scales: &[f32],
        biases: &[f32],
    ) -> Result<Vec<f32>, String> {
        let meta = &weight.meta;
        let num_elements: usize = meta.shape.iter().map(|&d| d as usize).product();
        let mut result = vec![0.0f32; num_elements];

        let group_size = meta.group_size;
        let num_groups = meta.num_groups;

        if scales.len() < num_groups || biases.len() < num_groups {
            return Err(format!(
                "dequantize: expected {} scales/biases groups, got scales={} biases={}",
                num_groups,
                scales.len(),
                biases.len()
            ));
        }

        if meta.bits == 8 {
            // Simple path: one u8 per value
            for g in 0..num_groups {
                let scale = scales[g];
                let bias = biases[g];
                let offset = g * group_size;
                for i in 0..group_size {
                    if offset + i < weight.raw_data.len() && offset + i < num_elements {
                        let qval = weight.raw_data[offset + i] as f32;
                        result[offset + i] = qval * scale + bias;
                    }
                }
            }
        } else if meta.bits == 4 {
            // 4-bit: 2 values packed per byte. byte = [val_high << 4 | val_low]
            // group_size elements occupy ceil(group_size * 4 / 8) bytes
            let elems_per_byte: usize = 2;
            let bytes_per_group = (group_size * (meta.bits as usize) + 7) / 8;
            for g in 0..num_groups {
                let scale = scales[g];
                let bias = biases[g];
                let elem_offset = g * group_size;
                let byte_offset = g * bytes_per_group;
                for i in 0..group_size {
                    let dst_idx = elem_offset + i;
                    if dst_idx >= num_elements {
                        break;
                    }
                    let byte_idx = byte_offset + i / elems_per_byte;
                    if byte_idx >= weight.raw_data.len() {
                        break;
                    }
                    let byte_val = weight.raw_data[byte_idx];
                    let qval = if i % elems_per_byte == 0 {
                        (byte_val & 0x0F) as f32 // low nibble
                    } else {
                        ((byte_val >> 4) & 0x0F) as f32 // high nibble
                    };
                    result[dst_idx] = qval * scale + bias;
                }
            }
        } else if meta.bits == 2 {
            // 2-bit: 4 values packed per byte. byte = [v3<<6 | v2<<4 | v1<<2 | v0]
            // group_size elements occupy ceil(group_size * 2 / 8) bytes
            let elems_per_byte: usize = 4;
            let bytes_per_group = (group_size * (meta.bits as usize) + 7) / 8;
            for g in 0..num_groups {
                let scale = scales[g];
                let bias = biases[g];
                let elem_offset = g * group_size;
                let byte_offset = g * bytes_per_group;
                for i in 0..group_size {
                    let dst_idx = elem_offset + i;
                    if dst_idx >= num_elements {
                        break;
                    }
                    let byte_idx = byte_offset + i / elems_per_byte;
                    if byte_idx >= weight.raw_data.len() {
                        break;
                    }
                    let byte_val = weight.raw_data[byte_idx];
                    let shift = (i % elems_per_byte) * 2;
                    let qval = ((byte_val >> shift) & 0x03) as f32;
                    result[dst_idx] = qval * scale + bias;
                }
            }
        } else {
            return Err(format!(
                "dequantize: bits={} not supported (only 2, 4, 8)",
                meta.bits
            ));
        }

        Ok(result)
    }
}

impl Default for AccelerateBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorBackend for AccelerateBackend {
    fn create_f32(&mut self, data: &[f32], shape: &[i32]) -> Result<TensorHandle, String> {
        if shape.is_empty() || shape.iter().any(|&d| d <= 0) {
            return Err(format!(
                "create_f32: shape {:?} contains invalid dimensions",
                shape
            ));
        }
        let expected: usize = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| format!("create_f32: shape product overflow for {:?}", shape))?;
        if data.len() != expected {
            return Err(format!(
                "create_f32: data length {} != shape product {} for shape {:?}",
                data.len(),
                expected,
                shape,
            ));
        }
        let idx = if let Some(idx) = self.free_list.pop() {
            let new_gen = self.generations[idx]
                .checked_add(1)
                .ok_or_else(|| format!("AccelerateBackend: generation overflow at slot {idx}"))?;
            self.generations[idx] = new_gen;
            self.tensors[idx] = Some(TensorStorage::Owned(data.to_vec()));
            self.shapes[idx] = Some(shape.to_vec());
            idx
        } else {
            let idx = self.tensors.len();
            self.tensors.push(Some(TensorStorage::Owned(data.to_vec())));
            self.generations.push(1);
            self.shapes.push(Some(shape.to_vec()));
            idx
        };
        Ok(TensorHandle {
            slot: idx as u32,
            generation: self.generations[idx],
        })
    }

    fn create_u32(&mut self, data: &[u32], shape: &[i32]) -> Result<TensorHandle, String> {
        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        self.create_f32(&data_f32, shape)
    }

    fn create_f32_from_bf16_bits(
        &mut self,
        data: &[u16],
        shape: &[i32],
    ) -> Result<TensorHandle, String> {
        let data_f32: Vec<f32> = data
            .iter()
            .map(|&bits| f32::from_bits((bits as u32) << 16))
            .collect();
        self.create_f32(&data_f32, shape)
    }

    fn create_owned_from_bytes(
        &mut self,
        data: &[u8],
        shape: &[i32],
        dtype: DType,
    ) -> Result<TensorHandle, String> {
        match dtype {
            DType::F32 => {
                if data.len() % 4 != 0 {
                    return Err("create_owned_from_bytes: F32 data length not multiple of 4".into());
                }
                let n = data.len() / 4;
                let mut data_f32 = vec![0.0f32; n];
                // SAFETY: both slices are valid f32 byte representations
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const f32,
                        data_f32.as_mut_ptr(),
                        n,
                    );
                }
                self.create_f32(&data_f32, shape)
            }
            DType::F16 | DType::BF16 => {
                Err("create_owned_from_bytes: F16/BF16 not supported in AccelerateBackend".into())
            }
            DType::I8 | DType::U8 => {
                let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                self.create_f32(&data_f32, shape)
            }
            DType::I32 => {
                if data.len() % 4 != 0 {
                    return Err("create_owned_from_bytes: I32 data length not multiple of 4".into());
                }
                let n = data.len() / 4;
                // SAFETY: reinterpret I32 bytes as i32 slice
                let data_i32: &[i32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n) };
                let data_f32: Vec<f32> = data_i32.iter().map(|&v| v as f32).collect();
                self.create_f32(&data_f32, shape)
            }
            DType::U32 => {
                if data.len() % 4 != 0 {
                    return Err("create_owned_from_bytes: U32 data length not multiple of 4".into());
                }
                let n = data.len() / 4;
                // SAFETY: reinterpret U32 bytes as u32 slice
                let data_u32: &[u32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, n) };
                let data_f32: Vec<f32> = data_u32.iter().map(|&v| v as f32).collect();
                self.create_f32(&data_f32, shape)
            }
        }
    }

    fn quantized_matmul(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        scales: TensorHandle,
        biases: TensorHandle,
    ) -> Result<TensorHandle, String> {
        // Validate input
        let x_data = self.data(x)?.to_vec();
        let x_shape = self.stored_shape(x)?.to_vec();
        if x_shape.len() != 2 {
            return Err("quantized_matmul: input must be 2D".into());
        }
        if x_shape[0] as u32 != op.m || x_shape[1] as u32 != op.k {
            return Err(format!(
                "quantized_matmul: input shape mismatch ({:?} vs op m={} k={})",
                x_shape, op.m, op.k
            ));
        }

        // Pre-fetch scales and biases before the mutable borrow.
        let scales_data = self.data(scales)?.to_vec();
        let biases_data = self.data(biases)?.to_vec();

        // Lazy dequantization
        let weight = self.get_weight_mut(w)?;
        if weight.dequantized.is_none() {
            let deq = Self::dequantize_weight(weight, &scales_data, &biases_data)?;
            weight.dequantized = Some(deq);
        }
        let w_f32 = weight.dequantized.as_ref().unwrap();

        // Verify weight shape
        let expected_k = op.k as usize;
        let expected_n = op.n as usize;
        let w_expected_len = expected_k * expected_n;
        if w_f32.len() != w_expected_len {
            return Err(format!(
                "quantized_matmul: dequantized weight length {} != {} (k={} n={})",
                w_f32.len(),
                w_expected_len,
                op.k,
                op.n
            ));
        }

        // cblas_sgemm: C = x @ W (both in row-major)
        let m = op.m as i32;
        let n = op.n as i32;
        let k = op.k as i32;
        let w_data = w_f32.to_vec();
        self.make_output(&[m, n], |out| {
            unsafe {
                accelerate_ffi::cblas_sgemm(
                    accelerate_ffi::CBLAS_ROW_MAJOR,
                    accelerate_ffi::CBLAS_NO_TRANS,
                    accelerate_ffi::CBLAS_NO_TRANS,
                    m,
                    n,
                    k,
                    1.0,
                    x_data.as_ptr(),
                    k,
                    w_data.as_ptr(),
                    n,
                    0.0,
                    out.as_mut_ptr(),
                    n,
                );
            }
            Ok(())
        })
    }

    fn matmul(
        &mut self,
        op: &MatmulOp,
        a: TensorHandle,
        b: TensorHandle,
    ) -> Result<TensorHandle, String> {
        // Shape validation before FFI
        let a_shape = self.stored_shape(a)?.to_vec();
        let b_shape = self.stored_shape(b)?.to_vec();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(format!(
                "matmul: requires exactly 2D tensors, got {}D and {}D",
                a_shape.len(),
                b_shape.len()
            ));
        }
        let a_m = a_shape[0] as u32;
        let a_k = a_shape[1] as u32;
        let b_k = b_shape[0] as u32;
        let b_n = b_shape[1] as u32;
        if a_m != op.m {
            return Err(format!("A.M={a_m} != op.m={}", op.m));
        }
        if a_k != op.k || b_k != op.k {
            return Err(format!("K mismatch: A.K={a_k}, B.K={b_k}, op.k={}", op.k));
        }
        if b_n != op.n {
            return Err(format!("B.N={b_n} != op.n={}", op.n));
        }
        if op.m == 0 || op.n == 0 || op.k == 0 {
            return Err("matmul: dimensions must be positive".into());
        }

        let m = i32::try_from(op.m).map_err(|_| format!("matmul: M={} exceeds i32", op.m))?;
        let n = i32::try_from(op.n).map_err(|_| format!("matmul: N={} exceeds i32", op.n))?;
        let k = i32::try_from(op.k).map_err(|_| format!("matmul: K={} exceeds i32", op.k))?;

        // No-copy access to resident buffers
        let a_data = self.data(a)?.to_vec();
        let b_data = self.data(b)?.to_vec();

        self.make_output(&[m, n], |out| {
            unsafe {
                accelerate_ffi::cblas_sgemm(
                    accelerate_ffi::CBLAS_ROW_MAJOR,
                    accelerate_ffi::CBLAS_NO_TRANS,
                    accelerate_ffi::CBLAS_NO_TRANS,
                    m,
                    n,
                    k,
                    1.0f32, // alpha — passed by value
                    a_data.as_ptr(),
                    k,
                    b_data.as_ptr(),
                    n,
                    0.0f32, // beta — passed by value
                    out.as_mut_ptr(),
                    n,
                );
            }
            Ok(())
        })
    }

    fn rms_norm(
        &mut self,
        op: &RmsNormOp,
        x: TensorHandle,
        weight: TensorHandle,
    ) -> Result<TensorHandle, String> {
        let x_data = self.data(x)?.to_vec();
        let weight_data = self.data(weight)?.to_vec();
        let x_shape = self.stored_shape(x)?.to_vec();

        let dim = op.dim as usize;
        let n = x_shape.iter().product::<i32>() as usize;
        let batch = if dim > 0 { n / dim } else { 1 };

        if weight_data.len() != dim {
            return Err(format!(
                "rms_norm: weight length {} != dim {}",
                weight_data.len(),
                dim
            ));
        }

        let dim_i = dim as i32;

        self.make_output(&x_shape, |out| {
            for b in 0..batch {
                let row_start = b * dim;
                let x_row = &x_data[row_start..row_start + dim];
                let out_row = &mut out[row_start..row_start + dim];

                // Compute x_sq = x_row * x_row via vDSP_vmul
                let mut x_sq = vec![0.0f32; dim];
                unsafe {
                    accelerate_ffi::vDSP_vmul(
                        x_row.as_ptr(),
                        1,
                        x_row.as_ptr(),
                        1,
                        x_sq.as_mut_ptr(),
                        1,
                        dim_i,
                    );
                }

                // Sum x_sq via vDSP_sve
                let mut sum = 0.0f32;
                unsafe {
                    accelerate_ffi::vDSP_sve(x_sq.as_ptr(), 1, &mut sum, dim_i);
                }

                // inv_rms = 1 / sqrt(mean + eps)
                let mean = sum / dim as f32;
                let inv_rms = 1.0 / (mean + op.eps).sqrt();

                // Scale: out_row[i] = x_row[i] * inv_rms * weight[i]
                // First multiply by scalar inv_rms
                let scalar = [inv_rms];
                let mut scaled = vec![0.0f32; dim];
                unsafe {
                    accelerate_ffi::vDSP_vsmul(
                        x_row.as_ptr(),
                        1,
                        scalar.as_ptr(),
                        scaled.as_mut_ptr(),
                        1,
                        dim_i,
                    );
                }

                // Then multiply element-wise by weight
                unsafe {
                    accelerate_ffi::vDSP_vmul(
                        scaled.as_ptr(),
                        1,
                        weight_data.as_ptr(),
                        1,
                        out_row.as_mut_ptr(),
                        1,
                        dim_i,
                    );
                }
            }
            Ok(())
        })
    }

    fn rope(&mut self, op: &RoPEOp, x: TensorHandle) -> Result<TensorHandle, String> {
        let x_data = self.data(x)?.to_vec();
        let x_shape = self.stored_shape(x)?.to_vec();

        let head_dim = op.head_dim as usize;
        let n = x_shape.iter().product::<i32>() as usize;
        let num_vectors = if head_dim > 0 { n / head_dim } else { 0 };

        if op.positions.len() != num_vectors {
            return Err(format!(
                "rope: {} positions but {} vectors (head_dim={})",
                op.positions.len(),
                num_vectors,
                head_dim
            ));
        }

        let half_dim = head_dim / 2;
        let inv_base: f32 = 10000.0f32;

        self.make_output(&x_shape, |out| {
            for v in 0..num_vectors {
                let pos = op.positions[v] as f32;
                let row_start = v * head_dim;

                for i in 0..half_dim {
                    let theta = pos * inv_base.powf(-2.0 * i as f32 / head_dim as f32);
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();

                    let idx_even = row_start + 2 * i;
                    let idx_odd = idx_even + 1;

                    let x_e = x_data[idx_even];
                    let x_o = x_data[idx_odd];

                    out[idx_even] = x_e * cos_t - x_o * sin_t;
                    out[idx_odd] = x_e * sin_t + x_o * cos_t;
                }
            }
            Ok(())
        })
    }

    fn add(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String> {
        let a_data = self.data(a)?.to_vec();
        let b_data = self.data(b)?.to_vec();
        let a_shape = self.stored_shape(a)?.to_vec();

        let n = a_shape.iter().product::<i32>() as usize;
        let n_i = n as i32;

        if b_data.len() != n {
            return Err(format!(
                "add: tensor lengths differ ({} vs {})",
                n,
                b_data.len()
            ));
        }

        self.make_output(&a_shape, |out| {
            unsafe {
                accelerate_ffi::vDSP_vadd(
                    a_data.as_ptr(),
                    1,
                    b_data.as_ptr(),
                    1,
                    out.as_mut_ptr(),
                    1,
                    n_i,
                );
            }
            Ok(())
        })
    }

    fn multiply(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String> {
        let a_data = self.data(a)?.to_vec();
        let b_data = self.data(b)?.to_vec();
        let a_shape = self.stored_shape(a)?.to_vec();

        let n = a_shape.iter().product::<i32>() as usize;
        let n_i = n as i32;

        if b_data.len() != n {
            return Err(format!(
                "multiply: tensor lengths differ ({} vs {})",
                n,
                b_data.len()
            ));
        }

        self.make_output(&a_shape, |out| {
            unsafe {
                accelerate_ffi::vDSP_vmul(
                    a_data.as_ptr(),
                    1,
                    b_data.as_ptr(),
                    1,
                    out.as_mut_ptr(),
                    1,
                    n_i,
                );
            }
            Ok(())
        })
    }

    fn silu(&mut self, x: TensorHandle) -> Result<TensorHandle, String> {
        let x_data = self.data(x)?.to_vec();
        let x_shape = self.stored_shape(x)?.to_vec();

        let n = x_shape.iter().product::<i32>() as usize;
        let n_i = n as i32;

        self.make_output(&x_shape, |out| {
            // Compute silu(x) = x * sigmoid(x) using the identity:
            //   sigmoid(x) = exp(x) / (1 + exp(x))
            //   silu(x) = x * exp(x) / (1 + exp(x))
            let mut exp_x = vec![0.0f32; n];
            unsafe {
                accelerate_ffi::vvexpf(exp_x.as_mut_ptr(), x_data.as_ptr(), &n_i);
            }
            // Compute x * exp_x
            let mut x_exp = vec![0.0f32; n];
            unsafe {
                accelerate_ffi::vDSP_vmul(
                    x_data.as_ptr(),
                    1,
                    exp_x.as_ptr(),
                    1,
                    x_exp.as_mut_ptr(),
                    1,
                    n_i,
                );
            }
            // Compute 1 + exp_x
            let mut one_plus_exp = vec![0.0f32; n];
            unsafe {
                let ones = vec![1.0f32; n];
                accelerate_ffi::vDSP_vadd(
                    ones.as_ptr(),
                    1,
                    exp_x.as_ptr(),
                    1,
                    one_plus_exp.as_mut_ptr(),
                    1,
                    n_i,
                );
            }

            // out = (x * exp_x) / (1 + exp_x) = silu(x)
            unsafe {
                accelerate_ffi::vDSP_vdiv(
                    one_plus_exp.as_ptr(),
                    1,
                    x_exp.as_ptr(),
                    1,
                    out.as_mut_ptr(),
                    1,
                    n_i,
                );
            }

            Ok(())
        })
    }

    fn transpose(&mut self, x: TensorHandle, dims: &[i32]) -> Result<TensorHandle, String> {
        let x_data = self.data(x)?.to_vec();
        let x_shape = self.stored_shape(x)?.to_vec();

        if dims.len() != x_shape.len() {
            return Err(format!(
                "transpose: dims length {} != shape rank {}",
                dims.len(),
                x_shape.len()
            ));
        }

        // Validate dims permutation
        let mut seen = vec![false; x_shape.len()];
        for &d in dims {
            let idx = d as usize;
            if idx >= x_shape.len() || seen[idx] {
                return Err(format!("transpose: invalid dims permutation {:?}", dims));
            }
            seen[idx] = true;
        }

        if x_shape.len() == 2 && dims[0] == 1 && dims[1] == 0 {
            // Fast path: vDSP_mtrans for 2D transpose [1,0]
            let m = x_shape[0];
            let n = x_shape[1];
            let new_shape = vec![n, m];
            self.make_output(&new_shape, |out| {
                unsafe {
                    accelerate_ffi::vDSP_mtrans(x_data.as_ptr(), 1, out.as_mut_ptr(), 1, m, n);
                }
                Ok(())
            })
        } else {
            // General transpose via coordinate mapping
            let new_shape: Vec<i32> = dims.iter().map(|&d| x_shape[d as usize]).collect();
            let n = x_shape.iter().product::<i32>() as usize;

            // Build strides
            let mut old_strides = vec![0i32; x_shape.len()];
            let mut new_strides = vec![0i32; new_shape.len()];
            old_strides[x_shape.len() - 1] = 1;
            new_strides[new_shape.len() - 1] = 1;
            for i in (0..x_shape.len() - 1).rev() {
                old_strides[i] = old_strides[i + 1] * x_shape[i + 1];
                new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
            }

            self.make_output(&new_shape, |out| {
                for linear in 0..n as i32 {
                    let mut old_idx = 0i32;
                    let mut remaining = linear;
                    for d in 0..x_shape.len() {
                        let coord = remaining / new_strides[d];
                        remaining %= new_strides[d];
                        old_idx += coord * old_strides[dims[d as usize] as usize];
                    }
                    out[linear as usize] = x_data[old_idx as usize];
                }
                Ok(())
            })
        }
    }

    fn reshape(&mut self, x: TensorHandle, shape: &[i32]) -> Result<TensorHandle, String> {
        let x_data = self.data(x)?.to_vec();
        let x_shape = self.stored_shape(x)?.to_vec();

        let old_product: usize = x_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| format!("reshape: old shape product overflow for {:?}", x_shape))?;
        let new_product: usize = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| format!("reshape: new shape product overflow for {:?}", shape))?;

        if old_product != new_product {
            return Err(format!(
                "reshape: element count mismatch (old={:?}, new={:?}, old_n={}, new_n={})",
                x_shape, shape, old_product, new_product
            ));
        }

        // Clone the data and register with the new shape
        self.create_f32(&x_data, shape)
    }

    fn softmax(&mut self, x: TensorHandle, axis: i32) -> Result<TensorHandle, String> {
        let x_data = self.data(x)?.to_vec();
        let x_shape = self.stored_shape(x)?.to_vec();

        // Normalize axis
        let rank = x_shape.len() as i32;
        let ax = if axis < 0 { axis + rank } else { axis };
        if ax < 0 || ax >= rank {
            return Err(format!(
                "softmax: axis {} out of range for rank {}",
                axis, rank
            ));
        }
        let ax = ax as usize;

        // Compute number of rows (along axis) and row length (along axis dimension)
        let outer: usize = x_shape[..ax]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .unwrap_or(1);
        let dim = x_shape[ax] as usize;
        let inner: usize = x_shape[ax + 1..]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .unwrap_or(1);

        let row_len = dim;
        let num_rows = outer * inner;
        let _n = x_shape.iter().product::<i32>() as usize;
        let row_len_i = row_len as i32;

        self.make_output(&x_shape, |out| {
            // For each "row" along the softmax axis
            for r in 0..num_rows {
                // Each row: dim elements at stride `inner`
                // Source: x_data[r * inner * dim + ...], with contiguous inner dims (for axis=-1)
                // For general axis, we need strided access
                // For simplicity: assume axis is the last dimension (most common case)
                // or if inner == 1, we have contiguous rows

                // Find the contiguous block for this row
                // Row starts at: (r / inner) * (dim * inner) + (r % inner)
                // For contiguous case (axis == rank-1, inner == 1):
                let base = r * dim;

                // 1. Find max
                let mut max_val = x_data[base];
                for i in 1..dim {
                    let v = x_data[base + i];
                    if v > max_val {
                        max_val = v;
                    }
                }

                // 2. Subtract max from each element => row_minus_max
                let row_start = base;
                let row_end = base + dim;
                for i in row_start..row_end {
                    out[i] = x_data[i] - max_val;
                }

                // 3. Compute exp of each element (in-place on out row)
                unsafe {
                    accelerate_ffi::vvexpf(
                        out.as_mut_ptr().add(row_start),
                        out.as_ptr().add(row_start),
                        &row_len_i,
                    );
                }

                // 4. Compute sum of exp values
                let mut exp_sum = 0.0f32;
                unsafe {
                    accelerate_ffi::vDSP_sve(
                        out.as_ptr().add(row_start),
                        1,
                        &mut exp_sum,
                        row_len_i,
                    );
                }

                // 5. Divide each exp by the sum
                unsafe {
                    accelerate_ffi::vDSP_vdiv(
                        &exp_sum,
                        0,
                        out.as_ptr().add(row_start),
                        1,
                        out.as_mut_ptr().add(row_start),
                        1,
                        row_len_i,
                    );
                }
            }
            Ok(())
        })
    }

    fn index_select(
        &mut self,
        x: TensorHandle,
        indices: &[u32],
        axis: i32,
    ) -> Result<TensorHandle, String> {
        let x_data = self.data(x)?.to_vec();
        let x_shape = self.stored_shape(x)?.to_vec();

        let rank = x_shape.len() as i32;
        let ax = if axis < 0 { axis + rank } else { axis };
        if ax < 0 || ax >= rank {
            return Err(format!(
                "index_select: axis {} out of range for rank {}",
                axis, rank
            ));
        }
        let ax = ax as usize;

        if x_shape.len() == 1 {
            // 1D: use vDSP_vgathr
            let n = indices.len();
            let new_shape = vec![n as i32];
            let indices_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
            self.make_output(&new_shape, |out| {
                unsafe {
                    accelerate_ffi::vDSP_vgathr(
                        x_data.as_ptr(),
                        indices_i32.as_ptr(),
                        1,
                        out.as_mut_ptr(),
                        1,
                        n as i32,
                    );
                }
                Ok(())
            })
        } else if x_shape.len() == 2 && ax == 0 {
            // 2D, axis=0: copy selected rows
            let rows = x_shape[0] as usize;
            let cols = x_shape[1] as usize;
            let out_rows = indices.len();
            let new_shape = vec![out_rows as i32, x_shape[1]];
            self.make_output(&new_shape, |out| {
                for (out_idx, &idx) in indices.iter().enumerate() {
                    let src_idx = idx as usize;
                    if src_idx >= rows {
                        return Err(format!(
                            "index_select: index {} out of bounds for dimension size {}",
                            idx, rows
                        ));
                    }
                    let src_start = src_idx * cols;
                    let dst_start = out_idx * cols;
                    out[dst_start..dst_start + cols]
                        .copy_from_slice(&x_data[src_start..src_start + cols]);
                }
                Ok(())
            })
        } else if x_shape.len() == 2 && ax == 1 {
            // 2D, axis=1: copy selected columns
            let rows = x_shape[0] as usize;
            let cols = x_shape[1] as usize;
            let out_cols = indices.len();
            let new_shape = vec![x_shape[0], out_cols as i32];
            self.make_output(&new_shape, |out| {
                for r in 0..rows {
                    let src_row_start = r * cols;
                    let dst_row_start = r * out_cols;
                    for (out_c, &idx) in indices.iter().enumerate() {
                        let src_idx = idx as usize;
                        if src_idx >= cols {
                            return Err(format!(
                                "index_select: index {} out of bounds for dimension size {}",
                                idx, cols
                            ));
                        }
                        out[dst_row_start + out_c] = x_data[src_row_start + src_idx];
                    }
                }
                Ok(())
            })
        } else {
            // General ND: for each position along axis, copy sub-slices
            let _n = x_shape.iter().product::<i32>() as usize;
            let outer: usize = x_shape[..ax]
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
                .unwrap_or(1);
            let dim = x_shape[ax] as usize;
            let inner: usize = x_shape[ax + 1..]
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
                .unwrap_or(1);

            let out_dim = indices.len();
            let slice_len = inner;
            let mut new_shape = x_shape.to_vec();
            new_shape[ax] = out_dim as i32;
            self.make_output(&new_shape, |out| {
                for o in 0..outer {
                    let src_base = o * dim * inner;
                    let dst_base = o * out_dim * inner;
                    for (dst_i, &idx) in indices.iter().enumerate() {
                        let src_idx = idx as usize;
                        if src_idx >= dim {
                            return Err(format!(
                                "index_select: index {} out of bounds for dimension size {}",
                                idx, dim
                            ));
                        }
                        let src_off = src_base + src_idx * inner;
                        let dst_off = dst_base + dst_i * inner;
                        out[dst_off..dst_off + slice_len]
                            .copy_from_slice(&x_data[src_off..src_off + slice_len]);
                    }
                }
                Ok(())
            })
        }
    }

    fn evaluate(
        &mut self,
        group_id: u64,
        outputs: &[TensorHandle],
    ) -> Result<EvaluationReceipt, String> {
        // Validate every output handle before issuing receipt
        for &h in outputs {
            let _ = self.data(h)?;
        }
        let (active, cached) = self.active_memory();
        Ok(EvaluationReceipt {
            group_id,
            graph_build_ns: 0,
            submit_ns: 0,
            sync_ns: 0,
            output_count: outputs.len(),
            active_memory_after: active,
            cache_memory_after: cached,
            observed_substrate: Some("cpu".into()),
            eval_calls: 0,
        })
    }

    fn read_f32(&mut self, handle: TensorHandle) -> Result<ReadbackReceipt, String> {
        let data = self.data(handle)?.to_vec();
        Ok(ReadbackReceipt {
            data,
            forced_eval: false,
            sync_ns: 0,
            observed_substrate: Some("cpu".into()),
        })
    }

    fn shape(&self, handle: TensorHandle) -> Result<Vec<i32>, String> {
        self.stored_shape(handle).map(|s| s.to_vec())
    }

    fn release(&mut self, handle: TensorHandle) -> Result<(), String> {
        let slot = handle.slot as usize;
        let generation = handle.generation;
        if slot >= self.tensors.len() || self.generations[slot] != generation {
            return Err(format!(
                "AccelerateBackend: invalid handle slot={slot} gen={generation}"
            ));
        }
        if self.tensors[slot].is_none() {
            return Err(format!("AccelerateBackend: slot {slot} already released"));
        }
        self.tensors[slot] = None;
        self.shapes[slot] = None;
        self.free_list.push(slot);
        Ok(())
    }

    fn active_memory(&self) -> (u64, u64) {
        let active: u64 = self
            .tensors
            .iter()
            .filter_map(|t| t.as_ref())
            .map(|s| match s {
                TensorStorage::Owned(v) => (v.len() * 4) as u64,
                TensorStorage::External { len: _, .. } => 0, // tracked by IosurfaceAllocator
            })
            .sum();
        (active, 0)
    }

    fn bind_external(
        &mut self,
        _owner_token: u64,
        data: &[u8],
        shape: &[i32],
        dtype: DType,
    ) -> Result<TensorHandle, String> {
        if dtype != DType::F32 {
            return Err("AccelerateBackend: bind_external supports F32 only".into());
        }
        let ptr = data.as_ptr() as *const f32;
        let len = data.len() / 4; // bytes to f32 count
        let storage = TensorStorage::External { ptr, len };

        let idx = if let Some(idx) = self.free_list.pop() {
            self.generations[idx] = self.generations[idx]
                .checked_add(1)
                .ok_or_else(|| "generation overflow".to_string())?;
            self.tensors[idx] = Some(storage);
            self.shapes[idx] = Some(shape.to_vec());
            idx
        } else {
            let idx = self.tensors.len();
            self.tensors.push(Some(storage));
            self.generations.push(1);
            self.shapes.push(Some(shape.to_vec()));
            idx
        };
        Ok(TensorHandle {
            slot: idx as u32,
            generation: self.generations[idx],
        })
    }

    fn backend_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            can_gpu: false,
            can_cpu: true,
            supports_quantized: true,
            supports_bf16_native: false,
            backend_name: self.name.clone(),
        }
    }
    fn concatenate(&mut self, tensors: &[TensorHandle], axis: i32) -> Result<TensorHandle, String> {
        if tensors.is_empty() {
            return Err("concatenate: empty tensor list".into());
        }
        if tensors.len() == 1 {
            // Single tensor — just copy the handle's data
            let d = self.data(tensors[0])?.to_vec();
            let s = self.stored_shape(tensors[0])?.to_vec();
            return self.create_f32(&d, &s);
        }

        // Validate all shapes match except on the concat axis
        let ref_shape = self.stored_shape(tensors[0])?.to_vec();
        let ax = if axis < 0 {
            (ref_shape.len() as i32 + axis) as usize
        } else {
            axis as usize
        };
        if ax >= ref_shape.len() {
            return Err(format!(
                "concatenate: axis {} out of range for shape {:?}",
                axis, ref_shape
            ));
        }

        let mut total_on_axis: usize = ref_shape[ax] as usize;
        let mut all_data: Vec<Vec<f32>> = Vec::with_capacity(tensors.len());
        all_data.push(self.data(tensors[0])?.to_vec());

        for (i, &h) in tensors[1..].iter().enumerate() {
            let s = self.stored_shape(h)?.to_vec();
            if s.len() != ref_shape.len() {
                return Err(format!(
                    "concatenate: tensor {} has {} dims, expected {}",
                    i + 1,
                    s.len(),
                    ref_shape.len()
                ));
            }
            for (j, (&a, &b)) in s.iter().zip(ref_shape.iter()).enumerate() {
                if j != ax && a != b {
                    return Err(format!(
                        "concatenate: tensor {} shape mismatch at dim {}: {} != {}",
                        i + 1,
                        j,
                        a,
                        b
                    ));
                }
            }
            total_on_axis += s[ax] as usize;
            all_data.push(self.data(h)?.to_vec());
        }

        // Compute per-element sizes
        let inner: usize = ref_shape[ax + 1..]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .unwrap_or(1);
        let outer_per_input: Vec<usize> = all_data
            .iter()
            .map(|d| d.len() / (ref_shape[ax] as usize * inner))
            .collect();

        let total_outer: usize = ref_shape
            .iter()
            .take(ax)
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .unwrap_or(1);
        let mut out = vec![0.0f32; total_outer * total_on_axis * inner];

        for o in 0..total_outer {
            let dst_base = o * total_on_axis * inner;
            let mut write_offset = 0;
            for (ti, data) in all_data.iter().enumerate() {
                let dim_ax = ref_shape[ax] as usize;
                let slice_len = dim_ax * inner;
                let src_base = o * outer_per_input[ti] * slice_len;
                let dst_slice = dst_base + write_offset;
                out[dst_slice..dst_slice + slice_len]
                    .copy_from_slice(&data[src_base..src_base + slice_len]);
                write_offset += slice_len;
            }
        }

        let mut new_shape = ref_shape.to_vec();
        new_shape[ax] = total_on_axis as i32;
        self.create_f32(&out, &new_shape)
    }

    fn slice(
        &mut self,
        x: TensorHandle,
        start: &[i32],
        stop: &[i32],
        step: &[i32],
    ) -> Result<TensorHandle, String> {
        let x_data = self.data(x)?.to_vec();
        let x_shape = self.stored_shape(x)?.to_vec();
        let ndim = x_shape.len();

        if start.len() != ndim || stop.len() != ndim || step.len() != ndim {
            return Err(format!(
                "slice: start/stop/step must all have {} entries (tensor dims), got {} {} {}",
                ndim,
                start.len(),
                stop.len(),
                step.len()
            ));
        }

        // Compute output shape
        let mut out_shape = Vec::with_capacity(ndim);
        let mut strides = Vec::with_capacity(ndim);
        for d in 0..ndim {
            let s = start[d].max(0).min(x_shape[d] - 1);
            let e = stop[d].max(s + 1).min(x_shape[d]);
            let st = step[d].max(1);
            let dim_size = ((e - s) + st - 1) / st;
            out_shape.push(dim_size);
            strides.push(st);
        }

        // Compute output size
        let out_len: usize = out_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .unwrap_or(0);
        let mut out = vec![0.0f32; out_len];

        // General ND iteration
        // Iterate over ND using linear index + strides
        for o in 0..out_len {
            // Recover ND coordinates from linear index
            let mut remaining = o;
            let mut src_linear = 0;
            for d in (0..ndim).rev() {
                let pos_d =
                    (start[d] + (remaining % out_shape[d] as usize) as i32 * strides[d]) as usize;
                let dim_stride = x_shape[d + 1..]
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
                    .unwrap_or(1);
                src_linear += pos_d * dim_stride;
                remaining /= out_shape[d] as usize;
            }
            out[o] = x_data[src_linear];
        }

        self.create_f32(&out, &out_shape)
    }

    fn cast(&mut self, x: TensorHandle, dtype: DType) -> Result<TensorHandle, String> {
        match dtype {
            DType::F32 => {
                // Already F32 — copy data and register with new handle
                let data = self.data(x)?.to_vec();
                let shape = self.stored_shape(x)?.to_vec();
                self.create_f32(&data, &shape)
            }
            DType::I8 | DType::U8 | DType::I32 | DType::U32 => {
                // F32 → target: read as F32 then convert
                let data = self.data(x)?.to_vec();
                let shape = self.stored_shape(x)?.to_vec();
                match dtype {
                    DType::I8 => {
                        let converted: Vec<f32> = data.iter().map(|&v| (v as i8) as f32).collect();
                        self.create_f32(&converted, &shape)
                    }
                    DType::U8 => {
                        let converted: Vec<f32> = data.iter().map(|&v| (v as u8) as f32).collect();
                        self.create_f32(&converted, &shape)
                    }
                    DType::I32 => {
                        let converted: Vec<f32> = data.iter().map(|&v| (v as i32) as f32).collect();
                        self.create_f32(&converted, &shape)
                    }
                    DType::U32 => {
                        let converted: Vec<f32> = data.iter().map(|&v| (v as u32) as f32).collect();
                        self.create_f32(&converted, &shape)
                    }
                    _ => Err(format!(
                        "cast to {:?} not supported in AccelerateBackend",
                        dtype
                    )),
                }
            }
            DType::F16 | DType::BF16 => Err(
                "AccelerateBackend: cast to F16/BF16 not supported (no native half type)".into(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{MatmulOp, RmsNormOp};

    #[test]
    fn test_matmul_2x3_3x2() {
        let mut backend = AccelerateBackend::new();
        let a = backend
            .create_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap();
        let b = backend
            .create_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])
            .unwrap();
        let op = MatmulOp { m: 2, k: 3, n: 2 };
        let c = backend.matmul(&op, a, b).unwrap();
        let result = backend.read_f32(c).unwrap();
        assert_eq!(result.data.len(), 4);
        let expected = [58.0, 64.0, 139.0, 154.0];
        for (got, exp) in result.data.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-4,
                "matmul: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_matmul_1x4_4x1() {
        let mut backend = AccelerateBackend::new();
        let a = backend.create_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let b = backend.create_f32(&[5.0, 6.0, 7.0, 8.0], &[4, 1]).unwrap();
        let op = MatmulOp { m: 1, k: 4, n: 1 };
        let c = backend.matmul(&op, a, b).unwrap();
        let result = backend.read_f32(c).unwrap();
        assert_eq!(result.data, vec![70.0]);
    }

    #[test]
    fn test_rms_norm_basic() {
        let mut backend = AccelerateBackend::new();
        let x = backend.create_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let w = backend.create_f32(&[0.5, 1.0, 1.5, 2.0], &[4]).unwrap();
        let op = RmsNormOp { dim: 4, eps: 1e-6 };
        let y = backend.rms_norm(&op, x, w).unwrap();
        let result = backend.read_f32(y).unwrap();
        assert_eq!(result.data.len(), 4);
        let inv_rms = 1.0 / (7.5_f64).sqrt() as f32;
        let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .zip(vec![0.5, 1.0, 1.5, 2.0].into_iter())
            .map(|(x, w)| x * inv_rms * w)
            .collect();
        for (got, exp) in result.data.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-3,
                "rms_norm: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_add_basic() {
        let mut backend = AccelerateBackend::new();
        let a = backend.create_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = backend.create_f32(&[4.0, 5.0, 6.0], &[3]).unwrap();
        let c = backend.add(a, b).unwrap();
        let result = backend.read_f32(c).unwrap();
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_silu_basic() {
        let mut backend = AccelerateBackend::new();
        let x = backend.create_f32(&[0.0, 1.0, -1.0, 2.0], &[4]).unwrap();
        let y = backend.silu(x).unwrap();
        let result = backend.read_f32(y).unwrap();
        eprintln!("silu result: {:?}", result.data);
        assert!((result.data[0] - 0.0).abs() < 1e-3);
        assert!((result.data[1] - 0.731).abs() < 5e-3);
        assert!((result.data[2] - -0.269).abs() < 5e-3);
        assert!((result.data[3] - 1.762).abs() < 5e-3);
    }
}
