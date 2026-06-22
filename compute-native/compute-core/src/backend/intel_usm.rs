//! Intel USM (Unified Shared Memory) — zero-copy buffer abstraction for Intel iGPUs.
//! Ported from ggml-sycl's USM pattern and oneAPI SYCL malloc_shared semantics.
//!
//! USM allocations can be accessed from both host CPU and Intel iGPU without
//! explicit copy operations. This is the Linux analogue of Apple Unified Memory.
//!
//! # Feature gate
//!
//! This module is compiled only when `feature = "intel"` is enabled.

/// Private result alias using `String` error, matching the backend pattern.
pub type Result<T, E = String> = std::result::Result<T, E>;

/// Level Zero driver discovery result.
#[derive(Clone, Debug)]
pub struct LevelZeroProbe {
    pub driver_version: String,
    pub device_count: u32,
    pub has_usm_shared: bool,
    pub has_usm_device: bool,
    pub max_compute_units: u32,
    pub device_name: String,
    pub available: bool,
}

impl LevelZeroProbe {
    /// Probe for Level Zero drivers and Intel iGPU.
    /// On systems without Level Zero, returns a clean "not available" result.
    pub fn probe() -> Self {
        // In real mode (Linux + level-zero installed):
        //   calls zeDriverGet() -> zeDeviceGet() -> zeDeviceGetProperties()
        // In stub mode (macOS or no driver):
        LevelZeroProbe {
            driver_version: String::new(),
            device_count: 0,
            has_usm_shared: false,
            has_usm_device: false,
            max_compute_units: 0,
            device_name: "none".into(),
            available: false,
        }
    }

    pub fn is_available(&self) -> bool {
        self.available
    }
}

/// USM allocation type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsmType {
    /// Shared allocation — accessible by both host CPU and device
    Shared,
    /// Device-only allocation — requires explicit copy
    Device,
    /// Host-only allocation — accessible only by CPU
    Host,
}

/// A USM buffer descriptor.
#[derive(Clone, Debug)]
pub struct UsmBuffer {
    pub allocation: UsmType,
    pub byte_size: u64,
    pub alignment: u64,
    pub device_ptr: Option<u64>,   // Device virtual address (Level Zero)
    pub host_ptr: Option<*mut u8>, // Host pointer (always set for shared/host)
}

impl UsmBuffer {
    pub fn allocate(_size: u64, _usm_type: UsmType, _alignment: u64) -> Result<Self> {
        // Stub: return an error with a clear message
        // Real: calls zeMemAllocShared/zeMemAllocDevice
        Err(format!(
            "USM allocation not available: Intel Level Zero driver required (stub mode)",
        ))
    }

    pub fn free(&mut self) {
        self.device_ptr = None;
        self.host_ptr = None;
    }

    pub fn is_valid(&self) -> bool {
        self.device_ptr.is_some() || self.host_ptr.is_some()
    }
}

/// USM memory pool — manages multiple USM allocations with a generational slot-map.
pub struct UsmMemoryPool {
    buffers: Vec<Option<UsmBuffer>>,
    generations: Vec<u32>,
    free_list: Vec<usize>,
    total_allocated_bytes: u64,
}

impl UsmMemoryPool {
    pub fn new() -> Self {
        UsmMemoryPool {
            buffers: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            total_allocated_bytes: 0,
        }
    }

    pub fn allocate(&mut self, size: u64, usm_type: UsmType, alignment: u64) -> Result<UsmHandle> {
        let buf = UsmBuffer::allocate(size, usm_type, alignment)?;
        self.total_allocated_bytes += size;
        if let Some(idx) = self.free_list.pop() {
            let gen = self.generations[idx];
            self.buffers[idx] = Some(buf);
            Ok(UsmHandle {
                index: idx as u32,
                generation: gen,
            })
        } else {
            let gen = 1;
            self.buffers.push(Some(buf));
            self.generations.push(gen);
            Ok(UsmHandle {
                index: (self.buffers.len() - 1) as u32,
                generation: gen,
            })
        }
    }

    pub fn get(&self, handle: UsmHandle) -> Option<&UsmBuffer> {
        self.buffers
            .get(handle.index as usize)
            .and_then(|s| s.as_ref())
            .filter(|_| handle.generation == self.generations[handle.index as usize])
    }

    pub fn release(&mut self, handle: UsmHandle) {
        if let Some(Some(buf)) = self.buffers.get_mut(handle.index as usize) {
            if self.generations[handle.index as usize] == handle.generation {
                buf.free();
                self.generations[handle.index as usize] += 1;
                self.free_list.push(handle.index as usize);
                self.buffers[handle.index as usize] = None;
            }
        }
    }

    pub fn total_allocated(&self) -> u64 {
        self.total_allocated_bytes
    }
}

/// Handle referencing a USM allocation in the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UsmHandle {
    pub index: u32,
    pub generation: u32,
}

/// Dispatch cascade for Intel iGPU matrix operations.
/// Ported from ggml-sycl's DMMV -> MMVQ -> MMQ -> oneDNN cascade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchLevel {
    /// Direct dequantize-on-the-fly during matmul vector (single token decode)
    Dmmv,
    /// Mixed-precision matmul vector quantized (small batch decode)
    Mmvq,
    /// Mixed-precision matmul quantized (prefill, large batch)
    Mmq,
    /// oneDNN library GEMM (fp32/bf16, largest matmuls)
    OneDnnGemm,
    /// Fall back to CPU
    FallbackCpu,
}

impl DispatchLevel {
    /// Select the dispatch level based on operation parameters.
    pub fn select(m: u32, n: u32, k: u32, is_quantized: bool) -> Self {
        let _ = (n, k); // unused in stub, reserved for future heuristics
        if m == 1 && is_quantized {
            DispatchLevel::Dmmv
        } else if m <= 4 && is_quantized {
            DispatchLevel::Mmvq
        } else if m <= 128 {
            DispatchLevel::Mmq
        } else {
            DispatchLevel::OneDnnGemm
        }
    }
}
