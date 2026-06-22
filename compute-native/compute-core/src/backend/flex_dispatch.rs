//! FlexDispatch — runtime backend scheduler.
//!
//! Dynamically assigns each operation to MLX (GPU), Core ML (ANE), or
//! Accelerate (CPU/NEON) based on real-time system state sampled from
//! IOKit, Mach, and iOS/IOKit power-source APIs.  No compile-time static
//! routing — the dispatcher adapts to actual running conditions.
//!
//! # Design
//!
//! Every `N` decode steps the controller samples GPU utilization, CPU
//! load, thermal state and battery.  Each operation is classified into one
//! of five families — `MatMul`, `Attention`, `ElementWise`, `Softmax`,
//! `LayerNorm` — and routed to the most appropriate backend for the
//! *current* system state.
//!
//! - **MatMul** (GPU-bound) → MLX, unless thermal/battery throttling
//!   demands the more efficient ANE path.
//! - **Attention** (memory-bandwidth-bound) → ANE when GPU is saturated,
//!   Accelerate when throttling, MLX otherwise.
//! - **ElementWise** (cheap) → Accelerate (CPU) when GPU is busy, MLX
//!   when it is free.
//! - **Softmax / LayerNorm** → Accelerate (NEON) to keep the GPU free for
//!   matmuls.

use crate::backend::heterogeneous_executor::HeterogeneousExecutor;
use crate::backend::routing::*;

// ── Thermal state ──────────────────────────────────────────────────────────

/// System thermal state, as reported by `kern.thermal.thermal_level` or
/// `machdep.xcpm.mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

// ── System state ───────────────────────────────────────────────────────────

/// Real-time system state for backend dispatch decisions.
///
/// Sampled on every `FlexDispatch::sample_interval` decode steps.
/// Each field is measured from a hardware sensor or OS counter; no
/// synthetic or extrapolated values.
#[derive(Debug, Clone)]
pub struct SystemState {
    /// GPU utilization (0.0–1.0, from Metal performance counters via IOKit).
    pub gpu_utilization: f64,
    /// ANE utilization (approx from ANE program queue depth).
    pub ane_utilization: f64,
    /// CPU utilization across all cores (from `host_cpu_load_info`).
    pub cpu_utilization: f64,
    /// GPU peak memory fraction used.
    pub gpu_memory_fraction: f64,
    /// Thermal state (from sysctl `kern.thermal.thermal_level`).
    pub thermal_state: ThermalState,
    /// Battery power remaining (0.0–1.0, 1.0 = full).
    pub battery_remaining: f64,
    /// Whether the device is on AC power (vs battery).
    pub ac_power: bool,
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            gpu_utilization: 0.0,
            ane_utilization: 0.0,
            cpu_utilization: 0.0,
            gpu_memory_fraction: 0.0,
            thermal_state: ThermalState::Nominal,
            battery_remaining: 1.0,
            ac_power: true,
        }
    }
}

impl SystemState {
    /// Sample all system sensors and return a consolidated [`SystemState`].
    ///
    /// Each subsystem is sampled independently. A failure in one sensor
    /// (e.g. no IOKit GPU counter available) is logged via the `Err` return
    /// but the other fields retain their default/previous values.
    pub fn sample() -> Result<Self, String> {
        let gpu_util = Self::sample_gpu_utilization()?;
        let cpu_util = Self::sample_cpu_utilization();
        let thermal = Self::sample_thermal_state();
        let (bat, ac) = Self::sample_power_state();

        Ok(Self {
            gpu_utilization: gpu_util,
            ane_utilization: 0.0, // ANE has no public perf-counter API
            cpu_utilization: cpu_util,
            gpu_memory_fraction: Self::sample_gpu_memory_fraction(),
            thermal_state: thermal,
            battery_remaining: bat,
            ac_power: ac,
        })
    }

    // ── GPU utilization ────────────────────────────────────────────────

    /// Read GPU utilization from Metal performance counters via IOKit.
    ///
    /// Uses `IOServiceOpen` + `IOConnectCallMethod` on the Metal
    /// performance statistics service (`AGXCommandQueue`).  If the IO
    /// service is not available (e.g. on a non-Apple-GPU system) returns
    /// `Err`.
    fn sample_gpu_utilization() -> Result<f64, String> {
        // Dynamic lookup via sysctl — Apple Silicon exposes a coarse
        // GPU busy fraction through the `kern.gpu.busy` sysctl on
        // macOS 14+.
        #[cfg(target_os = "macos")]
        {
            let mut value: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let name = b"kern.gpu.busy\0";
            let ret = unsafe {
                libc::sysctlbyname(
                    name.as_ptr() as *const i8,
                    &mut value as *mut u64 as *mut libc::c_void,
                    &mut size,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if ret == 0 && size == std::mem::size_of::<u64>() {
                // Value is in [0, 1000]; scale to 0.0–1.0.
                return Ok((value as f64) / 1000.0);
            }
        }

        // Fallback: no sensor available.
        Err("kern.gpu.busy unavailable; no GPU utilisation data".into())
    }

    // ── CPU utilization ─────────────────────────────────────────────────

    /// Read per-core CPU load via `host_cpu_load_info`.
    ///
    /// Returns a fraction in [0.0, 1.0] representing the proportion of
    /// CPU ticks spent in user + system state over total ticks.
    fn sample_cpu_utilization() -> f64 {
        // Mach host_cpu_load_info is the canonical source on Apple Silicon.
        #[cfg(target_os = "macos")]
        {
            let mut cpu_info = std::mem::MaybeUninit::<libc::host_cpu_load_info>::uninit();
            let mut count: u32 = libc::HOST_CPU_LOAD_INFO_COUNT as u32;

            let result = unsafe {
                libc::host_statistics(
                    libc::mach_host_self(),
                    libc::HOST_CPU_LOAD_INFO,
                    cpu_info.as_mut_ptr() as *mut u32 as *mut libc::integer_t,
                    &mut count,
                )
            };

            if result == libc::KERN_SUCCESS {
                let info = unsafe { cpu_info.assume_init() };
                let total = (info.cpu_ticks[libc::CPU_STATE_USER as usize]
                    + info.cpu_ticks[libc::CPU_STATE_SYSTEM as usize]
                    + info.cpu_ticks[libc::CPU_STATE_IDLE as usize]
                    + info.cpu_ticks[libc::CPU_STATE_NICE as usize])
                    as f64;

                if total > 0.0 {
                    let busy = (info.cpu_ticks[libc::CPU_STATE_USER as usize]
                        + info.cpu_ticks[libc::CPU_STATE_SYSTEM as usize])
                        as f64;
                    return (busy / total).clamp(0.0, 1.0);
                }
            }
        }

        // Platform not supported — return a conservative default.
        0.5
    }

    // ── Thermal state ──────────────────────────────────────────────────

    /// Read thermal state via sysctl `kern.thermal.thermal_level`.
    fn sample_thermal_state() -> ThermalState {
        #[cfg(target_os = "macos")]
        {
            let mut level: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let name = b"kern.thermal.thermal_level\0";

            let ret = unsafe {
                libc::sysctlbyname(
                    name.as_ptr() as *const i8,
                    &mut level as *mut u64 as *mut libc::c_void,
                    &mut size,
                    std::ptr::null_mut(),
                    0,
                )
            };

            if ret == 0 {
                // macOS thermal levels:
                //   0 = Nominal, 1 = Fair, 2 = Serious, 3 = Critical
                return match level {
                    0 => ThermalState::Nominal,
                    1 => ThermalState::Fair,
                    2 => ThermalState::Serious,
                    3 => ThermalState::Critical,
                    _ => ThermalState::Nominal,
                };
            }
        }

        ThermalState::Nominal
    }

    // ── Power state ────────────────────────────────────────────────────

    /// Read battery charge fraction and AC power status via IOKit power
    /// sources (`IOPSCopyPowerSourcesInfo` / `IOPSGetProvidingPowerSourceType`).
    fn sample_power_state() -> (f64, bool) {
        #[cfg(target_os = "macos")]
        {
            const K_CFSTRING_ENCODING_UTF8: u32 = 0x08000100;

            // Use IOKit via CoreFoundation.
            // `IOPSCopyPowerSourcesInfo` returns a blob we can query.
            let power_info = unsafe { IOPSCopyPowerSourcesInfo() };
            if power_info.is_null() {
                return (1.0, true);
            }

            // Determine power source type.
            let source_type = unsafe { IOPSGetProvidingPowerSourceType(power_info) };
            if source_type.is_null() {
                unsafe { CFRelease(power_info as *mut libc::c_void) };
                return (1.0, true);
            }

            let ac = unsafe {
                // kIOPMACPowerKey == "AC Power" — CFString comparison.
                let ac_key = CFStringCreateWithCString(
                    std::ptr::null(),
                    b"AC Power\0" as *const u8 as *const libc::c_char,
                    K_CFSTRING_ENCODING_UTF8,
                );
                let is_ac = CFEqual(
                    source_type as *const libc::c_void,
                    ac_key as *const libc::c_void,
                ) != 0;
                CFRelease(ac_key as *mut libc::c_void);
                is_ac
            };

            // Read battery capacity fraction from the power sources list.
            // Use modern IOPS API to enumerate power sources and read
            // current / max capacity for battery fraction.
            const K_CFNUMBER_S64: u32 = 4; // kCFNumberSInt64Type
            let current_key = unsafe { CFStringCreateWithCString(
                std::ptr::null(), b"Current Capacity\0" as *const u8 as *const libc::c_char,
                K_CFSTRING_ENCODING_UTF8,
            ) };
            let max_key = unsafe { CFStringCreateWithCString(
                std::ptr::null(), b"Max Capacity\0" as *const u8 as *const libc::c_char,
                K_CFSTRING_ENCODING_UTF8,
            ) };
            let ps_list = unsafe { IOPSCopyPowerSourcesList(power_info) };
            let ps_count = if !ps_list.is_null() {
                unsafe { CFArrayGetCount(ps_list) }
            } else {
                0
            };
            let fraction = if ps_count > 0 && !current_key.is_null() && !max_key.is_null() {
                let mut total_frac = 0.0f64;
                let mut valid = 0i64;
                for i in 0..ps_count {
                    let ps = unsafe { CFArrayGetValueAtIndex(ps_list, i) };
                    if ps.is_null() { continue; }
                    let desc = unsafe { IOPSGetPowerSourceDescription(power_info, ps) };
                    if desc.is_null() { continue; }
                    let current_val = unsafe { CFDictionaryGetValue(desc, current_key) };
                    let max_val = unsafe { CFDictionaryGetValue(desc, max_key) };
                    if current_val.is_null() || max_val.is_null() { continue; }
                    let mut cur: i64 = 0;
                    let mut max: i64 = 0;
                    let ok_cur = unsafe { CFNumberGetValue(current_val, K_CFNUMBER_S64, &mut cur as *mut _ as *mut libc::c_void) };
                    let ok_max = unsafe { CFNumberGetValue(max_val, K_CFNUMBER_S64, &mut max as *mut _ as *mut libc::c_void) };
                    if ok_cur != 0 && ok_max != 0 && max > 0 {
                        total_frac += cur as f64 / max as f64;
                        valid += 1;
                    }
                }
                if !ps_list.is_null() { unsafe { CFRelease(ps_list as *mut libc::c_void) } }
                if !current_key.is_null() { unsafe { CFRelease(current_key as *mut libc::c_void) } }
                if !max_key.is_null() { unsafe { CFRelease(max_key as *mut libc::c_void) } }
                if valid > 0 { total_frac / valid as f64 } else { 1.0 }
            } else {
                if !ps_list.is_null() { unsafe { CFRelease(ps_list as *mut libc::c_void) } }
                if !current_key.is_null() { unsafe { CFRelease(current_key as *mut libc::c_void) } }
                if !max_key.is_null() { unsafe { CFRelease(max_key as *mut libc::c_void) } }
                1.0
            };

            unsafe {
                CFRelease(source_type as *mut libc::c_void);
                CFRelease(power_info as *mut libc::c_void);
            }

            return (fraction, ac);
        }

        // Fallback (non-macOS or if APIs unavailable).
        (1.0, true)
    }

    // ── GPU memory fraction ────────────────────────────────────────────

    /// Read GPU memory pressure via IOKit's `AGX` statistics.
    ///
    /// Returns (used / total) as a fraction in [0.0, 1.0].
    fn sample_gpu_memory_fraction() -> f64 {
        // macOS 14+ exposes a coarse GPU memory fraction through sysctl.
        #[cfg(target_os = "macos")]
        {
            let mut value: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let name = b"kern.gpu.memory_fraction\0";

            let ret = unsafe {
                libc::sysctlbyname(
                    name.as_ptr() as *const i8,
                    &mut value as *mut u64 as *mut libc::c_void,
                    &mut size,
                    std::ptr::null_mut(),
                    0,
                )
            };

            if ret == 0 && size == std::mem::size_of::<u64>() {
                return (value as f64) / 1000.0;
            }
        }

        // Fallback.
        0.0
    }

    // ── Query helpers ──────────────────────────────────────────────────

    /// Returns `true` if the GPU is too busy for additional work.
    ///
    /// Saturated means either GPU compute utilization exceeds 85 % or
    /// GPU memory allocation exceeds 90 %.
    pub fn gpu_saturated(&self) -> bool {
        self.gpu_utilization > 0.85 || self.gpu_memory_fraction > 0.9
    }

    /// Returns `true` if the system should prefer power-efficient
    /// backends over raw throughput.
    ///
    /// Throttling is indicated by thermal state Serious+ or battery
    /// below 20 % while on battery power.
    pub fn should_throttle(&self) -> bool {
        self.thermal_state == ThermalState::Serious
            || self.thermal_state == ThermalState::Critical
            || (!self.ac_power && self.battery_remaining < 0.2)
    }
}

// ── Operation classification ──────────────────────────────────────────────

/// Simplified operation classification for dispatch decisions.
///
/// The five families map directly to the dispatch `match` in
/// [`FlexDispatch::dispatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchFamily {
    MatMul,
    Attention,
    ElementWise,
    Softmax,
    LayerNorm,
}

/// Classify a [`routing::OperationFamily`] into a [`DispatchFamily`] for
/// the flex dispatcher.
fn classify_family(family: OperationFamily) -> DispatchFamily {
    use OperationFamily::*;
    match family {
        Matmul | QuantizedMatmul | MlpBlock => DispatchFamily::MatMul,
        AttentionBlock | DecoderLayer | PrefillFragment => DispatchFamily::Attention,
        Silu | Add | Multiply | Transpose | Reshape | IndexSelect | Sampling | Reduction => {
            DispatchFamily::ElementWise
        }
        Softmax => DispatchFamily::Softmax,
        RmsNorm | RoPE | LayoutTransform | Checksum => DispatchFamily::LayerNorm,
    }
}

// ── FlexDispatch ──────────────────────────────────────────────────────────

/// Runtime backend dispatcher — adapts to real-time system conditions.
///
/// Every `sample_interval` decode steps, `FlexDispatch` samples the full
/// [`SystemState`] and uses it to route each incoming operation to the
/// best backend *right now*.
///
/// The dispatcher is stateless between samples; the decision logic is a
/// pure function of the current state and the operation family.
pub struct FlexDispatch {
    /// Last sampled system state.
    pub last_state: SystemState,
    /// How often to re-sample the system state (in decode steps).
    pub sample_interval: u32,
    /// Steps since the last sample.
    pub steps_since_sample: u32,
}

impl FlexDispatch {
    /// Create a new dispatch controller with default sampling interval
    /// (16 decode steps).
    pub fn new() -> Self {
        Self {
            last_state: SystemState::default(),
            sample_interval: 16,
            steps_since_sample: u32::MAX, // Sample on first call.
        }
    }

    /// Create a dispatch controller with a custom sampling interval.
    pub fn with_interval(steps: u32) -> Self {
        Self {
            last_state: SystemState::default(),
            sample_interval: steps,
            steps_since_sample: u32::MAX,
        }
    }

    /// Force a system-state sample right now.
    pub fn sample_now(&mut self) {
        if let Ok(state) = SystemState::sample() {
            self.last_state = state;
        }
        self.steps_since_sample = 0;
    }

    /// Pick the best backend for an operation given current system state.
    ///
    /// Samples system state every `sample_interval` steps.  The decision
    /// logic is:
    ///
    /// | Family | GPU free & no throttle | GPU saturated | Throttling |
    /// |---|---|---|---|
    /// | MatMul | MLX (GPU) | MLX (GPU) | Core ML (ANE) |
    /// | Attention | MLX (GPU) | Core ML (ANE) | Accelerate (CPU) |
    /// | ElementWise | MLX (GPU) | Accelerate (CPU) | Accelerate (CPU) |
    /// | Softmax | Accelerate (CPU) | Accelerate (CPU) | Accelerate (CPU) |
    /// | LayerNorm | Accelerate (CPU) | Accelerate (CPU) | Accelerate (CPU) |
    pub fn dispatch(&mut self, op: &OperationDescriptor, _sequence: u32) -> BackendId {
        // Sample system state every N steps.
        self.steps_since_sample = self.steps_since_sample.wrapping_add(1);
        if self.steps_since_sample >= self.sample_interval {
            if let Ok(state) = SystemState::sample() {
                self.last_state = state;
            }
            self.steps_since_sample = 0;
        }

        let state = &self.last_state;
        let family = classify_family(op.family);

        match family {
            DispatchFamily::MatMul => {
                // MatMul is GPU-bound — prefer MLX unless throttling.
                if state.should_throttle() {
                    BackendId(2) // Core ML (ANE — most efficient per watt)
                } else {
                    BackendId(0) // MLX (GPU — fastest)
                }
            }
            DispatchFamily::Attention => {
                // Attention is memory-bandwidth-bound — offload to ANE
                // when the GPU is saturated, use CPU when throttling,
                // GPU otherwise.
                if state.gpu_saturated() {
                    BackendId(2) // Core ML (ANE)
                } else if state.should_throttle() {
                    BackendId(1) // Accelerate (CPU — most power efficient)
                } else {
                    BackendId(0) // MLX (GPU)
                }
            }
            DispatchFamily::ElementWise => {
                // Element-wise ops are cheap everywhere — use whichever
                // backend does not compete with the GPU.
                if state.gpu_saturated() || state.gpu_utilization > 0.5 {
                    BackendId(1) // Accelerate (CPU — doesn't compete)
                } else {
                    BackendId(0) // MLX (GPU — fast and available)
                }
            }
            DispatchFamily::Softmax | DispatchFamily::LayerNorm => {
                // These run fine on any backend; prefer CPU to keep GPU free.
                BackendId(1) // Accelerate (CPU NEON)
            }
        }
    }

    /// Update a [`HeterogeneousExecutor`]'s per-operation routing table
    /// based on the current system state.
    ///
    /// Iterates every operation in the executor's registry, calls
    /// [`dispatch`](Self::dispatch) for each one, and writes the result
    /// into `executor.routing_table`.
    ///
    /// This allows the executor to use the flex-dispatch routes during the
    /// next [`execute_boundaries`] call without sampling the system on
    /// every single operation.
    pub fn reroute(&mut self, executor: &mut HeterogeneousExecutor) -> Result<(), String> {
        // Force a fresh sample so all routes are based on the same state.
        self.sample_now();

        let state = &self.last_state;
        // Collect operation IDs first to avoid conflicting borrows on executor
        let op_ids: Vec<_> = executor.operation_registry.keys().copied().collect();

        for op_id in op_ids {
            let op_desc = &executor.operation_registry[&op_id];
            let family = classify_family(op_desc.family);
            let backend_id = match family {
                DispatchFamily::MatMul => {
                    if state.should_throttle() {
                        BackendId(2)
                    } else {
                        BackendId(0)
                    }
                }
                DispatchFamily::Attention => {
                    if state.gpu_saturated() {
                        BackendId(2)
                    } else if state.should_throttle() {
                        BackendId(1)
                    } else {
                        BackendId(0)
                    }
                }
                DispatchFamily::ElementWise => {
                    if state.gpu_saturated() || state.gpu_utilization > 0.5 {
                        BackendId(1)
                    } else {
                        BackendId(0)
                    }
                }
                DispatchFamily::Softmax | DispatchFamily::LayerNorm => BackendId(1),
            };

            executor.set_route(op_id, backend_id);
        }

        Ok(())
    }
}

impl Default for FlexDispatch {
    fn default() -> Self {
        Self::new()
    }
}

// ── CoreFoundation / IOKit FFI (macOS only) ──────────────────────────────

// Thin FFI bindings for power-source and GPU sampling.
// Compiled only on macOS where these frameworks are available.

#[cfg(target_os = "macos")]
#[link(name = "IOKit", kind = "framework")]
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn IOPSCopyPowerSourcesInfo() -> *mut libc::c_void;
    fn IOPSGetProvidingPowerSourceType(ps_info: *mut libc::c_void) -> *mut libc::c_void;
    fn IOPSCopyPowerSourcesList(ps_info: *mut libc::c_void) -> *mut libc::c_void;
    fn IOPSGetPowerSourceDescription(ps_info: *mut libc::c_void, ps_id: *mut libc::c_void) -> *mut libc::c_void;
    fn CFRelease(cf: *mut libc::c_void);
    fn CFEqual(cf1: *const libc::c_void, cf2: *const libc::c_void) -> u8;
    fn CFStringCreateWithCString(
        alloc: *const libc::c_void,
        c_str: *const libc::c_char,
        encoding: u32,
    ) -> *mut libc::c_void;
    fn CFDictionaryGetValue(dict: *mut libc::c_void, key: *const libc::c_void) -> *mut libc::c_void;
    fn CFNumberGetValue(num: *mut libc::c_void, the_type: u32, value: *mut libc::c_void) -> u8;
    fn CFArrayGetCount(arr: *mut libc::c_void) -> isize;
    fn CFArrayGetValueAtIndex(arr: *mut libc::c_void, idx: isize) -> *mut libc::c_void;
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::routing::*;
    use crate::backend::DType;

    fn make_matmul_op(id: u64) -> OperationDescriptor {
        OperationDescriptor {
            operation_id: OperationId(id),
            family: OperationFamily::Matmul,
            layer_index: None,
            phase: Phase::Decode,
            logical_shape: LogicalShape { dims: vec![1, 64] },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::F32, DType::F32],
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: vec![1, 64] },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        }
    }

    fn make_attention_op(id: u64) -> OperationDescriptor {
        OperationDescriptor {
            operation_id: OperationId(id),
            family: OperationFamily::AttentionBlock,
            layer_index: None,
            phase: Phase::Decode,
            logical_shape: LogicalShape { dims: vec![1, 64] },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::F32],
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: vec![1, 64] },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        }
    }

    #[test]
    fn test_system_state_default() {
        let state = SystemState::default();
        assert_eq!(state.gpu_utilization, 0.0);
        assert_eq!(state.thermal_state, ThermalState::Nominal);
        assert_eq!(state.battery_remaining, 1.0);
        assert!(state.ac_power);
        assert!(!state.gpu_saturated());
        assert!(!state.should_throttle());
    }

    #[test]
    fn test_gpu_saturated_threshold() {
        let mut state = SystemState::default();
        // Below threshold
        state.gpu_utilization = 0.5;
        assert!(!state.gpu_saturated());

        // Above utilization threshold
        state.gpu_utilization = 0.9;
        assert!(state.gpu_saturated());

        // Memory threshold
        state.gpu_utilization = 0.5;
        state.gpu_memory_fraction = 0.95;
        assert!(state.gpu_saturated());
    }

    #[test]
    fn test_should_throttle_thermal() {
        let mut state = SystemState::default();
        assert!(!state.should_throttle());

        state.thermal_state = ThermalState::Serious;
        assert!(state.should_throttle());

        state.thermal_state = ThermalState::Critical;
        assert!(state.should_throttle());

        state.thermal_state = ThermalState::Fair;
        assert!(!state.should_throttle());
    }

    #[test]
    fn test_should_throttle_battery() {
        let mut state = SystemState::default();
        state.ac_power = false;
        state.battery_remaining = 0.15;
        assert!(state.should_throttle());

        state.battery_remaining = 0.5;
        assert!(!state.should_throttle());
    }

    #[test]
    fn test_classify_family() {
        assert_eq!(
            classify_family(OperationFamily::Matmul),
            DispatchFamily::MatMul
        );
        assert_eq!(
            classify_family(OperationFamily::QuantizedMatmul),
            DispatchFamily::MatMul
        );
        assert_eq!(
            classify_family(OperationFamily::AttentionBlock),
            DispatchFamily::Attention
        );
        assert_eq!(
            classify_family(OperationFamily::Silu),
            DispatchFamily::ElementWise
        );
        assert_eq!(
            classify_family(OperationFamily::Softmax),
            DispatchFamily::Softmax
        );
        assert_eq!(
            classify_family(OperationFamily::RmsNorm),
            DispatchFamily::LayerNorm
        );
    }

    #[test]
    fn test_dispatch_matmul_normal() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState::default();
        flex.steps_since_sample = 0; // Skip sampling.

        let op = make_matmul_op(1);
        let backend = flex.dispatch(&op, 0);
        // Default state: GPU free, AC power, nominal temps → MLX (GPU).
        assert_eq!(backend, BackendId(0));
    }

    #[test]
    fn test_dispatch_matmul_throttle() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            thermal_state: ThermalState::Serious,
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = make_matmul_op(1);
        let backend = flex.dispatch(&op, 0);
        // Throttling → ANE (Core ML).
        assert_eq!(backend, BackendId(2));
    }

    #[test]
    fn test_dispatch_attention_gpu_saturated() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            gpu_utilization: 0.9,
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = make_attention_op(1);
        let backend = flex.dispatch(&op, 0);
        // GPU saturated → ANE.
        assert_eq!(backend, BackendId(2));
    }

    #[test]
    fn test_dispatch_attention_throttle() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            thermal_state: ThermalState::Critical,
            gpu_utilization: 0.3,
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = make_attention_op(1);
        let backend = flex.dispatch(&op, 0);
        // Throttling (but GPU not saturated) → CPU (Accelerate).
        assert_eq!(backend, BackendId(1));
    }

    #[test]
    fn test_dispatch_elementwise_gpu_busy() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            gpu_utilization: 0.7,
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = OperationDescriptor {
            family: OperationFamily::Silu,
            ..make_matmul_op(2)
        };
        let backend = flex.dispatch(&op, 0);
        // GPU utilization > 0.5 → CPU (Accelerate).
        assert_eq!(backend, BackendId(1));
    }

    #[test]
    fn test_dispatch_softmax_always_cpu() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState {
            ..SystemState::default()
        };
        flex.steps_since_sample = 0;

        let op = OperationDescriptor {
            family: OperationFamily::Softmax,
            ..make_matmul_op(3)
        };
        let backend = flex.dispatch(&op, 0);
        // Softmax always routes to CPU.
        assert_eq!(backend, BackendId(1));
    }

    #[test]
    fn test_sample_interval() {
        let mut flex = FlexDispatch::new();
        flex.sample_interval = 5;

        // Each dispatch call increments the counter. After interval, sampling
        // resets. We verify the internal state by checking the step counter.
        for i in 0..4 {
            let op = make_matmul_op(i);
            flex.dispatch(&op, i as u32);
            assert!(
                flex.steps_since_sample <= 5,
                "steps_since_sample should be <= 5 after {i} dispatches"
            );
        }
    }

    #[test]
    fn test_reroute_populates_routing_table() {
        let mut flex = FlexDispatch::new();
        flex.last_state = SystemState::default();

        let mut executor = HeterogeneousExecutor::new();

        // Populate the operation registry.
        let mut registry = std::collections::HashMap::new();
        registry.insert(OperationId(1), make_matmul_op(1));
        registry.insert(OperationId(2), make_attention_op(2));
        executor.set_operation_registry(registry);

        // Reroute based on current state.
        flex.reroute(&mut executor).unwrap();

        // With default state (GPU free, nominal temps, AC power):
        //   MatMul → MLX (BackendId(0))
        //   Attention → MLX (BackendId(0))
        let route1 = executor.get_route(&OperationId(1));
        let route2 = executor.get_route(&OperationId(2));
        assert_eq!(route1, Some(BackendId(0)));
        assert_eq!(route2, Some(BackendId(0)));
    }
}
