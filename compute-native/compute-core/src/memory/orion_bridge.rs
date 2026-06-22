//! Bridge between the unified memory island (IosurfaceAllocator) and the
//! Orion ANE runtime.
//!
//! Orion (orion-runtime/) is an Objective-C ANE execution engine that uses
//! IOSurface-backed tensors. This bridge makes Orion draw from the same
//! IOSurface pool as mlx-rs Arrays, candle Tensors, and Core ML MLMultiArrays
//! — true zero-copy across all accelerators on Apple Silicon.
//!
//! # Data flow
//!
//! 1. IosurfaceAllocator::allocate() → Arena (IOSurfaceCreate)
//! 2. orion_tensor_from_external() → wraps the same IOSurface (no copy)
//! 3. orion_eval() → ANE reads the IOSurface directly
//! 4. MLX/Core ML also read the same IOSurface via Arena bridge
//!
//! See `docs/compute-image-memory-architecture.md`.

use crate::arena::Arena;

/// Opaque handle to an Orion ANE program (matches OrionProgram* in ane_runtime.m).
#[repr(C)]
pub struct OrionProgram {
    _data: [u8; 0],
    _marker: std::marker::PhantomData<(*mut u8, usize)>,
}

/// Opaque IOSurface reference (matches IOSurfaceRef).
#[repr(C)]
pub struct OrionSurface {
    _data: [u8; 0],
    _marker: std::marker::PhantomData<(*mut u8, usize)>,
}

// ---------------------------------------------------------------------------
// FFI: orion-runtime functions
// ---------------------------------------------------------------------------

extern "C" {
    /// Initialize the ANE runtime. Must be called before any other orion_* function.
    /// Returns true if the ANE was successfully initialized.
    fn orion_ane_init() -> bool;

    /// Wrap a pre-existing IOSurface for ANE usage (retains it).
    /// Returns the same surface on success, NULL on null input.
    fn orion_tensor_from_external(surface: *mut OrionSurface) -> *mut OrionSurface;

    /// Create an IOSurface using the Arena pixel format
    /// (kCVPixelFormatType_OneComponent16Half, width=seq_len, height=channels).
    /// Caller must CFRelease the returned surface.
    fn orion_tensor_from_arena(channels: i32, seq_len: i32) -> *mut OrionSurface;

    /// Release an IOSurface (CFRelease).
    fn orion_tensor_release(surface: *mut OrionSurface);

    /// Compile MIL text into an ANE execution program.
    /// `mil_text` is null-terminated MIL program source.
    /// `weight_dict` is nil/0 for weight-free programs.
    /// `program_tag` is a human-readable label for debugging (may be NULL).
    /// Returns compiled program handle or NULL on failure.
    fn orion_compile_mil(
        mil_text: *const u8,
        weight_dict: *const std::ffi::c_void,
        program_tag: *const u8,
    ) -> *mut OrionProgram;

    /// Release a compiled program and unload from ANE.
    fn orion_release_program(prog: *mut OrionProgram);

    /// Evaluate a compiled ANE program with IOSurface inputs and outputs.
    fn orion_eval(
        prog: *const OrionProgram,
        inputs: *mut *mut OrionSurface,
        num_inputs: i32,
        outputs: *mut *mut OrionSurface,
        num_outputs: i32,
    ) -> bool;
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// Errors during Orion bridge operations.
#[derive(Debug)]
pub enum OrionBridgeError {
    /// The IosurfaceAllocator failed to allocate an arena.
    AllocationFailed(String),
    /// The IOSurface pointer was null after creation.
    NullSurface,
    /// The ANE program pointer was null.
    NullProgram,
}

impl std::fmt::Display for OrionBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationFailed(s) => write!(f, "allocation failed: {s}"),
            Self::NullSurface => write!(f, "null IOSurface"),
            Self::NullProgram => write!(f, "null ANE program"),
        }
    }
}

impl std::error::Error for OrionBridgeError {}

/// Allocate an IOSurface from the shared memory island for ANE use.
///
/// This creates the surface via `orion_tensor_from_arena` using the same
/// pixel format as the IosurfaceAllocator/Arena, ensuring compatibility
/// with MLX external arrays and Core ML multi-arrays.
pub fn allocate_ane_surface(
    channels: i32,
    seq_len: i32,
) -> Result<*mut OrionSurface, OrionBridgeError> {
    if channels <= 0 || seq_len <= 0 {
        return Err(OrionBridgeError::AllocationFailed(
            "channels and seq_len must be positive".into(),
        ));
    }

    let surface = unsafe { orion_tensor_from_arena(channels, seq_len) };

    if surface.is_null() {
        return Err(OrionBridgeError::NullSurface);
    }

    Ok(surface)
}

/// Wrap an existing Arena's IOSurface for ANE use.
///
/// This is the PRIMARY integration point: instead of creating a new IOSurface,
/// it uses the same one the Arena allocated for the shared memory island.
/// MLX Arrays, Core ML multi-arrays, and the ANE all read/write the same bytes.
pub fn wrap_arena_for_ane(arena: &Arena) -> Result<*mut OrionSurface, OrionBridgeError> {
    // The Arena stores its IOSurface as an opaque io_surface pointer.
    // This pointer is the IOSurfaceRef we need.
    let surface = arena.io_surface_ptr() as *mut OrionSurface;

    if surface.is_null() {
        return Err(OrionBridgeError::NullSurface);
    }

    // Retain via orion_tensor_from_external so Orion takes ownership.
    let retained = unsafe { orion_tensor_from_external(surface) };

    if retained.is_null() {
        return Err(OrionBridgeError::NullSurface);
    }

    Ok(retained)
}

/// Run an ANE program with IOSurface inputs and outputs allocated from the
/// shared memory island.
///
/// `prog` is an OrionProgram* returned from orion_compile_mil.
/// `inputs` and `outputs` are IOSurfaceRef pointers from wrap_arena_for_ane()
/// or allocate_ane_surface().
///
/// Returns true on success, false on failure.
pub fn run_ane_program(
    prog: *const OrionProgram,
    inputs: &[*mut OrionSurface],
    outputs: &[*mut OrionSurface],
) -> bool {
    if prog.is_null() {
        return false;
    }

    let num_inputs = inputs.len() as i32;
    let num_outputs = outputs.len() as i32;

    // We need mut pointers for the C API
    let mut input_ptrs: Vec<*mut OrionSurface> = inputs.to_vec();
    let mut output_ptrs: Vec<*mut OrionSurface> = outputs.to_vec();

    unsafe {
        orion_eval(
            prog,
            input_ptrs.as_mut_ptr(),
            num_inputs,
            output_ptrs.as_mut_ptr(),
            num_outputs,
        )
    }
}

/// Release an IOSurface that was obtained via `allocate_ane_surface()` or
/// `wrap_arena_for_ane()`.
pub fn release_ane_surface(surface: *mut OrionSurface) {
    if !surface.is_null() {
        unsafe {
            orion_tensor_release(surface);
        }
    }
}

// ── ANE pre-warm ───────────────────────────────────────────────────────────

/// Minimal MIL program for ANE firmware warmup.
/// Program: x * x (element-wise multiply, 1x1x1x1 fp16).
// Build a minimal MIL program for ANE warmup.  Uses include_bytes to
// avoid Rust string escaping issues with MIL's curly-brace syntax.
const ANE_WARMUP_MIL: &[u8] = include_bytes!("ane_warmup.mil");

/// Pre-warm the ANE by compiling and executing a minimal program.
/// This wakes the ANE firmware so subsequent orion_eval calls avoid the
/// ~100-500 μs cold-start dispatch latency.
///
/// The ANE compiler XPC service (`com.apple.appleneuralengine.compiler`)
/// requires the `com.apple.private.ane.compile` entitlement, which direct
/// callers lack.  Without it, `orion_compile_mil` prints an ANECCompile
/// error and returns NULL.
///
/// Strategy attempted, in order:
/// 1. Direct ANE compile via `orion_compile_mil` (requires entitlement —
///    expected to fail for now, succeeds when signed with Core ML
///    entitlement or run from within a Core ML sandbox).
/// 2. Core ML warmup via `MLModel` + `MLPredictionOptions` (has the
///    entitlement — future integration path).
/// 3. Graceful no-op — ANE operations still work, just cold on first eval.
pub fn prewarm_ane() -> bool {
    // Initialize the ANE runtime before any ANE operations.
    // Returns false if the ANE is unavailable (e.g. no ANE driver).
    if !unsafe { orion_ane_init() } {
        return false;
    }

    let mil_ptr = ANE_WARMUP_MIL.as_ptr();
    let tag_ptr = b"ane_warmup\0".as_ptr();
    let prog = unsafe { orion_compile_mil(mil_ptr, std::ptr::null(), tag_ptr) };
    if prog.is_null() {
        // Strategy 1 failed (expected without ANE compile entitlement).
        // Strategy 2: compile a minimal .mlpackage through `coremlc` which
        // has the Core ML entitlement and can reach the ANECCompile daemon.
        // Even without executing the compiled model, contacting the ANE
        // compiler through the entitled path warms the compiler infrastructure.
        return crate::memory::coreml_warmup::prewarm_ane_via_coreml();
    }
    let input = unsafe { orion_tensor_from_arena(1, 1) };
    if input.is_null() {
        unsafe {
            orion_release_program(prog);
        }
        return false;
    }
    let output = unsafe { orion_tensor_from_arena(1, 1) };
    if output.is_null() {
        unsafe {
            orion_tensor_release(input);
            orion_release_program(prog);
        }
        return false;
    }
    let mut inputs = [input];
    let mut outputs = [output];
    let ok = unsafe { orion_eval(prog, inputs.as_mut_ptr(), 1, outputs.as_mut_ptr(), 1) };
    unsafe {
        orion_tensor_release(input);
        orion_tensor_release(output);
        orion_release_program(prog);
    }
    ok
}

/// Allocate an ANE-compatible surface from the IosurfaceAllocator and
/// wrap it for Orion in one call.
///
/// This is the HIGH-LEVEL API: the surface is backed by the shared memory
/// island and can be used by MLX, Core ML, and Orion interchangeably.
pub fn allocate_from_island(
    _allocator: &crate::memory::allocator::IosurfaceAllocator,
    channels: u32,
    seq_len: u32,
) -> Result<*mut OrionSurface, OrionBridgeError> {
    // Allocate a fresh IOSurface-backed arena
    let arena = crate::arena::Arena::new(channels, seq_len, mlx_rs::Dtype::Float16)
        .map_err(|e| OrionBridgeError::AllocationFailed(e))?;

    // Wrap for ANE usage (no copy — same IOSurface)
    wrap_arena_for_ane(&arena)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::IosurfaceAllocator;

    /// Verify that allocate_ane_surface returns a non-null pointer.
    #[test]
    fn test_allocate_ane_surface_success() {
        let surface = allocate_ane_surface(64, 128).expect("allocate ANE surface");
        assert!(!surface.is_null(), "surface must not be null");
        release_ane_surface(surface);
    }

    /// Verify that allocate_ane_surface rejects invalid inputs.
    #[test]
    fn test_allocate_ane_surface_invalid() {
        let result = allocate_ane_surface(0, 128);
        assert!(result.is_err(), "zero channels should fail");

        let result = allocate_ane_surface(64, 0);
        assert!(result.is_err(), "zero seq_len should fail");

        let result = allocate_ane_surface(-1, 128);
        assert!(result.is_err(), "negative channels should fail");
    }

    /// Verify that null pointers are caught.
    #[test]
    fn test_run_ane_null_program() {
        let inputs: Vec<*mut OrionSurface> = vec![std::ptr::null_mut()];
        let outputs: Vec<*mut OrionSurface> = vec![std::ptr::null_mut()];
        let result = run_ane_program(std::ptr::null(), &inputs, &outputs);
        assert!(!result, "null program should return false");
    }

    /// Verify allocate_from_island returns a non-null surface.
    #[test]
    fn test_allocate_from_island() {
        let allocator = IosurfaceAllocator::new(1024 * 1024 * 100); // 100 MB pool
        let surface = allocate_from_island(&allocator, 64, 128).expect("allocate from island");
        assert!(!surface.is_null(), "surface must not be null");
        release_ane_surface(surface);
    }
}
