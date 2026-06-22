//! Metal compute kernel dispatch — launches fused Metal kernels.
//!
//! Uses the `metal` crate to create command buffers, set pipeline state,
//! bind buffers from the unified arena, and dispatch compute work.

use crate::worker_dispatch::LoadedMetalKernel;
use metal::MTLDevice;

/// Dispatch a fused Metal kernel and return execution time in microseconds.
pub fn dispatch_fused_kernel(
    kernel: &LoadedMetalKernel,
    device: &metal::Device,
    command_buffer: &metal::CommandBufferRef,
) -> Result<u64, String> {
    let start = std::time::Instant::now();

    // Compile library and create pipeline state from the loaded metallib data.
    let lib = device.new_library_with_data(kernel.library_data())
        .map_err(|e| format!("failed to create Metal library: {:?}", e))?;
    let function = lib.get_function(kernel.function_name(), None)
        .map_err(|e| format!("failed to get entry point '{}': {:?}", kernel.function_name(), e))?;
    let pipeline_state = device.new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("failed to create compute pipeline state: {:?}", e))?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline_state);

    // Bind buffers according to the dispatch recipe.
    // The buffer_slot_map maps buffer indices to logical names.
    // Actual Metal buffers are bound from the arena outside this function.
    // For now, buffer binding is handled by the caller via encoder.set_buffer().

    let tg = kernel.artifact.dispatch.threads_per_threadgroup;
    let gg = kernel.artifact.dispatch.threadgroups_per_grid;

    encoder.dispatch_thread_groups(
        metal::MTLSize {
            width: gg[0] as u64,
            height: gg[1] as u64,
            depth: gg[2] as u64,
        },
        metal::MTLSize {
            width: tg[0] as u64,
            height: tg[1] as u64,
            depth: tg[2] as u64,
        },
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(start.elapsed().as_micros() as u64)
}
