// Tensix FFI bridge — C API implementation for Tenstorrent device operations.
//
// Compiled in one of two modes:
//   -DTENSIX_STUB_MODE  (default, macOS dev):  all ops return errors
//   -DTENSIX_REAL_MODE  (Linux + TT_METAL_HOME): delegates to Metalium/TTNN
//
// Stub mode produces a linkable .o with all symbols defined, so Rust's
// extern "C" bindings link cleanly on any host.

#include "bridge.h"

#include <cstdint>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>

// ── Error helpers ──────────────────────────────────────────────────────────

/// Fill a TensixError for return to the caller.
static void set_error(TensixError* err, int32_t code, const char* fmt, ...) {
    if (!err) return;
    err->code = code;
    va_list args;
    va_start(args, fmt);
    vsnprintf(err->message, sizeof(err->message), fmt, args);
    va_end(args);
}

/// Clear the error (call before every fallible entry point).
static void clear_error(TensixError* err) {
    if (!err) return;
    err->code = 0;
    err->message[0] = '\0';
}

// ═══════════════════════════════════════════════════════════════════════════
//  STUB MODE  —  every real operation returns an error
// ═══════════════════════════════════════════════════════════════════════════
#ifdef TENSIX_STUB_MODE

// ── Device lifecycle ───────────────────────────────────────────────────────

TensixDevice* tensix_open_device(int32_t device_id, TensixError* err) {
    (void)device_id;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
    return NULL;
}

void tensix_close_device(TensixDevice* dev) {
    (void)dev;
    // no-op: stub device was never allocated
}

int32_t tensix_device_core_count(TensixDevice* dev) {
    (void)dev;
    return 0;
}

void tensix_device_arch(TensixDevice* dev, uint8_t* out, size_t out_len) {
    (void)dev;
    if (out && out_len > 0) {
        out[0] = '\0';
    }
}

// ── Buffer management ──────────────────────────────────────────────────────

TensixBuffer* tensix_allocate_buffer(
    TensixDevice* dev, uint64_t bytes,
    TensixMemoryType mem_type, TensixError* err)
{
    (void)dev;
    (void)bytes;
    (void)mem_type;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
    return NULL;
}

void tensix_deallocate_buffer(TensixBuffer* buf) {
    (void)buf;
    // no-op: stub buffer was never allocated
}

void tensix_write_to_buffer(
    TensixBuffer* buf, const float* data, uint64_t count,
    uint64_t offset, TensixError* err)
{
    (void)buf;
    (void)data;
    (void)count;
    (void)offset;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

void tensix_read_from_buffer(
    float* data, TensixBuffer* buf, uint64_t count,
    uint64_t offset, TensixError* err)
{
    (void)data;
    (void)buf;
    (void)count;
    (void)offset;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

// ── Op dispatch ────────────────────────────────────────────────────────────

void tensix_matmul(
    TensixDevice* dev, TensixBuffer* a, TensixBuffer* b,
    TensixBuffer* c, const TensixMatmulParams* params, TensixError* err)
{
    (void)dev; (void)a; (void)b; (void)c; (void)params;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

void tensix_sdpa(
    TensixDevice* dev, TensixBuffer* q, TensixBuffer* k, TensixBuffer* v,
    TensixBuffer* out, const TensixSdpaParams* params, TensixError* err)
{
    (void)dev; (void)q; (void)k; (void)v; (void)out; (void)params;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

void tensix_rms_norm(
    TensixDevice* dev, TensixBuffer* x, TensixBuffer* weight,
    TensixBuffer* out, const TensixNormParams* params, TensixError* err)
{
    (void)dev; (void)x; (void)weight; (void)out; (void)params;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

void tensix_rope(
    TensixDevice* dev, TensixBuffer* x, TensixBuffer* cos, TensixBuffer* sin,
    TensixBuffer* out, const TensixRopeParams* params, TensixError* err)
{
    (void)dev; (void)x; (void)cos; (void)sin; (void)out; (void)params;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

void tensix_silu(
    TensixDevice* dev, TensixBuffer* x, TensixBuffer* out, TensixError* err)
{
    (void)dev; (void)x; (void)out;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

void tensix_add(
    TensixDevice* dev, TensixBuffer* a, TensixBuffer* b,
    TensixBuffer* out, TensixError* err)
{
    (void)dev; (void)a; (void)b; (void)out;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

// ── Program compilation ────────────────────────────────────────────────────

TensixProgram* tensix_compile_program(
    TensixDevice* dev, const uint8_t* program_spec_json,
    size_t json_len, TensixError* err)
{
    (void)dev;
    (void)program_spec_json;
    (void)json_len;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
    return NULL;
}

void tensix_execute_program(
    TensixDevice* dev, TensixProgram* prog, TensixError* err)
{
    (void)dev;
    (void)prog;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

void tensix_free_program(TensixProgram* prog) {
    (void)prog;
    // no-op: stub program was never allocated
}

// ── Synchronization ────────────────────────────────────────────────────────

void tensix_synchronize_device(TensixDevice* dev) {
    (void)dev;
    // no-op in stub mode
}

// ── Profiling ──────────────────────────────────────────────────────────────

void tensix_read_profiler(
    TensixDevice* dev, TensixProfileEvent* event, TensixError* err)
{
    (void)dev;
    (void)event;
    set_error(err, -1, "Tensix requires Linux + tt-metal runtime");
}

void tensix_reset_profiler(TensixDevice* dev) {
    (void)dev;
    // no-op in stub mode
}

// ═══════════════════════════════════════════════════════════════════════════
//  REAL MODE  —  delegates to Metalium / TTNN C++ APIs
// ═══════════════════════════════════════════════════════════════════════════
#elif defined(TENSIX_REAL_MODE)

// ── Includes ───────────────────────────────────────────────────────────────
//
// These headers ship with tt-metal when TT_METAL_HOME is set.
// #include <tt-metalium/device.hpp>
// #include <tt-metalium/mesh_device.hpp>
// #include <tt-metalium/host_api.hpp>
// #include <tt-metalium/buffer.hpp>
// #include <tt-metalium/program.hpp>
// #include <tt-metalium/kernel.hpp>
// #include <device/tt_arch_types.h>
// #include <ttnn/operations/core/core.hpp>
// #include <ttnn/operations/matmul.hpp>
// #include <ttnn/operations/normalization.hpp>
// #include <ttnn/operations/embedding.hpp>
// #include <ttnn/operations/activation.hpp>
// #include <ttnn/operations/eltwise/binary.hpp>
// #include <fmt/core.h>
// #include <nlohmann/json.hpp>

// ── Device lifecycle ───────────────────────────────────────────────────────

TensixDevice* tensix_open_device(int32_t device_id, TensixError* err) {
    clear_error(err);
    /*
    TODO(REAL_MODE): Replace with real Metalium init.

    try {
        TensixDevice* dev = (TensixDevice*)std::malloc(sizeof(TensixDevice));
        // tt::tt_metal::MeshDeviceConfig config;
        // auto mesh = tt::tt_metal::MeshDevice::create(config);
        // dev->ptr = mesh->get_device(device_id).release();
        // dev->ptr = nullptr; // placeholder
        return dev;
    } catch (const std::exception& e) {
        set_error(err, -1, "tensix_open_device failed: %s", e.what());
        return NULL;
    }
    */
    (void)device_id;
    set_error(err, -1, "tensix_open_device: not yet implemented (REAL_MODE skeleton)");
    return NULL;
}

void tensix_close_device(TensixDevice* dev) {
    /*
    TODO(REAL_MODE):
    if (dev && dev->ptr) {
        auto* d = static_cast<tt::tt_metal::Device*>(dev->ptr);
        delete d;
    }
    */
    std::free(dev);
}

int32_t tensix_device_core_count(TensixDevice* dev) {
    /*
    TODO(REAL_MODE):
    if (dev && dev->ptr) {
        auto* d = static_cast<tt::tt_metal::Device*>(dev->ptr);
        return static_cast<int32_t>(d->compute_cores().size());
    }
    */
    (void)dev;
    return 0;
}

void tensix_device_arch(TensixDevice* dev, uint8_t* out, size_t out_len) {
    /*
    TODO(REAL_MODE):
    if (dev && dev->ptr && out && out_len > 0) {
        auto* d = static_cast<tt::tt_metal::Device*>(dev->ptr);
        auto arch = d->arch();
        const char* name = tt::tt_metal::get_arch_name(arch);
        snprintf((char*)out, out_len, "%s", name);
        return;
    }
    */
    (void)dev;
    if (out && out_len > 0) out[0] = '\0';
}

// ── Buffer management ──────────────────────────────────────────────────────

TensixBuffer* tensix_allocate_buffer(
    TensixDevice* dev, uint64_t bytes,
    TensixMemoryType mem_type, TensixError* err)
{
    /*
    TODO(REAL_MODE): Replace with real buffer allocation.

    clear_error(err);
    try {
        TensixBuffer* buf = (TensixBuffer*)std::malloc(sizeof(TensixBuffer));

        // Map TensixMemoryType -> tt::tt_metal::BufferType
        // tt::tt_metal::BufferType bt;
        // switch (mem_type) {
        //     case TensixMemoryType_DRAM:  bt = tt::tt_metal::BufferType::DRAM;  break;
        //     case TensixMemoryType_Host:  bt = tt::tt_metal::BufferType::SYSTEM_MEMORY; break;
        //     case TensixMemoryType_Trace: bt = tt::tt_metal::BufferType::TRACE; break;
        // }

        // buf->ptr = new tt::tt_metal::Buffer(bytes, ...);
        // buf->bytes = bytes;
        // return buf;
    } catch (...) {}
    */
    (void)dev; (void)bytes; (void)mem_type;
    set_error(err, -1, "tensix_allocate_buffer: not yet implemented (REAL_MODE skeleton)");
    return NULL;
}

void tensix_deallocate_buffer(TensixBuffer* buf) {
    /*
    TODO(REAL_MODE):
    if (buf && buf->ptr) {
        auto* b = static_cast<tt::tt_metal::Buffer*>(buf->ptr);
        delete b;
    }
    */
    std::free(buf);
}

void tensix_write_to_buffer(
    TensixBuffer* buf, const float* data, uint64_t count,
    uint64_t offset, TensixError* err)
{
    /*
    TODO(REAL_MODE):
    clear_error(err);
    if (!buf) { set_error(err, -1, "null buffer"); return; }
    // auto* b = static_cast<tt::tt_metal::Buffer*>(buf->ptr);
    // tt::tt_metal::detail::WriteToBuffer(*b, data);
    */
    (void)buf; (void)data; (void)count; (void)offset;
    set_error(err, -1, "tensix_write_to_buffer: not yet implemented (REAL_MODE skeleton)");
}

void tensix_read_from_buffer(
    float* data, TensixBuffer* buf, uint64_t count,
    uint64_t offset, TensixError* err)
{
    /*
    TODO(REAL_MODE):
    clear_error(err);
    if (!buf) { set_error(err, -1, "null buffer"); return; }
    // auto* b = static_cast<tt::tt_metal::Buffer*>(buf->ptr);
    // tt::tt_metal::detail::ReadFromBuffer(*b, data);
    */
    (void)data; (void)buf; (void)count; (void)offset;
    set_error(err, -1, "tensix_read_from_buffer: not yet implemented (REAL_MODE skeleton)");
}

// ── Op dispatch ────────────────────────────────────────────────────────────

void tensix_matmul(
    TensixDevice* dev, TensixBuffer* a, TensixBuffer* b,
    TensixBuffer* c, const TensixMatmulParams* params, TensixError* err)
{
    /*
    TODO(REAL_MODE):
    // ttnn::operations::matmul or TTNN matmul op:
    // auto input_a = ttnn::from_buffer(...);
    // auto input_b = ttnn::from_buffer(...);
    // auto output = ttnn::matmul(input_a, input_b, ...);
    // ttnn::to_buffer(output, c->ptr);
    */
    (void)dev; (void)a; (void)b; (void)c; (void)params;
    set_error(err, -1, "tensix_matmul: not yet implemented (REAL_MODE skeleton)");
}

void tensix_sdpa(
    TensixDevice* dev, TensixBuffer* q, TensixBuffer* k, TensixBuffer* v,
    TensixBuffer* out, const TensixSdpaParams* params, TensixError* err)
{
    /*
    TODO(REAL_MODE):
    // ttnn::transformer::scaled_dot_product_attention(...);
    */
    (void)dev; (void)q; (void)k; (void)v; (void)out; (void)params;
    set_error(err, -1, "tensix_sdpa: not yet implemented (REAL_MODE skeleton)");
}

void tensix_rms_norm(
    TensixDevice* dev, TensixBuffer* x, TensixBuffer* weight,
    TensixBuffer* out, const TensixNormParams* params, TensixError* err)
{
    /*
    TODO(REAL_MODE):
    // ttnn::rms_norm(x_tensor, weight_tensor, params->eps);
    */
    (void)dev; (void)x; (void)weight; (void)out; (void)params;
    set_error(err, -1, "tensix_rms_norm: not yet implemented (REAL_MODE skeleton)");
}

void tensix_rope(
    TensixDevice* dev, TensixBuffer* x, TensixBuffer* cos, TensixBuffer* sin,
    TensixBuffer* out, const TensixRopeParams* params, TensixError* err)
{
    /*
    TODO(REAL_MODE):
    // ttnn::experimental::rotary_embedding(...);
    */
    (void)dev; (void)x; (void)cos; (void)sin; (void)out; (void)params;
    set_error(err, -1, "tensix_rope: not yet implemented (REAL_MODE skeleton)");
}

void tensix_silu(
    TensixDevice* dev, TensixBuffer* x, TensixBuffer* out, TensixError* err)
{
    /*
    TODO(REAL_MODE):
    // ttnn::silu(x_tensor);
    */
    (void)dev; (void)x; (void)out;
    set_error(err, -1, "tensix_silu: not yet implemented (REAL_MODE skeleton)");
}

void tensix_add(
    TensixDevice* dev, TensixBuffer* a, TensixBuffer* b,
    TensixBuffer* out, TensixError* err)
{
    /*
    TODO(REAL_MODE):
    // ttnn::add(a_tensor, b_tensor);
    */
    (void)dev; (void)a; (void)b; (void)out;
    set_error(err, -1, "tensix_add: not yet implemented (REAL_MODE skeleton)");
}

// ── Program compilation ────────────────────────────────────────────────────

TensixProgram* tensix_compile_program(
    TensixDevice* dev, const uint8_t* program_spec_json,
    size_t json_len, TensixError* err)
{
    /*
    TODO(REAL_MODE): Parse JSON spec and build a tt::tt_metal::Program
    with kernel descriptors, CB configs, and runtime args.

    // auto json = nlohmann::json::parse(program_spec_json, program_spec_json + json_len);
    // TensixProgram* prog = (TensixProgram*)std::malloc(sizeof(TensixProgram));
    // prog->ptr = new tt::tt_metal::Program;
    // ... compile kernels, set runtime args ...
    // return prog;
    */
    (void)dev; (void)program_spec_json; (void)json_len;
    set_error(err, -1, "tensix_compile_program: not yet implemented (REAL_MODE skeleton)");
    return NULL;
}

void tensix_execute_program(
    TensixDevice* dev, TensixProgram* prog, TensixError* err)
{
    /*
    TODO(REAL_MODE):
    // tt::tt_metal::EnqueueProgram(command_queue,
    //     *static_cast<tt::tt_metal::Program*>(prog->ptr), false);
    */
    (void)dev; (void)prog;
    set_error(err, -1, "tensix_execute_program: not yet implemented (REAL_MODE skeleton)");
}

void tensix_free_program(TensixProgram* prog) {
    /*
    TODO(REAL_MODE):
    if (prog && prog->ptr) {
        delete static_cast<tt::tt_metal::Program*>(prog->ptr);
    }
    */
    std::free(prog);
}

// ── Synchronization ────────────────────────────────────────────────────────

void tensix_synchronize_device(TensixDevice* dev) {
    /*
    TODO(REAL_MODE):
    if (dev && dev->ptr) {
        auto* d = static_cast<tt::tt_metal::Device*>(dev->ptr);
        tt::tt_metal::Synchronize(*d);
    }
    */
    (void)dev;
}

// ── Profiling ──────────────────────────────────────────────────────────────

void tensix_read_profiler(
    TensixDevice* dev, TensixProfileEvent* event, TensixError* err)
{
    /*
    TODO(REAL_MODE): Query device-level profiling counters.
    Metalium exposes DPRINT profiling; TTNN has per-op profiling hooks.
    */
    (void)dev; (void)event;
    set_error(err, -1, "tensix_read_profiler: not yet implemented (REAL_MODE skeleton)");
}

void tensix_reset_profiler(TensixDevice* dev) {
    /*
    TODO(REAL_MODE): Reset device-level profiling counters.
    */
    (void)dev;
}

// ═══════════════════════════════════════════════════════════════════════════
//  UNKNOWN MODE  —  should never happen
// ═══════════════════════════════════════════════════════════════════════════
#else
#error "Define either TENSIX_STUB_MODE or TENSIX_REAL_MODE"
#endif
