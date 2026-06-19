#ifndef TENSIX_FFI_BRIDGE_H
#define TENSIX_FFI_BRIDGE_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types */
typedef struct TensixDevice TensixDevice;
typedef struct TensixBuffer TensixBuffer;
typedef struct TensixProgram TensixProgram;

/* Error struct — 256-byte message + i32 code */
typedef struct {
    char     message[256];
    int32_t  code;
} TensixError;

/* Memory domain */
typedef enum {
    TensixMemoryType_DRAM  = 0,
    TensixMemoryType_Host  = 1,
    TensixMemoryType_Trace = 2,
} TensixMemoryType;

/* Op parameter structs */
typedef struct {
    uint32_t M, N, K;
    uint8_t  transpose_a, transpose_b;
    uint8_t  dtype;  /* 0=f32, 1=f16, 2=bf16 */
} TensixMatmulParams;

typedef struct {
    uint32_t batch, num_q_heads, num_kv_heads, head_dim;
    uint32_t seq_len_q, seq_len_kv;
    float    scale;
    uint8_t  dtype;
} TensixSdpaParams;

typedef struct {
    uint32_t dim;
    float    eps;
    uint8_t  dtype;
} TensixNormParams;

typedef struct {
    uint32_t dim;
    float    theta;
    int32_t  max_seq_len;
} TensixRopeParams;

typedef struct {
    uint64_t kernel_cycles;
    uint64_t sync_ns;
    uint64_t dram_bytes_read;
    uint64_t dram_bytes_written;
    uint32_t core_count;
    float    cb_occupancy;
    float    noc_utilization;
} TensixProfileEvent;

/* ── Device ────────────────────────────────────────────────────────── */
TensixDevice* tensix_open_device(int32_t device_id, TensixError* err);
void          tensix_close_device(TensixDevice* dev);
int32_t       tensix_device_core_count(TensixDevice* dev);
void          tensix_device_arch(TensixDevice* dev, uint8_t* out, size_t out_len);

/* ── Buffer ────────────────────────────────────────────────────────── */
TensixBuffer* tensix_allocate_buffer(TensixDevice* dev, uint64_t bytes,
                                     TensixMemoryType mem_type, TensixError* err);
void          tensix_deallocate_buffer(TensixBuffer* buf);
void          tensix_write_to_buffer(TensixBuffer* buf, const float* data,
                                     uint64_t count, uint64_t offset, TensixError* err);
void          tensix_read_from_buffer(float* data, TensixBuffer* buf,
                                      uint64_t count, uint64_t offset, TensixError* err);

/* ── Compute ops ───────────────────────────────────────────────────── */
void tensix_matmul(TensixDevice* dev, TensixBuffer* a, TensixBuffer* b,
                   TensixBuffer* c, const TensixMatmulParams* params, TensixError* err);
void tensix_sdpa(TensixDevice* dev, TensixBuffer* q, TensixBuffer* k,
                 TensixBuffer* v, TensixBuffer* out,
                 const TensixSdpaParams* params, TensixError* err);
void tensix_rms_norm(TensixDevice* dev, TensixBuffer* x, TensixBuffer* weight,
                     TensixBuffer* out, const TensixNormParams* params, TensixError* err);
void tensix_rope(TensixDevice* dev, TensixBuffer* x, TensixBuffer* cos,
                 TensixBuffer* sin, TensixBuffer* out,
                 const TensixRopeParams* params, TensixError* err);
void tensix_silu(TensixDevice* dev, TensixBuffer* x,
                 TensixBuffer* out, TensixError* err);
void tensix_add(TensixDevice* dev, TensixBuffer* a, TensixBuffer* b,
                TensixBuffer* out, TensixError* err);

/* ── Program compilation ──────────────────────────────────────────── */
TensixProgram* tensix_compile_program(TensixDevice* dev,
                                      const uint8_t* program_spec_json,
                                      size_t json_len, TensixError* err);
void           tensix_execute_program(TensixDevice* dev,
                                      TensixProgram* prog, TensixError* err);
void           tensix_free_program(TensixProgram* prog);

/* ── Sync ──────────────────────────────────────────────────────────── */
void tensix_synchronize_device(TensixDevice* dev);

/* ── Profiling ─────────────────────────────────────────────────────── */
void tensix_read_profiler(TensixDevice* dev, TensixProfileEvent* event,
                          TensixError* err);
void tensix_reset_profiler(TensixDevice* dev);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* TENSIX_FFI_BRIDGE_H */
