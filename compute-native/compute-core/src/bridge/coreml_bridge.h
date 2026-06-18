// Tribunus Core ML Bridge — consolidated public C API header.
//
// Thread-safety contract:
//   All functions are thread-safe unless noted otherwise.
//   MLModel objects are thread-safe for prediction per Apple documentation.
//   MLState objects are NOT safe for concurrent prediction — callers must
//   serialize access to a single MLState.  Different MLState objects may be
//   used concurrently.
//   Async completion handlers run on arbitrary dispatch queues — the handler
//   MUST NOT assume it is on any specific queue.
//
// Null-safety contract:
//   All pointer parameters are checked for null before dereference.
//   Functions return -1 or equivalent error code on null input.
//   Passing NULL to destroy/release functions is a no-op.

#pragma once
#include <stdint.h>
#include "coreml_arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Opaque handles ─────────────────────────────────────────────────────────

/// Opaque handle for a Core ML model (MLModel*).
/// Created via tribunus_coreml_load_model, released via tribunus_coreml_free_model.
typedef struct TribunusCoreMlModel TribunusCoreMlModel;

/// Opaque handle for a Core ML state object (MLState*).
/// Created via tribunus_coreml_state_create, released via tribunus_coreml_state_destroy.
typedef struct TribunusCoreMlState TribunusCoreMlState;

/// Opaque handle for an in-flight stateful prediction request.
/// Created via tribunus_coreml_predict_stateful_async, released via
/// tribunus_coreml_stateful_request_destroy (or the refcount-autoreleased callback).
/// Safe to call destroy from any thread once; the request internally uses a
/// refcount to prevent use-after-free between the caller and the completion handler.
typedef struct TribunusCoreMlStatefulRequest TribunusCoreMlStatefulRequest;

// ── Model lifecycle (stateless) ─────────────────────────────────────────────

/// Load a compiled Core ML model (.mlmodelc directory).
/// Thread-safe: may be called from any thread.
/// @param out_model  [out] receives the opaque model handle (NULL-terminated on error).
/// @param model_path path to the .mlmodelc bundle directory.  Must be non-NULL.
/// @param compute_units MLComputeUnits as int64_t (0=CPU, 1=CPU+GPU, 2=CPU+ANE, 3=All).
/// @return 0 on success, negative on error (error details printed to stderr).
int tribunus_coreml_load_model(
    void** out_model,
    const char* model_path,
    int64_t compute_units
);

/// Release a loaded model.  Safe to call with NULL.
void tribunus_coreml_free_model(void* model_ptr);

// ── State lifecycle ─────────────────────────────────────────────────────────

/// Create a new MLState from a loaded model.
/// Thread-safe: MLModel's newState may be called from any thread.
/// @param out_state [out] receives the opaque state handle (NULL on error).
/// @param model_ptr opaque model handle from tribunus_coreml_load_model.
/// @return 0 on success, negative on error.
int tribunus_coreml_state_create(
    TribunusCoreMlState** out_state,
    void* model_ptr
);

/// Destroy a state object.  Safe to call with NULL.
void tribunus_coreml_state_destroy(TribunusCoreMlState* state);

// ── Stateless prediction ───────────────────────────────────────────────────

/// Run stateless prediction: input arena -> model -> output arena.
/// Thread-safe for concurrent calls on the same model (MLModel is reentrant).
/// @return 0 on success, negative on error.
int tribunus_coreml_predict(
    void* model_ptr,
    const char* input_name,
    const TribunusArenaInfo* input_arena_info,
    const char* output_name,
    const TribunusArenaInfo* output_arena_info
);

/// Run stateless prediction using IOSurface/CVPixelBuffer input.
/// Thread-safe for concurrent calls on the same model.
/// @return 0 on success, negative on error.
int tribunus_coreml_predict_pixelbuffer(
    void* model_ptr,
    const char* input_name,
    const TribunusArenaInfo* input_arena,
    const char* output_name,
    TribunusArenaInfo* output_arena
);

// ── Stateful prediction (synchronous) ───────────────────────────────────────

/// Run stateful prediction: input arena -> model + state -> output arena.
/// The MLState is read and updated atomically by CoreML.
/// NOT safe for concurrent calls on the same MLState — serialize access.
/// Thread-safe for concurrent calls on different MLState objects.
/// @return 0 on success, negative on error.
int tribunus_coreml_predict_stateful(
    void* model_ptr,
    TribunusCoreMlState* state,
    const char* input_name,
    void* input_arena_info,    // const TribunusArenaInfo*
    const char* output_name,
    void* output_arena_info    // TribunusArenaInfo* (output is written here)
);

// ── Stateful prediction (asynchronous) ──────────────────────────────────────

/// Start an async stateful prediction.  Returns immediately.
/// The completion handler runs on an arbitrary dispatch queue.
/// NOT safe for concurrent calls on the same MLState — serialize access.
///
/// @param out_request [out] receives an opaque request handle.
///        The caller owns one reference; call tribunus_coreml_stateful_request_destroy
///        to release it (or let the Rust Drop impl handle it).
/// @return 0 on success, negative on error (request not created).
int tribunus_coreml_predict_stateful_async(
    TribunusCoreMlStatefulRequest** out_request,
    void* model_ptr,
    TribunusCoreMlState* state,
    const char* input_name,
    void* input_arena_info,
    const char* output_name,
    void* output_arena_info
);

/// Check if an async request has completed.
/// Thread-safe (atomic read).
/// @return 1 if complete, 0 if still pending, -1 on NULL request.
int tribunus_coreml_stateful_request_is_complete(TribunusCoreMlStatefulRequest* request);

/// Set the Rust waker to wake when the async request completes.
/// Thread-safe.  May be called from any thread, including from the completion
/// handler itself.  If the request has already completed, the waker is called
/// immediately.
void tribunus_coreml_stateful_request_set_waker(
    TribunusCoreMlStatefulRequest* request,
    void* waker
);

/// Block until the async request completes.
/// NOT safe to call from the completion handler's dispatch queue (deadlock risk).
/// @return 0 on success, negative on prediction failure, -1 on NULL request.
int tribunus_coreml_stateful_request_wait(TribunusCoreMlStatefulRequest* request);

/// Release one reference on an async request handle.
/// Internally uses a refcount — safe to call once from the creator thread even
/// if the completion handler is still in-flight.  The underlying memory is freed
/// only when both the creator and the completion handler release their references.
/// Thread-safe (atomic refcount).
void tribunus_coreml_stateful_request_destroy(TribunusCoreMlStatefulRequest* request);

// ── Shutdown ────────────────────────────────────────────────────────────────

/// Graceful shutdown: signal all in-flight async requests, prevent new ones,
/// and release global resources.
///
/// Call once at process shutdown.  After this function returns:
///   - No new async requests will be accepted (returns error).
///   - All pending requests have been signalled/completed.
///   - Callers waiting on tribunus_coreml_stateful_request_wait will unblock.
///
/// Thread-safe.  Idempotent — subsequent calls are no-ops.
void tribunus_coreml_shutdown(void);

#ifdef __cplusplus
}
#endif
