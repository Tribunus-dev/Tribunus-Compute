#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for an ANE compiled program
typedef struct TribunusAneProgram TribunusAneProgram;

// Initialize AppleNeuralEngine private framework dynamic loading.
// Returns 1 if successful, 0 if not available/failed.
int tribunus_ane_init(void);

// Compile MIL program text to ANE program.
// Returns 0 on success, negative on error.
int tribunus_ane_compile_mil(
    TribunusAneProgram** out_program,
    const char* mil_text,
    const char* program_tag
);

// Compile MIL program with weight dictionary to ANE program.
// weight_dict is a CFRetained NSDictionary* (as void*) from build_ane_weight_dict().
// Caller retains ownership of weight_dict.
// Returns 0 on success, negative on error.
int tribunus_ane_compile_mil_with_weights(
    TribunusAneProgram** out_program,
    const char* mil_text,
    void* weight_dict,  // NSDictionary* - CFRetained
    const char* program_tag
);

// Evaluate a compiled program with input and output IOSurfaces.
// Returns 1 on success, 0 on failure/error.
int tribunus_ane_eval(
    TribunusAneProgram* program,
    void** inputs, int num_inputs,
    void** outputs, int num_outputs
);

// Unload and release ANE program.
void tribunus_ane_release_program(TribunusAneProgram* program);

// Get current compile count.
int tribunus_ane_compile_count(void);

// Reload weight file on disk and reload the model.
// Returns 1 on success, 0 on failure.
int tribunus_ane_program_reload_weights(
    TribunusAneProgram* program,
    const char* weight_path,
    const void* weight_data,
    uint64_t weight_size
);

#ifdef __cplusplus
}
#endif
