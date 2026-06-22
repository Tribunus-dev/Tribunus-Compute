// C header for ane_weight_dict.mm — built as a static library.
//
// Used by Rust FFI in memory/ane_weight_dict.rs to build the weight_dict
// NSDictionary for orion_compile_mil() from MappedSegment data.

#ifndef ORION_WEIGHT_DICT_BRIDGE_H
#define ORION_WEIGHT_DICT_BRIDGE_H

#include <stddef.h>

/// A single weight blob entry passed from Rust.
typedef struct {
    const char* blob_path;   // BLOBFILE path in MIL text
    const void* data;        // Pointer into MappedSegment memory
    unsigned long length;    // Byte length
    unsigned long long offset;  // Byte offset within the weight file
} OrionWeightBlobEntry;

/// Build an NSDictionary weight_dict from blob entries.
/// The returned pointer is a retained NSDictionary* — caller must CFRelease.
void* build_ane_weight_dict(const OrionWeightBlobEntry* blobs, unsigned long count);

#endif // ORION_WEIGHT_DICT_BRIDGE_H
