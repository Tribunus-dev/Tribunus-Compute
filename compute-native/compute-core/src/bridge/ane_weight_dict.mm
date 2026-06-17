// ObjC bridge: build weight_dict NSDictionary for orion_compile_mil.
//
// The weight_dict maps BLOBFILE paths (referenced in MIL text) to
// NSData objects wrapping mmap'd segment memory (no copy).  The
// ANE compiler reads weights directly from the same MappedSegment
// pages that MLX and Accelerate use — zero-copy across all backends.

#import <Foundation/Foundation.h>

typedef struct {
    const char* blob_path;
    const void* data;
    unsigned long length;
    unsigned long offset;
} OrionWeightBlobEntry;

/// Build an NSDictionary weight_dict from blob entries using
/// `dataWithBytesNoCopy:length:freeWhenDone:NO` (no data copy).
/// Returns a CFRetained void* (NSDictionary*) — caller must CFRelease.
extern "C" void* build_ane_weight_dict(
    const OrionWeightBlobEntry* blobs,
    unsigned long count
) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];

    for (unsigned long i = 0; i < count; i++) {
        NSString *key = [NSString stringWithUTF8String:blobs[i].blob_path];
        NSData *data = [NSData dataWithBytesNoCopy:(void *)blobs[i].data
                                            length:blobs[i].length
                                      freeWhenDone:NO];
        NSDictionary *entry = @{@"data": data, @"offset": @(blobs[i].offset)};
        dict[key] = entry;
    }

    return (void*)CFBridgingRetain(dict);
}
