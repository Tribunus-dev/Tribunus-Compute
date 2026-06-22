// Tribunus SharedTensorArena — IOSurface + CVPixelBuffer backed allocator.
// Phase 1: Replaces posix_memalign with real IOSurface + CVPixelBuffer storage.

#import "coreml_arena.h"
#import <IOSurface/IOSurface.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <string.h>
#import <stdint.h>

// kCVPixelFormatType_OneComponent16Half is 0x4C303068 = 'L00h'
// It may not be defined in older SDKs, so define it ourselves.
#ifndef kCVPixelFormatType_OneComponent16Half
#define kCVPixelFormatType_OneComponent16Half 0x4C303068
#endif

// kCVPixelFormatType_OneComponent32Float is 0x4C303066 = 'L00f'
// It may not be defined in older SDKs, so define it ourselves.
#ifndef kCVPixelFormatType_OneComponent32Float
#define kCVPixelFormatType_OneComponent32Float 0x4C303066
#endif

extern "C" {

int tribunus_arena_alloc(TribunusArenaInfo* info,
                          int32_t logical_dim0,
                          int32_t logical_dim1,
                          int32_t dtype) {
    @autoreleasepool {
        if (!info || logical_dim0 <= 0 || logical_dim1 <= 0) return -1;
        memset(info, 0, sizeof(TribunusArenaInfo));

        // Select pixel format based on dtype
        bool is_float16 = (dtype == 0);  // MLX Dtype::Float16 = 0
        int bytes_per_elem = is_float16 ? 2 : 4;
        uint32_t pixel_format = is_float16
            ? kCVPixelFormatType_OneComponent16Half
            : kCVPixelFormatType_OneComponent32Float;

        int32_t width = logical_dim1;
        int32_t height = logical_dim0;
        int32_t byte_size = width * height * bytes_per_elem;

        NSDictionary* surfaceAttrs = @{
            (id)kIOSurfaceWidth: @(width),
            (id)kIOSurfaceHeight: @(height),
            (id)kIOSurfaceBytesPerElement: @(bytes_per_elem),
            (id)kIOSurfacePixelFormat: @(pixel_format),
        };

        IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)surfaceAttrs);
        if (!surface) return -2;

        CVPixelBufferRef cvBuffer = NULL;
        CVReturn cvRet = CVPixelBufferCreateWithIOSurface(
            kCFAllocatorDefault, surface, NULL, &cvBuffer);
        if (cvRet != kCVReturnSuccess || !cvBuffer) {
            CFRelease(surface);
            return -3;
        }

        CVReturn lockRet = CVPixelBufferLockBaseAddress(cvBuffer, 0);
        if (lockRet != kCVReturnSuccess) {
            CVPixelBufferUnlockBaseAddress(cvBuffer, 0);
            CFRelease(cvBuffer);
            CFRelease(surface);
            return -4;
        }

        void* base = CVPixelBufferGetBaseAddress(cvBuffer);
        if (!base) {
            CVPixelBufferUnlockBaseAddress(cvBuffer, 0);
            CFRelease(cvBuffer);
            CFRelease(surface);
            return -5;
        }

        size_t bpr = CVPixelBufferGetBytesPerRow(cvBuffer);
        memset(base, 0, byte_size);

        info->width = width;
        info->height = height;
        info->logical_dim0 = logical_dim0;
        info->logical_dim1 = logical_dim1;
        info->pixel_format = pixel_format;
        info->byte_size = byte_size;
        info->bytes_per_row = (uint32_t)bpr;
        info->base_address = base;
        info->cv_buffer = (void*)CFRetain(cvBuffer);
        info->io_surface = (void*)CFRetain(surface);

        return 0;
    }
}
int tribunus_arena_alloc_f32(TribunusArenaInfo* info,
                              int32_t logical_dim0,
                              int32_t logical_dim1) {
    @autoreleasepool {
        if (!info || logical_dim0 <= 0 || logical_dim1 <= 0) return -1;
        memset(info, 0, sizeof(TribunusArenaInfo));

        int32_t width = logical_dim1;
        int32_t height = logical_dim0;
        int32_t byte_size = width * height * 4; // FP32 = 4 bytes per element

        // Create CVPixelBuffer directly with the proven kCVPixelFormatType_OneComponent32Float.
        // This internally creates an IOSurface.
        CVPixelBufferRef cvBuffer = NULL;
        uint32_t pixelFormat = kCVPixelFormatType_OneComponent32Float;
        CVReturn cvRet = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, pixelFormat, NULL, &cvBuffer);
        if (cvRet != kCVReturnSuccess || !cvBuffer) return -2;

        // Get the underlying IOSurface.
        IOSurfaceRef surface = CVPixelBufferGetIOSurface(cvBuffer);
        if (!surface) {
            CFRelease(cvBuffer);
            return -3;
        }
        CFRetain(surface);

        // Lock the buffer permanently.
        CVReturn lockRet = CVPixelBufferLockBaseAddress(cvBuffer, 0);
        if (lockRet != kCVReturnSuccess) {
            CFRelease(surface);
            CFRelease(cvBuffer);
            return -4;
        }

        void* base = CVPixelBufferGetBaseAddress(cvBuffer);
        if (!base) {
            CVPixelBufferUnlockBaseAddress(cvBuffer, 0);
            CFRelease(surface);
            CFRelease(cvBuffer);
            return -5;
        }

        size_t bpr = CVPixelBufferGetBytesPerRow(cvBuffer);
        memset(base, 0, byte_size);

        info->width = width;
        info->height = height;
        info->logical_dim0 = logical_dim0;
        info->logical_dim1 = logical_dim1;
        info->pixel_format = pixelFormat;
        info->byte_size = byte_size;
        info->bytes_per_row = (uint32_t)bpr;
        info->base_address = base;
        info->cv_buffer = (void*)CFRetain(cvBuffer);
        info->io_surface = (void*)surface;

        return 0;
    }
}

void tribunus_arena_free(TribunusArenaInfo* info) {
    @autoreleasepool {
        if (!info) return;

        if (info->cv_buffer) {
            CVPixelBufferRef cvBuffer = (CVPixelBufferRef)info->cv_buffer;
            CVPixelBufferUnlockBaseAddress(cvBuffer, 0);
            CFRelease(info->cv_buffer);
        }
        if (info->io_surface) {
            CFRelease(info->io_surface);
        }

        memset(info, 0, sizeof(TribunusArenaInfo));
    }
}

int32_t tribunus_arena_io_surface_id(const TribunusArenaInfo* info) {
    @autoreleasepool {
        if (!info || !info->io_surface) return -1;
        IOSurfaceRef surface = (IOSurfaceRef)info->io_surface;
        return (int32_t)IOSurfaceGetID(surface);
    }
}

int tribunus_arena_lock(TribunusArenaInfo* info) {
    @autoreleasepool {
        // Buffer is locked at allocation for the full arena lifetime (Phase 1).
        // Lock/unlock are API placeholders for future lease-based ownership.
        (void)info;
        return 0;
    }
}

int tribunus_arena_unlock(TribunusArenaInfo* info) {
    @autoreleasepool {
        // Buffer is locked for the full arena lifetime.
        (void)info;
        return 0;
    }
}

int tribunus_arena_alloc_bytes(TribunusArenaInfo* info, int32_t byte_count) {
    @autoreleasepool {
        if (!info || byte_count <= 0) return -1;
        memset(info, 0, sizeof(TribunusArenaInfo));
        CVPixelBufferRef cvBuffer = NULL;
        CVReturn cvRet = CVPixelBufferCreate(
            kCFAllocatorDefault, byte_count, 1,
            kCVPixelFormatType_OneComponent8, NULL, &cvBuffer);
        if (cvRet != kCVReturnSuccess || !cvBuffer) return -2;
        IOSurfaceRef surface = CVPixelBufferGetIOSurface(cvBuffer);
        if (!surface) { CFRelease(cvBuffer); return -3; }
        CFRetain(surface);
        CVReturn lockRet = CVPixelBufferLockBaseAddress(cvBuffer, 0);
        if (lockRet != kCVReturnSuccess) { CFRelease(surface); CFRelease(cvBuffer); return -4; }
        void* base = CVPixelBufferGetBaseAddress(cvBuffer);
        if (!base) { CVPixelBufferUnlockBaseAddress(cvBuffer, 0); CFRelease(surface); CFRelease(cvBuffer); return -5; }
        size_t bpr = CVPixelBufferGetBytesPerRow(cvBuffer);
        memset(base, 0, byte_count);
        info->width = byte_count; info->height = 1;
        info->logical_dim0 = byte_count; info->logical_dim1 = 1;
        info->pixel_format = kCVPixelFormatType_OneComponent8;
        info->byte_size = byte_count; info->bytes_per_row = (uint32_t)bpr;
        info->base_address = base;
        info->cv_buffer = (void*)CFRetain(cvBuffer);
        info->io_surface = (void*)CFRetain(surface);
        return 0;
    }
}

} // extern "C"
