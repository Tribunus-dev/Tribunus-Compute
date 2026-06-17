# ComputeImage Memory Architecture — Zero-Copy Pipeline

The canonical data flow for ALL compute operations in Tribunus Compute. Every
subsystem (mlx-rs Arrays, candle Tensors, Core ML MLArrays) must construct its
tensor types from ComputeImage's MappedSegment memory — never from its own
private allocator.

## The Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          1. COMPILE PHASE                                    │
│                                                                              │
│  Source model → compile_with_authority() → ComputeImage directory            │
│                                                                              │
│  ├── manifest.json     tensor table, alias table, residency plan            │
│  ├── segment_000.bin   execution-ordered tensor bytes (mmap target)         │
│  ├── segment_001.bin   quantized weights, scales, biases                   │
│  └── ...                                                                     │
│                                                                              │
│  All tensors are laid out in execution order with known offsets.             │
│  No runtime rearrangement needed.                                            │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          2. LOAD PHASE (mmap)                                │
│                                                                              │
│  MappedSegment::new(path) → Arc<MappedSegment>                               │
│    ↓                                                                          │
│  mmap(..., MAP_PRIVATE | MAP_NORESERVE, PROT_READ, fd, 0)                    │
│    ↓                                                                          │
│  Kernel maps file pages into process address space.                          │
│  On Apple Silicon, these are PHYSICALLY THE SAME PAGES the GPU reads.        │
│  → NO memory allocated. NO copy.                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                     3. TENSOR CONSTRUCTION (zero copy)                       │
│                                                                              │
│  TensorTable entry:                                                          │
│    name: "model.layers.0.self_attn.q_proj.weight"                            │
│    segment: "segment_000.bin"                                                │
│    offset: 4096    (aligned to 16 bytes)                                      │
│    byte_length: 16384                                                         │
│    logical_shape: [4096, 4096]                                                │
│    storage_dtype: "U8"                                                       │
│    quantization: { groups: 64, group_size: 64 }                               │
│                                                                              │
│  Pointer calculation:                                                        │
│    ptr = segment.data_ptr() + tensor_entry.offset                             │
│                                                                              │
│  mlx-rs Array (read-only, no copy):                                          │
│    let storage = StaticStorage::new(ptr, byte_length);                        │
│    let arr = unsafe { new_external_array(                                    │
│        Arc::new(storage),                                                    │
│        &logical_shape,                                                       │
│        mlx_dtype,                                                            │
│    )? };                                                                     │
│                                                                              │
│  candle Tensor (read-only, no copy):                                         │
│    let storage = CpuStorage::from_ptr(ptr, len, dtype);                      │
│    let tensor = Tensor::from_cpu_storage(storage, &shape, device);           │
│    // On Apple Silicon, CpuStorage reads unified memory pages                │
│    // → Metal ops can reference the same pages via shared buffers            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                     4. INFERENCE (in-place on mmap'd memory)                 │
│                                                                              │
│  prefill: token→KV  ────→  attention  ────→  logits  ────→  sample          │
│    KV cache writes    │     reads weights       writes logits                │
│    to Arena (IOSurf)  │     from MappedSegment  to Arena (IOSurf)            │
│                       │     (zero copy)                                      │
│                       ▼                                                      │
│  All weight tensor reads hit the mmap'd pages directly.                      │
│  KV cache goes to IOSurface-backed Arena (the only allocator).              │
│  Logits are written to a small IOSurface Arena buffer.                       │
│                                                                              │
│  NO memory is allocated for weights at runtime.                               │
└──────────────────────────────────────────────────────────────────────────────┘

## Subsystem Conformance

### mlx-rs Arrays ✓ (already compliant)
- `MappedSegment` implements `ExternalStorage` ✓
- `new_external_array()` creates no-copy Arrays ✓
- Used by `kv_cache.rs`, Gemma model code ✓
- OminiX models already use mlx-rs — need to wire via `new_external_array()`

### IOSurface Arena ✓ (already compliant)
- `Arena::new()` creates IOSurface-backed FP16 memory ✓
- `arena_to_mlx_array()` → `new_external_array()` → mlx-rs Array ✓
- Used for KV cache (grows dynamically, requires write) ✓
- The only ALLOCATOR in the system — everything else is zero-copy mmap

### candle Tensors ✗ (needs bridge)
- Must NOT use `MetalDevice::allocator` for weight tensors
- Must construct from `CpuStorage::from_ptr(segment_ptr + offset, ...)`
- The `UnifiedMemoryBlock` bridge provides the raw pointer contract
- The `bytes_to_mlx_array()` function shows the pattern — candle needs equivalent

### mistralrs ✗ (inherits from candle)
- Uses candle internally — if candle is compliant, mistralrs follows
- mistralrs-quant's quantization kernels need buffer access from shared memory

### KV Cache ✓ (already compliant)
- Uses `mlx_rs::Array` for keys/values via `kv_cache.rs`
- Uses IOSurface Arena for dynamic growth
- Ring buffer for sliding layers, concatenation for global layers

## Implementation: load_tensor_from_compute_image

```rust
/// Create a no-copy mlx-rs Array from a ComputeImage tensor entry.
///
/// # Zero-copy guarantee
/// The returned Array reads directly from the MappedSegment's mmap'd pages.
/// No memory is allocated beyond the small Array handle struct.
pub fn load_tensor_from_compute_image(
    segment: &Arc<MappedSegment>,
    entry: &TensorEntry,
) -> Result<Array, String> {
    let ptr = unsafe { segment.data_ptr().add(entry.offset as usize) };
    let storage = Arc::new(unsafe {
        StaticStorage::new(ptr, entry.byte_length as usize)
    });
    let shape: Vec<i32> = entry.logical_shape.iter().map(|&d| d as i32).collect();
    unsafe { new_external_array(storage, &shape, entry.storage_dtype.into()) }
}
```

## Implementation: load_candle_tensor_from_compute_image

```rust
/// Create a candle Tensor from a ComputeImage tensor entry.
///
/// On Apple Silicon, the returned Tensor shares pages with the mmap'd segment.
/// No memory copy occurs.
pub fn load_candle_tensor_from_compute_image(
    segment: &Arc<MappedSegment>,
    entry: &TensorEntry,
    device: &candle_core::Device,
) -> Result<candle_core::Tensor, String> {
    let ptr = unsafe { segment.data_ptr().add(entry.offset as usize) };
    let len = entry.byte_length as usize;
    let dtype = convert_storage_dtype(&entry.storage_dtype)?;
    let shape = convert_logical_shape(&entry.logical_shape)?;

    // Create CpuStorage wrapping the mmap'd memory
    // CpuStorage::from_ptr is a no-copy operation
    let storage = unsafe { candle_core::CpuStorage::from_ptr(ptr as *mut u8, len, dtype) }
        .map_err(|e| format!("CpuStorage::from_ptr: {e}"))?;

    candle_core::Tensor::from_cpu_storage(storage, &shape, device)
        .map_err(|e| format!("Tensor::from_cpu_storage: {e}"))
}
```

## Rules

1. **NO subsystem allocates weight-tensor memory at runtime.**
   Weight reads always come from mmap'd ComputeImage segments.

2. **The only runtime memory allocator is the IOSurface Arena.**
   KV cache, logit buffers, scratch buffers use `IosurfaceAllocator`.

3. **mlx-rs and candle must construct tensors from MappedSegment pointers.**
   The `StaticStorage` bridge enables this for mlx-rs; `CpuStorage::from_ptr`
   enables it for candle.

4. **CUDA/Metal private storage mode is NOT used for weight tensors.**
   `StorageModeShared` is required so CPU and GPU share the same pages.
   (This is the default on Apple Silicon.)
