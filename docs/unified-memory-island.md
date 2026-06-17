# IOSurface Unified Memory Island Architecture

Apple Silicon Macs use a Unified Memory Architecture (UMA) where the CPU, GPU, and
Neural Engine share the same physical memory pool. However, Tribunus Compute currently
has THREE completely independent memory subsystems that are unaware of each other.

## Current State — Three Silos

```
mlx-rs Arrays              candle MetalStorage        IOSurface Arena
┌──────────────────┐      ┌───────────────────┐     ┌──────────────────┐
│ MLX allocator    │      │ Bucket allocator   │     │ IOSurfaceCreate  │
│ Metal buffers    │      │ StorageModeShared  │     │ CVPixelBuffer    │
│ Arc<Buffer>      │      │ BufferMap buckets  │     │ new_external_arr │
│ opaque to others │      │ opaque to others   │     │ opaque to others │
└──────────────────┘      └───────────────────┘     └──────────────────┘
        │                         │                         │
        │                         │                         │
        ▼                         ▼                         ▼
   Apple Silicon Unified Memory (shared physical pool)
```

Each subsystem:
- Has its own allocator (no shared memory pool)
- Has its own tracking (no shared pressure awareness)
- Requires data copies to cross boundaries
- Is a separate Metal `Buffer` allocation from the same `MTLDevice`

## Target Architecture

```
                    ┌──────────────────────────────────────┐
                    │        Unified Memory Island          │
                    │                                      │
                    │  ┌────────────────────────────────┐  │
                    │  │    IOSurfaceArenaAllocator      │  │
                    │  │  (single MTLDevice + IOSurface  │  │
                    │  │    pool, tracked by ArenaId)    │  │
                    │  └──────┬──────┬──────┬───────────┘  │
                    │         │      │      │              │
                    │    ┌────┘      │      └──────┐       │
                    │    ▼           ▼              ▼       │
                    │ ┌──────┐ ┌──────────┐ ┌──────────┐   │
                    │ │ MLX  │ │ Candle   │ │ Core ML  │   │
                    │ │ Array│ │  Tensor  │ │ MLArray  │   │
                    │ └──────┘ └──────────┘ └──────────┘   │
                    │                                      │
                    │  No copies between subsystems —      │
                    │  all point to same IOSurface memory  │
                    └──────────────────────────────────────┘
```

## Implementation Plan

### 1. Unified Allocator (`memory/allocator.rs`)

A single IOSurface-backed allocator that all subsystems draw from.

```rust
pub struct IosurfaceAllocator {
    device: MTLDevice,
    active_arenas: HashMap<ArenaId, Arena>,
    total_allocated_bytes: AtomicU64,
    max_pool_bytes: u64,
}

impl IosurfaceAllocator {
    /// Allocate an IOSurface-backed arena of `size` bytes.
    pub fn allocate(&self, size: usize, dtype: Dtype) -> Result<Arena>;

    /// Reinterpret an existing Arena as an mlx-rs Array (zero-copy).
    pub fn as_mlx_array(&self, arena: &Arena, shape: &[i32]) -> Result<Array>;

    /// Reinterpret an existing Arena as a candle Tensor (zero-copy).
    pub fn as_candle_tensor<S: Shape>(
        &self, arena: &Arena, shape: S, dtype: DType
    ) -> Result<Tensor>;

    /// Total bytes allocated across all arenas.
    pub fn total_allocated(&self) -> u64;

    /// Current memory pressure (fraction of max_pool_bytes used).
    pub fn pressure(&self) -> f64;
}
```

### 2. ExternalStorage for IOSurface (`memory/iosurface_storage.rs`)

Implements the existing `ExternalStorage` trait for IOSurface-backed memory,
making it directly usable by `new_external_array()`.

```rust
pub struct IosurfaceStorage {
    arena: Arc<Arena>,
}

impl ExternalStorage for IosurfaceStorage {
    fn data_ptr(&self) -> *const u8 { self.arena.base_ptr() as *const u8 }
    fn byte_len(&self) -> usize { self.arena.byte_len() }
}
```

### 3. Candle MetalBuffer Bridge (`memory/candle_bridge.rs`)

A bridge that exposes candle's Metal-backed `Storage` as `ExternalStorage`,
allowing zero-copy conversion between candle Tensors and mlx-rs Arrays.

```rust
pub struct CandleMetalBridge {
    /// Wraps a candle Tensor's Metal buffer pointer as ExternalStorage.
    pub fn as_external_storage(tensor: &Tensor) -> Result<IosurfaceStorage>;

    /// Create a candle::Tensor that shares memory with an mlx-rs Array.
    pub fn from_mlx_array(arr: &Array) -> Result<Tensor>;
}
```

### 4. Memory Pressure Integration (`memory/pressure.rs`)

Extends the existing `worker_memory.rs` monitoring with per-allocator tracking:

```rust
pub struct UnifiedMemoryTelemetry {
    pub machine: MachineProfile,
    pub allocator: AllocatorStats,      // IOSurface pool usage
    pub mlx_allocator: MlxMemorySnapshot,  // MLX allocator stats
    pub process_rss: u64,               // from worker_memory
    pub swap_usage: u64,                // from system
    pub candle_allocator: CandleAllocatorStats,
}

pub fn sample_unified_memory() -> UnifiedMemoryTelemetry;
```

### 5. Zero-Copy Operation Table

| From \ To | mlx-rs Array | candle Tensor | Core ML MLArray |
|-----------|-------------|---------------|-----------------|
| mlx-rs Array | — | via `CandleMetalBridge` | via `ExternalStorage` |
| candle Tensor | via `CandleMetalBridge` | — | via `ExternalStorage` |
| Core ML MLArray | via `new_external_array()` | via `CandleMetalBridge` | — |

All operations are pointer-only — no data copies.

## Migration Steps

1. Create `memory/allocator.rs` with the unified IOSurface allocator
2. Create `memory/iosurface_storage.rs` implementing ExternalStorage
3. Create `memory/candle_bridge.rs` for mlx-rs ↔ candle
4. Update `arena.rs` to use the unified allocator
5. Add `UnifiedMemoryTelemetry` to worker_memory
6. Wire the omlx-style MemoryMonitor to track unified stats
7. Update docs to reflect the unified architecture
