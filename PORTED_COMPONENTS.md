# Ported Components from External Repositories

This document summarizes all components ported from the following repositories into Tribunus Compute:

- **OminiX-MLX** — High-performance ML inference on Apple Silicon (model implementations)
- **candle** — Minimalist ML framework for Rust (full port)
- **Orion** — ANE runtime for Apple Silicon (complete)
- **mistral.rs** — Fast LLM inference in Rust (full port)
- **Tribunus** — Main monorepo (compute-core, mlx-rs sync)

## Directory Structure

```
Tribunus Compute/
├── Cargo.toml                          # Root workspace (Tribunus native crates)
├── compute-native/                     # N-API binding layer (existing)
│   ├── src/                            # N-API bridge + glue
│   └── compute-core/                   # Core compute logic (from main Tribunus)
│       ├── src/ (110+ files)
│       ├── tests/ (15+ files)
│       └── Cargo.toml
├── mlx-rs-fork/                        # MLX Rust bindings (synced with main Tribunus)
│   ├── mlx-rs/, mlx-sys/, mlx-macros/, ...  # evidence feature added
├── mlx-rs-core/                        # Shared MLX infra (from OminiX-MLX, already ported)
├── mlx-rs-lm/                          # MLX LM utilities (from OminiX-MLX, NEW)
├── orion-runtime/                      # ANE runtime (from Orion, already ported)
├── models/                             # Model implementations (from OminiX-MLX, 19 crates)
│   ├── gemma4-mlx/                     # Gemma 4 LLM
│   ├── mixtral-mlx/                    # Mixtral MoE LLM
│   ├── mistral-mlx/                    # Mistral LLM (already ported)
│   ├── glm4-mlx/                       # GLM-4 LLM
│   ├── glm4-moe-mlx/                   # GLM-4 MoE LLM
│   ├── glm-4.7-flash-mlx/             # GLM-4.7 Flash LLM
│   ├── qwen3-mlx/                      # Qwen3 LLM (already ported)
│   ├── qwen3-vl-mlx/                   # Qwen3 Vision-Language
│   ├── qwen-image-mlx/                 # Qwen Image generation
│   ├── qwen3-asr-mlx/                  # Qwen3 ASR
│   ├── qwen3-tts-mlx/                  # Qwen3 TTS
│   ├── qwen3-tts-core/                 # Qwen3 TTS core (pure Rust)
│   ├── qwen3.5-35B-mlx/               # Qwen3.5 35B LLM
│   ├── funasr-mlx/                     # FunASR (speech recognition)
│   ├── funasr-nano-mlx/               # FunASR Nano
│   ├── funasr-qwen4b-mlx/             # FunASR Qwen4B
│   ├── gpt-sovits-mlx/                 # GPT-SoVITS voice synthesis
│   ├── deepseek-ocr2-mlx/             # DeepSeek OCR
│   ├── zimage-mlx/                     # Z-Image-Turbo generation
│   ├── flux-klein-mlx/                 # FLUX.2-klein image generation
│   ├── step-audio2-mlx/               # Step-Audio2 speech model
│   ├── moxin-vlm-mlx/                  # MoXin Vision-Language
│   ├── minicpm-sala-mlx/              # MiniCPM-SALA hybrid attention
│   └── step-audio2-mlx/                # Step-Audio-2
├── candle/                             # Candle ML framework (sub-workspace, 15 crates)
│   ├── Cargo.toml                      # Candle workspace
│   ├── candle-core/                    # Core tensor library
│   ├── candle-nn/                      # Neural network layers
│   ├── candle-transformers/            # Transformer implementations
│   ├── candle-kernels/                 # Custom CUDA kernels
│   ├── candle-metal-kernels/          # Metal GPU kernels
│   ├── candle-flash-attn/             # Flash attention
│   ├── candle-flash-attn-v3/          # Flash attention v3
│   ├── candle-onnx/                    # ONNX runtime support
│   ├── candle-datasets/               # Dataset utilities
│   ├── candle-examples/               # Reference model implementations
│   ├── candle-pyo3/                   # Python bindings
│   ├── candle-book/                   # Tutorial book
│   ├── candle-ug/                     # Micrograd
│   ├── candle-wasm-tests/             # WASM tests
│   ├── candle-wasm-examples/          # WASM examples
│   └── tensor-tools/                   # Tensor debugging tools
└── mistralrs/                          # Mistral.rs inference engine (sub-workspace, 14 crates)
    ├── Cargo.toml                      # Mistral.rs workspace
    ├── mistralrs-macros/              # Proc macros
    ├── mistralrs-core/                 # Core inference engine (358 files)
    ├── mistralrs-quant/               # Quantization (GGUF, GPTQ, AWQ, HQQ, ISQ)
    ├── mistralrs-paged-attn/          # PagedAttention (Metal + CUDA)
    ├── mistralrs-flash-attn/          # Flash attention
    ├── mistralrs-vision/              # Vision model support
    ├── mistralrs-code-exec/           # Code execution sandbox
    ├── mistralrs-sandbox/             # Safety sandboxing
    ├── mistralrs-audio/               # Audio model support
    ├── mistralrs-server-core/         # OpenAI-compatible API server
    ├── mistralrs-mcp/                 # MCP server
    ├── mistralrs-cli/                  # CLI interface
    ├── mistralrs-pyo3/                # Python bindings
    └── mistralrs (main)               # Main orchestrator
```

## Porting Summary

| Repo | Ported | Files | Status |
|------|--------|-------|--------|
| OminiX-MLX | 19 model crates + mlx-rs-lm | ~400 files | Complete |
| candle | 15 crates (full port) | ~1,000 files | Complete |
| Orion | 1 runtime (already ported) | ~200 files | Complete |
| mistral.rs | 14 crates (full port) | ~600 files | Complete |
| Tribunus main | compute-core + mlx-rs sync | Already present | Complete |

## Dependency Architecture

### Root workspace (Tribunus Compute/Cargo.toml)
Contains: compute-native, mlx-rs-core, mlx-rs-lm, models/* (19 crates)
Dependencies: mlx-rs-fork (path), external crates.io

### Candle sub-workspace (Tribunus Compute/candle/Cargo.toml)
Contains: 15 candle crates
Dependencies: Internal path deps

### Mistral.rs sub-workspace (Tribunus Compute/mistralrs/Cargo.toml)
Contains: 14 mistralrs crates
Dependencies: candle sub-workspace (path), external crates.io

## Key Adaptations

### OminiX-MLX models dependency rewrites
- `{ path = "../mlx-rs" }` → `{ path = "../../mlx-rs-fork/mlx-rs" }`
- `{ path = "../mlx-rs/mlx-sys" }` → `{ path = "../../mlx-rs-fork/mlx-sys" }`
- `{ path = "../mlx-rs-core" }` → `{ path = "../../mlx-rs-core" }`
- `{ workspace = true }` expanded to explicit version numbers

### Candle crates
- Workspace deps preserved via candle/Cargo.toml
- Path deps between crates preserved

### Mistral.rs crates
- Candle git deps → local path deps to candle/ sub-workspace
- Internal path deps preserved via mistralrs workspace

### mlx-rs-fork sync
- Added `evidence` feature (serde + serde_json support)
- Matches main Tribunus monorepo

# omlx — Python LLM Inference Server (Reference + Rust Scaffolding)

See `ref/omlx/README.md`, `docs/omlx-*.md`, and `src/{quantization,cache,memory,scheduling}/` for details.

| Component | Source | Lines | Status |
|-----------|--------|-------|--------|
| Python reference files | omlx/ | ~25K, ~317 files | Copied to ref/omlx/ |
| oQ Dynamic Quantization | ref/omlx/oq.py | 4.2K | Design doc + Rust scaffold |
| Prefix Cache | ref/omlx/cache/prefix_cache.py | 3K | Design doc + Rust scaffold |
| Paged SSD Cache | ref/omlx/cache/paged_ssd_cache.py | 3.5K | Design doc + Rust scaffold |
| Memory Management | ref/omlx/memory_monitor.py + process_memory_enforcer.py | 2.3K | Design doc + Rust scaffold |
| TurboQuant KV | ref/omlx/turboquant_kv.py | 495 | Design doc + Rust scaffold |
| Scheduler | ref/omlx/scheduler.py | 10K | Design doc + Rust scaffold |
| VLM Engine | ref/omlx/engine/vlm.py | 3.5K | Reference only |
| Tool Calling | ref/omlx/api/tool_calling.py | 2K | Reference only |
| Harmony Adapter | ref/omlx/adapter/harmony.py | 495 | Reference only |

# Unified Memory Island — Apple Silicon Memory Architecture Alignment

Design doc: `docs/unified-memory-island.md`

Apple Silicon Macs use a Unified Memory Architecture (UMA) where CPU, GPU,
and Neural Engine share the same physical memory pool.  Tribunus Compute now
has a single IOSurface-backed memory island that all subsystems draw from.

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Allocator | `src/memory/allocator.rs` | ~300 | Single IOSurface-backed allocator for all subsystems |
| IOSurface Storage | `src/memory/iosurface_storage.rs` | ~180 | IOSurface memory as `ExternalStorage` for mlx-rs |
| Candle Bridge | `src/memory/candle_bridge.rs` | ~450 | Zero-copy bridge (candle ↔ mlx-rs via unified memory) |
| Telemetry | `src/memory/telemetry.rs` | ~180 | Unified memory pressure across all backends |
| Monitor | `src/memory/monitor.rs` | ~60 | Real-time memory pressure monitoring (omlx-style) |
| Enforcer | `src/memory/enforcer.rs` | ~80 | Proactive OOM prevention |
| Pool | `src/memory/pool.rs` | ~80 | Memory-aware engine lifecycle |

### Architecture

```
                  Unified Memory Island
                ┌──────────────────────────┐
                │    IosurfaceAllocator    │
                │  (single MTLDevice pool) │
                └────┬────┬────┬───────────┘
                     │    │    │
                     ▼    ▼    ▼
               ┌────┐ ┌────┐ ┌────┐
               │MLX │ │Cndl│ │CML │
               │Arr │ │Ten.│ │MLAr│
               └────┘ └────┘ └────┘
               Zero copy — same IOSurface bytes
```

### Key adaptations

- `IosurfaceStorage` implements `ExternalStorage` so IOSurface memory is directly usable
  by `new_external_array()` — no copy into mlx-rs Arrays
- `UnifiedMemoryBlock` provides a raw memory region viewable as either a candle Tensor
  or mlx-rs Array through Apple Silicon's shared-Metal-buffer memory
- `UnifiedMemoryTelemetry` aggregates allocator stats from mlx-rs, candle buckets, and
  the IOSurface pool into a single pressure metric
