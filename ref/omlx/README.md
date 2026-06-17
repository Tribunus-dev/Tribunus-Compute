# omlx Reference Implementation

This directory contains Python source files from [omlx](https://github.com/jundot/omlx),
an LLM inference server optimized for Apple Silicon.

## Purpose

These files serve as reference implementations for porting omlx's novel algorithms
into Tribunus Compute's Rust-native inference engine. Each module corresponds to
a design doc in `docs/` and a Rust scaffold in `compute-core/src/`.

## Key Modules

### Cache (ref/omlx/cache/)
- `prefix_cache.py` — Block-aware prefix caching with automatic prefix discovery (3K lines)
- `paged_cache.py` — Block-based paged attention cache (1.7K lines)
- `paged_ssd_cache.py` — SSD-backed KV cache with safetensors serialization (3.5K lines)
- `boundary_snapshot_store.py` — Cache boundary management for long contexts (1K lines)
- `hybrid_cache.py` — Hybrid in-memory/SSD cache strategy (338 lines)
- `vision_feature_cache.py` — VLM vision feature caching (502 lines)

### Engine (ref/omlx/engine/)
- `vlm.py` — Vision-Language Model engine (3.5K lines)
- `batched.py` — Continuous batching engine (964 lines)
- `dflash.py` — Draft-model speculative decoding engine (1.5K lines)
- `base.py` — Base engine class (490 lines)

### Core (ref/omlx/)
- `scheduler.py` — Continuous batching scheduler (10K lines)
- `oq.py` — Dynamic load-time quantization (oQ2-oQ8) (4.2K lines)
- `turboquant_kv.py` — KV cache quantization wrapper (495 lines)
- `memory_monitor.py` — Real-time memory monitoring (844 lines)
- `process_memory_enforcer.py` — Proactive OOM prevention (1.4K lines)
- `server.py` — OpenAI-compatible API server (6.3K lines)
- `model_discovery.py` — Automatic model detection (1.2K lines)
- `engine_pool.py` — Multi-engine lifecycle management (1.4K lines)
- `engine_core.py` — Core engine orchestration (1.2K lines)

### API (ref/omlx/api/)
- `tool_calling.py` — Structured output / tool calling (2K lines)
- `thinking.py` — Chain of thought API (543 lines)

### Adapter (ref/omlx/adapter/)
- `harmony.py` — Harmony format model loading (495 lines)
- `gemma4.py` — Gemma 4 adapter (499 lines)

## Porting Status

| Algorithm | Design Doc | Rust Scaffold | Status |
|-----------|-----------|---------------|--------|
| oQ Quantization | `docs/omlx-oq-quantization.md` | `src/quantization/oq.rs` | Scaffold |
| Prefix Cache | `docs/omlx-prefix-cache.md` | `src/cache/prefix_cache.rs` | Scaffold |
| Paged SSD Cache | `docs/omlx-ssd-cache.md` | `src/cache/paged_ssd_cache.rs` | Scaffold |
| Memory Management | `docs/omlx-memory-management.md` | `src/memory/*.rs` | Scaffold |
| TurboQuant KV | `docs/omlx-turboquant-kv.md` | `src/quantization/turboquant_kv.rs` | Scaffold |
| Scheduler | `docs/omlx-scheduler.md` | `src/scheduling/*.rs` | Scaffold |
| VLM Engine | — | — | Reference Only |
| Tool Calling | — | — | Reference Only |
| Harmony Adapter | — | — | Reference Only |
