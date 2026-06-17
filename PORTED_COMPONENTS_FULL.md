# Complete Porting Summary for Tribunus Compute

## Overview

This document provides a complete summary of all components ported from external repositories (OminiX-MLX, candle, Orion) and the main Tribunus monorepo to the Tribunus Compute repository.

## Architecture Comparison

### Main Tribunus Monorepo (Source of Truth)
```
Tribunus/
├── mlx-rs/                    # Tribunus fork of mlx-rs
├── mlx-sys/                   # Tribunus fork of mlx-sys  
├── mlx-internal-macros/
├── mlx-macros/
├── mlx-lm/
├── mlx-lm-utils/
├── mlx-tests/
└── packages/
    ├── compute-native/        # Thin NAPI bindings (386 lines lib.rs)
    │   └── src/lib.rs        # Delegates to compute-core
    │   └── Cargo.toml          # Workspace with compute-core
    │
    └── compute-core/          # Core compute logic
        ├── src/               # All implementation
        ├── Cargo.toml
        └── tests/
```

### Tribunus Compute Repo (Target)
```
Tribunus Compute/
├── compute-native/          # Current: Combined (407 lines lib.rs)
│   ├── src/
│   │   ├── lib.rs           # Both bindings AND logic
│   │   ├── bridge.rs
│   │   ├── session.rs
│   │   └── ... (60+ files)
│   └── Cargo.toml
│
├── mlx-rs-fork/            # Existing: Tribunus fork of mlx-rs
│   ├── mlx-rs/
│   ├── mlx-sys/
│   └── ...
│
├── mlx-rs-core/             # NEW: From OminiX-MLX
│   └── ...
│
├── models/                  # NEW: From OminiX-MLX
│   ├── mistral-mlx/
│   └── qwen3-mlx/
│
└── orion-runtime/          # NEW: From Orion
    └── ...
```

## What Has Been Ported

### ✅ From OminiX-MLX

1. **mlx-rs-core/**: Complete shared infrastructure crate
   - KV cache implementations
   - RoPE utilities
   - Attention mask creation
   - Custom Metal kernels (fused_swiglu, fused_modulate, flash_attention)
   - Audio processing
   - Speculative decoding
   - Token generation infrastructure

2. **models/mistral-mlx/**: Mistral LLM implementation
3. **models/qwen3-mlx/**: Qwen3 LLM implementation

### ✅ From Orion

**orion-runtime/**: Complete ANE runtime
- Core ANE access (ane_runtime.m/h, ane_program_cache.m/h)
- Compiler (builder.c/h, codegen.m/h, graph.c/h, etc.)
- Inference kernels (prefill_ane, decode_ane, decode_cpu, kv_cache)
- Training kernels
- Model loading (weight_loader)
- Applications (CLI)
- Tokenizers
- Makefile build system

### ✅ From Main Tribunus Monorepo

**compute-native/compute-core/**: Core logic separated from bindings
- Placement profiles
- Backend conformance tests
- Decode attribution
- Coreml lifecycle
- Pipeline parity contracts
- Treatment qualification
- Multiple research and evidence profiles

## What Needs To Be Done

### 🔧 Architectural Decision Required

The main Tribunus monorepo has evolved to use a **split architecture** where:
- `compute-native` = thin NAPI/FFI layer
- `compute-core` = all compute logic

The current Tribunus Compute repo uses a **combined architecture** where everything is in `compute-native`.

**You need to decide:**

#### Option A: Migrate to Split Architecture (Recommended)
```bash
# 1. Create compute-core as a workspace member
cd Tribunus Compute
mkdir -p compute-core/src

# 2. Move all logic from compute-native/src/ (except lib.rs) to compute-core/src/
# 3. Update compute-native/src/lib.rs to delegate to compute-core
# 4. Update compute-native/Cargo.toml to include compute-core as workspace member
# 5. Update compute-native/Cargo.toml dependencies to use workspace = true
```

Benefits:
- Matches main monorepo structure
- Cleaner separation of concerns
- Easier testing without NAPI overhead
- Better maintainability

#### Option B: Merge compute-core into compute-native
```bash
# Copy all from compute-core/src/ into compute-native/src/
cp -r compute-native/compute-core/src/* compute-native/src/
# Update compute-native/src/lib.rs to include new modules
```

Benefits:
- Keeps current structure
- Less restructuring
- Faster migration

### 📦 Specific Changes to Port

From main monorepo's `packages/compute-native/Cargo.toml`:

1. **Workspace structure**:
   ```toml
   [workspace]
   members = [
       ".",
       "compute-core",
   ]
   ```

2. **Workspace dependencies**:
   ```toml
   [workspace.dependencies]
   mlx-rs = { path = "../mlx-rs-fork/mlx-rs" }
   mlx-sys = { path = "../mlx-rs-fork/mlx-sys" }
   ```

3. **Additional dependencies**:
   - `rayon = "1"`
   - `async-trait = "0.1"`
   - `coreml-proto = "0.1.0"`
   - `prost = "0.14.4"`
   - `prost-types = "0.14.4"`
   - `tempfile = "3"`
   - `tribunus-evidence-schema` (internal, may need to be removed)

4. **Features**:
   ```toml
   [features]
   default = ["stub-backend"]
   private-development = []
   stub-backend = ["mlx-rs/stub"]
   mlx-backend = ["mlx-rs/metal", "mlx-rs/accelerate", "mlx-rs/safetensors"]
   ```

5. **Release profiles**: Multiple optimized profiles for different scenarios

### 🔄 mlx-rs-fork Synchronization

The main Tribunus monorepo has its own `mlx-rs/`, `mlx-sys/`, etc. at the root level. The Tribunus Compute repo has `mlx-rs-fork/`. 

Check if `mlx-rs-fork/` is in sync with the main monorepo's `mlx-rs/`. If not, consider:
- Copying the latest from main monorepo
- Or ensuring both are updated to the same version

## Files Created

✅ **From OminiX-MLX**:
- `mlx-rs-core/` (complete crate)
- `models/mistral-mlx/` (complete crate)
- `models/qwen3-mlx/` (complete crate)

✅ **From Orion**:
- `orion-runtime/` (complete runtime with all source files)

✅ **From Main Tribunus Monorepo**:
- `compute-native/compute-core/` (complete core implementation)
- `compute-native/Cargo.toml.backup` (main monorepo version)
- `compute-native/src/lib_new.rs` (main monorepo version)

## Recommendations

1. **Review compute-core**: The `compute-native/compute-core/` directory contains the latest core logic from the main monorepo. This should be integrated.

2. **Decide on architecture**: Choose between split (recommended) or combined architecture.

3. **Update dependencies**: Add missing dependencies from main monorepo's Cargo.toml.

4. **Sync mlx-rs-fork**: Ensure it's up to date with main monorepo's mlx-rs.

5. **Integrate mlx-rs-core**: The new mlx-rs-core crate can be used by compute-core to avoid duplicating KV cache, attention, etc.

## Quick Start

To see what's been ported:
```bash
cd /Users/user/Developer/GitHub/Tribunus Compute

# List all new directories
tree -L 2 -I '.git|node_modules|target' .

# Check the new crates
ls -la mlx-rs-core/src/
ls -la models/
ls -la orion-runtime/
ls -la compute-native/compute-core/src/
```

## Questions for You

1. Should I proceed with migrating to the split architecture (Option A)?
2. Or would you prefer to merge compute-core into the existing combined architecture (Option B)?
3. Should I also sync the mlx-rs-fork with the main monorepo's mlx-rs?
4. Are there specific components from candle you still want ported (despite the type incompatibility)?
