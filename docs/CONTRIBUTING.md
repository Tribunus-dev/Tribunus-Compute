# Contributing to Tribunus Compute Kernel

## 1. Philosophy

The Tribunus Compute Kernel is an Apple Silicon-native inference runtime that
bridges MLX, Core ML, and IOSurface-backed zero-copy tensor transport.  It
powers the Tribunus agent platform's on-device compute layer.

**Correctness and reproducible evidence take precedence over throughput or
feature breadth.**

Every claim about performance, memory behavior, or numerical correctness must
be backed by a repeatable test, a fixture, or a qualification run.  We do not
optimize on speculation: profile first, measure before and after, commit the
receipts.  A feature that cannot be qualified on real Apple Silicon hardware
in CI-free form (at minimum with `cargo test -- --nocapture` on a developer
machine) is not ready for merge.

The project maintains frozen ABI contracts (`tribunus-iosurface-fp16-arena-v1`,
`tribunus-coreml-stateful-island-v1`, `tribunus-hybrid-compute-image-v1`)
and a frozen capability-namespace in `capability.rs`.  Backward-incompatible
changes to these require a new version — not a silent mutation.

## 2. Getting Started

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| macOS | ≥ 14.0 | Required for Core ML, Metal, IOSurface APIs |
| Apple Silicon | M1+ | aarch64 only; no x86_64 emulation supported |
| Rust | 1.85+ | edition 2021, resolver 2 |
| Bun | 1.2+ | For JS-side tests and napi build orchestration |
| Python | 3.13 ARM64 | Only needed for Core ML artifact compilation |
| coremltools | 9.0 | Installed via `.venv-coreml-compiler` (see below) |
| Xcode | 16+ | For ObjC++ bridge compilation (`-fobjc-arc`, `-fblocks`) |

**macOS SDK frameworks** (auto-linked by `build.rs`):
- `CoreML.framework`
- `CoreVideo.framework`
- `IOSurface.framework`

### Workspace setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Tribunus-dev/Tribunus-Compute
cd Tribunus-Compute

# The mlx-rs-fork submodule points at the compat/mlx-core-0.31.2 branch.
# If you already cloned without --recurse-submodules:
git submodule update --init --recursive

# Verify fork pointer
cd mlx-rs-fork && git log --oneline -1
# Expected: at or past 93ed8db (mlx-rs 0.25.3-tribunus.1, mlx-sys 0.6.0-tribunus.1)
cd ..
```

### Building

```bash
# Build the Rust crate (release — LTO, stripped, single codegen unit)
cargo build --release -p tribunus-compute-native

# Build the napi addon (produces tribunus-compute-native.aarch64-apple-darwin.node)
cd compute-native && bun run build && cd ..

# Development build (faster iteration, debug symbols)
cargo build -p tribunus-compute-native
cd compute-native && bun run build:debug && cd ..

# Core ML compiler environment (optional, for compile_pipeline work)
python3.13 -m venv .venv-coreml-compiler
source .venv-coreml-compiler/bin/activate
pip install coremltools==9.0
deactivate
```

### Running tests

```bash
# Rust unit and integration tests
cargo test -p tribunus-compute-native -- --nocapture

# Rust tests with release profile (slower compile, reflects production paths)
cargo test --release -p tribunus-compute-native -- --nocapture

# JS-side napi smokes
cd compute-native && bun test && cd ..

# Run a specific test module
cargo test -p tribunus-compute-native arena::tests -- --nocapture

# Run qualification tests (requires Metal-capable Apple Silicon)
cargo test -p tribunus-compute-native --test qualification -- --nocapture
```

## 3. Development Workflow

### Branch strategy

The repository uses a simple trunk-based model:

- `main` — latest release-ready state.  Every merge must pass review and
  at minimum compile cleanly.
- Short-lived feature branches off `main`, named `feat/<description>` or
  `fix/<description>`.
- The mlx-rs fork lives on branch `compat/mlx-core-0.31.2`.  See
  §8 (MLX Fork) for submodule changes.

### Commit style

- Commits are atomic: one logical change per commit.
- The commit message subject line is capitalized, imperative, ≤ 72 characters.
  The body (if needed) wraps at 72 columns and explains *why* more than *what*.
- Prefix when helpful: `arena:`, `kv_cache:`, `supervisor:`, `mlx:`,
  `build:`, `qual:`, `doc:`, `chore:`.
- Every commit must compile.  It is acceptable to split work across commits
  that each pass `cargo check`.

### PR process

1. Open a draft PR early; mark it ready for review once tests pass and CI
   (or equivalent local verification) is green.
2. At least one other contributor must approve.  For the `compute-native/`
   crate, changes to frozen ABIs, capability constants, or receipt schemas
   require sign-off from a maintainer.
3. Squash merge into `main`.

## 4. Testing Standards

### What needs tests

Every new public API, every bug fix, and every structural refactor must
include tests.  Specifically:

- **Arena lifecycle**: All valid and illegal transitions, pool acquire/release
  cycles, budget enforcement, max-per-key limits.
- **Capability detection**: Detection logic and serde round-trips.
- **Supervisor**: Job lifecycle (submit, cancel, shutdown), worker
  registration, cancellation propagation.
- **KV cache**: Sliding and global eviction, concurrent access, clear.
- **Errors**: Builder pattern, Display impls, every variant.
- **Hybrid profiles**: Serde, schema validation, tensor-flow correctness.
- **Receipts**: Emitter structs, copy classification, serde round-trips.
- **Compute image**: Residency invariants, active memory bounds, quantized
  matmul parity (fused vs dequantize+matmul), concurrent execution.
- **Frozen ABIs**: Structural identity tests that assert receipt JSON schemas
  and capability constant names do not drift.

### Fixture-based testing

Prefer fixture-based tests over mocks.  A fixture is a real MLX array, a
real IOSurface arena, a safetensors header, or a pre-compiled ComputeImage
directory — not a trait object pretending to be one.

The `compute_image` tests use the following fixture pattern (4 tests):
- `handles=4→29→4` — registry handle count after compute image lifecycle
- `mlx_active=72128→71288` — MLX Metal active memory bounded between
  compile and run phases
- Quantized matmul parity — fused quantized matmul matches
  dequantize+float-matmul bitwise
- Concurrent — 2×100 + 4×50 heavy quantized matmul, no SIGSEGV

To add a new fixture-based test, create a real model config and safetensors
shard in the `tests/fixtures/` directory (or generate one programmatically
at test time using the crate's own `config.rs` builders).  The fixture
directory structure is:

```
tests/fixtures/<name>/
  config.json
  model-00001-of-NNNNN.safetensors   (optional, for real weight loading)
```

### Qualification testing

Qualification tests are gated behind `#[cfg(feature = "qualification")]` or
live in `tests/qualification.rs`.  They require real Apple Silicon hardware
with Metal.  See §7.

### What does NOT need tests

- Trivial getter/setter boilerplate (unless it has error paths).
- Documentation examples that duplicate unit tests.
- Bridge code that is pure FFI dispatch (test the abstraction that calls it,
  not the dispatch function itself).

## 5. Code Style

### rustfmt

All Rust code must be formatted with `rustfmt` using the project defaults
(no `rustfmt.toml` overrides).  CI enforces this.

```bash
cargo fmt --check
```

### Clippy

Warnings are denied at the workspace level:

```toml
[workspace.lints.clippy]
all = "warn"
pedantic = "warn"
nursery = "warn"
cargo = "warn"
```

The project uses `#![deny(clippy::all, clippy::pedantic, clippy::nursery,
clippy::cargo)]` in `compute-native/src/lib.rs`.  Run before pushing:

```bash
cargo clippy --all-targets -- -D warnings
```

If a lint fires on justified code, suppress it with `#[allow(...)]` on the
minimum enclosing scope and add a comment explaining why.  Do not suppress
at the crate level.

### Unsafe code policy

`unsafe` is allowed only in well-defined circumstances:

1. **FFI calls to mlx-sys C bindings.**  Each call must be wrapped in a safe
   function that documents safety invariants (e.g. "`ptr` must be non-null
   and point to a valid `mlx_array` returned by `mlx_array_new_*`").
2. **IOSurface / Core Video / Core ML ObjC++ bridge.**  The `.mm` files
   interact with Objective-C runtime and Core Foundation; the Rust side
   treats them as safe extern calls.
3. **`Send + Sync` impls on internal registries.**  Only when required by
   `lazy_static` or `once_cell`.  Document why the type is actually thread-
   safe (e.g. interior mutability via `parking_lot::RwLock`).

The workspace enforces `[lints.rust] unsafe_code = "warn"`.  New `unsafe`
blocks must be justified in the PR description or commit message.

### Naming conventions

- Rust: `snake_case` for functions, `CamelCase` for types, `SCREAMING_SNAKE`
  for constants and statics.
- Frozen capability constants in `capability.rs` use the `CAP_` prefix with
  `SCREAMING_SNAKE` (e.g. `CAP_IOSURFACE_CREATION`).
- Receipt struct field names use `snake_case` (serialized to JSON as-is).
- JS/TS: `camelCase` for functions, `PascalCase` for types.

### Unused code

Unused public items that serve external API contracts (napi exports, public
config types) may carry `#[allow(dead_code)]` with a comment indicating the
external consumption site.  All other dead code should be removed.

## 6. Documentation

### What needs docs

- **Every public item** in the `compute-native` crate must have a
  doc comment (`///` or `//!`).  The workspace enforces `missing_docs =
  "warn"`.
- **Frozen ABIs** (arena layout, capability names, receipt schemas) must be
  documented in `STATUS.md` and the `///` comment on the defining type.
- **napi-rs exports** in `lib.rs` need doc comments that describe the JS
  calling convention, parameter types, and return value.
- **Build system changes** (`build.rs`, `Cargo.toml` features) must explain
  the target OS condition and linking rationale.

### Where docs live

| Content | Location |
|---|---|
| Crate-level docs | `compute-native/src/lib.rs` (`//!` module doc) |
| Component docs | Doc comments on the component's root module file |
| Frozen ABIs, capability table, receipt schemas | `compute-native/STATUS.md` |
| API alignment review | `compute-native/CODE_REVIEW.md` |
| Environment preflight | `compute-native/environment.json` |
| Core ML compiler | `compute-native/tools/coreml-compiler/` |
| This contributing guide | `docs/CONTRIBUTING.md` |
| Issue tracking | GitHub Issues, OMP task board (`docs/json/omp/`) |

### Doc comment style

- First sentence is a summary line.  Blank line before details.
- Use markdown: backtick code, bullet lists, links to types.
- Include a `# Panics` or `# Safety` section where applicable.
- For napi exports, show the JS call pattern in a fenced code block:

```rust
/// Execute the full 48-layer model from a compiled ComputeImage.
///
/// ```js
/// const tokenId = compute.runFullModelFromImage("./image-dir", inputIds);
/// ```
///
/// Returns the next token ID directly — no logits cross the FFI boundary.
```

## 7. Qualification

Qualification is the process of running a defined set of tests on real
Apple Silicon hardware and recording the results as a signed or hashed
artifact.  It is required before claiming a capability, a performance
budget, or memory bound as "verified."

### Running qualification

```bash
# Full qualification suite
cargo test --release -p tribunus-compute-native --test qualification -- --nocapture

# Qualification produces a receipt JSON in the current directory:
#   qualification-receipt-<timestamp>.json
```

### Hardware requirements

| Model | Minimum | Notes |
|---|---|---|
| Apple Silicon | M1 | All qualification |
| Memory | 16 GB | All qualification (48-layer compute image) |

Optional — expanded qualification:
| Apple Silicon | M2 Ultra or M3 Max | Core ML stateful island, IOSurface throughput tests |

### What qualification covers

1. **Arena lifecycle** — creation, transition through all states, pool
   reuse, clean destruction.
2. **MLX Metal memory** — active memory bounds before/during/after compute
   image compile and run.
3. **Quantized matmul parity** — fused quantized matmul output matches
   dequantize + float matmul bitwise (relative tolerance ≤ 1 ULP per
   quantization error bounds).
4. **Concurrent execution** — 2×100 + 4×50 heavy quantized matmul
   invocations with no SIGSEGV, no handle leak.
5. **Zero-copy round trip** — MLX → IOSurface → Core ML → IOSurface → MLX
   with `application_copy_free = true` classification.
6. **KV cache** — sliding and global eviction, concurrent reader safety,
   clear isolation.

### Artifact format

Each qualification run produces a JSON receipt file containing:

```jsonc
{
  "version": "1.0",
  "timestamp": "2026-06-08T12:00:00Z",
  "host": {
    "model": "Mac14,2",        // or equivalent sysctl hw.model
    "chip": "Apple M1",
    "memory_gb": 16,
    "macos_version": "26.5.1"
  },
  "rust_toolchain": "1.85.0",
  "tests": {
    "total": 12,
    "passed": 12,
    "failed": 0,
    "skipped": 0
  },
  "capabilities": {
    "metal_available": true,
    "accelerate_available": true,
    "supports_quantized_matmul": true,
    "supports_arena_pooling": true,
    "supports_kv_cache": true,
    // ... all capabilities from capability.rs
  },
  "memory_measurements": {
    "compile_active_memory_bytes": 72128,
    "run_active_memory_bytes": 71288
  },
  "receipts": [
    // ArenaCreationReceipt, CoreMlPredictionReceipt, HybridJobReceipt
  ]
}
```

Receipt files are named `qualification-receipt-<YYYY-MM-DD>.json` and should
be committed alongside capability-support documentation changes.

## 8. MLX Fork

The crate depends on a Tribunus fork of `mlx-rs` at version 0.25.3,
compatible with MLX Core 0.31.2 (upstream mlx-rs targets Core 0.25.x).
The fork lives as a git submodule in `mlx-rs-fork/`, branch
`compat/mlx-core-0.31.2`.

### Submodule anatomy

```
mlx-rs-fork/
  mlx-rs/       — Rust bindings (crate mlx-rs 0.25.3-tribunus.1)
  mlx-sys/      — FFI bindings to MLX C (crate mlx-sys 0.6.0-tribunus.1)
  mlx-c/        — Vendored MLX C v0.6.0 (patched)
```

The workspace `Cargo.toml` references submodule paths directly:

```toml
mlx-rs = { path = "../mlx-rs-fork/mlx-rs", features = ["metal", "accelerate", "safetensors"] }
mlx-sys = { path = "../mlx-rs-fork/mlx-sys" }
```

### Making fork changes

If you need to change the mlx-rs bindings (e.g. add a new FFI wrapper, fix
an upstream bug, update to a newer mlx-c):

```bash
cd mlx-rs-fork
git checkout compat/mlx-core-0.31.2
# Make changes in mlx-rs/, mlx-sys/, or mlx-c/
git add -A && git commit -m "fix: ..."
git push origin compat/mlx-core-0.31.2
cd ..
git add mlx-rs-fork
git commit -m "chore: bump mlx-rs-fork to <new-commit-hash>"
```

### Updating the pointer

After the fork branch has advanced:

```bash
git submodule update --remote mlx-rs-fork
git add mlx-rs-fork
git commit -m "chore: update mlx-rs-fork to <description>"
```

### Patch notes

The fork carries the following patches on top of upstream mlx-rs 0.25.3:

1. **FFT API**: Added `FFTNorm::Backward` variant for MLX Core 0.31.2.
2. **Version pins**: `mlx-rs = 0.25.3-tribunus.1`, `mlx-sys = 0.6.0-tribunus.1`.
3. **Quantization fix**: Pass non-null mode string `"affine"` to quantize ops
   (upstream segfaulted with null pointer).
4. **Metal + Accelerate**: Both features enabled on all builds.

These are documented in the fork branch's commit history and in the
`README.md` fork-commits table at the project root.

## 9. Review Process

### What reviewers check

- **Correctness**: Does the change handle edge cases (empty input, zero
  dimensions, concurrent access, cancellation)?  Are error paths tested?
- **Evidence**: Are performance claims backed by qualification receipts?
  Are memory bounds asserted in tests?  Are fixture-based tests used where
  a real MLX array or IOSurface is involved?
- **Safety**: Every `unsafe` block — is it justified, scoped minimally, and
  documented with invariants?  No new `unsafe` without a safety comment.
- **ABI stability**: Do changes to arena layout, capability constants, or
  receipt schemas require a frozen-ABI version bump?  If so, the PR must
  include the version bump and migration path.
- **Coverage**: Do new exports have tests?  Do bug fixes have a regression
  test?  Are documentation examples tested (`cargo test --doc`)?
- **Style**: `cargo fmt --check`, `cargo clippy -- -D warnings`.
  Unsuppressed lint warnings are a blocking finding.
- **Submodule discipline**: Fork changes go through the `mlx-rs-fork`
  submodule; the compute-native crate never directly edits files under
  `mlx-rs-fork/` in the same commit.

### Merge criteria

A PR may be merged (squash) when:

1. At least one approving review (no unresolved comments).
2. `cargo check --all-targets` passes.
3. `cargo test -p tribunus-compute-native` passes.
4. `cargo clippy --all-targets -- -D warnings` passes.
5. `cargo fmt --check` passes.
6. Changes to frozen ABIs, capability constants, or receipt schemas have
   explicit maintainer sign-off.
7. The `mlx-rs-fork` submodule pointer is at the intended commit (not stale
   from a previous merge).

Items 2–5 can be verified locally if CI is unavailable; paste the
verification output into the PR.

### CLA / DCO

By contributing, you agree that your contributions are licensed under the
same AGPL-3.0-only terms as the project (see §10).  Every commit must be
signed off per the Developer Certificate of Origin:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Use `git commit -s` to append the sign-off.  Contributions without a
sign-off may be rejected.

## 10. License


## 10. License and Contributor Agreement

The Tribunus Compute Kernel is dual-licensed under **AGPL-3.0-only** and a
separate **commercial license**. See [README.md §12](../README.md#12-license)
and [docs/LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) for the full policy.

### Contributor License Agreement

To preserve the ability to continue dual-licensing the kernel, all substantial
external contributions require a contributor license agreement (CLA) that
grants Tribunus, Inc. sufficient copyright and patent rights to include the
contribution under both the AGPL and the commercial license.

Before submitting a substantial pull request:

1. Contact `license@tribunus.io` with a summary of your intended contribution.
2. You will receive a lightweight CLA form.
3. Sign and return it before the PR is merged.

Trivial contributions (typo fixes, one-line corrections, build configuration
adjustments) do not require a CLA. If you are unsure whether your contribution
is substantial, open an issue first.

Without a CLA, external contributions limit the project's ability to offer
commercial licenses. If an unsigned contribution is merged, it creates code
that cannot be included in commercial-licensed distributions without the
contributor's separate permission.

### Third-Party Dependencies

- **mlx-rs / mlx-sys**: MIT OR Apache-2.0 (upstream), AGPL-3.0-only (Tribunus patches)
- **napi-rs / napi-derive**: MIT
- **safetensors**: Apache-2.0
- **MLX C**: MIT
- **Core ML / IOSurface / Core Video**: Proprietary Apple frameworks — linked
  at runtime on macOS; no distribution of framework binaries

SPDX-License-Identifier: AGPL-3.0-only
