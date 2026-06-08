# Releasing

This document describes the release process for the Tribunus Compute Kernel.

---

## 1. Release Philosophy

Every release **must** identify the exact dependency tuple that the kernel
was built and qualified against:

| Layer | Identifier | Source |
|-------|------------|--------|
| Kernel version | `compute-vX.Y.Z` | This repo |
| tribunus-mlx-rs version | `vA.B.C-tribunus.N` | `mlx-rs-fork/mlx-rs/Cargo.toml` |
| tribunus-mlx-sys version | `vA.B.C-tribunus.N` | `mlx-rs-fork/mlx-sys/Cargo.toml` |
| MLX C version | `vA.B.C` (patched) | Vendored in `mlx-rs-fork/` |
| MLX Core version | `vA.B.C` | `mlx/version.h` / upstream mlx |

These five values are the **release tuple**. They must be declared in the
GitHub Release body and embedded in any generated receipt or provenance
artifact. Changing any tuple member between releases changes the contract;
the CHANGELOG must note which values moved.

### Why the tuple matters

The kernel is tightly coupled to all five layers. A patch to the MLX C
bindings or a bump to tribunus-mlx-sys can change quantization output,
memory telemetry accuracy, or FFI crash surface. The tuple documents the
exact boundary.

---

## 2. Versioning Scheme

The kernel follows the scheme:

```
compute-vX.Y.Z[-PRERELEASE]
```

- **X.Y.Z** — Semantic Versioning (MAJOR.MINOR.PATCH).
- **compute-v** prefix — disambiguates kernel releases from the mlx-rs fork
  releases and parent monorepo releases.
- **PRERELEASE** — optional qualifier (`alpha.1`, `beta.2`, `rc.1`). Pre-release
  versions are not published to end users.

Semver rules apply:

| Bump | When |
|------|------|
| MAJOR | Breaking change to the N-API binding surface (function signatures drop or change), breaking ABI change to a frozen ABI contract, or breaking change to a receipt schema field. |
| MINOR | New capability (e.g. `supports_external_array` flips from `false` to `true`), new N-API export, new frozen ABI, new receipt schema, new qualified path. |
| PATCH | Bug fix, performance improvement, test-only changes, documentation, or build system change that does not alter any binding, capability, or schema. |

Pre-release versions use dot-separated qualifiers:

- `compute-v0.2.0-alpha.1` — first alpha of the 0.2.0 minor release.
- `compute-v0.2.0-rc.1` — release candidate.

Pre-releases satisfy the same qualification as stable releases but are not
tagged as a "Latest" release on GitHub.

---

## 3. Release Checklist

### Prerequisites

- You are on the release branch (see Section 6) with a clean working tree
  (`git status` reports nothing dirty).
- You have committed or stashed all in-progress work.
- All CI/CD checks pass on the target commit.

### Step 1 — Run full qualification suite on target hardware

The kernel targets `aarch64-apple-darwin`. Run on a physical Apple Silicon
Mac (no emulation).

```shell
# Full test suite (all modules)
cargo test --workspace --manifest-path compute-native/Cargo.toml

# Release build — LTO may expose codegen issues not present in debug
cargo build --release --manifest-path compute-native/Cargo.toml

# N-API smoke test (requires the Node.js addon to load)
node -e "require('./compute-native/index.js')"

# Concurrency stress (quantized matmul — SIGSEGV gate)
cargo test --manifest-path compute-native/Cargo.toml concurrent
```

All tests must pass. No SIGSEGV. No unexpected `panic!`. No new warnings
introduced by the release build.

If you are qualifying a capability change (e.g. new feature gate), run the
targeted qualification test for that capability separately and include the
output in the release notes.

### Step 2 — Generate receipt artifacts

Receipts are the provenance for the release tuple. Generate them from the
build artifacts:

```shell
# Extract the release tuple from Cargo.toml files
MLX_RS_VERSION=$(grep 'version' mlx-rs-fork/mlx-rs/Cargo.toml | head -1 | awk '{print $3}' | tr -d '"')
MLX_SYS_VERSION=$(grep 'version' mlx-rs-fork/mlx-sys/Cargo.toml | head -1 | awk '{print $3}' | tr -d '"')
KERNEL_VERSION=$(grep 'version' compute-native/Cargo.toml | head -1 | awk '{print $3}' | tr -d '"')

# Build the binary and capture hashes
sha256sum compute-native/target/release/libtribunus_compute_native.dylib > compute-native/release/receipt.sha256
```

Store the receipt alongside the release artifacts. It belongs on the GitHub
Release as an attached file.

### Step 3 — Update CHANGELOG.md

1. Move the items from `[Unreleased]` into a new dated section header
   for the version being released.
2. Ensure every tuple change is noted (mlx-rs fork bump, mlx-sys bump,
   MLX C patch version, MLX Core version).
3. Add a new `[Unreleased]` header at the top for the next cycle.
4. Add the compare link at the bottom of the file.

Commit the CHANGELOG update:

```shell
git add CHANGELOG.md
git commit -m "docs: prepare compute-vX.Y.Z changelog"
```

### Step 4 — Tag the commit

```shell
git tag -a compute-vX.Y.Z -m "compute-vX.Y.Z"
git push origin compute-vX.Y.Z
```

The tag **must** start with `compute-v` to distinguish it from mlx-rs fork
tags and parent monorepo tags.

### Step 5 — Build release artifacts

```shell
cargo build --release --manifest-path compute-native/Cargo.toml
cd compute-native
napi build --platform --release
```

This produces:

- `compute-native/tribunus-compute-native.aarch64-apple-darwin.node` — the
  N-API addon.
- `compute-native/index.js` — the JS loader.
- `compute-native/index.d.ts` — TypeScript type declarations.
- `target/release/libtribunus_compute_native.dylib` — standalone dylib
  (when `crate-type = ["cdylib"]` is active).
- `target/release/tribunus-compute-worker` — standalone binary.
- `target/release/tribunus-fake-worker` — test/fake binary.

Collect these into a release directory:

```shell
mkdir -p compute-native/release
cp compute-native/target/release/libtribunus_compute_native.dylib compute-native/release/
cp compute-native/target/release/tribunus-compute-worker compute-native/release/
cp compute-native/target/release/tribunus-fake-worker compute-native/release/
cp compute-native/tribunus-compute-native.aarch64-apple-darwin.node compute-native/release/
cp compute-native/index.js compute-native/release/
cp compute-native/index.d.ts compute-native/release/
```

### Step 6 — Publish to GitHub Releases

1. Draft a new release on GitHub, targeting the tag `compute-vX.Y.Z`.
2. Title: `compute-vX.Y.Z`
3. Body — include:

   - The **release tuple** table (Section 1).
   - A summary of changes (copy the relevant CHANGELOG section).
   - Any known qualification gaps.
   - Link to the full receipt.

4. Attach all release artifacts from `compute-native/release/`.
5. Attach the receipt (`receipt.sha256`).
6. Mark as "Latest" for stable releases; uncheck "Latest" for pre-releases.
7. Publish.

---

## 4. MLX Fork Release

The `mlx-rs-fork/` subdirectory is a compatibility fork of the upstream
`mlx-rs` crate. Fork releases are versioned independently and tagged with
their own prefix.

### Versioning

- `tribunus-mlx-rs` follows the upstream `mlx-rs` version with a
  `-tribunus.N` pre-release suffix:
  `0.25.3-tribunus.1`, `0.25.3-tribunus.2`, etc.
- `tribunus-mlx-sys` follows the MLX C version with the same suffix:
  `0.6.0-tribunus.1`, `0.6.0-tribunus.2`, etc.
- The suffix `N` increments for each Tribunus-specific patch to the fork
  (bug fix, FFI patch, or compatibility shim).
- Fork tags use the `mlx-rs-` and `mlx-sys-` prefixes:
  `mlx-rs-0.25.3-tribunus.1`, `mlx-sys-0.6.0-tribunus.1`.

### Relationship to kernel releases

A fork release **precedes** a kernel release when the fork changes. The
sequence is:

1. Bump fork version / apply patches.
2. Tag fork release (`mlx-rs-0.25.3-tribunus.2`).
3. Update kernel `Cargo.toml` to reference the new fork version.
4. Qualify the kernel against the updated fork.
5. Cut kernel release (`compute-vX.Y.Z`).

Fork-only releases (kernel unchanged) are valid independently — the fork
changelog documents what changed. The kernel release notes then reference
the fork tag.

### Upstream sync

When upstream `mlx-rs` or `MLX C` releases a new version, the fork is
rebased onto the upstream tag and re-patched. The fork version number
jumps to match the new upstream version, with a fresh `-tribunus.1`
suffix.

---

## 5. Rollback

### When to roll back

A release is rolled back when it introduces a regression that was not
caught by the qualification suite — for example:

- A SIGSEGV that only manifests on specific hardware.
- A correctness regression in quantized matmul output.
- A broken N-API binding that crashes consumers.
- A capability report that misreports hardware support.

### Procedure

1. **Tag the revert commit** on the release branch:
   ```shell
   git revert --no-edit compute-vX.Y.Z
   git tag compute-vX.Y.Z-revert
   git push origin compute-vX.Y.Z-revert
   ```

2. **Delete the release tag** from the remote:
   ```shell
   git push --delete origin compute-vX.Y.Z
   ```

3. **Delete the GitHub Release** (Settings → Releases → delete).

4. **Restore the previous release** as the "Latest" release.

5. **Create an issue** documenting:
   - The regression (with reproduction steps).
   - What the qualification suite missed.
   - How the qualification suite must be improved before the next release.

6. **Cut a patch release** once the fix is committed, following the normal
   checklist. The patch version increments the PATCH component
   (`compute-vX.Y.(Z+1)`).

### Pre-release rollback

Pre-release tags (`alpha`, `beta`, `rc`) can be deleted without a revert
commit. Simply delete the tag and GitHub Release, fix the issue, and cut
a new pre-release with an incremented qualifier
(e.g. `compute-v0.2.0-rc.2` → `compute-v0.2.0-rc.3`).

---

## 6. Supported Release Branches

| Branch | Provides | Patch policy |
|--------|----------|-------------|
| `main` | Latest development | Unstable; may accept breakage. |
| `release/compute-v0.1.x` | v0.1.x stable | Critical bug fixes only (SIGSEGV, data corruption, N-API binding failure). No new features. |
| `release/mlx-rs-0.25.x-tribunus` | mlx-rs fork v0.25.x-tribunus | Upstream-sync patches and critical FFI fixes only. No kernel changes. |

The `release/compute-v0.1.x` branch is created at the `compute-v0.1.0` tag.
Subsequent patch releases (`compute-v0.1.1`, `compute-v0.1.2`) are tagged
from this branch and merged back to `main`.

### Creating a new release branch

```shell
git branch release/compute-v0.X.x compute-v0.X.0
git push origin release/compute-v0.X.x
```

New major or minor releases create new release branches. The oldest release
branch is retired (no further patches) once two newer major/minor releases
exist.
