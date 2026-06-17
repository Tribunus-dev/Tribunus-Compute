//! Cache strategies ported from omlx.
//!
//! - `prefix_cache`: Block-aware prefix caching with automatic prefix discovery
//! - `paged_ssd_cache`: SSD-backed paged KV cache with safetensors serialization
//!
//! Reference implementations in `ref/omlx/cache/`.
//! Design docs in `docs/omlx-prefix-cache.md` and `docs/omlx-ssd-cache.md`.

pub mod paged_ssd_cache;
pub mod prefix_cache;
