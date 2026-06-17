//! Quantization strategies ported from omlx.
//!
//! - `oq`: oQ dynamic quantization (load-time mixed-precision)
//! - `turboquant_kv`: TurboQuant KV cache quantization
//!
//! Reference implementations in `ref/omlx/oq.py` and `ref/omlx/turboquant_kv.py`.
//! Design docs in `docs/omlx-oq-quantization.md` and `docs/omlx-turboquant-kv.md`.

pub mod oq;
pub mod turboquant_kv;
