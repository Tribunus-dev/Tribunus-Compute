//! Kernel variant selection — compile-time artifact selection, candidate
//! benchmark evidence, and selection receipts for SealedComputeImageExecutable.
//!
//! At compile time the compiler benchmarks candidate kernel implementations
//! against the target profile's hardware contract.  The best-performing
//! candidate is selected per operation/shape-class pair, and a
//! [`KernelSelectionReceipt`] records the selection policy version, candidate
//! artifacts, resource-fit and numerical qualification outcomes, and the
//! chosen winner.  These receipts become part of the
//! [`CompileTimeReceiptBundle`] embedded in the sealed executable.
//!
//! [`KernelCandidateEvidence`] captures per-candidate benchmark results
//! (median/min latency, resource-fit pass/fail, numerical pass/fail) that
//! feed the selection policy.
//!
//! [`KernelConfiguration`] records the tiling parameters and pipeline id
//! for a candidate kernel variant.
//!
//! [`CompileTimeReceiptBundle`]: crate::compute_image::executable::schema::CompileTimeReceiptBundle

pub mod selection;

pub mod evidence;
pub use evidence::*;

pub mod compatibility;
pub use compatibility::*;

pub use selection::*;
