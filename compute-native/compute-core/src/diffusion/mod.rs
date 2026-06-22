//! Discrete masked diffusion module.
//!
//! Provides CPU-side sampling, canvas management, and scheduling primitives
//! for diffusion-based token generation. These operate on `Vec<f32>` and
//! `Vec<u32>` — no MLX Array dependency.

pub mod canvas;
pub mod sampler;
pub mod scheduler;

pub use canvas::TokenCanvas;
pub use sampler::{ConvergenceResult, DiffusionSampler, SamplerOutput};
pub use scheduler::DiffusionScheduler;
