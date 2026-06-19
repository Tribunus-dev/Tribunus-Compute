pub mod auth;
pub mod cpu;
pub mod benchmark;
pub mod models;
pub mod rate_limiter;

#[cfg(feature = "mlx-backend")]
pub mod admin;
#[cfg(feature = "mlx-backend")]
pub mod engine;
#[cfg(feature = "mlx-backend")]
pub mod routes;
