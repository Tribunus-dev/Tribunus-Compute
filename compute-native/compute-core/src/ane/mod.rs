//! ANE (Apple Neural Engine) draft model module.
//!
//! Provides [`AneDraftModel`], a [`DraftModel`](crate::speculative::DraftModel)
//! implementation that runs a small Core ML language model entirely on the
//! Neural Engine via the IOSurface zero-copy path.

pub mod draft_model;
pub mod hot_row_predictor;
pub mod moe_scheduler;
pub mod sink_detector;
pub mod weight_row_cache;
