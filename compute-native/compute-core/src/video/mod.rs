//! Multi-modal video support for compute-core.
//!
//! Video is a sequence of frames processed by the vision encoder per-frame,
//! with temporal cross-attention between frames.  This module provides:
//!
//! - [`decoder`]: Frame extraction from video files (MP4, MOV, WebM).
//! - [`encoder`]: Temporal video encoding (per-frame vision + temporal agg).
//! - [`types`]: Shared multi-modal input types (Image, Audio, Video).

pub mod decoder;
pub mod encoder;

pub use decoder::{extract_frames, MAX_VIDEO_FRAMES};
pub use encoder::{FrameEncoder, TemporalAggregation, VideoConfig, VideoEncoder};
