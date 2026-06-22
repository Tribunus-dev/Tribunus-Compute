//! Structured JSON logging to stderr.
//!
//! Each log line is a single-line JSON object with:
//! - `timestamp`: ISO 8601 UTC
//! - `level`: debug / info / warn / error
//! - `module`: Rust module path (via `module_path!()`)
//! - `trace_id`: optional request-scoped identifier
//! - `message`: human-readable description
//! - `fields`: optional extra key-value payload

use serde::Serialize;
use std::collections::HashMap;

// ── Log level ───────────────────────────────────────────────────────────────

/// Log severity level.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

// ── Log entry ───────────────────────────────────────────────────────────────

/// A single structured log entry suitable for JSON serialization.
#[derive(Debug, Serialize)]
pub struct LogEntry<'a> {
    pub timestamp: String,
    pub level: LogLevel,
    pub module: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fields: Option<HashMap<String, serde_json::Value>>,
}

// ── Core log functions ──────────────────────────────────────────────────────

/// Emit a structured JSON log line to stderr.
///
/// The entry is serialised as a single-line JSON object and written to stderr
/// followed by a newline.  If serialisation fails (should not happen for
/// reasonable inputs) the error is silently dropped.
pub fn log(level: LogLevel, module: &'static str, msg: &str) {
    emit(LogEntry {
        timestamp: crate::now_iso8601(),
        level,
        module,
        trace_id: None,
        message: msg.to_owned(),
        fields: None,
    });
}

/// Emit a structured JSON log line with extra key-value fields.
pub fn log_with_fields(
    level: LogLevel,
    module: &'static str,
    msg: &str,
    fields: HashMap<String, serde_json::Value>,
) {
    emit(LogEntry {
        timestamp: crate::now_iso8601(),
        level,
        module,
        trace_id: None,
        message: msg.to_owned(),
        fields: Some(fields),
    });
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn emit(entry: LogEntry<'_>) {
    if let Ok(line) = serde_json::to_string(&entry) {
        // Writing atomically to stderr – use_line_writer would buffer;
        // we write directly for immediate visibility in production logs.
        use std::io::Write;
        let _ = writeln!(std::io::stderr().lock(), "{line}");
    }
}

// ── Convenience macros ──────────────────────────────────────────────────────

/// Log at Info level.  Expands to `log(LogLevel::Info, module_path!(), $msg)`.
#[macro_export]
macro_rules! log_info {
    ($msg:expr $(,)?) => {
        $crate::logging::log(
            $crate::logging::LogLevel::Info,
            module_path!(),
            $msg,
        )
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::logging::log(
            $crate::logging::LogLevel::Info,
            module_path!(),
            &format!($fmt, $($arg)*),
        )
    };
}

/// Log at Error level.
#[macro_export]
macro_rules! log_error {
    ($msg:expr $(,)?) => {
        $crate::logging::log(
            $crate::logging::LogLevel::Error,
            module_path!(),
            $msg,
        )
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::logging::log(
            $crate::logging::LogLevel::Error,
            module_path!(),
            &format!($fmt, $($arg)*),
        )
    };
}

/// Log at Warn level.
#[macro_export]
macro_rules! log_warn {
    ($msg:expr $(,)?) => {
        $crate::logging::log(
            $crate::logging::LogLevel::Warn,
            module_path!(),
            $msg,
        )
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::logging::log(
            $crate::logging::LogLevel::Warn,
            module_path!(),
            &format!($fmt, $($arg)*),
        )
    };
}

/// Log at Debug level.
#[macro_export]
macro_rules! log_debug {
    ($msg:expr $(,)?) => {
        $crate::logging::log(
            $crate::logging::LogLevel::Debug,
            module_path!(),
            $msg,
        )
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::logging::log(
            $crate::logging::LogLevel::Debug,
            module_path!(),
            &format!($fmt, $($arg)*),
        )
    };
}

// Re-export macros as crate-level items for Rust 2021 edition compatibility.
// #[macro_export] macros in submodules are not automatically in scope for
// binary targets within the same crate. The pub(crate) use allows importing
// them with `use crate::log_info;`.
pub use log_debug;
pub use log_error;
pub use log_info;
pub use log_warn;
