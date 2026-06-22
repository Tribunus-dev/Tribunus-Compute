//! Crash breadcrumb — writes structured JSON markers before and after each
//! native MLX/CoreML call so that a crash can be attributed to the exact
//! operation in progress.
//!
//! File: `$TRIBUNUS_BREADCRUMB_DIR/breadcrumbs-{pid}.jsonl`
//! (default: `/tmp/tribunus-breadcrumbs/`). Truncated on first open.
//! Each line is a JSON object — "before" entries carry the full operation
//! descriptor; "after" entries record success and elapsed microseconds.
//!
//! A crash during a native call leaves the last fsync'd "before" entry as
//! the crash breadcrumb trail.

use parking_lot::Mutex;
use serde_json::json;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::LazyLock;

static BREADCRUMB_WRITER: LazyLock<Mutex<Option<File>>> = LazyLock::new(|| Mutex::new(None));

/// Lazily initialise the breadcrumb file.
///
/// Creates the directory if missing, opens (or truncates) the per-PID file,
/// and stores the handle for the lifetime of the process.  Idempotent after
/// the first call.
fn ensure_initialized(pid: u32) {
    let mut guard = BREADCRUMB_WRITER.lock();
    if guard.is_some() {
        return;
    }
    let dir = std::env::var("TRIBUNUS_BREADCRUMB_DIR")
        .unwrap_or_else(|_| "/tmp/tribunus-breadcrumbs/".to_string());
    let _ = fs::create_dir_all(&dir);
    let path = PathBuf::from(&dir).join(format!("breadcrumbs-{}.jsonl", pid));
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .expect("failed to open breadcrumb file");
    *guard = Some(file);
}

/// Write a breadcrumb **before** entering a native MLX/CoreML call, then
/// fsync the file so the entry survives a crash.
///
/// Parameters describe the operation about to execute: which transformer
/// layer, which projection, the backend, materialisation class, input and
/// weight shapes, and quantisation parameters.
pub fn before_native(
    pid: u32,
    layer: u32,
    phase: &str,
    projection: &str,
    backend: &str,
    materialization: &str,
    input_shape: &[i32],
    weight_shape: &[i32],
    bits: u8,
    group_size: u32,
) {
    ensure_initialized(pid);

    let ts = crate::now_iso8601();
    let line = json!({
        "ts": ts,
        "pid": pid,
        "layer": layer,
        "phase": phase,
        "projection": projection,
        "backend": backend,
        "materialization": materialization,
        "input_shape": input_shape,
        "weight_shape": weight_shape,
        "bits": bits,
        "group_size": group_size,
    });

    let mut guard = BREADCRUMB_WRITER.lock();
    match &mut *guard {
        Some(file) => {
            let _ = writeln!(file, "{}", line);
            let _ = file.sync_all(); // fsync — critical for crash survival
        }
        None => {}
    }
}

/// Write a breadcrumb **after** a native call completes successfully.
///
/// No fsync — a crash during the native call means this line is never
/// written, and the last fsync'd `before_native` entry pinpoints the
/// crash site.
pub fn after_native(_pid: u32, elapsed_us: u64) {
    let ts = crate::now_iso8601();
    let line = json!({
        "ts": ts,
        "ok": true,
        "elapsed_us": elapsed_us,
    });

    // The file is guaranteed to be initialised by the first `before_native`
    // call that preceded this one.  If there was none, skip silently.
    let mut guard = BREADCRUMB_WRITER.lock();
    match &mut *guard {
        Some(file) => {
            let _ = writeln!(file, "{}", line);
            // Deliberately *no* fsync here.
        }
        None => {}
    }
}
