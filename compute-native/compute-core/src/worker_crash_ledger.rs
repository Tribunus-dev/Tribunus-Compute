//! Crash ledger that records worker-process termination events to an
//! append-only JSONL file.
//!
//! Each crash event records the worker PID, exit code, optional signal,
//! active request id, and the last crash breadcrumb written before the
//! native call that may have caused the crash. The file lives at
//! `$TRIBUNUS_CRASH_DIR/crashes.jsonl` (default `~/.tribunus/crashes/`).

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::LazyLock;

use parking_lot::Mutex;

use crate::now_iso8601;

// ── Path helpers ────────────────────────────────────────────────────────────

/// Directory for the crash ledger file.
fn crash_dir() -> PathBuf {
    std::env::var("TRIBUNUS_CRASH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".tribunus").join("crashes")
        })
}

/// Full path to the JSONL crash file.
fn crash_file_path() -> PathBuf {
    crash_dir().join("crashes.jsonl")
}

// ── Global writer handle ────────────────────────────────────────────────────

/// Lazily initialised writer handle. The `File` is opened on the first
/// [`WorkerCrashLedger::record`] call so we never touch the filesystem
/// until there is actually a crash to persist.
static CRASH_LEDGER: LazyLock<Mutex<Option<File>>> = LazyLock::new(|| Mutex::new(None));

// ── Public API ──────────────────────────────────────────────────────────────

/// Append-only crash record ledger.
///
/// All methods are thread-safe. The writer is opened lazily and kept
/// open across calls so that each `record` only needs a fast lock + write.
pub struct WorkerCrashLedger;

impl WorkerCrashLedger {
    /// Record one worker crash event.
    ///
    /// Creates the crash directory and ledger file on the first call.
    /// The JSON line is written then `fsync`'d so that a crash immediately
    /// after this returns will not lose the record.
    pub fn record(
        pid: u32,
        exit_code: i32,
        signal: Option<i32>,
        active_request_id: Option<String>,
        last_breadcrumb: Option<String>,
    ) {
        let mut guard = CRASH_LEDGER.lock();

        // Lazily create the directory and open the file on first write.
        if guard.is_none() {
            let dir = crash_dir();
            if let Err(e) = fs::create_dir_all(&dir) {
                log_error!("Failed to create crash ledger directory {:?}: {}", dir, e);
                return;
            }
            let path = crash_file_path();
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(file) => {
                    *guard = Some(file);
                }
                Err(e) => {
                    log_error!("Failed to open crash ledger {:?}: {}", path, e);
                    return;
                }
            }
        }

        let record = serde_json::json!({
            "timestamp": now_iso8601(),
            "worker_pid": pid,
            "exit_code": exit_code,
            "signal": signal,
            "active_request_id": active_request_id,
            "last_breadcrumb": last_breadcrumb,
        });

        if let Some(file) = &mut *guard {
            let line = serde_json::to_string(&record).unwrap_or_default();
            let _ = writeln!(file, "{line}");
            let _ = file.flush();
            let _ = file.sync_all();
        }
    }

    /// Return the most recent unique crash signatures, deduplicated by
    /// `(exit_code, signal)`.
    ///
    /// Iterates the ledger file from the end, collecting entries whose
    /// `(exit_code, signal)` pair has not been seen yet. Stops when
    /// `limit` entries have been collected or the file is exhausted.
    /// Returns an empty `Vec` when the ledger file does not exist or
    /// cannot be read.
    pub fn recent_signatures(limit: usize) -> Vec<HashMap<String, serde_json::Value>> {
        let path = crash_file_path();
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };

        let mut entries: Vec<HashMap<String, serde_json::Value>> = Vec::new();
        for line in BufReader::new(file).lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(value) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&line) {
                entries.push(value);
            }
        }

        // Iterate in reverse, deduplicating by (exit_code, signal).
        let mut seen: HashSet<(i64, Option<i64>)> = HashSet::new();
        let mut result: Vec<HashMap<String, serde_json::Value>> = Vec::new();
        for entry in entries.into_iter().rev() {
            let key = (
                entry.get("exit_code").and_then(|v| v.as_i64()).unwrap_or(0),
                match entry.get("signal") {
                    Some(v) if v.is_null() => None,
                    Some(v) => v.as_i64(),
                    None => None,
                },
            );
            if seen.insert(key) {
                result.push(entry);
                if result.len() >= limit {
                    break;
                }
            }
        }

        result
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// Helper: return a temporary path for the crash ledger.
    fn tmp_crash_dir() -> PathBuf {
        let dir = std::env::temp_dir()
            .join("tribunus_test_crash_ledger")
            .join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_record_appends_jsonl() {
        let dir = tmp_crash_dir();
        // Override the env var for the test scope.
        std::env::set_var("TRIBUNUS_CRASH_DIR", dir.to_str().unwrap());

        WorkerCrashLedger::record(
            1001,
            -6,
            Some(6),
            Some("req-1".into()),
            Some("attention:layer=12:QProj".into()),
        );

        let path = crash_file_path();
        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();

        let parsed: HashMap<String, serde_json::Value> =
            serde_json::from_str(contents.trim()).unwrap();
        assert_eq!(
            parsed.get("worker_pid").and_then(|v| v.as_u64()),
            Some(1001)
        );
        assert_eq!(parsed.get("exit_code").and_then(|v| v.as_i64()), Some(-6));
        assert_eq!(parsed.get("signal").and_then(|v| v.as_i64()), Some(6));
        assert_eq!(
            parsed.get("active_request_id").and_then(|v| v.as_str()),
            Some("req-1")
        );
        assert_eq!(
            parsed.get("last_breadcrumb").and_then(|v| v.as_str()),
            Some("attention:layer=12:QProj")
        );
        assert!(parsed.get("timestamp").and_then(|v| v.as_str()).is_some());
    }

    #[test]
    fn test_record_multiple_lines() {
        let dir = tmp_crash_dir();
        std::env::set_var("TRIBUNUS_CRASH_DIR", dir.to_str().unwrap());

        WorkerCrashLedger::record(1, 0, None, None, None);
        WorkerCrashLedger::record(2, -11, Some(11), Some("req-2".into()), None);

        let path = crash_file_path();
        let lines: Vec<String> = BufReader::new(File::open(&path).unwrap())
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .collect();

        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_recent_signatures_dedup() {
        let dir = tmp_crash_dir();
        std::env::set_var("TRIBUNUS_CRASH_DIR", dir.to_str().unwrap());

        // Write entries directly so we control order precisely.
        let path = crash_file_path();
        fs::create_dir_all(&dir).unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();

        fn write_entry(file: &mut File, ec: i32, sig: Option<i32>, rid: &str) {
            let record = serde_json::json!({
                "timestamp": now_iso8601(),
                "worker_pid": 42,
                "exit_code": ec,
                "signal": sig,
                "active_request_id": rid,
                "last_breadcrumb": null,
            });
            let line = serde_json::to_string(&record).unwrap();
            writeln!(file, "{line}").unwrap();
        }

        // Three distinct (exit_code, signal) combos, one duplicate.
        write_entry(&mut file, 1, None, "a");
        write_entry(&mut file, 1, None, "b"); // dup of (1, None)
        write_entry(&mut file, 2, Some(11), "c");
        write_entry(&mut file, 3, Some(6), "d");
        file.sync_all().unwrap();
        drop(file);

        // Reset the lazy writer so `recent_signatures` reads from the file.
        *CRASH_LEDGER.lock() = None;

        let sigs = WorkerCrashLedger::recent_signatures(10);
        // We should see 3 unique signatures, in reverse file order.
        // The duplicate (1, None) at entry "b" is deduped — the first
        // encountered when iterating backwards is "b", then "a" is skipped.
        // The three unique ones come from entries d, c, (b or a).
        assert_eq!(
            sigs.len(),
            3,
            "expected 3 unique signatures, got {}",
            sigs.len()
        );
    }

    #[test]
    fn test_recent_signatures_limit() {
        let dir = tmp_crash_dir();
        std::env::set_var("TRIBUNUS_CRASH_DIR", dir.to_str().unwrap());

        let path = crash_file_path();
        fs::create_dir_all(&dir).unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();

        for i in 0..10 {
            let record = serde_json::json!({
                "timestamp": now_iso8601(),
                "worker_pid": 42,
                "exit_code": i,
                "signal": null,
                "active_request_id": format!("req-{i}"),
                "last_breadcrumb": null,
            });
            let line = serde_json::to_string(&record).unwrap();
            writeln!(file, "{line}").unwrap();
        }
        file.sync_all().unwrap();
        drop(file);

        *CRASH_LEDGER.lock() = None;

        let sigs = WorkerCrashLedger::recent_signatures(3);
        assert_eq!(sigs.len(), 3);
        // Most recent entries, reversed, so exit codes 9, 8, 7.
        assert_eq!(sigs[0].get("exit_code").and_then(|v| v.as_i64()), Some(9));
        assert_eq!(sigs[2].get("exit_code").and_then(|v| v.as_i64()), Some(7));
    }

    #[test]
    fn test_recent_signatures_empty_file() {
        let dir = tmp_crash_dir();
        std::env::set_var("TRIBUNUS_CRASH_DIR", dir.to_str().unwrap());
        fs::create_dir_all(&dir).unwrap();

        let sigs = WorkerCrashLedger::recent_signatures(10);
        assert!(sigs.is_empty());
    }

    #[test]
    fn test_no_file_returns_empty() {
        // Use a nonexistent dir so `recent_signatures` finds nothing.
        std::env::set_var("TRIBUNUS_CRASH_DIR", "/tmp/tribunus_test_nonexistent_XXXX");
        let sigs = WorkerCrashLedger::recent_signatures(10);
        assert!(sigs.is_empty());
    }
}
