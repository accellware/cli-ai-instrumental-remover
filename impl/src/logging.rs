//! Logging subsystem.
//!
//! Wraps `tracing-subscriber` and integrates with `indicatif::MultiProgress`
//! so log lines do not get clobbered by spinner / progress-bar redraws on the
//! same TTY.
//!
//! Resolution order for the log level (highest priority first):
//!
//! 1. `--log-level <level>` CLI flag (explicit override)
//! 2. `-v` / `-vv` / `-vvv` CLI count
//! 3. `RUST_LOG` environment variable
//! 4. Default: `WARN`
//!
//! All output goes to **stderr** to keep stdout clean for scripting.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use indicatif::MultiProgress;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

/// Global handle to the `MultiProgress` used for progress bars.
///
/// The tracing writer reads this so it can call
/// [`MultiProgress::suspend`] around each log emission, preventing log
/// lines from being overdrawn by spinner / bar redraws.
static PROGRESS: OnceLock<MultiProgress> = OnceLock::new();

/// Optional file sink for `--log-file`. Wrapped in a `Mutex` because
/// `tracing-subscriber`'s `fmt` layer expects a writer that is `Sync`-friendly
/// across threads.
static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();

/// Register the `MultiProgress` instance used by the pipeline.
///
/// Called once from `pipeline::run` before any progress bar is added. After
/// this call, every tracing event automatically pauses the bars while it
/// writes its line, then resumes the redraw — no flicker, no clobbering.
pub fn set_progress(mp: MultiProgress) {
    let _ = PROGRESS.set(mp);
}

/// Resolve the desired filter directives, in precedence order.
fn resolve_filter(
    explicit: Option<&str>,
    verbosity: u8,
    rust_log: Option<String>,
) -> EnvFilter {
    // 1. Explicit `--log-level` wins.
    if let Some(level) = explicit {
        return EnvFilter::try_new(level)
            .unwrap_or_else(|_| EnvFilter::new("warn"));
    }

    // 2. `-v` count.
    if verbosity > 0 {
        let level = match verbosity {
            1 => "info",
            2 => "debug",
            _ => "trace",
        };
        // Scope the verbose level to our own crate so we don't drown in
        // ort/ffmpeg internals; keep third-party at WARN.
        return EnvFilter::try_new(format!("warn,music_separator={level}"))
            .unwrap_or_else(|_| EnvFilter::new(level));
    }

    // 3. RUST_LOG env.
    if let Some(s) = rust_log {
        if !s.is_empty() {
            if let Ok(f) = EnvFilter::try_new(&s) {
                return f;
            }
        }
    }

    // 4. Default.
    EnvFilter::new("warn")
}

/// Initialize the global tracing subscriber.
///
/// This must be called *first thing* from `main` — before config load,
/// before any `tracing::*!` macro is invoked. Calling it twice is harmless
/// (the second `try_init` is a no-op).
///
/// `verbosity` is the count of `-v` flags. `explicit_level` is the value of
/// `--log-level` if provided. `log_file` optionally tees a copy of every
/// event to a file.
pub fn init(
    verbosity: u8,
    explicit_level: Option<&str>,
    log_file: Option<&Path>,
) -> Result<(), String> {
    let filter = resolve_filter(
        explicit_level,
        verbosity,
        std::env::var("RUST_LOG").ok(),
    );

    // Open the optional log file before building the subscriber, so we can
    // surface "could not open" errors back to the caller.
    if let Some(path) = log_file {
        let f = File::create(path)
            .map_err(|e| format!("could not open log file {}: {}", path.display(), e))?;
        let _ = LOG_FILE.set(Mutex::new(f));
    }

    // The format depends on whether we are at DEBUG/TRACE (developer view)
    // or INFO and below (clean human view).
    let dev_view = verbosity >= 2
        || matches!(explicit_level, Some("debug") | Some("trace"));

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(ProgressAwareWriter)
        .with_ansi(supports_color());

    let result = if dev_view {
        builder
            .with_target(true)
            .with_line_number(true)
            .with_file(true)
            .compact()
            .try_init()
    } else {
        builder
            .with_target(false)
            .without_time()
            .compact()
            .try_init()
    };

    result.map_err(|e| e.to_string())
}

/// Heuristic: ANSI is fine on a TTY and disabled when stderr is redirected.
fn supports_color() -> bool {
    use std::io::IsTerminal;
    io::stderr().is_terminal()
}

/// `MakeWriter` that wraps each emit in `MultiProgress::suspend` (when a
/// `MultiProgress` has been registered) and optionally tees to a log file.
#[derive(Default, Clone, Copy)]
struct ProgressAwareWriter;

impl<'a> MakeWriter<'a> for ProgressAwareWriter {
    type Writer = ProgressAwareWriterInner;

    fn make_writer(&'a self) -> Self::Writer {
        ProgressAwareWriterInner
    }
}

struct ProgressAwareWriterInner;

impl Write for ProgressAwareWriterInner {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Mirror to the optional log file (best effort — never fail the main
        // write because of file errors).
        if let Some(file) = LOG_FILE.get() {
            if let Ok(mut g) = file.lock() {
                let _ = g.write_all(buf);
            }
        }

        if let Some(mp) = PROGRESS.get() {
            let mut written = 0;
            mp.suspend(|| {
                let mut err = io::stderr().lock();
                written = err.write(buf).unwrap_or(0);
            });
            Ok(written)
        } else {
            io::stderr().lock().write(buf)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = LOG_FILE.get() {
            if let Ok(mut g) = file.lock() {
                let _ = g.flush();
            }
        }
        io::stderr().lock().flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_filter_explicit_wins_over_verbosity() {
        // `--log-level error` must trump `-vvv`.
        let f = resolve_filter(Some("error"), 3, Some("debug".to_string()));
        // EnvFilter doesn't expose its directives directly, but Display works.
        assert!(f.to_string().contains("error"));
    }

    #[test]
    fn resolve_filter_verbosity_wins_over_rust_log() {
        let f = resolve_filter(None, 1, Some("trace".to_string()));
        assert!(f.to_string().contains("info"));
    }

    #[test]
    fn resolve_filter_rust_log_used_when_no_flags() {
        let f = resolve_filter(None, 0, Some("music_separator=trace".to_string()));
        assert!(f.to_string().contains("trace"));
    }

    #[test]
    fn resolve_filter_default_is_warn() {
        let f = resolve_filter(None, 0, None);
        assert!(f.to_string().contains("warn"));
    }

    #[test]
    fn resolve_filter_empty_rust_log_falls_through_to_default() {
        let f = resolve_filter(None, 0, Some(String::new()));
        assert!(f.to_string().contains("warn"));
    }
}
