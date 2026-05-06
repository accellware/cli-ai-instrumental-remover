mod config;
mod error;
mod ffmpeg;
mod inference;
mod logging;
mod model_data;
mod pipeline;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "music-separator", about = "Remove background music from video files")]
struct Args {
    /// Path to the input video file
    #[arg(long)]
    input: PathBuf,

    /// Increase log verbosity. Repeat for more detail:
    /// `-v` = info, `-vv` = debug, `-vvv` = trace.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Explicit log level. Wins over `-v` and `RUST_LOG`.
    /// One of: off, error, warn, info, debug, trace.
    #[arg(long, value_name = "LEVEL")]
    log_level: Option<String>,

    /// Optional path to also write log lines to (in addition to stderr).
    /// File is overwritten on each run.
    #[arg(long, value_name = "PATH")]
    log_file: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    // Initialize tracing first, before anything else can emit logs.
    if let Err(e) = logging::init(
        args.verbose,
        args.log_level.as_deref(),
        args.log_file.as_deref(),
    ) {
        eprintln!("Error: failed to initialize logging: {e}");
        std::process::exit(1);
    }

    tracing::debug!(?args, "parsed CLI arguments");
    tracing::info!(
        verbose = args.verbose,
        log_level = ?args.log_level,
        log_file = ?args.log_file,
        "music-separator starting"
    );

    let config = match config::load() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, "failed to load config");
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = pipeline::run(&args.input, &config) {
        tracing::error!(error = %e, "pipeline failed");
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
