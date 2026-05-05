mod config;
mod error;
mod ffmpeg;
mod inference;
mod model_data;
mod pipeline;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "music-separator", about = "Remove background music from video files")]
struct Args {
    /// Path to the input video file
    #[arg(long)]
    input: PathBuf,
}

fn main() {
    let args = Args::parse();

    let config = match config::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = pipeline::run(&args.input, &config) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
