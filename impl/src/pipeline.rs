use std::path::Path;
use std::fs;
use uuid::Uuid;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use crate::config::Config;
use crate::error::AppError;
use crate::{model_data, ffmpeg};
use crate::inference::Separator;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_spinner(mp: &MultiProgress, msg: &'static str) -> ProgressBar {
    let pb = mp.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} {msg}")
            .unwrap(),
    );
    pb.set_message(msg);
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

/// Attempt to delete a temp file, printing a warning on failure.
fn try_remove(path: &Path) {
    if let Err(e) = fs::remove_file(path) {
        eprintln!(
            "Warning: could not delete temp file {}: {}",
            path.display(),
            e
        );
    }
}

/// Delete both temp files (best-effort).
fn cleanup_temps(extracted: &Path, vocals: &Path) {
    try_remove(extracted);
    try_remove(vocals);
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

pub fn run(input_path: &Path, config: &Config) -> Result<(), AppError> {
    let mp = MultiProgress::new();

    // ── Step 1: Validate ───────────────────────────────────────────────────────
    let pb1 = make_spinner(&mp, "[1/5] Validating inputs...");

    if !config.model_path.exists() {
        pb1.abandon();
        return Err(AppError::ModelNotFound(config.model_path.clone()));
    }
    if !input_path.exists() {
        pb1.abandon();
        return Err(AppError::InputVideoNotFound(input_path.to_path_buf()));
    }

    let model_data_path = config
        .model_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("model_data.json");

    let model_params_list = model_data::load(&model_data_path)
        .inspect_err(|_| pb1.abandon())?;

    let model_filename = config
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| { pb1.abandon(); AppError::ModelNotFound(config.model_path.clone()) })?;

    let model_params = model_data::find_by_name(&model_params_list, model_filename)
        .inspect_err(|_| pb1.abandon())?
        .clone();

    fs::create_dir_all(&config.output_dir)
        .map_err(|e| AppError::OutputDirCreate(e.to_string()))
        .inspect_err(|_| pb1.abandon())?;

    pb1.finish_with_message("[1/5] Validating inputs... ✓");

    // ── Step 2: Probe audio ────────────────────────────────────────────────────
    let pb2 = make_spinner(&mp, "[2/5] Probing audio...");

    let audio_info = ffmpeg::probe_audio(input_path)?;

    if audio_info.is_none() {
        let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = input_path.extension().unwrap_or_default().to_string_lossy();
        let out_name = if ext.is_empty() {
            format!("{}_no_music", stem)
        } else {
            format!("{}_no_music.{}", stem, ext)
        };
        let output_path = config.output_dir.join(&out_name);
        fs::copy(input_path, &output_path)
            .map_err(|e| AppError::FileCopy(e.to_string()))?;
        eprintln!("No audio track found; copying file to output.");
        pb2.finish_with_message("[2/5] Probing audio... ✓");
        return Ok(());
    }

    pb2.finish_with_message("[2/5] Probing audio... ✓");

    // ── Step 3: Extract audio ──────────────────────────────────────────────────
    let pb3 = make_spinner(&mp, "[3/5] Extracting audio...");

    let temp_dir = std::env::temp_dir();
    let extracted_wav = temp_dir.join(format!("{}_extracted.wav", Uuid::new_v4()));
    let vocals_wav = temp_dir.join(format!("{}_vocals.wav", Uuid::new_v4()));

    if let Err(e) = ffmpeg::extract_audio(input_path, &extracted_wav) {
        cleanup_temps(&extracted_wav, &vocals_wav);
        return Err(e);
    }

    pb3.finish_with_message("[3/5] Extracting audio... ✓");

    // ── Step 4: Run inference ──────────────────────────────────────────────────
    let pb4 = mp.add(ProgressBar::new(100));
    pb4.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40} {pos}%")
            .unwrap(),
    );
    pb4.set_message("[4/5] Running inference...");

    let mut separator = match Separator::new(config, model_params) {
        Ok(s) => s,
        Err(e) => {
            cleanup_temps(&extracted_wav, &vocals_wav);
            return Err(e);
        }
    };

    if let Err(e) = separator.separate_vocals(&extracted_wav, &vocals_wav, |p| {
        pb4.set_position((p * 100.0) as u64);
    }) {
        cleanup_temps(&extracted_wav, &vocals_wav);
        return Err(e);
    }

    pb4.set_position(100);
    pb4.finish_with_message("[4/5] Running inference...");

    // ── Step 5: Remux ──────────────────────────────────────────────────────────
    let pb5 = make_spinner(&mp, "[5/5] Remuxing video...");

    let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = input_path.extension().unwrap_or_default().to_string_lossy();
    let out_name = if ext.is_empty() {
        format!("{}_no_music", stem)
    } else {
        format!("{}_no_music.{}", stem, ext)
    };
    let output_path = config.output_dir.join(&out_name);

    if let Err(e) = ffmpeg::remux_with_audio(input_path, &vocals_wav, &output_path) {
        cleanup_temps(&extracted_wav, &vocals_wav);
        return Err(e);
    }

    pb5.finish_with_message("[5/5] Remuxing video... ✓");

    // ── Step 6: Cleanup ────────────────────────────────────────────────────────
    try_remove(&extracted_wav);
    try_remove(&vocals_wav);

    eprintln!("Done → {}", output_path.display());

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ExecutionProvider};

    #[test]
    fn run_returns_error_for_missing_input() {
        // model_path points to the test binary (which exists at runtime) so the
        // model-exists check passes and we reach the input-exists check.
        let exe = std::env::current_exe().unwrap();
        let config = Config {
            model_path: exe.clone(),
            output_dir: std::env::temp_dir().join("ms_test_output"),
            execution_provider: ExecutionProvider::Cpu,
            chunk_size: 261120,
        };
        let result = run(Path::new("/nonexistent/video.mp4"), &config);
        assert!(
            matches!(result, Err(AppError::InputVideoNotFound(_))),
            "expected InputVideoNotFound, got: {result:?}"
        );
    }
}
