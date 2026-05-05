# Implementation Blueprint & Prompt Plan: `music-separator`

## Blueprint

### Phase Analysis

Dependency graph driving prompt ordering:

```
error.rs
  └─► config.rs
  └─► model_data.rs
  └─► ffmpeg.rs (probe → extract → remux)
  └─► inference.rs (stft/istft → resample → onnx session → chunk loop)
        └─ depends on: config.rs, model_data.rs
  └─► pipeline.rs
        └─ depends on: config.rs, model_data.rs, ffmpeg.rs, inference.rs
  └─► main.rs
        └─ depends on: pipeline.rs, config.rs
```

Each prompt below produces tested, integrated code. Nothing is left orphaned.

### Build Order Summary

| Prompt | Module(s)                     | Depends On                      | Testable After |
| ------ | ----------------------------- | ------------------------------- | -------------- |
| 1      | `error.rs` + stubs            | nothing                         | immediately    |
| 2      | `config.rs`                   | `error.rs`                      | immediately    |
| 3      | `model_data.rs`               | `error.rs`                      | immediately    |
| 4      | `ffmpeg.rs` (probe + extract) | `error.rs`                      | compile-only   |
| 5      | `ffmpeg.rs` (remux)           | prompt 4                        | compile-only   |
| 6      | `inference.rs` DSP primitives | `error.rs`                      | immediately    |
| 7      | `inference.rs` preprocessing  | prompt 6                        | immediately    |
| 8      | `inference.rs` ONNX session   | prompts 6+7, config, model_data | partial        |
| 9      | `pipeline.rs`                 | all modules above               | partial        |
| 10     | `main.rs` + integration tests | `pipeline.rs`, `config.rs`      | full binary    |

---

## Prompts

---

### Prompt 1 — Project Scaffolding & Error Types

**Goal:** Establish the Cargo project, all dependencies, module stubs, and the central error type. Every subsequent prompt compiles against this foundation.

```text
Create a new Rust binary crate named `music-separator` with the following Cargo.toml dependencies:

[package]
name = "music-separator"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
ort = { version = "2", features = ["cuda"] }
ffmpeg-next = "7"
rustfft = "6"
hound = "3"
indicatif = "0.17"
uuid = { version = "1", features = ["v4"] }
thiserror = "1"
anyhow = "1"

[dev-dependencies]
tempfile = "3"

Create the following empty module files (just `// TODO` stubs with `pub mod` declarations):
- src/main.rs
- src/config.rs
- src/model_data.rs
- src/ffmpeg.rs
- src/inference.rs
- src/pipeline.rs
- src/error.rs

In `src/main.rs`, declare all modules:

mod config;
mod error;
mod ffmpeg;
mod inference;
mod model_data;
mod pipeline;

Implement `src/error.rs` with a single unified `AppError` enum using `thiserror`. It must cover these variants:

- `ConfigNotFound` — config.json not found in CWD
- `ConfigParse(String)` — JSON parse error with field context
- `ModelNotFound(PathBuf)` — .onnx file path not found
- `ModelDataNotFound(PathBuf)` — model_data.json not found
- `ModelDataParse(String)` — model_data.json parse error
- `ModelNotInRegistry(String)` — model filename not found in model_data.json
- `InputVideoNotFound(PathBuf)` — input video file not found
- `OutputDirCreate(String)` — failed to create output_dir
- `FfmpegProbe(String)` — FFmpeg probe failure
- `FfmpegExtract(String)` — FFmpeg audio extraction failure
- `FfmpegRemux(String)` — FFmpeg remux failure
- `OnnxLoad(String)` — ONNX model load failure
- `OnnxInference(String)` — ONNX inference failure
- `AudioRead(String)` — WAV read error
- `AudioWrite(String)` — WAV write error
- `FileCopy(String)` — file copy failure

Each variant must implement `Display` via `thiserror`'s `#[error("...")]` attribute with a human-readable message.

Write a unit test in `error.rs` that verifies each variant formats to a non-empty string via `to_string()`.

The project must compile cleanly with `cargo build` (stubs are fine, no logic yet).
```

---

### Prompt 2 — Config Module

**Goal:** Load, parse, and validate `config.json`. First real business logic.

```text
Implement `src/config.rs` for the `music-separator` project.

The `Config` struct must deserialize from JSON and contain these fields:
- `model_path: PathBuf`
- `output_dir: PathBuf`
- `execution_provider: ExecutionProvider` (an enum: `Cpu`, `Cuda`, serde rename to lowercase strings "cpu" / "cuda")
- `chunk_size: usize`

Implement a function:

pub fn load() -> Result<Config, AppError>

It must:
1. Look for `config.json` in `std::env::current_dir()`. Return `AppError::ConfigNotFound` if absent.
2. Read and deserialize the file. Return `AppError::ConfigParse` with the serde error message if malformed.
3. Validate that `chunk_size > 0`. Return `AppError::ConfigParse` with message "chunk_size must be greater than 0" if not.
4. Validate that `execution_provider` is one of the two known variants (serde handles this automatically via the enum, but add a clear error if serde fails).
5. Return the parsed `Config` on success.

Do NOT validate model_path or output_dir existence here — that is done in the pipeline.

Write unit tests in a `#[cfg(test)]` block using `tempfile::tempdir()` to write temporary `config.json` files and assert:
- Valid config deserializes correctly
- Missing file returns `AppError::ConfigNotFound`
- Malformed JSON returns `AppError::ConfigParse`
- `chunk_size: 0` returns `AppError::ConfigParse`
- Unknown `execution_provider` value returns `AppError::ConfigParse`
- "cuda" deserializes to `ExecutionProvider::Cuda`

All tests must pass with `cargo test`.
```

---

### Prompt 3 — Model Data Module

**Goal:** Parse `model_data.json` and look up inference parameters by model filename.

```text
Implement `src/model_data.rs` for the `music-separator` project.

`model_data.json` is structured as a map from MD5 hash strings to model parameter objects. Example:

{
  "77d07b2667ddf05b9e3175941b4454a0": {
    "compensate": 1.021,
    "mdx_dim_f_set": 3072,
    "mdx_dim_t_set": 8,
    "mdx_n_fft_scale_set": 7680,
    "primary_stem": "Vocals",
    "name": "UVR-MDX-NET-Voc_FT.onnx"
  }
}

Define:

pub struct ModelParams {
    pub compensate: f32,
    pub mdx_dim_f_set: usize,
    pub mdx_dim_t_set: usize,
    pub mdx_n_fft_scale_set: usize,
    pub primary_stem: String,
    pub name: String,
}

Implement:

pub fn load(model_data_path: &Path) -> Result<Vec<ModelParams>, AppError>
pub fn find_by_name(params: &[ModelParams], model_filename: &str) -> Result<&ModelParams, AppError>

`load` must:
1. Return `AppError::ModelDataNotFound` if the file does not exist.
2. Return `AppError::ModelDataParse` if JSON parsing fails.
3. Return the values of the map as a `Vec<ModelParams>` (ignore the hash keys).

`find_by_name` must:
1. Search the slice for an entry whose `name` field matches `model_filename` (filename only, not full
   path — use `Path::file_name()` on the model_path from config).
2. Return `AppError::ModelNotInRegistry(model_filename.to_string())` if not found.

Write unit tests:
- `load` on a valid inline JSON string written to a tempfile returns the correct Vec
- `load` on a missing file returns `ModelDataNotFound`
- `load` on malformed JSON returns `ModelDataParse`
- `find_by_name` finds "UVR-MDX-NET-Voc_FT.onnx" correctly
- `find_by_name` returns `ModelNotInRegistry` for an unknown name

All tests must pass with `cargo test`.
```

---

### Prompt 4 — FFmpeg: Audio Probe & Extraction

**Goal:** Implement the two FFmpeg functions needed before inference — detecting whether audio exists and demuxing it to WAV.

```text
Implement the probe and extraction functions in `src/ffmpeg.rs` for the `music-separator` project
using the `ffmpeg-next` crate.

Implement:

pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn probe_audio(video_path: &Path) -> Result<Option<AudioInfo>, AppError>
pub fn extract_audio(video_path: &Path, output_wav: &Path) -> Result<AudioInfo, AppError>

`probe_audio`:
1. Call `ffmpeg_next::format::input(video_path)` — wrap errors as `AppError::FfmpegProbe`.
2. Iterate streams looking for one with `medium() == ffmpeg_next::media::Type::Audio`.
3. If found, decode the codec parameters to get sample rate and channel count.
   Return `Some(AudioInfo { ... })`.
4. If no audio stream found, return `None`.

`extract_audio`:
1. Open the input video with `ffmpeg_next::format::input`.
2. Find the best audio stream.
3. Open the output WAV file with `ffmpeg_next::format::output` using the `wav` muxer.
4. Add an audio stream to the output copying the codec parameters.
5. Demux packets from the audio stream and write them to the WAV output, flushing at the end.
6. Return the `AudioInfo` (sample_rate, channels) from the source stream.
7. Wrap all ffmpeg-next errors into `AppError::FfmpegExtract`.

Call `ffmpeg_next::init()` at the top of each public function (it is idempotent).

For unit tests, note that integration-level FFmpeg tests require real media files.
Write a compile-time test only:

#[test]
fn probe_audio_missing_file_returns_error() {
    let result = probe_audio(Path::new("/nonexistent/file.mp4"));
    assert!(result.is_err());
}

All code must compile with `cargo build`.
```

---

### Prompt 5 — FFmpeg: Video Remux

**Goal:** Complete the `ffmpeg.rs` module by adding the remux function that swaps the audio stream.

```text
Add the remux function to `src/ffmpeg.rs` in the `music-separator` project.

Implement:

pub fn remux_with_audio(
    video_path: &Path,
    vocals_wav: &Path,
    output_path: &Path,
) -> Result<(), AppError>

This function must:
1. Open the original video with `ffmpeg_next::format::input(video_path)` — wrap errors as
   `AppError::FfmpegRemux`.
2. Open the vocals WAV with `ffmpeg_next::format::input(vocals_wav)`.
3. Open the output file with `ffmpeg_next::format::output(output_path)`.
4. Copy all video streams from the original input to the output with `-c:v copy` semantics
   (add output stream with codec parameters copied from input stream).
5. Copy the audio stream from `vocals_wav` into the output with `-c:a copy` semantics.
6. Write the output header.
7. Interleave and write all packets from both sources (video packets from original, audio packets
   from vocals_wav), adjusting PTS/DTS to the output stream's time base using `packet.rescale_ts`.
8. Write the output trailer.
9. Wrap all errors as `AppError::FfmpegRemux`.

Add a compile-only test:

#[test]
fn remux_missing_input_returns_error() {
    let result = remux_with_audio(
        Path::new("/nonexistent/video.mp4"),
        Path::new("/nonexistent/audio.wav"),
        Path::new("/tmp/out.mp4"),
    );
    assert!(result.is_err());
}

All existing tests must still pass. `cargo build` must succeed.
```

---

### Prompt 6 — Audio DSP: STFT, iSTFT, and Hann Window

**Goal:** Implement the signal processing primitives that MDX-Net inference depends on. Pure math — no ONNX yet.

```text
Implement the DSP primitives in `src/inference.rs` for the `music-separator` project using the
`rustfft` crate.

Implement these public functions (not yet connected to ONNX):

pub fn hann_window(size: usize) -> Vec<f32>

pub fn stft(
    signal: &[f32],
    fft_size: usize,
    hop_length: usize,
    window: &[f32],
) -> Vec<Vec<[f32; 2]>>   // [frame][bin] as [real, imag]

pub fn istft(
    frames: &[Vec<[f32; 2]>],
    fft_size: usize,
    hop_length: usize,
    window: &[f32],
    signal_length: usize,
) -> Vec<f32>

Rules:
- `hann_window(n)`: return `0.5 * (1.0 - cos(2π * i / n))` for i in 0..n
- `stft`: pad signal with `fft_size/2` zeros on each side (center padding). For each hop-aligned
  frame, apply the window element-wise, run a forward FFT via `rustfft`, return only the first
  `fft_size/2 + 1` bins (one-sided spectrum).
- `istft`: inverse of stft. For each frame, run inverse FFT on the one-sided spectrum (mirror
  conjugate to reconstruct full spectrum), apply window, overlap-add into output buffer, normalize
  by the sum-of-squared windows (OLA normalization). Trim output to `signal_length`.

Write unit tests:
- `hann_window(4)` returns values close to `[0.0, 0.75, 0.75, 0.0]` within `1e-5`
- STFT round-trip: generate a 44100-sample 440 Hz sine wave, run `stft` then `istft`, verify the
  reconstructed signal matches the original with max absolute error < `0.01`
- STFT of a zero signal returns all-zero frames
- `istft` output length equals `signal_length`

All tests must pass with `cargo test`.
```

---

### Prompt 7 — Audio DSP: Resampling & Preprocessing

**Goal:** Implement sample rate conversion and the mono/normalization helpers that prepare audio for inference and restore it afterward.

```text
Add audio preprocessing and postprocessing functions to `src/inference.rs` in the
`music-separator` project.

Implement:

pub fn resample(signal: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32>
pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32>
pub fn interleave_stereo(left: &[f32], right: &[f32]) -> Vec<f32>
pub fn normalize(signal: &[f32]) -> (Vec<f32>, f32)  // returns (normalized, scale_factor)
pub fn denormalize(signal: &[f32], scale: f32) -> Vec<f32>

Rules:
- `resample`: implement linear interpolation resampling. Compute the ratio
  `from_rate as f64 / to_rate as f64`. For each output sample index, compute the corresponding
  fractional input index and linearly interpolate between the two surrounding input samples.
- `to_mono`: if `channels == 1`, return a clone. If `channels == 2`, average pairs of interleaved
  samples `(L + R) / 2`. For N channels, average every N consecutive samples.
- `interleave_stereo`: zip left and right into `[L0, R0, L1, R1, ...]`.
- `normalize`: divide by the max absolute value of the signal. Return the normalized signal and the
  scale factor used (the max abs value). If max is 0.0, return the signal unchanged and scale = 1.0.
- `denormalize`: multiply each sample by `scale`.

Write unit tests:
- `to_mono` on `[1.0, 0.0, 0.0, 1.0]` with `channels=2` returns `[0.5, 0.5]`
- `resample` from 44100 to 22050 on a 44100-sample signal returns a 22050-sample signal
- `resample` from 22050 to 44100 on a 22050-sample signal returns a 44100-sample signal
- `normalize` on `[2.0, -4.0, 1.0]` returns `([0.5, -1.0, 0.25], 4.0)`
- `denormalize` reverses `normalize` within `1e-6`
- `interleave_stereo` on `[1.0, 2.0]` and `[3.0, 4.0]` returns `[1.0, 3.0, 2.0, 4.0]`

All existing and new tests must pass with `cargo test`.
```

---

### Prompt 8 — ONNX Inference: Model Session & Chunk Loop

**Goal:** Load the ONNX model and implement the full chunk-based vocals separation pipeline, wiring together all DSP from the previous two prompts.

```text
Implement the ONNX inference pipeline in `src/inference.rs` for the `music-separator` project
using the `ort` crate (v2).

Add:

use crate::config::{Config, ExecutionProvider};
use crate::model_data::ModelParams;
use crate::error::AppError;

pub struct Separator {
    session: ort::Session,
    params: ModelParams,
    chunk_size: usize,
}

impl Separator {
    pub fn new(config: &Config, params: ModelParams) -> Result<Self, AppError>
    pub fn separate_vocals(
        &self,
        wav_path: &Path,
        output_path: &Path,
        progress_cb: impl Fn(f32),   // called with 0.0..=1.0 progress fraction
    ) -> Result<(), AppError>
}

`Separator::new`:
1. Build an `ort::SessionBuilder`. If `config.execution_provider == ExecutionProvider::Cuda`,
   add the CUDA execution provider. Otherwise use CPU.
2. Load the model from `config.model_path`. Wrap errors as `AppError::OnnxLoad`.
3. Store session, params, chunk_size from config.

`separate_vocals`:
1. Read the WAV file with `hound::WavReader`. Wrap errors as `AppError::AudioRead`. Store original
   sample rate and channel count.
2. Collect all samples as `Vec<f32>` (convert i16/i32 samples by dividing by their max value to
   reach float32 range).
3. Call `to_mono(samples, channels)`.
4. If sample_rate != 44100, call `resample(mono, sample_rate, 44100)`. Record whether resampling
   happened.
5. Call `normalize(resampled)` to get `(normalized, scale)`.
6. Compute hop_length = `params.mdx_n_fft_scale_set / 4`.
7. Compute window = `hann_window(params.mdx_n_fft_scale_set)`.
8. Split the normalized signal into overlapping chunks of `self.chunk_size` samples with 50% overlap
   (hop = chunk_size / 2). Pad the last chunk with zeros if needed.
9. For each chunk:
   a. Run `stft(chunk, mdx_n_fft_scale_set, hop_length, &window)` → complex spectrogram frames.
   b. Reshape into a float32 tensor of shape `[1, 4, mdx_dim_f_set, mdx_dim_t_set]` (real and imag
      interleaved as the 4 channels: [real_ch0, imag_ch0, real_ch1, imag_ch1] — MDX-Net expects 4
      channels).
   c. Run `session.run(inputs![tensor])` → wrap errors as `AppError::OnnxInference`.
   d. Extract the output tensor, reshape back to complex spectrogram frames.
   e. Run `istft(output_frames, mdx_n_fft_scale_set, hop_length, &window, chunk.len())`.
   f. Call `progress_cb(chunk_index as f32 / total_chunks as f32)`.
10. Overlap-add all output chunks back into a full-length signal.
11. Apply `compensate` multiplier: `output *= params.compensate`.
12. Call `denormalize(output, scale)`.
13. If resampling happened in step 4, resample back from 44100 to original sample_rate.
14. If original was stereo, call `interleave_stereo(&output, &output)` to duplicate mono to stereo.
15. Write the output to `output_path` using `hound::WavWriter` with the original sample_rate and
    channel count. Wrap errors as `AppError::AudioWrite`.

Write one unit test:

#[test]
fn separator_new_fails_on_missing_model() {
    // Build a config pointing to a nonexistent model
    // Assert Separator::new returns AppError::OnnxLoad
}

All existing tests must pass. `cargo build` must succeed.
```

---

### Prompt 9 — Pipeline Orchestrator & Progress Reporting

**Goal:** Wire all modules into the 6-step pipeline with `indicatif` spinners and a progress bar.

```text
Implement `src/pipeline.rs` for the `music-separator` project.

Implement:

pub fn run(input_path: &Path, config: &Config) -> Result<(), AppError>

This function orchestrates the full pipeline in order. Use `indicatif::MultiProgress` and
`indicatif::ProgressBar` for all user-facing output. All progress output must go to stderr.

Stage display format:

[1/5] Validating inputs...         ✓
[2/5] Extracting audio...          ✓
[3/5] Running inference...         ██████████░░░░░ 67%
[4/5] Remuxing video...            ✓
[5/5] Cleaning up...               ✓

Done → output/interview_no_music.mp4

Stages 1, 2, 4, and 5 use a spinner style (`ProgressBar::new_spinner`) that finish with
`finish_with_message("✓")`.
Stage 3 uses `ProgressBar::new(100)` with a bar style and percentage, updated via the `progress_cb`
closure passed to `Separator::separate_vocals`.

Step-by-step logic:

Step 1 — Validate:
- Verify `config.model_path` exists → `AppError::ModelNotFound`
- Verify `input_path` exists → `AppError::InputVideoNotFound`
- Load `model_data.json` from the same directory as `config.model_path`
- Call `model_data::find_by_name` for the model filename
- Create `config.output_dir` with `fs::create_dir_all` → `AppError::OutputDirCreate`

Step 2 — Probe audio:
- Call `ffmpeg::probe_audio(input_path)`
- If `None`: compute output path as `{output_dir}/{stem}_no_music.{ext}`, copy the input file
  with `fs::copy`, print a notice message to stderr, return `Ok(())`

Step 3 — Extract audio:
- Generate two UUIDs: one for extracted WAV (`{uuid}_extracted.wav`), one for vocals WAV
  (`{uuid}_vocals.wav`), both in `std::env::temp_dir()`
- Call `ffmpeg::extract_audio(input_path, &extracted_wav_path)`

Step 4 — Run inference:
- Construct `Separator::new(config, model_params)`
- Call `separator.separate_vocals(&extracted_wav_path, &vocals_wav_path, progress_cb)`

Step 5 — Remux:
- Compute output path: `{output_dir}/{original_stem}_no_music.{original_ext}`
- Call `ffmpeg::remux_with_audio(input_path, &vocals_wav_path, &output_path)`

Step 6 — Cleanup:
- Attempt `fs::remove_file` for both temp files
- On failure, print a warning to stderr; do not return an error
- Print final: `Done → {output_path}`

On any error in steps 1–5, attempt cleanup of temp files before returning the error.

Write a unit test:

#[test]
fn run_returns_error_for_missing_input() {
    let config = /* minimal valid config pointing to real model */;
    let result = run(Path::new("/nonexistent/video.mp4"), &config);
    assert!(matches!(result, Err(AppError::InputVideoNotFound(_))));
}

All existing tests must pass. `cargo build` must succeed.
```

---

### Prompt 10 — CLI Entry Point & Final Wiring

**Goal:** Implement `main.rs`, wire the full application together, add integration test scaffolding, and verify the complete binary builds and runs end-to-end.

```text
Implement `src/main.rs` for the `music-separator` project.

Use `clap` with the `derive` feature to define the CLI:

#[derive(Parser)]
#[command(name = "music-separator", about = "Remove background music from video files")]
struct Args {
    /// Path to the input video file
    #[arg(long)]
    input: PathBuf,
}

`main` must:
1. Parse `Args` with `Args::parse()`.
2. Call `config::load()`. On error, print "Error: {e}" to stderr and `std::process::exit(1)`.
3. Call `pipeline::run(&args.input, &config)`. On error, print "Error: {e}" to stderr and
   `std::process::exit(1)`.
4. On success, exit with code 0 (implicit).

No `anyhow` or `?` in `main` — catch all errors explicitly and exit with code 1.

---

Also create `tests/integration_test.rs` with the following test structure (use the `assert_cmd`
crate — add it to `[dev-dependencies]`):

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[test]
fn cli_exits_1_when_no_config() {
    // Run in a temp dir with no config.json present
    // Assert exit code is 1
    // Assert stderr contains "config"
}

#[test]
fn cli_exits_1_when_input_not_found() {
    // Write a valid config.json to a temp dir
    // Run with --input /nonexistent/video.mp4 from that dir
    // Assert exit code is 1
    // Assert stderr contains the missing file path
}

#[test]
fn cli_shows_help() {
    // Run with --help
    // Assert exit code is 0
    // Assert stdout contains "--input"
}

Add `assert_cmd = "2"` to `[dev-dependencies]` in Cargo.toml.

After implementation, run `cargo test` and `cargo build --release`. Fix any compilation errors.
The binary must:
- Print help with `--help`
- Exit 1 with a clear message when `config.json` is absent
- Exit 1 with a clear message when `--input` file does not exist
- Exit 1 when model file in config does not exist
- Build successfully for the host platform in release mode

All tests must pass with `cargo test`.
```
