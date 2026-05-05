# `music-separator` — Implementation Checklist

Track progress by checking off each item as it is completed.

---

## Prompt 1 — Project Scaffolding & Error Types

### Cargo Setup

- [x] Create new Rust binary crate named `music-separator`
- [x] Add all `[dependencies]` to `Cargo.toml`: `clap`, `serde`, `serde_json`, `ort`, `ffmpeg-next`, `rustfft`, `hound`, `indicatif`, `uuid`, `thiserror`, `anyhow`
- [x] Add `[dev-dependencies]`: `tempfile`
- [x] Verify `cargo build` succeeds on a clean project

### Module Stubs

- [x] Create `src/main.rs` with all `mod` declarations
- [x] Create `src/error.rs` stub
- [x] Create `src/config.rs` stub
- [x] Create `src/model_data.rs` stub
- [x] Create `src/ffmpeg.rs` stub
- [x] Create `src/inference.rs` stub
- [x] Create `src/pipeline.rs` stub

### `src/error.rs` — AppError Enum

- [x] Define `AppError` enum with `thiserror`
- [x] `ConfigNotFound` variant with human-readable `#[error]` message
- [x] `ConfigParse(String)` variant
- [x] `ModelNotFound(PathBuf)` variant
- [x] `ModelDataNotFound(PathBuf)` variant
- [x] `ModelDataParse(String)` variant
- [x] `ModelNotInRegistry(String)` variant
- [x] `InputVideoNotFound(PathBuf)` variant
- [x] `OutputDirCreate(String)` variant
- [x] `FfmpegProbe(String)` variant
- [x] `FfmpegExtract(String)` variant
- [x] `FfmpegRemux(String)` variant
- [x] `OnnxLoad(String)` variant
- [x] `OnnxInference(String)` variant
- [x] `AudioRead(String)` variant
- [x] `AudioWrite(String)` variant
- [x] `FileCopy(String)` variant

### Tests — `error.rs`

- [x] Unit test: every variant formats to a non-empty string via `to_string()`
- [x] `cargo test` passes (1/1 against full crate)

### Notes

- `ort` pinned to `2.0.0-rc.12` — no stable 2.x release on crates.io.
- Build environment: installed FFmpeg 7.1 shared dev libs to `~/ffmpeg-7.1` and LLVM (for libclang) to `C:\Program Files\LLVM`. Paths are wired via gitignored `impl/.cargo/config.toml`'s `[env]` block.
- **`ffmpeg-next` made `optional = true` and gated behind a `ffmpeg` feature** in `Cargo.toml`. Reason: `ffmpeg-sys-next 7.1.3` has hardcoded `size_of` assertions that don't match FFmpeg 7.1's opaque-struct API on Windows (10 layout errors against multiple `AVFilter*` types and `tm`). Re-enabling will be addressed in Prompt 4 — likely by bumping to `ffmpeg-next 8.x` or pinning to a known-good `ffmpeg-sys-next` version.

---

## Prompt 2 — Config Module

### `src/config.rs` — Types

- [x] Define `ExecutionProvider` enum with `Cpu` and `Cuda` variants
- [x] Serde rename: `"cpu"` → `Cpu`, `"cuda"` → `Cuda`
- [x] Define `Config` struct with fields: `model_path`, `output_dir`, `execution_provider`, `chunk_size`
- [x] Derive `Deserialize` on both types

### `src/config.rs` — `load()` Function

- [x] Look for `config.json` in `std::env::current_dir()`
- [x] Return `AppError::ConfigNotFound` if file absent
- [x] Parse JSON and return `AppError::ConfigParse` on malformed input
- [x] Validate `chunk_size > 0`, return `AppError::ConfigParse` if zero
- [x] Return parsed `Config` on success

### Tests — `config.rs`

- [x] Valid config deserializes all fields correctly
- [x] Missing `config.json` → `AppError::ConfigNotFound`
- [x] Malformed JSON → `AppError::ConfigParse`
- [x] `chunk_size: 0` → `AppError::ConfigParse`
- [x] Unknown `execution_provider` string → `AppError::ConfigParse`
- [x] `"cuda"` deserializes to `ExecutionProvider::Cuda`
- [x] `cargo test` passes (7/7)

---

## Prompt 3 — Model Data Module

### `src/model_data.rs` — Types

- [ ] Define `ModelParams` struct with fields: `compensate`, `mdx_dim_f_set`, `mdx_dim_t_set`, `mdx_n_fft_scale_set`, `primary_stem`, `name`
- [ ] Derive `Deserialize` on `ModelParams`

### `src/model_data.rs` — Functions

- [ ] Implement `load(model_data_path: &Path) -> Result<Vec<ModelParams>, AppError>`
  - [ ] Return `AppError::ModelDataNotFound` if file absent
  - [ ] Return `AppError::ModelDataParse` if JSON parsing fails
  - [ ] Deserialize hash map and return its values as `Vec<ModelParams>`
- [ ] Implement `find_by_name(params: &[ModelParams], model_filename: &str) -> Result<&ModelParams, AppError>`
  - [ ] Match on `name` field (filename only, not full path)
  - [ ] Return `AppError::ModelNotInRegistry` if not found

### Tests — `model_data.rs`

- [ ] `load` on valid JSON tempfile returns correct `Vec`
- [ ] `load` on missing file → `ModelDataNotFound`
- [ ] `load` on malformed JSON → `ModelDataParse`
- [ ] `find_by_name` correctly finds `"UVR-MDX-NET-Voc_FT.onnx"`
- [ ] `find_by_name` on unknown name → `ModelNotInRegistry`
- [ ] `cargo test` passes

---

## Prompt 4 — FFmpeg: Audio Probe & Extraction

### `src/ffmpeg.rs` — Types & Init

- [ ] Define `AudioInfo` struct with `sample_rate: u32` and `channels: u16`
- [ ] Call `ffmpeg_next::init()` at the top of each public function

### `src/ffmpeg.rs` — `probe_audio`

- [ ] Open input with `ffmpeg_next::format::input`
- [ ] Wrap open errors as `AppError::FfmpegProbe`
- [ ] Iterate streams looking for an audio stream
- [ ] Return `Some(AudioInfo)` if audio stream found with correct sample_rate and channels
- [ ] Return `None` if no audio stream present

### `src/ffmpeg.rs` — `extract_audio`

- [ ] Open input video
- [ ] Find best audio stream
- [ ] Open output WAV with `wav` muxer
- [ ] Add audio stream to output copying codec parameters
- [ ] Demux and write audio packets, flush at end
- [ ] Return `AudioInfo` from source stream
- [ ] Wrap all errors as `AppError::FfmpegExtract`

### Tests — `ffmpeg.rs` (Prompt 4)

- [ ] `probe_audio` on nonexistent file returns `Err`
- [ ] `cargo build` succeeds

---

## Prompt 5 — FFmpeg: Video Remux

### `src/ffmpeg.rs` — `remux_with_audio`

- [ ] Open original video input — wrap as `AppError::FfmpegRemux`
- [ ] Open vocals WAV input
- [ ] Open output file
- [ ] Copy all video streams with `-c:v copy` semantics
- [ ] Copy audio stream from vocals WAV with `-c:a copy` semantics
- [ ] Write output header
- [ ] Interleave and write all packets, rescaling PTS/DTS with `packet.rescale_ts`
- [ ] Write output trailer
- [ ] Wrap all errors as `AppError::FfmpegRemux`

### Tests — `ffmpeg.rs` (Prompt 5)

- [ ] `remux_with_audio` on nonexistent inputs returns `Err`
- [ ] All previous tests still pass
- [ ] `cargo build` succeeds

---

## Prompt 6 — Audio DSP: STFT, iSTFT, and Hann Window

### `src/inference.rs` — DSP Primitives

- [ ] Implement `hann_window(size: usize) -> Vec<f32>` using `0.5 * (1 - cos(2π*i/n))`
- [ ] Implement `stft(signal, fft_size, hop_length, window) -> Vec<Vec<[f32; 2]>>`
  - [ ] Center-pad signal with `fft_size/2` zeros on each side
  - [ ] Apply window element-wise per frame
  - [ ] Run forward FFT via `rustfft`
  - [ ] Return only first `fft_size/2 + 1` bins (one-sided spectrum)
- [ ] Implement `istft(frames, fft_size, hop_length, window, signal_length) -> Vec<f32>`
  - [ ] Mirror-conjugate one-sided spectrum to full spectrum
  - [ ] Run inverse FFT per frame
  - [ ] Apply window element-wise
  - [ ] Overlap-add into output buffer
  - [ ] Normalize by sum-of-squared windows (OLA normalization)
  - [ ] Trim output to `signal_length`

### Tests — DSP Primitives

- [ ] `hann_window(4)` ≈ `[0.0, 0.75, 0.75, 0.0]` within `1e-5`
- [ ] STFT round-trip on 440 Hz sine (44100 samples): max absolute error < `0.01`
- [ ] STFT on zero signal returns all-zero frames
- [ ] `istft` output length equals `signal_length`
- [ ] `cargo test` passes

---

## Prompt 7 — Audio DSP: Resampling & Preprocessing

### `src/inference.rs` — Preprocessing Functions

- [ ] Implement `resample(signal, from_rate, to_rate) -> Vec<f32>` using linear interpolation
- [ ] Implement `to_mono(samples, channels) -> Vec<f32>`
  - [ ] Pass through if `channels == 1`
  - [ ] Average pairs for stereo; average N samples for N channels
- [ ] Implement `interleave_stereo(left, right) -> Vec<f32>` → `[L0, R0, L1, R1, ...]`
- [ ] Implement `normalize(signal) -> (Vec<f32>, f32)`
  - [ ] Divide by max absolute value
  - [ ] Return `(normalized, scale_factor)`
  - [ ] Handle zero signal: return unchanged with scale = 1.0
- [ ] Implement `denormalize(signal, scale) -> Vec<f32>` → multiply each sample by scale

### Tests — Preprocessing

- [ ] `to_mono([1.0, 0.0, 0.0, 1.0], channels=2)` → `[0.5, 0.5]`
- [ ] `resample` from 44100→22050 on 44100-sample signal → 22050-sample output
- [ ] `resample` from 22050→44100 on 22050-sample signal → 44100-sample output
- [ ] `normalize([2.0, -4.0, 1.0])` → `([0.5, -1.0, 0.25], 4.0)`
- [ ] `denormalize(normalize(x))` round-trips within `1e-6`
- [ ] `interleave_stereo([1.0, 2.0], [3.0, 4.0])` → `[1.0, 3.0, 2.0, 4.0]`
- [ ] All previous tests still pass
- [ ] `cargo test` passes

---

## Prompt 8 — ONNX Inference: Model Session & Chunk Loop

### `src/inference.rs` — `Separator` Struct

- [ ] Define `Separator` struct with fields: `session`, `params`, `chunk_size`
- [ ] Implement `Separator::new(config, params) -> Result<Self, AppError>`
  - [ ] Build `ort::SessionBuilder`
  - [ ] Add CUDA execution provider if `ExecutionProvider::Cuda`
  - [ ] Load model from `config.model_path`
  - [ ] Wrap load errors as `AppError::OnnxLoad`

### `src/inference.rs` — `separate_vocals`

- [ ] Read WAV with `hound::WavReader` → `AppError::AudioRead`
- [ ] Collect samples as `Vec<f32>` (normalize i16/i32 to float range)
- [ ] Call `to_mono(samples, channels)`
- [ ] Resample to 44100 Hz if needed; record whether resampling occurred
- [ ] Call `normalize` to get `(normalized, scale)`
- [ ] Compute `hop_length = mdx_n_fft_scale_set / 4`
- [ ] Compute `hann_window(mdx_n_fft_scale_set)`
- [ ] Split signal into overlapping chunks (50% overlap), zero-pad last chunk
- [ ] Per chunk:
  - [ ] Run `stft` → complex spectrogram
  - [ ] Reshape to tensor `[1, 4, mdx_dim_f_set, mdx_dim_t_set]`
  - [ ] Run ONNX forward pass → `AppError::OnnxInference`
  - [ ] Reshape output tensor back to complex frames
  - [ ] Run `istft` to reconstruct time-domain chunk
  - [ ] Call `progress_cb(chunk_index / total_chunks)`
- [ ] Overlap-add all output chunks into full-length signal
- [ ] Apply `compensate` multiplier
- [ ] Call `denormalize(output, scale)`
- [ ] Resample back to original sample rate if resampling occurred
- [ ] Duplicate mono to stereo via `interleave_stereo` if original was stereo
- [ ] Write output WAV with `hound::WavWriter` at original sample rate and channels → `AppError::AudioWrite`

### Tests — Inference

- [ ] `Separator::new` fails on missing model file → `AppError::OnnxLoad`
- [ ] All previous tests still pass
- [ ] `cargo build` succeeds

---

## Prompt 9 — Pipeline Orchestrator & Progress Reporting

### `src/pipeline.rs` — `run` Function

- [ ] Implement `pub fn run(input_path: &Path, config: &Config) -> Result<(), AppError>`
- [ ] Set up `indicatif::MultiProgress` targeting stderr

### Step 1 — Validate (spinner)

- [ ] Verify `config.model_path` exists → `AppError::ModelNotFound`
- [ ] Verify `input_path` exists → `AppError::InputVideoNotFound`
- [ ] Load `model_data.json` from the model file's parent directory
- [ ] Call `model_data::find_by_name` for the model filename
- [ ] Create `config.output_dir` with `fs::create_dir_all` → `AppError::OutputDirCreate`
- [ ] Finish spinner with `✓`

### Step 2 — Probe audio (spinner)

- [ ] Call `ffmpeg::probe_audio(input_path)`
- [ ] If no audio: compute `{output_dir}/{stem}_no_music.{ext}`, copy input file, print notice to stderr, return `Ok(())`
- [ ] Finish spinner with `✓`

### Step 3 — Extract audio (spinner)

- [ ] Generate UUID-based temp paths in `std::env::temp_dir()`
- [ ] Call `ffmpeg::extract_audio(input_path, &extracted_wav_path)`
- [ ] Finish spinner with `✓`

### Step 4 — Run inference (progress bar)

- [ ] Construct `Separator::new(config, model_params)`
- [ ] Call `separator.separate_vocals` with a `progress_cb` that updates the bar to percentage
- [ ] Finish bar at 100%

### Step 5 — Remux (spinner)

- [ ] Compute output path: `{output_dir}/{original_stem}_no_music.{original_ext}`
- [ ] Call `ffmpeg::remux_with_audio(input_path, &vocals_wav_path, &output_path)`
- [ ] Finish spinner with `✓`

### Step 6 — Cleanup

- [ ] `fs::remove_file` for extracted WAV; on failure, print warning to stderr
- [ ] `fs::remove_file` for vocals WAV; on failure, print warning to stderr
- [ ] Print `Done → {output_path}` to stderr
- [ ] On any error in steps 1–5, attempt cleanup before returning the error

### Tests — `pipeline.rs`

- [ ] `run` with nonexistent input path → `AppError::InputVideoNotFound`
- [ ] All previous tests still pass
- [ ] `cargo build` succeeds

---

## Prompt 10 — CLI Entry Point & Final Wiring

### `src/main.rs`

- [ ] Define `Args` struct with `clap` derive: `--input <PathBuf>`
- [ ] Set `name = "music-separator"` and `about` description on the command
- [ ] Call `config::load()`, print error to stderr and `exit(1)` on failure
- [ ] Call `pipeline::run(&args.input, &config)`, print error to stderr and `exit(1)` on failure
- [ ] No `?` or `anyhow` in `main` — all errors handled explicitly

### `tests/integration_test.rs`

- [ ] Add `assert_cmd = "2"` to `[dev-dependencies]` in `Cargo.toml`
- [ ] Test: `cli_exits_1_when_no_config` — run in temp dir without `config.json`, assert exit code 1 and stderr contains "config"
- [ ] Test: `cli_exits_1_when_input_not_found` — valid config, nonexistent `--input`, assert exit code 1 and stderr contains the missing path
- [ ] Test: `cli_shows_help` — run with `--help`, assert exit code 0 and stdout contains `--input`

### Final Verification

- [ ] `cargo test` — all unit and integration tests pass
- [ ] `cargo build --release` — release binary builds without errors or warnings
- [ ] Binary prints help with `--help`
- [ ] Binary exits 1 with clear message when `config.json` is absent
- [ ] Binary exits 1 with clear message when `--input` file does not exist
- [ ] Binary exits 1 with clear message when model file in config does not exist

---

## End-to-End & Manual QA

### Functional Testing

- [ ] Process a real `.mp4` file with a music + speech audio track
- [ ] Verify output file plays correctly (VLC, browser)
- [ ] Verify background music is audibly reduced in the output
- [ ] Verify the output video stream is bit-for-bit identical to input (no re-encode)
- [ ] Verify output sample rate matches the input audio's original sample rate

### Format Compatibility

- [ ] Test with `.mp4` input
- [ ] Test with `.mkv` input
- [ ] Test with `.mov` input
- [ ] Test with `.avi` input

### Edge Cases

- [ ] Input video with no audio track → file copied to output, exit 0
- [ ] Input video with audio at a sample rate other than 44100 Hz → output rate matches original
- [ ] Input video with stereo audio → output is stereo
- [ ] Input video with mono audio → output is mono
- [ ] Very short video (< 1 chunk worth of audio) → processes without crash

### Cross-Platform

- [ ] Build and run successfully on Windows
- [ ] Build and run successfully on Linux
- [ ] Build and run successfully on macOS

### Config & Error Conditions

- [ ] Missing `config.json` → descriptive error, exit 1
- [ ] Malformed `config.json` → descriptive error, exit 1
- [ ] `chunk_size: 0` in config → descriptive error, exit 1
- [ ] `model_path` points to nonexistent file → descriptive error, exit 1
- [ ] `model_data.json` missing from model directory → descriptive error, exit 1
- [ ] Model filename not listed in `model_data.json` → descriptive error, exit 1
- [ ] `output_dir` cannot be created (permissions) → descriptive error, exit 1
- [ ] `execution_provider: "cuda"` on a CPU-only machine → clear error message, exit 1
- [ ] Temp file cleanup failure → warning printed, process still exits 0

### Output Naming

- [ ] `interview.mp4` → `{output_dir}/interview_no_music.mp4`
- [ ] `clip.mkv` → `{output_dir}/clip_no_music.mkv`
- [ ] `recording.mov` → `{output_dir}/recording_no_music.mov`
