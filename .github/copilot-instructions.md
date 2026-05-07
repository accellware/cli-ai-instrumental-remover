# music-separator — Copilot Instructions

A cross-platform Rust CLI that strips background music from video files using MDX-Net ONNX inference. The video stream is never re-encoded; only the audio track is replaced with the isolated vocals stem.

## Architecture

```
src/
├── main.rs        — CLI entry: clap Args { --input }, calls config::load() then pipeline::run()
├── config.rs      — Config struct, load() reads config.json from CWD
├── model_data.rs  — ModelParams, load() parses model_data.json, find_by_name() lookup
├── ffmpeg.rs      — probe_audio(), extract_audio(), remux_with_audio()
├── inference.rs   — DSP primitives + Separator struct (ONNX session + chunk loop)
├── pipeline.rs    — run(): orchestrates the 6-step pipeline with indicatif progress
└── error.rs       — AppError enum (thiserror), one variant per failure mode
```

**Data flow:** input video → probe audio → extract WAV → ONNX inference → remux → cleanup

## Code Conventions

- `main.rs` must not use `?`. Catch all errors explicitly, print to stderr, call `std::process::exit(1)`.
- Use `AppError` variants (thiserror) throughout all modules. `anyhow` is only allowed in `main`.
- Every public function in `ffmpeg.rs` must call `ffmpeg_next::init()` at the top (it is idempotent).
- Progress reporting (indicatif spinners and bars) goes to **stderr**, never stdout. Use `MultiProgress` targeting stderr.
- Temp files use UUID naming: `<uuid>_extracted.wav`, `<uuid>_vocals.wav` in `std::env::temp_dir()`.
- Temp file cleanup failures are warnings only — never propagate as errors.

## Pipeline Steps

1. **Validate** — check model file, input file, load `model_data.json` from model's parent dir, create `output_dir`
2. **Probe audio** — if no audio stream: copy input verbatim with `_no_music` suffix, print notice, return `Ok(())`
3. **Extract audio** — demux to temp WAV, preserve original sample rate and channel count
4. **ONNX inference** — resample→44100 Hz, mono, normalize, STFT chunks, forward pass, iSTFT, overlap-add, apply compensate, denormalize, resample back
5. **Remux** — `-c:v copy` from original + vocals WAV → `{output_dir}/{stem}_no_music.{ext}`
6. **Cleanup** — delete both temp WAVs; warn on failure

## Inference Details

- MDX-Net requires **44100 Hz mono float32** input.
- STFT: `fft_size = mdx_n_fft_scale_set`, `hop = fft_size / 4`, Hann window, one-sided spectrum (`fft_size/2 + 1` bins).
- Tensor shape into ONNX: `[1, 4, mdx_dim_f_set, mdx_dim_t_set]` — 4 channels are `[real_ch0, imag_ch0, real_ch1, imag_ch1]`.
- Chunks with **50% overlap**; use overlap-add for reconstruction.
- After iSTFT: multiply by `params.compensate`, then `denormalize(output, scale)`.
- If original was stereo: call `interleave_stereo(&output, &output)` to duplicate mono→stereo before writing WAV.

## Error Handling

| Condition                     | Behavior                                    |
| ----------------------------- | ------------------------------------------- |
| `config.json` absent from CWD | `AppError::ConfigNotFound`, exit 1          |
| No audio stream in input      | copy verbatim, exit 0                       |
| Temp file cleanup fails       | print warning, exit 0                       |
| Any other failure             | `AppError` variant, print to stderr, exit 1 |

## Build & Test

```bash
# Build
cargo build
cargo build --release

# Test (all unit + integration)
cargo test

# Windows: FFmpeg env vars are wired in impl/.cargo/config.toml (gitignored)
# LLVM/libclang must be installed for ffmpeg-sys-next bindings
```

## Testing Conventions

- Unit tests live in `#[cfg(test)]` blocks inside each module.
- Use `tempfile::tempdir()` for any test that writes files.
- Integration tests are in `tests/integration_test.rs` using `assert_cmd`.
- FFmpeg tests that require real media files are **compile-only** — test the error path (nonexistent file) only.
- STFT round-trip test: generate a 440 Hz sine at 44100 samples, verify `max_abs_error < 0.01` after STFT→iSTFT.

## Config File (`config.json`, loaded from CWD, not committed)

```json
{
  "model_path": "models/mdxnet/UVR-MDX-NET-Voc_FT.onnx",
  "output_dir": "./output",
  "execution_provider": "cpu"
}
```

`model_data.json` must be co-located with the `.onnx` file. The per-inference chunk length is fixed by the model architecture (`1024 * ((1 << mdx_dim_t_set) - 1)`) and is not user-configurable.

### Optional `tuning` block

All ORT/CUDA tuning is opt-in. Top-level `tuning` may include any of: `intra_threads`, `inter_threads`, `parallel_execution`, `memory_pattern`. The nested `tuning.cuda` may include: `device_id`, `gpu_mem_limit_mb` (MiB), `arena_extend_strategy` (`"next_power_of_two"` | `"same_as_requested"`), `cudnn_conv_algo_search` (`"exhaustive"` | `"heuristic"` | `"default"`), `cudnn_conv_use_max_workspace`, `tf32`, `prefer_nhwc`. Each key maps 1:1 to the matching ort 2.x `SessionBuilder` / `ort::ep::CUDA` builder method. Omitted fields keep ORT's default; unknown keys are rejected.

## Dependencies

| Crate                    | Purpose                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------ |
| `clap 4.x` (derive)      | CLI argument parsing                                                                 |
| `ort 2.x` (cuda feature) | ONNX Runtime bindings                                                                |
| `ffmpeg-next 8.x`        | Audio probe, extract, remux (upgraded from 7.x for FFmpeg 7.1 API compat on Windows) |
| `rustfft 6.x`            | STFT / iSTFT                                                                         |
| `hound 3.x`              | WAV read/write                                                                       |
| `indicatif 0.17.x`       | Progress bars/spinners                                                               |
| `thiserror 1.x`          | AppError enum                                                                        |
| `anyhow 1.x`             | Error context in main only                                                           |
