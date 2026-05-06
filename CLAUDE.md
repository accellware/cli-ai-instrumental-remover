# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Cross-platform Rust CLI that strips background music from a video file using an MDX-Net ONNX model. The video stream is **never re-encoded** — only the audio track is replaced with the isolated stem.

```
music-separator --input interview.mp4   →   output/interview_no_music.mp4
```

## Repository layout

The Cargo crate is **not at the repo root** — it lives under `impl/`. All `cargo` commands must be run from `impl/`.

```
.
├── impl/                      Rust crate (run cargo from here)
│   ├── Cargo.toml
│   ├── src/                   main.rs, config.rs, pipeline.rs, ffmpeg.rs,
│   │                          inference.rs, model_data.rs, error.rs
│   ├── tests/integration_test.rs
│   └── config.json            (gitignored — created locally for runs)
├── models/mdxnet/             ONNX models + model_data.json (committed)
├── specs/                     draft.md, prompt_plan.md, todo.md (design notes)
├── tests/raw_vid.mp4          sample input for manual end-to-end runs
└── .github/copilot-instructions.md   richer architecture/convention notes
```

## Build & run

```bash
# From impl/ — feature flag is REQUIRED for any real run
cargo build --release --features ffmpeg

# Without --features ffmpeg: compiles and unit tests pass, but ffmpeg.rs
# stubs return errors at runtime. Use the no-feature build only for testing
# pure-DSP code (inference.rs, etc.).

# Run (config.json must be in CWD)
./target/release/music-separator --input path/to/video.mp4
```

### Tests

```bash
cargo test                           # all unit + integration
cargo test --features ffmpeg         # include ffmpeg-gated tests
cargo test <name>                    # single test by name substring
cargo test --test integration_test   # only integration tests
```

FFmpeg tests that need real media are **compile-only** — they exercise the error path against a nonexistent file. STFT round-trip test asserts `max_abs_error < 0.01` on a 440 Hz sine.

### Windows build prerequisites

- LLVM / `libclang` (required by `bindgen` for `ffmpeg-sys-next`).
- FFmpeg **shared** build (7.x or 8.x); set `FFMPEG_DIR` to the extracted root, add `%FFMPEG_DIR%\bin` to `PATH`.
- Per-machine env vars are wired through `impl/.cargo/config.toml` which is **gitignored** — don't commit it.
- `add-ffmpeg-path.ps1` at the repo root is a helper for setting these.

## Architecture

**Data flow:** `input video → probe audio → extract WAV → ONNX inference → remux → cleanup`

The 6-step pipeline lives in `pipeline.rs::run()` and uses `indicatif::MultiProgress` (always to **stderr**):

1. **Validate** — model file, input file, load `model_data.json` from the model's parent dir, create `output_dir`.
2. **Probe audio** — if no audio stream: copy input verbatim with `_no_music` suffix, exit 0.
3. **Extract audio** — demux to a temp WAV at original sample rate / channel count.
4. **ONNX inference** — see "Inference details" below.
5. **Remux** — `-c:v copy` from original + vocals WAV → `{output_dir}/{stem}_no_music.{ext}`.
6. **Cleanup** — delete both temp WAVs; cleanup failures are warnings only, never errors.

### Module responsibilities

| Module          | Role                                                                                   |
| --------------- | -------------------------------------------------------------------------------------- |
| `main.rs`       | clap `Args { --input }`, calls `config::load()` then `pipeline::run()`                 |
| `config.rs`     | `Config` struct; `load()` reads `config.json` from CWD                                 |
| `model_data.rs` | `ModelParams`; parses `model_data.json` next to the `.onnx`; `find_by_name()`          |
| `ffmpeg.rs`     | `probe_audio`, `extract_audio`, `remux_with_audio` (gated by `feature = "ffmpeg"`)     |
| `inference.rs`  | DSP primitives + `Separator` (ONNX session + chunk loop)                               |
| `pipeline.rs`   | 6-step orchestrator                                                                    |
| `error.rs`      | `AppError` enum via `thiserror` — one variant per failure mode                         |

### Inference details

- Model input is **44100 Hz mono float32**. Caller resamples down then back up.
- STFT: `fft_size = mdx_n_fft_scale_set`, `hop = fft_size / 4`, Hann window, one-sided spectrum (`fft_size/2 + 1` bins).
- Tensor shape: `[1, 4, mdx_dim_f_set, mdx_dim_t_set]` — channels are `[real_ch0, imag_ch0, real_ch1, imag_ch1]`.
- Chunks use **50% overlap**; reconstruct via overlap-add.
- After iSTFT: multiply by `params.compensate`, then `denormalize(output, scale)`.
- If original was stereo: duplicate mono → stereo via `interleave_stereo(&output, &output)` before writing the WAV.

## Conventions

- **`main.rs` must not use `?`.** Catch all errors explicitly, print to stderr, `std::process::exit(1)`.
- All other modules return `AppError` (thiserror). `anyhow` is allowed **only** in `main.rs`.
- Every public function in `ffmpeg.rs` must call `ffmpeg_next::init()` at the top (idempotent).
- Progress (indicatif spinners/bars) goes to **stderr**, never stdout. stdout stays clean for scripting.
- Temp files: UUID-named (`<uuid>_extracted.wav`, `<uuid>_vocals.wav`) in `std::env::temp_dir()`. Cleanup failures warn, never error.
- Output naming is fixed: `{output_dir}/{original_stem}_no_music.{original_ext}`. Original extension and video stream are preserved bit-exact.

### Error-to-exit mapping

| Condition                     | Behavior                                  |
| ----------------------------- | ----------------------------------------- |
| `config.json` absent from CWD | `AppError::ConfigNotFound`, exit 1        |
| No audio stream in input      | copy verbatim with suffix, exit 0         |
| Temp file cleanup fails       | print warning, exit 0                     |
| Any other failure             | `AppError` variant, print stderr, exit 1  |

## Config & models

`config.json` (loaded from CWD, **not committed**):

```json
{
  "model_path": "models/mdxnet/UVR-MDX-NET-Voc_FT.onnx",
  "output_dir": "./output",
  "execution_provider": "cpu",
  "chunk_size": 261120
}
```

`execution_provider` is `"cpu"` or `"cuda"`. `model_data.json` **must be co-located** with the chosen `.onnx` file. Three models ship under `models/mdxnet/`: `UVR-MDX-NET-Voc_FT.onnx` (vocals), `UVR_MDXNET_KARA_2.onnx` (instrumental), `UVR-MDX-NET-Inst_HQ_3.onnx` (HQ instrumental). The pipeline always writes whatever the model's `primary_stem` is.

## Key crates

`clap 4` (derive), `ort 2.x-rc` (cuda feature), `ffmpeg-next 8` (optional via `ffmpeg` feature), `rustfft 6`, `hound 3`, `indicatif 0.17`, `uuid 1` (v4), `thiserror 1`, `anyhow 1`. Dev: `tempfile`, `assert_cmd`.

`ort` is pinned to a 2.x RC — when bumping, re-check that the CUDA EP downloads correctly and that `Session` / `SessionOutputs` API hasn't shifted.
