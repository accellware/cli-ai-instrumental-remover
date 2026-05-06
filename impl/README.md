# music-separator

A cross-platform CLI that strips background music from a video file using an
MDX-Net ONNX model for audio source separation. The video stream is never
re-encoded — only the audio track is replaced with the isolated stem.

```
music-separator --input interview.mp4
# → output/interview_no_music.mp4
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Build](#build)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Docker Image Preparation](#docker-image-preparation)
6. [Running](#running)
7. [Available Models](#available-models)
8. [Progress Output](#progress-output)
9. [Logging](#logging)
10. [Output Naming](#output-naming)
11. [Error Reference](#error-reference)
12. [CUDA / GPU Acceleration](#cuda--gpu-acceleration)
13. [Development](#development)

---

## Prerequisites

| Dependency              | Version           | Notes                                 |
| ----------------------- | ----------------- | ------------------------------------- |
| Rust toolchain          | 1.76+             | `rustup update stable`                |
| FFmpeg shared libraries | 7.x or 8.x        | Must be on `PATH` / `FFMPEG_DIR`      |
| LLVM / libclang         | any recent        | Required by `bindgen` at compile time |
| ONNX Runtime            | bundled via `ort` | Downloaded automatically              |
| CUDA + cuDNN            | optional          | Only for `execution_provider: "cuda"` |

### Installing FFmpeg (Windows)

Download the **shared** build from <https://www.gyan.dev/ffmpeg/builds/> (e.g.
`ffmpeg-8.1-full_build-shared.7z`), extract it, then set the environment
variables so the compiler and linker can find the headers and `.lib` files:

```powershell
# In your shell profile or .cargo/config.toml [env] block:
FFMPEG_DIR   = "C:\ffmpeg-8.1"          # root of the extracted archive
LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
```

Add `%FFMPEG_DIR%\bin` to your system `PATH` so the `.dll` files are found at
runtime.

### Installing FFmpeg (Linux)

```bash
# Debian / Ubuntu
sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
                 libswresample-dev libswscale-dev libavfilter-dev clang

# Arch
sudo pacman -S ffmpeg clang
```

### Installing FFmpeg (macOS)

```bash
brew install ffmpeg llvm
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
```

---

## Build

```bash
# Debug build (fast compile, slower runtime)
cargo build --features ffmpeg

# Release build (optimized — use this for real audio processing)
cargo build --release --features ffmpeg
```

The compiled binary is placed at:

- Debug: `target/debug/music-separator`
- Release: `target/release/music-separator`

> **Note**: building without `--features ffmpeg` produces a binary that
> compiles fine and passes all tests, but will exit with an error when you
> actually try to process a video. Always include `--features ffmpeg` for use.

---

## Setup

The tool expects two things in the directory you run it from:

1. A `config.json` file (see [Configuration](#configuration))
2. The ONNX model file and a `model_data.json` at the path referenced by
   `model_path` in that config

The repository already ships three models under `../models/mdxnet/`:

```
models/
└── mdxnet/
    ├── model_data.json
    ├── UVR-MDX-NET-Voc_FT.onnx       ← vocals separation (recommended)
    ├── UVR_MDXNET_KARA_2.onnx         ← instrumental / karaoke removal
    └── UVR-MDX-NET-Inst_HQ_3.onnx     ← high-quality instrumental
```

Copy or symlink the `models/` folder (or just its `mdxnet/` subdirectory)
next to your `config.json`, or use absolute paths in the config.

---

## Configuration

Create `config.json` in the directory where you will run the binary:

```json
{
  "model_path": "models/mdxnet/UVR-MDX-NET-Voc_FT.onnx",
  "output_dir": "./output",
  "execution_provider": "cpu"
}
```

### Fields

| Field                | Type                | Required | Description                                                                                                                                 |
| -------------------- | ------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_path`         | string              | yes      | Relative or absolute path to the `.onnx` model file. The `model_data.json` must be in the **same directory**.                               |
| `output_dir`         | string              | yes      | Directory where the processed video is written. Created automatically if it does not exist.                                                 |
| `execution_provider` | `"cpu"` \| `"cuda"` | yes      | Inference device. Use `"cpu"` unless you have an NVIDIA GPU with CUDA installed.                                                            |

The per-inference chunk length is fixed by the model architecture (`1024 * ((1 << mdx_dim_t_set) - 1)`) and is not configurable.

---

## Docker Image Preparation

These steps package the CLI and models into Docker images so you can run the
tool without installing Rust/FFmpeg on the host.

Run all Docker commands from the **repository root** (the directory that
contains `Dockerfile`, `Dockerfile.gpu`, and `docker-compose.yml`).

### What is already in the repo

- `Dockerfile` for CPU execution (`execution_provider: "cpu"`)
- `Dockerfile.gpu` for NVIDIA GPU execution (`execution_provider: "cuda"`)
- `docker/config.json` (CPU container config)
- `docker/config.cuda.json` (GPU container config)
- `docker-compose.yml` with services `music-separator` and `music-separator-gpu`

### Step 1: Prepare input/output folders

```powershell
New-Item -ItemType Directory -Force inputs, out | Out-Null
Copy-Item tests/raw_vid.mp4 inputs/
```

### Step 2: Build the CPU image

```bash
docker compose build music-separator
```

Optional direct build:

```bash
docker build -t music-separator:latest .
```

### Step 3: Run the CPU container

```bash
docker compose run --rm music-separator --input /inputs/raw_vid.mp4
```

Expected output file:

```text
out/raw_vid_no_music.mp4
```

### Step 4: Verify GPU host prerequisites (GPU only)

Requirements:

- NVIDIA driver with CUDA 12.4+ support
- `nvidia-container-toolkit` installed and configured for Docker

Quick validation:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Step 5: Build the GPU image

```bash
docker compose build music-separator-gpu
```

Optional direct build:

```bash
docker build -f Dockerfile.gpu -t music-separator:gpu .
```

### Step 6: Run the GPU container

```bash
docker compose run --rm music-separator-gpu --input /inputs/raw_vid.mp4
```

If your Compose environment does not pass GPU devices through, use:

```bash
docker run --rm --gpus all \
  -v ${PWD}/inputs:/inputs:ro \
  -v ${PWD}/out:/out \
  music-separator:gpu --input /inputs/raw_vid.mp4
```

### Notes

- Both images bake `models/mdxnet/*` into `/app/models/mdxnet`.
- The container uses `/app/config.json` copied from:
  - `docker/config.json` for CPU image
  - `docker/config.cuda.json` for GPU image
- First build is slow (FFmpeg 8.1.1 from source + release Rust build); later
  builds reuse Docker layer cache.

---

## Running

```bash
# From the directory that contains config.json:
./target/release/music-separator --input path/to/video.mp4
```

Or, if the binary is on your `PATH`:

```bash
music-separator --input path/to/video.mp4
```

### Full example

```bash
# Assume the repo root is your working directory
cd /path/to/music-separator-repo

# Write config
cat > config.json <<'EOF'
{
  "model_path": "models/mdxnet/UVR-MDX-NET-Voc_FT.onnx",
  "output_dir": "./output",
  "execution_provider": "cpu"
}
EOF

# Run
./impl/target/release/music-separator --input my_video.mp4
```

On success, the processed file appears at `output/my_video_no_music.mp4`.

### Help

```
music-separator --help
```

```
Remove background music from video files

Usage: music-separator --input <INPUT>

Options:
      --input <INPUT>  Path to the input video file
  -h, --help           Print help
```

---

## Available Models

All three models ship in `models/mdxnet/`. Point `model_path` at whichever fits
your use case:

| Model file                   | Primary stem     | Best for                                          |
| ---------------------------- | ---------------- | ------------------------------------------------- |
| `UVR-MDX-NET-Voc_FT.onnx`    | **Vocals**       | Removing background music, keeping speech/singing |
| `UVR_MDXNET_KARA_2.onnx`     | **Instrumental** | Karaoke tracks, keeping the backing track         |
| `UVR-MDX-NET-Inst_HQ_3.onnx` | **Instrumental** | High-quality instrumental separation              |

The model output is whatever the model's `primary_stem` is — the binary always
writes that stem's audio into the output video.

---

## Progress Output

All progress is printed to **stderr**, keeping **stdout** clean for scripting:

```
⠸ [1/5] Validating inputs... ✓
⠸ [2/5] Probing audio...     ✓
⠸ [3/5] Extracting audio...  ✓
  [4/5] Running inference... ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 25%
```

After inference completes:

```
  [4/5] Running inference...
⠸ [5/5] Remuxing video...    ✓
Done → output/my_video_no_music.mp4
```

If the input file has **no audio track**, the file is copied verbatim to
`output_dir` with the `_no_music` suffix and the tool exits with code 0:

```
No audio track found; copying file to output.
Done → output/silent_video_no_music.mp4
```

---

## Logging

The CLI uses the [`tracing`] ecosystem. By default it is **silent** at
`WARN` level so stdout/stderr stay clean for scripting; bump verbosity
when you need to see what the pipeline is doing.

### Flags

| Flag                  | Meaning                                                                          |
| --------------------- | -------------------------------------------------------------------------------- |
| (no flag)             | `WARN` (default)                                                                 |
| `-v`                  | `INFO` — one line per pipeline step, resolved config, model parameters           |
| `-vv`                 | `DEBUG` — file paths, audio stream layout, per-chunk inference progress          |
| `-vvv`                | `TRACE` — everything                                                             |
| `--log-level <LEVEL>` | Explicit override: `off`, `error`, `warn`, `info`, `debug`, or `trace`           |
| `--log-file <PATH>`   | Also tee logs to this file (overwritten each run)                                |

**Precedence:** `--log-level` > `-v` count > `RUST_LOG` env var > default `WARN`.

All log output goes to **stderr** — stdout stays clean. Logs play nicely
with the indicatif progress bars: bars are momentarily paused while a log
line is written so they never overwrite each other.

### `RUST_LOG`

Power users can use the standard env-filter syntax to dial different
crates independently:

```bash
RUST_LOG=music_separator=debug,ort=info music-separator --input clip.mp4
```

### Worked example

```bash
music-separator -vv --input clip.mp4
```

```
 INFO music-separator starting verbose=2 log_level=None log_file=None
 INFO config loaded model_path="models/mdxnet/UVR-MDX-NET-Voc_FT.onnx" output_dir="./output" execution_provider=Cpu
 INFO pipeline starting input="clip.mp4" output_dir="./output" execution_provider=Cpu
 INFO [1/5] validating inputs
 INFO model parameters resolved name="UVR-MDX-NET-Voc_FT.onnx" primary_stem="Vocals" ...
 INFO [2/5] probing audio stream
 INFO audio stream detected sample_rate=48000 channels=2
 INFO [3/5] extracting audio to temp WAV
 INFO [4/5] running ONNX inference
 INFO ONNX session initialised execution_provider=Cpu model="..." model_bytes=Some(...)
 INFO audio loaded for inference total_samples=... sample_rate=48000 channels=2
 INFO chunked input; running inference total_chunks=N chunk_size=261120 fft_size=7680 hop_length=1024
TRACE processed chunk chunk_index=1 of=N samples=261120
...
 INFO [5/5] remuxing video with isolated stem
 INFO pipeline finished output="output/clip_no_music.mp4"
```

[`tracing`]: https://docs.rs/tracing

---

## Output Naming

The output filename is always `{original_stem}_no_music.{original_extension}`,
placed under `output_dir`:

| Input           | Output                          |
| --------------- | ------------------------------- |
| `interview.mp4` | `output/interview_no_music.mp4` |
| `clip.mkv`      | `output/clip_no_music.mkv`      |
| `recording.mov` | `output/recording_no_music.mov` |
| `session.avi`   | `output/session_no_music.avi`   |

The original file extension and video stream are preserved exactly (no
re-encode). Only the audio track is replaced.

---

## Error Reference

All errors are printed to **stderr** as `Error: <message>` and exit with
code 1.

| Error message                                        | Cause                                              | Fix                                                                      |
| ---------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------ |
| `config.json not found in current working directory` | `config.json` missing                              | Create `config.json` in the directory you run the binary from            |
| `failed to parse config.json: …`                     | Invalid JSON or missing/wrong field                | Check the JSON syntax and all required fields                            |
| `model file not found: <path>`                       | `model_path` in config does not exist              | Verify the path points to the `.onnx` file                               |
| `model_data.json not found at: <path>`               | `model_data.json` missing from the model directory | Ensure `model_data.json` is in the same folder as the `.onnx` file       |
| `model not listed in model_data.json: <name>`        | Model filename has no entry in `model_data.json`   | Use one of the supported model filenames listed above                    |
| `input video file not found: <path>`                 | The `--input` path does not exist                  | Check the path passed to `--input`                                       |
| `failed to create output directory: …`               | Permission or path error                           | Check write access to the `output_dir` parent                            |
| `ffmpeg probe failed: …`                             | FFmpeg could not open the input                    | Verify FFmpeg libraries are installed and the file is a valid media file |
| `ffmpeg audio extraction failed: …`                  | FFmpeg demux error                                 | Check FFmpeg installation; try running `ffprobe` on the file             |
| `ffmpeg remux failed: …`                             | FFmpeg mux error                                   | Ensure the output directory is writable and has enough disk space        |
| `failed to load ONNX model: …`                       | ONNX Runtime could not open the model              | Verify `model_path` points to a valid `.onnx` file                       |
| `ONNX inference failed: …`                           | Runtime error during inference                     | Check available memory; try the CPU execution provider or a smaller model |

### CUDA errors

If `execution_provider` is `"cuda"` and CUDA is unavailable, ONNX Runtime
returns an error caught as `failed to load ONNX model`. Either install CUDA +
cuDNN or switch to `"cpu"`.

---

## CUDA / GPU Acceleration

Set `execution_provider` to `"cuda"` in `config.json`:

```json
{
  "execution_provider": "cuda",
  ...
}
```

Requirements:

- NVIDIA GPU with CUDA Compute Capability 3.5+
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.x or 12.x)
- [cuDNN](https://developer.nvidia.com/cudnn) matching your CUDA version
- Both `cuda` and `cudnn` libraries on `PATH` / `LD_LIBRARY_PATH`

The `ort` crate downloads the ONNX Runtime CUDA execution provider
automatically at compile time. No extra build flags are needed.

---

## Development

### Running tests

```bash
# All unit tests (no FFmpeg required)
cargo test

# Build + test with FFmpeg feature enabled
cargo build --features ffmpeg && cargo test
```

### Project layout

```
impl/
├── Cargo.toml
├── src/
│   ├── main.rs         CLI entry point — arg parsing, config load, pipeline dispatch
│   ├── config.rs       Config struct, JSON deserialization, validation
│   ├── pipeline.rs     6-step orchestrator, indicatif progress bars
│   ├── ffmpeg.rs       probe_audio, extract_audio, remux_with_audio
│   ├── inference.rs    STFT/iSTFT, resampling, Separator struct, ONNX chunk loop
│   ├── model_data.rs   model_data.json deserialization, model lookup
│   └── error.rs        AppError enum (thiserror)
└── tests/
    └── integration_test.rs
```

### Key design decisions

- **FFmpeg feature gate** — `ffmpeg-next` is optional (`--features ffmpeg`).
  When the feature is absent the stubs in `ffmpeg.rs` return descriptive
  errors, keeping the pure-Rust DSP code fully testable without FFmpeg
  installed.
- **No video re-encode** — `remux_with_audio` copies the video stream
  packet-by-packet (`-c:v copy` equivalent), so output file size and quality
  are identical to the input for the video track.
- **Temp files** — extracted WAV and vocals WAV are written to
  `std::env::temp_dir()` with UUID names and deleted after remux, even on
  error paths.
