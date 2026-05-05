# Specification: `music-separator` CLI

## 1. Overview

A cross-platform command-line tool written entirely in Rust that strips background music from a video file using an MDX-Net ONNX model for audio source separation. The video stream is never re-encoded; only the audio track is replaced with the isolated vocals stem.

---

## 2. Target Platforms

- Windows, Linux, macOS (single cross-platform Rust binary)

---

## 3. CLI Interface

```
music-separator --input <video_file>
```

### Arguments

| Flag             | Required | Description                                               |
| ---------------- | -------- | --------------------------------------------------------- |
| `--input <path>` | Yes      | Path to the input video file (any format FFmpeg supports) |

### Config Loading

`config.json` is automatically loaded from the **current working directory**. There is no override flag. If the file is not found, the CLI exits with an error message and a non-zero exit code.

---

## 4. Configuration File (`config.json`)

### Schema

```json
{
  "model_path": "models/mdxnet/UVR-MDX-NET-Voc_FT.onnx",
  "output_dir": "./output",
  "execution_provider": "cpu",
  "chunk_size": 261120
}
```

### Fields

| Field                | Type    | Required | Description                                                                |
| -------------------- | ------- | -------- | -------------------------------------------------------------------------- |
| `model_path`         | string  | Yes      | Relative or absolute path to the `.onnx` model file                        |
| `output_dir`         | string  | Yes      | Directory where the processed video is saved. Created if it does not exist |
| `execution_provider` | string  | Yes      | `"cpu"` or `"cuda"` (GPU via ONNX Runtime)                                 |
| `chunk_size`         | integer | Yes      | Number of audio samples per inference chunk (e.g. `261120`)                |

---

## 5. Processing Pipeline

### Step 1 — Validation

- Verify `config.json` exists in the current working directory → error and exit if not
- Verify model file at `model_path` exists → error and exit if not
- Verify input video file path exists → error and exit if not
- Load and parse `model_data.json` co-located with the model file to retrieve inference parameters for the configured model

### Step 2 — Probe Audio Track

- Use FFmpeg bindings to probe whether the input video has an audio stream
- **If no audio track is present:**
  - Copy the input file verbatim to `output_dir` with the `_no_music` suffix and the original file extension
  - Print an informational message
  - Exit successfully (exit code 0)

### Step 3 — Extract Audio

- Use FFmpeg to demux the audio stream from the video into a **WAV file**
- Preserve the **original sample rate** and **channel count**
- Write to the system temp directory (`std::env::temp_dir()`)
- Temp filename pattern: `<uuid>_extracted.wav`

### Step 4 — ONNX Inference (Vocals Separation)

- Load the ONNX model via the `ort` crate using the execution provider from config
- Read inference parameters from `model_data.json` for the active model:
  - `mdx_dim_f_set` — frequency dimension
  - `mdx_dim_t_set` — time dimension
  - `mdx_n_fft_scale_set` — FFT window size
  - `compensate` — output gain compensation factor
- Audio preprocessing:
  - Resample audio to **44100 Hz** (MDX-Net requirement) if different from the original
  - Convert to **mono float32** for inference
- Chunk processing loop (chunk size from config):
  - Apply **Short-Time Fourier Transform (STFT)** with the model's FFT parameters
  - Run the ONNX model forward pass
  - Apply **Inverse STFT (iSTFT)** to reconstruct the time-domain signal
  - Apply the `compensate` gain factor to the chunk output
- The output stem is **Vocals** (keeps speech/singing, removes background music)
- Reconstruct full vocals audio by concatenating processed chunks with overlap-add
- Write the separated vocals audio to the system temp directory as `<uuid>_vocals.wav`
- Preserve the **original sample rate** in the output WAV (resample back if step above resampled)

### Step 5 — Remux Video

- Use FFmpeg to replace the audio stream in the original video:
  - `-c:v copy` — no video re-encoding
  - Map video stream from the original input
  - Map audio stream from `<uuid>_vocals.wav`
  - Encode audio to match original audio codec where required by the container format, otherwise use `-c:a copy` if WAV is compatible
- Output file path: `{output_dir}/{original_stem}_no_music.{original_extension}`
  - Example: `output/interview_no_music.mp4`

### Step 6 — Cleanup

- Delete all temp files created in steps 3 and 4: `<uuid>_extracted.wav`, `<uuid>_vocals.wav`
- If cleanup fails for any file, print a warning but do **not** fail the process

---

## 6. Progress Reporting

Display stage-by-stage status to stdout using the `indicatif` crate:

```
[1/5] Validating inputs...         ✓
[2/5] Extracting audio...          ✓
[3/5] Running inference...         ██████████░░░░░ 67%
[4/5] Remuxing video...            ✓
[5/5] Cleaning up...               ✓

Done → output/interview_no_music.mp4
```

- Stages 1, 2, 4, and 5 display a spinner that resolves to `✓` on completion
- Stage 3 (inference) shows a percentage progress bar updated per processed chunk
- All progress output goes to **stderr** so stdout remains clean for scripting

---

## 7. Error Handling

| Condition                                | Behavior                                           |
| ---------------------------------------- | -------------------------------------------------- |
| `config.json` not found in CWD           | Print descriptive error, exit code 1               |
| `config.json` is malformed JSON          | Print parse error with field context, exit code 1  |
| Model `.onnx` file not found             | Print descriptive error, exit code 1               |
| `model_data.json` not found or malformed | Print descriptive error, exit code 1               |
| Model not listed in `model_data.json`    | Print descriptive error, exit code 1               |
| Input video file not found               | Print descriptive error, exit code 1               |
| Video has no audio track                 | Copy file to output dir, print notice, exit code 0 |
| `output_dir` cannot be created           | Print OS error, exit code 1                        |
| FFmpeg extraction failure                | Print FFmpeg stderr output, exit code 1            |
| ONNX Runtime load/inference failure      | Print runtime error, exit code 1                   |
| FFmpeg remux failure                     | Print FFmpeg stderr output, exit code 1            |
| Temp file cleanup failure                | Print warning, continue, exit code 0               |

All error messages are printed to **stderr**.

---

## 8. Architecture & Module Layout

```
music-separator/
├── config.json                  # User configuration (not committed)
├── Cargo.toml
├── src/
│   ├── main.rs                  # CLI entry point: arg parsing, config loading, pipeline dispatch
│   ├── config.rs                # Config struct, JSON deserialization, validation
│   ├── pipeline.rs              # Orchestrates the full 5-step pipeline, progress reporting
│   ├── ffmpeg.rs                # Audio extraction, video probe, video remux via FFmpeg bindings
│   ├── inference.rs             # ONNX model loading, STFT/iSTFT, chunk inference loop
│   ├── model_data.rs            # model_data.json deserialization, model parameter lookup
│   └── error.rs                 # Unified error enum (thiserror), context wrappers
└── models/
    └── mdxnet/
        ├── model_data.json
        └── UVR-MDX-NET-Voc_FT.onnx
```

### Data Flow

```
Input video
    │
    ▼
[ffmpeg.rs] probe audio → no audio → copy to output_dir
    │
    ▼
[ffmpeg.rs] extract audio → <uuid>_extracted.wav (original sample rate)
    │
    ▼
[inference.rs] load ONNX model + model_data params
    │   resample to 44100 Hz (if needed), convert to mono float32
    │   chunk loop: STFT → ONNX forward → iSTFT → compensate
    │   resample back to original sample rate
    ▼
<uuid>_vocals.wav
    │
    ▼
[ffmpeg.rs] remux: original video stream + vocals WAV → output video
    │
    ▼
[pipeline.rs] cleanup temp files
```

---

## 9. Key Dependencies (`Cargo.toml`)

| Crate                  | Version (approx) | Purpose                                    |
| ---------------------- | ---------------- | ------------------------------------------ |
| `clap`                 | 4.x              | CLI argument parsing (`derive` feature)    |
| `serde` + `serde_json` | 1.x              | JSON config and model_data deserialization |
| `ort`                  | 2.x              | ONNX Runtime Rust bindings (CPU + CUDA EP) |
| `ffmpeg-next`          | 7.x              | FFmpeg bindings for probe, extract, remux  |
| `rustfft`              | 6.x              | STFT / iSTFT implementation                |
| `hound`                | 3.x              | WAV file read/write                        |
| `indicatif`            | 0.17.x           | Progress bars and spinners                 |
| `uuid`                 | 1.x              | Temp file unique naming (`v4` feature)     |
| `thiserror`            | 1.x              | Ergonomic error type definitions           |
| `anyhow`               | 1.x              | Error context propagation in `main`        |

---

## 10. Output Naming Convention

| Input           | Output                                |
| --------------- | ------------------------------------- |
| `interview.mp4` | `{output_dir}/interview_no_music.mp4` |
| `clip.mkv`      | `{output_dir}/clip_no_music.mkv`      |
| `recording.mov` | `{output_dir}/recording_no_music.mov` |

The original file extension is always preserved. The output directory is created automatically if it does not exist.

---

## 11. Inference Details

The MDX-Net models operate on spectrogram representations. The following process must be implemented in `inference.rs`:

1. **Resample** input audio to 44100 Hz using linear or sinc interpolation
2. **Mix to mono** by averaging channels
3. **Normalize** signal to float32 range `[-1.0, 1.0]`
4. **STFT** using:
   - FFT size: `mdx_n_fft_scale_set`
   - Hop length: `mdx_n_fft_scale_set / 4`
   - Window: Hann window
5. **Reshape** spectrogram to model input shape `[1, 4, mdx_dim_f_set, mdx_dim_t_set]`
6. **Run ONNX forward pass** — output is the vocals spectrogram mask
7. **iSTFT** to reconstruct time-domain vocals signal
8. **Apply compensate** multiplier to output waveform
9. **Resample back** to original sample rate if resampling occurred in step 1
10. **Reconstruct stereo** if original was stereo (apply the mono separation mask to each channel independently or duplicate mono output)

---

## 12. Testing Plan

### Unit Tests

| Module          | Test Cases                                                                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.rs`     | Valid config parses correctly; missing required fields return error; unknown execution_provider is rejected                                                   |
| `model_data.rs` | Known model hash resolves to correct parameters; unknown model returns descriptive error                                                                      |
| `inference.rs`  | STFT round-trip (STFT then iSTFT on a sine wave produces near-identical output); chunk boundary handling produces no audible clicks (overlap-add correctness) |
| `ffmpeg.rs`     | Probe returns `false` for a video-only file; extract produces a valid WAV; remux produces a file with the correct number of streams                           |
| `error.rs`      | All error variants format correctly to human-readable strings                                                                                                 |

### Integration Tests

| Scenario                          | Expected Result                                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| Valid video with audio track      | Output video in `output_dir` with `_no_music` suffix, vocals-only audio, identical video stream |
| Video with no audio track         | Input video copied verbatim to `output_dir` with `_no_music` suffix, exit code 0                |
| Missing input file                | Descriptive error on stderr, exit code 1, no output files created                               |
| Missing model file                | Descriptive error on stderr, exit code 1, no temp files left on disk                            |
| Missing `config.json`             | Descriptive error on stderr, exit code 1                                                        |
| CUDA provider on CPU-only machine | Graceful fallback error message or automatic fallback to CPU (document behavior explicitly)     |

### Manual / End-to-End Tests

- Process a known video and verify via listening that background music is reduced
- Verify output video plays correctly in VLC and browser players
- Verify output file size is reasonable (video stream unchanged)
- Test with: `.mp4`, `.mkv`, `.mov`, `.avi` input formats
- Test on Windows, Linux, and macOS

---

## 13. Constraints & Assumptions

- FFmpeg libraries must be installed on the target system (dynamic linking via `ffmpeg-next`)
- ONNX Runtime native library is bundled or linked via `ort`'s download feature
- CUDA execution provider requires CUDA and cuDNN installed on the system; the CLI does not validate this beyond the ONNX Runtime error
- The model used (`UVR-MDX-NET-Voc_FT.onnx`) outputs the **Vocals** primary stem as its direct output
- Audio is always converted to 44100 Hz internally for MDX-Net inference regardless of original sample rate; the output WAV is resampled back before remux
- The temp directory is the OS default (`std::env::temp_dir()`); no configuration is exposed for it
- Temp files are always cleaned up after processing, even on error (use Rust's `Drop` or explicit cleanup in error paths)
