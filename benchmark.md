# Benchmark

End-to-end wall-time measurements for the GPU docker image
(`music-separator:gpu`, built from `Dockerfile.gpu`) running against
`tests/raw_vid.mp4`.

Driven by [`scripts/bench.ps1`](scripts/bench.ps1):

```powershell
./scripts/bench.ps1 -Iterations 3 -WarmupIterations 1
```

## Environment

| Item            | Value                                                    |
| --------------- | -------------------------------------------------------- |
| Date            | 2026-05-07                                               |
| GPU             | NVIDIA RTX A3000 Laptop GPU (6144 MiB)                   |
| NVIDIA driver   | 591.86                                                   |
| Host OS         | Windows 11 Pro 26200                                     |
| Container base  | `nvidia/cuda:12.6.3-cudnn` (Ubuntu 24.04)                |
| cuDNN (in-image)| 9.5.1                                                    |
| ONNX Runtime    | `ort 2.0.0-rc.12` with CUDA EP (graph opt level 4 — max) |
| Docker          | 29.4.1                                                   |

## Workload

| Item                    | Value                                                       |
| ----------------------- | ----------------------------------------------------------- |
| Input file              | `tests/raw_vid.mp4`                                         |
| Duration                | 2648.073 s (≈ 44 min 8 s)                                   |
| Video                   | H.264                                                       |
| Audio                   | AAC, 44.1 kHz, stereo                                       |
| Total audio samples     | 233 560 064                                                 |
| Model                   | `UVR-MDX-NET-Voc_FT.onnx` (vocals, primary_stem)            |
| Chunks                  | 894                                                         |
| `chunk_size`            | 261 120 samples (= `1024 × ((1 << 8) − 1)`)                 |
| `fft_size` / `hop`      | 7680 / 1024                                                 |
| Config                  | built-in `/app/config.json` (= `docker/config.cuda.json`)   |
| Tuning block            | omitted (ORT and CUDA EP defaults preserved)                |

## Configuration under test

```json
{
  "model_path": "/app/models/mdxnet/UVR-MDX-NET-Voc_FT.onnx",
  "output_dir": "/out",
  "execution_provider": "cuda"
}
```

No `tuning` block, so ORT defaults apply: `graph_optimization_level = Level3`
(reported as level 4 in the session log = `ALL`), `enable_mem_pattern = 1`,
`arena_extend_strategy = next_power_of_two`, auto thread-pool sizing,
`cudnn_conv_algo_search = exhaustive` (ORT default for CUDA EP).

## Results — full pipeline wall time

3 timed iterations after 1 warmup.

| Run     | Wall time (s) |
| ------- | ------------: |
| warmup  |       232.896 |
| run 1   |       228.183 |
| run 2   |       232.212 |
| run 3   |       233.492 |

| Statistic | Value (s) |
| --------- | --------: |
| mean      |   231.295 |
| min       |   228.183 |
| max       |   233.492 |
| stddev    |     2.771 |

**Real-time factor: ≈ 11.45×** (2648.073 s of audio processed in 231.295 s
of wall time). On this hardware/config the pipeline runs ~11× faster than
real time end-to-end (probe + extract + inference + remux).

## Peak GPU memory

Peak BFC arena allocation reported by ORT during inference (CUDA + CudaPinned):

- Session init: ~133 MiB
- Chunk-loop steady state: ~2.28 GiB (4 arena extensions reaching 1 GiB chunks)

Headroom on the 6144 MiB A3000 is comfortable; the model fits without
needing `tuning.cuda.gpu_mem_limit_mb`.

## Notes

- **Step-level timings.** Per-step wall times are reported by `indicatif`
  spinners (e.g. `[4/5] Running inference... ✓ (Xs)`) which write to stderr
  only when stderr is a TTY. The bench script captures stdout+stderr to a
  file, so those `✓ (Xs)` lines are suppressed by indicatif's own non-TTY
  fallback. Aggregating step-4 (inference) wall time independently is
  tracked in [`specs/optimization-todo.md`](specs/optimization-todo.md)
  section 1.
- Run-to-run wall time on the same image+config is stable (stddev ≈ 1.2%
  of mean). This is the baseline against which `tuning` overrides should
  be A/B'd via `bench.ps1 -Config <override.json>`.

## Reproducing

```powershell
# Default: 3 timed runs, 1 warmup, against the image's baked-in config.cuda.json
./scripts/bench.ps1

# Build the image first
./scripts/bench.ps1 -Build

# Compare a tuning override:
./scripts/bench.ps1 -Config ./bench-configs/tf32-on.json -Iterations 5
```

Per-run logs land in `out/bench/`; the latest log path is printed at the
end of the script.