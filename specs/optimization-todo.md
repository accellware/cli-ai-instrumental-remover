# Music Separator Optimization TODO

This checklist tracks performance work for model execution in the current GPU container workflow.

## 1. Baseline and Benchmarking Protocol

- [ ] Define the benchmark input set (short, medium, long clips) and keep it fixed for comparisons.
- [ ] Run release-mode baseline with GPU and standard logging (no -vv).
- [ ] Capture metrics per run:
  - [ ] Full pipeline wall time
  - [ ] Step 4 (model inference) wall time
  - [ ] Chunk count
  - [ ] Average chunk latency
  - [ ] P95 chunk latency
  - [ ] Peak memory usage
- [ ] Record environment details for every run (GPU model, driver, CUDA image tag, ORT version).
- [ ] Add a simple benchmark command script for reproducible runs.

## 2. Quick Wins (Low Risk, High Impact)

- [x] Reduce per-chunk debug logging overhead by defaulting chunk-level logs to trace only.
- [x] Keep progress feedback, but throttle update frequency (e.g., every N chunks).
- [x] Avoid expensive formatting in hot loops when logs are disabled. (`tracing::trace!` with field-style args; macro short-circuits when filtered.)
- [ ] Compare runtime with and without verbose logging to quantify logging tax.

## 3. Memory and Allocation Reduction

- [x] Replace full input chunk materialization with streaming chunk iteration.
- [x] Remove output chunk staging vector and overlap-add directly into the final output buffer.
- [x] Reuse tensor input buffer across chunks. (Now passed as a `TensorRef` view, not consumed.)
- [x] Reuse frame/output temporary buffers across chunks. (`fft_buf`, `padded_chunk`, `out_real`, `out_imag`, `istft_accum`, `chunk_time`.)
- [x] Minimize per-iteration Vec allocations in STFT, iSTFT, and tensor packing. (Hot path no longer uses `Vec<Vec<[f32;2]>>`; STFT writes directly into the ONNX tensor layout, iSTFT reads from flat `out_real`/`out_imag` arrays.)
- [ ] Validate no regression in output length and signal continuity after streaming refactor. (Unit suite green, including STFT round-trip; needs an end-to-end ear/level check on a real clip.)

## 4. FFT and DSP Path Optimization

- [x] Reuse FFT plans and reusable scratch buffers for STFT/iSTFT operations. (`forward_fft`/`inverse_fft` are `Arc<dyn Fft<f32>>` built once in `Separator::new`; `fft_scratch` is reused via `process_with_scratch`.)
- [x] Audit windowing and overlap-add loops for avoidable copies. (Padded-chunk leading/trailing zeros are written once at allocation time; `istft_norm` is precomputed once and reused as the OLA denominator.)
- [ ] Benchmark CPU time split: resample, STFT, ORT run, iSTFT, overlap-add.
- [ ] Consider specialized data layout changes only if benchmark shows clear gain.

## 5. ONNX Runtime and CUDA Session Tuning

- [x] Audit current SessionBuilder configuration and enable supported graph optimization settings. (ort 2.0.0-rc.12 defaults to `GraphOptimizationLevel::Level3` per the upstream docstring — no override needed.)
- [ ] Tune thread and execution options for GPU path where applicable.
- [x] Add a warmup inference pass before timed chunk loop. (`Separator::warmup()` runs one zero-input inference to amortise EP/cuDNN first-call cost.)
- [ ] Evaluate IO binding or equivalent path to reduce host-device transfer overhead.
- [ ] Validate provider library loading and keep runtime image dependencies explicit.

## 6. Model Tradeoff Profiles

- [ ] Define profile presets:
  - [ ] Quality-first (current default behavior)
  - [ ] Balanced
  - [ ] Speed-first
- [ ] Evaluate alternative shipped models for throughput vs quality.
- [ ] Document expected speed and quality deltas for each profile.
- [ ] Ensure profile selection preserves existing output naming and pipeline contracts.

## 7. Validation Gates and Acceptance Criteria

- [ ] Acceptance target: at least 30 percent reduction in Step 4 time on fixed benchmark set.
- [ ] Acceptance target: no material degradation in vocal quality on reference clips.
- [ ] Acceptance target: no increase in failure rate across representative inputs.
- [ ] Add regression checklist before merge:
  - [ ] CPU path still works
  - [ ] CUDA path still works
  - [ ] No-audio passthrough behavior unchanged
  - [ ] Temp file cleanup warnings remain non-fatal

## 8. Tracking Table

Fill this table as work lands.

| Date | Commit | Change   | Full Time | Step 4 Time | Avg Chunk | P95 Chunk | Peak Memory | Notes |
| ---- | ------ | -------- | --------- | ----------- | --------- | --------- | ----------- | ----- |
| TBD  | TBD    | Baseline | TBD       | TBD         | TBD       | TBD       | TBD         | TBD   |
