// DSP primitives + Separator (ONNX session + chunk loop).
//
// Hot-path design notes
// ─────────────────────
// The chunk loop in `Separator::separate_vocals` avoids per-chunk allocations:
//   * FFT plans (forward + inverse) and their scratch buffers are built once.
//   * The padded-input, FFT, tensor-input, output-spectrum, OLA accumulator
//     and chunk-time buffers are owned by `Separator` and reused.
//   * STFT writes directly into the ONNX tensor layout — no `Vec<Vec<...>>`.
//   * The ORT input tensor is a borrowed view (`TensorRef::from_array_view`)
//     of the reusable input buffer; ownership is not transferred per call.
//   * Outer overlap-add accumulates straight into the final output buffer;
//     the previous `output_chunks: Vec<Vec<f32>>` staging is gone.

use std::path::Path;
use std::sync::Arc;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use crate::config::{Config, ExecutionProvider};
use crate::model_data::ModelParams;
use crate::error::AppError;

/// MDX-Net STFT hop length (samples). Fixed by the model architecture: every
/// shipped MDX-Net model uses a 1024-sample hop, regardless of `mdx_n_fft_scale_set`.
/// The chunk length fed to the model is then `HOP_LENGTH * (n_time_model - 1)`,
/// where `n_time_model = 1 << params.mdx_dim_t_set`.
const HOP_LENGTH: usize = 1024;

/// Symmetric Hann window of length `size`.
///
/// Formula: `w[i] = 0.5 * (1 - cos(2π·i / (size-1)))` for `i in 0..size`.
///
/// Notes
/// - size == 0  → empty Vec
/// - size == 1  → `[1.0]` (degenerate case; denominator would be zero)
pub fn hann_window(size: usize) -> Vec<f32> {
    match size {
        0 => vec![],
        1 => vec![1.0],
        n => {
            let denom = (n - 1) as f32;
            (0..n)
                .map(|i| {
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / denom).cos())
                })
                .collect()
        }
    }
}

/// Short-Time Fourier Transform (allocating reference implementation, used by tests).
///
/// The hot path uses `Separator`'s reusable buffers instead.
#[allow(dead_code)]
pub fn stft(
    signal: &[f32],
    fft_size: usize,
    hop_length: usize,
    window: &[f32],
) -> Vec<Vec<[f32; 2]>> {
    let pad = fft_size / 2;
    let mut padded = vec![0.0_f32; pad + signal.len() + pad];
    padded[pad..pad + signal.len()].copy_from_slice(signal);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut scratch = vec![Complex::<f32> { re: 0.0, im: 0.0 }; fft.get_inplace_scratch_len()];
    let mut buf = vec![Complex::<f32> { re: 0.0, im: 0.0 }; fft_size];

    let n_bins = fft_size / 2 + 1;
    let mut frames = Vec::new();
    let mut pos = 0;

    while pos + fft_size <= padded.len() {
        for i in 0..fft_size {
            buf[i] = Complex { re: padded[pos + i] * window[i], im: 0.0 };
        }
        fft.process_with_scratch(&mut buf, &mut scratch);

        let mut frame = Vec::with_capacity(n_bins);
        for c in &buf[..n_bins] {
            frame.push([c.re, c.im]);
        }
        frames.push(frame);

        pos += hop_length;
    }

    frames
}

/// Inverse Short-Time Fourier Transform (allocating reference implementation, used by tests).
#[allow(dead_code)]
pub fn istft(
    frames: &[Vec<[f32; 2]>],
    fft_size: usize,
    hop_length: usize,
    window: &[f32],
    signal_length: usize,
) -> Vec<f32> {
    let n_bins = fft_size / 2 + 1;

    let ola_length = if frames.is_empty() {
        fft_size
    } else {
        (frames.len() - 1) * hop_length + fft_size
    };
    let buf_len = ola_length.max(fft_size / 2 + signal_length);

    let mut output = vec![0.0_f32; buf_len];
    let mut norm = vec![0.0_f32; buf_len];

    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(fft_size);
    let mut scratch = vec![Complex::<f32> { re: 0.0, im: 0.0 }; ifft.get_inplace_scratch_len()];
    let mut buf = vec![Complex::<f32> { re: 0.0, im: 0.0 }; fft_size];

    let inv_n = 1.0_f32 / fft_size as f32;

    for (frame_idx, frame) in frames.iter().enumerate() {
        let frame_start = frame_idx * hop_length;

        for c in buf.iter_mut() {
            *c = Complex { re: 0.0, im: 0.0 };
        }
        for i in 0..n_bins {
            buf[i] = Complex { re: frame[i][0], im: frame[i][1] };
        }
        for i in n_bins..fft_size {
            let j = fft_size - i;
            buf[i] = Complex { re: frame[j][0], im: -frame[j][1] };
        }

        ifft.process_with_scratch(&mut buf, &mut scratch);

        for (i, c) in buf.iter().enumerate() {
            let sample = c.re * inv_n * window[i];
            output[frame_start + i] += sample;
            norm[frame_start + i] += window[i] * window[i];
        }
    }

    for (o, &n) in output.iter_mut().zip(norm.iter()) {
        if n > 1e-8 {
            *o /= n;
        }
    }

    let trim_start = fft_size / 2;
    let trim_end = trim_start + signal_length;

    if trim_end <= output.len() {
        output[trim_start..trim_end].to_vec()
    } else {
        let mut result = output[trim_start..].to_vec();
        result.resize(signal_length, 0.0);
        result
    }
}

/// Resample `signal` from `from_rate` Hz to `to_rate` Hz using linear interpolation.
pub fn resample(signal: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return signal.to_vec();
    }
    if signal.is_empty() {
        return Vec::new();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (signal.len() as f64 / ratio).round() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_i = src_pos.floor() as usize;
        let frac = src_pos - src_pos.floor();
        let clamped_next = (src_i + 1).min(signal.len() - 1);
        output.push(signal[src_i] * (1.0 - frac) as f32 + signal[clamped_next] * frac as f32);
    }

    output
}

/// Mix a multi-channel interleaved `signal` down to mono by averaging each frame.
pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }

    let c = channels as usize;
    let output_len = samples.len() / c;
    let inv_c = 1.0_f32 / channels as f32;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let sum: f32 = samples[i * c..i * c + c].iter().sum();
        output.push(sum * inv_c);
    }

    output
}

/// Interleave `left` and `right` channels into a single stereo buffer `[L0, R0, L1, R1, ...]`.
pub fn interleave_stereo(left: &[f32], right: &[f32]) -> Vec<f32> {
    let n = left.len().min(right.len());
    let mut output = Vec::with_capacity(2 * n);
    for i in 0..n {
        output.push(left[i]);
        output.push(right[i]);
    }
    output
}

/// Normalise `signal` so the peak absolute value is 1.0.
pub fn normalize(signal: &[f32]) -> (Vec<f32>, f32) {
    let max_abs = signal.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if max_abs < 1e-8 {
        return (signal.to_vec(), 1.0);
    }
    let normalized = signal.iter().map(|x| x / max_abs).collect();
    (normalized, max_abs)
}

/// Undo a previous [`normalize`] call by multiplying each sample by `scale`.
pub fn denormalize(signal: &[f32], scale: f32) -> Vec<f32> {
    signal.iter().map(|x| x * scale).collect()
}

// ── Separator ─────────────────────────────────────────────────────────────────

/// Wraps an ONNX Runtime session, the MDX-Net params, and reusable DSP buffers.
pub struct Separator {
    session: ort::session::Session,
    params: ModelParams,
    chunk_size: usize,

    // Model-derived dims.
    fft_size: usize,
    hop_length: usize,
    n_time_model: usize,
    n_bins_model: usize,
    n_bins_full: usize,

    // FFT plans + scratch — built once, reused per chunk/per frame.
    forward_fft: Arc<dyn Fft<f32>>,
    inverse_fft: Arc<dyn Fft<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    fft_buf: Vec<Complex<f32>>,

    // Precomputed window + its OLA energy (norm = sum_t window[i - t*hop]^2).
    window: Vec<f32>,
    istft_norm: Vec<f32>,

    // Reusable per-chunk buffers.
    padded_chunk: Vec<f32>,            // pad + chunk_size + pad, FFT-pad zeros stay 0
    tensor_input: Vec<f32>,            // 4 * n_bins_model * n_time_model
    out_real: Vec<f32>,                // n_bins_model * n_time_model
    out_imag: Vec<f32>,                // n_bins_model * n_time_model
    istft_accum: Vec<f32>,             // (n_time_model-1)*hop + fft_size
    chunk_time: Vec<f32>,              // chunk_size
}

impl std::fmt::Debug for Separator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Separator")
            .field("params", &self.params)
            .field("chunk_size", &self.chunk_size)
            .field("fft_size", &self.fft_size)
            .field("hop_length", &self.hop_length)
            .field("n_time_model", &self.n_time_model)
            .field("n_bins_model", &self.n_bins_model)
            .finish()
    }
}

impl Separator {
    /// Create a new `Separator` by building an ORT session from `config` and
    /// pre-allocating every reusable buffer the chunk loop needs.
    pub fn new(config: &Config, params: ModelParams) -> Result<Self, AppError> {
        tracing::debug!(
            model_path = %config.model_path.display(),
            execution_provider = ?config.execution_provider,
            "building ONNX session"
        );

        let builder = ort::session::Session::builder()
            .map_err(|e| AppError::OnnxLoad(e.to_string()))?;

        let mut builder = match config.execution_provider {
            ExecutionProvider::Cuda => builder
                .with_execution_providers([
                    ort::ep::CUDA::default().build().error_on_failure(),
                ])
                .map_err(|e| AppError::OnnxLoad(e.to_string()))?,
            ExecutionProvider::Cpu => builder,
        };

        let session = builder
            .commit_from_file(&config.model_path)
            .map_err(|e| AppError::OnnxLoad(e.to_string()))?;

        let model_size = std::fs::metadata(&config.model_path)
            .map(|m| m.len())
            .ok();
        tracing::info!(
            execution_provider = ?config.execution_provider,
            model = %config.model_path.display(),
            model_bytes = ?model_size,
            "ONNX session initialised"
        );

        // ── Pre-compute model-derived DSP dims and reusable buffers ──────────
        let fft_size = params.mdx_n_fft_scale_set;
        let hop_length = HOP_LENGTH;
        let n_time_model = 1usize << params.mdx_dim_t_set;
        let n_bins_model = params.mdx_dim_f_set;
        let n_bins_full = fft_size / 2 + 1;
        // chunk_size is fully determined by the model: feeding any other value
        // would either zero-pad the time dimension of the input tensor (quality
        // hit) or truncate the chunk's audio (waste). Always derive it.
        let chunk_size = hop_length * (n_time_model - 1);

        let mut planner = FftPlanner::<f32>::new();
        let forward_fft = planner.plan_fft_forward(fft_size);
        let inverse_fft = planner.plan_fft_inverse(fft_size);
        let scratch_len = forward_fft
            .get_inplace_scratch_len()
            .max(inverse_fft.get_inplace_scratch_len());
        let fft_scratch = vec![Complex::<f32> { re: 0.0, im: 0.0 }; scratch_len];
        let fft_buf = vec![Complex::<f32> { re: 0.0, im: 0.0 }; fft_size];

        let window = hann_window(fft_size);

        let pad = fft_size / 2;
        let padded_chunk = vec![0.0_f32; pad + chunk_size + pad];

        let plane = n_bins_model * n_time_model;
        let tensor_input = vec![0.0_f32; 4 * plane];
        let out_real = vec![0.0_f32; plane];
        let out_imag = vec![0.0_f32; plane];

        let accum_len = (n_time_model.saturating_sub(1)) * hop_length + fft_size;
        let accum_len = accum_len.max(fft_size / 2 + chunk_size);
        let istft_accum = vec![0.0_f32; accum_len];

        // Precompute the OLA window-energy norm: norm[i] = Σ_t window[i - t*hop]^2.
        let mut istft_norm = vec![0.0_f32; accum_len];
        for t in 0..n_time_model {
            let frame_start = t * hop_length;
            for i in 0..fft_size {
                if frame_start + i < istft_norm.len() {
                    istft_norm[frame_start + i] += window[i] * window[i];
                }
            }
        }

        let chunk_time = vec![0.0_f32; chunk_size];

        Ok(Self {
            session,
            params,
            chunk_size,
            fft_size,
            hop_length,
            n_time_model,
            n_bins_model,
            n_bins_full,
            forward_fft,
            inverse_fft,
            fft_scratch,
            fft_buf,
            window,
            istft_norm,
            padded_chunk,
            tensor_input,
            out_real,
            out_imag,
            istft_accum,
            chunk_time,
        })
    }

    /// Run a single inference pass with all-zero input. Lets the EP perform any
    /// first-call kernel selection / cuDNN heuristic search before the timed loop.
    fn warmup(&mut self) -> Result<(), AppError> {
        for x in self.tensor_input.iter_mut() {
            *x = 0.0;
        }
        let shape = [1usize, 4, self.n_bins_model, self.n_time_model];
        let tensor_view: &[f32] = &self.tensor_input;
        let tensor = ort::value::TensorRef::<f32>::from_array_view((shape, tensor_view))
            .map_err(|e| AppError::OnnxInference(e.to_string()))?;
        let _outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(|e| AppError::OnnxInference(e.to_string()))?;
        Ok(())
    }

    /// Compute STFT of the current padded chunk and pack it directly into
    /// `tensor_input` in the ONNX `[1, 4, n_bins_model, n_time_model]` layout.
    /// Channels 0/1 = real/imag for input ch0; channels 2/3 = duplicate for ch1.
    fn stft_into_tensor(&mut self) {
        let plane = self.n_bins_model * self.n_time_model;
        for x in self.tensor_input.iter_mut() {
            *x = 0.0;
        }

        let mut pos = 0usize;
        for t in 0..self.n_time_model {
            if pos + self.fft_size > self.padded_chunk.len() {
                break;
            }
            for i in 0..self.fft_size {
                self.fft_buf[i] = Complex {
                    re: self.padded_chunk[pos + i] * self.window[i],
                    im: 0.0,
                };
            }
            self.forward_fft
                .process_with_scratch(&mut self.fft_buf, &mut self.fft_scratch);

            for f in 0..self.n_bins_model {
                let c = self.fft_buf[f];
                let idx = f * self.n_time_model + t;
                self.tensor_input[idx] = c.re;
                self.tensor_input[plane + idx] = c.im;
                self.tensor_input[2 * plane + idx] = c.re;
                self.tensor_input[3 * plane + idx] = c.im;
            }

            pos += self.hop_length;
        }
    }

    /// Inverse-STFT from `out_real` / `out_imag` into `chunk_time`.
    fn istft_chunk(&mut self) {
        for s in self.istft_accum.iter_mut() {
            *s = 0.0;
        }

        let inv_n = 1.0_f32 / self.fft_size as f32;

        for t in 0..self.n_time_model {
            // Reconstruct full conjugate-symmetric spectrum:
            //   bins 0..n_bins_full from model output; bins ≥ n_bins_model are zero;
            //   bins n_bins_full..fft_size are conjugate mirrors.
            for c in self.fft_buf.iter_mut() {
                *c = Complex { re: 0.0, im: 0.0 };
            }
            for f in 0..self.n_bins_model {
                let idx = f * self.n_time_model + t;
                self.fft_buf[f] = Complex {
                    re: self.out_real[idx],
                    im: self.out_imag[idx],
                };
            }
            for i in 1..self.n_bins_full - 1 {
                let mirror = self.fft_size - i;
                let c = self.fft_buf[i];
                self.fft_buf[mirror] = Complex { re: c.re, im: -c.im };
            }

            self.inverse_fft
                .process_with_scratch(&mut self.fft_buf, &mut self.fft_scratch);

            let frame_start = t * self.hop_length;
            for i in 0..self.fft_size {
                let s = self.fft_buf[i].re * inv_n * self.window[i];
                self.istft_accum[frame_start + i] += s;
            }
        }

        // OLA normalise using the precomputed window-energy denominator.
        for (a, &n) in self.istft_accum.iter_mut().zip(self.istft_norm.iter()) {
            if n > 1e-8 {
                *a /= n;
            }
        }

        // Trim the centre-pad offset and copy into chunk_time.
        let trim_start = self.fft_size / 2;
        for i in 0..self.chunk_size {
            let idx = trim_start + i;
            self.chunk_time[i] = if idx < self.istft_accum.len() {
                self.istft_accum[idx]
            } else {
                0.0
            };
        }
    }

    /// Separate vocals from `wav_path` and write the result to `output_path`.
    ///
    /// `progress_cb` is called with values in `[0.0, 1.0]` as processing advances.
    pub fn separate_vocals(
        &mut self,
        wav_path: &Path,
        output_path: &Path,
        progress_cb: impl Fn(f32),
    ) -> Result<(), AppError> {
        // ── Step 1: Read WAV ──────────────────────────────────────────────────
        let mut reader = hound::WavReader::open(wav_path)
            .map_err(|e| AppError::AudioRead(e.to_string()))?;
        let spec = reader.spec();
        let orig_sample_rate = spec.sample_rate;
        let orig_channels = spec.channels;

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| s.map_err(|e| AppError::AudioRead(e.to_string())))
                .collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1_i32 << (bits - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| {
                        s.map(|v| v as f32 / max_val)
                            .map_err(|e| AppError::AudioRead(e.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };

        tracing::info!(
            total_samples = samples.len(),
            sample_rate = orig_sample_rate,
            channels = orig_channels,
            "audio loaded for inference"
        );

        // ── Step 2: Mono + resample to 44100 ─────────────────────────────────
        let mono = to_mono(&samples, orig_channels);
        let needs_resample = orig_sample_rate != 44100;
        let resampled = if needs_resample {
            tracing::debug!(
                from = orig_sample_rate,
                to = 44100u32,
                "resampling to model rate"
            );
            resample(&mono, orig_sample_rate, 44100)
        } else {
            mono
        };

        // ── Step 3: Normalize ─────────────────────────────────────────────────
        let (normalized, scale) = normalize(&resampled);

        // ── Step 4: DSP / chunking parameters ────────────────────────────────
        let chunk_size = self.chunk_size;
        let outer_hop = chunk_size / 2;
        let pad = self.fft_size / 2;

        // Number of chunks (matches the legacy chunking semantics: any tail
        // shorter than chunk_size still produces one final chunk, zero-padded).
        let total_chunks = if normalized.is_empty() {
            0
        } else if normalized.len() <= chunk_size {
            1
        } else {
            // Number of full hops we can take before the chunk window fits
            // entirely past the signal end.
            let extra = normalized.len() - chunk_size;
            extra.div_ceil(outer_hop) + 1
        };

        tracing::info!(
            total_chunks,
            chunk_size = self.chunk_size,
            fft_size = self.fft_size,
            hop_length = self.hop_length,
            "chunked input; running inference"
        );

        // ── Step 5: Final-output buffers ─────────────────────────────────────
        let output_len = normalized.len();
        let mut full_output = vec![0.0_f32; output_len];
        let mut weight = vec![0.0_f32; output_len];

        // ── Step 6: Warmup pass (skip when no real chunks would run) ─────────
        if total_chunks > 0 {
            self.warmup()?;
        }

        // Throttle progress callback: at most ~100 updates over the whole run,
        // plus one update on the final chunk.
        let progress_every = (total_chunks / 100).max(1);

        // ── Step 7: Streaming chunk loop ─────────────────────────────────────
        let mut start = 0usize;
        let mut idx = 0usize;
        while start < normalized.len() {
            let end = (start + chunk_size).min(normalized.len());
            let usable = end - start;

            // Load chunk into reusable padded buffer; FFT-pad zeros stay zero.
            self.padded_chunk[pad..pad + usable]
                .copy_from_slice(&normalized[start..end]);
            if usable < chunk_size {
                for s in &mut self.padded_chunk[pad + usable..pad + chunk_size] {
                    *s = 0.0;
                }
            }

            // a) STFT directly into the tensor input layout.
            self.stft_into_tensor();

            // b) ONNX forward pass — borrow tensor_input as a TensorRef view.
            //    We split-borrow `self`: tensor_input (immut) and session (mut)
            //    are disjoint fields, which the borrow checker accepts.
            {
                let shape = [1usize, 4, self.n_bins_model, self.n_time_model];
                let tensor_view: &[f32] = &self.tensor_input;
                let tensor =
                    ort::value::TensorRef::<f32>::from_array_view((shape, tensor_view))
                        .map_err(|e| AppError::OnnxInference(e.to_string()))?;

                let outputs = self
                    .session
                    .run(ort::inputs![tensor])
                    .map_err(|e| AppError::OnnxInference(e.to_string()))?;

                let (_shape, output_flat) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| AppError::OnnxInference(e.to_string()))?;

                // Copy out channel-0 real / imag into our flat buffers.
                let plane = self.n_bins_model * self.n_time_model;
                if output_flat.len() < 2 * plane {
                    return Err(AppError::OnnxInference(format!(
                        "model output too small: got {} floats, expected at least {}",
                        output_flat.len(),
                        2 * plane
                    )));
                }
                self.out_real.copy_from_slice(&output_flat[..plane]);
                self.out_imag.copy_from_slice(&output_flat[plane..2 * plane]);
            }

            // c) iSTFT into the chunk_time buffer.
            self.istft_chunk();

            // d) Outer overlap-add directly into the final output.
            let copy_len = self.chunk_time.len().min(output_len.saturating_sub(start));
            for i in 0..copy_len {
                full_output[start + i] += self.chunk_time[i];
                weight[start + i] += 1.0;
            }

            tracing::trace!(
                chunk_index = idx + 1,
                of = total_chunks,
                samples = usable,
                "processed chunk"
            );

            if idx % progress_every == 0 {
                progress_cb(idx as f32 / total_chunks.max(1) as f32);
            }

            if start + chunk_size >= normalized.len() {
                break;
            }
            start += outer_hop;
            idx += 1;
        }

        // Outer OLA normalisation.
        for i in 0..output_len {
            if weight[i] > 0.0 {
                full_output[i] /= weight[i];
            }
        }

        // ── Step 8: Compensate, denormalize, optionally resample back ─────────
        for s in full_output.iter_mut() {
            *s *= self.params.compensate;
        }

        let mut final_output = denormalize(&full_output, scale);

        if needs_resample {
            final_output = resample(&final_output, 44100, orig_sample_rate);
        }

        // ── Step 9: Restore stereo if needed, write WAV ───────────────────────
        let write_samples = if orig_channels == 2 {
            interleave_stereo(&final_output, &final_output)
        } else {
            final_output
        };

        let out_spec = hound::WavSpec {
            channels: orig_channels,
            sample_rate: orig_sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(output_path, out_spec)
            .map_err(|e| AppError::AudioWrite(e.to_string()))?;
        for s in &write_samples {
            writer
                .write_sample(*s)
                .map_err(|e| AppError::AudioWrite(e.to_string()))?;
        }
        writer
            .finalize()
            .map_err(|e| AppError::AudioWrite(e.to_string()))?;

        progress_cb(1.0);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_values() {
        let w = hann_window(4);
        let expected = [0.0_f32, 0.75, 0.75, 0.0];
        for (a, &b) in w.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
    }

    #[test]
    fn stft_round_trip_sine_wave() {
        let sample_rate = 44100usize;
        let freq = 440.0_f32;
        let signal: Vec<f32> = (0..sample_rate)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let fft_size = 2048;
        let hop_length = fft_size / 4;
        let window = hann_window(fft_size);

        let frames = stft(&signal, fft_size, hop_length, &window);
        let reconstructed = istft(&frames, fft_size, hop_length, &window, signal.len());

        assert_eq!(reconstructed.len(), signal.len());
        let max_err = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 0.01, "max absolute error: {max_err}");
    }

    #[test]
    fn stft_zero_signal_returns_zero_frames() {
        let signal = vec![0.0_f32; 1024];
        let fft_size = 512;
        let hop = fft_size / 4;
        let window = hann_window(fft_size);
        let frames = stft(&signal, fft_size, hop, &window);
        assert!(!frames.is_empty());
        for frame in &frames {
            for &[re, im] in frame {
                assert!((re).abs() < 1e-6, "expected zero real, got {re}");
                assert!((im).abs() < 1e-6, "expected zero imag, got {im}");
            }
        }
    }

    #[test]
    fn istft_output_length_equals_signal_length() {
        let signal = vec![1.0_f32; 2048];
        let fft_size = 512;
        let hop = fft_size / 4;
        let window = hann_window(fft_size);
        let frames = stft(&signal, fft_size, hop, &window);
        let out = istft(&frames, fft_size, hop, &window, signal.len());
        assert_eq!(out.len(), signal.len());
    }

    #[test]
    fn to_mono_stereo_averages_pairs() {
        let samples = vec![1.0_f32, 0.0, 0.0, 1.0];
        let mono = to_mono(&samples, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn resample_downsample_44100_to_22050() {
        let signal: Vec<f32> = (0..44100).map(|i| i as f32 / 44100.0).collect();
        let down = resample(&signal, 44100, 22050);
        assert_eq!(down.len(), 22050);
    }

    #[test]
    fn resample_upsample_22050_to_44100() {
        let signal: Vec<f32> = (0..22050).map(|i| i as f32 / 22050.0).collect();
        let up = resample(&signal, 22050, 44100);
        assert_eq!(up.len(), 44100);
    }

    #[test]
    fn normalize_divides_by_max_abs() {
        let signal = vec![2.0_f32, -4.0, 1.0];
        let (norm, scale) = normalize(&signal);
        assert!((scale - 4.0).abs() < 1e-6);
        assert!((norm[0] - 0.5).abs() < 1e-6);
        assert!((norm[1] - (-1.0)).abs() < 1e-6);
        assert!((norm[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn denormalize_roundtrips_normalize() {
        let signal = vec![1.0_f32, -2.0, 0.5, 0.0, 3.0];
        let (norm, scale) = normalize(&signal);
        let recovered = denormalize(&norm, scale);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn interleave_stereo_produces_lr_pairs() {
        let left = vec![1.0_f32, 2.0];
        let right = vec![3.0_f32, 4.0];
        let out = interleave_stereo(&left, &right);
        assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn separator_new_fails_on_missing_model() {
        use std::path::PathBuf;
        use crate::config::{Config, ExecutionProvider};
        use crate::model_data::ModelParams;

        let config = Config {
            model_path: PathBuf::from("/nonexistent/model.onnx"),
            output_dir: PathBuf::from("./output"),
            execution_provider: ExecutionProvider::Cpu,
        };
        let params = ModelParams {
            compensate: 1.0,
            mdx_dim_f_set: 3072,
            mdx_dim_t_set: 8,
            mdx_n_fft_scale_set: 7680,
            primary_stem: "Vocals".to_string(),
            name: "model.onnx".to_string(),
        };
        let result = Separator::new(&config, params);
        assert!(
            matches!(result, Err(AppError::OnnxLoad(_))),
            "expected OnnxLoad, got: {result:?}"
        );
    }
}
