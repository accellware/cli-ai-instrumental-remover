// DSP primitives — pure Rust, no ONNX, no FFmpeg.
// Implemented in Prompt 6.
// Prompt 8 adds the Separator struct and separate_vocals method.

use std::path::Path;
use rustfft::{FftPlanner, num_complex::Complex};
use crate::config::{Config, ExecutionProvider};
use crate::model_data::ModelParams;
use crate::error::AppError;

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

/// Short-Time Fourier Transform.
///
/// Steps
/// 1. Centre-pad `signal` with `fft_size / 2` zeros on each side.
/// 2. Slide a window of width `fft_size` in steps of `hop_length` over the
///    padded signal.
/// 3. For each frame: multiply by `window`, run a forward FFT, keep the
///    one-sided spectrum (`fft_size / 2 + 1` bins).
///
/// Returns `Vec<Vec<[f32; 2]>>` — outer = frames, inner = bins, each bin
/// is `[real, imag]`.
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

    let n_bins = fft_size / 2 + 1;
    let mut frames = Vec::new();
    let mut pos = 0;

    while pos + fft_size <= padded.len() {
        // Apply window and convert to complex.
        let mut buf: Vec<Complex<f32>> = padded[pos..pos + fft_size]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex { re: s * w, im: 0.0 })
            .collect();

        fft.process(&mut buf);

        // Keep only the one-sided (positive-frequency) spectrum.
        frames.push(
            buf[..n_bins]
                .iter()
                .map(|c| [c.re, c.im])
                .collect::<Vec<_>>(),
        );

        pos += hop_length;
    }

    frames
}

/// Inverse Short-Time Fourier Transform.
///
/// Reconstructs the time-domain signal from one-sided STFT frames produced
/// by [`stft`] using overlap-add (OLA) synthesis.
///
/// Steps
/// 1. Allocate an OLA buffer of length `(frames.len()-1)*hop_length + fft_size`
///    (padded to at least `fft_size/2 + signal_length` to handle the trim).
/// 2. For each frame:
///    a. Reconstruct the full conjugate-symmetric complex spectrum.
///    b. Run an inverse FFT and normalise by `1 / fft_size`.
///    c. Multiply by `window` and overlap-add into the output buffer.
///    d. Accumulate `window[i]²` into a normalisation buffer.
/// 3. Divide each output sample by its accumulated window energy (skip
///    positions where the energy is < 1e-8 to avoid division by zero).
/// 4. Strip the centre-padding offset: return
///    `output[fft_size/2 .. fft_size/2 + signal_length]`.
pub fn istft(
    frames: &[Vec<[f32; 2]>],
    fft_size: usize,
    hop_length: usize,
    window: &[f32],
    signal_length: usize,
) -> Vec<f32> {
    let n_bins = fft_size / 2 + 1;

    // OLA buffer covers all frames.
    let ola_length = if frames.is_empty() {
        fft_size
    } else {
        (frames.len() - 1) * hop_length + fft_size
    };

    // Must also be long enough that we can read fft_size/2..fft_size/2+signal_length.
    let buf_len = ola_length.max(fft_size / 2 + signal_length);

    let mut output = vec![0.0_f32; buf_len];
    let mut norm = vec![0.0_f32; buf_len];

    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(fft_size);

    for (frame_idx, frame) in frames.iter().enumerate() {
        let frame_start = frame_idx * hop_length;

        // --- Reconstruct full conjugate-symmetric spectrum ---
        let mut buf = vec![Complex { re: 0.0_f32, im: 0.0_f32 }; fft_size];

        // Positive-frequency bins (0..n_bins) are stored as-is.
        for i in 0..n_bins {
            buf[i] = Complex { re: frame[i][0], im: frame[i][1] };
        }

        // Negative-frequency bins (n_bins..fft_size) are conjugate mirrors.
        // bin[fft_size - i] = conj(bin[i]) for i in 1..n_bins-1
        for i in n_bins..fft_size {
            let j = fft_size - i; // j falls in 1..n_bins-1
            buf[i] = Complex { re: frame[j][0], im: -frame[j][1] };
        }

        // --- IFFT + scale ---
        ifft.process(&mut buf);
        let inv_n = 1.0_f32 / fft_size as f32;

        // --- Window, overlap-add, and accumulate window energy ---
        for (i, c) in buf.iter().enumerate() {
            let sample = c.re * inv_n * window[i];
            output[frame_start + i] += sample;
            norm[frame_start + i] += window[i] * window[i];
        }
    }

    // --- OLA normalisation ---
    for (o, &n) in output.iter_mut().zip(norm.iter()) {
        if n > 1e-8 {
            *o /= n;
        }
    }

    // --- Trim: remove the centre-pad offset added by stft ---
    let trim_start = fft_size / 2;
    let trim_end = trim_start + signal_length;

    if trim_end <= output.len() {
        output[trim_start..trim_end].to_vec()
    } else {
        // Edge case: buffer shorter than expected — pad with zeros.
        let mut result = output[trim_start..].to_vec();
        result.resize(signal_length, 0.0);
        result
    }
}

/// Resample `signal` from `from_rate` Hz to `to_rate` Hz using linear interpolation.
///
/// Special cases:
/// - `from_rate == to_rate` → returns `signal.to_vec()` (no-op).
/// - `signal.is_empty()` → returns empty Vec.
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
///
/// - `channels == 0` or `channels == 1` → returns `signal.to_vec()`.
/// - Trailing samples that don't form a complete frame are silently dropped.
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
///
/// Output length is `2 * min(left.len(), right.len())`.
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
///
/// Returns `(normalised_signal, scale)` where `scale` is the peak magnitude
/// used for division.  If the peak is below 1e-8 the original signal is
/// returned unchanged with `scale = 1.0` (avoids division by zero).
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

/// Wraps an ONNX Runtime session and the associated MDX-Net model parameters.
#[derive(Debug)]
pub struct Separator {
    session: ort::session::Session,
    params: ModelParams,
    chunk_size: usize,
}

impl Separator {
    /// Create a new `Separator` by building an ORT session from `config`.
    pub fn new(config: &Config, params: ModelParams) -> Result<Self, AppError> {
        tracing::debug!(
            model_path = %config.model_path.display(),
            execution_provider = ?config.execution_provider,
            chunk_size = config.chunk_size,
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

        Ok(Self {
            session,
            params,
            chunk_size: config.chunk_size,
        })
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

        // ── Step 4: DSP parameters ────────────────────────────────────────────
        // MDX-Net specifics:
        //   - hop is fixed at 1024 (independent of fft_size).
        //   - dim_t in the model is `2 ** mdx_dim_t_set` (the JSON stores the
        //     log2 exponent, e.g. 8 → 256 time frames).
        //   - chunk_size in the config is expected to equal `1024 * (dim_t - 1)`.
        let fft_size = self.params.mdx_n_fft_scale_set;
        let hop_length = 1024usize;
        let window = hann_window(fft_size);
        let n_bins_full = fft_size / 2 + 1;

        // ── Step 5: Split into overlapping chunks (50 % overlap) ─────────────
        let hop = self.chunk_size / 2;
        let mut chunks: Vec<Vec<f32>> = Vec::new();
        let mut start = 0usize;
        while start < normalized.len() {
            let end = (start + self.chunk_size).min(normalized.len());
            let mut chunk = normalized[start..end].to_vec();
            if chunk.len() < self.chunk_size {
                chunk.resize(self.chunk_size, 0.0); // zero-pad last chunk
            }
            chunks.push(chunk);
            if start + self.chunk_size >= normalized.len() {
                break;
            }
            start += hop;
        }
        let total_chunks = chunks.len();

        tracing::info!(
            total_chunks,
            chunk_size = self.chunk_size,
            fft_size,
            hop_length,
            "chunked input; running inference"
        );

        // ── Step 6: Chunk processing loop ─────────────────────────────────────
        let mut output_chunks: Vec<Vec<f32>> = Vec::with_capacity(total_chunks);

        for (idx, chunk) in chunks.iter().enumerate() {
            tracing::debug!(
                chunk_index = idx + 1,
                of = total_chunks,
                samples = chunk.len(),
                "processing chunk"
            );
            let n_bins_model = self.params.mdx_dim_f_set;
            // mdx_dim_t_set is stored as log2 — actual model time dim is 2^value.
            let n_time_model = 1usize << self.params.mdx_dim_t_set;

            // a) STFT
            let frames = stft(chunk, fft_size, hop_length, &window);

            // b) Build input tensor [1, 4, n_bins_model, n_time_model]
            let frames_to_use: Vec<&Vec<[f32; 2]>> =
                frames.iter().take(n_time_model).collect();

            let mut tensor_data = vec![0.0_f32; 4 * n_bins_model * n_time_model];
            for t in 0..n_time_model {
                for f in 0..n_bins_model {
                    let (re, im) =
                        if t < frames_to_use.len() && f < frames_to_use[t].len() {
                            (frames_to_use[t][f][0], frames_to_use[t][f][1])
                        } else {
                            (0.0_f32, 0.0_f32)
                        };
                    // channels 0 (real) and 1 (imag) for input ch0
                    tensor_data[0 * n_bins_model * n_time_model + f * n_time_model + t] = re;
                    tensor_data[1 * n_bins_model * n_time_model + f * n_time_model + t] = im;
                    // channels 2 (real) and 3 (imag) — copies for ch1
                    tensor_data[2 * n_bins_model * n_time_model + f * n_time_model + t] = re;
                    tensor_data[3 * n_bins_model * n_time_model + f * n_time_model + t] = im;
                }
            }

            // c) Run ONNX forward pass — scoped so `outputs` is dropped before
            //    the next iteration borrows `self.session` again.
            let out_frames: Vec<Vec<[f32; 2]>> = {
                let tensor = ort::value::Tensor::<f32>::from_array(
                    ([1usize, 4, n_bins_model, n_time_model], tensor_data),
                )
                .map_err(|e| AppError::OnnxInference(e.to_string()))?;

                let outputs = self
                    .session
                    .run(ort::inputs![tensor])
                    .map_err(|e| AppError::OnnxInference(e.to_string()))?;

                // d) Extract output and rebuild complex frames
                let (_out_shape, output_flat) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| AppError::OnnxInference(e.to_string()))?;

                // iSTFT needs the full one-sided spectrum (fft_size/2 + 1 bins).
                // The model only outputs n_bins_model bins; the remaining
                // high-frequency bins are left as zeros.
                let mut frames_out: Vec<Vec<[f32; 2]>> =
                    vec![vec![[0.0, 0.0]; n_bins_full]; n_time_model];
                for t in 0..n_time_model {
                    for f in 0..n_bins_model {
                        let base = f * n_time_model + t;
                        let re = output_flat.get(base).copied().unwrap_or(0.0);
                        let imag_base = n_bins_model * n_time_model + base;
                        let im = output_flat.get(imag_base).copied().unwrap_or(0.0);
                        frames_out[t][f] = [re, im];
                    }
                }
                frames_out
                // `outputs` is dropped here, releasing the mutable borrow on
                // `self.session`.
            };

            // e) iSTFT → time-domain chunk
            let chunk_output = istft(&out_frames, fft_size, hop_length, &window, chunk.len());
            output_chunks.push(chunk_output);
            progress_cb(idx as f32 / total_chunks as f32);
        }

        // ── Step 7: Overlap-add reconstruction ───────────────────────────────
        let output_len = normalized.len();
        let mut full_output = vec![0.0_f32; output_len];
        let mut weight = vec![0.0_f32; output_len];

        let mut pos = 0usize;
        for chunk_out in &output_chunks {
            let copy_len = chunk_out.len().min(output_len.saturating_sub(pos));
            for i in 0..copy_len {
                full_output[pos + i] += chunk_out[i];
                weight[pos + i] += 1.0;
            }
            if pos + self.chunk_size >= output_len {
                break;
            }
            pos += self.chunk_size / 2;
        }

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
            chunk_size: 261120,
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
