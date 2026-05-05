// DSP primitives — pure Rust, no ONNX, no FFmpeg.
// Implemented in Prompt 6.

use rustfft::{FftPlanner, num_complex::Complex};

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
}
