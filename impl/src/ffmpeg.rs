use std::path::Path;
use crate::error::AppError;

#[derive(Debug, Clone, Copy)]
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: u16,
}

#[cfg(feature = "ffmpeg")]
mod imp {
    use super::*;
    use ffmpeg_next as ffmpeg;

    pub fn probe_audio(video_path: &Path) -> Result<Option<AudioInfo>, AppError> {
        ffmpeg::init().map_err(|e| AppError::FfmpegProbe(e.to_string()))?;

        tracing::debug!(path = %video_path.display(), "probing input streams");

        let ictx = ffmpeg::format::input(video_path)
            .map_err(|e| {
                tracing::warn!(path = %video_path.display(), error = %e, "ffmpeg probe open failed");
                AppError::FfmpegProbe(e.to_string())
            })?;

        let stream_count = ictx.streams().count();
        tracing::debug!(stream_count, "input opened");

        for stream in ictx.streams() {
            let medium = stream.parameters().medium();
            tracing::debug!(index = stream.index(), medium = ?medium, "stream");
            if medium == ffmpeg::media::Type::Audio {
                let params = stream.parameters();
                let sample_rate = unsafe { (*params.as_ptr()).sample_rate as u32 };
                let channels = unsafe { (*params.as_ptr()).ch_layout.nb_channels as u16 };
                tracing::debug!(
                    sample_rate, channels,
                    "audio stream selected"
                );
                return Ok(Some(AudioInfo { sample_rate, channels }));
            }
        }

        tracing::debug!("no audio stream found in input");
        Ok(None)
    }

    pub fn extract_audio(video_path: &Path, output_wav: &Path) -> Result<AudioInfo, AppError> {
        ffmpeg::init().map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        tracing::debug!(
            input = %video_path.display(),
            output = %output_wav.display(),
            "extract_audio start"
        );

        let mut ictx = ffmpeg::format::input(video_path)
            .map_err(|e| {
                tracing::warn!(error = %e, "ffmpeg input open failed");
                AppError::FfmpegExtract(e.to_string())
            })?;

        let (audio_stream_index, sample_rate, channels, decoder_params) = {
            let in_stream = ictx
                .streams()
                .best(ffmpeg::media::Type::Audio)
                .ok_or_else(|| AppError::FfmpegExtract("no audio stream".to_string()))?;
            let params = in_stream.parameters();
            let sr = unsafe { (*params.as_ptr()).sample_rate as u32 };
            let ch = unsafe { (*params.as_ptr()).ch_layout.nb_channels as u16 };
            (in_stream.index(), sr, ch, params)
        };

        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(decoder_params)
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;
        let mut decoder = codec_ctx
            .decoder()
            .audio()
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        tracing::debug!(
            stream_index = audio_stream_index,
            sample_rate,
            channels,
            decoder_format = ?decoder.format(),
            "audio decoder ready"
        );

        // Resample the decoder's native format (often FLTP) → packed S16 at the
        // same sample rate and channel layout, so we can write a standard WAV.
        let dst_format =
            ffmpeg::format::Sample::I16(ffmpeg::format::sample::Type::Packed);
        let dst_layout = decoder.channel_layout();
        let mut resampler = decoder
            .resampler(dst_format, dst_layout, sample_rate)
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        let wav_spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(output_wav, wav_spec)
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        let mut decoded = ffmpeg::frame::Audio::empty();
        let mut resampled = ffmpeg::frame::Audio::empty();

        let write_resampled =
            |frame: &ffmpeg::frame::Audio,
             writer: &mut hound::WavWriter<std::io::BufWriter<std::fs::File>>|
             -> Result<(), AppError> {
                let n = frame.samples() * channels as usize;
                if n == 0 {
                    return Ok(());
                }
                let bytes = frame.data(0);
                let needed = n * 2;
                if bytes.len() < needed {
                    return Err(AppError::FfmpegExtract(format!(
                        "resampler returned {} bytes, expected at least {}",
                        bytes.len(),
                        needed
                    )));
                }
                for chunk in bytes[..needed].chunks_exact(2) {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    writer
                        .write_sample(s)
                        .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;
                }
                Ok(())
            };

        for (stream, packet) in ictx.packets() {
            if stream.index() != audio_stream_index {
                continue;
            }
            decoder
                .send_packet(&packet)
                .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;
            while decoder.receive_frame(&mut decoded).is_ok() {
                resampler
                    .run(&decoded, &mut resampled)
                    .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;
                write_resampled(&resampled, &mut writer)?;
            }
        }

        decoder
            .send_eof()
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            resampler
                .run(&decoded, &mut resampled)
                .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;
            write_resampled(&resampled, &mut writer)?;
        }

        // Drain any samples buffered inside the resampler.
        while resampler.delay().is_some() {
            match resampler.flush(&mut resampled) {
                Ok(_) => write_resampled(&resampled, &mut writer)?,
                Err(e) => return Err(AppError::FfmpegExtract(e.to_string())),
            }
            if resampled.samples() == 0 {
                break;
            }
        }

        writer
            .finalize()
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        Ok(AudioInfo { sample_rate, channels })
    }

    pub fn remux_with_audio(
        video_path: &Path,
        vocals_wav: &Path,
        output_path: &Path,
    ) -> Result<(), AppError> {
        ffmpeg::init().map_err(|e| AppError::FfmpegRemux(e.to_string()))?;
        super::remux_via_ffmpeg_cli(video_path, vocals_wav, output_path)
    }
}

// Remux by spawning the `ffmpeg` CLI. We delegate this step instead of using
// the ffmpeg-next encoder/muxer API because:
//   - The vocals WAV is float PCM, which most container muxers (notably MP4)
//     don't accept — the audio must be re-encoded (e.g. to AAC) on the way out.
//     Doing that via the library requires an encoder + AVAudioFifo for
//     frame-size alignment + per-format sample-format negotiation, none of
//     which is exposed cleanly by ffmpeg-next 8.
//   - The CLI is already a runtime dependency: the shared libs we link
//     against come from the same FFmpeg install whose `bin/` is on PATH.
// The video stream is still copied (-c:v copy), preserving the project's
// "never re-encode video" guarantee.
fn remux_via_ffmpeg_cli(
    video_path: &Path,
    vocals_wav: &Path,
    output_path: &Path,
) -> Result<(), AppError> {
    use std::process::{Command, Stdio};

    tracing::debug!(
        video = %video_path.display(),
        audio = %vocals_wav.display(),
        output = %output_path.display(),
        "remux_with_audio start (ffmpeg CLI)"
    );

    let args = [
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        &video_path.to_string_lossy(),
        "-i",
        &vocals_wav.to_string_lossy(),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        &output_path.to_string_lossy(),
    ];
    tracing::debug!(?args, "spawning ffmpeg");

    let output = Command::new("ffmpeg")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| {
            AppError::FfmpegRemux(format!(
                "could not spawn ffmpeg (is it on PATH?): {e}"
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let code = output
            .status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "<signal>".to_string());
        tracing::warn!(exit_code = %code, stderr = %stderr, "ffmpeg remux failed");
        return Err(AppError::FfmpegRemux(format!(
            "ffmpeg exited with code {code}: {stderr}"
        )));
    }

    tracing::debug!("ffmpeg remux finished");
    Ok(())
}

pub fn probe_audio(video_path: &Path) -> Result<Option<AudioInfo>, AppError> {
    #[cfg(feature = "ffmpeg")]
    return imp::probe_audio(video_path);
    #[cfg(not(feature = "ffmpeg"))]
    {
        let _ = video_path;
        return Err(AppError::FfmpegProbe(
            "ffmpeg feature is not enabled; rebuild with --features ffmpeg".to_string(),
        ));
    }
}

pub fn extract_audio(video_path: &Path, output_wav: &Path) -> Result<AudioInfo, AppError> {
    #[cfg(feature = "ffmpeg")]
    return imp::extract_audio(video_path, output_wav);
    #[cfg(not(feature = "ffmpeg"))]
    {
        let _ = video_path;
        let _ = output_wav;
        return Err(AppError::FfmpegExtract(
            "ffmpeg feature is not enabled; rebuild with --features ffmpeg".to_string(),
        ));
    }
}

pub fn remux_with_audio(
    video_path: &Path,
    vocals_wav: &Path,
    output_path: &Path,
) -> Result<(), AppError> {
    #[cfg(feature = "ffmpeg")]
    return imp::remux_with_audio(video_path, vocals_wav, output_path);
    #[cfg(not(feature = "ffmpeg"))]
    {
        let _ = video_path;
        let _ = vocals_wav;
        let _ = output_path;
        return Err(AppError::FfmpegRemux(
            "ffmpeg feature is not enabled; rebuild with --features ffmpeg".to_string(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Always-compiled smoke test — verifies the public symbols exist and the
    // stub (or real impl) returns Err for a nonexistent file.
    #[test]
    fn probe_audio_missing_file_returns_error() {
        let result = probe_audio(std::path::Path::new("/nonexistent/file.mp4"));
        assert!(result.is_err());
    }

    #[test]
    fn remux_missing_input_returns_error() {
        let result = remux_with_audio(
            std::path::Path::new("/nonexistent/video.mp4"),
            std::path::Path::new("/nonexistent/audio.wav"),
            std::path::Path::new("/tmp/out.mp4"),
        );
        assert!(result.is_err());
    }
}
