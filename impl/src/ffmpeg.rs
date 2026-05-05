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

        let ictx = ffmpeg::format::input(video_path)
            .map_err(|e| AppError::FfmpegProbe(e.to_string()))?;

        for stream in ictx.streams() {
            if stream.parameters().medium() == ffmpeg::media::Type::Audio {
                let params = stream.parameters();
                let sample_rate = params.rate() as u32;
                let channels = params.channels() as u16;
                return Ok(Some(AudioInfo { sample_rate, channels }));
            }
        }

        Ok(None)
    }

    pub fn extract_audio(video_path: &Path, output_wav: &Path) -> Result<AudioInfo, AppError> {
        ffmpeg::init().map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        let mut ictx = ffmpeg::format::input(video_path)
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        let mut octx = ffmpeg::format::output(output_wav)
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        // Use a block so in_stream (borrows ictx) and out_stream (borrows octx)
        // are both dropped before ictx.packets() needs &mut ictx.
        let (audio_stream_index, sample_rate, channels, in_time_base) = {
            let in_stream = ictx
                .streams()
                .best(ffmpeg::media::Type::Audio)
                .ok_or_else(|| AppError::FfmpegExtract("no audio stream".to_string()))?;

            let codec = ffmpeg::encoder::find(ffmpeg::codec::Id::PCM_S16LE)
                .ok_or_else(|| AppError::FfmpegExtract("PCM_S16LE encoder not found".to_string()))?;
            let mut out_stream = octx
                .add_stream(codec)
                .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;
            out_stream.set_parameters(in_stream.parameters());

            (
                in_stream.index(),
                in_stream.parameters().rate() as u32,
                in_stream.parameters().channels() as u16,
                in_stream.time_base(),
            )
            // in_stream and out_stream are both dropped here
        };

        octx.write_header()
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        // Get out_time_base after write_header so the muxer has finalised it.
        // ffmpeg::Rational is Copy so the borrow ends immediately.
        let out_time_base = octx
            .stream(0)
            .ok_or_else(|| AppError::FfmpegExtract("output stream 0 missing after write_header".to_string()))?
            .time_base();

        for (stream, mut packet) in ictx.packets() {
            if stream.index() != audio_stream_index {
                continue;
            }
            packet.rescale_ts(in_time_base, out_time_base);
            packet.set_stream(0);
            packet
                .write_interleaved(&mut octx)
                .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;
        }

        octx.write_trailer()
            .map_err(|e| AppError::FfmpegExtract(e.to_string()))?;

        Ok(AudioInfo { sample_rate, channels })
    }
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
}
