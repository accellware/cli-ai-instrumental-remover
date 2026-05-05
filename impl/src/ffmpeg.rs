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
                let sample_rate = unsafe { (*params.as_ptr()).sample_rate as u32 };
                let channels = unsafe { (*params.as_ptr()).ch_layout.nb_channels as u16 };
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
                unsafe { (*in_stream.parameters().as_ptr()).sample_rate as u32 },
                unsafe { (*in_stream.parameters().as_ptr()).ch_layout.nb_channels as u16 },
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

    pub fn remux_with_audio(
        video_path: &Path,
        vocals_wav: &Path,
        output_path: &Path,
    ) -> Result<(), AppError> {
        ffmpeg::init().map_err(|e| AppError::FfmpegRemux(e.to_string()))?;

        let mut video_ictx = ffmpeg::format::input(video_path)
            .map_err(|e| AppError::FfmpegRemux(e.to_string()))?;
        let mut audio_ictx = ffmpeg::format::input(vocals_wav)
            .map_err(|e| AppError::FfmpegRemux(e.to_string()))?;
        let mut octx = ffmpeg::format::output(output_path)
            .map_err(|e| AppError::FfmpegRemux(e.to_string()))?;

        // video_stream_map: (video_in_stream_index, video_in_time_base, output_stream_index)
        // audio info: (audio_in_stream_index, audio_in_time_base, audio_output_stream_index)
        let (video_stream_map, audio_in_idx, audio_in_tb, audio_out_idx) = {
            let mut vmap: Vec<(usize, ffmpeg::Rational, usize)> = Vec::new();
            let mut out_idx = 0usize;

            for in_stream in video_ictx.streams() {
                if in_stream.parameters().medium() == ffmpeg::media::Type::Video {
                    let mut out_stream = octx
                        .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
                        .map_err(|e| AppError::FfmpegRemux(e.to_string()))?;
                    out_stream.set_parameters(in_stream.parameters());
                    vmap.push((in_stream.index(), in_stream.time_base(), out_idx));
                    out_idx += 1;
                }
            }

            let audio_in_stream = audio_ictx
                .streams()
                .best(ffmpeg::media::Type::Audio)
                .ok_or_else(|| AppError::FfmpegRemux("vocals WAV has no audio stream".to_string()))?;
            let a_in_idx = audio_in_stream.index();
            let a_in_tb = audio_in_stream.time_base();

            {
                let mut out_audio = octx
                    .add_stream(ffmpeg::encoder::find(ffmpeg::codec::Id::None))
                    .map_err(|e| AppError::FfmpegRemux(e.to_string()))?;
                out_audio.set_parameters(audio_in_stream.parameters());
            }

            (vmap, a_in_idx, a_in_tb, out_idx)
            // all borrows on video_ictx, audio_ictx, octx dropped here
        };

        octx.write_header()
            .map_err(|e| AppError::FfmpegRemux(e.to_string()))?;

        // Write all video packets from original
        for (stream, mut packet) in video_ictx.packets() {
            if let Some(&(_, in_tb, out_idx)) = video_stream_map
                .iter()
                .find(|(in_idx, _, _)| *in_idx == stream.index())
            {
                let out_tb = octx
                    .stream(out_idx)
                    .ok_or_else(|| AppError::FfmpegRemux("output video stream missing".to_string()))?
                    .time_base();
                packet.rescale_ts(in_tb, out_tb);
                packet.set_stream(out_idx);
                packet
                    .write_interleaved(&mut octx)
                    .map_err(|e| AppError::FfmpegRemux(e.to_string()))?;
            }
        }

        // Write all audio packets from vocals WAV
        for (stream, mut packet) in audio_ictx.packets() {
            if stream.index() != audio_in_idx {
                continue;
            }
            let out_tb = octx
                .stream(audio_out_idx)
                .ok_or_else(|| AppError::FfmpegRemux("output audio stream missing".to_string()))?
                .time_base();
            packet.rescale_ts(audio_in_tb, out_tb);
            packet.set_stream(audio_out_idx);
            packet
                .write_interleaved(&mut octx)
                .map_err(|e| AppError::FfmpegRemux(e.to_string()))?;
        }

        octx.write_trailer()
            .map_err(|e| AppError::FfmpegRemux(e.to_string()))?;

        Ok(())
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
