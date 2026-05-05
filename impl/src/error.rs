use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("config.json not found in current working directory")]
    ConfigNotFound,

    #[error("failed to parse config.json: {0}")]
    ConfigParse(String),

    #[error("model file not found: {0}")]
    ModelNotFound(PathBuf),

    #[error("model_data.json not found at: {0}")]
    ModelDataNotFound(PathBuf),

    #[error("failed to parse model_data.json: {0}")]
    ModelDataParse(String),

    #[error("model not listed in model_data.json: {0}")]
    ModelNotInRegistry(String),

    #[error("input video file not found: {0}")]
    InputVideoNotFound(PathBuf),

    #[error("failed to create output directory: {0}")]
    OutputDirCreate(String),

    #[error("ffmpeg probe failed: {0}")]
    FfmpegProbe(String),

    #[error("ffmpeg audio extraction failed: {0}")]
    FfmpegExtract(String),

    #[error("ffmpeg remux failed: {0}")]
    FfmpegRemux(String),

    #[error("failed to load ONNX model: {0}")]
    OnnxLoad(String),

    #[error("ONNX inference failed: {0}")]
    OnnxInference(String),

    #[error("failed to read WAV audio: {0}")]
    AudioRead(String),

    #[error("failed to write WAV audio: {0}")]
    AudioWrite(String),

    #[error("failed to copy file: {0}")]
    FileCopy(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_formats_to_non_empty_string() {
        let errors: Vec<AppError> = vec![
            AppError::ConfigNotFound,
            AppError::ConfigParse("x".to_string()),
            AppError::ModelNotFound(PathBuf::from("x")),
            AppError::ModelDataNotFound(PathBuf::from("x")),
            AppError::ModelDataParse("x".to_string()),
            AppError::ModelNotInRegistry("x".to_string()),
            AppError::InputVideoNotFound(PathBuf::from("x")),
            AppError::OutputDirCreate("x".to_string()),
            AppError::FfmpegProbe("x".to_string()),
            AppError::FfmpegExtract("x".to_string()),
            AppError::FfmpegRemux("x".to_string()),
            AppError::OnnxLoad("x".to_string()),
            AppError::OnnxInference("x".to_string()),
            AppError::AudioRead("x".to_string()),
            AppError::AudioWrite("x".to_string()),
            AppError::FileCopy("x".to_string()),
        ];

        assert_eq!(errors.len(), 16);

        for err in &errors {
            assert!(!err.to_string().is_empty());
        }
    }
}
