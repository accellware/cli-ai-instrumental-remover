use std::path::PathBuf;
use serde::Deserialize;
use crate::error::AppError;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model_path: PathBuf,
    pub output_dir: PathBuf,
    pub execution_provider: ExecutionProvider,
    pub chunk_size: usize,
}

pub fn load() -> Result<Config, AppError> {
    let path = std::env::current_dir()
        .map_err(|e| AppError::ConfigParse(format!("could not determine cwd: {e}")))?
        .join("config.json");

    tracing::debug!(path = %path.display(), "looking for config.json");

    if !path.exists() {
        tracing::error!(path = %path.display(), "config.json not found");
        return Err(AppError::ConfigNotFound);
    }

    let contents = std::fs::read_to_string(&path)
        .map_err(|e| AppError::ConfigParse(format!("read error: {e}")))?;

    tracing::debug!(bytes = contents.len(), raw = %contents, "read config.json");

    let cfg = serde_json::from_str::<Config>(&contents)
        .map_err(|e| AppError::ConfigParse(e.to_string()))?;

    if cfg.chunk_size == 0 {
        return Err(AppError::ConfigParse(
            "chunk_size must be greater than 0".to_string(),
        ));
    }

    tracing::info!(
        model_path = %cfg.model_path.display(),
        output_dir = %cfg.output_dir.display(),
        execution_provider = ?cfg.execution_provider,
        chunk_size = cfg.chunk_size,
        "config loaded"
    );

    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use tempfile::tempdir;

    fn cwd_lock() -> &'static Mutex<()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
    }

    struct CwdGuard {
        original: PathBuf,
        _lock: MutexGuard<'static, ()>,
    }

    impl CwdGuard {
        fn new(target: &Path) -> Self {
            let lock = match cwd_lock().lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            let original = std::env::current_dir().unwrap();
            std::env::set_current_dir(target).unwrap();
            Self {
                original,
                _lock: lock,
            }
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    fn write_config(dir: &Path, contents: &str) {
        std::fs::write(dir.join("config.json"), contents).unwrap();
    }

    #[test]
    fn valid_config_loads_all_fields() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_path": "models/foo.onnx", "output_dir": "./out", "execution_provider": "cpu", "chunk_size": 261120}"#,
        );
        let _guard = CwdGuard::new(dir.path());

        let cfg = load().unwrap();
        assert_eq!(cfg.model_path, PathBuf::from("models/foo.onnx"));
        assert_eq!(cfg.output_dir, PathBuf::from("./out"));
        assert_eq!(cfg.execution_provider, ExecutionProvider::Cpu);
        assert_eq!(cfg.chunk_size, 261120);
    }

    #[test]
    fn missing_config_returns_config_not_found() {
        let dir = tempdir().unwrap();
        let _guard = CwdGuard::new(dir.path());

        assert!(matches!(load(), Err(AppError::ConfigNotFound)));
    }

    #[test]
    fn malformed_json_returns_config_parse() {
        let dir = tempdir().unwrap();
        write_config(dir.path(), r#""not json""#);
        let _guard = CwdGuard::new(dir.path());

        assert!(matches!(load(), Err(AppError::ConfigParse(_))));
    }

    #[test]
    fn chunk_size_zero_returns_config_parse() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_path": "models/foo.onnx", "output_dir": "./out", "execution_provider": "cpu", "chunk_size": 0}"#,
        );
        let _guard = CwdGuard::new(dir.path());

        let err = load().unwrap_err();
        match err {
            AppError::ConfigParse(msg) => assert!(
                msg.contains("chunk_size must be greater than 0"),
                "got: {msg}"
            ),
            other => panic!("expected ConfigParse, got: {other:?}"),
        }
    }

    #[test]
    fn unknown_execution_provider_returns_config_parse() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_path": "models/foo.onnx", "output_dir": "./out", "execution_provider": "tpu", "chunk_size": 261120}"#,
        );
        let _guard = CwdGuard::new(dir.path());

        assert!(matches!(load(), Err(AppError::ConfigParse(_))));
    }

    #[test]
    fn cuda_string_deserializes_to_cuda_variant() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_path": "models/foo.onnx", "output_dir": "./out", "execution_provider": "cuda", "chunk_size": 261120}"#,
        );
        let _guard = CwdGuard::new(dir.path());

        let cfg = load().unwrap();
        assert_eq!(cfg.execution_provider, ExecutionProvider::Cuda);
    }
}
