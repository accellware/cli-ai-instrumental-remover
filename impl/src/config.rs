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

    /// Optional ORT/CUDA tuning. Any field omitted here keeps the ORT default;
    /// see `Tuning` for the full schema.
    #[serde(default)]
    pub tuning: Tuning,
}

/// Session-level tuning knobs. All fields are optional — `None`/missing keeps
/// ORT's own default behaviour. The nested `cuda` block only applies when
/// `execution_provider == "cuda"`.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Tuning {
    /// Threads used to parallelise execution within a single op
    /// (maps to `SessionBuilder::with_intra_threads`).
    pub intra_threads: Option<usize>,

    /// Threads used when `parallel_execution` is enabled — i.e. for executing
    /// independent ops concurrently (`with_inter_threads`).
    pub inter_threads: Option<usize>,

    /// Enable parallel-op execution mode (`with_parallel_execution`).
    pub parallel_execution: Option<bool>,

    /// Toggle predictive memory-pattern optimisation (`with_memory_pattern`).
    /// Disable when input shapes vary across runs.
    pub memory_pattern: Option<bool>,

    pub cuda: CudaTuning,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaTuning {
    /// CUDA device index (`with_device_id`).
    pub device_id: Option<i32>,

    /// Cap the CUDA EP's memory arena, in MiB (`with_memory_limit`, applied as
    /// `value * 1024 * 1024` bytes). Actual GPU usage may exceed this since
    /// some allocations bypass the arena.
    pub gpu_mem_limit_mb: Option<usize>,

    /// How the CUDA arena grows when more memory is needed.
    pub arena_extend_strategy: Option<ArenaExtendStrategyOpt>,

    /// cuDNN convolution algorithm search mode. Exhaustive picks the fastest
    /// kernel at warmup but uses more memory; Heuristic skips the search.
    pub cudnn_conv_algo_search: Option<ConvAlgorithmSearchOpt>,

    /// Allow exhaustive cuDNN search to use unbounded workspace
    /// (`with_conv_max_workspace`). False caps it at 32 MB and may pick a
    /// slower kernel.
    pub cudnn_conv_use_max_workspace: Option<bool>,

    /// Enable TensorFloat-32 for `MatMul`/`Conv` on Ampere+ Tensor cores
    /// (`with_tf32`). Faster but reduced precision; off by default.
    pub tf32: Option<bool>,

    /// Prefer NHWC layout for ops that support it (`with_prefer_nhwc`).
    pub prefer_nhwc: Option<bool>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArenaExtendStrategyOpt {
    NextPowerOfTwo,
    SameAsRequested,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConvAlgorithmSearchOpt {
    Exhaustive,
    Heuristic,
    Default,
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

    tracing::info!(
        model_path = %cfg.model_path.display(),
        output_dir = %cfg.output_dir.display(),
        execution_provider = ?cfg.execution_provider,
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
            r#"{"model_path": "models/foo.onnx", "output_dir": "./out", "execution_provider": "cpu"}"#,
        );
        let _guard = CwdGuard::new(dir.path());

        let cfg = load().unwrap();
        assert_eq!(cfg.model_path, PathBuf::from("models/foo.onnx"));
        assert_eq!(cfg.output_dir, PathBuf::from("./out"));
        assert_eq!(cfg.execution_provider, ExecutionProvider::Cpu);
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
    fn unknown_execution_provider_returns_config_parse() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_path": "models/foo.onnx", "output_dir": "./out", "execution_provider": "tpu"}"#,
        );
        let _guard = CwdGuard::new(dir.path());

        assert!(matches!(load(), Err(AppError::ConfigParse(_))));
    }

    #[test]
    fn cuda_string_deserializes_to_cuda_variant() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_path": "models/foo.onnx", "output_dir": "./out", "execution_provider": "cuda"}"#,
        );
        let _guard = CwdGuard::new(dir.path());

        let cfg = load().unwrap();
        assert_eq!(cfg.execution_provider, ExecutionProvider::Cuda);
    }

    #[test]
    fn tuning_block_omitted_yields_all_none_defaults() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_path": "models/foo.onnx", "output_dir": "./out", "execution_provider": "cpu"}"#,
        );
        let _guard = CwdGuard::new(dir.path());

        let cfg = load().unwrap();
        assert!(cfg.tuning.intra_threads.is_none());
        assert!(cfg.tuning.inter_threads.is_none());
        assert!(cfg.tuning.parallel_execution.is_none());
        assert!(cfg.tuning.memory_pattern.is_none());
        assert!(cfg.tuning.cuda.device_id.is_none());
        assert!(cfg.tuning.cuda.gpu_mem_limit_mb.is_none());
        assert!(cfg.tuning.cuda.arena_extend_strategy.is_none());
        assert!(cfg.tuning.cuda.cudnn_conv_algo_search.is_none());
        assert!(cfg.tuning.cuda.cudnn_conv_use_max_workspace.is_none());
        assert!(cfg.tuning.cuda.tf32.is_none());
        assert!(cfg.tuning.cuda.prefer_nhwc.is_none());
    }

    #[test]
    fn tuning_block_populated_parses_all_fields() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{
                "model_path": "models/foo.onnx",
                "output_dir": "./out",
                "execution_provider": "cuda",
                "tuning": {
                    "intra_threads": 4,
                    "inter_threads": 2,
                    "parallel_execution": true,
                    "memory_pattern": false,
                    "cuda": {
                        "device_id": 0,
                        "gpu_mem_limit_mb": 4096,
                        "arena_extend_strategy": "same_as_requested",
                        "cudnn_conv_algo_search": "heuristic",
                        "cudnn_conv_use_max_workspace": true,
                        "tf32": true,
                        "prefer_nhwc": true
                    }
                }
            }"#,
        );
        let _guard = CwdGuard::new(dir.path());

        let cfg = load().unwrap();
        assert_eq!(cfg.tuning.intra_threads, Some(4));
        assert_eq!(cfg.tuning.inter_threads, Some(2));
        assert_eq!(cfg.tuning.parallel_execution, Some(true));
        assert_eq!(cfg.tuning.memory_pattern, Some(false));
        assert_eq!(cfg.tuning.cuda.device_id, Some(0));
        assert_eq!(cfg.tuning.cuda.gpu_mem_limit_mb, Some(4096));
        assert_eq!(
            cfg.tuning.cuda.arena_extend_strategy,
            Some(ArenaExtendStrategyOpt::SameAsRequested)
        );
        assert_eq!(
            cfg.tuning.cuda.cudnn_conv_algo_search,
            Some(ConvAlgorithmSearchOpt::Heuristic)
        );
        assert_eq!(cfg.tuning.cuda.cudnn_conv_use_max_workspace, Some(true));
        assert_eq!(cfg.tuning.cuda.tf32, Some(true));
        assert_eq!(cfg.tuning.cuda.prefer_nhwc, Some(true));
    }

    #[test]
    fn unknown_tuning_field_is_rejected() {
        let dir = tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{
                "model_path": "models/foo.onnx",
                "output_dir": "./out",
                "execution_provider": "cpu",
                "tuning": { "bogus": 1 }
            }"#,
        );
        let _guard = CwdGuard::new(dir.path());

        assert!(matches!(load(), Err(AppError::ConfigParse(_))));
    }
}
