use std::collections::HashMap;
use std::path::Path;
use serde::Deserialize;
use crate::error::AppError;

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelParams {
    pub compensate: f32,
    pub mdx_dim_f_set: usize,
    pub mdx_dim_t_set: usize,
    pub mdx_n_fft_scale_set: usize,
    pub primary_stem: String,
    pub name: String,
}

pub fn load(model_data_path: &Path) -> Result<Vec<ModelParams>, AppError> {
    if !model_data_path.exists() {
        return Err(AppError::ModelDataNotFound(model_data_path.to_path_buf()));
    }
    let contents = std::fs::read_to_string(model_data_path)
        .map_err(|e| AppError::ModelDataParse(format!("read error: {e}")))?;
    let map: HashMap<String, ModelParams> = serde_json::from_str(&contents)
        .map_err(|e| AppError::ModelDataParse(e.to_string()))?;
    Ok(map.into_values().collect())
}

pub fn find_by_name<'a>(
    params: &'a [ModelParams],
    model_filename: &str,
) -> Result<&'a ModelParams, AppError> {
    params
        .iter()
        .find(|p| p.name == model_filename)
        .ok_or_else(|| AppError::ModelNotInRegistry(model_filename.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tempfile::tempdir;

    fn sample_params() -> Vec<ModelParams> {
        vec![
            ModelParams {
                compensate: 1.021,
                mdx_dim_f_set: 3072,
                mdx_dim_t_set: 8,
                mdx_n_fft_scale_set: 7680,
                primary_stem: "Vocals".to_string(),
                name: "UVR-MDX-NET-Voc_FT.onnx".to_string(),
            },
            ModelParams {
                compensate: 1.035,
                mdx_dim_f_set: 2048,
                mdx_dim_t_set: 8,
                mdx_n_fft_scale_set: 5120,
                primary_stem: "Instrumental".to_string(),
                name: "UVR_MDXNET_KARA_2.onnx".to_string(),
            },
        ]
    }

    #[test]
    fn load_parses_valid_json_into_vec() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("model_data.json");
        let json = r#"{
    "77d07b2667ddf05b9e3175941b4454a0": {
        "compensate": 1.021,
        "mdx_dim_f_set": 3072,
        "mdx_dim_t_set": 8,
        "mdx_n_fft_scale_set": 7680,
        "primary_stem": "Vocals",
        "name": "UVR-MDX-NET-Voc_FT.onnx"
    },
    "1d64a6d2c30f709b8c9b4ce1366d96ee": {
        "compensate": 1.035,
        "mdx_dim_f_set": 2048,
        "mdx_dim_t_set": 8,
        "mdx_n_fft_scale_set": 5120,
        "primary_stem": "Instrumental",
        "name": "UVR_MDXNET_KARA_2.onnx"
    }
}"#;
        std::fs::write(&path, json).expect("write json");

        let result = load(&path).expect("load should succeed");
        assert_eq!(result.len(), 2);

        let names: HashSet<String> = result.iter().map(|p| p.name.clone()).collect();
        assert!(names.contains("UVR-MDX-NET-Voc_FT.onnx"));
        assert!(names.contains("UVR_MDXNET_KARA_2.onnx"));
    }

    #[test]
    fn load_missing_file_returns_model_data_not_found() {
        let dir = tempdir().expect("create tempdir");
        let nonexistent_path = dir.path().join("does_not_exist.json");

        let err = load(&nonexistent_path).expect_err("should fail for missing file");
        assert!(matches!(err, AppError::ModelDataNotFound(p) if p == nonexistent_path));
    }

    #[test]
    fn load_malformed_json_returns_model_data_parse() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "not json").expect("write bad json");

        let err = load(&path).expect_err("should fail for malformed json");
        assert!(matches!(err, AppError::ModelDataParse(_)));
    }

    #[test]
    fn find_by_name_returns_matching_model() {
        let params = sample_params();
        let found = find_by_name(&params, "UVR-MDX-NET-Voc_FT.onnx").expect("should find");
        assert_eq!(found.name, "UVR-MDX-NET-Voc_FT.onnx");
        assert_eq!(found.compensate, 1.021);
    }

    #[test]
    fn find_by_name_unknown_returns_model_not_in_registry() {
        let params = sample_params();
        let err = find_by_name(&params, "missing.onnx").expect_err("should not find");
        assert!(matches!(err, AppError::ModelNotInRegistry(name) if name == "missing.onnx"));
    }
}
