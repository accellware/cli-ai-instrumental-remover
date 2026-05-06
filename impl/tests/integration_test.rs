use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[test]
fn cli_exits_1_when_no_config() {
    let dir = tempdir().unwrap();
    let mut cmd = Command::cargo_bin("music-separator").unwrap();
    cmd.current_dir(dir.path())
        .args(["--input", "/nonexistent/video.mp4"]);
    let output = cmd.output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("config"),
        "expected 'config' in stderr, got: {stderr}"
    );
}

#[test]
fn cli_exits_1_when_input_not_found() {
    let dir = tempdir().unwrap();
    // Write a minimal valid config.json to the temp dir
    let config_content = r#"{
        "model_path": "/nonexistent/model.onnx",
        "output_dir": "./output",
        "execution_provider": "cpu"
    }"#;
    fs::write(dir.path().join("config.json"), config_content).unwrap();

    let mut cmd = Command::cargo_bin("music-separator").unwrap();
    cmd.current_dir(dir.path())
        .args(["--input", "/nonexistent/video.mp4"]);
    let output = cmd.output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    // We expect either ModelNotFound or InputVideoNotFound in stderr
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.is_empty(),
        "expected non-empty stderr error message"
    );
}

#[test]
fn cli_shows_help() {
    let mut cmd = Command::cargo_bin("music-separator").unwrap();
    cmd.arg("--help");
    let output = cmd.output().unwrap();
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--input"),
        "expected '--input' in help output, got: {stdout}"
    );
}

#[test]
fn cli_help_lists_logging_flags() {
    let mut cmd = Command::cargo_bin("music-separator").unwrap();
    cmd.arg("--help");
    let output = cmd.output().unwrap();
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--verbose") || stdout.contains("-v"),
        "expected verbose flag in help output, got: {stdout}"
    );
    assert!(
        stdout.contains("--log-level"),
        "expected --log-level in help output, got: {stdout}"
    );
    assert!(
        stdout.contains("--log-file"),
        "expected --log-file in help output, got: {stdout}"
    );
}

#[test]
fn cli_accepts_verbose_flag() {
    // -v should be parsed without error; the run will then exit 1 because
    // there is no config.json in the temp dir, which is fine — we only
    // care that clap accepts the flag.
    let dir = tempdir().unwrap();
    let mut cmd = Command::cargo_bin("music-separator").unwrap();
    cmd.current_dir(dir.path())
        .args(["-v", "--input", "/nonexistent/video.mp4"]);
    let output = cmd.output().unwrap();
    // Exit 1 (config missing) — but NOT 2 (which clap uses for parse errors).
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit 1, got: {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_accepts_repeated_verbose_flags() {
    let dir = tempdir().unwrap();
    let mut cmd = Command::cargo_bin("music-separator").unwrap();
    cmd.current_dir(dir.path())
        .args(["-vvv", "--input", "/nonexistent/video.mp4"]);
    let output = cmd.output().unwrap();
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn cli_accepts_log_level_flag() {
    let dir = tempdir().unwrap();
    let mut cmd = Command::cargo_bin("music-separator").unwrap();
    cmd.current_dir(dir.path())
        .args([
            "--log-level", "debug",
            "--input", "/nonexistent/video.mp4",
        ]);
    let output = cmd.output().unwrap();
    assert_eq!(output.status.code(), Some(1));
}
