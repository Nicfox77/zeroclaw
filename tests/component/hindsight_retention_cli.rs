//! Local operator controls must persist across CLI processes without contacting an LLM.
use serde_json::Value;
use std::{
    path::Path,
    process::{Command, Output},
};

fn configure(directory: &Path, backend: &str) {
    std::fs::write(
        directory.join("config.toml"),
        format!(
            r#"schema_version = 4
locale = "en"
[memory]
backend = "{backend}"
[memory.hindsight]
mode = "external"
api_url = "http://127.0.0.1:9"
[memory.hindsight.retention]
deferred = true
[channels]
session_persistence = true
session_backend = "sqlite"
"#
        ),
    )
    .unwrap();
}
fn cli(directory: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_zeroclaw"))
        .arg("--config-dir")
        .arg(directory)
        .args(args)
        .env("ZEROCLAW_DATA_DIR", directory.join("data"))
        .env("RUST_LOG", "off")
        .env("LC_ALL", "C")
        .env("TERM", "dumb")
        .output()
        .unwrap()
}
fn retention(directory: &Path, args: &[&str]) -> Value {
    let mut command = vec!["memory", "retention"];
    command.extend_from_slice(args);
    let output = cli(directory, &command);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid retention status: {e}: {stdout}"))
}
#[test]
fn operator_can_drain_pause_and_restore_schedule_across_processes_without_a_service() {
    let directory = tempfile::tempdir().unwrap();
    configure(directory.path(), "hindsight");
    let before = retention(directory.path(), &["status"]);
    assert_eq!(before["accepting_new_work"], false);
    assert_eq!(before["control"]["state"], "automatic");
    let draining = retention(directory.path(), &["drain", "--minutes", "10"]);
    assert_eq!(draining["accepting_new_work"], true);
    assert_eq!(draining["control"]["state"], "draining");
    assert_eq!(
        retention(directory.path(), &["status"])["control"],
        draining["control"]
    );
    assert_eq!(
        retention(directory.path(), &["pause"])["accepting_new_work"],
        false
    );
    let retry = cli(directory.path(), &["memory", "retry-pending"]);
    assert!(
        retry.status.success(),
        "{}",
        String::from_utf8_lossy(&retry.stderr)
    );
    assert_eq!(
        retention(directory.path(), &["status"])["control"]["state"],
        "paused"
    );
    let resumed = retention(directory.path(), &["resume"]);
    assert_eq!(resumed["control"]["state"], "automatic");
    assert_eq!(resumed["accepting_new_work"], false);
}
#[test]
fn invalid_drain_and_wrong_backend_fail_without_creating_admission_state() {
    let directory = tempfile::tempdir().unwrap();
    configure(directory.path(), "hindsight");
    for minutes in ["0", "1441"] {
        assert!(
            !cli(
                directory.path(),
                &["memory", "retention", "drain", "--minutes", minutes]
            )
            .status
            .success()
        );
    }
    assert_eq!(
        retention(directory.path(), &["status"])["control"]["state"],
        "automatic"
    );
    let other = tempfile::tempdir().unwrap();
    configure(other.path(), "sqlite");
    assert!(
        !cli(other.path(), &["memory", "retention", "drain"])
            .status
            .success()
    );
    assert!(!other.path().join("data/hindsight/delivery.db").exists());
}
