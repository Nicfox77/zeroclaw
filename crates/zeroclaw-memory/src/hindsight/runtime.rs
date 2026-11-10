//! Official hindsight-embed lifecycle, separate from the HTTP client.
use super::{
    client::Client,
    ledger::{Ledger, hash},
};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::process::Command;
use zeroclaw_config::{
    hindsight::{LlmConfig, RuntimeMode},
    schema::Config,
};

pub fn directory(config: &Config) -> PathBuf {
    if config
        .memory
        .hindsight
        .runtime
        .data_dir
        .as_os_str()
        .is_empty()
    {
        config.data_dir.join("hindsight")
    } else {
        config.memory.hindsight.runtime.data_dir.clone()
    }
}
fn profile(config: &Config) -> String {
    format!(
        "zeroclaw-{}",
        &hash(directory(config).as_os_str().as_encoded_bytes())[..24]
    )
}
fn llm_env(
    config: &Config,
    llm: &LlmConfig,
    prefix: &str,
    env: &mut BTreeMap<String, String>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !llm.provider.trim().is_empty() && !llm.model.trim().is_empty(),
        "hindsight_generative_llm_required"
    );
    env.insert(format!("{prefix}_PROVIDER"), llm.provider.clone());
    env.insert(format!("{prefix}_MODEL"), llm.model.clone());
    if let Some(url) = &llm.base_url {
        env.insert(format!("{prefix}_BASE_URL"), url.clone());
    }
    let key = if !llm.credential_provider.is_empty() {
        anyhow::ensure!(llm.api_key.is_none(), "hindsight_ambiguous_llm_credentials");
        let (family, alias) = llm
            .credential_provider
            .split_once('.')
            .ok_or_else(|| anyhow::Error::msg("hindsight_invalid_credential_reference"))?;
        config
            .providers
            .models
            .find(family, alias)
            .ok_or_else(|| anyhow::Error::msg("hindsight_credential_reference_missing"))?
            .api_key
            .as_ref()
    } else {
        llm.api_key.as_ref()
    };
    if let Some(key) = key {
        anyhow::ensure!(
            !key.starts_with("enc2:") && !key.starts_with("enc:"),
            "hindsight_secret_not_unlocked"
        );
        env.insert(format!("{prefix}_API_KEY"), key.clone());
    }
    Ok(())
}
fn local_model(root: &Path, model: &str) -> String {
    let revision = match model {
        "hindsight-ai/hs-embed-1" => Some("3164caa4db95a03faac8a327f914f7c8f42d2f98"),
        "hindsight-ai/hs-rerank-1" => Some("927f5be31280298a51cf29af8a29cb111ed812c5"),
        _ => None,
    };
    revision
        .map(|rev| {
            root.join("models")
                .join(format!("models--{}", model.replace('/', "--")))
                .join("snapshots")
                .join(rev)
        })
        .filter(|p| p.is_dir())
        .map_or_else(|| model.into(), |p| p.to_string_lossy().into_owned())
}
/// Explicit mapping to settings supported by the pinned upstream release.
/// Returned values include secrets; callers must never log this map.
pub fn environment(config: &Config) -> anyhow::Result<BTreeMap<String, String>> {
    let h = &config.memory.hindsight;
    h.validate()?;
    let root = directory(config);
    let mut env = BTreeMap::new();
    if let Some(path) = &h.runtime.env_file {
        check_private(path)?;
        for line in std::fs::read_to_string(path)?.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| anyhow::Error::msg("hindsight_invalid_environment_file"))?;
            // A protected file supplies provider credentials/settings, never
            // listeners, executable paths or a competing database backend.
            anyhow::ensure!(
                [
                    "HINDSIGHT_API_LLM",
                    "HINDSIGHT_API_RETAIN_LLM",
                    "HINDSIGHT_API_REFLECT_LLM",
                    "HINDSIGHT_API_CONSOLIDATION_LLM",
                    "HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM"
                ]
                .iter()
                .any(|prefix| ["PROVIDER", "MODEL", "API_KEY", "BASE_URL"]
                    .iter()
                    .any(|suffix| key == format!("{prefix}_{suffix}"))),
                "hindsight_environment_key_not_allowed"
            );
            env.insert(
                key.into(),
                value.trim_matches('"').trim_matches('\'').into(),
            );
        }
    }
    if h.llm.provider.is_empty() {
        anyhow::ensure!(
            env.contains_key("HINDSIGHT_API_LLM_PROVIDER")
                && env.contains_key("HINDSIGHT_API_LLM_MODEL"),
            "hindsight_generative_llm_required"
        );
    } else {
        llm_env(config, &h.llm, "HINDSIGHT_API_LLM", &mut env)?;
    }
    for (op, llm) in &h.llm_overrides {
        llm_env(
            config,
            llm,
            &format!("HINDSIGHT_API_{}_LLM", op.to_ascii_uppercase()),
            &mut env,
        )?;
    }
    let url = zeroclaw_config::hindsight::validate_url(&h.api_url)?;
    for (key, value) in [
        (
            "HINDSIGHT_API_HOST",
            url.host_str().unwrap_or("127.0.0.1").to_owned(),
        ),
        (
            "HINDSIGHT_API_PORT",
            url.port_or_known_default().unwrap_or(9077).to_string(),
        ),
        (
            "HINDSIGHT_EMBED_API_VERSION",
            h.runtime.package_version.clone(),
        ),
        ("HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT", "0".into()),
        (
            "HINDSIGHT_API_DATABASE_URL",
            format!("pg0://{}", profile(config)),
        ),
        ("PG0_HOME", root.join("pg0").to_string_lossy().into_owned()),
        ("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "local".into()),
        (
            "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL",
            local_model(&root, &h.models.embedding),
        ),
        ("HINDSIGHT_API_RERANKER_PROVIDER", "local".into()),
        (
            "HINDSIGHT_API_RERANKER_LOCAL_MODEL",
            local_model(&root, &h.models.reranker),
        ),
        // Missions apply to the ordinary extraction/consolidation modes.
        // RETAIN_CUSTOM_INSTRUCTIONS is ignored unless extraction is custom.
        (
            "HINDSIGHT_API_RETAIN_MISSION",
            zeroclaw_api::memory_traits::AGENT_MEMORY_PERSPECTIVE.into(),
        ),
        (
            "HINDSIGHT_API_OBSERVATIONS_MISSION",
            zeroclaw_api::memory_traits::AGENT_MEMORY_PERSPECTIVE.into(),
        ),
        (
            "HINDSIGHT_API_REFLECT_MISSION",
            zeroclaw_api::memory_traits::AGENT_MEMORY_PERSPECTIVE.into(),
        ),
        ("HINDSIGHT_API_ENABLE_OBSERVATIONS", "true".into()),
        ("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "true".into()),
        ("HF_HUB_OFFLINE", "1".into()),
        ("UV_OFFLINE", "1".into()),
        (
            "HF_HOME",
            root.join("models").to_string_lossy().into_owned(),
        ),
        ("CUDA_VISIBLE_DEVICES", String::new()),
        ("HIP_VISIBLE_DEVICES", String::new()),
        ("OMP_NUM_THREADS", "4".into()),
        ("MKL_NUM_THREADS", "4".into()),
    ] {
        env.insert(key.into(), value);
    }
    if h.models.decision_enabled {
        env.insert(
            "HINDSIGHT_API_DECISION_MODEL_PROVIDER".into(),
            "hindsight".into(),
        );
        env.insert(
            "HINDSIGHT_API_DECISION_MODEL_BASE_URL".into(),
            h.models.decision_endpoint.clone(),
        );
        env.insert(
            "HINDSIGHT_API_DECISION_MODEL_MODEL".into(),
            h.models.decision.clone(),
        );
        env.insert(
            "HINDSIGHT_API_DECISION_MODEL_MAX_CONCURRENT".into(),
            "1".into(),
        );
        env.insert("HINDSIGHT_API_DECISION_MODEL_TIMEOUT".into(), "300".into());
        if let Some(key) = &h.models.decision_api_key {
            env.insert("HINDSIGHT_API_DECISION_MODEL_API_KEY".into(), key.clone());
        } else if local_decision_endpoint(&h.models.decision_endpoint)?
            && root.join("decision.key").exists()
        {
            check_private(&root.join("decision.key"))?;
            env.insert(
                "HINDSIGHT_API_DECISION_MODEL_API_KEY".into(),
                std::fs::read_to_string(root.join("decision.key"))?
                    .trim()
                    .into(),
            );
        }
    }
    if h.models.decision_recall_rerank {
        env.insert(
            "HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER".into(),
            "hindsight".into(),
        );
    }
    Ok(env)
}
fn local_decision_endpoint(value: &str) -> anyhow::Result<bool> {
    let url = zeroclaw_config::hindsight::validate_url(value)?;
    Ok(url.host_str().is_some_and(|host| {
        host.trim_matches(['[', ']'])
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
    }))
}
fn check_private(path: &Path) -> anyhow::Result<()> {
    let meta = std::fs::symlink_metadata(path)?;
    anyhow::ensure!(
        meta.is_file() && !meta.file_type().is_symlink(),
        "hindsight_environment_requires_regular_file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        anyhow::ensure!(
            meta.permissions().mode() & 0o077 == 0,
            "hindsight_environment_permissions_require_0600"
        );
    }
    Ok(())
}
fn private_directory(path: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}
/// Explicit setup writes the official protected profile. No package/model downloads
/// occur here or during startup; install pinned dependencies and weights first.
pub fn setup(config: &Config) -> anyhow::Result<PathBuf> {
    let env = environment(config)?;
    let root = directory(config);
    private_directory(&root)?;
    let home =
        std::env::var_os("HOME").ok_or_else(|| anyhow::Error::msg("hindsight_home_unavailable"))?;
    let profiles = PathBuf::from(home).join(".hindsight/profiles");
    private_directory(&profiles)?;
    let target = profiles.join(format!("{}.env", profile(config)));
    let temp = profiles.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp)?;
    for (key, value) in env {
        anyhow::ensure!(
            !value.contains(['\n', '\r', '\0']),
            "hindsight_invalid_environment_value"
        );
        // Official profile parser strips surrounding quotes but does not shell-evaluate.
        writeln!(file, "{key}={value}")?;
    }
    file.sync_all()?;
    std::fs::rename(temp, &target)?;
    Ok(target)
}

fn environment_hash(config: &Config) -> anyhow::Result<String> {
    Ok(hash(&serde_json::to_vec(&environment(config)?)?))
}
/// This records the environment actually used by the owned process. It is an
/// observed boot fact, never a replacement for live configuration or policy.
fn record_started(config: &Config) -> anyhow::Result<()> {
    let env = environment(config)?;
    let value = serde_json::json!({"environment_hash":environment_hash(config)?,"started_at":chrono::Utc::now().to_rfc3339(),"embedding":config.memory.hindsight.models.embedding,"reranker":config.memory.hindsight.models.reranker,"decision":config.memory.hindsight.models.decision,"llm_provider":env.get("HINDSIGHT_API_LLM_PROVIDER"),"llm_model":env.get("HINDSIGHT_API_LLM_MODEL")});
    let root = directory(config);
    let temp = root.join(format!(".runtime-{}.tmp", uuid::Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp)?;
    file.write_all(&serde_json::to_vec(&value)?)?;
    file.sync_all()?;
    std::fs::rename(temp, root.join("runtime.json"))?;
    Ok(())
}
pub fn check_applied(config: &Config) -> anyhow::Result<()> {
    if config.memory.hindsight.mode != RuntimeMode::Managed {
        return Ok(());
    }
    let path = directory(config).join("runtime.json");
    check_private(&path).map_err(|_| anyhow::Error::msg("hindsight_managed_runtime_unverified"))?;
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(bytes.len() <= 4096, "hindsight_runtime_manifest_invalid");
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        value["environment_hash"].as_str() == Some(environment_hash(config)?.as_str()),
        "hindsight_runtime_configuration_reload_required"
    );
    Ok(())
}
async fn stop_command(binary: &Path, profile: &str) -> anyhow::Result<()> {
    let result = tokio::time::timeout(
        Duration::from_secs(30),
        Command::new(binary)
            .args(["-p", profile, "daemon", "stop"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .status(),
    )
    .await;
    anyhow::ensure!(
        matches!(result,Ok(Ok(status)) if status.success()),
        "hindsight_managed_shutdown_failed"
    );
    Ok(())
}
pub struct Managed {
    lock: Option<File>,
    binary: PathBuf,
    profile: String,
    supervisor: Option<tokio::task::JoinHandle<()>>,
}
fn resolved_binary(binary: &Path) -> anyhow::Result<PathBuf> {
    if binary.is_absolute() {
        return Ok(binary.into());
    }
    anyhow::ensure!(
        binary.components().count() == 1,
        "hindsight_binary_requires_absolute_path"
    );
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .map(|p| p.join(binary))
        .find(|p| p.is_file())
        .ok_or_else(|| anyhow::Error::msg("hindsight_binary_not_found"))
}
async fn verify_packages(binary: &Path) -> anyhow::Result<()> {
    // Neither upstream executable has a side-effect-free --version switch.
    // Query Python distribution metadata in the owning virtual environment.
    let python = binary
        .parent()
        .ok_or_else(|| anyhow::Error::msg("hindsight_binary_parent_required"))?
        .join("python");
    anyhow::ensure!(
        python.is_file(),
        "hindsight_managed_requires_pinned_virtualenv"
    );
    let output=tokio::time::timeout(Duration::from_secs(10),Command::new(python).args(["-c","import importlib.metadata as m; print(m.version('hindsight-embed')); print(m.version('hindsight-api'))"]).kill_on_drop(true).output()).await??;
    anyhow::ensure!(
        output.status.success() && output.stdout.len() < 1024,
        "hindsight_package_verification_failed"
    );
    let versions = std::str::from_utf8(&output.stdout)?;
    anyhow::ensure!(
        versions.lines().count() == 2
            && versions
                .lines()
                .all(|v| v == zeroclaw_config::hindsight::TESTED_VERSION),
        "hindsight_installed_package_version_mismatch"
    );
    Ok(())
}
async fn start_command(config: &Config, binary: &Path) -> anyhow::Result<()> {
    let path = setup(config)?;
    check_private(&path)?;
    let mut command = Command::new(binary);
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("HINDSIGHT_") {
            command.env_remove(name);
        }
    }
    command
        .envs(environment(config)?)
        .args(["-p", &profile(config), "daemon", "start"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    let result = tokio::time::timeout(
        Duration::from_secs(config.memory.hindsight.startup_timeout_secs),
        command.status(),
    )
    .await;
    anyhow::ensure!(
        matches!(result,Ok(Ok(status)) if status.success()),
        "hindsight_managed_start_failed_check_profile_daemon_logs"
    );
    Ok(())
}
impl Managed {
    pub async fn start(
        live: zeroclaw_config::live::LiveConfigHandle,
    ) -> anyhow::Result<Option<Self>> {
        let config = live.read().clone();
        if config.memory.backend != "hindsight"
            || config.memory.hindsight.mode == RuntimeMode::External
        {
            return Ok(None);
        }
        config.validate_hindsight_memory()?;
        let binary = resolved_binary(&config.memory.hindsight.runtime.binary_path)?;
        verify_packages(&binary).await?;
        private_directory(&directory(&config))?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory(&config).join("managed.lock"))?;
        lock.try_lock()
            .map_err(|_| anyhow::Error::msg("hindsight_managed_instance_already_owned"))?;
        let mut owned = Self {
            lock: Some(lock),
            binary: binary.clone(),
            profile: profile(&config),
            supervisor: None,
        };
        // A previous owner may have crashed while its dedicated API survived.
        // Restart through the official manager so this process never claims
        // newly written settings were applied to an already running daemon.
        stop_command(&binary, &owned.profile).await?;
        start_command(&config, &binary).await?;
        let client = Client::new(&config.memory.hindsight)?;
        let deadline = tokio::time::Instant::now()
            + Duration::from_secs(config.memory.hindsight.startup_timeout_secs);
        loop {
            if client.health().await.is_ok() {
                break;
            }
            anyhow::ensure!(
                tokio::time::Instant::now() < deadline,
                "hindsight_managed_health_timeout"
            );
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        record_started(&config)?;
        let expected_profile = owned.profile.clone();
        owned.supervisor = Some(zeroclaw_spawn::spawn!(async move {
            let mut failures = 0u32;
            loop {
                tokio::time::sleep(Duration::from_secs(15)).await;
                let config = live.read().clone();
                if config.memory.backend != "hindsight"
                    || config.memory.hindsight.mode != RuntimeMode::Managed
                    || profile(&config) != expected_profile
                {
                    continue;
                }
                if resolved_binary(&config.memory.hindsight.runtime.binary_path)
                    .ok()
                    .as_ref()
                    != Some(&binary)
                {
                    continue;
                }
                let configured = check_applied(&config).is_ok();
                let healthy = match Client::new(&config.memory.hindsight) {
                    Ok(client) => client.health().await.is_ok(),
                    Err(_) => false,
                };
                if healthy && configured {
                    failures = 0;
                    continue;
                }
                failures = failures.saturating_add(1);
                if failures >= 3 || !configured {
                    let restarted = async {
                        stop_command(&binary, &expected_profile).await?;
                        start_command(&config, &binary).await?;
                        record_started(&config)
                    }
                    .await;
                    if restarted.is_err() {
                        zeroclaw_log::record!(
                            WARN,
                            zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Note),
                            "hindsight_managed_restart_failed"
                        );
                    }
                    tokio::time::sleep(Duration::from_secs(
                        (5u64.saturating_mul(1 << failures.min(6))).min(300),
                    ))
                    .await;
                }
            }
        }));
        Ok(Some(owned))
    }
    pub async fn shutdown(mut self) -> anyhow::Result<()> {
        if let Some(supervisor) = self.supervisor.take() {
            supervisor.abort();
            if let Err(error) = supervisor.await
                && !error.is_cancelled()
            {
                return Err(error.into());
            }
        }
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            Command::new(&self.binary)
                .args(["-p", &self.profile, "daemon", "stop"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .kill_on_drop(true)
                .status(),
        )
        .await;
        anyhow::ensure!(
            matches!(result,Ok(Ok(status)) if status.success()),
            "hindsight_managed_shutdown_failed"
        );
        drop(self.lock.take());
        Ok(())
    }
}
impl Drop for Managed {
    fn drop(&mut self) {
        if let Some(supervisor) = self.supervisor.take() {
            supervisor.abort();
        }
        let Some(lock) = self.lock.take() else {
            return;
        };
        let binary = self.binary.clone();
        let profile = self.profile.clone();
        std::thread::spawn(move || {
            let result = std::process::Command::new(binary)
                .args(["-p", &profile, "daemon", "stop"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            if !matches!(result,Ok(status) if status.success()) {
                zeroclaw_log::record!(
                    WARN,
                    zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Note),
                    "hindsight_managed_shutdown_failed"
                );
            }
            drop(lock);
        });
    }
}
pub async fn status(config: &Config) -> anyhow::Result<serde_json::Value> {
    let client = Client::new(&config.memory.hindsight)?;
    let health = client.health().await;
    let database = health
        .as_ref()
        .ok()
        .and_then(|v| v.get("database"))
        .cloned();
    let ledger = Ledger::open(&config.data_dir)?;
    let h = &config.memory.hindsight;
    let applied = if h.mode == RuntimeMode::Managed {
        Some(check_applied(config).is_ok())
    } else {
        None
    };
    let decision_available = if h.models.decision_enabled {
        let mut transport = h.clone();
        transport.api_url = h.models.decision_endpoint.clone();
        transport.mode = RuntimeMode::External;
        transport.api_key = h.models.decision_api_key.clone();
        if transport.api_key.is_none()
            && local_decision_endpoint(&h.models.decision_endpoint)?
            && directory(config).join("decision.key").exists()
        {
            let path = directory(config).join("decision.key");
            check_private(&path)?;
            transport.api_key = Some(std::fs::read_to_string(path)?.trim().into());
        }
        Some(Client::new(&transport)?.health().await.is_ok())
    } else {
        None
    };
    let owner_present = if h.mode == RuntimeMode::Managed {
        match OpenOptions::new()
            .read(true)
            .write(true)
            .open(directory(config).join("managed.lock"))
        {
            Ok(file) => match file.try_lock() {
                Ok(()) => Some(false),
                Err(std::fs::TryLockError::WouldBlock) => Some(true),
                Err(_) => None,
            },
            Err(_) => Some(false),
        }
    } else {
        None
    };
    let process_state = if h.mode == RuntimeMode::External {
        "external"
    } else if health.is_err() {
        "unavailable"
    } else if owner_present != Some(true) {
        "unowned"
    } else if applied == Some(true) {
        "running"
    } else {
        "configuration_unverified"
    };
    let env = environment(config).ok();
    let llm_provider = env
        .as_ref()
        .and_then(|e| e.get("HINDSIGHT_API_LLM_PROVIDER"));
    let llm_model = env.as_ref().and_then(|e| e.get("HINDSIGHT_API_LLM_MODEL"));
    Ok(
        serde_json::json!({"backend":config.memory.backend,"mode":h.mode,"configuration_applied":applied,"process_state":process_state,"managed_owner_present":owner_present,"managed_profile":profile(config),"api_available":health.is_ok(),"api_error":health.err().map(|e|e.to_string()),"database":database,"embedding":h.models.embedding,"reranker":h.models.reranker,"decision_model":h.models.decision,"decision_available":decision_available,"decision_recall_rerank":h.models.decision_recall_rerank,"llm_provider":llm_provider,"llm_model":llm_model,"deliveries":ledger.delivery_counts()?,"diagnostics":ledger.diagnostics()?}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protected_environment_rejects_unsupported_settings() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hindsight.env");
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path).unwrap();
        writeln!(file, "HINDSIGHT_API_LLM_UNSUPPORTED_PROVIDER=anything").unwrap();
        let mut config = Config::default();
        config.memory.hindsight.runtime.env_file = Some(path);
        assert!(
            environment(&config)
                .unwrap_err()
                .to_string()
                .contains("environment_key_not_allowed")
        );
    }
    #[test]
    fn environment_maps_separate_llm_and_local_models_without_hosted_fallback() {
        let mut c = Config::default();
        c.memory.hindsight.llm.provider = "minimax".into();
        c.memory.hindsight.llm.model = "MiniMax-M3".into();
        let env = environment(&c).unwrap();
        assert_eq!(env["HINDSIGHT_API_LLM_MODEL"], "MiniMax-M3");
        assert_eq!(env["HINDSIGHT_API_EMBEDDINGS_PROVIDER"], "local");
        assert_eq!(env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"], "0");
        assert!(!env.contains_key("HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER"));
        assert_eq!(env["HINDSIGHT_API_HOST"], "127.0.0.1");
        assert_eq!(
            env["HINDSIGHT_API_RETAIN_MISSION"],
            zeroclaw_api::memory_traits::AGENT_MEMORY_PERSPECTIVE
        );
        assert!(!env.contains_key("HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS"));
    }
}

#[cfg(test)]
mod managed_service_tests {
    use super::*;
    /// Uses an explicitly selected local installation and existing decrypted
    /// provider credentials. Never saves or switches the operator's config.
    #[tokio::test]
    #[ignore = "requires an explicitly provisioned local runtime and configured generative provider"]
    async fn real_managed_lifecycle_and_recall_ranking_benchmark() {
        let mut config = Config::load_or_init().await.unwrap();
        let data = tempfile::tempdir().unwrap();
        config.data_dir = data.path().into();
        let root = PathBuf::from(
            std::env::var_os("ZEROCLAW_TEST_HINDSIGHT_MANAGED_ROOT")
                .expect("explicit local root required"),
        );
        config.memory.backend = "hindsight".into();
        config.memory.types.enabled = false;
        config.memory.rerank_enabled = false;
        config.memory.consolidation_extract_facts = false;
        config.knowledge.enabled = false;
        config.channels.session_persistence = true;
        config.channels.session_backend = "sqlite".into();
        for agent in config.agents.values_mut() {
            agent.memory.backend = zeroclaw_config::multi_agent::MemoryBackendKind::Hindsight;
            agent.workspace.read_memory_from.clear();
        }
        let h = &mut config.memory.hindsight;
        h.mode = RuntimeMode::Managed;
        h.runtime.binary_path = root.join("runtime/bin/hindsight-embed");
        h.runtime.data_dir = root;
        h.api_url =
            std::env::var("ZEROCLAW_TEST_HINDSIGHT_URL").expect("explicit service URL required");
        h.llm.provider =
            std::env::var("ZEROCLAW_TEST_HINDSIGHT_LLM_PROVIDER").expect("provider required");
        h.llm.model = std::env::var("ZEROCLAW_TEST_HINDSIGHT_LLM_MODEL").expect("model required");
        h.llm.credential_provider = std::env::var("ZEROCLAW_TEST_HINDSIGHT_LLM_REF")
            .expect("existing credential reference required")
            .into();
        h.models.decision_recall_rerank = false;
        let live = zeroclaw_config::live::LiveConfig::new(config.clone());
        let owned = Managed::start(live.handle()).await.unwrap().unwrap();
        assert!(
            Managed::start(live.handle()).await.is_err(),
            "a second process owner must not be admitted"
        );
        assert!(check_applied(&config).is_ok());
        let client = Client::new(&config.memory.hindsight).unwrap();
        let bank = format!("zeroclaw-managed-fixture-{}", uuid::Uuid::new_v4());
        client.ensure_bank(&bank).await.unwrap();
        // Verify the real upstream prompt assembly, not only environment names.
        for operation in ["retain", "consolidation", "reflect"] {
            let preview = client
                .request(
                    reqwest::Method::POST,
                    &["v1", "default", "banks", &bank, "prompts", "preview"],
                    Some(&serde_json::json!({"operation":operation})),
                    false,
                )
                .await
                .unwrap();
            assert!(
                preview
                    .to_string()
                    .contains(zeroclaw_api::memory_traits::AGENT_MEMORY_PERSPECTIVE),
                "perspective mission missing from real {operation} prompt"
            );
        }
        let operation = uuid::Uuid::new_v4().to_string();
        let item=super::super::client::Item {content:"User: I decided the fictional Larch project uses SQLite for its permanent conversation archive. It keeps the deployment lightweight. Assistant: I will keep its rolling summaries separate from long-term memory.".into(),document_id:"fixture-turn".into(),timestamp:"2026-01-01T00:00:00Z".into(),context:"Fictional architecture test conversation".into(),metadata:Default::default()};
        client.retain(&bank, &[item], &operation).await.unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
        loop {
            let operation = client.operation(&bank, &operation).await.unwrap();
            if operation.status == "completed" {
                break;
            }
            assert_ne!(operation.status, "failed");
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        let mut report = Vec::new();
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let rows = client
                .recall(&bank, "Larch conversation storage design", "mid", 512, true)
                .await
                .unwrap();
            assert!(!rows.results.is_empty());
            report.push(serde_json::json!({"mode":"ordinary_reranker","milliseconds":started.elapsed().as_millis(),"results":rows.results.len()}));
        }
        owned.shutdown().await.unwrap();
        config.memory.hindsight.models.decision_recall_rerank = true;
        live.publish(live.next_revision().unwrap(), config.clone())
            .unwrap();
        let owned = Managed::start(live.handle()).await.unwrap().unwrap();
        assert!(check_applied(&config).is_ok());
        let client = Client::new(&config.memory.hindsight).unwrap();
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let rows = client
                .recall(&bank, "Larch conversation storage design", "mid", 512, true)
                .await
                .unwrap();
            assert!(!rows.results.is_empty());
            report.push(serde_json::json!({"mode":"decision_reranker","milliseconds":started.elapsed().as_millis(),"results":rows.results.len()}));
        }
        client
            .request(
                reqwest::Method::DELETE,
                &["v1", "default", "banks", &bank],
                None,
                false,
            )
            .await
            .unwrap();
        owned.shutdown().await.unwrap();
        // Restore the recommended ordinary-recall profile for subsequent tests.
        config.memory.hindsight.models.decision_recall_rerank = false;
        setup(&config).unwrap();
        eprintln!("{}", serde_json::to_string(&report).unwrap());
    }
}
