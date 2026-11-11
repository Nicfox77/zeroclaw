//! Canonical configuration for the exclusive Hindsight memory backend.
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf};
use zeroclaw_macros::Configurable;

pub const TESTED_VERSION: &str = "1.2.0";

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    #[default]
    Managed,
    External,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ReflectMode {
    Off,
    #[default]
    Adaptive,
    Always,
}
impl crate::traits::HasPropKind for RuntimeMode {
    const PROP_KIND: crate::traits::PropKind = crate::traits::PropKind::Enum;
}
impl crate::traits::HasPropKind for ReflectMode {
    const PROP_KIND: crate::traits::PropKind = crate::traits::PropKind::Enum;
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "memory.hindsight.llm"]
#[serde(default)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub base_url: Option<String>,
    /// Resolve only credentials from this existing model-provider alias at use time.
    /// Generative provider, model and endpoint remain independently configured above.
    #[credential_class = "public_value"]
    pub credential_provider: crate::providers::ModelProviderRef,
    #[secret]
    #[credential_class = "encrypted_secret"]
    #[cfg_attr(feature = "schema-export", schemars(extend("x-secret" = true)))]
    pub api_key: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "memory.hindsight.models"]
#[serde(default)]
pub struct Models {
    pub embedding: String,
    pub reranker: String,
    pub decision: String,
    pub decision_enabled: bool,
    pub decision_recall_rerank: bool,
    pub decision_endpoint: String,
    #[secret]
    #[credential_class = "encrypted_secret"]
    #[cfg_attr(feature = "schema-export", schemars(extend("x-secret" = true)))]
    pub decision_api_key: Option<String>,
}
impl Default for Models {
    fn default() -> Self {
        Self {
            embedding: "hindsight-ai/hs-embed-1".into(),
            reranker: "hindsight-ai/hs-rerank-1".into(),
            decision: "hs-decide-1-flash".into(),
            decision_enabled: true,
            decision_recall_rerank: false,
            decision_endpoint: "http://127.0.0.1:8000".into(),
            decision_api_key: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "memory.hindsight.runtime"]
#[serde(default)]
pub struct Runtime {
    pub binary_path: PathBuf,
    /// Empty resolves beneath the installation data directory, never the current directory.
    pub data_dir: PathBuf,
    pub package_version: String,
    /// Optional protected environment file. Runtime-generated secret files are mode 0600.
    #[credential_class = "path_only_reference"]
    pub env_file: Option<PathBuf>,
}
impl Default for Runtime {
    fn default() -> Self {
        Self {
            binary_path: "hindsight-embed".into(),
            data_dir: PathBuf::new(),
            package_version: TESTED_VERSION.into(),
            env_file: None,
        }
    }
}

/// An explicitly configured durable knowledge page; unrelated to rolling summaries.
#[derive(Clone, Debug, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "memory.hindsight.mental_models"]
#[serde(default)]
pub struct MentalModel {
    pub name: String,
    pub source_query: String,
    pub max_tokens: usize,
    pub min_refresh_interval_seconds: u64,
}
impl Default for MentalModel {
    fn default() -> Self {
        Self {
            name: String::new(),
            source_query: String::new(),
            max_tokens: 2048,
            min_refresh_interval_seconds: 3600,
        }
    }
}

/// Admission policy for new Hindsight retention work; transcript capture is independent.
#[derive(Clone, Debug, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "memory.hindsight.retention"]
#[serde(default)]
pub struct Retention {
    /// Queue writes for the worker rather than extracting inside a chat turn.
    pub deferred: bool,
    /// Optional daily HH:MM-HH:MM window, start inclusive and end exclusive.
    /// Without a window, deferred retention requires a manual drain request.
    pub window: Option<String>,
    /// IANA timezone for the daily window, including daylight-saving transitions.
    pub timezone: String,
}
impl Default for Retention {
    fn default() -> Self {
        Self {
            deferred: false,
            window: None,
            timezone: "UTC".into(),
        }
    }
}
impl Retention {
    fn bounds(&self) -> anyhow::Result<Option<(chrono::NaiveTime, chrono::NaiveTime)>> {
        self.window
            .as_deref()
            .map(|window| {
                let (start, end) = window
                    .split_once('-')
                    .ok_or_else(|| anyhow::Error::msg("hindsight_invalid_retention_window"))?;
                let parse = |value: &str| -> anyhow::Result<chrono::NaiveTime> {
                    anyhow::ensure!(value.len() == 5, "hindsight_invalid_retention_window");
                    chrono::NaiveTime::parse_from_str(value, "%H:%M")
                        .map_err(|_| anyhow::Error::msg("hindsight_invalid_retention_window"))
                };
                let start = parse(start)?;
                let end = parse(end)?;
                anyhow::ensure!(start != end, "hindsight_invalid_retention_window");
                Ok((start, end))
            })
            .transpose()
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        self.timezone
            .parse::<chrono_tz::Tz>()
            .map_err(|_| anyhow::Error::msg("hindsight_invalid_retention_timezone"))?;
        self.bounds()?;
        Ok(())
    }
    pub fn allows_at(&self, now: chrono::DateTime<chrono::Utc>) -> anyhow::Result<bool> {
        if !self.deferred {
            return Ok(true);
        }
        let Some((start, end)) = self.bounds()? else {
            return Ok(false);
        };
        let zone = self
            .timezone
            .parse::<chrono_tz::Tz>()
            .map_err(|_| anyhow::Error::msg("hindsight_invalid_retention_timezone"))?;
        let time = now.with_timezone(&zone).time();
        Ok(if start < end {
            time >= start && time < end
        } else {
            time >= start || time < end
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "memory.hindsight"]
#[serde(default)]
pub struct HindsightConfig {
    pub mode: RuntimeMode,
    pub api_url: String,
    #[secret]
    #[credential_class = "encrypted_secret"]
    #[cfg_attr(feature = "schema-export", schemars(extend("x-secret" = true)))]
    pub api_key: Option<String>,
    /// Human-readable prefix; a collision-resistant scope digest is always appended.
    pub bank_template: String,
    pub startup_timeout_secs: u64,
    pub request_timeout_secs: u64,
    pub reflect_timeout_secs: u64,
    pub auto_retain: bool,
    #[nested]
    pub retention: Retention,
    pub auto_recall: bool,
    pub auto_reflect: bool,
    pub reflect_mode: ReflectMode,
    pub recall_budget: String,
    pub recall_max_tokens: usize,
    pub reflect_budget: String,
    pub prefer_observations: bool,
    /// Bridge bounded canonical workspace identity into operator-bank missions.
    /// Other principals never receive the installation owner's profile.
    pub identity_context: bool,
    pub retain_async: bool,
    pub retry_failed_retains: bool,
    pub queue_capacity: usize,
    #[nested]
    pub models: Models,
    #[nested]
    pub runtime: Runtime,
    #[nested]
    pub llm: LlmConfig,
    /// Upstream operation names: retain, reflect, consolidation.
    #[nested]
    pub llm_overrides: HashMap<String, LlmConfig>,
    #[nested]
    pub mental_models: HashMap<String, MentalModel>,
}
impl Default for HindsightConfig {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Managed,
            api_url: "http://127.0.0.1:9077".into(),
            api_key: None,
            bank_template: "zeroclaw-{agent}".into(),
            startup_timeout_secs: 120,
            request_timeout_secs: 15,
            reflect_timeout_secs: 120,
            auto_retain: true,
            retention: Retention::default(),
            auto_recall: true,
            auto_reflect: true,
            reflect_mode: ReflectMode::Adaptive,
            recall_budget: "mid".into(),
            recall_max_tokens: 2048,
            reflect_budget: "mid".into(),
            prefer_observations: true,
            identity_context: false,
            retain_async: true,
            retry_failed_retains: true,
            queue_capacity: 10_000,
            models: Models::default(),
            runtime: Runtime::default(),
            llm: LlmConfig::default(),
            llm_overrides: HashMap::new(),
            mental_models: HashMap::new(),
        }
    }
}

pub fn validate_url(value: &str) -> anyhow::Result<url::Url> {
    let url = url::Url::parse(value).map_err(|_| anyhow::Error::msg("hindsight_invalid_url"))?;
    anyhow::ensure!(
        url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "hindsight_url_credentials_or_parameters_forbidden"
    );
    let local = match url.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        _ => false,
    };
    anyhow::ensure!(
        url.scheme() == "https" || (url.scheme() == "http" && local),
        "hindsight_requires_https_or_literal_loopback"
    );
    anyhow::ensure!(url.host().is_some(), "hindsight_missing_host");
    Ok(url)
}
impl HindsightConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        self.retention.validate()?;
        let url = validate_url(&self.api_url)?;
        anyhow::ensure!(
            url.path() == "/" || url.path().is_empty(),
            "hindsight_api_url_requires_origin"
        );
        anyhow::ensure!(
            self.mode != RuntimeMode::Managed || self.api_key.is_none(),
            "hindsight_managed_api_auth_not_supported"
        );
        if !self.runtime.data_dir.as_os_str().is_empty() {
            anyhow::ensure!(
                self.runtime.data_dir.is_absolute(),
                "hindsight_runtime_directory_requires_absolute_path"
            );
        }
        if let Some(path) = &self.runtime.env_file {
            anyhow::ensure!(
                path.is_absolute(),
                "hindsight_environment_requires_absolute_path"
            );
        }
        anyhow::ensure!(
            self.retain_async,
            "hindsight_durable_delivery_requires_async_retains"
        );
        anyhow::ensure!(
            self.bank_template.len() <= 80,
            "hindsight_bank_template_too_long"
        );
        for model in [
            &self.models.embedding,
            &self.models.reranker,
            &self.models.decision,
        ] {
            anyhow::ensure!(
                !model.trim().is_empty() && model.len() <= 2048,
                "hindsight_model_identifier_required"
            );
        }
        if self.mode == RuntimeMode::Managed {
            anyhow::ensure!(
                url.scheme() == "http",
                "hindsight_managed_api_requires_loopback_http"
            );
            anyhow::ensure!(
                matches!(url.host(), Some(url::Host::Ipv4(ip)) if ip.is_loopback())
                    || matches!(url.host(), Some(url::Host::Ipv6(ip)) if ip.is_loopback()),
                "hindsight_managed_requires_loopback"
            );
            anyhow::ensure!(
                self.runtime.package_version == TESTED_VERSION,
                "hindsight_untested_runtime_version"
            );
        }
        anyhow::ensure!(
            self.bank_template.contains("{agent}")
                && self
                    .bank_template
                    .replace("{agent}", "")
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "hindsight_invalid_bank_template"
        );
        for seconds in [
            self.startup_timeout_secs,
            self.request_timeout_secs,
            self.reflect_timeout_secs,
        ] {
            anyhow::ensure!(seconds > 0 && seconds <= 86400, "hindsight_invalid_timeout");
        }
        anyhow::ensure!(
            (1..=16384).contains(&self.recall_max_tokens)
                && (1..=100_000).contains(&self.queue_capacity),
            "hindsight_invalid_budget_or_capacity"
        );
        for budget in [&self.recall_budget, &self.reflect_budget] {
            anyhow::ensure!(
                ["low", "mid", "high"].contains(&budget.as_str()),
                "hindsight_invalid_budget"
            );
        }
        anyhow::ensure!(
            !self.models.decision_recall_rerank || self.models.decision_enabled,
            "hindsight_decision_recall_requires_decision_model"
        );
        let decision_url = validate_url(&self.models.decision_endpoint)?;
        anyhow::ensure!(
            decision_url.path() == "/" || decision_url.path().is_empty(),
            "hindsight_decision_endpoint_requires_origin"
        );
        anyhow::ensure!(
            self.mental_models.len() <= 32,
            "hindsight_too_many_mental_models"
        );
        for (id, model) in &self.mental_models {
            anyhow::ensure!(
                !id.is_empty()
                    && id.len() <= 64
                    && id
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "hindsight_invalid_mental_model_id"
            );
            anyhow::ensure!(
                !model.name.trim().is_empty()
                    && !model.source_query.trim().is_empty()
                    && model.source_query.len() <= 8192
                    && (256..=8192).contains(&model.max_tokens)
                    && model.min_refresh_interval_seconds >= 60,
                "hindsight_invalid_mental_model"
            );
        }
        for operation in self.llm_overrides.keys() {
            anyhow::ensure!(
                ["retain", "reflect", "consolidation", "mental_model_refresh"]
                    .contains(&operation.as_str()),
                "hindsight_invalid_llm_operation"
            );
        }
        for llm in std::iter::once(&self.llm).chain(self.llm_overrides.values()) {
            anyhow::ensure!(
                llm.api_key.is_none() || llm.credential_provider.is_empty(),
                "hindsight_ambiguous_llm_credentials"
            );
            anyhow::ensure!(
                llm.provider.is_empty() == llm.model.is_empty(),
                "hindsight_llm_provider_and_model_required_together"
            );
            if let Some(url) = &llm.base_url {
                validate_url(url)?;
            }
        }
        Ok(())
    }
}

/// Private-memory identity only: this never creates an RPC principal or grant.
/// A linked personal connection explicitly shares the local operator's memory;
/// other admitted peers are isolated by adapter type, persistent alias and ID.
pub fn channel_principal(
    config: &crate::schema::Config,
    agent: &str,
    kind: &str,
    alias: &str,
    sender: &str,
    linked: bool,
) -> anyhow::Result<String> {
    anyhow::ensure!(
        !sender.is_empty() && !alias.is_empty(),
        "hindsight_channel_identity_required"
    );
    let channel_key = format!("{kind}.{alias}");
    anyhow::ensure!(
        config.agent_for_channel(&channel_key) == Some(agent),
        "hindsight_channel_agent_revoked"
    );
    anyhow::ensure!(
        config
            .get_prop(&format!("channels.{kind}.{alias}.enabled"))
            .is_ok_and(|v| v == "true"),
        "hindsight_channel_disabled"
    );
    let peers = config.channel_external_peers(kind, alias);
    anyhow::ensure!(
        crate::schema::peer_policy_admits(&peers, &[sender], |a, b| a == b),
        "hindsight_channel_principal_revoked"
    );
    let policy = &config
        .agent(agent)
        .ok_or_else(|| anyhow::Error::msg("hindsight_agent_revoked"))?
        .main_session;
    let binding = match kind {
        "telegram" => policy.telegram.get(alias),
        "simplex" => policy.simplex.get(alias),
        _ => None,
    };
    if linked && policy.enabled && binding.is_some_and(|id| id == sender) {
        Ok(zeroclaw_api::principal::PrincipalId::SHARED_OPERATOR.into())
    } else {
        Ok(format!(
            "channel-peer:{}",
            serde_json::to_string(&(kind, alias, sender))?
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boxed_settings_preserve_schema_properties_and_secret_metadata() {
        let mut config = crate::schema::Config::default();
        config
            .set_prop("memory.hindsight.auto_recall", "false")
            .unwrap();
        assert!(!config.memory.hindsight.auto_recall);
        assert_eq!(
            config.get_prop("memory.hindsight.auto_recall").unwrap(),
            "false"
        );
        assert!(crate::schema::Config::prop_is_secret(
            "memory.hindsight.llm.api_key"
        ));
        let text = toml::to_string(&config).unwrap();
        let restored: crate::schema::Config = toml::from_str(&text).unwrap();
        assert!(!restored.memory.hindsight.auto_recall);
    }
    #[test]
    fn defaults_and_boundary_validation() {
        let mut c = HindsightConfig::default();
        c.validate().unwrap();
        for u in [
            "http://example.com",
            "http://localhost:9077",
            "http://127.0.0.1@evil.com",
            "http://127.0.0.1/?key=secret",
            "file:///tmp/test",
        ] {
            assert!(validate_url(u).is_err(), "{u}");
        }
        c.bank_template = "shared".into();
        assert!(c.validate().is_err());
        c.bank_template = "test-{agent}".into();
        c.models.decision_enabled = false;
        c.models.decision_recall_rerank = true;
        assert!(c.validate().is_err());
    }
}

#[cfg(test)]
mod retention_tests {
    use super::Retention;
    fn at(value: &str) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339(value)
            .unwrap()
            .with_timezone(&chrono::Utc)
    }
    #[test]
    fn deferred_window_tracks_pacific_dst_and_excludes_closing_time() {
        let policy = Retention {
            deferred: true,
            window: Some("04:00-08:00".into()),
            timezone: "America/Los_Angeles".into(),
        };
        policy.validate().unwrap();
        for (time, allowed) in [
            ("2026-01-10T11:59:59Z", false),
            ("2026-01-10T12:00:00Z", true),
            ("2026-01-10T15:59:59Z", true),
            ("2026-01-10T16:00:00Z", false),
            ("2026-07-10T10:59:59Z", false),
            ("2026-07-10T11:00:00Z", true),
            ("2026-07-10T14:59:59Z", true),
            ("2026-07-10T15:00:00Z", false),
            ("2026-03-08T11:00:00Z", true),
            ("2026-11-01T12:00:00Z", true),
        ] {
            assert_eq!(policy.allows_at(at(time)).unwrap(), allowed, "{time}");
        }
    }
    #[test]
    fn manual_only_and_overnight_windows_have_distinct_admission() {
        let mut policy = Retention {
            deferred: true,
            ..Default::default()
        };
        assert!(!policy.allows_at(at("2026-01-01T12:00:00Z")).unwrap());
        policy.window = Some("23:00-02:00".into());
        for (time, allowed) in [
            ("2026-01-01T23:00:00Z", true),
            ("2026-01-02T01:59:59Z", true),
            ("2026-01-02T02:00:00Z", false),
            ("2026-01-02T12:00:00Z", false),
        ] {
            assert_eq!(policy.allows_at(at(time)).unwrap(), allowed);
        }
        assert!(
            Retention::default()
                .allows_at(at("2026-01-01T12:00:00Z"))
                .unwrap()
        );
    }
    #[test]
    fn invalid_retention_policy_is_rejected_and_nested_props_roundtrip() {
        for window in [
            "04:00-04:00",
            "25:00-08:00",
            "4:00-08:00",
            "04:00-08:00-extra",
            "04:00",
        ] {
            let policy = Retention {
                window: Some(window.into()),
                ..Default::default()
            };
            assert!(policy.validate().is_err(), "{window}");
        }
        assert!(
            Retention {
                timezone: "invalid-zone".into(),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        let mut config = crate::schema::Config::default();
        config
            .set_prop("memory.hindsight.retention.deferred", "true")
            .unwrap();
        config
            .set_prop("memory.hindsight.retention.window", "04:00-08:00")
            .unwrap();
        config
            .set_prop("memory.hindsight.retention.timezone", "America/Los_Angeles")
            .unwrap();
        let restored: crate::schema::Config =
            toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        assert!(restored.memory.hindsight.retention.deferred);
        assert_eq!(
            restored.memory.hindsight.retention.window.as_deref(),
            Some("04:00-08:00")
        );
    }
}
