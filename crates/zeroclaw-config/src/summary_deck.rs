//! Runtime-profile policy for derived, rotating conversation summaries.
use crate::providers::ModelProviderRef;
use serde::{Deserialize, Serialize};
use zeroclaw_macros::Configurable;

#[derive(Clone, Debug, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "runtime_profile.summary_deck"]
#[serde(default)]
pub struct SummaryDeckConfig {
    /// Summarize completed turns in the background. Requires native session persistence.
    pub enabled: bool,
    /// Fraction of the available history budget reserved for summaries (0 < value < 1).
    pub summary_share: f64,
    /// Start summarization at this fraction of the recent-history allocation.
    pub trigger_ratio: f64,
    /// Maximum estimated tokens in one generated summary; provenance headers also count against the deck allocation.
    pub max_summary_tokens: usize,
    /// Bound each background inference attempt, in seconds.
    pub timeout_secs: u64,
    /// Optional configured model-provider reference; empty uses the session's current route.
    pub provider: ModelProviderRef,
}

impl Default for SummaryDeckConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            summary_share: 0.15,
            trigger_ratio: 0.8,
            max_summary_tokens: 1_000,
            timeout_secs: 60,
            provider: ModelProviderRef::default(),
        }
    }
}

impl SummaryDeckConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.summary_share > 0.0 && self.summary_share < 1.0,
            "summary_deck.summary_share must be between 0 and 1"
        );
        anyhow::ensure!(
            self.trigger_ratio > 0.0 && self.trigger_ratio < 1.0,
            "summary_deck.trigger_ratio must be between 0 and 1"
        );
        anyhow::ensure!(
            self.max_summary_tokens >= 64,
            "summary_deck.max_summary_tokens must be at least 64"
        );
        anyhow::ensure!(
            self.timeout_secs > 0 && self.timeout_secs <= 600,
            "summary_deck.timeout_secs must be between 1 and 600"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_deck_defaults_and_invalid_budgets() {
        let policy = SummaryDeckConfig::default();
        assert!(!policy.enabled);
        assert!(policy.validate().is_ok());
        for share in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(
                SummaryDeckConfig {
                    summary_share: share,
                    ..policy.clone()
                }
                .validate()
                .is_err()
            );
        }
        for ratio in [0.0, 1.0, -0.1, f64::NAN] {
            assert!(
                SummaryDeckConfig {
                    trigger_ratio: ratio,
                    ..policy.clone()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            SummaryDeckConfig {
                max_summary_tokens: 0,
                ..policy.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            SummaryDeckConfig {
                timeout_secs: 0,
                ..policy
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn summary_deck_is_owned_by_the_runtime_profile() {
        let profile: crate::schema::RuntimeProfileConfig = toml::from_str(
            "[summary_deck]\nenabled = true\nsummary_share = 0.2\nprovider = 'custom.summary'\n",
        )
        .unwrap();
        assert!(profile.summary_deck.enabled);
        assert_eq!(profile.summary_deck.summary_share, 0.2);
        let restored: crate::schema::RuntimeProfileConfig =
            toml::from_str(&toml::to_string(&profile).unwrap()).unwrap();
        assert_eq!(restored.summary_deck.provider.trim(), "custom.summary");
    }
}
