//! Explicit routing bindings for one personal conversation across transports.
//! Membership here joins histories; existing transport authentication and peer
//! allowlists still decide whether an input may execute.
use crate::schema::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zeroclaw_api::channel::ChannelMessage;
use zeroclaw_macros::Configurable;

#[derive(Clone, Debug, Default, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "agent.main_session"]
#[serde(default)]
pub struct MainSessionConfig {
    /// Route explicitly linked connections to one durable conversation for this agent.
    pub enabled: bool,
    /// Join authenticated Web operator connections. This does not grant gateway access.
    pub web: bool,
    /// Telegram bot alias to stable numeric owner user ID. Only that user's direct chat joins.
    pub telegram: HashMap<String, String>,
}

/// The agent identity creates the canonical personal-conversation identity.
/// No transport maintains an independently editable destination or alias map.
pub fn session_id(agent: &str) -> String {
    format!("main-session-{agent}")
}

pub fn is_main_session(config: &Config, session: &str) -> bool {
    config
        .agents
        .iter()
        .any(|(alias, agent)| agent.main_session.enabled && session_id(alias) == session)
}

pub fn web_session(config: &Config, agent: &str) -> Option<String> {
    let policy = &config.agent(agent)?.main_session;
    (policy.enabled && policy.web).then(|| session_id(agent))
}

pub fn channel_session(config: &Config, agent: &str, message: &ChannelMessage) -> Option<String> {
    let policy = &config.agent(agent)?.main_session;
    if !policy.enabled
        || message.channel != "telegram"
        || message.passive_context
        || message.thread_ts.is_some()
    {
        return None;
    }
    let owner = policy.telegram.get(message.channel_alias.as_deref()?)?;
    // Use the adapter-stamped platform ID, never a mutable display username.
    (message.platform_sender_id.as_deref() == Some(owner.as_str())
        && message.reply_target == *owner)
        .then(|| session_id(agent))
}

impl MainSessionConfig {
    pub fn validate(&self, config: &Config, agent: &str) -> anyhow::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        anyhow::ensure!(
            self.web || !self.telegram.is_empty(),
            "main_session_requires_explicit_connection_bindings"
        );
        anyhow::ensure!(
            !self.web || config.gateway.session_persistence,
            "main_session_requires_gateway_persistence"
        );
        anyhow::ensure!(
            !self.web || config.gateway.require_pairing,
            "main_session_web_requires_authenticated_gateway"
        );
        anyhow::ensure!(
            self.telegram.is_empty() || config.channels.session_persistence,
            "main_session_requires_channel_persistence"
        );
        for (alias, owner) in &self.telegram {
            anyhow::ensure!(
                owner.parse::<u64>().is_ok_and(|id| id > 0)
                    && owner.bytes().all(|byte| byte.is_ascii_digit()),
                "main_session_telegram_requires_stable_numeric_user_id"
            );
            anyhow::ensure!(
                config.agent_for_channel(&format!("telegram.{alias}")) == Some(agent),
                "main_session_telegram_binding_must_belong_to_the_agent"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::AliasedAgentConfig;

    fn config() -> Config {
        let mut config = Config::default();
        config.agents.insert(
            "assistant".into(),
            AliasedAgentConfig {
                channels: vec!["telegram.personal".into()],
                main_session: MainSessionConfig {
                    enabled: true,
                    web: true,
                    telegram: HashMap::from([("personal".into(), "10001".into())]),
                },
                ..Default::default()
            },
        );
        config
    }

    fn message() -> ChannelMessage {
        ChannelMessage {
            id: "fixture".into(),
            sender: "mutable-name".into(),
            platform_sender_id: Some("10001".into()),
            reply_target: "10001".into(),
            content: "fixture message".into(),
            channel: "telegram".into(),
            channel_alias: Some("personal".into()),
            timestamp: 0,
            thread_ts: None,
            interruption_scope_id: None,
            attachments: vec![],
            subject: None,
            internal_sop_event: None,
            passive_context: false,
            explicitly_addressed: true,
            conversation_scope: zeroclaw_api::channel::ChannelConversationScope::Sender,
            references: vec![],
            voice_origin: false,
        }
    }

    #[test]
    fn only_explicit_private_connections_join_the_main_conversation() {
        let mut config = config();
        let original = message();
        assert_eq!(
            web_session(&config, "assistant"),
            channel_session(&config, "assistant", &original)
        );
        let mut other = original.clone();
        other.platform_sender_id = Some("10002".into());
        assert!(channel_session(&config, "assistant", &other).is_none());
        let mut group = original.clone();
        group.reply_target = "-10003".into();
        assert!(channel_session(&config, "assistant", &group).is_none());
        let mut unstamped = original.clone();
        unstamped.platform_sender_id = None;
        unstamped.sender = "10001".into();
        assert!(channel_session(&config, "assistant", &unstamped).is_none());
        let mut different_bot = original.clone();
        different_bot.channel_alias = Some("other".into());
        assert!(channel_session(&config, "assistant", &different_bot).is_none());
        config
            .agents
            .get_mut("assistant")
            .unwrap()
            .main_session
            .enabled = false;
        assert!(web_session(&config, "assistant").is_none());
        assert!(channel_session(&config, "assistant", &original).is_none());
    }

    #[test]
    fn main_session_validation_rejects_wildcards_and_unowned_bots() {
        let mut config = config();
        assert!(
            config.agents["assistant"]
                .main_session
                .validate(&config, "assistant")
                .is_ok()
        );
        for identity in ["*", "@fixture", "0", "10001 "] {
            config
                .agents
                .get_mut("assistant")
                .unwrap()
                .main_session
                .telegram
                .insert("personal".into(), identity.into());
            assert!(
                config.agents["assistant"]
                    .main_session
                    .validate(&config, "assistant")
                    .is_err()
            );
        }
    }
}
