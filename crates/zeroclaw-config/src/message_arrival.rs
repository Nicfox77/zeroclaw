//! Message arrival policy. An explicit mode is authoritative; the old boolean
//! is consulted only when no mode is configured.
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, zeroclaw_macros::ConfigEnum,
)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum MessageArrivalMode {
    /// Run each message as a separate turn, in receive order.
    Queue,
    /// Cancel the active turn when a newer message arrives.
    Interrupt,
    /// Deliver compatible follow-ups between tool rounds of the active turn.
    Steer,
}

impl MessageArrivalMode {
    pub fn resolve(explicit: Option<Self>, legacy_interrupt: bool) -> Self {
        explicit.unwrap_or(if legacy_interrupt {
            Self::Interrupt
        } else {
            Self::Queue
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_modes_override_only_the_legacy_fallback() {
        assert_eq!(
            MessageArrivalMode::resolve(None, false),
            MessageArrivalMode::Queue
        );
        assert_eq!(
            MessageArrivalMode::resolve(None, true),
            MessageArrivalMode::Interrupt
        );
        for mode in [
            MessageArrivalMode::Queue,
            MessageArrivalMode::Interrupt,
            MessageArrivalMode::Steer,
        ] {
            for legacy in [false, true] {
                assert_eq!(MessageArrivalMode::resolve(Some(mode), legacy), mode);
            }
        }
    }

    #[test]
    fn supported_config_controls_set_enum_modes() {
        let mut config = crate::schema::Config::default();
        config
            .channels
            .simplex
            .insert("fixture".into(), crate::simplex::SimplexConfig::default());
        config
            .channels
            .telegram
            .insert("fixture".into(), crate::schema::TelegramConfig::default());
        for channel in ["simplex", "telegram"] {
            let path = format!("channels.{channel}.fixture.message_arrival");
            config.set_prop(&path, "steer").unwrap();
            assert!(config.set_prop(&path, "invalid-mode").is_err());
        }
        let encoded = toml::to_string(&config).unwrap();
        let restored: crate::schema::Config = toml::from_str(&encoded).unwrap();
        assert_eq!(
            restored.channels.simplex["fixture"].message_arrival,
            Some(MessageArrivalMode::Steer)
        );
        assert_eq!(
            restored.channels.telegram["fixture"].message_arrival,
            Some(MessageArrivalMode::Steer)
        );
    }

    #[test]
    fn channel_mode_survives_serde_with_legacy_configs() {
        for mode in ["queue", "interrupt", "steer"] {
            let config: crate::simplex::SimplexConfig = serde_json::from_value(serde_json::json!({
                "message_arrival": mode, "interrupt_on_new_message": true,
            }))
            .unwrap();
            assert_eq!(
                serde_json::to_value(config).unwrap()["message_arrival"],
                mode
            );
        }
        let config: crate::simplex::SimplexConfig =
            serde_json::from_value(serde_json::json!({"interrupt_on_new_message": true})).unwrap();
        assert_eq!(config.message_arrival, None);
        assert!(
            serde_json::from_value::<crate::simplex::SimplexConfig>(
                serde_json::json!({"message_arrival":"typo"})
            )
            .is_err()
        );
    }
}
