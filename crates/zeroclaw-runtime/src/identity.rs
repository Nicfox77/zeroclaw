//! Compatibility facade for the shared identity parser and renderer.
use anyhow::Result;
use std::path::Path;
use zeroclaw_config::schema::IdentityConfig;
pub use zeroclaw_infra::aieos::{
    AieosIdentity, CapabilitiesSection, HistorySection, IdentitySection, InterestsSection,
    LinguisticsSection, MotivationsSection, Names, OceanTraits, PhysicalitySection,
    PsychologySection, aieos_to_system_prompt,
};

pub fn load_aieos_identity(
    config: &IdentityConfig,
    workspace_dir: &Path,
) -> Result<Option<AieosIdentity>> {
    zeroclaw_infra::aieos::load_aieos_identity(
        &config.format,
        config.aieos_path.as_deref(),
        config.aieos_inline.as_deref(),
        workspace_dir,
    )
}

pub fn is_aieos_configured(config: &IdentityConfig) -> bool {
    config.format == "aieos" && (config.aieos_path.is_some() || config.aieos_inline.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_aieos_configured_true_with_path() {
        let config = IdentityConfig {
            format: "aieos".into(),
            aieos_path: Some("identity.json".into()),
            aieos_inline: None,
        };
        assert!(is_aieos_configured(&config));
    }

    #[test]
    fn is_aieos_configured_true_with_inline() {
        let config = IdentityConfig {
            format: "aieos".into(),
            aieos_path: None,
            aieos_inline: Some("{\"identity\":{}}".into()),
        };
        assert!(is_aieos_configured(&config));
    }

    #[test]
    fn is_aieos_configured_false_openclaw_format() {
        let config = IdentityConfig {
            format: "openclaw".into(),
            aieos_path: Some("identity.json".into()),
            aieos_inline: None,
        };
        assert!(!is_aieos_configured(&config));
    }

    #[test]
    fn is_aieos_configured_false_no_config() {
        let config = IdentityConfig {
            format: "aieos".into(),
            aieos_path: None,
            aieos_inline: None,
        };
        assert!(!is_aieos_configured(&config));
    }

    #[test]
    fn load_aieos_identity_from_file_supports_generator_shape() {
        let json = r#"{
            "identity": {
                "names": { "first": "Nova" },
                "bio": { "gender": "Non-binary" }
            },
            "psychology": {
                "traits": { "mbti": "ENTP" },
                "moral_compass": { "alignment": "Chaotic Good" }
            }
        }"#;

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("identity.json");
        std::fs::write(&path, json).unwrap();

        let config = IdentityConfig {
            format: "aieos".into(),
            aieos_path: Some("identity.json".into()),
            aieos_inline: None,
        };

        let identity = load_aieos_identity(&config, temp.path()).unwrap().unwrap();
        assert_eq!(
            identity.identity.unwrap().names.unwrap().first.as_deref(),
            Some("Nova")
        );
        assert_eq!(identity.psychology.unwrap().mbti.as_deref(), Some("ENTP"));
    }
}
