//! Canonical AIEOS name normalization shared by identity prompts and speaker labels.
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Names {
    #[serde(default)]
    pub first: Option<String>,
    #[serde(default)]
    pub last: Option<String>,
    #[serde(default)]
    pub nickname: Option<String>,
    #[serde(default)]
    pub full: Option<String>,
}

pub fn normalize_names(value: Option<&Value>) -> Option<Names> {
    let value = value?;

    let mut names = Names {
        first: value.get("first").and_then(scalar_to_string),
        last: value.get("last").and_then(scalar_to_string),
        nickname: value.get("nickname").and_then(scalar_to_string),
        full: value.get("full").and_then(scalar_to_string),
    };

    if names.full.is_none()
        && let (Some(first), Some(last)) = (&names.first, &names.last)
    {
        names.full = Some(format!("{first} {last}"));
    }

    if names.first.is_none()
        && names.last.is_none()
        && names.nickname.is_none()
        && names.full.is_none()
    {
        return None;
    }

    Some(names)
}

fn scalar_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_owned()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}
impl Names {
    pub fn display_name(&self) -> Option<&str> {
        self.nickname
            .as_deref()
            .or(self.full.as_deref())
            .or(self.first.as_deref())
            .or(self.last.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_name_fields_share_identity_normalization_and_display_precedence() {
        let data = serde_json::json!({"first":" Nova ","last":"Example","nickname":"Nov"});
        let names = normalize_names(Some(&data)).unwrap();
        assert_eq!(names.full.as_deref(), Some("Nova Example"));
        assert_eq!(names.display_name(), Some("Nov"));
        assert!(normalize_names(Some(&serde_json::json!({"first":"  "}))).is_none());
    }
}
