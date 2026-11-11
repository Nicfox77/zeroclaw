//! Retention admission over live policy and the existing durable delivery ledger.
use super::ledger::Ledger;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use zeroclaw_config::{hindsight::Retention, live::LiveConfigHandle};

/// Operator intent is created here, never inferred from delivery status.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Control {
    #[default]
    Automatic,
    Paused,
    Draining {
        until: i64,
    },
}

pub fn allowed(policy: &Retention, ledger: &Ledger, now: DateTime<Utc>) -> anyhow::Result<bool> {
    match ledger.retention_control()? {
        Control::Paused => Ok(false),
        Control::Draining { until } if until > now.timestamp() => Ok(true),
        _ => policy.allows_at(now),
    }
}

/// Readiness does not reserve the accelerator while bank preparation awaits.
pub(super) fn ready(
    live: &LiveConfigHandle,
    ledger: &Ledger,
    operation: &str,
) -> anyhow::Result<bool> {
    let config = live.read();
    if config.memory.backend != "hindsight" {
        return Ok(false);
    }
    let policy = &config.memory.hindsight.retention;
    Ok(allowed(policy, ledger, Utc::now())?
        && (!policy.deferred
            || ledger
                .retention_in_flight()?
                .is_none_or(|active| active == operation)))
}

/// Called again at every mutating retention request, after awaited preparation.
/// A durable single-operation claim keeps uncertain acknowledgements serialized.
pub(super) fn admit(
    live: &LiveConfigHandle,
    ledger: &Ledger,
    operation: &str,
) -> anyhow::Result<bool> {
    let config = live.read();
    if config.memory.backend != "hindsight" {
        return Ok(false);
    }
    let policy = &config.memory.hindsight.retention;
    if !allowed(policy, ledger, Utc::now())? {
        return Ok(false);
    }
    if policy.deferred {
        ledger.claim_retention(operation)
    } else {
        Ok(true)
    }
}

pub fn status(policy: &Retention, ledger: &Ledger) -> anyhow::Result<serde_json::Value> {
    let now = Utc::now();
    Ok(serde_json::json!({
        "deferred": policy.deferred,
        "window": policy.window,
        "timezone": policy.timezone,
        "control": ledger.retention_control()?,
        "accepting_new_work": allowed(policy, ledger, now)?,
        "in_flight": ledger.retention_in_flight()?,
        "deliveries": ledger.delivery_counts()?,
        "checked_at": now.to_rfc3339(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn operator_intent_and_unknown_submission_survive_reopen_until_upstream_terminal() {
        let directory = tempfile::tempdir().unwrap();
        let policy = Retention {
            deferred: true,
            ..Default::default()
        };
        let now = Utc::now();
        let ledger = Ledger::open(directory.path()).unwrap();
        assert!(!allowed(&policy, &ledger, now).unwrap());
        ledger
            .set_retention_control(&Control::Draining {
                until: now.timestamp() + 60,
            })
            .unwrap();
        let first = ledger.operation("bank", "one", "one").unwrap();
        let second = ledger.operation("bank", "two", "two").unwrap();
        assert!(ledger.claim_retention(&first).unwrap());
        assert!(!ledger.claim_retention(&second).unwrap());
        drop(ledger);
        let ledger = Ledger::open(directory.path()).unwrap();
        assert!(allowed(&policy, &ledger, now).unwrap());
        assert!(ledger.claim_retention(&first).unwrap());
        assert!(!ledger.claim_retention(&second).unwrap());
        ledger.update(&first, "completed").unwrap();
        assert!(ledger.claim_retention(&second).unwrap());
        ledger.set_retention_control(&Control::Paused).unwrap();
        assert!(!allowed(&policy, &ledger, now).unwrap());
        ledger.set_retention_control(&Control::Automatic).unwrap();
        assert!(!allowed(&policy, &ledger, now).unwrap());
        ledger
            .set_retention_control(&Control::Draining {
                until: now.timestamp() - 1,
            })
            .unwrap();
        assert!(!allowed(&policy, &ledger, now).unwrap());
    }
    #[test]
    fn readiness_has_no_claim_and_admission_resolves_changed_live_policy() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(directory.path()).unwrap();
        let mut cfg = zeroclaw_config::schema::Config::default();
        cfg.memory.backend = "hindsight".into();
        cfg.memory.hindsight.retention.deferred = true;
        let live = zeroclaw_config::live::LiveConfig::new(cfg.clone());
        let operation = ledger.operation("bank", "fixture", "fixture").unwrap();
        assert!(!ready(&live.handle(), &ledger, &operation).unwrap());
        ledger
            .set_retention_control(&Control::Draining {
                until: Utc::now().timestamp() + 60,
            })
            .unwrap();
        assert!(ready(&live.handle(), &ledger, &operation).unwrap());
        assert_eq!(ledger.retention_in_flight().unwrap(), None);
        ledger.set_retention_control(&Control::Automatic).unwrap();
        cfg.memory.hindsight.retention.deferred = false;
        live.publish(live.next_revision().unwrap(), cfg).unwrap();
        assert!(admit(&live.handle(), &ledger, &operation).unwrap());
        ledger.set_retention_control(&Control::Paused).unwrap();
        assert!(!admit(&live.handle(), &ledger, &operation).unwrap());
    }
    #[test]
    fn local_note_transport_failure_does_not_release_uncertain_inflight_claim() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(directory.path()).unwrap();
        let item = super::super::client::Item {
            update_mode: None,
            content: "Fixture note".into(),
            document_id: "fixture".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            context: "fixture".into(),
            metadata: Default::default(),
        };
        let scope = zeroclaw_api::memory_traits::PrincipalScope::new("fixture-owner");
        let note = ledger
            .queue_note("bank", "fixture", &scope, &[item], 10)
            .unwrap();
        assert!(ledger.claim_retention(&note).unwrap());
        ledger.defer_note(&note, true).unwrap();
        let other = ledger.operation("bank", "other", "other").unwrap();
        assert!(!ledger.claim_retention(&other).unwrap());
        ledger.update(&note, "failed").unwrap(); // An observed upstream failure is terminal.
        assert!(ledger.claim_retention(&other).unwrap());
    }
}
