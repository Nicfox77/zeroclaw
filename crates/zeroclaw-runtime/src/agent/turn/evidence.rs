//! Authoritative, turn-local evidence for consumers that retain completed
//! conversations.
//!
//! This is deliberately not a transcript cache. The turn engine records only
//! the accepted no-tool model response and steering text immediately before
//! that text enters provider-visible history. Runtime notices, injected
//! context, and tool output never enter this record.

use std::sync::{Arc, Mutex};

use zeroclaw_api::ingress::{SourceClass, Transport, TurnOrigin};
use zeroclaw_api::principal::PrincipalId;

use super::SteeringInput;

/// One admitted correction that actually entered the model's current turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedSteering {
    pub text: String,
    pub principal: PrincipalId,
    pub ingress: zeroclaw_api::ingress::IngressContext,
}

/// Completed evidence for one logical turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnEvidence {
    pub turn_id: String,
    pub principal: PrincipalId,
    pub model_final: String,
    pub steering: Vec<AppliedSteering>,
}

#[derive(Default)]
struct EvidenceState {
    turn_id: Option<String>,
    model_final: Option<String>,
    steering: Vec<AppliedSteering>,
    invalid: bool,
}

/// Owned handle shared by the caller and the spawned turn task.
///
/// A caller creates one capture per logical turn. The handle does not rely on
/// task-local state, so it remains usable when `execute_turn_prepared` moves
/// the turn into a spawned task.
#[derive(Clone)]
pub struct TurnEvidenceCapture {
    principal: Arc<PrincipalId>,
    state: Arc<Mutex<EvidenceState>>,
}

impl TurnEvidenceCapture {
    /// Bind evidence for one turn to its already-admitted original principal.
    #[must_use]
    pub fn new(principal: PrincipalId) -> Self {
        Self {
            principal: Arc::new(principal),
            state: Arc::new(Mutex::new(EvidenceState::default())),
        }
    }

    /// The principal an applied correction must match.
    #[must_use]
    pub fn principal(&self) -> &PrincipalId {
        &self.principal
    }

    /// Return completed evidence only when a model-authored final was accepted.
    ///
    /// No final means the caller must not retain the turn. This fails closed for
    /// cache hits, cancellations, runtime notices, and turn errors.
    #[must_use]
    pub fn snapshot(&self) -> Option<TurnEvidence> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.invalid {
            return None;
        }
        Some(TurnEvidence {
            turn_id: state.turn_id.clone()?,
            principal: (*self.principal).clone(),
            model_final: state.model_final.clone()?,
            steering: state.steering.clone(),
        })
    }

    pub(crate) fn begin_turn(&self, turn_id: &str) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.invalid {
            return;
        }
        match state.turn_id.as_deref() {
            Some(existing) if existing != turn_id => state.invalid = true,
            Some(_) => {}
            None => state.turn_id = Some(turn_id.to_owned()),
        }
    }

    pub(crate) fn replace_model_final(&self, text: &str) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.invalid || state.turn_id.is_none() {
            state.invalid = true;
            state.model_final = None;
            return;
        }
        state.model_final = Some(text.to_owned());
    }

    pub(crate) fn clear_model_final(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.invalid || state.turn_id.is_none() {
            state.invalid = true;
            state.model_final = None;
            return;
        }
        state.model_final = None;
    }

    /// Validate and record a correction at the point it will be appended to
    /// provider-visible history.
    pub(crate) fn apply_steering(&self, input: &SteeringInput) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(ingress) = input.ingress() else {
            state.invalid = true;
            state.model_final = None;
            state.steering.clear();
            return;
        };
        let same_principal = ingress.sender.as_deref() == Some(self.principal.as_str());
        let external_user = ingress.source_class == SourceClass::External
            && ingress.internal_principal.is_none()
            && !matches!(ingress.transport, Transport::Internal)
            && !matches!(
                ingress.origin,
                TurnOrigin::Cron | TurnOrigin::Daemon | TurnOrigin::SubTurn
            );
        if state.invalid || state.turn_id.is_none() || !same_principal || !external_user {
            state.invalid = true;
            state.model_final = None;
            state.steering.clear();
            return;
        }
        state.steering.push(AppliedSteering {
            text: input.text().to_owned(),
            principal: (*self.principal).clone(),
            ingress: ingress.clone(),
        });
    }
}

impl std::fmt::Debug for TurnEvidenceCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f.debug_struct("TurnEvidenceCapture")
            .field("principal", &self.principal)
            .field("turn_id", &state.turn_id)
            .field("has_model_final", &state.model_final.is_some())
            .field("steering_count", &state.steering.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ingress(
        sender: &str,
        origin: TurnOrigin,
        internal_principal: bool,
    ) -> zeroclaw_api::ingress::IngressContext {
        zeroclaw_api::ingress::IngressContext {
            message_id: Some("steer-1".into()),
            source_class: SourceClass::External,
            sender: Some(sender.into()),
            transport: Transport::Rpc,
            trust: zeroclaw_api::ingress::TrustClass::Untrusted,
            origin,
            internal_principal: internal_principal.then_some(
                zeroclaw_api::ingress::InternalPrincipal::Daemon {
                    task: "synthetic".into(),
                },
            ),
        }
    }

    #[test]
    fn same_principal_external_steering_is_recorded_without_runtime_context() {
        let principal = PrincipalId::from("user:alice");
        let capture = TurnEvidenceCapture::new(principal.clone());
        capture.begin_turn("turn-1");
        capture.apply_steering(
            &SteeringInput::new("Use the corrected target.").with_ingress(ingress(
                "user:alice",
                TurnOrigin::Interactive,
                false,
            )),
        );
        capture.replace_model_final("The corrected target is production.");

        let evidence = capture.snapshot().unwrap();
        assert_eq!(evidence.model_final, "The corrected target is production.");
        assert_eq!(evidence.steering.len(), 1);
        assert_eq!(evidence.steering[0].text, "Use the corrected target.");
        assert_eq!(evidence.steering[0].principal, principal);
    }

    #[test]
    fn missing_foreign_or_internal_provenance_invalidates_the_whole_capture() {
        let principal = PrincipalId::from("user:alice");
        for input in [
            SteeringInput::new("missing identity"),
            SteeringInput::new("foreign identity").with_ingress(ingress(
                "user:mallory",
                TurnOrigin::Interactive,
                false,
            )),
            SteeringInput::new("runtime injection").with_ingress(ingress(
                "user:alice",
                TurnOrigin::Daemon,
                true,
            )),
        ] {
            let capture = TurnEvidenceCapture::new(principal.clone());
            capture.begin_turn("turn-1");
            capture.replace_model_final("candidate final");
            capture.apply_steering(&input);
            assert!(capture.snapshot().is_none());
        }
    }

    #[test]
    fn snapshot_requires_an_accepted_model_final() {
        let capture = TurnEvidenceCapture::new(PrincipalId::from("user:alice"));
        capture.begin_turn("turn-1");
        assert!(capture.snapshot().is_none());
        capture.replace_model_final("only a model response is eligible");
        assert_eq!(
            capture.snapshot().unwrap().model_final,
            "only a model response is eligible"
        );
    }
}
