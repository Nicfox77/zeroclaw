//! Durable steering observations under an already admitted foreground turn.
//! No second foreground lease, engine execution, or replay of an effect occurs.
use super::capture::ProviderCaptureScope;
use super::*;
use zeroclaw_api::ingress::IngressContext;
use zeroclaw_api::model_provider::ChatMessage;
use zeroclaw_api::session_events::{
    ContentPayload, Digest, EventData, MessageId, PayloadId, PayloadRead, ProviderPurpose,
    RecordedContentPart, StoredPayloadRef, TargetRef, canonical_json,
};

pub(super) async fn record(
    scope: &ProviderCaptureScope,
    message: &ChatMessage,
    ingress: Option<&IngressContext>,
) -> anyhow::Result<bool> {
    anyhow::ensure!(
        scope.purpose == ProviderPurpose::Main,
        "maintenance_cannot_accept_steering"
    );
    anyhow::ensure!(message.role == "user", "steering_requires_user_input");
    anyhow::ensure!(
        message.attachments.is_empty(),
        "steering_attachment_capture_not_supported"
    );
    let turn = scope
        .turn
        .clone()
        .ok_or_else(|| anyhow::Error::msg("steering_requires_admitted_turn"))?;
    let id = match ingress.filter(|i| i.message_id.as_deref().is_some_and(|id| !id.is_empty())) {
        Some(ingress) => MessageId::parse(format!(
            "steering-{}",
            Digest::from_bytes(&canonical_json(ingress)?).as_str()
        ))?,
        None => MessageId::new(),
    };
    let leaf = PayloadRecord::new(
        PayloadId::parse(format!("steering-text-{id}"))?,
        "session.text/v1".into(),
        message.content.as_bytes().to_vec(),
    )?;
    let content = PayloadRecord::new(
        PayloadId::parse(format!("steering-content-{id}"))?,
        "session.content/v1".into(),
        canonical_json(&ContentPayload {
            message_id: id.clone(),
            parts: vec![RecordedContentPart::Text {
                text_ref: StoredPayloadRef::from(&leaf),
            }],
        })?,
    )?;
    let provenance = PayloadRecord::new(
        PayloadId::parse(format!("steering-ingress-{id}"))?,
        "session.ingress/v1".into(),
        canonical_json(&serde_json::json!({"source":"runtime_steering", "ingress":ingress}))?,
    )?;
    let command = PayloadRecord::new(
        PayloadId::parse(format!("steering-command-{id}"))?,
        "session.control/v1".into(),
        canonical_json(
            &serde_json::json!({"command":"steer", "target_turn_id":turn, "content_ref":StoredPayloadRef::from(&content)}),
        )?,
    )?;
    let timestamp = message.timestamp;
    let owner = Arc::clone(&scope.coordinator);
    let stream = scope.stream.clone();
    let branch = scope.branch.clone();
    let resolver = Arc::clone(&scope.resolver);
    let recorded = scope
        .coordinator
        .blocking(move || {
            let lock = owner.lock_for(&stream);
            let _guard = lock.lock().unwrap_or_else(|error| error.into_inner());
            resolver.authorize_context(&stream, &branch)?;
            owner.require_known_turn(&stream, &branch, Some(&turn))?;
            let key = format!("steering-input:{id}");
            let payloads = vec![leaf, content.clone(), provenance.clone(), command.clone()];
            if owner.store.lookup_receipt(&stream, &key)?.is_some() {
                for expected in &payloads {
                    match owner.store.read_payload(&expected.payload_id)? {
                        PayloadRead::Present(actual)
                            if actual.codec == expected.codec && actual.body == expected.body => {}
                        _ => return Err(AppendError::IdempotencyConflict(key).into()),
                    }
                }
                return Ok(false);
            }
            let start = owner
                .store
                .read_latest_event(&stream, &branch, Some(&turn), &EventKind::TurnStarted)?
                .ok_or_else(|| {
                    SessionError::InvalidAdmission("steering_turn_not_started".into())
                })?;
            let EventData::TurnStarted {
                accepted_event_id, ..
            } = start.body.data
            else {
                return Err(SessionError::InvalidAdmission(
                    "steering_start_kind_mismatch".into(),
                ));
            };
            let actor = EventActor::System {
                component: "runtime_steering".into(),
            };
            let mut control = SessionEvent::typed(
                stream.clone(),
                EventData::ControlInputRecorded {
                    control_ref: StoredPayloadRef::from(&command),
                    expected_target_ref: TargetRef::Turn {
                        turn_id: turn.clone(),
                    },
                    ingress_ref: StoredPayloadRef::from(&provenance),
                },
                actor.clone(),
            )
            .map_err(AppendError::from)?;
            control.branch_id = Some(branch.clone());
            control.turn_id = Some(turn.clone());
            let mut user = SessionEvent::typed(
                stream.clone(),
                EventData::UserContentRecorded {
                    accepted_event_id,
                    content_ref: StoredPayloadRef::from(&content),
                },
                actor,
            )
            .map_err(AppendError::from)?;
            user.branch_id = Some(branch);
            user.turn_id = Some(turn);
            user.causation_id = Some(control.event_id.clone());
            if let Some(timestamp) = timestamp {
                user.occurred_at = timestamp;
            }
            let head = owner.store.head(&stream)?;
            resolver.authorize(&EffectIntent {
                session_id: stream.clone(),
                expected_head: head.clone(),
                idempotency_key: key.clone(),
                event: control.clone(),
                payloads: payloads.clone(),
            })?;
            owner.store.append_batch(AppendRequest {
                session_id: stream.clone(),
                expected_head: head,
                idempotency_key: key,
                events: vec![control, user],
                payloads,
                erasure_reference_commitment: None,
            })?;
            let _ = owner.wakeups.send(stream);
            Ok(true)
        })
        .await?;
    Ok(recorded)
}
