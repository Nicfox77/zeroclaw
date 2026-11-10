//! Derive interrupted assistant context from canonical provider observations.
//! Original text leaves remain the source of truth and erasure dependencies.
use super::*;
use zeroclaw_api::model_provider::ProviderContentPosition;
use zeroclaw_api::session_events::*;

const INTERRUPTED: &str = "\n\n[Interrupted: my response stopped before completion because the execution process was lost. This is only the text recorded before the interruption; it does not establish that any planned work completed.]";

impl SessionCoordinator {
    /// Called under the stream lock, after ownership loss has been proved.
    /// No effects are executed and no provider completion is invented.
    pub(super) fn interrupted_context(
        &self,
        started: &SessionEvent,
        history: &[SessionEvent],
        cause: &EventId,
        events: &mut Vec<SessionEvent>,
        payloads: &mut Vec<PayloadRecord>,
    ) -> Result<(), SessionError> {
        let EventData::ProviderAttemptStarted {
            prepared_event_id, ..
        } = &started.body.data
        else {
            return Ok(());
        };
        let Some(prepared) = history
            .iter()
            .find(|event| &event.event_id == prepared_event_id)
        else {
            return Err(ReadError::Gap("interrupted provider preparation missing".into()).into());
        };
        if !matches!(
            prepared.body.data,
            EventData::ProviderRequestPrepared {
                purpose: ProviderPurpose::Main,
                ..
            }
        ) {
            return Ok(());
        }
        let mut segments: Vec<(Option<ProviderContentPosition>, StoredPayloadRef)> = Vec::new();
        for event in history.iter().filter(|event| {
            event.invocation_id == started.invocation_id && event.sequence > started.sequence
        }) {
            let EventData::ProviderResponseProgressObserved { observation_ref } = &event.body.data
            else {
                continue;
            };
            let observation: ProviderObservationPayload = self.recovery_payload(observation_ref)?;
            for operation in observation.operations {
                match operation {
                    ProviderObservation::ProtocolSegmentObserved {
                        position,
                        component,
                        segment_ref,
                        ..
                    } if component == "text" => {
                        let block: ResponseBlock = self.recovery_payload(&segment_ref)?;
                        if let ResponseBlock::Text { text_ref } = block {
                            segments.push((Some(position), text_ref));
                        }
                    }
                    ProviderObservation::BytesAppended {
                        component,
                        bytes_ref,
                        ..
                    } if component == "text" => {
                        segments.push((None, bytes_ref));
                    }
                    _ => {}
                }
            }
        }
        if segments.is_empty() {
            return Ok(());
        }
        // Match normal capture ordering: protocol coordinates when complete,
        // otherwise the observed append order. Within a coordinate sort is stable.
        order_provider_segments(&mut segments, |segment| segment.0.as_ref());
        let mut parts = Vec::new();
        let mut nonempty = false;
        for (_, text_ref) in segments {
            let record = match self.store.read_payload(&text_ref.payload_id)? {
                PayloadRead::Present(record) => record,
                // Never resurrect an erased response, including a partial one.
                _ => return Ok(()),
            };
            if StoredPayloadRef::from(&record) != text_ref || record.codec != "session.text/v1" {
                return Err(
                    ReadError::Corrupt("interrupted text reference mismatch".into()).into(),
                );
            }
            let text =
                std::str::from_utf8(&record.body).map_err(|e| ReadError::Corrupt(e.to_string()))?;
            nonempty |= !text.trim().is_empty();
            parts.push(RecordedContentPart::Text { text_ref });
        }
        if !nonempty {
            return Ok(());
        }
        let branch = started
            .branch_id
            .as_ref()
            .ok_or(SessionError::InvocationMismatch)?;
        let input = match events.iter().rev().find(|event| {
            event.branch_id.as_ref() == Some(branch)
                && event.body.kind() == EventKind::ContextProjectionCreated
        }) {
            Some(event) => event.clone(),
            None => self
                .branch_context(&started.session_id, branch)?
                .ok_or_else(|| ReadError::Gap("interrupted context missing".into()))?,
        };
        let EventData::ContextProjectionCreated {
            result_ref,
            source_ranges,
            summary_ref,
            summary_source_refs,
            ..
        } = &input.body.data
        else {
            return Err(SessionError::InvocationMismatch);
        };
        let mut context: serde_json::Value = if let Some(record) = payloads
            .iter()
            .find(|p| p.payload_id == result_ref.payload_id)
        {
            serde_json::from_slice(&record.body).map_err(|e| ReadError::Corrupt(e.to_string()))?
        } else {
            self.recovery_payload(result_ref)?
        };
        let mut refs = match context.get("message_refs") {
            Some(refs) => serde_json::from_value(refs.clone())
                .map_err(|e| ReadError::Corrupt(e.to_string()))?,
            None => context_read::context_refs(self.store.as_ref(), &input)?,
        };
        let marker = PayloadRecord::new(
            PayloadId::new(),
            "session.text/v1".into(),
            INTERRUPTED.as_bytes().to_vec(),
        )
        .map_err(AppendError::from)?;
        parts.push(RecordedContentPart::Text {
            text_ref: StoredPayloadRef::from(&marker),
        });
        let id = MessageId::new();
        let content = PayloadRecord::json(
            "session.content/v1",
            &ContentPayload {
                message_id: id.clone(),
                parts,
            },
        )
        .map_err(AppendError::from)?;
        let mut recorded = SessionEvent::typed(
            started.session_id.clone(),
            EventData::AssistantContentRecorded {
                content_ref: StoredPayloadRef::from(&content),
                // ProviderBlockRef names completed results only. The original
                // attempt is linked by causation; parts reference its exact leaves.
                provenance_refs: vec![],
            },
            EventActor::System {
                component: "execution_recovery".into(),
            },
        )
        .map_err(AppendError::from)?;
        recorded.branch_id = started.branch_id.clone();
        recorded.turn_id = started.turn_id.clone();
        recorded.causation_id = Some(started.event_id.clone());
        refs.push(MessageRef {
            message_id: id,
            source: ContextMessageSource::Content {
                content_event_id: recorded.event_id.clone(),
                content_ref: StoredPayloadRef::from(&content),
            },
            recalled_memory: None,
        });
        context["message_refs"] =
            serde_json::to_value(refs).map_err(|e| ReadError::Corrupt(e.to_string()))?;
        let context =
            PayloadRecord::json("session.context/v1", &context).map_err(AppendError::from)?;
        // The transform link is also the canonical witness for inherited
        // recall snapshots. Recovery creates a local writer binding only;
        // it carries no historical or newly invented effect permissions.
        let binding = PayloadRecord::json("session.execution-binding/v1", &serde_json::json!({
            "source":"verified_execution_loss_recovery", "execution_incarnation":self.execution_incarnation,
            "recovery_event_id":cause,
        })).map_err(AppendError::from)?;
        let spec = PayloadRecord::json("session.transform/v1", &serde_json::json!({
            "reason_key":"interrupted_assistant_response", "algorithm_version":"execution-recovery/v1", "budget":0,
        })).map_err(AppendError::from)?;
        let mut transform = SessionEvent::typed(
            started.session_id.clone(),
            EventData::ContextTransformStarted {
                input_projection_event_id: input.event_id.clone(),
                transform_spec_ref: StoredPayloadRef::from(&spec),
                execution_binding_ref: StoredPayloadRef::from(&binding),
            },
            recorded.actor.clone(),
        )
        .map_err(AppendError::from)?;
        transform.branch_id = started.branch_id.clone();
        transform.turn_id = started.turn_id.clone();
        transform.causation_id = Some(cause.clone());
        let mut projection = SessionEvent::typed(
            started.session_id.clone(),
            EventData::ContextProjectionCreated {
                source_ranges: source_ranges.clone(),
                summary_ref: summary_ref.clone(),
                summary_source_refs: summary_source_refs.clone(),
                budget: 0,
                reason_key: "interrupted_assistant_response".into(),
                algorithm_version: "execution-recovery/v1".into(),
                result_ref: StoredPayloadRef::from(&context),
                transform_started_event_id: Some(transform.event_id.clone()),
            },
            recorded.actor.clone(),
        )
        .map_err(AppendError::from)?;
        projection.branch_id = started.branch_id.clone();
        projection.turn_id = started.turn_id.clone();
        projection.causation_id = Some(transform.event_id.clone());
        events.extend([recorded, transform, projection]);
        payloads.extend([marker, content, context, binding, spec]);
        Ok(())
    }

    fn recovery_payload<T: serde::de::DeserializeOwned>(
        &self,
        reference: &StoredPayloadRef,
    ) -> Result<T, SessionError> {
        let record = context_read::present(self.store.as_ref(), reference)?;
        serde_json::from_slice(&record.body).map_err(|e| ReadError::Corrupt(e.to_string()).into())
    }
}
