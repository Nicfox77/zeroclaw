//! Disposable session-list metadata, derived from one fixed event prefix.
use super::*;
use zeroclaw_api::session_events::ModelResolution;

impl SessionQuery {
    async fn summary_reducer(
        &self,
        stream: SessionId,
    ) -> Result<zeroclaw_api::session_events::catalog::SessionReducer, SessionError> {
        use zeroclaw_api::session_events::catalog::SessionReducer;
        self.policy
            .authorize_actor(&self.actor, &stream, SessionAccess::Read)?;
        let store = self.coordinator.store.clone();
        let scope = stream.clone();
        let (mut reducer, revision, head, expected_revision) = self
            .coordinator
            .blocking(move || {
                let revision = store.erasure_revision()?;
                let expected_revision = store.canonical_revision()?;
                let head = store.head(&scope)?;
                Ok((
                    SessionReducer::load_checkpoint(store.as_ref(), &scope, &head)?,
                    revision,
                    head,
                    expected_revision,
                ))
            })
            .await?;
        let mut after = reducer.cursor();
        let checkpoint_sequence = after.sequence;
        let mut through = Some(head);
        loop {
            let page = self
                .read_page(stream.clone(), after, through.clone(), 256)
                .await?;
            through = Some(page.through.clone());
            for event in &page.events {
                let seed_count = if let EventData::ContextProjectionCreated { reason_key, .. } =
                    &event.body.data
                    && (reason_key == "host_seed" || reason_key == "legacy_import")
                {
                    let PayloadRead::Present(record) = self
                        .read_payload(
                            stream.clone(),
                            PayloadSelector {
                                event_sequence: event.sequence,
                                reference_path: vec![0],
                            },
                        )
                        .await?
                    else {
                        return Err(ReadError::Corrupt("seed identities unavailable".into()).into());
                    };
                    let value: serde_json::Value = serde_json::from_slice(&record.body)
                        .map_err(|error| ReadError::Corrupt(error.to_string()))?;
                    Some(
                        value
                            .get(if reason_key == "host_seed" {
                                "seed_messages"
                            } else {
                                "legacy_messages"
                            })
                            .and_then(serde_json::Value::as_array)
                            .ok_or_else(|| ReadError::Corrupt("seed identities missing".into()))?
                            .len(),
                    )
                } else {
                    None
                };
                let targets = if let EventData::ContentSuperseded {
                    target_event_ids, ..
                } = &event.body.data
                {
                    let ids = target_event_ids.clone();
                    let store = self.coordinator.store.clone();
                    self.coordinator
                        .blocking(move || {
                            ids.into_iter()
                                .map(|id| Ok((id.clone(), store.read_event(&id)?)))
                                .collect::<Result<std::collections::HashMap<_, _>, SessionError>>()
                        })
                        .await?
                } else {
                    Default::default()
                };
                reducer.apply_indexed(event, seed_count, &|id| {
                    Ok(targets.get(id).cloned().flatten())
                })?;
                if let EventData::TurnAccepted { ingress_ref, .. } = &event.body.data {
                    let reference_index = event
                        .body
                        .data
                        .stored_refs()
                        .map_err(AppendError::from)?
                        .iter()
                        .position(|reference| reference == ingress_ref)
                        .ok_or_else(|| ReadError::Corrupt("ingress reference missing".into()))?;
                    let PayloadRead::Present(record) = self
                        .read_payload(
                            stream.clone(),
                            PayloadSelector {
                                event_sequence: event.sequence,
                                reference_path: vec![reference_index],
                            },
                        )
                        .await?
                    else {
                        return Err(ReadError::Corrupt("ingress unavailable".into()).into());
                    };
                    let value = serde_json::from_slice(&record.body)
                        .map_err(|error| ReadError::Corrupt(error.to_string()))?;
                    reducer.observe_ingress(&value)?;
                }
            }
            after = page
                .events
                .last()
                .map_or(page.after.clone(), |event| EventCursor {
                    session_id: stream.clone(),
                    sequence: event.sequence,
                });
            if !page.has_more {
                break;
            }
        }
        reducer.compact();
        let store = self.coordinator.store.clone();
        reducer = self
            .coordinator
            .blocking(move || {
                if reducer.cursor().sequence != checkpoint_sequence {
                    reducer.save_checkpoint(
                        store.as_ref(),
                        revision.as_ref(),
                        expected_revision,
                    )?;
                }
                Ok(reducer)
            })
            .await?;
        Ok(reducer)
    }

    /// Small current-state projection; reading it never replays tools/providers.
    pub async fn current_state(
        &self,
        stream: SessionId,
    ) -> Result<serde_json::Value, SessionError> {
        let reducer = self.summary_reducer(stream.clone()).await?;
        self.policy
            .authorize_actor(&self.actor, &stream, SessionAccess::Read)?;
        let active = reducer.active_turn();
        Ok(
            serde_json::json!({"session_id":stream,"state":if active.is_some(){"running"}else if reducer.latest_outcome_unknown(){"unknown"}else{"idle"},"turn_id":active,"source_cursor":reducer.cursor()}),
        )
    }

    pub async fn summary(&self, stream: SessionId) -> Result<SessionSummary, SessionError> {
        let mut reducer = self.summary_reducer(stream.clone()).await?;
        let through = Some(reducer.cursor());
        let has_copy_seed = reducer.has_copy_seed();
        let pending: Vec<_> = reducer
            .pending_metadata()
            .map(|(field, sequence, reference)| (field.to_owned(), sequence, reference.clone()))
            .collect();
        for (field, sequence, reference) in pending {
            let value = match self
                .read_payload(
                    stream.clone(),
                    PayloadSelector {
                        event_sequence: sequence,
                        reference_path: vec![0],
                    },
                )
                .await?
            {
                PayloadRead::Present(record) if record.codec == "session.text/v1" => {
                    serde_json::Value::String(
                        String::from_utf8(record.body)
                            .map_err(|error| ReadError::Corrupt(error.to_string()))?,
                    )
                }
                PayloadRead::Present(record) => serde_json::from_slice(&record.body)
                    .map_err(|error| ReadError::Corrupt(error.to_string()))?,
                PayloadRead::ErasurePending | PayloadRead::Redacted
                    if zeroclaw_api::session_events::payloads::content_leaf(&reference.codec) =>
                {
                    serde_json::Value::Null
                }
                _ => {
                    return Err(ReadError::Corrupt("structural metadata unavailable".into()).into());
                }
            };
            reducer.resolve_metadata(&field, value)?;
        }
        self.policy
            .authorize_actor(&self.actor, &stream, SessionAccess::Read)?;
        let mut summary = reducer
            .finish()?
            .ok_or_else(|| ReadError::Gap("session root missing".into()))?
            .session;
        if has_copy_seed {
            summary.message_count = self
                .message_window(stream, through, None, Some(0))
                .await?
                .next_cursor
                .map_or(0, |cursor| cursor.before);
        }
        Ok(summary)
    }
}

impl SessionQuery {
    /// Rejected selections are observations, never the effective override.
    pub async fn model_selection(
        &self,
        mut stream: SessionId,
    ) -> Result<Option<zeroclaw_api::session_events::capture::ModelOverrideSelection>, SessionError>
    {
        let mut frozen = None;
        let mut selected_branch = self.branch.clone();
        let mut visited = std::collections::HashSet::new();
        loop {
            if !visited.insert(stream.clone()) {
                return Err(ReadError::Corrupt("copy source cycle".into()).into());
            }
            self.policy
                .authorize_actor(&self.actor, &stream, SessionAccess::Read)?;
            let scope = stream.clone();
            let store = self.coordinator.store.clone();
            let branch = selected_branch.clone();
            let bound = frozen.take();
            let (cause, copy) = self
                .coordinator
                .blocking(move || {
                    let actual = store.head(&scope)?;
                    let through = bound.unwrap_or(actual.sequence);
                    if through > actual.sequence {
                        return Err(ReadError::Gap("copy source prefix unavailable".into()).into());
                    }
                    let branch = match branch {
                        Some(branch) => branch,
                        None => store
                            .read_page(
                                &scope,
                                &EventCursor::beginning(scope.clone()),
                                Some(&EventCursor {
                                    session_id: scope.clone(),
                                    sequence: through,
                                }),
                                1,
                            )?
                            .events
                            .first()
                            .and_then(|event| event.branch_id.clone())
                            .ok_or_else(|| {
                                ReadError::Gap("model selection branch absent".into())
                            })?,
                    };
                    let lineage = zeroclaw_api::session_events::lineage::read_frozen_lineage(
                        store.as_ref(),
                        &scope,
                        &branch,
                        through,
                    )?;
                    let mut applied: Option<SessionEvent> = None;
                    let mut seed: Option<SessionEvent> = None;
                    for (branch, bound) in lineage {
                        let mut before = bound;
                        while let Some(event) = store.read_latest_event_before(
                            &scope,
                            &branch,
                            None,
                            &EventKind::SessionModelOverrideChanged,
                            before,
                        )? {
                            if matches!(
                                &event.body.data,
                                EventData::SessionModelOverrideChanged {
                                    resolution: ModelResolution::Applied { .. }
                                }
                            ) {
                                if applied
                                    .as_ref()
                                    .is_none_or(|prior| event.sequence > prior.sequence)
                                {
                                    applied = Some(event);
                                }
                                break;
                            }
                            before = event.sequence.saturating_sub(1);
                        }
                        if let Some(event) = store.read_context_projection(
                            &scope,
                            &branch,
                            None,
                            "copy_on_write_seed",
                            bound,
                        )? && seed
                            .as_ref()
                            .is_none_or(|prior| event.sequence > prior.sequence)
                        {
                            seed = Some(event);
                        }
                    }
                    if let Some(event) = applied {
                        return Ok((
                            Some(event.causation_id.ok_or_else(|| {
                                ReadError::Corrupt("model selection origin absent".into())
                            })?),
                            None,
                        ));
                    }
                    let copy = if let Some(event) = seed {
                        let EventData::ContextProjectionCreated { result_ref, .. } =
                            &event.body.data
                        else {
                            return Err(
                                ReadError::Corrupt("copy source kind mismatch".into()).into()
                            );
                        };
                        let record = zeroclaw_api::session_events::context_read::present(
                            store.as_ref(),
                            result_ref,
                        )?;
                        let body = serde_json::from_slice(&record.body)
                            .map_err(|error| ReadError::Corrupt(error.to_string()))?;
                        zeroclaw_api::session_events::context_read::copy_source(&body)?
                    } else {
                        None
                    };
                    Ok((None, copy))
                })
                .await?;
            self.policy
                .authorize_actor(&self.actor, &stream, SessionAccess::Read)?;
            let Some(cause) = cause else {
                if let Some(copy) = copy {
                    stream = copy.cursor.session_id;
                    frozen = Some(copy.cursor.sequence);
                    selected_branch = Some(copy.branch_id);
                    continue;
                }
                return Ok(None);
            };
            let store = self.coordinator.store.clone();
            let control = self
                .coordinator
                .blocking(move || {
                    Ok(store
                        .read_event(&cause)?
                        .ok_or_else(|| ReadError::Gap("model selection input absent".into()))?)
                })
                .await?;
            if control.session_id != stream
                || control.body.kind() != EventKind::ControlInputRecorded
            {
                return Err(
                    ReadError::Corrupt("model selection input scope mismatch".into()).into(),
                );
            }
            let PayloadRead::Present(value) = self
                .read_payload(
                    stream,
                    PayloadSelector {
                        event_sequence: control.sequence,
                        reference_path: vec![0],
                    },
                )
                .await?
            else {
                return Err(SessionError::StaleTarget(
                    "model selection input erased".into(),
                ));
            };
            let value: serde_json::Value = serde_json::from_slice(&value.body)
                .map_err(|error| ReadError::Corrupt(error.to_string()))?;
            return Ok(Some(
                serde_json::from_value(
                    value
                        .get("selection")
                        .cloned()
                        .ok_or_else(|| ReadError::Corrupt("model selection missing".into()))?,
                )
                .map_err(|error| ReadError::Corrupt(error.to_string()))?,
            ));
        }
    }
}
