use super::*;
use crate::session::summary_deck::{ResolvedSummaryDeck, SummaryDeckRuntime};
use std::sync::atomic::AtomicUsize;
use zeroclaw_config::summary_deck::SummaryDeckConfig;

fn recalled_fixture() -> zeroclaw_api::model_provider::RecalledMemoryContext {
    zeroclaw_api::model_provider::RecalledMemoryContext {
        entries: vec![zeroclaw_api::model_provider::RecalledMemoryEntry {
            key: "fixture_fact".into(),
            content: "retained recall evidence".into(),
            stored_on: None,
        }],
    }
}

#[tokio::test]
async fn recalled_context_preserves_identity_reopens_and_rotates_without_entering_the_transcript() {
    for backend in ["sqlite", "jsonl"] {
        let directory = tempfile::tempdir().expect("directory");
        let (port, authority) = session(directory.path(), backend);
        let mut snapshot = add_turn(&port, 0).await.expect("turn");
        let before = port.latest_context().await.expect("context");
        let original =
            crate::session::context_messages::context_refs(port.owner.store.as_ref(), &before)
                .expect("refs");
        let user = snapshot
            .messages
            .iter_mut()
            .find_map(|message| match message {
                ConversationMessage::Chat(chat) if chat.role == "user" => Some(chat),
                _ => None,
            })
            .expect("user");
        let raw = user.content.clone();
        user.recalled_memory = Some(recalled_fixture());
        port.replace_context(snapshot, "recall_fixture")
            .await
            .expect("attach recall");
        let after = port.latest_context().await.expect("context");
        let refs =
            crate::session::context_messages::context_refs(port.owner.store.as_ref(), &after)
                .expect("refs");
        let annotated = refs
            .iter()
            .find(|item| item.recalled_memory.is_some())
            .expect("annotation");
        assert!(
            original
                .iter()
                .any(|item| item.message_id == annotated.message_id
                    && item.source == annotated.source)
        );
        let PayloadRead::Present(payload) = port
            .owner
            .store
            .read_payload(
                &annotated
                    .recalled_memory
                    .as_ref()
                    .expect("reference")
                    .payload_id,
            )
            .expect("snapshot")
        else {
            panic!("snapshot")
        };
        assert!(
            !String::from_utf8(payload.body)
                .expect("json")
                .contains(&raw),
            "recall leaves must not copy the original user's text"
        );
        let stream = port.stream.clone();
        let branch = port.branch.clone();
        drop(port);
        let store = zeroclaw_infra::session_events::open_event_store(directory.path(), backend)
            .expect("reopen");
        let port = EngineSession::new(
            Arc::new(SessionCoordinator::new(store)),
            stream,
            branch,
            PrincipalId::shared_operator(),
            authority,
        )
        .with_summary_deck(Arc::new(Runtime::new()));
        let restored = port
            .read_context_snapshot(port.latest_context().await.expect("context"))
            .await
            .expect("snapshot")
            .expect("present");
        assert!(restored.messages.iter().any(|message|matches!(message,ConversationMessage::Chat(chat) if chat.content==raw && chat.recalled_memory.as_ref()==Some(&recalled_fixture()))));
        for n in 1..12 {
            let snapshot = add_turn(&port, n).await.expect("turn");
            port.prepare_context(snapshot).await.expect("prepare");
            wait_job(&port).await.expect("summary");
        }
        let current = port
            .read_context_snapshot(port.latest_context().await.expect("context"))
            .await
            .expect("snapshot")
            .expect("present");
        assert!(current.messages.iter().all(|message|!matches!(message,ConversationMessage::Chat(chat) if chat.recalled_memory.is_some())));
        let transcript = port
            .history_reader()
            .expect("reader")
            .history(port.stream.as_str(), 100)
            .await
            .expect("history")
            .expect("present");
        assert!(
            transcript
                .messages
                .iter()
                .any(|message| message.text == raw)
        );
        assert!(
            transcript
                .messages
                .iter()
                .all(|message| !message.text.contains("retained recall evidence"))
        );
    }
}

#[tokio::test]
async fn recall_references_observe_erasure_and_never_resurrect_the_user_source() {
    for backend in ["sqlite", "jsonl"] {
        let directory = tempfile::tempdir().expect("directory");
        let (port, authority) = session(directory.path(), backend);
        let mut snapshot = add_turn(&port, 0).await.expect("turn");
        let user = snapshot
            .messages
            .iter_mut()
            .find_map(|message| match message {
                ConversationMessage::Chat(chat) if chat.role == "user" => Some(chat),
                _ => None,
            })
            .expect("user");
        user.recalled_memory = Some(recalled_fixture());
        port.replace_context(snapshot, "recall_fixture")
            .await
            .expect("attach recall");
        let input = port.latest_context().await.expect("context");
        let refs =
            crate::session::context_messages::context_refs(port.owner.store.as_ref(), &input)
                .expect("refs");
        let annotated = refs
            .iter()
            .find(|item| item.recalled_memory.is_some())
            .expect("annotation");
        let ContextMessageSource::Content { content_ref, .. } = &annotated.source else {
            panic!("original source")
        };
        let PayloadRead::Present(content) = port
            .owner
            .store
            .read_payload(&content_ref.payload_id)
            .expect("content")
        else {
            panic!("content")
        };
        let content: ContentPayload = serde_json::from_slice(&content.body).expect("content");
        let RecordedContentPart::Text { text_ref } = content.parts[0].clone() else {
            panic!("text")
        };
        for (index, target) in [
            annotated.recalled_memory.clone().expect("recall reference"),
            text_ref,
        ]
        .into_iter()
        .enumerate()
        {
            let intent = SessionEvent::typed(
                port.stream.clone(),
                EventData::PayloadErasureRequested {
                    targets: vec![target],
                    reason_category: "operator_requested".into(),
                },
                port.actor.clone(),
            )
            .expect("intent");
            port.owner
                .admit_erasure(
                    PrincipalId::shared_operator(),
                    authority.clone(),
                    AppendRequest {
                        session_id: port.stream.clone(),
                        expected_head: port.owner.store.head(&port.stream).expect("head"),
                        idempotency_key: format!("erase-recall-{index}"),
                        events: vec![intent],
                        payloads: vec![],
                        erasure_reference_commitment: None,
                    },
                )
                .await
                .expect("erasure");
            let resolved = crate::session::context_messages::resolve_message(
                port.owner.store.as_ref(),
                &port.stream,
                annotated,
            );
            if index == 0 {
                let ConversationMessage::Chat(chat) =
                    resolved.expect("resolve user").expect("user remains")
                else {
                    panic!("user")
                };
                assert!(chat.recalled_memory.is_none());
                assert!(
                    !zeroclaw_api::session_events::context_read::message_available(
                        port.owner.store.as_ref(),
                        &port.stream,
                        annotated
                    )
                    .expect("summary visibility")
                );
            } else {
                assert!(
                    !matches!(resolved, Ok(Some(_))),
                    "erased user text cannot reappear through annotated context"
                );
            }
        }
    }
}

#[test]
fn summary_budget_counts_tool_arguments_reasoning_and_utf8() {
    let calls = ConversationMessage::AssistantToolCalls {
        text: None,
        tool_calls: vec![zeroclaw_api::model_provider::ToolCall {
            id: "fixture-call".into(),
            name: "file_write".into(),
            arguments: "x".repeat(100_000),
            extra_content: None,
        }],
        reasoning_content: Some("r".repeat(4_000)),
    };
    let unicode = ConversationMessage::Chat(ChatMessage::user("架".repeat(4_000)));
    let expected = calls.chat_messages()[0].estimated_text_tokens()
        + unicode.chat_messages()[0].estimated_text_tokens();
    let actual = crate::session::summary_deck::tokens(&[calls, unicode]);
    assert_eq!(actual, expected);
    assert!(
        actual > 29_000,
        "structured payloads must trigger summarization before the dispatch guard"
    );
}

#[tokio::test]
async fn summary_publication_reads_only_the_checkpoint_tail_and_captured_invocations() {
    for archive_size in [0, 10_000] {
        let directory = tempfile::tempdir().expect("directory");
        let (mut port, _) = session(directory.path(), "sqlite");
        let store = Arc::new(super::scaling_tests::CountedStore::new(
            port.owner.store.clone(),
        ));
        port.owner = Arc::new(SessionCoordinator::new(store.clone()));
        port = port.with_summary_deck(Arc::new(Runtime::new()));
        let title = PayloadRecord::new(
            PayloadId::new(),
            "session.text/v1".into(),
            b"fixture".to_vec(),
        )
        .expect("title");
        for batch in 0..archive_size / 1_000 {
            let events = (0..1_000)
                .map(|_| {
                    SessionEvent::typed(
                        port.stream.clone(),
                        EventData::SessionMetadataChanged {
                            field: "title".into(),
                            new_value_ref: StoredPayloadRef::from(&title),
                        },
                        EventActor::default(),
                    )
                    .expect("event")
                })
                .collect();
            store
                .append_batch(AppendRequest {
                    session_id: port.stream.clone(),
                    expected_head: store.head(&port.stream).expect("head"),
                    idempotency_key: format!("archive-{batch}"),
                    events,
                    payloads: if batch == 0 {
                        vec![title.clone()]
                    } else {
                        vec![]
                    },
                    erasure_reference_commitment: None,
                })
                .expect("archive");
        }
        port.replace_context(
            ContextSnapshot {
                messages: vec![ConversationMessage::Chat(ChatMessage::user(
                    "current context",
                ))],
                history_has_trim_breadcrumb: Some(false),
            },
            "scaling_fixture",
        )
        .await
        .expect("current context boundary");
        // Warm the shared checkpoint as a normal status poll would. Subsequent
        // publication must read new work, never the archived prefix again.
        zeroclaw_api::session_events::catalog::SessionReducer::materialize(
            store.as_ref(),
            &port.stream,
        )
        .expect("checkpoint");
        store.archived_through.store(
            store.head(&port.stream).expect("head").sequence as usize,
            Ordering::SeqCst,
        );
        store.read_events.store(0, Ordering::SeqCst);
        for n in 0..10 {
            let snapshot = add_turn(&port, n).await.expect("turn");
            port.prepare_context(snapshot).await.expect("prepare");
            wait_job(&port).await.expect("summary");
        }
        assert_eq!(
            store.archived_reads.load(Ordering::SeqCst),
            0,
            "summary publication replayed the archived prefix; total page reads {}",
            store.read_events.load(Ordering::SeqCst)
        );
        let current = port
            .read_context_snapshot(port.latest_context().await.expect("projection"))
            .await
            .expect("snapshot")
            .expect("present");
        assert!(
            serde_json::to_string(&current.messages)
                .expect("messages")
                .contains("Historical summary deck")
        );
    }
}

#[tokio::test]
async fn retained_summaries_skip_raw_sources_but_observe_new_erasures() {
    for backend in ["sqlite", "jsonl"] {
        let directory = tempfile::tempdir().expect("directory");
        let (mut port, authority) = session(directory.path(), backend);
        port = port.with_summary_deck(Arc::new(Runtime::new()));
        for n in 0..10 {
            let snapshot = add_turn(&port, n).await.expect("turn");
            port.prepare_context(snapshot).await.expect("prepare");
            wait_job(&port).await.expect("summary");
        }
        let input = port.latest_context().await.expect("projection");
        let EventData::ContextProjectionCreated { result_ref, .. } = &input.body.data else {
            panic!("projection");
        };
        let PayloadRead::Present(body) = port
            .owner
            .store
            .read_payload(&result_ref.payload_id)
            .expect("body")
        else {
            panic!("present");
        };
        let body: serde_json::Value = serde_json::from_slice(&body.body).expect("context");
        let sources: Vec<MessageRef> =
            serde_json::from_value(body["summary_deck"]["cards"][0]["source"].clone())
                .expect("summary sources");
        let source = sources.first().expect("source");
        let leaf = match &source.source {
            ContextMessageSource::Conversation { payload_ref, .. } => payload_ref.clone(),
            ContextMessageSource::Content { content_ref, .. } => {
                let PayloadRead::Present(content) = port
                    .owner
                    .store
                    .read_payload(&content_ref.payload_id)
                    .expect("content")
                else {
                    panic!("present");
                };
                let content: ContentPayload =
                    serde_json::from_slice(&content.body).expect("content");
                let RecordedContentPart::Text { text_ref } = &content.parts[0] else {
                    panic!("text");
                };
                text_ref.clone()
            }
        };
        let mut snapshot = port
            .read_context_snapshot(input.clone())
            .await
            .expect("snapshot")
            .expect("present");
        let counted = Arc::new(super::scaling_tests::CountedStore::new(
            port.owner.store.clone(),
        ));
        counted.forbid(leaf.payload_id.clone());
        port.owner = Arc::new(SessionCoordinator::new(counted.clone()));
        crate::session::summary_deck::reconcile(
            &port.owner,
            &port.branch,
            &input,
            &EventId::new(),
            &mut snapshot,
        )
        .expect("render without original bytes");
        counted.clear_forbidden();
        let intent = SessionEvent::typed(
            port.stream.clone(),
            EventData::PayloadErasureRequested {
                targets: vec![leaf],
                reason_category: "operator_requested".into(),
            },
            port.actor.clone(),
        )
        .expect("intent");
        port.owner
            .admit_erasure(
                PrincipalId::shared_operator(),
                authority,
                AppendRequest {
                    session_id: port.stream.clone(),
                    expected_head: counted.head(&port.stream).expect("head"),
                    idempotency_key: "erase-summary-source".into(),
                    events: vec![intent],
                    payloads: vec![],
                    erasure_reference_commitment: None,
                },
            )
            .await
            .expect("erasure");
        assert!(
            crate::session::summary_deck::reconcile(
                &port.owner,
                &port.branch,
                &input,
                &EventId::new(),
                &mut snapshot
            )
            .is_err(),
            "a new erasure must invalidate the retained summary"
        );
    }
}

#[derive(Clone)]
struct Runtime {
    main: Option<SessionId>,
    enabled: Arc<AtomicBool>,
    fail: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
    gate: Option<Arc<tokio::sync::Semaphore>>,
}
impl Runtime {
    fn new() -> Self {
        Self {
            main: None,
            enabled: Arc::new(AtomicBool::new(true)),
            fail: Arc::new(AtomicBool::new(false)),
            calls: Arc::new(AtomicUsize::new(0)),
            gate: None,
        }
    }
}
impl SummaryDeckRuntime for Runtime {
    fn main_session(&self) -> anyhow::Result<Option<SessionId>> {
        Ok(self.main.clone())
    }
    fn resolve(&self) -> anyhow::Result<Option<ResolvedSummaryDeck>> {
        if !self.enabled.load(Ordering::SeqCst) {
            return Ok(None);
        }
        Ok(Some(ResolvedSummaryDeck {
            config: SummaryDeckConfig {
                enabled: true,
                max_summary_tokens: 128,
                timeout_secs: 2,
                ..Default::default()
            },
            context_budget: 1_200,
            max_turns: 8,
            source_budget: 4_000,
        }))
    }
    fn executor(&self) -> anyhow::Result<ContextSummaryExecutor<'static>> {
        let runtime = self.clone();
        Ok(Box::new(move |source| {
            Box::pin(async move {
                let observation = current_provider_admission()
                    .expect("captured summary")
                    .admit(ProviderDispatchIntent {
                        requested_route: None,
                        previous_attempt_event_id: None,
                        provider_ref: "fixture".into(),
                        model: "fixture".into(),
                        messages: source
                            .iter()
                            .flat_map(ConversationMessage::chat_messages)
                            .collect(),
                        tools: None,
                        temperature: None,
                        thinking: None,
                        tool_round_policy: zeroclaw_api::model_provider::ToolRoundPolicy::Batch,
                    })
                    .await?;
                let n = runtime.calls.fetch_add(1, Ordering::SeqCst) + 1;
                if let Some(gate) = runtime.gate {
                    gate.acquire_owned().await?.forget();
                }
                if runtime.fail.load(Ordering::SeqCst) {
                    anyhow::bail!("fixture_summary_failed");
                }
                let text = format!("Summary {n}: keep architecture decisions and their rationale.");
                let leaf = PayloadRecord::new(
                    PayloadId::new(),
                    "session.text/v1".into(),
                    text.as_bytes().to_vec(),
                )?;
                let response = PayloadRecord::json(
                    "session.response/v1",
                    &ProviderResponsePayload {
                        blocks: vec![ResponseBlock::Text {
                            text_ref: StoredPayloadRef::from(&leaf),
                        }],
                        completion: ContentCompletion::Complete,
                        finish_evidence_ref: None,
                        usage_ref: None,
                        reasoning_absence: Some("not_returned".into()),
                    },
                )?;
                observation
                    .observe(
                        EventData::ProviderAttemptCompleted {
                            response_ref: StoredPayloadRef::from(&response),
                            served_attribution_ref: None,
                            last_progress_event_id: None,
                        },
                        vec![leaf, response],
                    )
                    .await?;
                Ok(text)
            })
        }))
    }
}

async fn wait_job(port: &EngineSession) -> anyhow::Result<()> {
    let job = port
        .owner
        .summary_deck_jobs
        .lock()
        .unwrap()
        .get(&port.branch)
        .cloned();
    if let Some(job) = job {
        let mut rx = job.result.clone();
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            rx.wait_for(Option::is_some),
        )
        .await??;
        let result = rx.borrow().clone().expect("job result");
        result.map_err(|error| anyhow::Error::msg(format!("{error:#}")))?;
        port.owner
            .summary_deck_jobs
            .lock()
            .unwrap()
            .remove(&port.branch);
    }
    Ok(())
}

async fn add_turn(port: &EngineSession, n: usize) -> anyhow::Result<ContextSnapshot> {
    let prompt = format!("Architecture question {n}: preserve this original decision evidence.");
    let mut incoming = input();
    incoming.parts = vec![EngineInputPart::Text(prompt.clone())];
    port.record_context_input(incoming, &prompt, "fixture_message")
        .await?;
    port.append_assistant(&format!(
        "Answer {n}: retain the database design rationale."
    ))
    .await?;
    port.read_context_snapshot(port.latest_context().await?)
        .await?
        .ok_or_else(|| anyhow::Error::msg("missing context"))
}

#[tokio::test]
async fn deck_rotates_without_deleting_originals_and_survives_reopen() {
    for backend in ["sqlite", "jsonl"] {
        let directory = tempfile::tempdir().unwrap();
        let (port, authority) = session(directory.path(), backend);
        let runtime = Arc::new(Runtime::new());
        let port = port.with_summary_deck(runtime.clone());
        for n in 0..30 {
            let snapshot = add_turn(&port, n).await.unwrap();
            port.prepare_context(snapshot).await.unwrap();
            wait_job(&port).await.unwrap();
        }
        assert!(runtime.calls.load(Ordering::SeqCst) >= 5);
        let current = port
            .read_context_snapshot(port.latest_context().await.unwrap())
            .await
            .unwrap()
            .unwrap();
        let text = serde_json::to_string(&current.messages).unwrap();
        assert!(text.contains("Historical summary deck"));
        assert!(
            !text.contains("Summary 1:"),
            "oldest summary must leave the deck"
        );
        let messages = port
            .history_reader()
            .unwrap()
            .history(port.stream.as_str(), 100)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            messages.messages.len(),
            60,
            "archive retains every original message"
        );
        assert!(
            messages
                .messages
                .iter()
                .any(|message| message.text.contains("Architecture question 0:"))
        );
        let identity = messages.messages[0].message_id.clone();
        let stream = port.stream.clone();
        let branch = port.branch.clone();
        drop(port);
        let store =
            zeroclaw_infra::session_events::open_event_store(directory.path(), backend).unwrap();
        let reopened = EngineSession::new(
            Arc::new(SessionCoordinator::new(store)),
            stream,
            branch,
            PrincipalId::shared_operator(),
            authority,
        )
        .with_summary_deck(runtime.clone());
        let restored = reopened
            .read_context_snapshot(reopened.latest_context().await.unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::to_value(restored).unwrap(),
            serde_json::to_value(current).unwrap()
        );
        let messages = reopened
            .history_reader()
            .unwrap()
            .history(reopened.stream.as_str(), 100)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(messages.messages.len(), 60);
        assert_eq!(messages.messages[0].message_id, identity);
    }
}

#[tokio::test]
async fn background_summary_keeps_chat_running_and_rebases_newer_turns() {
    for backend in ["sqlite", "jsonl"] {
        let directory = tempfile::tempdir().unwrap();
        let (port, _) = session(directory.path(), backend);
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let mut runtime = Runtime::new();
        runtime.gate = Some(gate.clone());
        let runtime = Arc::new(runtime);
        let port = port.with_summary_deck(runtime.clone());
        for n in 0..5 {
            add_turn(&port, n).await.unwrap();
        }
        let snapshot = add_turn(&port, 5).await.unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            port.prepare_context(snapshot),
        )
        .await
        .unwrap()
        .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while runtime.calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("summary must freeze its input before the foreground edit");
        let mut stale = add_turn(&port, 6).await.unwrap();
        stale
            .messages
            .push(ConversationMessage::Chat(ChatMessage::assistant(
                "Newer derived context must survive background publication.",
            )));
        port.replace_context(stale.clone(), "fixture_newer_context")
            .await
            .unwrap();
        let before = port.latest_context().await.unwrap();
        let derived =
            crate::session::context_messages::context_refs(port.owner.store.as_ref(), &before)
                .unwrap()
                .last()
                .unwrap()
                .clone();
        gate.add_permits(1);
        wait_job(&port).await.unwrap();
        // An engine finishing with its pre-publication view cannot restore the
        // summarized prefix or overwrite the newer deck.
        port.replace_context(stale, "fixture_inflight_context")
            .await
            .unwrap();
        let current = port
            .read_context_snapshot(port.latest_context().await.unwrap())
            .await
            .unwrap()
            .unwrap();
        let text = serde_json::to_string(&current.messages).unwrap();
        assert!(text.contains("Summary 1:"));
        assert!(text.contains("Architecture question 6:"));
        assert!(!text.contains("Architecture question 0:"));
        assert!(text.contains("Newer derived context must survive background publication."));
        let latest = port.latest_context().await.unwrap();
        assert!(
            crate::session::context_messages::context_refs(port.owner.store.as_ref(), &latest,)
                .unwrap()
                .contains(&derived),
            "{backend}: the canonical identity must survive rebasing"
        );
        assert_eq!(
            port.history_reader()
                .unwrap()
                .history(port.stream.as_str(), 100)
                .await
                .unwrap()
                .unwrap()
                .messages
                .len(),
            14,
            "{backend}: derived context must not duplicate the original transcript"
        );
    }
}

#[tokio::test]
async fn transformed_conversation_identity_cannot_select_another_payload() {
    for backend in ["sqlite", "jsonl"] {
        let directory = tempfile::tempdir().unwrap();
        let (port, _) = session(directory.path(), backend);
        let mut snapshot = add_turn(&port, 0).await.unwrap();
        for text in ["First derived annotation", "Second derived annotation"] {
            snapshot
                .messages
                .push(ConversationMessage::Chat(ChatMessage::assistant(text)));
        }
        port.replace_context(snapshot, "fixture_derived_annotations")
            .await
            .unwrap();
        let input = port.latest_context().await.unwrap();
        let mut refs =
            crate::session::context_messages::context_refs(port.owner.store.as_ref(), &input)
                .unwrap();
        let first = refs.len() - 2;
        assert!(matches!(
            refs[first].source,
            ContextMessageSource::Conversation { .. }
        ));
        // Both payloads are valid and in scope. A known message identity must
        // still bind to its own canonical payload, rather than another item.
        refs[first].source = refs[first + 1].source.clone();
        let transform = port
            .owner
            .begin_context_transform(
                crate::session::context::ContextTransformCommand {
                    stream: port.stream.clone(),
                    branch: port.branch.clone(),
                    turn: None,
                    expected_projection: Some(input.event_id.clone()),
                    intent: ContextTransformIntent {
                        reason_key: "fixture_forgery".into(),
                        algorithm_version: "fixture/v1".into(),
                        budget: 0,
                    },
                    execution_binding: Some(port.authority.maintenance_binding().unwrap()),
                },
                port.authority.clone(),
            )
            .await
            .unwrap();
        let started = transform.started_event_id().unwrap();
        let payload = PayloadRecord::json(
            "session.context/v1",
            &serde_json::json!({
                "message_refs": refs, "history_has_trim_breadcrumb": false,
            }),
        )
        .unwrap();
        let mut projection = SessionEvent::typed(
            port.stream.clone(),
            EventData::ContextProjectionCreated {
                source_ranges: vec![SourceRange {
                    session_id: port.stream.clone(),
                    from_seq: input.sequence,
                    to_seq: input.sequence,
                    disposition: RangeDisposition::Included,
                }],
                summary_ref: None,
                budget: 0,
                reason_key: "fixture_forgery".into(),
                algorithm_version: "fixture/v1".into(),
                result_ref: StoredPayloadRef::from(&payload),
                summary_source_refs: vec![],
                transform_started_event_id: Some(started.clone()),
            },
            EventActor::System {
                component: "context_transform".into(),
            },
        )
        .unwrap();
        projection.branch_id = Some(port.branch.clone());
        projection.causation_id = Some(started.clone());
        let head = port.owner.store.head(&port.stream).unwrap();
        let error = port
            .owner
            .append(AppendRequest {
                session_id: port.stream.clone(),
                expected_head: head.clone(),
                idempotency_key: format!("fixture_forgery:{started}"),
                events: vec![projection],
                payloads: vec![payload],
                erasure_reference_commitment: None,
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("context conversation source has no canonical origin"),
            "{backend}: {error}"
        );
        assert_eq!(port.owner.store.head(&port.stream).unwrap(), head);
        transform
            .failed("fixture_forgery_rejected".into())
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn failed_and_disabled_summaries_preserve_original_context() {
    for backend in ["sqlite", "jsonl"] {
        let directory = tempfile::tempdir().unwrap();
        let (port, _) = session(directory.path(), backend);
        let runtime = Arc::new(Runtime::new());
        runtime.fail.store(true, Ordering::SeqCst);
        runtime.enabled.store(false, Ordering::SeqCst);
        let port = port.with_summary_deck(runtime.clone());
        for n in 0..6 {
            add_turn(&port, n).await.unwrap();
        }
        let before = port
            .read_context_snapshot(port.latest_context().await.unwrap())
            .await
            .unwrap()
            .unwrap();
        runtime.enabled.store(true, Ordering::SeqCst);
        port.maintain_summary_deck().await.unwrap();
        assert!(
            wait_job(&port)
                .await
                .unwrap_err()
                .to_string()
                .contains("fixture_summary_failed")
        );
        let after = port
            .read_context_snapshot(port.latest_context().await.unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::to_value(&before).unwrap(),
            serde_json::to_value(&after).unwrap()
        );
        runtime.enabled.store(false, Ordering::SeqCst);
        let calls = runtime.calls.load(Ordering::SeqCst);
        port.prepare_context(before).await.unwrap();
        assert_eq!(runtime.calls.load(Ordering::SeqCst), calls);
    }
}

#[tokio::test]
async fn background_publication_rechecks_live_enablement_and_authority() {
    for revoke in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let (port, authority) = session(directory.path(), "sqlite");
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let mut runtime = Runtime::new();
        runtime.gate = Some(gate.clone());
        runtime.enabled.store(false, Ordering::SeqCst);
        let runtime = Arc::new(runtime);
        let port = port.with_summary_deck(runtime.clone());
        for n in 0..6 {
            add_turn(&port, n).await.unwrap();
        }
        let before = port
            .read_context_snapshot(port.latest_context().await.unwrap())
            .await
            .unwrap()
            .unwrap();
        runtime.enabled.store(true, Ordering::SeqCst);
        port.maintain_summary_deck().await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while runtime.calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        if revoke {
            authority.0.store(false, Ordering::SeqCst);
        } else {
            runtime.enabled.store(false, Ordering::SeqCst);
        }
        gate.add_permits(1);
        assert!(wait_job(&port).await.is_err());
        authority.0.store(true, Ordering::SeqCst);
        let after = port
            .read_context_snapshot(port.latest_context().await.unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::to_value(before).unwrap(),
            serde_json::to_value(after).unwrap()
        );
    }
}

#[tokio::test]
async fn main_conversation_refreshes_stale_connections_inside_the_execution_lane() {
    for backend in ["sqlite", "jsonl"] {
        let directory = tempfile::tempdir().unwrap();
        let (port, _) = session(directory.path(), backend);
        let mut runtime = Runtime::new();
        runtime.enabled.store(false, Ordering::SeqCst);
        runtime.main = Some(port.stream.clone());
        let port = port.with_summary_deck(Arc::new(runtime));
        add_turn(&port, 0).await.unwrap();
        let stale = port
            .read_context_snapshot(port.latest_context().await.unwrap())
            .await
            .unwrap()
            .unwrap();
        for (prompt, reply) in [
            ("Web architecture follow-up", "Web rationale"),
            ("Telegram continuation", "Telegram rationale"),
        ] {
            let mut incoming = input();
            incoming.parts = vec![EngineInputPart::Text(prompt.into())];
            let mut view = stale.clone();
            view.messages
                .push(ConversationMessage::Chat(ChatMessage::user(prompt)));
            let active = port.clone();
            port.execute(
                incoming,
                Box::pin(async move {
                    let mut selected = active.prepare_context(view).await?;
                    let context = serde_json::to_string(&selected.messages)?;
                    if prompt == "Telegram continuation" {
                        assert!(context.contains("Web architecture follow-up"));
                        assert!(context.contains("Web rationale"));
                    }
                    selected
                        .messages
                        .push(ConversationMessage::Chat(ChatMessage::assistant(reply)));
                    active
                        .replace_context(selected, "fixture_completed_turn")
                        .await?;
                    Ok(reply.into())
                }),
            )
            .await
            .unwrap();
        }
        let current = port
            .read_context_snapshot(port.latest_context().await.unwrap())
            .await
            .unwrap()
            .unwrap();
        let text = serde_json::to_string(&current.messages).unwrap();
        assert!(text.contains("Telegram continuation"));
        assert_eq!(text.matches("Web architecture follow-up").count(), 1);
    }
}
