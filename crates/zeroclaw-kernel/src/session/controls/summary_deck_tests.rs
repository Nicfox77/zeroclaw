use super::*;
use crate::session::summary_deck::{ResolvedSummaryDeck, SummaryDeckRuntime};
use std::sync::atomic::AtomicUsize;
use zeroclaw_config::summary_deck::SummaryDeckConfig;

#[derive(Clone)]
struct Runtime {
    enabled: Arc<AtomicBool>,
    fail: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
    gate: Option<Arc<tokio::sync::Semaphore>>,
}
impl Runtime {
    fn new() -> Self {
        Self {
            enabled: Arc::new(AtomicBool::new(true)),
            fail: Arc::new(AtomicBool::new(false)),
            calls: Arc::new(AtomicUsize::new(0)),
            gate: None,
        }
    }
}
impl SummaryDeckRuntime for Runtime {
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
    let directory = tempfile::tempdir().unwrap();
    let (port, _) = session(directory.path(), "sqlite");
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
    let stale = add_turn(&port, 6).await.unwrap();
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
