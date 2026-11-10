//! Resumable delivery from canonical transcript references, never a memory store.
use super::{
    HindsightMemory,
    client::Item,
    ledger::{Delivery, Ledger, Stream},
};
use std::sync::Arc;
use zeroclaw_api::memory_traits::PrincipalScope;
use zeroclaw_config::live::LiveConfigHandle;

/// The host supplies authorized native reads. Hidden/reasoning/injected content
/// is deliberately absent from this interface.
#[async_trait::async_trait]
pub trait TranscriptSource: Send + Sync {
    async fn revision(&self, stream: &Stream) -> anyhow::Result<String>;
    async fn next(&self, stream: &Stream) -> anyhow::Result<Vec<SourceEvent>>;
    async fn load(&self, stream: &Stream, terminal: &str) -> anyhow::Result<Option<Vec<Item>>>;
}
pub struct SourceEvent {
    pub sequence: u64,
    pub occurred_at: i64,
    pub committed_turn: Option<String>,
}

pub async fn tick(live: &LiveConfigHandle, source: &dyn TranscriptSource) -> anyhow::Result<()> {
    let (data, capacity, retry, auto_retain) = {
        let config = live.read();
        if config.memory.backend != "hindsight" {
            return Ok(());
        }
        config.validate_hindsight_memory()?;
        (
            config.data_dir.clone(),
            config.memory.hindsight.queue_capacity,
            config.memory.hindsight.retry_failed_retains,
            config.memory.hindsight.auto_retain,
        )
    };
    let ledger = Arc::new(Ledger::open(&data)?);
    resume_notes(live, &ledger, retry).await?;
    let streams = ledger.streams()?;
    for stream in &streams {
        if let Ok(revision) = source.revision(stream).await {
            for delivery in ledger.revision_candidates(&stream.id, &revision)? {
                let checked = if delivery.status == "erasing" {
                    Ok(None)
                } else {
                    source.load(stream, &delivery.terminal).await
                };
                match checked {
                    Ok(None) => {
                        ledger.delivery_state(&delivery.id, "erasing", None, None)?;
                        if erase(live, &ledger, &delivery).await.is_err() {
                            ledger.observe("erasure", Some("hindsight_erasure_pending"), 0)?;
                        }
                    }
                    Ok(Some(_)) => ledger.checked_revision(&delivery.id, &revision)?,
                    Err(_) => {
                        ledger.observe("erasure", Some("hindsight_source_recheck_failed"), 0)?
                    }
                }
            }
        }
    }
    resume_rebuilds(live, &ledger).await?;
    if !auto_retain {
        return Ok(());
    }
    // A bounded page per registered stream. Capacity leaves the durable cursor
    // behind the transcript rather than dropping turns or blocking chat.
    for stream in &streams {
        // Historical sources must be enqueued before a newer live turn.
        if ledger.stream_backfilling(&stream.id)? {
            continue;
        }
        if !live.read().agent(&stream.agent).is_some_and(|a| {
            a.enabled
                && a.memory.backend == zeroclaw_config::multi_agent::MemoryBackendKind::Hindsight
        }) {
            continue;
        }
        let events = match source.next(stream).await {
            Ok(events) => events,
            Err(_) => {
                ledger.observe(
                    "transcript_read",
                    Some("hindsight_transcript_unavailable"),
                    0,
                )?;
                continue;
            }
        };
        for event in events {
            if !ledger.advance(
                stream,
                event.sequence,
                event.committed_turn.as_deref(),
                capacity,
            )? {
                break;
            }
        }
    }
    enqueue_backfills(&ledger, source, &streams, capacity).await?;
    for delivery in ledger.pending(16, retry)? {
        let stream = streams
            .iter()
            .find(|s| s.id == delivery.stream)
            .ok_or_else(|| anyhow::Error::msg("hindsight_delivery_source_missing"))?;
        if let Err(error) = deliver(live, source, &ledger, stream, &delivery).await {
            // Persist a stable diagnostic key, never arbitrary upstream text or content.
            let key = if error.to_string() == "hindsight_document_format_requires_explicit_reimport"
            {
                "hindsight_document_format_requires_explicit_reimport".into()
            } else {
                error
                    .downcast_ref::<super::client::Error>()
                    .map(|e| format!("hindsight_{:?}", e.kind))
                    .unwrap_or_else(|| "hindsight_delivery_failed".into())
            };
            ledger.delivery_state(&delivery.id, "failed", None, Some(&key))?;
        }
    }
    resume_rebuilds(live, &ledger).await?;
    Ok(())
}
/// Merge canonical event-page frontiers rather than sorting a copied archive.
/// Refill a depleted source before emitting another source's newer event.
async fn enqueue_backfills(
    ledger: &Ledger,
    source: &dyn TranscriptSource,
    streams: &[Stream],
    capacity: usize,
) -> anyhow::Result<()> {
    let mut views: Vec<_> = ledger
        .backfills()?
        .into_iter()
        .filter_map(|job| {
            streams
                .iter()
                .find(|stream| stream.id == job.stream)
                .map(|stream| {
                    (
                        job,
                        stream.clone(),
                        std::collections::VecDeque::<SourceEvent>::new(),
                    )
                })
        })
        .collect();
    for _ in 0..256 {
        for index in (0..views.len()).rev() {
            let (job, stream, events) = &mut views[index];
            if job.cursor >= job.through {
                views.remove(index);
                continue;
            }
            if events.is_empty() {
                let mut reader = stream.clone();
                reader.cursor = job.cursor;
                match source.next(&reader).await {
                    Ok(page) => events.extend(
                        page.into_iter()
                            .take_while(|event| event.sequence <= job.through),
                    ),
                    Err(_) => {
                        ledger.observe(
                            "transcript_read",
                            Some("hindsight_transcript_unavailable"),
                            0,
                        )?;
                        return Ok(());
                    }
                }
                if events.is_empty() {
                    // Without this source's frontier, a newer source cannot
                    // safely overtake it. Preserve the durable import intent.
                    ledger.observe("transcript_read", Some("hindsight_backfill_source_gap"), 0)?;
                    return Ok(());
                }
            }
        }
        let Some(index) = views
            .iter()
            .enumerate()
            .filter_map(|(index, (_, _, events))| {
                events
                    .front()
                    .map(|event| (index, event.occurred_at, event.sequence))
            })
            .min_by_key(|(_, time, sequence)| (*time, *sequence))
            .map(|(index, _, _)| index)
        else {
            break;
        };
        let (job, stream, events) = &mut views[index];
        let event = events
            .pop_front()
            .ok_or_else(|| anyhow::Error::msg("hindsight_backfill_frontier_missing"))?;
        let eligible = job.since.is_none_or(|since| event.occurred_at >= since)
            && job.until.is_none_or(|until| event.occurred_at <= until);
        if !ledger.advance_backfill(
            job,
            stream,
            event.sequence,
            if eligible {
                event.committed_turn.as_deref()
            } else {
                None
            },
            capacity,
        )? {
            break;
        }
        job.cursor = event.sequence;
    }
    Ok(())
}

async fn deliver(
    live: &LiveConfigHandle,
    source: &dyn TranscriptSource,
    ledger: &Ledger,
    stream: &Stream,
    delivery: &Delivery,
) -> anyhow::Result<()> {
    let config = live.read().clone();
    let backend = HindsightMemory::new(
        &config.memory.hindsight,
        &config.data_dir,
        Some(&stream.agent),
        Some(live.clone()),
    )?;
    let scope = PrincipalScope::new(&stream.principal).with_agent(Some(stream.agent.clone()));
    let bank = backend.bank_unchecked(&scope)?;
    let source_revision = source.revision(stream).await?;
    let Some(mut items) = source.load(stream, &delivery.terminal).await? else {
        if delivery.operation.is_some() {
            ledger.delivery_state(&delivery.id, "erasing", None, None)?;
            erase(live, ledger, delivery).await?;
        } else {
            ledger.delivery_state(&delivery.id, "erased", None, None)?;
        }
        return Ok(());
    };
    if items.is_empty() {
        ledger.delivery_state(&delivery.id, "completed", None, None)?;
        return Ok(());
    }
    let scanner = crate::scanned::ScannedMemory::with_live_policy(
        crate::none::NoneMemory::new("none"),
        live.clone(),
    );
    for item in &mut items {
        item.content = scanner.process_content(
            &item.document_id,
            &super::eligible_content(&item.content),
            None,
        )?;
        if item.update_mode.as_deref() == Some("append") {
            let _: Vec<serde_json::Value> = serde_json::from_str(&item.content)?;
            anyhow::ensure!(
                !ledger.has_legacy_sources(&stream.id)?,
                "hindsight_document_format_requires_explicit_reimport"
            );
        }
        scanner
            .enforce_policy(
                &item.document_id,
                None,
                &zeroclaw_api::memory_traits::MemoryCategory::Conversation,
            )
            .await?;
    }
    let document = &items[0].document_id;
    anyhow::ensure!(
        items.iter().all(|i| &i.document_id == document),
        "hindsight_delivery_document_mismatch"
    );
    // Stable source identity, not generated text: a redaction policy change
    // cannot accidentally allocate another operation after uncertain acceptance.
    let operation = if let Some(operation) = &delivery.operation {
        let (recorded_bank, recorded_document) = ledger.operation_source(operation)?;
        anyhow::ensure!(
            recorded_bank == bank && recorded_document == *document,
            "hindsight_document_format_requires_explicit_reimport"
        );
        operation.clone()
    } else {
        let digest = if items[0].update_mode.as_deref() == Some("append") {
            format!("{}:{}", delivery.id, ledger.generation(&stream.id)?)
        } else {
            delivery.id.clone()
        };
        ledger.operation(&bank, document, &digest)?
    };
    let revision = source.revision(stream).await?;
    anyhow::ensure!(
        revision == source_revision,
        "hindsight_source_changed_during_delivery"
    );
    ledger.attach_validated_source(&delivery.id, &operation, &revision)?;
    if delivery.operation.is_some() {
        match backend.client.operation(&bank, &operation).await {
            Ok(op) => {
                ledger.update(&operation, &op.status)?;
                match op.status.as_str() {
                    "completed" => {
                        ledger.delivery_state(&delivery.id, "completed", None, None)?;
                        return Ok(());
                    }
                    "failed" | "cancelled" => {
                        anyhow::ensure!(
                            backend.policy()?.retry_failed_retains
                                || delivery.status == "retry_requested",
                            "hindsight_retain_failed_manual_retry_required"
                        );
                        backend
                            .client
                            .request(
                                reqwest::Method::POST,
                                &[
                                    "v1",
                                    "default",
                                    "banks",
                                    &bank,
                                    "operations",
                                    &operation,
                                    "retry",
                                ],
                                None,
                                false,
                            )
                            .await?;
                        return Ok(());
                    }
                    _ => return Ok(()),
                }
            }
            Err(error) if error.status == Some(404) => {}
            Err(error) => return Err(error.into()),
        }
    }
    backend.prepare_bank(&bank, &scope).await?;
    let revision = source.revision(stream).await?;
    anyhow::ensure!(
        revision == source_revision,
        "hindsight_source_changed_during_delivery"
    );
    backend.policy()?;
    backend.client.retain(&bank, &items, &operation).await?;
    ledger.update(&operation, "processing")?;
    // Acceptance is deliberately not completion. The next tick reconciles it.
    Ok(())
}

async fn erase(
    live: &LiveConfigHandle,
    ledger: &Ledger,
    delivery: &Delivery,
) -> anyhow::Result<()> {
    let Some(operation) = &delivery.operation else {
        ledger.delivery_state(&delivery.id, "erased", None, None)?;
        return Ok(());
    };
    let config = live.read().clone();
    anyhow::ensure!(
        config.memory.backend == "hindsight",
        "hindsight_backend_revoked"
    );
    let client = super::client::Client::new(&config.memory.hindsight)?;
    let (bank, document) = ledger.operation_source(operation)?;
    if document.starts_with("zeroclaw:conversation:") {
        ledger.begin_rebuild(&delivery.stream, &bank, &document)?;
        return Ok(());
    }
    match client.operation(&bank, operation).await {
        Ok(operation) if matches!(operation.status.as_str(), "pending" | "processing") => {
            return Ok(());
        }
        Ok(_) => {}
        Err(error) if error.status == Some(404) => {}
        Err(error) => return Err(error.into()),
    }
    match client
        .request(
            reqwest::Method::DELETE,
            &["v1", "default", "banks", &bank, "documents", &document],
            None,
            false,
        )
        .await
    {
        Ok(_) => {}
        Err(error) if error.status == Some(404) => {}
        Err(error) => return Err(error.into()),
    }
    if !refresh_erased_models(&client, ledger, &delivery.id, &bank).await? {
        return Ok(());
    }
    ledger.delivery_state(&delivery.id, "erased", None, None)?;
    Ok(())
}

/// A turn inside a growing document cannot be removed by deleting its former
/// per-turn document. Drain writers, delete the conversation, then replay valid
/// canonical references in order under a fresh generation. Retrieval stays fenced
/// through replay and mental-model refresh, including across process restarts.
async fn resume_rebuilds(live: &LiveConfigHandle, ledger: &Ledger) -> anyhow::Result<()> {
    for rebuild in ledger.rebuilds()? {
        let config = live.read().clone();
        anyhow::ensure!(
            config.memory.backend == "hindsight",
            "hindsight_backend_revoked"
        );
        let client = super::client::Client::new(&config.memory.hindsight)?;
        let result = async {
            match rebuild.phase.as_str() {
                "draining" => {
                    let pending = ledger.draining_operations(&rebuild.stream)?;
                    for operation in &pending {
                        let status = match client.operation(&rebuild.bank, operation).await {
                            Ok(operation) => operation.status,
                            // An uncertain submission may still be accepted.
                            // Absence is not proof of cancellation: keep the bank
                            // fenced until the writer has a known terminal state.
                            Err(error) => return Err(error.into()),
                        };
                        ledger.update(operation, &status)?;
                    }
                    if !pending.is_empty() {
                        return Ok(());
                    }
                    match client
                        .request(
                            reqwest::Method::DELETE,
                            &[
                                "v1",
                                "default",
                                "banks",
                                &rebuild.bank,
                                "documents",
                                &rebuild.document,
                            ],
                            None,
                            false,
                        )
                        .await
                    {
                        Ok(_) => {}
                        Err(error) if error.status == Some(404) => {}
                        Err(error) => return Err(error.into()),
                    }
                    ledger.replay_rebuild(&rebuild)?;
                }
                "replaying" if ledger.replay_complete(&rebuild.stream)? => {
                    ledger.rebuild_phase(&rebuild, "refreshing")?;
                }
                "refreshing" => {
                    if refresh_erased_models(&client, ledger, &rebuild.generation, &rebuild.bank)
                        .await?
                    {
                        ledger.rebuild_phase(&rebuild, "completed")?;
                    }
                }
                "replaying" => {}
                _ => anyhow::bail!("hindsight_invalid_rebuild_phase"),
            }
            Ok::<(), anyhow::Error>(())
        }
        .await;
        if result.is_err() {
            ledger.observe(
                "conversation_rebuild",
                Some("hindsight_conversation_erasure_rebuild_pending"),
                0,
            )?;
        } else if !ledger.rebuild_pending(&rebuild.bank)? {
            ledger.observe("conversation_rebuild", None, 0)?;
        }
    }
    Ok(())
}

/// Native deletion schedules automatic retraction refreshes, but does not wait
/// for them and leaves manual models alone. Keep the bank fenced until every
/// model has completed an explicit full refresh. Receipts survive a restart.
async fn refresh_erased_models(
    client: &super::client::Client,
    ledger: &Ledger,
    delivery: &str,
    bank: &str,
) -> anyhow::Result<bool> {
    let active = client
        .request_query(
            reqwest::Method::GET,
            &["v1", "default", "banks", bank, "operations"],
            None,
            false,
            &[
                ("type", "refresh_mental_model".into()),
                ("status", "processing".into()),
                ("limit", "1".into()),
            ],
        )
        .await?;
    let total = active
        .get("total")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow::Error::msg("hindsight_invalid_operations"))?;
    if total > 0 {
        return Ok(false);
    }
    let response = client
        .request_query(
            reqwest::Method::GET,
            &["v1", "default", "banks", bank, "mental-models"],
            None,
            false,
            &[("limit", "100".into())],
        )
        .await?;
    let models = response
        .get("items")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::Error::msg("hindsight_invalid_mental_models"))?;
    anyhow::ensure!(
        response.get("total").and_then(serde_json::Value::as_u64) == Some(models.len() as u64),
        "hindsight_erasure_model_limit"
    );
    let mut complete = true;
    for model in models {
        let id = model
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::Error::msg("hindsight_invalid_mental_model"))?;
        if let Some(operation) = ledger.erasure_refresh(delivery, id)? {
            match client.operation(bank, &operation).await {
                Ok(state) if state.status == "completed" => continue,
                Ok(state) if matches!(state.status.as_str(), "failed" | "cancelled") => {
                    client
                        .request(
                            reqwest::Method::POST,
                            &[
                                "v1",
                                "default",
                                "banks",
                                bank,
                                "operations",
                                &operation,
                                "retry",
                            ],
                            None,
                            false,
                        )
                        .await?;
                    complete = false;
                    continue;
                }
                Ok(_) => {
                    complete = false;
                    continue;
                }
                Err(error) if error.status == Some(404) => {}
                Err(error) => return Err(error.into()),
            }
        }
        // A full refresh deliberately preserves old content when no evidence
        // remains. Clear the obsolete baseline first, including its embedding,
        // so "nothing to synthesize" cannot keep a deleted fact retrievable.
        client
            .request(
                reqwest::Method::POST,
                &["v1", "default", "banks", bank, "mental-models", id, "clear"],
                None,
                false,
            )
            .await?;
        let refresh = client
            .request(
                reqwest::Method::POST,
                &[
                    "v1",
                    "default",
                    "banks",
                    bank,
                    "mental-models",
                    id,
                    "refresh",
                ],
                Some(&serde_json::json!({"mode":"full"})),
                false,
            )
            .await?;
        let operation = refresh
            .get("operation_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::Error::msg("hindsight_invalid_refresh_operation"))?;
        ledger.record_erasure_refresh(delivery, id, operation)?;
        complete = false;
    }
    Ok(complete)
}

async fn resume_notes(
    live: &LiveConfigHandle,
    ledger: &Ledger,
    retry_failed: bool,
) -> anyhow::Result<()> {
    for note in ledger.pending_notes(retry_failed)? {
        let result = async {
            let local_status = ledger.status(&note.operation)?;
            let retry_requested = local_status.as_deref() == Some("retry_requested");
            if local_status.as_deref() == Some("completed") {
                ledger.complete_note(&note.operation)?;
                return Ok(());
            }
            let config = live.read().clone();
            let backend = HindsightMemory::new(
                &config.memory.hindsight,
                &config.data_dir,
                Some(&note.agent),
                Some(live.clone()),
            )?;
            let bank = backend.bank(&note.scope)?;
            let (recorded, document) = ledger.operation_source(&note.operation)?;
            anyhow::ensure!(
                recorded == bank && note.items.iter().all(|i| i.document_id == document),
                "hindsight_note_scope_mismatch"
            );
            match backend.client.operation(&bank, &note.operation).await {
                Ok(operation) => {
                    ledger.update(&note.operation, &operation.status)?;
                    match operation.status.as_str() {
                        "completed" => ledger.complete_note(&note.operation)?,
                        "failed" | "cancelled" => {
                            anyhow::ensure!(
                                backend.policy()?.retry_failed_retains || retry_requested,
                                "hindsight_retain_failed_manual_retry_required"
                            );
                            backend
                                .client
                                .request(
                                    reqwest::Method::POST,
                                    &[
                                        "v1",
                                        "default",
                                        "banks",
                                        &bank,
                                        "operations",
                                        &note.operation,
                                        "retry",
                                    ],
                                    None,
                                    false,
                                )
                                .await?;
                            ledger.update(&note.operation, "processing")?;
                        }
                        _ => {}
                    }
                }
                Err(error) if error.status == Some(404) => {
                    let scanner = crate::scanned::ScannedMemory::with_live_policy(
                        crate::none::NoneMemory::new("none"),
                        live.clone(),
                    );
                    let mut items = note.items;
                    for item in &mut items {
                        item.content = scanner.process_content(
                            &item.document_id,
                            &super::eligible_content(&item.content),
                            note.scope.namespace.as_deref(),
                        )?;
                        scanner
                            .enforce_policy(
                                &item.document_id,
                                note.scope.namespace.as_deref(),
                                &zeroclaw_api::memory_traits::MemoryCategory::Core,
                            )
                            .await?;
                    }
                    backend.prepare_bank(&bank, &note.scope).await?;
                    backend.policy()?;
                    backend
                        .client
                        .retain(&bank, &items, &note.operation)
                        .await?;
                    ledger.update(&note.operation, "processing")?;
                }
                Err(error) => return Err(anyhow::Error::new(error)),
            }
            ledger.defer_note(&note.operation, false)?;
            Ok(())
        }
        .await;
        if result.is_err() {
            ledger.defer_note(&note.operation, true)?;
            ledger.observe("note_retain", Some("hindsight_note_delivery_failed"), 0)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "recovery_tests.rs"]
mod recovery_tests;

#[cfg(test)]
mod tests {
    use super::*;
    struct CanonicalPages;
    #[async_trait::async_trait]
    impl TranscriptSource for CanonicalPages {
        async fn revision(&self, _stream: &Stream) -> anyhow::Result<String> {
            Ok("fixture-frontier".into())
        }
        async fn load(
            &self,
            _stream: &Stream,
            _terminal: &str,
        ) -> anyhow::Result<Option<Vec<Item>>> {
            anyhow::bail!("fixture_only_exercises_ingestion_order")
        }
        async fn next(&self, stream: &Stream) -> anyhow::Result<Vec<SourceEvent>> {
            let sequence = stream.cursor + 1;
            if sequence > 2 {
                return Ok(vec![]);
            }
            Ok(vec![SourceEvent {
                sequence,
                occurred_at: if stream.session == "a" { 1000 } else { 2000 }
                    + (sequence as i64 - 1) * 2000,
                committed_turn: Some(format!("{}:{sequence}", stream.session)),
            }])
        }
    }
    #[tokio::test]
    async fn chronological_cross_session_replay_refills_pages_and_resumes_after_backpressure() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(directory.path()).unwrap();
        for session in ["a", "b"] {
            ledger
                .register_stream("fixture", "owner", session, "branch", 2)
                .unwrap();
        }
        let streams = ledger.streams().unwrap();
        for stream in &streams {
            ledger.backfill(&stream.id, 2, None, None).unwrap();
        }
        enqueue_backfills(&ledger, &CanonicalPages, &streams, 2)
            .await
            .unwrap();
        let first = ledger.pending(8, true).unwrap();
        assert_eq!(
            first
                .iter()
                .map(|item| item.terminal.as_str())
                .collect::<Vec<_>>(),
            vec!["a:1", "b:1"]
        );
        for item in first {
            ledger
                .delivery_state(&item.id, "completed", None, None)
                .unwrap();
        }
        drop(ledger);
        let ledger = Ledger::open(directory.path()).unwrap();
        enqueue_backfills(&ledger, &CanonicalPages, &ledger.streams().unwrap(), 2)
            .await
            .unwrap();
        assert_eq!(
            ledger
                .pending(8, true)
                .unwrap()
                .iter()
                .map(|item| item.terminal.as_str())
                .collect::<Vec<_>>(),
            vec!["a:2", "b:2"]
        );
        assert!(ledger.backfills().unwrap().is_empty());
    }
}
