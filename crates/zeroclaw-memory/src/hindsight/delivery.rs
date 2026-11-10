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
    if !auto_retain {
        return Ok(());
    }
    // A bounded page per registered stream. Capacity leaves the durable cursor
    // behind the transcript rather than dropping turns or blocking chat.
    for stream in &streams {
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
    for job in ledger.backfills()? {
        let Some(stream) = streams.iter().find(|s| s.id == job.stream) else {
            continue;
        };
        let mut reader = stream.clone();
        reader.cursor = job.cursor;
        let events = match source.next(&reader).await {
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
        for event in events
            .into_iter()
            .take_while(|event| event.sequence <= job.through)
        {
            let eligible = job.since.is_none_or(|s| event.occurred_at >= s)
                && job.until.is_none_or(|u| event.occurred_at <= u);
            if !ledger.advance_backfill(
                &job,
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
        }
    }
    for delivery in ledger.pending(16, retry)? {
        let stream = streams
            .iter()
            .find(|s| s.id == delivery.stream)
            .ok_or_else(|| anyhow::Error::msg("hindsight_delivery_source_missing"))?;
        if let Err(error) = deliver(live, source, &ledger, stream, &delivery).await {
            // Persist a stable diagnostic key, never arbitrary upstream text or content.
            let key = error
                .downcast_ref::<super::client::Error>()
                .map(|e| format!("hindsight_{:?}", e.kind))
                .unwrap_or_else(|| "hindsight_delivery_failed".into());
            ledger.delivery_state(&delivery.id, "failed", None, Some(&key))?;
        }
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
        ledger.delivery_state(&delivery.id, "erased", None, None)?;
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
    let operation = ledger.operation(&bank, document, &delivery.id)?;
    ledger.delivery_state(&delivery.id, "processing", Some(&operation), None)?;
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
                        backend.policy()?;
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
    let revision = source.revision(stream).await?;
    anyhow::ensure!(
        revision == source_revision,
        "hindsight_source_changed_during_delivery"
    );
    ledger.checked_revision(&delivery.id, &revision)?;
    backend.policy()?;
    backend.prepare_bank(&bank).await?;
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
    ledger.delivery_state(&delivery.id, "erased", None, None)?;
    Ok(())
}

async fn resume_notes(
    live: &LiveConfigHandle,
    ledger: &Ledger,
    retry_failed: bool,
) -> anyhow::Result<()> {
    for note in ledger.pending_notes(retry_failed)? {
        let result = async {
            if ledger.status(&note.operation)?.as_deref() == Some("completed") {
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
                            backend.policy()?;
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
                    backend.prepare_bank(&bank).await?;
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
