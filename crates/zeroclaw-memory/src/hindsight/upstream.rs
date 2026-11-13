//! Completed-turn delivery over upstream session ownership, without native history.
//! Only unsent delivery bodies are spooled; this is not a session history store.
use super::{
    client::Item,
    delivery::{SourceEvent, TranscriptSource},
    ledger::{Ledger, Stream, hash},
    source::TranscriptAuthority,
};
use rusqlite::{Connection, OptionalExtension, params};
use std::{collections::BTreeMap, sync::Mutex};
use zeroclaw_config::live::LiveConfigHandle;
use zeroclaw_infra::{session_backend::SessionMetadata, session_sqlite::SqliteSessionBackend};

tokio::task_local! { static TURN_PRINCIPAL: Option<zeroclaw_api::memory_traits::PrincipalScope>; }
/// Scope supplied by admitted channel execution, never by model tool arguments.
pub async fn scope_principal<T>(
    scope: Option<zeroclaw_api::memory_traits::PrincipalScope>,
    future: impl std::future::Future<Output = T>,
) -> T {
    TURN_PRINCIPAL.scope(scope, future).await
}

/// Read handles do not open or migrate session storage until a bank has sources.
pub struct SessionAuthority(pub LiveConfigHandle);
impl TranscriptAuthority for SessionAuthority {
    fn admitted_scope(&self) -> anyhow::Result<zeroclaw_api::memory_traits::PrincipalScope> {
        TURN_PRINCIPAL
            .try_with(Clone::clone)
            .ok()
            .flatten()
            .ok_or_else(|| anyhow::Error::msg("hindsight_principal_scope_required"))
    }
    fn check_sources(&self, ledger: &Ledger, bank: &str) -> anyhow::Result<()> {
        if !ledger.has_transcript_sources(bank)? {
            return Ok(());
        }
        SessionTranscripts::new(self.0.clone())?.check_sources(ledger, bank)
    }
}

pub struct SessionTranscripts {
    live: LiveConfigHandle,
    store: SqliteSessionBackend,
    outbox: Mutex<Connection>,
}
impl SessionTranscripts {
    pub fn new(live: LiveConfigHandle) -> anyhow::Result<Self> {
        let data = live.read().data_dir.clone();
        let ledger = Ledger::open(&data)?;
        anyhow::ensure!(
            ledger.streams()?.is_empty() || SqliteSessionBackend::db_path(&data).try_exists()?,
            "hindsight_source_store_missing"
        );
        let store = SqliteSessionBackend::new(&data)?;
        let mut outbox = Connection::open(SqliteSessionBackend::db_path(&data))?;
        outbox.busy_timeout(std::time::Duration::from_secs(5))?;
        outbox.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA secure_delete=ON;
          CREATE TABLE IF NOT EXISTS hindsight_pending(stream TEXT NOT NULL,sequence INTEGER NOT NULL,terminal TEXT NOT NULL UNIQUE,occurred INTEGER NOT NULL,fingerprint TEXT NOT NULL,body TEXT,PRIMARY KEY(stream,sequence));")?;
        // Upgrade the earlier standalone prototype without losing receipts or
        // pending payloads. Commit the new copy before retiring the old rows;
        // repeating after interruption is safe and does not revive released text.
        let legacy_path = data.join("hindsight/transcript-outbox.db");
        if legacy_path.try_exists()? {
            let mut legacy = Connection::open(&legacy_path)?;
            legacy.busy_timeout(std::time::Duration::from_secs(5))?;
            legacy.pragma_update(None, "secure_delete", "ON")?;
            let old = legacy.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            let tx = outbox.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            {
                let mut query = old.prepare(
                    "SELECT stream,sequence,terminal,occurred,fingerprint,body FROM pending",
                )?;
                let rows = query.query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, u64>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, Option<String>>(5)?,
                    ))
                })?;
                for row in rows {
                    let (stream, sequence, terminal, occurred, fingerprint, body) = row?;
                    tx.execute("INSERT OR IGNORE INTO hindsight_pending(stream,sequence,terminal,occurred,fingerprint,body) VALUES(?1,?2,?3,?4,?5,?6)", params![stream,sequence,terminal,occurred,fingerprint,body])?;
                    let matches: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM hindsight_pending WHERE stream=?1 AND sequence=?2 AND terminal=?3 AND fingerprint=?4)", params![stream,sequence,terminal,fingerprint], |r|r.get(0))?;
                    anyhow::ensure!(matches, "hindsight_outbox_migration_identity_conflict");
                }
            }
            tx.commit()?;
            old.execute("DELETE FROM pending", [])?;
            old.commit()?;
        }
        Ok(Self {
            store,
            live,
            outbox: Mutex::new(outbox),
        })
    }
    fn ledger(&self) -> anyhow::Result<Ledger> {
        Ledger::open(&self.live.read().data_dir)
    }
    fn principal(&self, metadata: &SessionMetadata) -> anyhow::Result<String> {
        if let Some(principal) = &metadata.principal_id {
            return Ok(principal.clone());
        }
        let channel = metadata
            .channel_id
            .as_deref()
            .ok_or_else(|| anyhow::Error::msg("hindsight_source_principal_missing"))?;
        let (kind, alias) = channel
            .split_once('.')
            .ok_or_else(|| anyhow::Error::msg("hindsight_source_channel_invalid"))?;
        let sender = metadata
            .sender_id
            .as_deref()
            .ok_or_else(|| anyhow::Error::msg("hindsight_source_principal_missing"))?;
        let agent = metadata
            .agent_alias
            .as_deref()
            .ok_or_else(|| anyhow::Error::msg("hindsight_source_agent_missing"))?;
        zeroclaw_config::hindsight::channel_principal(
            &self.live.read(),
            agent,
            kind,
            alias,
            sender,
            false,
        )
    }
    fn snapshot(&self, stream: &Stream) -> anyhow::Result<Option<SessionMetadata>> {
        anyhow::ensure!(
            SqliteSessionBackend::db_path(&self.live.read().data_dir).try_exists()?,
            "hindsight_source_store_missing"
        );
        let Some(snapshot) = self.store.read_metadata(&stream.session)? else {
            return Ok(None);
        };
        anyhow::ensure!(
            snapshot.agent_alias.as_deref() == Some(&stream.agent)
                && self.principal(&snapshot)? == stream.principal,
            "hindsight_source_owner_changed"
        );
        let cfg = self.live.read();
        anyhow::ensure!(
            cfg.memory.backend == "hindsight"
                && cfg.agent(&stream.agent).is_some_and(|a| a.enabled
                    && a.memory.backend
                        == zeroclaw_config::multi_agent::MemoryBackendKind::Hindsight),
            "hindsight_source_agent_revoked"
        );
        Ok(Some(snapshot))
    }
    fn revision_now(&self) -> anyhow::Result<String> {
        let mut revision = Vec::new();
        for stream in self
            .ledger()?
            .streams()?
            .into_iter()
            .filter(|s| s.branch.starts_with("upstream:"))
        {
            let snapshot = self.snapshot(&stream)?;
            revision.push((stream.id, snapshot.map(|s| s.created_at.to_rfc3339())));
        }
        Ok(hash(&serde_json::to_vec(&revision)?))
    }
    /// Called only by the admitted chat/channel owner after a successful turn.
    /// This outbox is the delivery authority until acknowledgment, not a history API.
    #[cfg(test)]
    fn capture(
        &self,
        expected: &zeroclaw_api::memory_traits::PrincipalScope,
        session: &str,
        turn: &str,
        user: &str,
        assistant: &str,
    ) -> anyhow::Result<()> {
        let mut c = self
            .outbox
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_outbox_lock_poisoned"))?;
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let Some(snapshot) = self.store.read_metadata(session)? else {
            return Ok(());
        };
        self.capture_on(&tx, &snapshot, expected, session, turn, user, assistant)?;
        tx.commit()?;
        Ok(())
    }

    /// The completed transcript and extraction payload either both commit or roll back.
    #[allow(clippy::too_many_arguments)] // One admitted turn and its final session mutation.
    pub fn commit_turn(
        &self,
        expected: &zeroclaw_api::memory_traits::PrincipalScope,
        session: &str,
        turn: &str,
        user: &str,
        assistant: &str,
        write: zeroclaw_infra::session_sqlite::CompletedSessionWrite<'_>,
    ) -> anyhow::Result<bool> {
        self.store
            .commit_completed_with(session, write, |tx, snapshot| {
                self.capture_on(tx, snapshot, expected, session, turn, user, assistant)
            })
    }

    #[allow(clippy::too_many_arguments)] // Borrowed turn fields and the storage-owned transaction.
    fn capture_on(
        &self,
        tx: &Connection,
        snapshot: &SessionMetadata,
        expected: &zeroclaw_api::memory_traits::PrincipalScope,
        session: &str,
        turn: &str,
        user: &str,
        assistant: &str,
    ) -> anyhow::Result<bool> {
        let cfg = self.live.read();
        if cfg.memory.backend != "hindsight" || !cfg.memory.hindsight.auto_retain {
            return Ok(true);
        }
        cfg.validate_hindsight_memory()?;
        let capacity = cfg.memory.hindsight.queue_capacity;
        anyhow::ensure!(
            expected.namespace.is_none() && expected.tenant_id.is_none(),
            "hindsight_source_scope_unsupported"
        );
        drop(cfg);
        let agent = snapshot
            .agent_alias
            .as_deref()
            .ok_or_else(|| anyhow::Error::msg("hindsight_source_agent_missing"))?;
        let principal = self.principal(snapshot)?;
        anyhow::ensure!(
            expected.principal_id == principal && expected.agent_alias.as_deref() == Some(agent),
            "hindsight_source_owner_changed"
        );
        let branch = format!("upstream:{}", snapshot.created_at.to_rfc3339());
        let ledger = self.ledger()?;
        ledger.register_stream(agent, &principal, session, &branch, 0)?;
        let stream = ledger
            .streams()?
            .into_iter()
            .find(|s| {
                s.agent == agent
                    && s.principal == principal
                    && s.session == session
                    && s.branch == branch
            })
            .ok_or_else(|| anyhow::Error::msg("hindsight_source_missing"))?;
        let cfg = self.live.read();
        anyhow::ensure!(
            cfg.agent(agent).is_some_and(|a| a.enabled
                && a.memory.backend == zeroclaw_config::multi_agent::MemoryBackendKind::Hindsight),
            "hindsight_source_agent_revoked"
        );
        drop(cfg);
        let terminal = hash(&serde_json::to_vec(&(&stream.id, turn))?);
        let occurred = chrono::Utc::now();
        let names = zeroclaw_config::speakers::resolve(&self.live.read(), agent, &principal)?;
        let content = serde_json::json!([
            {"role":"user","speaker":names.user,"timestamp":occurred.to_rfc3339(),"content":user},
            {"role":"assistant","speaker":names.assistant,"timestamp":occurred.to_rfc3339(),"content":assistant}
        ]).to_string();
        anyhow::ensure!(content.len() <= 128 * 1024, "hindsight_turn_too_large");
        let item = Item {
            content,
            document_id: format!("zeroclaw:conversation:{}", stream.id),
            update_mode: Some("append".into()),
            timestamp: occurred.to_rfc3339(),
            context: format!(
                "{} Timestamps are capture times.",
                super::source::conversation_context(&names)
            ),
            metadata: BTreeMap::new(),
        };
        let fingerprint = hash(&serde_json::to_vec(&(user, assistant))?);
        let previous: Option<String> = tx
            .query_row(
                "SELECT fingerprint FROM hindsight_pending WHERE terminal=?1",
                [&terminal],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(previous) = previous {
            anyhow::ensure!(previous == fingerprint, "hindsight_turn_identity_conflict");
            return Ok(false);
        }
        let count: usize = tx.query_row(
            "SELECT count(*) FROM hindsight_pending WHERE body IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        anyhow::ensure!(count < capacity, "hindsight_outbox_full");
        let sequence: u64 = tx.query_row(
            "SELECT coalesce(max(sequence),0)+1 FROM hindsight_pending WHERE stream=?1",
            [&stream.id],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO hindsight_pending(stream,sequence,terminal,occurred,fingerprint,body) VALUES(?1,?2,?3,?4,?5,?6)",
            params![
                stream.id,
                sequence,
                terminal,
                occurred.timestamp_millis(),
                fingerprint,
                serde_json::to_string(&vec![item])?
            ],
        )?;
        Ok(true)
    }
    /// Release acknowledged payloads; identity receipts remain for reconciliation.
    pub fn release_delivered(&self) -> anyhow::Result<()> {
        let ledger = self.ledger()?;
        let mut c = self
            .outbox
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_outbox_lock_poisoned"))?;
        let tx = c.transaction()?;
        let rows = {
            let mut q =
                tx.prepare("SELECT stream,terminal FROM hindsight_pending WHERE body IS NOT NULL")?;
            q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                .collect::<Result<Vec<_>, _>>()?
        };
        for (stream, terminal) in rows {
            if ledger.source_delivery_complete(&stream, &terminal)? {
                tx.execute(
                    "UPDATE hindsight_pending SET body=NULL WHERE terminal=?1",
                    [terminal],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }
}

impl TranscriptAuthority for SessionTranscripts {
    fn admitted_scope(&self) -> anyhow::Result<zeroclaw_api::memory_traits::PrincipalScope> {
        TURN_PRINCIPAL
            .try_with(Clone::clone)
            .ok()
            .flatten()
            .ok_or_else(|| anyhow::Error::msg("hindsight_principal_scope_required"))
    }
    fn check_sources(&self, ledger: &Ledger, bank: &str) -> anyhow::Result<()> {
        if !ledger.has_transcript_sources(bank)? {
            return Ok(());
        }
        ledger.record_source_audit(bank, &self.revision_now()?)
    }
}
#[async_trait::async_trait]
impl TranscriptSource for SessionTranscripts {
    async fn revision(&self, _stream: &Stream) -> anyhow::Result<String> {
        self.revision_now()
    }
    async fn next(&self, stream: &Stream) -> anyhow::Result<Vec<SourceEvent>> {
        anyhow::ensure!(
            stream.branch.starts_with("upstream:"),
            "hindsight_source_adapter_mismatch"
        );
        let c = self
            .outbox
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_outbox_lock_poisoned"))?;
        let mut q=c.prepare("SELECT sequence,occurred,terminal FROM hindsight_pending WHERE stream=?1 AND sequence>?2 ORDER BY sequence LIMIT 256")?;
        Ok(q.query_map(params![stream.id, stream.cursor], |r| {
            Ok(SourceEvent {
                sequence: r.get(0)?,
                occurred_at: r.get(1)?,
                committed_turn: Some(r.get(2)?),
            })
        })?
        .collect::<Result<_, _>>()?)
    }
    async fn exists(&self, stream: &Stream, terminal: &str) -> anyhow::Result<bool> {
        let Some(snapshot) = self.snapshot(stream)? else {
            return Ok(false);
        };
        if stream.branch != format!("upstream:{}", snapshot.created_at.to_rfc3339()) {
            return Ok(false);
        }
        let c = self
            .outbox
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_outbox_lock_poisoned"))?;
        Ok(c.query_row(
            "SELECT EXISTS(SELECT 1 FROM hindsight_pending WHERE stream=?1 AND terminal=?2)",
            params![stream.id, terminal],
            |r| r.get(0),
        )?)
    }
    async fn load(&self, stream: &Stream, terminal: &str) -> anyhow::Result<Option<Vec<Item>>> {
        if !self.exists(stream, terminal).await? {
            return Ok(None);
        }
        let c = self
            .outbox
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_outbox_lock_poisoned"))?;
        let body: Option<String> = c.query_row(
            "SELECT body FROM hindsight_pending WHERE stream=?1 AND terminal=?2",
            params![stream.id, terminal],
            |r| r.get(0),
        )?;
        let body = body.ok_or_else(|| {
            anyhow::Error::msg("hindsight_acknowledged_source_requires_native_history_for_replay")
        })?;
        Ok(Some(serde_json::from_str(&body)?))
    }
}

/// Persist the successful final session mutation and its durable ingestion payload.
#[allow(clippy::too_many_arguments)] // The host supplies one admitted completed turn.
pub fn commit_completed_turn(
    live: &LiveConfigHandle,
    expected: &zeroclaw_api::memory_traits::PrincipalScope,
    session: &str,
    turn: &str,
    user: &str,
    assistant: &str,
    write: zeroclaw_infra::session_sqlite::CompletedSessionWrite<'_>,
) -> anyhow::Result<bool> {
    SessionTranscripts::new(live.clone())?
        .commit_turn(expected, session, turn, user, assistant, write)
}

pub struct Worker(tokio::task::AbortHandle);
impl Drop for Worker {
    fn drop(&mut self) {
        self.0.abort();
    }
}
pub fn start(live: LiveConfigHandle) -> Worker {
    let task = zeroclaw_spawn::spawn!(async move {
        loop {
            if live.read().memory.backend == "hindsight" {
                let lease = (|| -> anyhow::Result<std::fs::File> {
                    Ledger::open(&live.read().data_dir)?;
                    let file = std::fs::OpenOptions::new()
                        .create(true)
                        .truncate(false)
                        .read(true)
                        .write(true)
                        .open(live.read().data_dir.join("hindsight/ingestion.lock"))?;
                    file.try_lock()?;
                    Ok(file)
                })();
                match lease {
                    Ok(_lease) => {
                        let result = async {
                            let source = SessionTranscripts::new(live.clone())?;
                            super::delivery::tick(&live, &source).await?;
                            source.release_delivered()
                        }
                        .await;
                        if result.is_err() {
                            zeroclaw_log::record!(
                                WARN,
                                zeroclaw_log::Event::new(
                                    module_path!(),
                                    zeroclaw_log::Action::Note
                                ),
                                "hindsight_delivery_tick_failed"
                            );
                        }
                    }
                    Err(error) => {
                        if !matches!(
                            error.downcast_ref::<std::fs::TryLockError>(),
                            Some(std::fs::TryLockError::WouldBlock)
                        ) {
                            zeroclaw_log::record!(
                                WARN,
                                zeroclaw_log::Event::new(
                                    module_path!(),
                                    zeroclaw_log::Action::Note
                                ),
                                "hindsight_ingestion_lock_failed"
                            );
                        }
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    });
    Worker(task.abort_handle())
}

#[cfg(test)]
mod tests {
    use super::super::delivery::recovery_tests::{FaultApi, config};
    use super::*;
    use zeroclaw_api::{memory_traits::PrincipalScope, model_provider::ChatMessage};
    use zeroclaw_infra::session_backend::SessionBackend;
    fn scope() -> PrincipalScope {
        PrincipalScope::new("fixture-owner").with_agent(Some("fixture".into()))
    }
    fn session(source: &SessionTranscripts) {
        source
            .store
            .append("rpc_fixture", &ChatMessage::user("Original source"))
            .unwrap();
        source
            .store
            .set_session_agent_alias("rpc_fixture", "fixture")
            .unwrap();
        source
            .store
            .set_session_principal("rpc_fixture", "fixture-owner")
            .unwrap();
    }
    #[test]
    fn legacy_outbox_moves_pending_payloads_and_receipts_without_reviving_released_text() {
        let dir = tempfile::tempdir().unwrap();
        let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), "http://127.0.0.1:1"));
        let source = SessionTranscripts::new(live.handle()).unwrap();
        session(&source);
        source
            .capture(&scope(), "rpc_fixture", "legacy-turn", "original", "reply")
            .unwrap();
        let row = source
            .outbox
            .lock()
            .unwrap()
            .query_row(
                "SELECT stream,sequence,terminal,occurred,fingerprint,body FROM hindsight_pending",
                [],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, u64>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, String>(5)?,
                    ))
                },
            )
            .unwrap();
        let legacy = Connection::open(dir.path().join("hindsight/transcript-outbox.db")).unwrap();
        legacy.execute_batch("CREATE TABLE pending(stream TEXT,sequence INTEGER,terminal TEXT,occurred INTEGER,fingerprint TEXT,body TEXT)").unwrap();
        let insert = |legacy: &Connection| {
            legacy
                .execute(
                    "INSERT INTO pending VALUES(?1,?2,?3,?4,?5,?6)",
                    params![row.0, row.1, row.2, row.3, row.4, row.5],
                )
                .unwrap();
        };
        insert(&legacy);
        source
            .outbox
            .lock()
            .unwrap()
            .execute("DELETE FROM hindsight_pending", [])
            .unwrap();
        drop(source);
        let source = SessionTranscripts::new(live.handle()).unwrap();
        assert_eq!(
            source
                .outbox
                .lock()
                .unwrap()
                .query_row("SELECT body FROM hindsight_pending", [], |r| r
                    .get::<_, String>(0))
                .unwrap(),
            row.5
        );
        assert_eq!(
            legacy
                .query_row("SELECT count(*) FROM pending", [], |r| r.get::<_, u64>(0))
                .unwrap(),
            0
        );
        source
            .outbox
            .lock()
            .unwrap()
            .execute("UPDATE hindsight_pending SET body=NULL", [])
            .unwrap();
        // Simulate interruption after copying but before retiring the old rows.
        insert(&legacy);
        drop(source);
        let source = SessionTranscripts::new(live.handle()).unwrap();
        assert!(
            source
                .outbox
                .lock()
                .unwrap()
                .query_row("SELECT body FROM hindsight_pending", [], |r| r
                    .get::<_, Option<String>>(0))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn final_reply_and_outbox_commit_together_and_capacity_failure_rolls_both_back() {
        use zeroclaw_infra::session_sqlite::CompletedSessionWrite;
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(dir.path(), "http://127.0.0.1:1");
        cfg.memory.hindsight.queue_capacity = 1;
        let live = zeroclaw_config::live::LiveConfig::new(cfg);
        let source = SessionTranscripts::new(live.handle()).unwrap();
        session(&source);
        let reply = ChatMessage::assistant("accepted final reply");
        assert!(
            source
                .commit_turn(
                    &scope(),
                    "rpc_fixture",
                    "atomic-one",
                    "original",
                    "accepted final reply",
                    CompletedSessionWrite::Append(&reply)
                )
                .unwrap()
        );
        assert!(
            source
                .commit_turn(
                    &scope(),
                    "rpc_fixture",
                    "atomic-one",
                    "original",
                    "accepted final reply",
                    CompletedSessionWrite::Append(&reply)
                )
                .unwrap()
        );
        assert_eq!(
            source.store.load("rpc_fixture").len(),
            2,
            "retry must not duplicate the final reply"
        );
        let before = source.store.load("rpc_fixture");
        assert_eq!(before.last().unwrap().content, "accepted final reply");
        assert!(
            source
                .commit_turn(
                    &scope(),
                    "rpc_fixture",
                    "atomic-two",
                    "second",
                    "must roll back",
                    CompletedSessionWrite::Append(&ChatMessage::assistant("must roll back"))
                )
                .is_err()
        );
        assert_eq!(source.store.load("rpc_fixture").len(), before.len());
        assert_eq!(
            source.store.load("rpc_fixture").last().unwrap().content,
            "accepted final reply"
        );
        drop(source);
        let source = SessionTranscripts::new(live.handle()).unwrap();
        assert_eq!(
            source.store.load("rpc_fixture").last().unwrap().content,
            "accepted final reply"
        );
        assert_eq!(
            source
                .outbox
                .lock()
                .unwrap()
                .query_row("SELECT count(*) FROM hindsight_pending", [], |r| r
                    .get::<_, u64>(0))
                .unwrap(),
            1
        );
        source.store.delete_session("rpc_fixture").unwrap();
        assert!(
            !source
                .commit_turn(
                    &scope(),
                    "rpc_fixture",
                    "deleted",
                    "ignored",
                    "ignored",
                    CompletedSessionWrite::Append(&reply)
                )
                .unwrap()
        );
        assert!(source.store.load("rpc_fixture").is_empty());
    }

    #[tokio::test]
    async fn completed_turn_survives_restart_and_trimming_without_creating_event_history() {
        let dir = tempfile::tempdir().unwrap();
        let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), "http://127.0.0.1:1"));
        let source = SessionTranscripts::new(live.handle()).unwrap();
        session(&source);
        source
            .capture(
                &scope(),
                "rpc_fixture",
                "turn-one",
                "Original source",
                "Accepted answer",
            )
            .unwrap();
        source
            .capture(
                &scope(),
                "rpc_fixture",
                "turn-one",
                "Original source",
                "Accepted answer",
            )
            .unwrap();
        source
            .store
            .replace_conversation_state("rpc_fixture", &[ChatMessage::user("Later context")], true)
            .unwrap();
        drop(source);
        let source = SessionTranscripts::new(live.handle()).unwrap();
        let stream = source.ledger().unwrap().streams().unwrap().remove(0);
        let events = source.next(&stream).await.unwrap();
        assert_eq!(events.len(), 1);
        let body = source
            .load(&stream, events[0].committed_turn.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert!(
            body[0].context.starts_with(
                "This is your conversation history. You are fixture; User is the user."
            )
        );
        assert!(
            body[0]
                .context
                .contains("Keep User's actions and preferences attributed to User.")
        );
        assert!(body[0].context.ends_with("Timestamps are capture times."));
        assert!(body[0].content.contains("Original source"));
        assert!(!body[0].content.contains("Later context"));
        assert!(!dir.path().join("sessions/events.db").exists());
        assert!(!dir.path().join("brain.db").exists());
    }
    #[tokio::test]
    async fn owner_change_and_unattributed_sessions_never_authorize_capture_or_disclosure() {
        let dir = tempfile::tempdir().unwrap();
        let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), "http://127.0.0.1:1"));
        let source = SessionTranscripts::new(live.handle()).unwrap();
        source
            .store
            .append("rpc_fixture", &ChatMessage::user("source"))
            .unwrap();
        assert!(
            source
                .capture(&scope(), "rpc_fixture", "one", "source", "reply")
                .is_err()
        );
        session(&source);
        let other = PrincipalScope::new("another-owner").with_agent(Some("fixture".into()));
        assert!(
            source
                .capture(&other, "rpc_fixture", "one", "source", "reply")
                .is_err()
        );
        source
            .capture(&scope(), "rpc_fixture", "one", "source", "reply")
            .unwrap();
        assert!(
            source
                .capture(&scope(), "rpc_fixture", "one", "different source", "reply")
                .unwrap_err()
                .to_string()
                .contains("identity_conflict")
        );
        assert!(
            source
                .capture(
                    &scope().with_tenant(Some("other".into())),
                    "rpc_fixture",
                    "two",
                    "source",
                    "reply"
                )
                .is_err()
        );
        let stream = source.ledger().unwrap().streams().unwrap().remove(0);
        let terminal = source
            .next(&stream)
            .await
            .unwrap()
            .remove(0)
            .committed_turn
            .unwrap();
        source
            .store
            .set_session_principal("rpc_fixture", "another-owner")
            .unwrap();
        assert!(source.load(&stream, &terminal).await.is_err());
        std::fs::rename(
            SqliteSessionBackend::db_path(dir.path()),
            dir.path().join("moved.db"),
        )
        .unwrap();
        assert!(source.exists(&stream, &terminal).await.is_err());
        assert!(SessionTranscripts::new(live.handle()).is_err());
        assert!(source.admitted_scope().is_err());
    }
    #[test]
    fn full_outbox_reports_failure_and_keeps_already_accepted_turn() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(dir.path(), "http://127.0.0.1:1");
        cfg.memory.hindsight.queue_capacity = 1;
        let live = zeroclaw_config::live::LiveConfig::new(cfg);
        let source = SessionTranscripts::new(live.handle()).unwrap();
        session(&source);
        source
            .capture(&scope(), "rpc_fixture", "one", "first", "reply")
            .unwrap();
        assert!(
            source
                .capture(&scope(), "rpc_fixture", "two", "second", "reply")
                .unwrap_err()
                .to_string()
                .contains("outbox_full")
        );
        source
            .capture(&scope(), "rpc_fixture", "one", "first", "reply")
            .unwrap();
    }
    #[tokio::test]
    async fn delivery_ack_releases_payload_and_session_deletion_erases_remote_document() {
        let api = FaultApi::start();
        api.state.lock().unwrap().status = "completed".into();
        let dir = tempfile::tempdir().unwrap();
        let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &api.url));
        let source = SessionTranscripts::new(live.handle()).unwrap();
        session(&source);
        source
            .capture(
                &scope(),
                "rpc_fixture",
                "one",
                "Fictional Larch uses SQLite.",
                "Understood.",
            )
            .unwrap();
        super::super::delivery::tick(&live.handle(), &source)
            .await
            .unwrap();
        source.ledger().unwrap().retry_pending().unwrap();
        super::super::delivery::tick(&live.handle(), &source)
            .await
            .unwrap();
        source.release_delivered().unwrap();
        assert_eq!(api.state.lock().unwrap().retain_posts, 1);
        let stream = source.ledger().unwrap().streams().unwrap().remove(0);
        let terminal: String = source
            .outbox
            .lock()
            .unwrap()
            .query_row("SELECT terminal FROM hindsight_pending", [], |r| r.get(0))
            .unwrap();
        assert!(source.exists(&stream, &terminal).await.unwrap());
        assert!(
            source
                .load(&stream, &terminal)
                .await
                .unwrap_err()
                .to_string()
                .contains("requires_native_history")
        );
        source.store.delete_session("rpc_fixture").unwrap();
        assert!(!source.exists(&stream, &terminal).await.unwrap());
        for _ in 0..3 {
            super::super::delivery::tick(&live.handle(), &source)
                .await
                .unwrap();
        }
        assert!(api.state.lock().unwrap().deletes > 0);
        assert_eq!(api.state.lock().unwrap().retain_posts, 1);
    }
}
