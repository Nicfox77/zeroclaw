//! Delivery identities and state only. No memory text, vectors, or transcript copies.
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Mutex};
use zeroclaw_api::memory_traits::PrincipalScope;

#[derive(Clone, Debug)]
pub struct Stream {
    pub id: String,
    pub agent: String,
    pub principal: String,
    pub session: String,
    pub branch: String,
    pub cursor: u64,
}
#[derive(Clone, Debug)]
pub struct Delivery {
    pub id: String,
    pub stream: String,
    pub terminal: String,
    pub operation: Option<String>,
    pub status: String,
}

#[derive(Clone, Debug)]
pub struct Backfill {
    pub id: String,
    pub stream: String,
    pub cursor: u64,
    pub through: u64,
    pub since: Option<i64>,
    pub until: Option<i64>,
}

pub struct Note {
    pub operation: String,
    pub agent: String,
    pub scope: PrincipalScope,
    pub items: Vec<super::client::Item>,
}

pub struct Ledger {
    connection: Mutex<Connection>,
}
impl Ledger {
    pub fn open(data: &Path) -> anyhow::Result<Self> {
        let directory = data.join("hindsight");
        std::fs::create_dir_all(&directory)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))?;
        }
        let mut connection = Connection::open(directory.join("delivery.db"))?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        connection.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS source_audits(bank_id TEXT PRIMARY KEY,revision TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS activity(operation TEXT PRIMARY KEY,last_success TEXT,last_error TEXT,last_error_at TEXT,duration_ms INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS agents(alias TEXT PRIMARY KEY,identity TEXT NOT NULL UNIQUE);
            CREATE TABLE IF NOT EXISTS banks(scope_key TEXT PRIMARY KEY,agent_identity TEXT NOT NULL,bank_id TEXT NOT NULL UNIQUE);
            CREATE TABLE IF NOT EXISTS streams(identity TEXT PRIMARY KEY,agent TEXT NOT NULL,principal TEXT NOT NULL,session TEXT NOT NULL,branch TEXT NOT NULL,cursor INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS backfills(identity TEXT PRIMARY KEY,stream TEXT NOT NULL,cursor INTEGER NOT NULL DEFAULT 0,through INTEGER NOT NULL,since_millis INTEGER,until_millis INTEGER,status TEXT NOT NULL DEFAULT 'pending');
            CREATE TABLE IF NOT EXISTS deliveries(identity TEXT PRIMARY KEY,stream TEXT NOT NULL,terminal TEXT NOT NULL,sequence INTEGER NOT NULL,operation TEXT,status TEXT NOT NULL DEFAULT 'pending',attempts INTEGER NOT NULL DEFAULT 0,next_attempt INTEGER NOT NULL DEFAULT 0,error_key TEXT,UNIQUE(stream,terminal));
            CREATE TABLE IF NOT EXISTS notes(operation TEXT PRIMARY KEY,agent TEXT NOT NULL,scope TEXT NOT NULL,items TEXT NOT NULL,next_attempt INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS operations(identity TEXT PRIMARY KEY,bank_id TEXT NOT NULL,document_id TEXT NOT NULL,operation_id TEXT NOT NULL UNIQUE,status TEXT NOT NULL,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);")?;
        let schema =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let has_revision = {
            let mut q = schema.prepare("PRAGMA table_info(deliveries)")?;
            q.query_map([], |r| r.get::<_, String>(1))?
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .any(|name| name == "checked_revision")
        };
        if !has_revision {
            schema.execute(
                "ALTER TABLE deliveries ADD COLUMN checked_revision TEXT",
                [],
            )?;
        }
        schema.execute_batch("CREATE INDEX IF NOT EXISTS delivery_active ON deliveries(status,next_attempt,sequence);
            CREATE INDEX IF NOT EXISTS delivery_stream ON deliveries(stream,sequence);
            CREATE INDEX IF NOT EXISTS delivery_operation ON deliveries(operation,checked_revision,status);
            CREATE INDEX IF NOT EXISTS operation_bank ON operations(bank_id,operation_id);
            CREATE INDEX IF NOT EXISTS operation_document ON operations(bank_id,document_id);
            CREATE INDEX IF NOT EXISTS operation_completed ON operations(status,updated_at);")?;
        schema.commit()?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }
    /// Observed delivery/transport facts only; no prompt, result, or credential text.
    pub fn observe(
        &self,
        operation: &str,
        error: Option<&str>,
        duration_ms: u64,
    ) -> anyhow::Result<()> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        c.execute("INSERT INTO activity(operation,last_success,last_error,last_error_at,duration_ms) VALUES(?1,CASE WHEN ?2 IS NULL THEN CURRENT_TIMESTAMP ELSE NULL END,?2,CASE WHEN ?2 IS NOT NULL THEN CURRENT_TIMESTAMP ELSE NULL END,?3) ON CONFLICT(operation) DO UPDATE SET last_success=CASE WHEN ?2 IS NULL THEN CURRENT_TIMESTAMP ELSE last_success END,last_error=?2,last_error_at=CASE WHEN ?2 IS NOT NULL THEN CURRENT_TIMESTAMP ELSE last_error_at END,duration_ms=?3",params![operation,error,duration_ms])?;
        Ok(())
    }
    pub fn diagnostics(&self) -> anyhow::Result<serde_json::Value> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let mut q=c.prepare("SELECT operation,last_success,last_error,last_error_at,duration_ms FROM activity ORDER BY operation")?;
        let activity=q.query_map([],|r|Ok(serde_json::json!({"operation":r.get::<_,String>(0)?,"last_success":r.get::<_,Option<String>>(1)?,"last_error":r.get::<_,Option<String>>(2)?,"last_error_at":r.get::<_,Option<String>>(3)?,"duration_ms":r.get::<_,u64>(4)?})))?.collect::<Result<Vec<_>,_>>()?;
        let last_retain: Option<String> = c.query_row(
            "SELECT max(updated_at) FROM operations WHERE status='completed'",
            [],
            |r| r.get(0),
        )?;
        let backfills: usize = c.query_row(
            "SELECT count(*) FROM backfills WHERE status='pending'",
            [],
            |r| r.get(0),
        )?;
        let mut q=c.prepare("SELECT DISTINCT error_key FROM deliveries WHERE error_key IS NOT NULL ORDER BY error_key LIMIT 32")?;
        let errors = q
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        let mut q = c.prepare("SELECT bank_id FROM banks ORDER BY bank_id LIMIT 100")?;
        let banks = q
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(
            serde_json::json!({"operations":activity,"last_successful_retain":last_retain,"pending_backfills":backfills,"delivery_errors":errors,"banks":banks}),
        )
    }
    pub fn register_stream(
        &self,
        agent: &str,
        principal: &str,
        session: &str,
        branch: &str,
        cursor: u64,
    ) -> anyhow::Result<()> {
        let id = hash(&serde_json::to_vec(&(agent, principal, session, branch))?);
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        c.execute("INSERT OR IGNORE INTO streams(identity,agent,principal,session,branch,cursor) VALUES(?1,?2,?3,?4,?5,?6)",params![id,agent,principal,session,branch,cursor])?;
        Ok(())
    }
    pub fn streams(&self) -> anyhow::Result<Vec<Stream>> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let mut q = c.prepare(
            "SELECT identity,agent,principal,session,branch,cursor FROM streams ORDER BY identity",
        )?;
        Ok(q.query_map([], |r| {
            Ok(Stream {
                id: r.get(0)?,
                agent: r.get(1)?,
                principal: r.get(2)?,
                session: r.get(3)?,
                branch: r.get(4)?,
                cursor: r.get(5)?,
            })
        })?
        .collect::<Result<_, _>>()?)
    }
    /// The cursor advances atomically with materialized references. At capacity it
    /// stays behind the canonical stream; one cursor represents any size backlog.
    pub fn advance(
        &self,
        stream: &Stream,
        through: u64,
        terminal: Option<&str>,
        capacity: usize,
    ) -> anyhow::Result<bool> {
        let mut c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let tx = c.transaction()?;
        if let Some(terminal) = terminal {
            let id = hash(&serde_json::to_vec(&(&stream.id, terminal))?);
            let exists: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM deliveries WHERE identity=?1)",
                [&id],
                |r| r.get(0),
            )?;
            if !exists {
                let count: usize = tx.query_row(
                    "SELECT (SELECT count(*) FROM deliveries WHERE status IN ('pending','processing','failed','erasing'))+(SELECT count(*) FROM notes)",
                    [],
                    |r| r.get(0),
                )?;
                if count >= capacity {
                    return Ok(false);
                }
                tx.execute(
                    "INSERT INTO deliveries(identity,stream,terminal,sequence) VALUES(?1,?2,?3,?4)",
                    params![id, stream.id, terminal, through],
                )?;
            }
        }
        tx.execute(
            "UPDATE streams SET cursor=max(cursor,?1) WHERE identity=?2",
            params![through, stream.id],
        )?;
        tx.commit()?;
        Ok(true)
    }
    pub fn pending(&self, limit: usize, retry_failed: bool) -> anyhow::Result<Vec<Delivery>> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let mut q = c.prepare("SELECT identity,stream,terminal,operation,status FROM deliveries WHERE (status IN ('pending','processing') OR (status='failed' AND ?1)) AND next_attempt<=unixepoch() ORDER BY sequence,identity LIMIT ?2")?;
        Ok(q.query_map(params![retry_failed, limit], |r| {
            Ok(Delivery {
                id: r.get(0)?,
                stream: r.get(1)?,
                terminal: r.get(2)?,
                operation: r.get(3)?,
                status: r.get(4)?,
            })
        })?
        .collect::<Result<_, _>>()?)
    }
    pub fn delivery_state(
        &self,
        id: &str,
        status: &str,
        operation: Option<&str>,
        error_key: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let tx = c.transaction()?;
        tx.execute("DELETE FROM source_audits WHERE bank_id IN (SELECT o.bank_id FROM operations o JOIN deliveries d ON o.operation_id=coalesce(?1,d.operation) WHERE d.identity=?2 AND (?3='erasing' OR d.operation IS NULL OR d.operation!=?1))",params![operation,id,status])?;
        tx.execute("UPDATE deliveries SET status=?1,operation=coalesce(?2,operation),error_key=?3,attempts=attempts+CASE WHEN ?3 IS NULL THEN 0 ELSE 1 END,next_attempt=unixepoch()+CASE WHEN ?3 IS NULL THEN 1 ELSE min(300,5*(1 << min(attempts,6))) END WHERE identity=?4",params![status,operation,error_key,id])?;
        tx.commit()?;
        Ok(())
    }
    pub fn backfill(
        &self,
        stream: &str,
        through: u64,
        since: Option<i64>,
        until: Option<i64>,
    ) -> anyhow::Result<String> {
        let id = hash(&serde_json::to_vec(&(stream, through, since, until))?);
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        c.execute("INSERT OR IGNORE INTO backfills(identity,stream,through,since_millis,until_millis) VALUES(?1,?2,?3,?4,?5)",params![id,stream,through,since,until])?;
        Ok(id)
    }
    pub fn backfills(&self) -> anyhow::Result<Vec<Backfill>> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let mut q=c.prepare("SELECT identity,stream,cursor,through,since_millis,until_millis FROM backfills WHERE status='pending' ORDER BY identity LIMIT 32")?;
        Ok(q.query_map([], |r| {
            Ok(Backfill {
                id: r.get(0)?,
                stream: r.get(1)?,
                cursor: r.get(2)?,
                through: r.get(3)?,
                since: r.get(4)?,
                until: r.get(5)?,
            })
        })?
        .collect::<Result<_, _>>()?)
    }
    pub fn advance_backfill(
        &self,
        job: &Backfill,
        stream: &Stream,
        through: u64,
        terminal: Option<&str>,
        capacity: usize,
    ) -> anyhow::Result<bool> {
        let mut c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let tx = c.transaction()?;
        if let Some(terminal) = terminal {
            let id = hash(&serde_json::to_vec(&(&stream.id, terminal))?);
            let exists: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM deliveries WHERE identity=?1)",
                [&id],
                |r| r.get(0),
            )?;
            if !exists {
                let count: usize = tx.query_row(
                    "SELECT (SELECT count(*) FROM deliveries WHERE status IN ('pending','processing','failed','erasing'))+(SELECT count(*) FROM notes)",
                    [],
                    |r| r.get(0),
                )?;
                if count >= capacity {
                    return Ok(false);
                }
                tx.execute(
                    "INSERT INTO deliveries(identity,stream,terminal,sequence) VALUES(?1,?2,?3,?4)",
                    params![id, stream.id, terminal, through],
                )?;
            }
        }
        tx.execute("UPDATE backfills SET cursor=max(cursor,?1),status=CASE WHEN ?1>=through THEN 'completed' ELSE 'pending' END WHERE identity=?2",params![through,job.id])?;
        tx.commit()?;
        Ok(true)
    }
    pub fn revision_candidates(
        &self,
        stream: &str,
        revision: &str,
    ) -> anyhow::Result<Vec<Delivery>> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let mut q=c.prepare("SELECT identity,stream,terminal,operation,status FROM deliveries WHERE stream=?1 AND status!='erased' AND (checked_revision IS NULL OR checked_revision!=?2 OR status='erasing') ORDER BY sequence LIMIT 64")?;
        Ok(q.query_map(params![stream, revision], |r| {
            Ok(Delivery {
                id: r.get(0)?,
                stream: r.get(1)?,
                terminal: r.get(2)?,
                operation: r.get(3)?,
                status: r.get(4)?,
            })
        })?
        .collect::<Result<_, _>>()?)
    }
    pub fn checked_revision(&self, id: &str, revision: &str) -> anyhow::Result<()> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        c.execute(
            "UPDATE deliveries SET checked_revision=?1 WHERE identity=?2",
            params![revision, id],
        )?;
        Ok(())
    }
    pub fn operation_source(&self, id: &str) -> anyhow::Result<(String, String)> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        Ok(c.query_row(
            "SELECT bank_id,document_id FROM operations WHERE operation_id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?)
    }
    /// A source-audit receipt records a completed validation decision for a
    /// native erasure frontier. It contains no cached permissions: callers
    /// still resolve live policy and the canonical native frontier every time.
    pub fn source_audited(&self, bank: &str, revision: &str) -> anyhow::Result<bool> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        Ok(c.query_row(
            "SELECT EXISTS(SELECT 1 FROM source_audits WHERE bank_id=?1 AND revision=?2)",
            params![bank, revision],
            |r| r.get(0),
        )?)
    }
    pub fn record_source_audit(&self, bank: &str, revision: &str) -> anyhow::Result<()> {
        let mut c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        // The proof and receipt are one transaction: a concurrently attached
        // source must not invalidate the proof before a stale receipt is saved.
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let blocked: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM deliveries d JOIN operations o ON d.operation=o.operation_id WHERE o.bank_id=?1 AND d.status!='erased' AND (d.status='erasing' OR d.checked_revision IS NULL OR d.checked_revision!=?2))",params![bank,revision],|r|r.get(0))?;
        anyhow::ensure!(!blocked, "hindsight_source_erasure_reconciliation_pending");
        tx.execute("INSERT INTO source_audits(bank_id,revision) VALUES(?1,?2) ON CONFLICT(bank_id) DO UPDATE SET revision=excluded.revision",params![bank,revision])?;
        tx.commit()?;
        Ok(())
    }
    pub fn has_transcript_sources(&self, bank: &str) -> anyhow::Result<bool> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        Ok(c.query_row("SELECT EXISTS(SELECT 1 FROM deliveries d JOIN operations o ON d.operation=o.operation_id WHERE o.bank_id=?1 AND d.status!='erased')",[bank],|r|r.get(0))?)
    }
    /// Standalone explicit notes have no canonical chat record to reference.
    /// Their screened delivery body is spooled only until upstream completion;
    /// it is never searched or used as a substitute memory backend.
    pub fn queue_note(
        &self,
        bank: &str,
        agent: &str,
        scope: &PrincipalScope,
        items: &[super::client::Item],
        capacity: usize,
    ) -> anyhow::Result<String> {
        anyhow::ensure!(items.len() == 1, "hindsight_note_requires_one_document");
        let document = &items[0].document_id;
        let fingerprint = hash(&serde_json::to_vec(&(
            &items[0].content,
            &items[0].context,
            &items[0].metadata,
        ))?);
        let prefix = format!("note:{fingerprint}:");
        let mut c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        // Operation rows create the canonical ordering of accepted note intents.
        // Fingerprints are delivery metadata, never a second memory authority.
        let previous: Option<(String,String,String)> = tx.query_row(
            "SELECT identity,operation_id,status FROM operations WHERE bank_id=?1 AND document_id=?2 ORDER BY rowid DESC LIMIT 1",params![bank,document],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))
        ).optional()?;
        let operation = match previous {
            Some((identity, operation, status))
                if identity.starts_with(&prefix) && status != "completed" =>
            {
                operation
            }
            _ => {
                let operation = uuid::Uuid::new_v4().to_string();
                tx.execute("INSERT INTO operations(identity,bank_id,document_id,operation_id,status) VALUES(?1,?2,?3,?4,'pending')",params![format!("{prefix}{operation}"),bank,document,operation])?;
                operation
            }
        };
        let exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM notes WHERE operation=?1)",
            [&operation],
            |r| r.get(0),
        )?;
        if !exists {
            let count:usize=tx.query_row("SELECT (SELECT count(*) FROM notes)+(SELECT count(*) FROM deliveries WHERE status IN ('pending','processing','failed','erasing'))",[],|r|r.get(0))?;
            anyhow::ensure!(count < capacity, "hindsight_delivery_queue_full");
            tx.execute(
                "INSERT INTO notes(operation,agent,scope,items) VALUES(?1,?2,?3,?4)",
                params![
                    operation,
                    agent,
                    serde_json::to_string(scope)?,
                    serde_json::to_string(items)?
                ],
            )?;
        }
        tx.commit()?;
        Ok(operation)
    }
    pub fn note_is_next(&self, operation: &str) -> anyhow::Result<bool> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        Ok(c.query_row("SELECT NOT EXISTS(SELECT 1 FROM notes n JOIN operations older ON n.operation=older.operation_id JOIN operations current ON current.operation_id=?1 WHERE older.bank_id=current.bank_id AND older.document_id=current.document_id AND older.rowid<current.rowid)",[operation],|r|r.get(0))?)
    }
    pub fn note(&self, operation: &str) -> anyhow::Result<Option<Note>> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let row = c
            .query_row(
                "SELECT agent,scope,items FROM notes WHERE operation=?1",
                [operation],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?;
        row.map(|(agent, scope, items)| {
            Ok(Note {
                operation: operation.into(),
                agent,
                scope: serde_json::from_str(&scope)?,
                items: serde_json::from_str(&items)?,
            })
        })
        .transpose()
    }
    pub fn pending_notes(&self, retry_failed: bool) -> anyhow::Result<Vec<Note>> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let mut q=c.prepare("SELECT n.operation,n.agent,n.scope,n.items FROM notes n JOIN operations o ON n.operation=o.operation_id WHERE n.next_attempt<=unixepoch() AND (o.status!='failed' OR ?1) AND NOT EXISTS(SELECT 1 FROM notes p JOIN operations older ON p.operation=older.operation_id WHERE older.bank_id=o.bank_id AND older.document_id=o.document_id AND older.rowid<o.rowid) ORDER BY o.rowid LIMIT 16")?;
        let rows = q
            .query_map([retry_failed], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        rows.into_iter()
            .map(|(operation, agent, scope, items)| {
                Ok(Note {
                    operation,
                    agent,
                    scope: serde_json::from_str(&scope)?,
                    items: serde_json::from_str(&items)?,
                })
            })
            .collect()
    }
    pub fn complete_note(&self, operation: &str) -> anyhow::Result<()> {
        let mut c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let tx = c.transaction()?;
        tx.execute("UPDATE operations SET status='completed',updated_at=CURRENT_TIMESTAMP WHERE operation_id=?1",[operation])?;
        tx.execute("DELETE FROM notes WHERE operation=?1", [operation])?;
        tx.commit()?;
        Ok(())
    }
    pub fn defer_note(&self, operation: &str, failed: bool) -> anyhow::Result<()> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        c.execute(
            "UPDATE notes SET next_attempt=unixepoch()+30 WHERE operation=?1",
            [operation],
        )?;
        if failed {
            c.execute("UPDATE operations SET status='failed',updated_at=CURRENT_TIMESTAMP WHERE operation_id=?1",[operation])?;
        }
        Ok(())
    }
    pub fn retry_pending(&self) -> anyhow::Result<usize> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let notes = c.execute("UPDATE notes SET next_attempt=0", [])?;
        c.execute("UPDATE operations SET status='pending' WHERE operation_id IN (SELECT operation FROM notes)",[])?;
        Ok(notes+c.execute("UPDATE deliveries SET status='pending',next_attempt=0,error_key=NULL WHERE status IN ('pending','processing','failed')",[])?)
    }
    pub fn delivery_counts(&self) -> anyhow::Result<std::collections::BTreeMap<String, usize>> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let mut q = c.prepare("SELECT status,count(*) FROM (SELECT status FROM deliveries UNION ALL SELECT o.status FROM notes n JOIN operations o ON n.operation=o.operation_id) GROUP BY status")?;
        Ok(q.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?)
    }
    pub fn agent(&self, alias: &str) -> anyhow::Result<String> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        c.execute(
            "INSERT OR IGNORE INTO agents(alias,identity) VALUES(?1,?2)",
            params![alias, uuid::Uuid::new_v4().to_string()],
        )?;
        Ok(
            c.query_row("SELECT identity FROM agents WHERE alias=?1", [alias], |r| {
                r.get(0)
            })?,
        )
    }
    pub fn bank(
        &self,
        agent_identity: &str,
        alias: &str,
        scope: &PrincipalScope,
        template: &str,
    ) -> anyhow::Result<String> {
        anyhow::ensure!(
            !scope.principal_id.trim().is_empty(),
            "hindsight_principal_required"
        );
        anyhow::ensure!(
            scope
                .namespace
                .as_deref()
                .is_none_or(|n| !n.trim().is_empty())
                && scope
                    .tenant_id
                    .as_deref()
                    .is_none_or(|n| !n.trim().is_empty()),
            "hindsight_invalid_scope_dimension"
        );
        let scope_key = hash(&serde_json::to_vec(&(
            agent_identity,
            &scope.principal_id,
            scope.namespace.as_deref().unwrap_or("default"),
            &scope.tenant_id,
        ))?);
        let slug: String = alias
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .take(24)
            .collect();
        let bank = format!("{}-{}", template.replace("{agent}", &slug), scope_key);
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        c.execute(
            "INSERT OR IGNORE INTO banks(scope_key,agent_identity,bank_id) VALUES(?1,?2,?3)",
            params![scope_key, agent_identity, bank],
        )?;
        Ok(c.query_row(
            "SELECT bank_id FROM banks WHERE scope_key=?1",
            [scope_key],
            |r| r.get(0),
        )?)
    }
    /// UUID is durably allocated before network submission. Repeated source identity reuses it.
    pub fn operation(&self, bank: &str, document: &str, digest: &str) -> anyhow::Result<String> {
        let key = hash(&serde_json::to_vec(&(bank, document, digest))?);
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        c.execute("INSERT OR IGNORE INTO operations(identity,bank_id,document_id,operation_id,status) VALUES(?1,?2,?3,?4,'pending')",params![key,bank,document,uuid::Uuid::new_v4().to_string()])?;
        Ok(c.query_row(
            "SELECT operation_id FROM operations WHERE identity=?1",
            [key],
            |r| r.get(0),
        )?)
    }
    pub fn update(&self, operation: &str, status: &str) -> anyhow::Result<()> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        let changed = c.execute(
            "UPDATE operations SET status=?1,updated_at=CURRENT_TIMESTAMP WHERE operation_id=?2",
            params![status, operation],
        )?;
        anyhow::ensure!(changed == 1, "hindsight_unknown_delivery_operation");
        Ok(())
    }
    pub fn status(&self, operation: &str) -> anyhow::Result<Option<String>> {
        let c = self
            .connection
            .lock()
            .map_err(|_| anyhow::Error::msg("hindsight_delivery_lock_poisoned"))?;
        Ok(c.query_row(
            "SELECT status FROM operations WHERE operation_id=?1",
            [operation],
            |r| r.get(0),
        )
        .optional()?)
    }
}
pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scope_binding_and_idempotency_survive_reopen_without_storing_content() {
        let d = tempfile::tempdir().unwrap();
        let l = Ledger::open(d.path()).unwrap();
        let agent = l.agent("fixture").unwrap();
        let a = l
            .bank(
                &agent,
                "fixture",
                &PrincipalScope::new("owner-a"),
                "test-{agent}",
            )
            .unwrap();
        let b = l
            .bank(
                &agent,
                "fixture",
                &PrincipalScope::new("owner-b"),
                "test-{agent}",
            )
            .unwrap();
        assert_ne!(a, b);
        let op = l.operation(&a, "source-turn", "hash-only").unwrap();
        drop(l);
        let l = Ledger::open(d.path()).unwrap();
        assert_eq!(agent, l.agent("fixture").unwrap());
        assert_eq!(op, l.operation(&a, "source-turn", "hash-only").unwrap());
        l.update(&op, "processing").unwrap();
        assert_eq!(l.status(&op).unwrap().as_deref(), Some("processing"));
    }
}

#[cfg(test)]
mod delivery_tests {
    use super::*;
    #[test]
    fn bounded_queue_preserves_unmaterialized_tail_and_replays_idempotently() {
        let d = tempfile::tempdir().unwrap();
        let l = Ledger::open(d.path()).unwrap();
        l.register_stream("agent", "owner", "conversation", "branch", 100)
            .unwrap();
        let s = l.streams().unwrap().remove(0);
        assert!(l.advance(&s, 101, Some("turn-one"), 1).unwrap());
        assert!(!l.advance(&s, 102, Some("turn-two"), 1).unwrap());
        assert_eq!(l.streams().unwrap()[0].cursor, 101);
        let pending = l.pending(10, true).unwrap();
        assert_eq!(pending.len(), 1);
        drop(l);
        let l = Ledger::open(d.path()).unwrap();
        assert!(l.advance(&s, 101, Some("turn-one"), 1).unwrap());
        assert_eq!(l.pending(10, true).unwrap().len(), 1);
        l.delivery_state(&pending[0].id, "completed", None, None)
            .unwrap();
        assert!(l.advance(&s, 102, Some("turn-two"), 1).unwrap());
        assert_eq!(l.streams().unwrap()[0].cursor, 102);
        assert_eq!(l.pending(10, true).unwrap().len(), 1);
    }
    #[test]
    fn explicit_backfill_resumes_without_rewinding_live_capture_or_duplicating_turns() {
        let d = tempfile::tempdir().unwrap();
        let l = Ledger::open(d.path()).unwrap();
        l.register_stream("agent", "owner", "conversation", "branch", 100)
            .unwrap();
        let s = l.streams().unwrap().remove(0);
        l.backfill(&s.id, 99, None, None).unwrap();
        let job = l.backfills().unwrap().remove(0);
        assert!(
            l.advance_backfill(&job, &s, 50, Some("old-turn"), 10)
                .unwrap()
        );
        assert_eq!(l.streams().unwrap()[0].cursor, 100);
        drop(l);
        let l = Ledger::open(d.path()).unwrap();
        let job = l.backfills().unwrap().remove(0);
        assert_eq!(job.cursor, 50);
        l.advance_backfill(&job, &s, 50, Some("old-turn"), 10)
            .unwrap();
        l.advance_backfill(&job, &s, 99, None, 10).unwrap();
        assert!(l.backfills().unwrap().is_empty());
        assert_eq!(l.pending(10, true).unwrap().len(), 1);
        l.register_stream("agent", "owner", "conversation", "branch", 0)
            .unwrap();
        assert_eq!(l.streams().unwrap()[0].cursor, 100);
    }
    #[test]
    fn implicit_and_explicit_default_namespace_share_one_bank() {
        let d = tempfile::tempdir().unwrap();
        let l = Ledger::open(d.path()).unwrap();
        let id = l.agent("a").unwrap();
        let scope = PrincipalScope::new("owner");
        assert_eq!(
            l.bank(&id, "a", &scope, "z-{agent}").unwrap(),
            l.bank(
                &id,
                "a",
                &scope.with_namespace(Some("default".into())),
                "z-{agent}"
            )
            .unwrap()
        );
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    #[test]
    fn standalone_note_body_survives_uncertain_delivery_and_is_removed_on_completion() {
        let d = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(d.path()).unwrap();
        let scope = PrincipalScope::new("owner").with_agent(Some("fixture".into()));
        let item = super::super::client::Item {
            content: "screened explicit note".into(),
            document_id: "document".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            context: "explicit note".into(),
            metadata: Default::default(),
        };
        let operation = ledger
            .queue_note("bank", "fixture", &scope, &[item], 1)
            .unwrap();
        drop(ledger);
        let ledger = Ledger::open(d.path()).unwrap();
        let note = ledger.note(&operation).unwrap().unwrap();
        assert_eq!(note.operation, operation);
        assert_eq!(note.items[0].content, "screened explicit note");
        assert_eq!(ledger.delivery_counts().unwrap()["pending"], 1);
        ledger.complete_note(&operation).unwrap();
        assert!(ledger.note(&operation).unwrap().is_none());
        assert_eq!(
            ledger.status(&operation).unwrap().as_deref(),
            Some("completed")
        );
    }
    #[test]
    fn keyed_note_versions_are_ordered_and_reverting_content_creates_a_new_intent() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(directory.path()).unwrap();
        let scope = PrincipalScope::new("owner").with_agent(Some("fixture".into()));
        let mut item = super::super::client::Item {
            content: "first value".into(),
            document_id: "note-document".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            context: "explicit note".into(),
            metadata: Default::default(),
        };
        let first = ledger
            .queue_note("bank", "fixture", &scope, &[item.clone()], 8)
            .unwrap();
        assert_eq!(
            first,
            ledger
                .queue_note("bank", "fixture", &scope, &[item.clone()], 8)
                .unwrap()
        );
        item.content = "second value".into();
        let second = ledger
            .queue_note("bank", "fixture", &scope, &[item.clone()], 8)
            .unwrap();
        assert_ne!(first, second);
        assert!(!ledger.note_is_next(&second).unwrap());
        assert_eq!(ledger.pending_notes(true).unwrap().len(), 1);
        ledger.complete_note(&first).unwrap();
        assert!(ledger.note_is_next(&second).unwrap());
        ledger.complete_note(&second).unwrap();
        item.content = "first value".into();
        let restored = ledger
            .queue_note("bank", "fixture", &scope, &[item], 8)
            .unwrap();
        assert_ne!(restored, first);
        assert_ne!(restored, second);
        assert_eq!(ledger.pending_notes(true).unwrap()[0].operation, restored);
        ledger.complete_note(&restored).unwrap();
        // A completed receipt proves an old write, not current remote content:
        // the Control Plane may since have removed or corrected that document.
        let item = super::super::client::Item {
            content: "first value".into(),
            document_id: "note-document".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            context: "explicit note".into(),
            metadata: Default::default(),
        };
        assert_ne!(
            restored,
            ledger
                .queue_note("bank", "fixture", &scope, &[item], 8)
                .unwrap()
        );
    }
    #[test]
    fn new_source_attachment_and_erasure_invalidate_audit_receipts() {
        let d = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(d.path()).unwrap();
        ledger
            .register_stream("agent", "owner", "session", "branch", 0)
            .unwrap();
        let stream = ledger.streams().unwrap().remove(0);
        ledger.advance(&stream, 1, Some("turn"), 10).unwrap();
        let delivery = ledger.pending(10, true).unwrap().remove(0);
        let operation = ledger.operation("bank", "document", "source").unwrap();
        ledger.record_source_audit("bank", "frontier").unwrap();
        ledger
            .delivery_state(&delivery.id, "processing", Some(&operation), None)
            .unwrap();
        assert!(!ledger.source_audited("bank", "frontier").unwrap());
        ledger.checked_revision(&delivery.id, "frontier").unwrap();
        assert!(ledger.record_source_audit("bank", "frontier").is_ok());
        ledger.record_source_audit("bank", "frontier").unwrap();
        ledger
            .delivery_state(&delivery.id, "processing", Some(&operation), None)
            .unwrap();
        assert!(ledger.source_audited("bank", "frontier").unwrap());
        ledger
            .delivery_state(&delivery.id, "erasing", None, None)
            .unwrap();
        assert!(!ledger.source_audited("bank", "frontier").unwrap());
        assert!(ledger.record_source_audit("bank", "frontier").is_err());
    }
}
