//! Validated historical admission indexes, with a bounded canonical replay tail.
//! No content leaves or current authority are stored in this disposable cache.
use super::*;
use zeroclaw_api::session_events::{
    payloads::PayloadDelta,
    payloads::{PayloadIndex, StoredPayload},
    rules::{HistoryDelta, HistoryIndex},
};

// Transform admission stores the canonical terminal identity, not a boolean.
const FORMAT: i64 = 2;
const INTERVAL: i64 = 4_096;

pub(super) fn load_for_read(conn: &Connection) -> Result<Option<ValidationSnapshot>, ReadError> {
    match load(conn) {
        Ok(snapshot) => Ok(snapshot),
        // Readers retain forward compatibility; future semantics prevent
        // admission acceleration, not ordinary envelope/payload reads.
        Err(AppendError::UnsupportedSchema(_) | AppendError::UnresolvedReachability(_)) => Ok(None),
        Err(AppendError::StorageUnavailable(error)) => Err(ReadError::StorageUnavailable(error)),
        Err(error) => Err(ReadError::Corrupt(error.to_string())),
    }
}

pub(super) fn ensure_schema(conn: &Connection) -> Result<(), AppendError> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS session_admission_checkpoint(
        singleton INTEGER PRIMARY KEY CHECK(singleton=1), format INTEGER NOT NULL,
        event_rowid INTEGER NOT NULL, event_id TEXT NOT NULL, payload_rowid INTEGER NOT NULL,
        erasure_revision TEXT NOT NULL, body BLOB NOT NULL, digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS session_admission_deltas(first_rowid INTEGER PRIMARY KEY,last_rowid INTEGER NOT NULL,event_id TEXT NOT NULL,payload_rowid INTEGER NOT NULL,body BLOB NOT NULL,digest TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS admission_event_update AFTER UPDATE ON session_events BEGIN DELETE FROM session_admission_checkpoint; END;
        CREATE TRIGGER IF NOT EXISTS admission_event_delete AFTER DELETE ON session_events BEGIN DELETE FROM session_admission_checkpoint; END;
        CREATE TRIGGER IF NOT EXISTS admission_payload_update AFTER UPDATE ON event_payloads BEGIN DELETE FROM session_admission_checkpoint; END;
        CREATE TRIGGER IF NOT EXISTS admission_payload_delete AFTER DELETE ON event_payloads BEGIN DELETE FROM session_admission_checkpoint; END;
        CREATE TRIGGER IF NOT EXISTS admission_delta_event_update AFTER UPDATE ON session_events BEGIN DELETE FROM session_admission_deltas; END;
        CREATE TRIGGER IF NOT EXISTS admission_delta_event_delete AFTER DELETE ON session_events BEGIN DELETE FROM session_admission_deltas; END;
        CREATE TRIGGER IF NOT EXISTS admission_delta_payload_update AFTER UPDATE ON event_payloads BEGIN DELETE FROM session_admission_deltas; END;
        CREATE TRIGGER IF NOT EXISTS admission_delta_payload_delete AFTER DELETE ON event_payloads BEGIN DELETE FROM session_admission_deltas; END;
    ").map_err(map_sqlite)
}

fn corrupt(error: impl std::fmt::Display) -> AppendError {
    AppendError::Corrupt(error.to_string())
}

pub(super) fn load(conn: &Connection) -> Result<Option<ValidationSnapshot>, AppendError> {
    type Row = (i64, i64, String, i64, String, Vec<u8>, String);
    let row:Option<Row>=conn.query_row("SELECT format,event_rowid,event_id,payload_rowid,erasure_revision,body,digest FROM session_admission_checkpoint WHERE singleton=1",[],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?))).optional().map_err(map_sqlite)?;
    let Some((
        format,
        mut event_rowid,
        event_id,
        mut payload_rowid,
        erasure_revision,
        body,
        digest,
    )) = row
    else {
        return Ok(None);
    };
    if format != FORMAT
        || event_rowid <= 0
        || payload_rowid < 0
        || Digest::from_bytes(&body).as_str() != digest
        || checkpoints::erasure_revision(conn)
            .map_err(corrupt)?
            .as_str()
            != erasure_revision
    {
        return Ok(None);
    }
    let Ok((mut history, mut payloads)) =
        rmp_serde::from_slice::<(HistoryIndex, PayloadIndex)>(&body)
    else {
        return Ok(None);
    };
    let boundary: Option<Vec<u8>> = conn
        .query_row(
            "SELECT event_json FROM session_events WHERE rowid=?1 AND event_id=?2",
            params![event_rowid, event_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(map_sqlite)?;
    let Some(boundary) = boundary else {
        return Ok(None);
    };
    let boundary = parse_event(&boundary).map_err(corrupt)?;
    if boundary.event_id.as_str() != event_id
        || history.get_event(&boundary.event_id) != Some(&boundary)
    {
        return Ok(None);
    }
    let mut statement=conn.prepare("SELECT first_rowid,last_rowid,event_id,payload_rowid,body,digest FROM session_admission_deltas WHERE last_rowid>?1 ORDER BY first_rowid").map_err(map_sqlite)?;
    let mut rows = statement.query(params![event_rowid]).map_err(map_sqlite)?;
    while let Some(row) = rows.next().map_err(map_sqlite)? {
        let first: i64 = row.get(0).map_err(map_sqlite)?;
        let last: i64 = row.get(1).map_err(map_sqlite)?;
        let id: String = row.get(2).map_err(map_sqlite)?;
        let payload_through: i64 = row.get(3).map_err(map_sqlite)?;
        let bytes: Vec<u8> = row.get(4).map_err(map_sqlite)?;
        let digest: String = row.get(5).map_err(map_sqlite)?;
        if first != event_rowid + 1
            || last < first
            || payload_through < payload_rowid
            || Digest::from_bytes(&bytes).as_str() != digest
        {
            break;
        }
        let Ok((historical_delta, payload_delta)) =
            rmp_serde::from_slice::<(HistoryDelta, PayloadDelta)>(&bytes)
        else {
            break;
        };
        let raw: Option<Vec<u8>> = conn
            .query_row(
                "SELECT event_json FROM session_events WHERE rowid=?1 AND event_id=?2",
                params![last, id],
                |row| row.get(0),
            )
            .optional()
            .map_err(map_sqlite)?;
        let Some(raw) = raw else {
            break;
        };
        let boundary = parse_event(&raw).map_err(corrupt)?;
        if boundary.event_id.as_str() != id
            || historical_delta.recorded_event(&boundary.event_id) != Some(&boundary)
        {
            break;
        }
        history.commit(historical_delta, &[]);
        payloads.commit(payload_delta);
        event_rowid = last;
        payload_rowid = payload_through;
    }
    let mut events = Vec::new();
    let mut heads = std::collections::HashMap::<String, u64>::new();
    let mut statement=conn.prepare("SELECT session_id,sequence,event_id,event_json FROM session_events WHERE rowid>?1 ORDER BY rowid").map_err(map_sqlite)?;
    let mut rows = statement.query(params![event_rowid]).map_err(map_sqlite)?;
    while let Some(row) = rows.next().map_err(map_sqlite)? {
        let stream: String = row.get(0).map_err(map_sqlite)?;
        let sequence: u64 = row.get(1).map_err(map_sqlite)?;
        let id: String = row.get(2).map_err(map_sqlite)?;
        let raw: Vec<u8> = row.get(3).map_err(map_sqlite)?;
        let event = parse_event(&raw).map_err(corrupt)?;
        let prior = if let Some(sequence) = heads.get(&stream) {
            *sequence
        } else {
            conn.query_row("SELECT COALESCE(MAX(sequence),0) FROM session_events WHERE session_id=?1 AND rowid<=?2",params![stream,event_rowid],|row|row.get::<_,u64>(0)).map_err(map_sqlite)?
        };
        if event.session_id.as_str() != stream
            || event.event_id.as_str() != id
            || event.sequence != sequence
            || prior.checked_add(1) != Some(sequence)
        {
            return Err(corrupt(
                "checkpoint tail has an event index or sequence gap",
            ));
        }
        heads.insert(stream, sequence);
        events.push(event);
    }
    let query = format!(
        "SELECT payload_id,codec,CASE WHEN codec IN ({}) THEN NULL ELSE body END FROM event_payloads WHERE rowid>?1",
        content_leaf_sql_list()
    );
    let mut statement = conn.prepare(&query).map_err(map_sqlite)?;
    let records = statement
        .query_map(params![payload_rowid], |row| {
            Ok(StoredPayload {
                reference: zeroclaw_api::session_events::StoredPayloadRef {
                    payload_id: PayloadId::from(row.get::<_, String>(0)?),
                    codec: row.get(1)?,
                },
                body: row.get(2)?,
            })
        })
        .map_err(map_sqlite)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(map_sqlite)?;
    history.extend_committed(&events)?;
    payloads.extend_committed(&events, records, &history, &|id| {
        conn.query_row(
            "SELECT length(body) FROM event_payloads WHERE payload_id=?1",
            params![id.as_str()],
            |row| row.get(0),
        )
        .map_err(map_sqlite)
    })?;
    Ok(Some(ValidationSnapshot::Admission {
        revision: checkpoints::canonical_revision(conn).map_err(corrupt)?,
        history: Box::new(history),
        payloads: Box::new(payloads),
    }))
}

/// Snapshot the already validated prefix BEFORE the incoming batch. Publication
/// shares its transaction; a failed append cannot publish a speculative index.
pub(super) fn maybe_save(
    conn: &Connection,
    history: &HistoryIndex,
    payloads: &PayloadIndex,
    force: bool,
) -> Result<(), AppendError> {
    let through: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(rowid),0) FROM session_events",
            [],
            |row| row.get(0),
        )
        .map_err(map_sqlite)?;
    if through == 0 {
        return Ok(());
    }
    let previous: i64 = conn
        .query_row(
            "SELECT event_rowid FROM session_admission_checkpoint WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(map_sqlite)?
        .unwrap_or(0);
    // Geometric full snapshots keep total rewrite work amortized linear in the
    // archive. Individual commits persist compact validated deltas in between.
    if !force && through.saturating_sub(previous) < INTERVAL.max(previous / 2) {
        return Ok(());
    }
    let event_id: String = conn
        .query_row(
            "SELECT event_id FROM session_events WHERE rowid=?1",
            params![through],
            |row| row.get(0),
        )
        .map_err(map_sqlite)?;
    let payload_through: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(rowid),0) FROM event_payloads",
            [],
            |row| row.get(0),
        )
        .map_err(map_sqlite)?;
    let body = rmp_serde::to_vec_named(&(history, payloads)).map_err(corrupt)?;
    let revision = checkpoints::erasure_revision(conn).map_err(corrupt)?;
    conn.execute("INSERT INTO session_admission_checkpoint(singleton,format,event_rowid,event_id,payload_rowid,erasure_revision,body,digest) VALUES(1,?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(singleton) DO UPDATE SET format=excluded.format,event_rowid=excluded.event_rowid,event_id=excluded.event_id,payload_rowid=excluded.payload_rowid,erasure_revision=excluded.erasure_revision,body=excluded.body,digest=excluded.digest",params![FORMAT,through,event_id,payload_through,revision.as_str(),body,Digest::from_bytes(&body).as_str()]).map_err(map_sqlite)?;
    conn.execute(
        "DELETE FROM session_admission_deltas WHERE last_rowid<=?1",
        params![through],
    )
    .map_err(map_sqlite)?;
    Ok(())
}

pub(super) fn save_delta(
    conn: &Connection,
    events: &[SessionEvent],
    history: &HistoryDelta,
    payloads: &PayloadDelta,
) -> Result<(), AppendError> {
    let present: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM session_admission_checkpoint WHERE singleton=1)",
            [],
            |row| row.get(0),
        )
        .map_err(map_sqlite)?;
    if !present {
        return Ok(());
    }
    let last: i64 = conn
        .query_row("SELECT MAX(rowid) FROM session_events", [], |row| {
            row.get(0)
        })
        .map_err(map_sqlite)?;
    let count = i64::try_from(events.len()).map_err(corrupt)?;
    let event = events
        .last()
        .ok_or_else(|| corrupt("empty admission delta"))?;
    let payload_through: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(rowid),0) FROM event_payloads",
            [],
            |row| row.get(0),
        )
        .map_err(map_sqlite)?;
    let body = rmp_serde::to_vec_named(&(history, payloads)).map_err(corrupt)?;
    conn.execute("INSERT INTO session_admission_deltas(first_rowid,last_rowid,event_id,payload_rowid,body,digest) VALUES(?1,?2,?3,?4,?5,?6)",params![last-count+1,last,event.event_id.as_str(),payload_through,body,Digest::from_bytes(&body).as_str()]).map_err(map_sqlite)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroclaw_api::session_events::{EventActor, EventData, StoredPayloadRef};

    fn metadata(
        store: &SqliteEventStore,
        stream: &SessionId,
        key: &str,
        reference: &StoredPayloadRef,
        count: usize,
        payloads: Vec<PayloadRecord>,
    ) {
        let events = (0..count)
            .map(|_| {
                SessionEvent::typed(
                    stream.clone(),
                    EventData::SessionMetadataChanged {
                        field: "title".into(),
                        new_value_ref: reference.clone(),
                    },
                    EventActor::default(),
                )
                .expect("event")
            })
            .collect();
        store
            .append_batch(AppendRequest {
                session_id: stream.clone(),
                expected_head: store.head(stream).expect("head"),
                idempotency_key: key.into(),
                events,
                payloads,
                erasure_reference_commitment: None,
            })
            .expect("append");
    }

    #[test]
    fn reopen_and_peer_append_restore_validated_prefix_without_full_history_replay() {
        let directory = tempfile::tempdir().expect("directory");
        let store = SqliteEventStore::open(directory.path()).expect("store");
        let stream = SessionId::new();
        store
            .append_batch(super::super::tests::request(&stream, "root", 0))
            .expect("root");
        let leaf = PayloadRecord::new(
            PayloadId::new(),
            "session.text/v1".into(),
            b"fixture-original-content-must-not-enter-admission-checkpoint".to_vec(),
        )
        .expect("leaf");
        let reference = StoredPayloadRef::from(&leaf);
        metadata(
            &store,
            &stream,
            "archive",
            &reference,
            5_000,
            vec![leaf.clone()],
        );
        metadata(&store, &stream, "checkpoint", &reference, 1, vec![]);
        let body: Vec<u8> = store
            .conn
            .lock()
            .query_row("SELECT body FROM session_admission_checkpoint", [], |row| {
                row.get(0)
            })
            .expect("checkpoint");
        assert!(
            !body.windows(leaf.body.len()).any(|part| part == leaf.body),
            "content leaves must not be duplicated into the checkpoint"
        );
        drop(store);
        let store = SqliteEventStore::open(directory.path()).expect("reopen");
        sqlite_history::VERIFIED_HISTORY_READS.with(|count| count.set(0));
        metadata(&store, &stream, "after-reopen", &reference, 1, vec![]);
        let delta_leaf = PayloadRecord::new(
            PayloadId::new(),
            "session.text/v1".into(),
            b"fixture-new-content-must-not-enter-admission-delta".to_vec(),
        )
        .expect("delta leaf");
        metadata(
            &store,
            &stream,
            "delta-content",
            &StoredPayloadRef::from(&delta_leaf),
            1,
            vec![delta_leaf.clone()],
        );
        {
            let conn = store.conn.lock();
            let mut statement = conn
                .prepare("SELECT body FROM session_admission_deltas")
                .expect("deltas");
            for body in statement
                .query_map([], |row| row.get::<_, Vec<u8>>(0))
                .expect("rows")
            {
                assert!(
                    !body
                        .expect("body")
                        .windows(delta_leaf.body.len())
                        .any(|part| part == delta_leaf.body),
                    "validated deltas must not duplicate content leaves"
                );
            }
        }
        sqlite_history::VERIFIED_HISTORY_READS.with(|count| {
            assert_eq!(
                count.get(),
                0,
                "reopening must replay only the checkpoint tail"
            )
        });
        let peer = SqliteEventStore::open(directory.path()).expect("peer");
        let other = SessionId::new();
        peer.append_batch(super::super::tests::request(&other, "other-root", 0))
            .expect("other session");
        metadata(&peer, &stream, "peer-append", &reference, 1, vec![]);
        metadata(&store, &stream, "after-peer", &reference, 1, vec![]);
        let compacted: i64 = store
            .conn
            .lock()
            .query_row(
                "SELECT event_rowid FROM session_admission_checkpoint",
                [],
                |row| row.get(0),
            )
            .expect("base cursor");
        assert_eq!(
            compacted, 5_001,
            "ordinary commits must journal deltas, not rewrite the full growing checkpoint"
        );
        sqlite_history::VERIFIED_HISTORY_READS.with(|count| {
            assert_eq!(
                count.get(),
                0,
                "peer commits must advance the validated prefix incrementally"
            )
        });
        assert_eq!(
            store
                .read_payload(&leaf.payload_id)
                .expect("retained payload"),
            PayloadRead::Present(leaf)
        );
        peer.conn
            .lock()
            .execute(
                "UPDATE session_admission_checkpoint SET body=?1",
                params![b"corrupt".as_slice()],
            )
            .expect("corrupt cache");
        drop(store);
        drop(peer);
        let store = SqliteEventStore::open(directory.path()).expect("reopen corrupted cache");
        metadata(&store, &stream, "repair-checkpoint", &reference, 1, vec![]);
        sqlite_history::VERIFIED_HISTORY_READS.with(|count| {
            assert_eq!(
                count.get(),
                1,
                "a corrupt cache must rebuild from canonical events"
            )
        });
        drop(store);
        let store = SqliteEventStore::open(directory.path()).expect("reopen repaired cache");
        metadata(&store, &stream, "after-repair", &reference, 1, vec![]);
        sqlite_history::VERIFIED_HISTORY_READS
            .with(|count| assert_eq!(count.get(), 1, "the rebuilt checkpoint must be reusable"));
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;
    use zeroclaw_api::session_events::{CaptureMode, EventData, StoredPayloadRef};

    #[test]
    fn old_admission_format_rebuilds_without_changing_canonical_events() {
        let directory = tempfile::tempdir().unwrap();
        let store = SqliteEventStore::open(directory.path()).unwrap();
        let stream = SessionId::new();
        let branch = zeroclaw_api::session_events::BranchId::new();
        let mut root = SessionEvent::typed(
            stream.clone(),
            EventData::SessionStarted {
                owning_agent: "fixture".into(),
                capture_mode: CaptureMode::Native,
            },
            zeroclaw_api::session_events::EventActor::default(),
        )
        .unwrap();
        root.branch_id = Some(branch.clone());
        let root_id = root.event_id.clone();
        let receipt = store
            .append_batch(AppendRequest {
                session_id: stream.clone(),
                expected_head: EventCursor::beginning(stream.clone()),
                idempotency_key: "fixture_root".into(),
                events: vec![root],
                payloads: vec![],
                erasure_reference_commitment: None,
            })
            .unwrap();
        {
            let conn = store.conn.lock();
            let validation = store.validation.lock();
            let Some(ValidationSnapshot::Admission {
                history, payloads, ..
            }) = validation.as_ref()
            else {
                panic!("fixture admission index missing");
            };
            maybe_save(&conn, history, payloads, true).unwrap();
            conn.execute("UPDATE session_admission_checkpoint SET format=1", [])
                .unwrap();
            assert!(load(&conn).unwrap().is_none());
        }
        let original = store.read_event(&root_id).unwrap().unwrap();
        drop(store);
        let store = SqliteEventStore::open(directory.path()).unwrap();
        let title = PayloadRecord::new(
            PayloadId::new(),
            "session.text/v1".into(),
            b"fixture title".to_vec(),
        )
        .unwrap();
        let mut metadata = SessionEvent::typed(
            stream.clone(),
            EventData::SessionMetadataChanged {
                field: "title".into(),
                new_value_ref: StoredPayloadRef::from(&title),
            },
            zeroclaw_api::session_events::EventActor::default(),
        )
        .unwrap();
        metadata.branch_id = Some(branch);
        store
            .append_batch(AppendRequest {
                session_id: stream,
                expected_head: receipt.last,
                idempotency_key: "fixture_title".into(),
                events: vec![metadata],
                payloads: vec![title],
                erasure_reference_commitment: None,
            })
            .unwrap();
        assert_eq!(store.read_event(&root_id).unwrap().unwrap(), original);
        let conn = store.conn.lock();
        let format: i64 = conn
            .query_row(
                "SELECT format FROM session_admission_checkpoint WHERE singleton=1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(format, FORMAT);
        assert!(load_for_read(&conn).unwrap().is_some());
    }
}
