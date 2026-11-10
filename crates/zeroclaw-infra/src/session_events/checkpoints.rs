//! Rebuildable read-model checkpoints. Events and payloads remain canonical.
use super::*;
use zeroclaw_api::session_events::{ProjectionCheckpoint, canonical_json};

pub(super) fn ensure_schema(conn: &Connection) -> Result<(), AppendError> {
    // Opening an existing store checks for unfinished physical cleanup. Keep
    // that probe proportional to redacted rows, not the complete payload archive.
    conn.execute_batch("CREATE INDEX IF NOT EXISTS session_redacted_payloads ON event_payloads(payload_id) WHERE body IS NULL;").map_err(map_sqlite)?;
    conn.execute_batch("CREATE INDEX IF NOT EXISTS session_metadata_events ON session_events(session_id,sequence) WHERE json_extract(event_json,'$.kind')='session_metadata_changed';").map_err(map_sqlite)?;
    conn.execute_batch("CREATE TABLE IF NOT EXISTS session_canonical_revision(singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL);
        INSERT OR IGNORE INTO session_canonical_revision VALUES(1,0);
        CREATE TABLE IF NOT EXISTS session_projection_checkpoints(
        session_id TEXT NOT NULL, kind TEXT NOT NULL, sequence INTEGER NOT NULL,
        event_id TEXT NOT NULL, erasure_revision TEXT NOT NULL, body BLOB NOT NULL,
        digest TEXT NOT NULL, PRIMARY KEY(session_id,kind));
        CREATE TRIGGER IF NOT EXISTS session_checkpoint_event_update AFTER UPDATE ON session_events BEGIN DELETE FROM session_projection_checkpoints; END;
        CREATE TRIGGER IF NOT EXISTS session_checkpoint_event_delete AFTER DELETE ON session_events BEGIN DELETE FROM session_projection_checkpoints; END;
        CREATE TRIGGER IF NOT EXISTS session_checkpoint_payload_update AFTER UPDATE ON event_payloads BEGIN DELETE FROM session_projection_checkpoints; END;
        CREATE TRIGGER IF NOT EXISTS session_checkpoint_payload_delete AFTER DELETE ON event_payloads BEGIN DELETE FROM session_projection_checkpoints; END;
    ").map_err(map_sqlite)?;
    for coordinate in ["causation_id", "invocation_id"] {
        conn.execute_batch(&format!("CREATE INDEX IF NOT EXISTS session_events_{coordinate}_kind ON session_events(session_id,json_extract(event_json,'$.{coordinate}'),json_extract(event_json,'$.kind'),sequence) WHERE json_extract(event_json,'$.body.codec_version')=1;")).map_err(map_sqlite)?;
    }
    // This creates the canonical-table change counter. Read-model/index writes
    // do not change it; INSERT, physical erasure and unsupported raw edits do.
    for table in ["session_events", "event_payloads"] {
        for operation in ["INSERT", "UPDATE", "DELETE"] {
            conn.execute_batch(&format!("CREATE TRIGGER IF NOT EXISTS {table}_canonical_{operation} AFTER {operation} ON {table} BEGIN UPDATE session_canonical_revision SET revision=revision+1 WHERE singleton=1; END;")).map_err(map_sqlite)?;
        }
    }
    Ok(())
}

pub(super) fn canonical_revision(conn: &Connection) -> Result<i64, ReadError> {
    conn.query_row(
        "SELECT revision FROM session_canonical_revision WHERE singleton=1",
        [],
        |row| row.get(0),
    )
    .map_err(read_error)
}

pub(super) fn erasure_revision(conn: &Connection) -> Result<Digest, ReadError> {
    let mut statement = conn.prepare("SELECT event_id FROM session_events WHERE json_extract(event_json,'$.kind')='payload_erasure_requested' AND json_extract(event_json,'$.body.codec_version')=1 ORDER BY event_id").map_err(read_error)?;
    let ids = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(read_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(read_error)?;
    Ok(Digest::from_bytes(
        &canonical_json(&ids).map_err(|error| ReadError::Corrupt(error.to_string()))?,
    ))
}

impl SqliteEventStore {
    pub(super) fn load_projection(
        &self,
        stream: &SessionId,
        kind: &str,
    ) -> Result<Option<ProjectionCheckpoint>, ReadError> {
        let mut connection = self.conn.lock();
        let conn = connection.transaction().map_err(read_error)?;
        type Row = (u64, String, String, Vec<u8>, String);
        let row: Option<Row> = conn.query_row("SELECT sequence,event_id,erasure_revision,body,digest FROM session_projection_checkpoints WHERE session_id=?1 AND kind=?2", params![stream.as_str(),kind], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?))).optional().map_err(read_error)?;
        let Some((sequence, event_id, revision, body, digest)) = row else {
            return Ok(None);
        };
        if sequence == 0
            || Digest::from_bytes(&body).as_str() != digest
            || erasure_revision(&conn)?.as_str() != revision
        {
            return Ok(None);
        }
        let boundary: Option<Vec<u8>> = conn.query_row("SELECT event_json FROM session_events WHERE session_id=?1 AND sequence=?2 AND event_id=?3", params![stream.as_str(),sequence,event_id], |row| row.get(0)).optional().map_err(read_error)?;
        let Some(boundary) = boundary else {
            return Ok(None);
        };
        let boundary = parse_event(&boundary)?;
        if boundary.session_id != *stream
            || boundary.sequence != sequence
            || boundary.event_id.as_str() != event_id
        {
            return Ok(None);
        }
        Ok(Some(ProjectionCheckpoint {
            through: EventCursor {
                session_id: stream.clone(),
                sequence,
            },
            erasure_revision: revision.into(),
            body,
            expected_revision: None,
        }))
    }

    pub(super) fn save_projection(
        &self,
        kind: &str,
        checkpoint: &ProjectionCheckpoint,
    ) -> Result<(), ReadError> {
        let mut connection = self.conn.lock();
        let conn = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(read_error)?;
        if checkpoint.expected_revision != Some(canonical_revision(&conn)?) {
            return Ok(());
        }
        if erasure_revision(&conn)? != checkpoint.erasure_revision {
            return Ok(());
        }
        let event_id: Option<String> = conn
            .query_row(
                "SELECT event_id FROM session_events WHERE session_id=?1 AND sequence=?2",
                params![
                    checkpoint.through.session_id.as_str(),
                    checkpoint.through.sequence
                ],
                |row| row.get(0),
            )
            .optional()
            .map_err(read_error)?;
        let Some(event_id) = event_id else {
            return Ok(());
        };
        conn.execute("INSERT INTO session_projection_checkpoints(session_id,kind,sequence,event_id,erasure_revision,body,digest) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(session_id,kind) DO UPDATE SET sequence=excluded.sequence,event_id=excluded.event_id,erasure_revision=excluded.erasure_revision,body=excluded.body,digest=excluded.digest WHERE excluded.sequence >= session_projection_checkpoints.sequence", params![checkpoint.through.session_id.as_str(),kind,checkpoint.through.sequence,event_id,checkpoint.erasure_revision.as_str(),checkpoint.body,Digest::from_bytes(&checkpoint.body).as_str()]).map_err(read_error)?;
        conn.commit().map_err(read_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrupt_or_mutated_projection_checkpoints_rebuild_and_metadata_writes_do_not_invalidate_admission()
     {
        let dir = tempfile::tempdir().expect("directory");
        let store = SqliteEventStore::open(dir.path()).expect("store");
        let stream = SessionId::new();
        store
            .append_batch(super::super::tests::request(&stream, "root", 0))
            .expect("root");
        let checkpoint = ProjectionCheckpoint {
            through: store.head(&stream).expect("head"),
            erasure_revision: store
                .erasure_revision()
                .expect("revision")
                .expect("supported"),
            body: b"fixture projection".to_vec(),
            expected_revision: store.canonical_revision().expect("revision"),
        };
        let revision = canonical_revision(&store.conn.lock()).expect("revision");
        store
            .save_projection_checkpoint("fixture/v1", &checkpoint)
            .expect("save");
        assert_eq!(
            canonical_revision(&store.conn.lock()).expect("revision"),
            revision,
            "derived writes must not force canonical admission reconstruction"
        );
        assert_eq!(
            store
                .load_projection_checkpoint(&stream, "fixture/v1")
                .expect("load")
                .expect("checkpoint")
                .body,
            checkpoint.body
        );
        let reopened = SqliteEventStore::open(dir.path()).expect("reopen");
        assert!(
            reopened
                .load_projection_checkpoint(&stream, "fixture/v1")
                .expect("load")
                .is_some()
        );
        reopened
            .conn
            .lock()
            .execute(
                "UPDATE session_projection_checkpoints SET body=?1",
                params![b"broken".as_slice()],
            )
            .expect("corrupt cache");
        assert!(
            store
                .load_projection_checkpoint(&stream, "fixture/v1")
                .expect("corrupt cache is disposable")
                .is_none()
        );
        store
            .save_projection_checkpoint("fixture/v1", &checkpoint)
            .expect("rebuild");
        reopened
            .conn
            .lock()
            .execute("UPDATE session_events SET event_json=event_json", [])
            .expect("unsupported canonical edit");
        assert!(
            store
                .load_projection_checkpoint(&stream, "fixture/v1")
                .expect("invalidated")
                .is_none()
        );
        assert!(canonical_revision(&store.conn.lock()).expect("revision") > revision);
        store
            .save_projection_checkpoint("fixture/v1", &checkpoint)
            .expect("stale publication is ignored");
        assert!(
            store
                .load_projection_checkpoint(&stream, "fixture/v1")
                .expect("load")
                .is_none()
        );
    }
}
