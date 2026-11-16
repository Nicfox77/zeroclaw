//! Opt-in profiling against an operator-supplied, isolated session-store copy.
use super::*;
use zeroclaw_api::session_events::{StoredPayloadRef, context_read};

struct SerialReads<'a>(&'a dyn SessionEventStore);
impl SessionEventStore for SerialReads<'_> {
    fn append_batch(&self, _: AppendRequest) -> Result<AppendReceipt, AppendError> {
        Err(AppendError::StorageUnavailable(
            "read-only benchmark adapter".into(),
        ))
    }
    fn read_event(&self, id: &EventId) -> Result<Option<SessionEvent>, ReadError> {
        self.0.read_event(id)
    }
    fn read_page(
        &self,
        s: &SessionId,
        a: &EventCursor,
        t: Option<&EventCursor>,
        n: usize,
    ) -> Result<EventPage, ReadError> {
        self.0.read_page(s, a, t, n)
    }
    fn head(&self, s: &SessionId) -> Result<EventCursor, ReadError> {
        self.0.head(s)
    }
    fn lookup_receipt(&self, s: &SessionId, key: &str) -> Result<Option<AppendReceipt>, ReadError> {
        self.0.lookup_receipt(s, key)
    }
    fn read_payload(&self, id: &PayloadId) -> Result<PayloadRead, ReadError> {
        self.0.read_payload(id)
    }
    fn erasure_scope(
        &self,
        refs: &[StoredPayloadRef],
    ) -> Result<(Vec<SessionId>, Digest), AppendError> {
        self.0.erasure_scope(refs)
    }
}

#[test]
#[ignore = "requires an isolated native-store copy and explicit projection ID"]
fn profile_fragmented_context_reads() {
    let path = std::path::PathBuf::from(
        std::env::var("ZEROCLAW_CONTEXT_BENCH_DATA").expect("isolated data path"),
    );
    assert!(
        path.join(".context-benchmark-copy").is_file(),
        "refuse to open a non-benchmark data directory"
    );
    let id =
        EventId::parse(std::env::var("ZEROCLAW_CONTEXT_BENCH_PROJECTION").expect("projection ID"))
            .expect("ID");
    let store = SqliteEventStore::open(&path).expect("copied store");
    let event = store.read_event(&id).expect("read").expect("event");
    let refs = context_read::context_refs(&store, &event).expect("references");
    let read = |source: &dyn SessionEventStore| {
        refs.iter()
            .map(|item| context_read::resolve_message(source, &event.session_id, item))
            .collect::<Result<Vec<_>, _>>()
            .expect("resolve")
    };
    let reference = serde_json::to_vec(&read(&store)).expect("reference view");
    let serial = SerialReads(&store);
    for round in 0..3 {
        for (mode, source) in [
            ("serial", &serial as &dyn SessionEventStore),
            ("batched", &store as &dyn SessionEventStore),
        ] {
            let started = std::time::Instant::now();
            let messages = read(source);
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            assert!(
                serde_json::to_vec(&messages).expect("view") == reference,
                "context representation changed"
            );
            eprintln!(
                "context_read_bench round={round} mode={mode} messages={} duration_ms={ms:.3}",
                refs.len()
            );
        }
    }
}
