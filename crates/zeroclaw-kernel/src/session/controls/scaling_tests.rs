use super::*;
use std::sync::atomic::AtomicUsize;

pub(super) struct CountedStore {
    inner: Arc<dyn SessionEventStore>,
    read_events: AtomicUsize,
    read_payloads: AtomicUsize,
    forbidden_payloads: std::sync::Mutex<std::collections::HashSet<PayloadId>>,
}

impl CountedStore {
    pub(super) fn new(inner: Arc<dyn SessionEventStore>) -> Self {
        Self {
            inner,
            read_events: AtomicUsize::new(0),
            read_payloads: AtomicUsize::new(0),
            forbidden_payloads: Default::default(),
        }
    }
    pub(super) fn forbid(&self, id: PayloadId) {
        self.forbidden_payloads.lock().expect("lock").insert(id);
    }
    pub(super) fn clear_forbidden(&self) {
        self.forbidden_payloads.lock().expect("lock").clear();
    }
}

impl SessionEventStore for CountedStore {
    fn append_batch(&self, request: AppendRequest) -> Result<AppendReceipt, AppendError> {
        self.inner.append_batch(request)
    }
    fn read_event(&self, id: &EventId) -> Result<Option<SessionEvent>, ReadError> {
        self.inner.read_event(id)
    }
    fn read_latest_event(
        &self,
        stream: &SessionId,
        branch: &BranchId,
        turn: Option<&TurnId>,
        kind: &EventKind,
    ) -> Result<Option<SessionEvent>, ReadError> {
        self.inner.read_latest_event(stream, branch, turn, kind)
    }
    fn read_page(
        &self,
        stream: &SessionId,
        after: &EventCursor,
        through: Option<&EventCursor>,
        limit: usize,
    ) -> Result<EventPage, ReadError> {
        let page = self.inner.read_page(stream, after, through, limit)?;
        self.read_events
            .fetch_add(page.events.len(), Ordering::SeqCst);
        Ok(page)
    }
    fn head(&self, stream: &SessionId) -> Result<EventCursor, ReadError> {
        self.inner.head(stream)
    }
    fn lookup_receipt(
        &self,
        stream: &SessionId,
        key: &str,
    ) -> Result<Option<AppendReceipt>, ReadError> {
        self.inner.lookup_receipt(stream, key)
    }
    fn read_payload(&self, id: &PayloadId) -> Result<PayloadRead, ReadError> {
        if self.forbidden_payloads.lock().expect("lock").contains(id) {
            return Err(ReadError::Corrupt(
                "raw summarized source was reread".into(),
            ));
        }
        self.read_payloads.fetch_add(1, Ordering::SeqCst);
        self.inner.read_payload(id)
    }
    fn erasure_revision(&self) -> Result<Option<Digest>, ReadError> {
        self.inner.erasure_revision()
    }
    fn read_payload_disposition(
        &self,
        reference: &StoredPayloadRef,
    ) -> Result<zeroclaw_api::session_events::payloads::PayloadDisposition, ReadError> {
        self.inner.read_payload_disposition(reference)
    }
    fn erasure_scope(
        &self,
        targets: &[StoredPayloadRef],
    ) -> Result<(Vec<SessionId>, Digest), AppendError> {
        self.inner.erasure_scope(targets)
    }
}

#[tokio::test]
async fn context_planning_work_is_bounded_by_active_context_and_new_events() {
    for archive_size in [0, 1_000, 10_000] {
        let directory = tempfile::tempdir().expect("directory");
        let (mut port, _) = session(directory.path(), "sqlite");
        let store = Arc::new(CountedStore::new(port.owner.store.clone()));
        port.owner = Arc::new(SessionCoordinator::new(store.clone()));
        let title = PayloadRecord::new(
            PayloadId::new(),
            "session.text/v1".into(),
            b"fixture".to_vec(),
        )
        .expect("payload");
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
        let snapshot = ContextSnapshot {
            messages: vec![
                ConversationMessage::Chat(ChatMessage::user("active question")),
                ConversationMessage::Chat(ChatMessage::assistant("active answer")),
            ],
            history_has_trim_breadcrumb: Some(false),
        };
        port.replace_context(snapshot.clone(), "scaling_fixture")
            .await
            .expect("install active context");
        let input = port.latest_context().await.expect("current context");
        let expected =
            crate::session::context_messages::context_refs(store.as_ref(), &input).expect("refs");
        store.read_events.store(0, Ordering::SeqCst);
        store.read_payloads.store(0, Ordering::SeqCst);
        let plan = port
            .owner
            .plan_context(
                &port.branch,
                &input,
                &EventId::new(),
                &snapshot.messages,
                &[],
                &[],
            )
            .expect("plan");
        assert_eq!(
            plan.refs, expected,
            "retained message identities must survive"
        );
        assert_eq!(
            store.read_events.load(Ordering::SeqCst),
            0,
            "archive size {archive_size}"
        );
        assert!(
            store.read_payloads.load(Ordering::SeqCst) <= 4,
            "archive size {archive_size}"
        );
        assert!(plan.ranges.len() <= 1, "source metadata must stay bounded");
        assert!(
            plan.payloads.is_empty(),
            "unchanged messages reuse canonical payloads"
        );
    }
}
