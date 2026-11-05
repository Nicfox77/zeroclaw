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
    fn load_projection_checkpoint(
        &self,
        stream: &SessionId,
        kind: &str,
    ) -> Result<Option<ProjectionCheckpoint>, ReadError> {
        self.inner.load_projection_checkpoint(stream, kind)
    }
    fn save_projection_checkpoint(
        &self,
        kind: &str,
        checkpoint: &ProjectionCheckpoint,
    ) -> Result<(), ReadError> {
        self.inner.save_projection_checkpoint(kind, checkpoint)
    }
    fn read_latest_event_before(
        &self,
        stream: &SessionId,
        branch: &BranchId,
        turn: Option<&TurnId>,
        kind: &EventKind,
        through: u64,
    ) -> Result<Option<SessionEvent>, ReadError> {
        self.inner
            .read_latest_event_before(stream, branch, turn, kind, through)
    }
    fn read_context_projection(
        &self,
        stream: &SessionId,
        branch: &BranchId,
        turn: Option<&TurnId>,
        reason: &str,
        through: u64,
    ) -> Result<Option<SessionEvent>, ReadError> {
        self.inner
            .read_context_projection(stream, branch, turn, reason, through)
    }
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

#[tokio::test]
async fn status_and_metadata_resume_checkpoint_and_read_only_new_events() {
    let directory = tempfile::tempdir().expect("directory");
    let (mut port, authority) = session(directory.path(), "sqlite");
    let store = Arc::new(CountedStore::new(port.owner.store.clone()));
    port.owner = Arc::new(SessionCoordinator::new(store.clone()));
    let title = PayloadRecord::new(
        PayloadId::new(),
        "session.text/v1".into(),
        b"fixture title".to_vec(),
    )
    .expect("title");
    let events = (0..10_000)
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
            idempotency_key: "metadata-archive".into(),
            events,
            payloads: vec![title],
            erasure_reference_commitment: None,
        })
        .expect("archive");
    let initial = port
        .query()
        .summary(port.stream.clone())
        .await
        .expect("cold projection");
    assert_eq!(initial.title(), Some("fixture title"));
    store.read_events.store(0, Ordering::SeqCst);
    assert_eq!(
        port.query()
            .current_state(port.stream.clone())
            .await
            .expect("status")["state"],
        "idle"
    );
    assert!(
        port.query()
            .model_selection(port.stream.clone())
            .await
            .expect("model selection")
            .is_none()
    );
    assert_eq!(
        store.read_events.load(Ordering::SeqCst),
        0,
        "status and model selection must not replay the archive"
    );
    let checkpoint = store
        .load_projection_checkpoint(&port.stream, "session-summary/v1")
        .expect("checkpoint")
        .expect("present");
    assert!(
        checkpoint.body.len() < 4_096,
        "ordinary checkpoint size must not grow with archived events"
    );
    let reopened = Arc::new(CountedStore::new(
        zeroclaw_infra::session_events::open_event_store(directory.path(), "sqlite")
            .expect("reopen"),
    ));
    port.owner = Arc::new(SessionCoordinator::new(reopened.clone()));
    assert_eq!(
        port.query()
            .summary(port.stream.clone())
            .await
            .expect("reopened summary")
            .title(),
        initial.title()
    );
    assert!(
        reopened.read_events.load(Ordering::SeqCst) <= 1,
        "reopening must resolve only latest metadata references"
    );
    authority.0.store(false, Ordering::SeqCst);
    assert!(matches!(
        port.query().current_state(port.stream.clone()).await,
        Err(SessionError::PermissionDenied(_))
    ));
    authority.0.store(true, Ordering::SeqCst);
    let before = reopened.head(&port.stream).expect("head").sequence;
    port.record_context_input(input(), "new input", "scaling_fixture")
        .await
        .expect("input");
    let after = reopened.head(&port.stream).expect("head").sequence;
    reopened.read_events.store(0, Ordering::SeqCst);
    assert_eq!(
        port.query()
            .summary(port.stream.clone())
            .await
            .expect("tail projection")
            .message_count,
        1
    );
    assert!(
        reopened.read_events.load(Ordering::SeqCst) <= (after - before) as usize + 2,
        "only the newly appended tail and latest metadata may be read"
    );
    let page = reopened
        .read_page(
            &port.stream,
            &EventCursor {
                session_id: port.stream.clone(),
                sequence: before,
            },
            None,
            100,
        )
        .expect("new events");
    let target = page
        .events
        .iter()
        .find(|event| matches!(event.body.data, EventData::UserContentRecorded { .. }))
        .expect("new content")
        .event_id
        .clone();
    port.hide_content(target, "fixture_hide")
        .await
        .expect("hide compacted content");
    assert_eq!(
        port.query()
            .summary(port.stream.clone())
            .await
            .expect("edited summary")
            .message_count,
        0,
        "edits must decrement compacted counts without replay"
    );
}
