//! Fault injection at the durable worker/HTTP boundary, with process-local
//! handles dropped between attempts to exercise restart reconciliation.
use super::*;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Default)]
pub(crate) struct ServerState {
    drop_ack: bool,
    pub(crate) retain_posts: usize,
    pub(crate) retained: Vec<serde_json::Value>,
    retry_posts: usize,
    recall_posts: usize,
    pub(crate) deletes: usize,
    operation: Option<String>,
    pub(crate) status: String,
    has_model: bool,
    refresh_posts: usize,
    refresh_status: String,
}
pub(crate) struct FaultApi {
    pub(crate) url: String,
    pub(crate) state: Arc<Mutex<ServerState>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl FaultApi {
    pub(crate) fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let state = Arc::new(Mutex::new(ServerState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let shared = state.clone();
        let stopping = stop.clone();
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                };
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(3)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut buffer = [0; 4096];
                let split = loop {
                    let n = stream.read(&mut buffer).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                    assert!(bytes.len() < 1024 * 1024);
                    if let Some(index) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        break index + 4;
                    }
                };
                let header = String::from_utf8(bytes[..split].to_vec()).unwrap();
                let length = header
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .and_then(|v| v.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                while bytes.len() < split + length {
                    let n = stream.read(&mut buffer).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                }
                let mut parts = header.lines().next().unwrap().split_whitespace();
                let method = parts.next().unwrap();
                let path = parts.next().unwrap().split('?').next().unwrap();
                let body: serde_json::Value = if length == 0 {
                    serde_json::Value::Null
                } else {
                    serde_json::from_slice(&bytes[split..split + length]).unwrap()
                };
                let mut s = shared.lock().unwrap();
                let (code, response) = if method == "POST" && path.ends_with("/memories") {
                    s.retain_posts += 1;
                    s.retained.push(body.clone());
                    s.operation = Some(body["operation_id"].as_str().unwrap().into());
                    if s.status.is_empty() {
                        s.status = "processing".into();
                    }
                    if s.drop_ack {
                        s.drop_ack = false;
                        continue;
                    }
                    (
                        200,
                        serde_json::json!({"operation_id":s.operation,"status":"accepted"}),
                    )
                } else if path.ends_with("/recall") {
                    s.recall_posts += 1;
                    (
                        200,
                        serde_json::json!({"results":[{"id":"fixture-fact","text":"The fictional Larch project uses SQLite."}]}),
                    )
                } else if path.ends_with("/retry") {
                    s.retry_posts += 1;
                    s.status = "processing".into();
                    (200, serde_json::json!({}))
                } else if path.ends_with("/operations/refresh-fixture") {
                    (200, serde_json::json!({"status":s.refresh_status}))
                } else if path.contains("/operations/") {
                    match &s.operation {
                        Some(id) if path.ends_with(id) => {
                            (200, serde_json::json!({"status":s.status}))
                        }
                        _ => (404, serde_json::json!({})),
                    }
                } else if path.ends_with("/operations") {
                    (200, serde_json::json!({"total":0,"operations":[]}))
                } else if method == "DELETE" && path.contains("/documents/") {
                    s.deletes += 1;
                    (200, serde_json::json!({"document_deleted":1}))
                } else if path.ends_with("/mental-models") {
                    let items = if s.has_model {
                        vec![serde_json::json!({"id":"fixture-model"})]
                    } else {
                        vec![]
                    };
                    (200, serde_json::json!({"total":items.len(),"items":items}))
                } else if path.ends_with("/refresh") {
                    assert_eq!(body["mode"], "full");
                    s.refresh_posts += 1;
                    s.refresh_status = "processing".into();
                    (200, serde_json::json!({"operation_id":"refresh-fixture"}))
                } else {
                    (200, serde_json::json!({}))
                };
                let body = response.to_string();
                write!(stream,"HTTP/1.1 {code} fixture\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
            }
        });
        Self {
            url,
            state,
            stop,
            thread: Some(thread),
        }
    }
}
impl Drop for FaultApi {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}
#[derive(Default)]
struct Source {
    erased: AtomicBool,
}
#[async_trait::async_trait]
impl TranscriptSource for Source {
    async fn revision(&self, _stream: &Stream) -> anyhow::Result<String> {
        Ok(if self.erased.load(Ordering::Relaxed) {
            "frontier-2"
        } else {
            "frontier-1"
        }
        .into())
    }
    async fn next(&self, stream: &Stream) -> anyhow::Result<Vec<SourceEvent>> {
        Ok(if stream.cursor == 0 {
            vec![SourceEvent {
                sequence: 1,
                occurred_at: 1000,
                committed_turn: Some("turn-fixture".into()),
            }]
        } else {
            vec![]
        })
    }
    async fn load(&self, _stream: &Stream, _terminal: &str) -> anyhow::Result<Option<Vec<Item>>> {
        Ok((!self.erased.load(Ordering::Relaxed)).then(|| {
            vec![Item {
                update_mode: None,
                content: "User: The fictional Larch project uses SQLite.".into(),
                document_id: "fixture-document".into(),
                timestamp: "2026-01-01T00:00:00Z".into(),
                context: "Fixture conversation".into(),
                metadata: Default::default(),
            }]
        }))
    }
}
pub(crate) fn config(directory: &std::path::Path, url: &str) -> zeroclaw_config::schema::Config {
    let mut c = zeroclaw_config::schema::Config {
        data_dir: directory.into(),
        ..Default::default()
    };
    c.memory.backend = "hindsight".into();
    c.memory.hindsight.mode = zeroclaw_config::hindsight::RuntimeMode::External;
    c.memory.hindsight.api_url = url.into();
    c.channels.session_persistence = true;
    c.channels.session_backend = "sqlite".into();
    let mut agent = zeroclaw_config::schema::AliasedAgentConfig::default();
    agent.memory.backend = zeroclaw_config::multi_agent::MemoryBackendKind::Hindsight;
    c.agents.insert("fixture".into(), agent);
    c
}
fn register(directory: &std::path::Path) {
    Ledger::open(directory)
        .unwrap()
        .register_stream("fixture", "fixture-owner", "session", "branch", 0)
        .unwrap();
}

#[tokio::test]
async fn uncertain_acknowledgment_and_restart_reconcile_without_duplicate_retention() {
    let server = FaultApi::start();
    server.state.lock().unwrap().drop_ack = true;
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    let source = Source::default();
    let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    tick(&live.handle(), &source).await.unwrap();
    let operation = server.state.lock().unwrap().operation.clone().unwrap();
    assert_eq!(
        Ledger::open(dir.path()).unwrap().delivery_counts().unwrap()["failed"],
        1
    );
    drop(live);
    server.state.lock().unwrap().status = "completed".into();
    let reopened = Ledger::open(dir.path()).unwrap();
    assert_eq!(reopened.retry_pending().unwrap(), 1);
    drop(reopened);
    let restarted = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    tick(&restarted.handle(), &source).await.unwrap();
    let ledger = Ledger::open(dir.path()).unwrap();
    assert_eq!(ledger.delivery_counts().unwrap()["completed"], 1);
    assert_eq!(
        ledger.status(&operation).unwrap().as_deref(),
        Some("completed")
    );
    assert_eq!(server.state.lock().unwrap().retain_posts, 1);
    assert!(!dir.path().join("brain.db").exists());
}

#[tokio::test]
async fn extraction_outage_requires_retry_and_erasure_waits_for_retain_and_model_refresh() {
    let server = FaultApi::start();
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    let source = Source::default();
    let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    tick(&live.handle(), &source).await.unwrap();
    server.state.lock().unwrap().status = "failed".into();
    Ledger::open(dir.path()).unwrap().retry_pending().unwrap();
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().retry_posts, 1);
    source.erased.store(true, Ordering::Relaxed);
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(
        server.state.lock().unwrap().deletes,
        0,
        "in-flight retain must settle before erasure"
    );
    {
        let mut s = server.state.lock().unwrap();
        s.status = "completed".into();
        s.has_model = true;
    }
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(
        Ledger::open(dir.path()).unwrap().delivery_counts().unwrap()["erasing"],
        1
    );
    assert_eq!(server.state.lock().unwrap().refresh_posts, 1);
    // The receipt survives restart and prevents another synthesis submission.
    let restarted = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    tick(&restarted.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().refresh_posts, 1);
    server.state.lock().unwrap().refresh_status = "completed".into();
    tick(&restarted.handle(), &source).await.unwrap();
    assert_eq!(
        Ledger::open(dir.path()).unwrap().delivery_counts().unwrap()["erased"],
        1
    );
    assert_eq!(server.state.lock().unwrap().retain_posts, 1);
}

#[derive(Default)]
struct ConversationSource {
    erased: AtomicBool,
}
#[async_trait::async_trait]
impl TranscriptSource for ConversationSource {
    async fn revision(&self, _stream: &Stream) -> anyhow::Result<String> {
        Ok(if self.erased.load(Ordering::Relaxed) {
            "revision-2"
        } else {
            "revision-1"
        }
        .into())
    }
    async fn next(&self, stream: &Stream) -> anyhow::Result<Vec<SourceEvent>> {
        Ok((1..=2)
            .filter(|n| *n > stream.cursor)
            .map(|n| SourceEvent {
                sequence: n,
                occurred_at: n as i64,
                committed_turn: Some(format!("turn-{n}")),
            })
            .collect())
    }
    async fn load(&self, _stream: &Stream, terminal: &str) -> anyhow::Result<Option<Vec<Item>>> {
        if terminal == "turn-1" && self.erased.load(Ordering::Relaxed) {
            return Ok(None);
        }
        Ok(Some(vec![Item {update_mode:Some("append".into()),document_id:"zeroclaw:conversation:fixture".into(),
            content:serde_json::json!([{"role":"user","speaker":"Alex","content":terminal,"timestamp":"2026-01-01T00:00:00Z"}]).to_string(),
            timestamp:"2026-01-01T00:00:00Z".into(),context:"fixture".into(),metadata:Default::default()}]))
    }
}
#[tokio::test]
async fn conversation_appends_reconcile_lost_ack_and_rebuild_only_surviving_turns() {
    let server = FaultApi::start();
    server.state.lock().unwrap().drop_ack = true;
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    let source = ConversationSource::default();
    let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().retain_posts, 1);
    server.state.lock().unwrap().status = "completed".into();
    let ledger = Ledger::open(dir.path()).unwrap();
    ledger.retry_pending().unwrap();
    tick(&live.handle(), &source).await.unwrap();
    tick(&live.handle(), &source).await.unwrap();
    ledger.retry_pending().unwrap();
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(ledger.delivery_counts().unwrap()["completed"], 2);
    assert_eq!(server.state.lock().unwrap().retain_posts, 1);
    source.erased.store(true, Ordering::Relaxed);
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().deletes, 1);
    assert_eq!(server.state.lock().unwrap().retain_posts, 2);
    let bank = ledger
        .operation_source(server.state.lock().unwrap().operation.as_deref().unwrap())
        .unwrap()
        .0;
    assert!(ledger.rebuild_pending(&bank).unwrap());
    ledger.retry_pending().unwrap();
    // Reopen the live policy and delivery ledger during replay.
    let restarted = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    for _ in 0..3 {
        tick(&restarted.handle(), &source).await.unwrap();
    }
    assert!(
        !Ledger::open(dir.path())
            .unwrap()
            .rebuild_pending(&bank)
            .unwrap()
    );
    let s = server.state.lock().unwrap();
    assert_eq!(
        s.retained[0]["items"][0]["document_id"],
        s.retained[1]["items"][0]["document_id"]
    );
    assert_eq!(s.retained[1]["items"][0]["update_mode"], "append");
    assert_ne!(s.retained[0]["operation_id"], s.retained[1]["operation_id"]);
    assert!(
        s.retained[1]["items"][0]["content"]
            .as_str()
            .unwrap()
            .contains("turn-2")
    );
    assert!(
        !s.retained[1]["items"][0]["content"]
            .as_str()
            .unwrap()
            .contains("turn-1")
    );
}

#[tokio::test]
async fn missing_uncertain_append_is_not_proof_that_its_writer_was_cancelled() {
    let server = FaultApi::start();
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    let source = ConversationSource::default();
    let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    tick(&live.handle(), &source).await.unwrap();
    let operation = server.state.lock().unwrap().operation.take().unwrap();
    source.erased.store(true, Ordering::Relaxed);
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().deletes, 0);
    let ledger = Ledger::open(dir.path()).unwrap();
    assert_eq!(ledger.rebuilds().unwrap()[0].phase, "draining");
    assert_eq!(
        ledger.status(&operation).unwrap().as_deref(),
        Some("processing")
    );
    {
        let mut state = server.state.lock().unwrap();
        state.operation = Some(operation);
        state.status = "completed".into();
    }
    tick(&live.handle(), &source).await.unwrap();
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().deletes, 1);
}

#[tokio::test]
async fn deferred_capture_survives_restart_and_manual_drain_does_not_flood_after_lost_ack() {
    let server = FaultApi::start();
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    Ledger::open(dir.path())
        .unwrap()
        .register_stream(
            "fixture",
            "another-fixture-owner",
            "another-session",
            "branch",
            0,
        )
        .unwrap();
    let source = Source::default();
    let mut cfg = config(dir.path(), &server.url);
    cfg.memory.hindsight.retention.deferred = true;
    let live = zeroclaw_config::live::LiveConfig::new(cfg.clone());
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().retain_posts, 0);
    assert_eq!(
        Ledger::open(dir.path()).unwrap().delivery_counts().unwrap()["pending"],
        2
    );
    drop(live);
    let ledger = Ledger::open(dir.path()).unwrap();
    ledger
        .set_retention_control(&super::super::retention::Control::Draining {
            until: chrono::Utc::now().timestamp() + 60,
        })
        .unwrap();
    drop(ledger);
    let restarted = zeroclaw_config::live::LiveConfig::new(cfg);
    server.state.lock().unwrap().drop_ack = true;
    tick(&restarted.handle(), &source).await.unwrap();
    assert_eq!(
        server.state.lock().unwrap().retain_posts,
        1,
        "unknown acknowledgement must hold the single-operation claim"
    );
    Ledger::open(dir.path()).unwrap().retry_pending().unwrap();
    tick(&restarted.handle(), &source).await.unwrap();
    assert_eq!(
        server.state.lock().unwrap().retain_posts,
        1,
        "poll the original UUID rather than starting another job"
    );
    server.state.lock().unwrap().status = "completed".into();
    Ledger::open(dir.path()).unwrap().retry_pending().unwrap();
    tick(&restarted.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().retain_posts, 2);
    let ledger = Ledger::open(dir.path()).unwrap();
    ledger
        .set_retention_control(&super::super::retention::Control::Paused)
        .unwrap();
    // Expire the normal delivery backoff without lifting admission's pause.
    ledger.retry_pending().unwrap();
    tick(&restarted.handle(), &source).await.unwrap();
    assert_eq!(
        ledger.delivery_counts().unwrap()["completed"],
        2,
        "accepted work still reconciles while paused"
    );
    assert_eq!(server.state.lock().unwrap().retain_posts, 2);
}

#[tokio::test]
async fn deferred_explicit_note_returns_without_extraction_and_existing_recall_remains_available() {
    let server = FaultApi::start();
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), &server.url);
    cfg.memory.hindsight.retention.deferred = true;
    let live = zeroclaw_config::live::LiveConfig::new(cfg.clone());
    let memory = HindsightMemory::new(
        &cfg.memory.hindsight,
        dir.path(),
        Some("fixture"),
        Some(live.handle()),
    )
    .unwrap();
    let scope = PrincipalScope::new("fixture-owner").with_agent(Some("fixture".into()));
    memory
        .retain_scoped(
            &scope,
            "fixture-note",
            "The fictional Larch project uses SQLite.",
            zeroclaw_api::memory_traits::MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
    assert_eq!(server.state.lock().unwrap().retain_posts, 0);
    let facts = memory
        .recall_scoped(
            &scope,
            "Which database?",
            5,
            (None, None),
            zeroclaw_api::memory_traits::ContextRecallMode::Standard,
        )
        .await
        .unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(server.state.lock().unwrap().recall_posts, 1);
    assert_eq!(server.state.lock().unwrap().retain_posts, 0);
    let ledger = Ledger::open(dir.path()).unwrap();
    assert_eq!(ledger.diagnostics().unwrap()["pending_notes"], 1);
    ledger
        .set_retention_control(&super::super::retention::Control::Draining {
            until: chrono::Utc::now().timestamp() + 60,
        })
        .unwrap();
    tick(&live.handle(), &Source::default()).await.unwrap();
    assert_eq!(server.state.lock().unwrap().retain_posts, 1);
}

#[tokio::test]
async fn source_erased_while_deferred_is_never_submitted_by_a_later_manual_drain() {
    let server = FaultApi::start();
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    let source = Source::default();
    let mut cfg = config(dir.path(), &server.url);
    cfg.memory.hindsight.retention.deferred = true;
    let live = zeroclaw_config::live::LiveConfig::new(cfg);
    tick(&live.handle(), &source).await.unwrap();
    source.erased.store(true, Ordering::Relaxed);
    Ledger::open(dir.path())
        .unwrap()
        .set_retention_control(&super::super::retention::Control::Draining {
            until: chrono::Utc::now().timestamp() + 60,
        })
        .unwrap();
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().retain_posts, 0);
    assert_eq!(
        Ledger::open(dir.path()).unwrap().delivery_counts().unwrap()["erased"],
        1
    );
}

struct BatchSource {
    turns: usize,
    padding: usize,
    erased: std::sync::atomic::AtomicUsize,
}
impl BatchSource {
    fn new(turns: usize, padding: usize) -> Self {
        Self {
            turns,
            padding,
            erased: std::sync::atomic::AtomicUsize::new(usize::MAX),
        }
    }
}
#[async_trait::async_trait]
impl TranscriptSource for BatchSource {
    async fn revision(&self, _: &Stream) -> anyhow::Result<String> {
        Ok(format!("frontier-{}", self.erased.load(Ordering::Relaxed)))
    }
    async fn next(&self, stream: &Stream) -> anyhow::Result<Vec<SourceEvent>> {
        Ok((stream.cursor as usize..self.turns)
            .map(|i| SourceEvent {
                sequence: (i + 1) as u64,
                occurred_at: i as i64,
                committed_turn: Some(format!("turn-{i}")),
            })
            .collect())
    }
    async fn load(&self, _: &Stream, terminal: &str) -> anyhow::Result<Option<Vec<Item>>> {
        let index: usize = terminal.strip_prefix("turn-").unwrap().parse()?;
        if index == self.erased.load(Ordering::Relaxed) {
            return Ok(None);
        }
        Ok(Some(vec![Item {
            content:serde_json::json!([{"role":"user","speaker":"Alex","timestamp":format!("2026-01-01T00:{index:02}:00Z"),"content":format!("message-{index} {}", "x".repeat(self.padding))}]).to_string(),
            document_id:"zeroclaw:conversation:fixture".into(),update_mode:Some("append".into()),timestamp:format!("2026-01-01T00:{index:02}:00Z"),context:"Conversation between Alex (the user) and Nova (the assistant).".into(),metadata:Default::default(),
        }]))
    }
}
#[tokio::test]
async fn queued_turns_share_one_ordered_append_and_restart_keeps_membership() {
    let server = FaultApi::start();
    server.state.lock().unwrap().drop_ack = true;
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    let source = BatchSource::new(3, 0);
    let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    tick(&live.handle(), &source).await.unwrap();
    let op = server.state.lock().unwrap().operation.clone().unwrap();
    let ledger = Ledger::open(dir.path()).unwrap();
    let leader = ledger.pending(1, true).unwrap();
    // The failed leader is backed off until explicit retry, but all sources
    // were already bound atomically before the lost HTTP acknowledgement.
    assert!(leader.is_empty());
    assert_eq!(
        ledger
            .draining_operations(&ledger.streams().unwrap()[0].id)
            .unwrap(),
        vec![op.clone()]
    );
    ledger.retry_pending().unwrap();
    let leader = ledger.pending(1, true).unwrap().remove(0);
    assert_eq!(
        ledger
            .batch_candidates(&leader, MAX_APPEND_TURNS)
            .unwrap()
            .len(),
        3
    );
    drop(ledger);
    drop(live);
    let restarted = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    server.state.lock().unwrap().status = "completed".into();
    tick(&restarted.handle(), &source).await.unwrap();
    assert_eq!(
        Ledger::open(dir.path()).unwrap().delivery_counts().unwrap()["completed"],
        3
    );
    let s = server.state.lock().unwrap();
    assert_eq!(s.retain_posts, 1);
    let items = s.retained[0]["items"].as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["update_mode"], "append");
    assert_eq!(items[0]["timestamp"], "2026-01-01T00:02:00Z");
    let text = items[0]["content"].as_str().unwrap();
    assert!(
        text.find("message-0").unwrap() < text.find("message-1").unwrap()
            && text.find("message-1").unwrap() < text.find("message-2").unwrap()
    );
    assert!(text.starts_with("Alex (2026-01-01T00:00:00Z):\n"));
    assert!(!text.contains("turn_id"));
}
#[tokio::test]
async fn append_batches_bound_turn_count_and_text_size_without_truncating_turns() {
    for (turns, padding, expected) in [
        (MAX_APPEND_TURNS + 2, 0, MAX_APPEND_TURNS),
        (3, MAX_APPEND_BYTES / 2, 1),
        (2, MAX_APPEND_BYTES + 10, 1),
    ] {
        let server = FaultApi::start();
        let dir = tempfile::tempdir().unwrap();
        register(dir.path());
        let source = BatchSource::new(turns, padding);
        let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
        tick(&live.handle(), &source).await.unwrap();
        let s = server.state.lock().unwrap();
        assert_eq!(s.retain_posts, 1);
        let text = s.retained[0]["items"][0]["content"].as_str().unwrap();
        assert_eq!(text.matches("Alex (").count(), expected);
        assert_eq!(text.matches('x').count(), expected * (padding + 1)); // speaker Alex contributes one x
    }
}
#[tokio::test]
async fn erasing_one_inflight_batch_member_fences_and_rebuilds_the_document() {
    let server = FaultApi::start();
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    let source = BatchSource::new(3, 0);
    let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    tick(&live.handle(), &source).await.unwrap();
    source.erased.store(1, Ordering::Relaxed);
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().deletes, 0);
    let ledger = Ledger::open(dir.path()).unwrap();
    assert!(!ledger.rebuilds().unwrap().is_empty());
    server.state.lock().unwrap().status = "completed".into();
    tick(&live.handle(), &source).await.unwrap();
    tick(&live.handle(), &source).await.unwrap();
    let s = server.state.lock().unwrap();
    assert_eq!(s.deletes, 1);
    assert_eq!(s.retain_posts, 2);
    let replay = s.retained[1]["items"][0]["content"].as_str().unwrap();
    assert!(
        replay.contains("message-0")
            && replay.contains("message-2")
            && !replay.contains("message-1")
    );
}

#[tokio::test]
async fn deferred_backlog_is_coalesced_only_after_manual_drain() {
    let server = FaultApi::start();
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    let source = BatchSource::new(4, 0);
    let mut cfg = config(dir.path(), &server.url);
    cfg.memory.hindsight.retention.deferred = true;
    let ledger = Ledger::open(dir.path()).unwrap();
    ledger
        .set_retention_control(&super::super::retention::Control::Paused)
        .unwrap();
    let live = zeroclaw_config::live::LiveConfig::new(cfg);
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().retain_posts, 0);
    assert_eq!(ledger.delivery_counts().unwrap()["pending"], 4);
    ledger
        .set_retention_control(&super::super::retention::Control::Draining {
            until: chrono::Utc::now().timestamp() + 60,
        })
        .unwrap();
    tick(&live.handle(), &source).await.unwrap();
    let state = server.state.lock().unwrap();
    assert_eq!(state.retain_posts, 1);
    assert_eq!(
        state.retained[0]["items"][0]["content"]
            .as_str()
            .unwrap()
            .matches("Alex (")
            .count(),
        4
    );
}

#[tokio::test]
async fn missing_batch_receipt_reuses_membership_without_absorbing_new_turns() {
    let server = FaultApi::start();
    server.state.lock().unwrap().drop_ack = true;
    let dir = tempfile::tempdir().unwrap();
    register(dir.path());
    let source = BatchSource::new(3, 0);
    let live = zeroclaw_config::live::LiveConfig::new(config(dir.path(), &server.url));
    tick(&live.handle(), &source).await.unwrap();
    let original = server.state.lock().unwrap().operation.clone().unwrap();
    let ledger = Ledger::open(dir.path()).unwrap();
    let stream = ledger.streams().unwrap().remove(0);
    ledger.advance(&stream, 4, Some("turn-3"), 16).unwrap();
    server.state.lock().unwrap().operation = None; // No accepted operation can be found upstream.
    ledger.retry_pending().unwrap();
    tick(&live.handle(), &source).await.unwrap();
    let state = server.state.lock().unwrap();
    assert_eq!(state.retain_posts, 2);
    for request in &state.retained {
        assert_eq!(request["operation_id"], original);
        let text = request["items"][0]["content"].as_str().unwrap();
        assert_eq!(text.matches("Alex (").count(), 3);
        assert!(!text.contains("message-3"));
    }
}
