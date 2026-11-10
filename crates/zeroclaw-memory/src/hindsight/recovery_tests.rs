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
struct ServerState {
    drop_ack: bool,
    retain_posts: usize,
    retained: Vec<serde_json::Value>,
    retry_posts: usize,
    deletes: usize,
    operation: Option<String>,
    status: String,
    has_model: bool,
    refresh_posts: usize,
    refresh_status: String,
}
struct FaultApi {
    url: String,
    state: Arc<Mutex<ServerState>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl FaultApi {
    fn start() -> Self {
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
fn config(directory: &std::path::Path, url: &str) -> zeroclaw_config::schema::Config {
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
    assert_eq!(server.state.lock().unwrap().retain_posts, 2);
    source.erased.store(true, Ordering::Relaxed);
    tick(&live.handle(), &source).await.unwrap();
    assert_eq!(server.state.lock().unwrap().deletes, 1);
    assert_eq!(server.state.lock().unwrap().retain_posts, 3);
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
        s.retained[2]["items"][0]["document_id"]
    );
    assert_eq!(s.retained[2]["items"][0]["update_mode"], "append");
    assert_ne!(s.retained[1]["operation_id"], s.retained[2]["operation_id"]);
    assert!(
        s.retained[2]["items"][0]["content"]
            .as_str()
            .unwrap()
            .contains("turn-2")
    );
    assert!(
        !s.retained[2]["items"][0]["content"]
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
