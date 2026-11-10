//! Bounded Hindsight 1.2 HTTP transport. Contains no process lifecycle or bank-selection policy.
use reqwest::{Client as HttpClient, Method, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use zeroclaw_config::hindsight::HindsightConfig;

pub const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Unavailable,
    Timeout,
    Unauthorized,
    Rejected,
    Protocol,
    TooLarge,
    CircuitOpen,
}
#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub status: Option<u16>,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "hindsight_{:?}{}",
            self.kind,
            self.status.map(|s| format!("_{s}")).unwrap_or_default()
        )
    }
}
impl std::error::Error for Error {}
impl Error {
    pub fn transient(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::Unavailable | ErrorKind::Timeout | ErrorKind::CircuitOpen
        ) || self.status == Some(429)
    }
}

#[derive(Default)]
struct Circuit {
    failures: u32,
    until: Option<Instant>,
}
#[derive(Clone)]
pub struct Client {
    http: HttpClient,
    origin: Url,
    // Immutable transport connection material, not permissions or bank policy.
    token: Option<String>,
    timeout: Duration,
    reflect_timeout: Duration,
    // Creates the transport's observed failure/cooldown facts, never memory content.
    circuit: Arc<Mutex<Circuit>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Item {
    pub content: String,
    pub document_id: String,
    /// Native append is reserved for ordered canonical conversation delivery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update_mode: Option<String>,
    pub timestamp: String,
    pub context: String,
    pub metadata: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Fact {
    pub id: String,
    pub text: String,
    #[serde(default, rename = "type", alias = "fact_type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub mentioned_at: Option<String>,
    #[serde(default)]
    pub occurred_start: Option<String>,
    #[serde(default)]
    pub document_id: Option<String>,
    #[serde(default)]
    pub metadata: Option<BTreeMap<String, String>>,
}
#[derive(Debug, Deserialize)]
pub struct Recall {
    pub results: Vec<Fact>,
}
#[derive(Debug, Deserialize)]
pub struct Operation {
    pub id: Option<String>,
    pub operation_id: Option<String>,
    pub status: String,
}

impl Client {
    pub fn new(config: &HindsightConfig) -> anyhow::Result<Self> {
        config.validate()?;
        Ok(Self {
            http: HttpClient::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(5))
                .build()?,
            origin: zeroclaw_config::hindsight::validate_url(&config.api_url)?,
            token: config.api_key.clone(),
            timeout: Duration::from_secs(config.request_timeout_secs),
            reflect_timeout: Duration::from_secs(config.reflect_timeout_secs),
            circuit: Arc::new(Mutex::new(Circuit::default())),
        })
    }

    pub fn matches_transport(&self, config: &HindsightConfig) -> bool {
        zeroclaw_config::hindsight::validate_url(&config.api_url)
            .is_ok_and(|url| url == self.origin)
            && self.token == config.api_key
            && self.timeout == Duration::from_secs(config.request_timeout_secs)
            && self.reflect_timeout == Duration::from_secs(config.reflect_timeout_secs)
    }
    pub async fn request(
        &self,
        method: Method,
        segments: &[&str],
        body: Option<&Value>,
        reflective: bool,
    ) -> Result<Value, Error> {
        self.request_query(method, segments, body, reflective, &[])
            .await
    }
    pub async fn request_query(
        &self,
        method: Method,
        segments: &[&str],
        body: Option<&Value>,
        reflective: bool,
        query: &[(&str, String)],
    ) -> Result<Value, Error> {
        {
            let state = self.circuit.lock().map_err(|_| Error {
                kind: ErrorKind::Unavailable,
                status: None,
            })?;
            if state.until.is_some_and(|until| until > Instant::now()) {
                return Err(Error {
                    kind: ErrorKind::CircuitOpen,
                    status: None,
                });
            }
        }
        let result = self.send(method, segments, body, reflective, query).await;
        let mut state = self.circuit.lock().map_err(|_| Error {
            kind: ErrorKind::Unavailable,
            status: None,
        })?;
        match &result {
            Ok(_) => {
                state.failures = 0;
                state.until = None;
            }
            Err(error) if error.transient() => {
                state.failures = state.failures.saturating_add(1);
                if state.failures >= 5 {
                    state.until = Some(
                        Instant::now()
                            + Duration::from_secs(
                                (30u64.saturating_mul(
                                    1 << ((state.failures / 5).saturating_sub(1).min(4)),
                                ))
                                .min(300),
                            ),
                    );
                }
            }
            Err(_) => {}
        }
        result
    }

    async fn send(
        &self,
        method: Method,
        segments: &[&str],
        body: Option<&Value>,
        reflective: bool,
        query: &[(&str, String)],
    ) -> Result<Value, Error> {
        if segments
            .iter()
            .any(|segment| matches!(*segment, "." | ".."))
        {
            return Err(Error {
                kind: ErrorKind::Rejected,
                status: None,
            });
        }
        let mut url = self.origin.clone();
        url.path_segments_mut()
            .map_err(|_| Error {
                kind: ErrorKind::Protocol,
                status: None,
            })?
            .clear()
            .extend(segments);
        let mut request = self
            .http
            .request(method, url)
            .query(query)
            .timeout(if reflective {
                self.reflect_timeout
            } else {
                self.timeout
            });
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        if let Some(body) = body {
            let bytes = serde_json::to_vec(body).map_err(|_| Error {
                kind: ErrorKind::Protocol,
                status: None,
            })?;
            if bytes.len() > MAX_REQUEST_BYTES {
                return Err(Error {
                    kind: ErrorKind::TooLarge,
                    status: None,
                });
            }
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(bytes);
        }
        let map_transport = |error: reqwest::Error| Error {
            kind: if error.is_timeout() {
                ErrorKind::Timeout
            } else {
                ErrorKind::Unavailable
            },
            status: None,
        };
        let mut response = request.send().await.map_err(map_transport)?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error {
                kind: if matches!(status.as_u16(), 401 | 403) {
                    ErrorKind::Unauthorized
                } else if status.is_server_error() || status.as_u16() == 429 {
                    ErrorKind::Unavailable
                } else {
                    ErrorKind::Rejected
                },
                status: Some(status.as_u16()),
            });
        }
        if response
            .content_length()
            .is_some_and(|len| len > MAX_RESPONSE_BYTES as u64)
        {
            return Err(Error {
                kind: ErrorKind::TooLarge,
                status: None,
            });
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(map_transport)? {
            if chunk.len() > MAX_RESPONSE_BYTES.saturating_sub(bytes.len()) {
                return Err(Error {
                    kind: ErrorKind::TooLarge,
                    status: None,
                });
            }
            bytes.extend_from_slice(&chunk);
        }
        if bytes.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&bytes).map_err(|_| Error {
            kind: ErrorKind::Protocol,
            status: None,
        })
    }
    pub async fn health(&self) -> Result<Value, Error> {
        self.request(Method::GET, &["health"], None, false).await
    }
    pub async fn retain(
        &self,
        bank: &str,
        items: &[Item],
        operation: &str,
    ) -> Result<Value, Error> {
        let body = serde_json::json!({"items":items,"async":true,"operation_id":operation});
        self.request(
            Method::POST,
            &["v1", "default", "banks", bank, "memories"],
            Some(&body),
            false,
        )
        .await
    }
    pub async fn recall(
        &self,
        bank: &str,
        query: &str,
        budget: &str,
        tokens: usize,
        observations: bool,
    ) -> Result<Recall, Error> {
        let body = serde_json::json!({"query":query,"budget":budget,"max_tokens":tokens,"prefer_observations":observations});
        let value = self
            .request(
                Method::POST,
                &["v1", "default", "banks", bank, "memories", "recall"],
                Some(&body),
                false,
            )
            .await?;
        serde_json::from_value(value).map_err(|_| Error {
            kind: ErrorKind::Protocol,
            status: None,
        })
    }
    pub async fn reflect(
        &self,
        bank: &str,
        query: &str,
        budget: &str,
        tokens: usize,
    ) -> Result<Value, Error> {
        let body = serde_json::json!({"query":query,"budget":budget,"max_tokens":tokens,"include":{"facts":{}}});
        self.request(
            Method::POST,
            &["v1", "default", "banks", bank, "reflect"],
            Some(&body),
            true,
        )
        .await
    }
    pub async fn operation(&self, bank: &str, operation: &str) -> Result<Operation, Error> {
        let value = self
            .request(
                Method::GET,
                &["v1", "default", "banks", bank, "operations", operation],
                None,
                false,
            )
            .await?;
        serde_json::from_value(value).map_err(|_| Error {
            kind: ErrorKind::Protocol,
            status: None,
        })
    }
    pub async fn ensure_bank(&self, bank: &str) -> Result<Value, Error> {
        self.request(
            Method::PUT,
            &["v1", "default", "banks", bank],
            Some(&serde_json::json!({})),
            false,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    async fn fixture(response: Vec<u8>) -> (Client, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = HindsightConfig {
            api_url: format!("http://{}", listener.local_addr().unwrap()),
            ..Default::default()
        };
        let server = zeroclaw_spawn::spawn!(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0; 16384];
            let _ = socket.read(&mut request).await.unwrap();
            socket.write_all(&response).await.unwrap();
        });
        (Client::new(&config).unwrap(), server)
    }
    #[tokio::test]
    async fn errors_are_typed_and_never_expose_private_response_bodies() {
        let (client,server)=fixture(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 18\r\nConnection: close\r\n\r\nprivate-key-secret".to_vec()).await;
        let error = client.health().await.unwrap_err();
        assert_eq!(error.kind, ErrorKind::Unauthorized);
        assert!(!error.to_string().contains("private-key"));
        server.await.unwrap();
    }
    #[tokio::test]
    async fn declared_response_size_is_bounded() {
        let (client, server) = fixture(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                MAX_RESPONSE_BYTES + 1
            )
            .into_bytes(),
        )
        .await;
        assert_eq!(client.health().await.unwrap_err().kind, ErrorKind::TooLarge);
        server.await.unwrap();
    }
    #[tokio::test]
    async fn malformed_json_is_an_error_not_an_empty_recall() {
        let (client, server) = fixture(
            b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\noops".to_vec(),
        )
        .await;
        assert_eq!(client.health().await.unwrap_err().kind, ErrorKind::Protocol);
        server.await.unwrap();
    }
    #[tokio::test]
    async fn redirect_is_not_followed() {
        let (client,server)=fixture(b"HTTP/1.1 302 Found\r\nLocation: http://example.com/private\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()).await;
        assert_eq!(client.health().await.unwrap_err().status, Some(302));
        server.await.unwrap();
    }
    #[tokio::test]
    async fn chunked_response_without_content_length_is_bounded() {
        let size = MAX_RESPONSE_BYTES + 1;
        let mut response = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{size:x}\r\n"
        )
        .into_bytes();
        response.resize(response.len() + size, b'x');
        response.extend_from_slice(b"\r\n0\r\n\r\n");
        let (client, server) = fixture(response).await;
        assert_eq!(client.health().await.unwrap_err().kind, ErrorKind::TooLarge);
        server.await.unwrap();
    }
    #[tokio::test]
    async fn circuit_stops_requests_after_five_transient_failures() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = HindsightConfig {
            api_url: format!("http://{}", listener.local_addr().unwrap()),
            ..Default::default()
        };
        let server = zeroclaw_spawn::spawn!(async move {
            for _ in 0..5 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = [0; 4096];
                let received = socket.read(&mut bytes).await.unwrap();
                assert!(received > 0, "fixture requires a request before responding");
                socket.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await.unwrap();
            }
        });
        let client = Client::new(&config).unwrap();
        for _ in 0..5 {
            assert_eq!(
                client.health().await.unwrap_err().kind,
                ErrorKind::Unavailable
            );
        }
        assert_eq!(
            client.health().await.unwrap_err().kind,
            ErrorKind::CircuitOpen
        );
        server.await.unwrap();
    }
}
