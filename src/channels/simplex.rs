//! SimpleX Chat channel implementation for ZeroClaw.
//!
//! This module provides native SimpleX Chat integration via the simplex-chat CLI daemon.
//! It features auto-download of the simplex-chat binary and automatic subprocess management.

use super::traits::{Channel, ChannelMessage, SendMessage};
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
#[cfg(unix)]
use std::fs::Permissions;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use tokio::fs;
use tokio::process::{Child, Command as AsyncCommand};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message as WsMessage};
use tracing::{debug, error, info, warn};

/// SimpleX's maximum message length (approximate, actual limit varies)
const SIMPLEX_MAX_MESSAGE_LENGTH: usize = 16000;
const SIMPLEX_ACK_REACTIONS: &[&str] = &["⚡", "👌", "👀", "🔥", "👍"];

// ============================================================================
// SimpleX Protocol Types
// ============================================================================

/// SimpleX Chat API request
#[derive(Debug, Serialize)]
struct SimplexRequest {
    /// JSON-RPC version (always "2.0")
    jsonrpc: String,
    /// Request correlation ID
    id: u64,
    /// Method name
    method: String,
    /// Method parameters
    params: serde_json::Value,
}

/// SimpleX Chat API response
#[derive(Debug, Deserialize)]
struct SimplexResponse {
    /// JSON-RPC version
    jsonrpc: Option<String>,
    /// Correlation ID (matches request)
    id: Option<u64>,
    /// Result (if successful)
    result: Option<serde_json::Value>,
    /// Error (if failed)
    error: Option<SimplexError>,
}

/// SimpleX error response
#[derive(Debug, Deserialize)]
struct SimplexError {
    code: i32,
    message: String,
}

/// SimpleX Chat event (received from daemon)
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum SimplexEvent {
    /// New message received
    #[serde(rename = "newChatItem")]
    NewChatItem { chatItem: ChatItem },
    
    /// Contact request received
    #[serde(rename = "newContactRequest")]
    NewContactRequest { contactRequest: ContactRequest },
    
    /// Contact connected
    #[serde(rename = "contactConnection")]
    ContactConnected { contact: Contact },
    
    /// Contact disconnected
    #[serde(rename = "contactDisconnected")]
    ContactDisconnected { contact: Contact },
    
    /// Other events (we'll log but ignore for now)
    #[serde(other)]
    Other,
}

/// Chat item (message)
#[derive(Debug, Deserialize)]
struct ChatItem {
    /// Chat info (contact or group)
    chatInfo: ChatInfo,
    /// Message content
    content: ChatContent,
    /// Message ID
    itemId: String,
}

/// Chat info (contact or group)
#[derive(Debug, Deserialize)]
struct ChatInfo {
    #[serde(rename = "type")]
    chat_type: String,
    contact: Option<Contact>,
}

/// Chat content
#[derive(Debug, Deserialize)]
struct ChatContent {
    #[serde(rename = "type")]
    content_type: String,
    msgContent: Option<MessageContent>,
}

/// Message content
#[derive(Debug, Deserialize)]
struct MessageContent {
    text: String,
}

/// Contact information
#[derive(Debug, Deserialize)]
struct Contact {
    contactId: String,
    localDisplayName: Option<String>,
}

/// Contact request
#[derive(Debug, Deserialize)]
struct ContactRequest {
    contactRequestId: String,
}

/// SimpleX Chat channel configuration
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SimplexConfig {
    /// WebSocket URL for simplex-chat daemon (default: ws://127.0.0.1:5225)
    #[serde(default = "default_websocket_url")]
    pub websocket_url: String,

    /// Bot display name in SimpleX
    pub bot_display_name: Option<String>,

    /// Allow file attachments from users
    #[serde(default)]
    pub allow_files: bool,

    /// ACK reaction enable/disable
    #[serde(default = "default_ack_enabled")]
    pub ack_enabled: bool,

    /// Auto-download simplex-chat binary if not found
    #[serde(default = "default_auto_download")]
    pub auto_download: bool,

    /// Custom path to simplex-chat binary (optional)
    pub binary_path: Option<String>,
}

fn default_websocket_url() -> String {
    "ws://127.0.0.1:5225".to_string()
}

fn default_ack_enabled() -> bool {
    true
}

fn default_auto_download() -> bool {
    true
}

impl Default for SimplexConfig {
    fn default() -> Self {
        Self {
            websocket_url: default_websocket_url(),
            bot_display_name: None,
            allow_files: false,
            ack_enabled: true,
            auto_download: true,
            binary_path: None,
        }
    }
}

/// SimpleX Chat channel
pub struct SimplexChannel {
    config: SimplexConfig,
    workspace_dir: PathBuf,
    daemon_process: Arc<Mutex<Option<Child>>>,
    ack_enabled: bool,
    /// WebSocket message sender (for sending messages to daemon)
    ws_sender: Arc<Mutex<Option<futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
        WsMessage,
    >>>>,
    /// Request ID counter for JSON-RPC
    request_id: Arc<Mutex<u64>>,
}

impl SimplexChannel {
    /// Create a new SimpleX channel
    pub fn new(config: SimplexConfig) -> Self {
        Self {
            config,
            workspace_dir: PathBuf::new(),
            daemon_process: Arc::new(Mutex::new(None)),
            ack_enabled: true,
            ws_sender: Arc::new(Mutex::new(None)),
            request_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Set the workspace directory
    pub fn with_workspace_dir(mut self, dir: PathBuf) -> Self {
        self.workspace_dir = dir;
        self
    }

    /// Set ACK reaction enabled/disabled
    pub fn with_ack_enabled(mut self, enabled: bool) -> Self {
        self.ack_enabled = enabled;
        self
    }

    /// Generate next request ID
    async fn next_request_id(&self) -> u64 {
        let mut id = self.request_id.lock().await;
        *id += 1;
        *id
    }

    /// Get the path to the simplex-chat binary
    fn get_binary_path(&self) -> PathBuf {
        if let Some(ref custom_path) = self.config.binary_path {
            PathBuf::from(custom_path)
        } else {
            let data = dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("zeroclaw")
                .join("bin");
            data.join("simplex-chat")
        }
    }

    /// Check if simplex-chat binary exists
    async fn binary_exists(&self) -> bool {
        let binary_path = self.get_binary_path();
        binary_path.exists()
    }

    /// Download simplex-chat binary from GitHub releases
    #[cfg(target_os = "linux")]
    async fn download_binary(&self) -> Result<PathBuf> {
        let binary_path = self.get_binary_path();
        
        // Create directory if needed
        if let Some(parent) = binary_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Determine architecture
        let arch = if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else {
            anyhow::bail!("Unsupported architecture for SimpleX binary");
        };

        // Download URL (simplified - actual URL structure may vary)
        let download_url = format!(
            "https://github.com/simplex-chat/simplex-chat/releases/latest/download/simplex-chat-{}-{}",
            std::env::consts::OS,
            arch
        );

        info!("Downloading simplex-chat binary from {}", download_url);

        // Use curl or wget to download
        let status = Command::new("curl")
            .args(["-L", "-o", &binary_path.display().to_string(), &download_url])
            .status()
            .context("Failed to download simplex-chat binary")?;

        if !status.success() {
            anyhow::bail!("Failed to download simplex-chat binary (curl exit code: {})", status);
        }

        // Make executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            use std::fs::Permissions;
            fs::set_permissions(&binary_path, Permissions::from_mode(0o755)).await?;
        }

        info!("Successfully downloaded simplex-chat binary to {}", binary_path.display());
        Ok(binary_path)
    }

    #[cfg(not(target_os = "linux"))]
    async fn download_binary(&self) -> Result<PathBuf> {
        anyhow::bail!(
            "Auto-download of simplex-chat binary is only supported on Linux. \
             Please install simplex-chat manually or set binary_path in config."
        );
    }

    /// Ensure simplex-chat binary is available (download if needed)
    async fn ensure_binary(&self) -> Result<PathBuf> {
        if self.binary_exists().await {
            return Ok(self.get_binary_path());
        }

        if self.config.auto_download {
            self.download_binary().await
        } else {
            anyhow::bail!(
                "simplex-chat binary not found at {} and auto_download is disabled",
                self.get_binary_path().display()
            );
        }
    }

    /// Start the simplex-chat daemon
    async fn start_daemon(&self) -> Result<()> {
        let binary_path = self.ensure_binary().await?;
        
        // Build command arguments
        let mut args = vec![
            "-p".to_string(),
            "5225".to_string(),
        ];

        if let Some(ref display_name) = self.config.bot_display_name {
            args.push("--create-bot-display-name".to_string());
            args.push(display_name.clone());
        }

        if self.config.allow_files {
            args.push("--create-bot-allow-files".to_string());
        }

        info!("Starting simplex-chat daemon: {} {}", binary_path.display(), args.join(" "));

        // Start the daemon process
        let mut child = AsyncCommand::new(&binary_path)
            .args(&args)
            .spawn()
            .context("Failed to start simplex-chat daemon")?;

        let pid = child.id().unwrap_or(0);
        info!("SimpleX daemon started with PID {}", pid);

        // Wait a bit for daemon to initialize
        sleep(Duration::from_secs(2)).await;

        // Store the process handle
        let mut daemon = self.daemon_process.lock().await;
        *daemon = Some(child);

        Ok(())
    }

    /// Stop the simplex-chat daemon
    pub async fn stop_daemon(&self) -> Result<()> {
        let mut daemon = self.daemon_process.lock().await;
        if let Some(ref mut child) = *daemon {
            info!("Stopping simplex-chat daemon");
            child.kill().await.context("Failed to kill simplex-chat daemon")?;
            *daemon = None;
        }
        Ok(())
    }

    /// Split message for SimpleX's character limit
    fn split_message(&self, message: &str) -> Vec<String> {
        if message.chars().count() <= SIMPLEX_MAX_MESSAGE_LENGTH {
            return vec![message.to_string()];
        }

        let mut chunks = Vec::new();
        let mut remaining = message;

        while !remaining.is_empty() {
            let chunk_size = SIMPLEX_MAX_MESSAGE_LENGTH.min(remaining.chars().count());
            let end_byte = remaining
                .char_indices()
                .nth(chunk_size)
                .map(|(idx, _)| idx)
                .unwrap_or(remaining.len());

            chunks.push(remaining[..end_byte].to_string());
            remaining = &remaining[end_byte..];
        }

        chunks
    }

    /// Handle incoming SimpleX event
    async fn handle_event(
        &self,
        event: SimplexEvent,
        sender: &tokio::sync::mpsc::Sender<ChannelMessage>,
    ) -> Result<()> {
        match event {
            SimplexEvent::NewChatItem { chatItem } => {
                // Extract message content
                if let Some(content) = chatItem.content.msgContent {
                    let text = content.text;
                    
                    // Get contact info
                    let contact_name = chatItem.chatInfo.contact
                        .as_ref()
                        .and_then(|c| c.localDisplayName.clone())
                        .unwrap_or_else(|| "Unknown".to_string());
                    
                    let contact_id = chatItem.chatInfo.contact
                        .as_ref()
                        .map(|c| c.contactId.clone())
                        .unwrap_or_default();

                    info!("Message from {}: {}", contact_name, text);

                    // Send ACK reaction if enabled
                    if self.ack_enabled && self.config.ack_enabled {
                        // Note: ACK reactions would require implementing the sendMessage API
                        // For now, we'll skip this
                        debug!("Would send ACK reaction for message from {}", contact_name);
                    }

                    // Forward to ZeroClaw core
                    let channel_msg = ChannelMessage {
                        id: chatItem.itemId.clone(),
                        sender: contact_name.clone(),
                        reply_target: contact_id.clone(),
                        content: text.clone(),
                        channel: "simplex".to_string(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        thread_ts: None,
                    };

                    sender.send(channel_msg).await?;
                }
            }
            SimplexEvent::NewContactRequest { contactRequest } => {
                info!("New contact request: {}", contactRequest.contactRequestId);
                // TODO: Auto-accept contact requests based on config
            }
            SimplexEvent::ContactConnected { contact } => {
                let name = contact.localDisplayName.unwrap_or_default();
                info!("Contact connected: {} ({})", name, contact.contactId);
            }
            SimplexEvent::ContactDisconnected { contact } => {
                let name = contact.localDisplayName.unwrap_or_default();
                info!("Contact disconnected: {} ({})", name, contact.contactId);
            }
            SimplexEvent::Other => {
                // Ignore unknown events
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Channel for SimplexChannel {
    fn name(&self) -> &str {
        "simplex"
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> Result<()> {
        // Ensure binary is available
        let binary_path = self.ensure_binary().await?;
        
        // Start daemon if not running
        self.start_daemon().await?;

        info!("SimpleX channel started successfully");
        info!("Binary path: {}", binary_path.display());
        info!("WebSocket URL: {}", self.config.websocket_url);

        // Connect to simplex-chat daemon via WebSocket
        info!("Connecting to SimpleX daemon at {}", self.config.websocket_url);
        
        let (ws_stream, _) = connect_async(&self.config.websocket_url)
            .await
            .context("Failed to connect to SimpleX daemon via WebSocket")?;
        
        info!("WebSocket connection established");

        // Split the WebSocket into sender and receiver
        let (ws_sender, mut ws_receiver) = ws_stream.split();

        // Store the sender for use in send() method (move, not clone - SplitSink doesn't implement Clone)
        {
            let mut sender_guard = self.ws_sender.lock().await;
            *sender_guard = Some(ws_sender);
        }

        // Send API readiness check using the shared sender
        let api_ready = SimplexRequest {
            jsonrpc: "2.0".to_string(),
            id: self.next_request_id().await,
            method: "apiReady".to_string(),
            params: serde_json::json!({}),
        };
        
        {
            let mut sender_guard = self.ws_sender.lock().await;
            if let Some(ref mut ws_sender) = *sender_guard {
                ws_sender
                    .send(WsMessage::Text(serde_json::to_string(&api_ready)?.into()))
                    .await
                    .context("Failed to send API ready command")?;
            }
        }

        // Start message receiving loop
        info!("Starting SimpleX message receiving loop");
        
        loop {
            tokio::select! {
                // Receive messages from WebSocket
                message = ws_receiver.next() => {
                    match message {
                        Some(Ok(WsMessage::Text(text))) => {
                            debug!("Received from SimpleX: {}", text);
                            
                            // Parse the message
                            if let Ok(response) = serde_json::from_str::<SimplexResponse>(&text) {
                                debug!("JSON-RPC response: {:?}", response);
                            } else if let Ok(event) = serde_json::from_str::<SimplexEvent>(&text) {
                                // Handle events
                                if let Err(e) = self.handle_event(event, &tx).await {
                                    error!("Error handling SimpleX event: {}", e);
                                }
                            } else {
                                warn!("Unknown message format from SimpleX: {}", text);
                            }
                        }
                        Some(Ok(WsMessage::Ping(data))) => {
                            debug!("Received ping, sending pong");
                            let mut sender_guard = self.ws_sender.lock().await;
                            if let Some(ref mut ws_sender) = *sender_guard {
                                let _ = ws_sender.send(WsMessage::Pong(data)).await;
                            }
                        }
                        Some(Ok(WsMessage::Close(_))) => {
                            warn!("SimpleX daemon closed connection");
                            break;
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error: {}", e);
                            // Attempt reconnection
                            sleep(Duration::from_secs(5)).await;
                            match connect_async(&self.config.websocket_url).await {
                                Ok((new_stream, _)) => {
                                    let (new_sender, new_receiver) = new_stream.split();
                                    // Update the stored sender
                                    {
                                        let mut sender_guard = self.ws_sender.lock().await;
                                        *sender_guard = Some(new_sender);
                                    }
                                    ws_receiver = new_receiver;
                                    info!("Reconnected to SimpleX daemon");
                                }
                                Err(e) => {
                                    error!("Failed to reconnect: {}", e);
                                }
                            }
                        }
                        None => {
                            warn!("WebSocket stream ended");
                            break;
                        }
                        _ => {}
                    }
                }
                
                // Periodic daemon health check
                _ = sleep(Duration::from_secs(30)) => {
                    let mut daemon = self.daemon_process.lock().await;
                    if let Some(ref mut child) = *daemon {
                        match child.try_wait() {
                            Ok(Some(status)) => {
                                warn!("SimpleX daemon exited with status: {}", status);
                                drop(daemon);
                                // Restart daemon
                                if let Err(e) = self.start_daemon().await {
                                    error!("Failed to restart SimpleX daemon: {}", e);
                                }
                            }
                            Ok(None) => {
                                // Daemon still running
                                debug!("SimpleX daemon health check: OK");
                            }
                            Err(e) => {
                                error!("Error checking daemon status: {}", e);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        let chat_id = message.recipient.clone();
        let text = message.content.clone();
        
        info!("Sending SimpleX message to {}: {}", chat_id, text);
        
        // Split message if needed
        let chunks = self.split_message(&text);
        
        // Get WebSocket sender (mutable for send operation)
        let mut sender_guard = self.ws_sender.lock().await;
        
        for chunk in chunks {
            // Create SimpleX API request
            let request = SimplexRequest {
                jsonrpc: "2.0".to_string(),
                id: self.next_request_id().await,
                method: "sendMessage".to_string(),
                params: serde_json::json!({
                    "chat": {
                        "type": "direct",
                        "contactId": chat_id
                    },
                    "message": {
                        "type": "text",
                        "text": chunk
                    }
                }),
            };
            
            // Send via WebSocket
            let json = serde_json::to_string(&request)?;
            sender_guard.as_mut()
                .context("WebSocket not connected")?
                .send(WsMessage::Text(json.into())).await?;
            
            debug!("Sent message chunk to SimpleX");
        }
        
        Ok(())
    }

    async fn health_check(&self) -> bool {
        // Check if binary exists or can be downloaded
        if self.ensure_binary().await.is_err() {
            return false;
        }
        
        // Check if daemon is running
        let daemon = self.daemon_process.lock().await;
        daemon.is_some()
    }
}

impl crate::config::traits::ChannelConfig for SimplexConfig {
    fn name() -> &'static str {
        "simplex"
    }

    fn desc() -> &'static str {
        "SimpleX Chat channel with native daemon integration"
    }
}

impl crate::config::traits::ConfigHandle for SimplexConfig {
    fn name(&self) -> &'static str {
        "simplex"
    }

    fn desc(&self) -> &'static str {
        "SimpleX Chat channel with native daemon integration"
    }
}
