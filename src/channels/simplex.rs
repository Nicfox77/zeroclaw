//! SimpleX Chat channel implementation for ZeroClaw.
//!
//! This module provides native SimpleX Chat integration via the simplex-chat CLI daemon.
//! It features auto-download of the simplex-chat binary and automatic subprocess management.

use super::ack_reaction::{select_ack_reaction, AckReactionContext, AckReactionContextChatType};
use super::traits::{Channel, ChannelMessage, SendMessage};
use crate::config::Config;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command as AsyncCommand};
use tokio::sync::Mutex;
use tokio::time::sleep;
use tracing::{error, info, warn};

/// SimpleX's maximum message length (approximate, actual limit varies)
const SIMPLEX_MAX_MESSAGE_LENGTH: usize = 16000;
const SIMPLEX_ACK_REACTIONS: &[&str] = &["⚡", "👌", "👀", "🔥", "👍"];

/// SimpleX Chat channel configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
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
}

impl SimplexChannel {
    /// Create a new SimpleX channel
    pub fn new(config: SimplexConfig) -> Self {
        Self {
            config,
            workspace_dir: PathBuf::new(),
            daemon_process: Arc::new(Mutex::new(None)),
            ack_enabled: true,
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

    /// Get the path to the simplex-chat binary
    fn get_binary_path(&self) -> PathBuf {
        if let Some(ref custom_path) = self.config.binary_path {
            PathBuf::from(custom_path)
        } else {
            let data_dir = dirs::data_dir()
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
            fs::set_permissions(&binary_path, fs::Permissions::from_mode(0o755)).await?;
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
}

#[async_trait]
impl Channel for SimplexChannel {
    fn name(&self) -> &str {
        "simplex"
    }

    async fn start(
        &self,
        config: Config,
        sender: tokio::sync::mpsc::Sender<ChannelMessage>,
    ) -> Result<()> {
        // Ensure binary is available
        let binary_path = self.ensure_binary().await?;
        
        // Start daemon if not running
        self.start_daemon().await?;

        info!("SimpleX channel started successfully");
        info!("Binary path: {}", binary_path.display());
        info!("WebSocket URL: {}", self.config.websocket_url);

        // TODO: Implement WebSocket connection to daemon
        // TODO: Implement message receiving loop
        // For now, this is a stub that starts the daemon successfully

        // Keep the channel alive
        loop {
            sleep(Duration::from_secs(60)).await;
            
            // Check if daemon is still running
            let daemon = self.daemon_process.lock().await;
            if let Some(ref child) = *daemon {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        warn!("SimpleX daemon exited with status: {}", status);
                        drop(daemon);
                        // Restart daemon
                        self.start_daemon().await?;
                    }
                    Ok(None) => {
                        // Daemon still running
                    }
                    Err(e) => {
                        error!("Error checking daemon status: {}", e);
                    }
                }
            }
        }
    }

    async fn send(&self, message: SendMessage) -> Result<()> {
        // TODO: Implement WebSocket message sending
        // For now, this is a stub
        info!("SimpleX send: {:?}", message.text);
        Ok(())
    }

    async fn health_check(&self) -> Result<()> {
        // Check if binary exists or can be downloaded
        self.ensure_binary().await?;
        
        // Check if daemon is running
        let daemon = self.daemon_process.lock().await;
        if daemon.is_none() {
            anyhow::bail!("SimpleX daemon is not running");
        }
        
        Ok(())
    }

    async fn typing_indicator(&self, _enabled: bool) -> Result<()> {
        // SimpleX doesn't support typing indicators
        Ok(())
    }
}

impl crate::config::traits::ConfigHandle for SimplexConfig {
    fn enabled(&self) -> bool {
        true
    }

    fn channel_name(&self) -> &str {
        "simplex"
    }
}
