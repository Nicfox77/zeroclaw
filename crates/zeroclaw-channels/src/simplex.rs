//! SimpleX transport adapter. The official CLI owns identities, encrypted
//! messaging and call signaling; its external WebRTC bridge owns live media.
//! Config and peer_groups remain the live policy authority at every boundary.
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use base64::Engine as _;
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    sync::{mpsc, oneshot, watch},
};
use tokio_util::sync::CancellationToken;
use zeroclaw_api::channel::{Channel, ChannelMessage, SendMessage};
use zeroclaw_config::live::LiveConfigHandle;

const RPC_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_FRAME: u64 = 8 * 1024 * 1024;

fn bridge_result(value: Value) -> Result<()> {
    if value["ok"] == true {
        return Ok(());
    }
    let error = anyhow::Error::msg("simplex_bridge_command_failed");
    if value["not_delivered"] == true {
        Err(zeroclaw_api::session_events::capture::delivery_definitely_failed(error))
    } else {
        Err(error)
    }
}

#[derive(Clone)]
pub struct SimplexChannel {
    inner: Arc<Inner>,
}
struct Inner {
    alias: String,
    config: LiveConfigHandle,
    writer: Mutex<Option<mpsc::Sender<Value>>>,
    pending: Mutex<HashMap<String, oneshot::Sender<Value>>>,
    drafts: Mutex<HashMap<String, VoiceDraft>>,
    /// Creates the unfinished spoken turn from completed STT chunks. It is
    /// neither durable chat history nor a policy snapshot; only the assembled
    /// turn enters shared dispatch, once silence closes it.
    transcripts: Mutex<HashMap<String, String>>,
    cancel: Mutex<CancellationToken>,
}
struct PendingRequest<'a> {
    inner: &'a Inner,
    id: String,
}
impl Drop for PendingRequest<'_> {
    fn drop(&mut self) {
        self.inner.pending.lock().remove(&self.id);
    }
}

impl SimplexChannel {
    pub fn new(alias: String, config: LiveConfigHandle) -> Self {
        Self {
            inner: Arc::new(Inner {
                alias,
                config,
                writer: Mutex::new(None),
                pending: Mutex::new(HashMap::new()),
                drafts: Mutex::new(HashMap::new()),
                transcripts: Mutex::new(HashMap::new()),
                cancel: Mutex::new(CancellationToken::new()),
            }),
        }
    }
    fn allowed(&self, contact: &str, voice: bool) -> bool {
        if !contact.parse::<u64>().is_ok_and(|id| id > 0)
            || !contact.bytes().all(|b| b.is_ascii_digit())
        {
            return false;
        }
        let config = self.inner.config.read();
        let peers = config.channel_external_peers("simplex", &self.inner.alias);
        config
            .channels
            .simplex
            .get(&self.inner.alias)
            .is_some_and(|c| c.enabled && (!voice || c.calls_enabled))
            && config
                .agent_for_channel(&format!("simplex.{}", self.inner.alias))
                .is_some()
            && peers
                .iter()
                .any(|id| zeroclaw_config::schema::peer_grant_identity(id) == Some(contact))
            && crate::allowlist::is_user_allowed(
                &peers,
                contact,
                crate::allowlist::Match::Sensitive,
            )
    }
    async fn request(&self, mut value: Value) -> Result<()> {
        let id = uuid::Uuid::new_v4().to_string();
        value["id"] = json!(id);
        let writer = self.inner.writer.lock().clone();
        let Some(writer) = writer else {
            return self.one_shot_request(value).await;
        };
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().insert(id.clone(), tx);
        let pending = PendingRequest {
            inner: &self.inner,
            id,
        };
        let timeout = if value["type"] == "audio" {
            Duration::from_secs(200)
        } else {
            RPC_TIMEOUT
        };
        let result = tokio::time::timeout(timeout, async {
            writer.send(value).await.context("simplex_bridge_closed")?;
            bridge_result(rx.await.context("simplex_bridge_disconnected")?)
        })
        .await
        .context("simplex_bridge_timeout")?;
        drop(pending);
        result
    }
    async fn one_shot_request(&self, value: Value) -> Result<()> {
        let settings = self
            .inner
            .config
            .read()
            .channels
            .simplex
            .get(&self.inner.alias)
            .cloned()
            .context("simplex_alias_missing")?;
        settings.validate()?;
        let (program, args) = settings
            .bridge_command
            .split_first()
            .context("simplex_bridge_command_missing")?;
        let mut child = tokio::process::Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let mut input = child.stdin.take().context("simplex_bridge_stdin_missing")?;
        let mut output = BufReader::new(
            child
                .stdout
                .take()
                .context("simplex_bridge_stdout_missing")?,
        );
        let result = tokio::time::timeout(RPC_TIMEOUT, async {
            for (id, mut command) in [
                (
                    "init",
                    json!({"type":"init", "websocket_url": settings.websocket_url}),
                ),
                ("request", value),
            ] {
                command["id"] = json!(id);
                let mut bytes = serde_json::to_vec(&command)?;
                bytes.push(b'\n');
                input.write_all(&bytes).await?;
                loop {
                    let mut line = String::new();
                    ensure!(
                        (&mut output).take(MAX_FRAME).read_line(&mut line).await? > 0
                            && line.ends_with('\n'),
                        "simplex_bridge_invalid_frame"
                    );
                    let response: Value = serde_json::from_str(&line)?;
                    if response["type"] == "result" && response["id"] == id {
                        bridge_result(response)?;
                        break;
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        })
        .await
        .context("simplex_bridge_timeout")?;
        drop(input);
        if tokio::time::timeout(Duration::from_secs(3), child.wait())
            .await
            .is_err()
        {
            child.kill().await?;
        }
        result
    }
    fn manager(&self, contact: &str, call: bool) -> Result<crate::tts::TtsManager> {
        ensure!(self.allowed(contact, call), "simplex_voice_not_authorized");
        let config = self.inner.config.read();
        ensure!(config.tts.enabled, "simplex_tts_disabled");
        let owner = config
            .agent_for_channel(&format!("simplex.{}", self.inner.alias))
            .context("simplex_unowned_channel")?;
        crate::tts::TtsManager::from_config_for_agent(&config, Some(owner))
    }
    fn wants_note(&self, contact: &str, force: bool) -> bool {
        use zeroclaw_config::multi_agent::OutputModality;
        let config = self.inner.config.read();
        let matches = |modality| {
            config
                .channel_modality_peers("simplex", &self.inner.alias, modality)
                .iter()
                .any(|peer| zeroclaw_config::schema::peer_grant_identity(peer) == Some(contact))
        };
        !matches(OutputModality::Text) && (force || matches(OutputModality::Voice))
    }
    async fn voice_notice(&self, contact: &str, outgoing: bool) -> Result<()> {
        if !self.allowed(contact, false) {
            return Ok(());
        }
        let text = zeroclaw_runtime::i18n::get_required_cli_string(if outgoing {
            "channel-simplex-voice-reply-failed"
        } else {
            "channel-simplex-voice-note-failed"
        });
        self.request(json!({"type":"send", "contact":contact, "text":text}))
            .await
    }
    async fn finish_reply(
        &self,
        recipient: &str,
        mut draft: Option<VoiceDraft>,
        text: &str,
        suppress_voice: bool,
        force_voice: bool,
    ) -> Result<()> {
        ensure!(
            self.allowed(recipient, false),
            "simplex_contact_not_authorized"
        );
        if let Some(draft) = &draft {
            ensure!(draft.contact == recipient, "simplex_draft_scope_mismatch");
        }
        let is_call = draft.as_ref().is_some_and(|d| !d.scope.is_empty());
        let mut voice = None;
        if !is_call && !suppress_voice && self.wants_note(recipient, force_voice) {
            let prepared = draft.as_mut().and_then(|d| d.note.take());
            let result = async {
                // Recheck live permission and provider configuration at finalization.
                let current = Arc::new(self.manager(recipient, false)?);
                let note = prepared.unwrap_or_else(|| NoteSpeech {
                    job: crate::tts::SpeechJob::start(current.clone()),
                    manager: current,
                });
                let parts = note.job.finish(text).await?;
                let audio = note.manager.concat_parts_to_m4a(&parts).await?;
                ensure!(audio.len() <= 5 * 1024 * 1024, "simplex_voice_size_limit");
                ensure!(
                    self.allowed(recipient, false),
                    "simplex_contact_not_authorized"
                );
                Ok::<_, anyhow::Error>(base64::engine::general_purpose::STANDARD.encode(audio))
            }
            .await;
            match result {
                Ok(audio) => voice = Some(audio),
                Err(_) => {
                    // Synthesis has no transport effect. Preserve the text response.
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail),
                        "SimpleX voice note synthesis failed; delivering text"
                    );
                }
            }
        }
        // Audio failure after text delivery must never trigger a duplicate text
        // fallback. Deliver audio first and let final text ACK be authoritative.
        if !suppress_voice
            && let Some(mut draft) = draft
            && let Err(error) = draft.finish(text).await
        {
            if zeroclaw_api::session_events::capture::is_capture_stopped(&error) {
                return Err(error);
            }
            // Text is the final delivery boundary. A handled speech failure
            // must not turn an unattempted text send into uncertain capture.
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_attrs(json!({"error": error.to_string()})),
                "SimpleX speech failed; delivering final text"
            );
        }
        ensure!(
            self.allowed(recipient, false),
            "simplex_contact_not_authorized"
        );
        if voice.is_some()
            && (!self.wants_note(recipient, force_voice) || self.manager(recipient, false).is_err())
        {
            voice = None;
        }
        self.request(json!({"type": "send", "contact": recipient, "text": text, "voice":voice}))
            .await
    }

    async fn event(&self, event: Value, tx: &mpsc::Sender<ChannelMessage>) -> Result<()> {
        let kind = event["type"].as_str().unwrap_or("");
        let contact = event["contact"].as_str().unwrap_or("");
        if kind == "call_ended" {
            let prefix = format!("{}/", event["call_id"].as_str().unwrap_or(""));
            self.inner
                .transcripts
                .lock()
                .retain(|scope, _| !scope.starts_with(&prefix));
            return Ok(());
        }
        if kind == "voice_transfer_failed" {
            return self.voice_notice(contact, event["outgoing"] == true).await;
        }
        if kind == "voice_invitation" {
            if !self.allowed(contact, false) {
                return Ok(());
            }
            let limits = {
                let config = self.inner.config.read();
                (
                    config.transcription.enabled,
                    config.transcription.max_duration_secs,
                    config
                        .transcription
                        .max_audio_bytes
                        .unwrap_or(5 * 1024 * 1024)
                        .min(5 * 1024 * 1024),
                )
            };
            if !limits.0
                || !event["duration"]
                    .as_u64()
                    .is_some_and(|n| n > 0 && n <= limits.1)
                || !event["bytes"]
                    .as_u64()
                    .is_some_and(|n| n > 0 && n <= limits.2 as u64)
            {
                return self.voice_notice(contact, false).await;
            }
            return self
                .request(json!({"type":"receive_voice", "contact":contact,
                "message_id":event["id"], "file_id":event["file_id"], "filename":event["filename"],
                "max_duration":limits.1}))
                .await;
        }
        if kind == "call_invitation" {
            let accept = self.allowed(contact, true) && self.manager(contact, true).is_ok() && {
                let config = self.inner.config.read();
                config.transcription.enabled
                    && config
                        .agent_for_channel(&format!("simplex.{}", self.inner.alias))
                        .and_then(|a| config.agents.get(a))
                        .is_some_and(|a| !a.transcription_provider.is_empty())
            };
            return self.request(json!({"type": if accept {"accept"} else {"reject"}, "contact": contact, "call_id": event["call_id"]})).await;
        }
        if !matches!(kind, "message" | "utterance" | "voice_note")
            || !self.allowed(contact, kind == "utterance")
        {
            return Ok(());
        }
        let call = kind == "utterance";
        let voice = call || kind == "voice_note";
        let mut text = if voice {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(event["audio"].as_str().context("simplex_audio_missing")?)?;
            ensure!(
                if call {
                    bytes.len() <= 1_000_000 && bytes.starts_with(b"RIFF")
                } else {
                    !bytes.is_empty() && bytes.len() <= 5 * 1024 * 1024
                },
                "simplex_invalid_utterance"
            );
            let manager = {
                let config = self.inner.config.read();
                ensure!(
                    config.transcription.enabled,
                    "simplex_transcription_disabled"
                );
                ensure!(
                    call || event["duration"].as_f64().is_some_and(
                        |n| n > 0.0 && n <= config.transcription.max_duration_secs as f64
                    ),
                    "simplex_voice_duration_limit"
                );
                let owner = config
                    .agent_for_channel(&format!("simplex.{}", self.inner.alias))
                    .context("simplex_unowned_channel")?;
                crate::transcription::TranscriptionManager::from_config_for_agent(
                    &config,
                    Some(owner),
                )?
            };
            let filename = if call {
                "call.wav"
            } else {
                event["filename"]
                    .as_str()
                    .context("simplex_voice_filename_missing")?
            };
            manager.transcribe(&bytes, filename).await?
        } else {
            event["text"]
                .as_str()
                .context("simplex_text_missing")?
                .to_string()
        };
        // Speech recognition is asynchronous. Recheck policy after it returns.
        if (!voice && text.trim().is_empty()) || !self.allowed(contact, call) {
            return Ok(());
        }
        if call {
            let scope = event["scope"]
                .as_str()
                .context("simplex_voice_scope_missing")?;
            let Some(complete) =
                self.assemble_transcript(scope, &text, event["finished"] == true)?
            else {
                return Ok(());
            };
            text = complete;
        }
        if text.trim().is_empty() {
            return Ok(());
        }
        tx.send(ChannelMessage {
            id: event["id"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            sender: contact.into(),
            platform_sender_id: Some(contact.into()),
            reply_target: contact.into(),
            content: text,
            channel: "simplex".into(),
            channel_alias: Some(self.inner.alias.clone()),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            thread_ts: call.then(|| event["scope"].as_str().unwrap_or("").to_string()),
            interruption_scope_id: call
                .then(|| event["call_id"].as_str().unwrap_or("").to_string()),
            voice_origin: voice,
            explicitly_addressed: true,
            ..Default::default()
        })
        .await
        .context("simplex_dispatch_closed")
    }
    fn assemble_transcript(
        &self,
        scope: &str,
        text: &str,
        finished: bool,
    ) -> Result<Option<String>> {
        let mut transcripts = self.inner.transcripts.lock();
        ensure!(
            transcripts.contains_key(scope) || transcripts.len() < 32,
            "simplex_unfinished_turn_limit"
        );
        let combined = transcripts.entry(scope.to_string()).or_default();
        ensure!(
            combined.len() + text.len() <= 256_000,
            "simplex_transcript_limit"
        );
        if !text.trim().is_empty() {
            if !combined.is_empty() {
                combined.push(' ');
            }
            combined.push_str(text.trim());
        }
        Ok(if finished {
            transcripts.remove(scope)
        } else {
            None
        })
    }
}

impl zeroclaw_api::attribution::Attributable for SimplexChannel {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        zeroclaw_api::attribution::Role::Channel(zeroclaw_api::attribution::ChannelKind::Simplex)
    }
    fn alias(&self) -> &str {
        &self.inner.alias
    }
}

#[async_trait]
impl Channel for SimplexChannel {
    fn name(&self) -> &str {
        "simplex"
    }
    fn uses_cancel_token(&self) -> bool {
        true
    }
    fn set_cancel_token(&self, token: CancellationToken) {
        *self.inner.cancel.lock() = token;
    }
    async fn health_check(&self) -> bool {
        self.request(json!({"type": "ping"})).await.is_ok()
    }
    fn supports_draft_updates(&self) -> bool {
        true
    }
    fn is_direct_message(&self, message: &ChannelMessage) -> bool {
        message.channel == "simplex"
            && message.platform_sender_id.as_deref() == Some(message.reply_target.as_str())
    }
    async fn listen(&self, tx: mpsc::Sender<ChannelMessage>) -> Result<()> {
        let settings = self
            .inner
            .config
            .read()
            .channels
            .simplex
            .get(&self.inner.alias)
            .cloned()
            .context("simplex_alias_missing")?;
        settings.validate()?;
        let (program, args) = settings
            .bridge_command
            .split_first()
            .context("simplex_bridge_command_missing")?;
        let mut child = tokio::process::Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("simplex_bridge_spawn_failed")?;
        let mut input = child.stdin.take().context("simplex_bridge_stdin_missing")?;
        let output = child
            .stdout
            .take()
            .context("simplex_bridge_stdout_missing")?;
        let (writer, mut writes) = mpsc::channel::<Value>(32);
        let (events, mut reads) = mpsc::channel(32);
        *self.inner.writer.lock() = Some(writer);
        let mut tasks = tokio::task::JoinSet::new();
        tasks.spawn(async move {
            while let Some(value) = writes.recv().await {
                let mut data = serde_json::to_vec(&value)?;
                data.push(b'\n');
                input.write_all(&data).await?;
            }
            Ok::<(), anyhow::Error>(())
        });
        let channel = self.clone();
        tasks.spawn(async move {
            let mut reader = BufReader::new(output);
            loop {
                let mut line = String::new();
                let count = (&mut reader).take(MAX_FRAME).read_line(&mut line).await?;
                ensure!(
                    count > 0 && line.ends_with('\n'),
                    "simplex_bridge_invalid_frame"
                );
                let value: Value =
                    serde_json::from_str(&line).context("simplex_bridge_invalid_json")?;
                if value["type"] == "result" {
                    if let Some(id) = value["id"].as_str()
                        && let Some(waiter) = channel.inner.pending.lock().remove(id)
                        && waiter.send(value).is_err()
                    {
                        continue; // the request owner already cancelled
                    }
                } else if matches!(value["type"].as_str(), Some("interrupted" | "call_ended")) {
                    // Drop synthesis immediately, even while STT is pending.
                    let contact = value["contact"].as_str().unwrap_or("");
                    let call_id = value["call_id"].as_str().unwrap_or("");
                    channel.inner.drafts.lock().retain(|_, draft| {
                        draft.contact != contact || draft.scope.split('/').next() != Some(call_id)
                    });
                    if value["type"] == "call_ended" {
                        events
                            .send(value)
                            .await
                            .context("simplex_event_loop_closed")?;
                    }
                } else {
                    events
                        .send(value)
                        .await
                        .context("simplex_event_loop_closed")?;
                }
            }
        });
        let channel = self.clone();
        tasks.spawn(async move {
            channel
                .request(json!({"type": "init", "websocket_url": settings.websocket_url}))
                .await?;
            while let Some(event) = reads.recv().await {
                let note = matches!(event["type"].as_str(), Some("voice_note" | "voice_invitation"));
                let contact = event["contact"].as_str().unwrap_or("").to_string();
                if let Err(error) = channel.event(event, &tx).await {
                    if note { channel.voice_notice(&contact, false).await?; }
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                            .with_attrs(json!({"error": if note { "simplex_voice_note_failed".to_string() } else { error.to_string() }})),
                        "SimpleX event processing failed"
                    );
                }
            }
            Ok::<(), anyhow::Error>(())
        });
        let cancel = self.inner.cancel.lock().clone();
        let result = tokio::select! {
            () = cancel.cancelled() => Ok(()),
            result = tasks.join_next() => match result {
                Some(Ok(result)) => result,
                Some(Err(error)) => Err(anyhow::Error::new(error).context("simplex_bridge_task_failed")),
                None => Err(anyhow::Error::msg("simplex_bridge_tasks_missing")),
            },
        };
        self.inner.writer.lock().take();
        self.inner.pending.lock().clear();
        self.inner.drafts.lock().clear();
        self.inner.transcripts.lock().clear();
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        // Closing stdin gives the helper a chance to close browser/media first.
        if tokio::time::timeout(Duration::from_secs(3), child.wait())
            .await
            .is_err()
        {
            child.kill().await?;
        }
        result
    }
    async fn send(&self, message: &SendMessage) -> Result<()> {
        ensure!(
            self.allowed(&message.recipient, false),
            "simplex_contact_not_authorized"
        );
        if message.thread_ts.is_none() {
            return self
                .finish_reply(
                    &message.recipient,
                    None,
                    &message.content,
                    message.suppress_voice,
                    message.force_voice,
                )
                .await;
        }
        self.request(
            json!({"type": "send", "contact": message.recipient, "text": message.content}),
        )
        .await?;
        if !message.suppress_voice
            && let Some(scope) = &message.thread_ts
        {
            let mut draft = VoiceDraft::new(self.clone(), message.recipient.clone(), scope.clone());
            draft.finish(&message.content).await?;
        }
        Ok(())
    }
    async fn send_draft(&self, message: &SendMessage) -> Result<Option<String>> {
        ensure!(
            self.allowed(&message.recipient, false),
            "simplex_contact_not_authorized"
        );
        let id = uuid::Uuid::new_v4().to_string();
        // Text-only turns do not allocate a speech worker. The stored draft is
        // only a delivery scope; no model or policy snapshot is retained.
        let mut draft = VoiceDraft::new(
            self.clone(),
            message.recipient.clone(),
            message.thread_ts.clone().unwrap_or_default(),
        );
        draft.force_voice = message.force_voice;
        self.inner.drafts.lock().insert(id.clone(), draft);
        Ok(Some(id))
    }
    async fn prepare_voice_draft(&self, recipient: &str, id: &str, text: &str) -> Result<()> {
        ensure!(
            self.allowed(recipient, false),
            "simplex_contact_not_authorized"
        );
        let mut drafts = self.inner.drafts.lock();
        if let Some(draft) = drafts.get_mut(id) {
            ensure!(draft.contact == recipient, "simplex_draft_scope_mismatch");
            if draft.scope.is_empty() && self.wants_note(recipient, draft.force_voice) {
                if draft.note.is_none() {
                    let manager = Arc::new(self.manager(recipient, false)?);
                    draft.note = Some(NoteSpeech {
                        job: crate::tts::SpeechJob::start(manager.clone()),
                        manager,
                    });
                }
                if let Some(note) = &draft.note {
                    note.job.update(text);
                }
            }
        }
        Ok(())
    }
    fn supports_voice_draft_finalization(&self) -> bool {
        true
    }
    async fn finalize_draft_with_modality(
        &self,
        recipient: &str,
        id: &str,
        text: &str,
        suppress_voice: bool,
        force_voice: bool,
    ) -> Result<()> {
        let draft = self.inner.drafts.lock().remove(id);
        self.finish_reply(recipient, draft, text, suppress_voice, force_voice)
            .await
    }
    async fn stream_voice_draft(
        &self,
        recipient: &str,
        id: &str,
        phase: usize,
        text: &str,
        finished: bool,
    ) -> Result<()> {
        ensure!(
            self.allowed(recipient, false),
            "simplex_contact_not_authorized"
        );
        if let Some(draft) = self.inner.drafts.lock().get_mut(id) {
            ensure!(draft.contact == recipient, "simplex_draft_scope_mismatch");
            if !draft.scope.is_empty() {
                draft.update(phase, text, finished);
            }
        }
        Ok(())
    }
    async fn finalize_draft(
        &self,
        recipient: &str,
        id: &str,
        text: &str,
        suppress_voice: bool,
    ) -> Result<()> {
        let draft = self.inner.drafts.lock().remove(id);
        self.finish_reply(recipient, draft, text, suppress_voice, false)
            .await
    }
    async fn cancel_draft(&self, recipient: &str, id: &str) -> Result<()> {
        let draft = self.inner.drafts.lock().remove(id);
        if let Some(draft) = draft
            && !draft.scope.is_empty()
        {
            self.request(json!({"type": "stop_audio", "contact": recipient, "scope": draft.scope}))
                .await?;
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
struct SpeechSnapshot {
    text: String,
    finished: bool,
}
// A draft owns its synthesis job and the same per-turn manager needed to join
// its output format. Neither is an additional policy authority.
struct NoteSpeech {
    manager: Arc<crate::tts::TtsManager>,
    job: crate::tts::SpeechJob,
}
struct VoiceDraft {
    contact: String,
    scope: String,
    channel: std::sync::Weak<Inner>,
    force_voice: bool,
    note: Option<NoteSpeech>,
    phase: usize,
    phase_start: usize,
    updates: watch::Sender<SpeechSnapshot>,
    worker: Option<tokio::task::JoinHandle<Result<()>>>,
    cancel: CancellationToken,
}
impl Drop for VoiceDraft {
    fn drop(&mut self) {
        // Cancel synthesis immediately, but let an admitted audio delivery
        // settle its receipt. Aborting that future would strand an intent.
        self.cancel.cancel();
    }
}
impl VoiceDraft {
    fn new(channel: SimplexChannel, contact: String, scope: String) -> Self {
        let (updates, _) = watch::channel(SpeechSnapshot::default());
        Self {
            contact,
            scope,
            channel: Arc::downgrade(&channel.inner),
            force_voice: false,
            note: None,
            phase: 0,
            phase_start: 0,
            updates,
            worker: None,
            cancel: CancellationToken::new(),
        }
    }
    fn update(&mut self, phase: usize, text: &str, phase_finished: bool) {
        let mut snapshot = self.updates.borrow().clone();
        if phase != self.phase {
            snapshot.text.push_str("\n\n");
            self.phase_start = snapshot.text.len();
            self.phase = phase;
        }
        if !phase_finished {
            snapshot.text.truncate(self.phase_start);
            snapshot.text.push_str(text);
        } else if !snapshot.text.ends_with("\n\n") {
            snapshot.text.push_str("\n\n");
        }
        self.updates.send_replace(snapshot);
        self.start();
    }
    fn start(&mut self) {
        if self.worker.is_some() || self.scope.is_empty() {
            return;
        }
        let channel = self.channel.clone();
        let contact = self.contact.clone();
        let scope = self.scope.clone();
        let mut rx = self.updates.subscribe();
        // Spawned tasks do not inherit task locals. Transfer the existing turn
        // coordinate; its port rechecks live admission at actual delivery.
        let delivery = crate::orchestrator::DeliveryScope::current();
        let cancel = self.cancel.clone();
        self.worker = Some(zeroclaw_spawn::spawn!(async move {
            let mut spoken = Vec::new();
            loop {
                if cancel.is_cancelled() {
                    return Ok(());
                }
                let snapshot = rx.borrow_and_update().clone();
                let channel = SimplexChannel {
                    inner: channel.upgrade().context("simplex_channel_closed")?,
                };
                let manager = channel.manager(&contact, true)?;
                let cfg = crate::speech_segment::SpeechSegmentConfig::default()
                    .with_max_chars(manager.max_text_length());
                let segments =
                    crate::speech_segment::ready_segments(&snapshot.text, &cfg, snapshot.finished);
                ensure!(
                    segments.starts_with(&spoken),
                    "simplex_live_speech_rewritten"
                );
                if let Some(segment) = segments.get(spoken.len()) {
                    let audio = tokio::select! {
                        () = cancel.cancelled() => return Ok(()),
                        result = manager.synthesize(segment) => result?,
                    };
                    ensure!(
                        !audio.is_empty() && audio.len() <= 5 * 1024 * 1024,
                        "simplex_invalid_tts_audio"
                    );
                    let latest = rx.borrow().clone();
                    let current =
                        crate::speech_segment::ready_segments(&latest.text, &cfg, latest.finished);
                    ensure!(
                        current.starts_with(&spoken) && current.get(spoken.len()) == Some(segment),
                        "simplex_live_speech_rewritten"
                    );
                    ensure!(
                        channel.allowed(&contact, true),
                        "simplex_call_not_authorized"
                    );
                    crate::orchestrator::DeliveryScope::observe(
                        delivery.as_ref(),
                        "live_call_audio",
                        "simplex",
                        &contact,
                        segment,
                        channel
                            .request(json!({"type": "audio", "contact": contact, "scope": scope,
                        "audio": base64::engine::general_purpose::STANDARD.encode(audio)})),
                    )
                    .await?;
                    spoken.push(segment.clone());
                    continue;
                }
                if snapshot.finished {
                    return Ok(());
                }
                drop(channel);
                tokio::select! {
                    () = cancel.cancelled() => return Ok(()),
                    changed = rx.changed() => {changed.context("simplex_speech_cancelled")?;}
                }
            }
        }));
    }
    async fn finish(&mut self, text: &str) -> Result<()> {
        if self.scope.is_empty() {
            return Ok(());
        }
        let mut snapshot = self.updates.borrow().clone();
        // Final sanitization can trim whitespace. Preserve earlier, already
        // audible phases and apply the final reply only to its current phase.
        if !snapshot.text.trim().ends_with(text.trim()) {
            snapshot.text.truncate(self.phase_start);
            snapshot.text.push_str(text);
        }
        snapshot.finished = true;
        self.updates.send_replace(snapshot);
        self.start();
        if let Some(worker) = &mut self.worker {
            worker.await.context("simplex_speech_worker_failed")??;
        }
        self.worker.take();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroclaw_config::{
        live::LiveConfig,
        multi_agent::{PeerGroupConfig, PeerUsername},
        schema::{AliasedAgentConfig, Config, SimplexConfig},
    };

    fn fixture() -> (SimplexChannel, LiveConfig) {
        let mut config = Config::default();
        config.channels.simplex.insert(
            "personal".into(),
            SimplexConfig {
                enabled: true,
                calls_enabled: true,
                websocket_url: "ws://127.0.0.1:5225".into(),
                bridge_command: vec!["fixture".into()],
                ..Default::default()
            },
        );
        config.agents.insert(
            "assistant".into(),
            AliasedAgentConfig {
                channels: vec!["simplex.personal".into()],
                ..Default::default()
            },
        );
        config.peer_groups.insert(
            "owner".into(),
            PeerGroupConfig {
                channel: "simplex.personal".into(),
                external_peers: vec![PeerUsername("17".into())],
                ..Default::default()
            },
        );
        let live = LiveConfig::new(config);
        (SimplexChannel::new("personal".into(), live.handle()), live)
    }

    #[test]
    fn bridge_rejections_preserve_known_failure_without_guessing_on_lost_acceptance() {
        use zeroclaw_api::session_events::capture::is_delivery_definitely_failed;
        assert!(bridge_result(json!({"ok":true})).is_ok());
        assert!(is_delivery_definitely_failed(
            &bridge_result(json!({"ok":false,"not_delivered":true})).unwrap_err()
        ));
        assert!(!is_delivery_definitely_failed(
            &bridge_result(json!({"ok":false})).unwrap_err()
        ));
    }

    #[test]
    fn continuous_speech_chunks_form_one_input_only_after_silence() {
        let (channel, _) = fixture();
        assert!(
            channel
                .assemble_transcript("call/turn", "First part.", false)
                .unwrap()
                .is_none()
        );
        assert!(
            channel
                .assemble_transcript("call/turn", "Second part.", false)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            channel
                .assemble_transcript("call/turn", "", true)
                .unwrap()
                .as_deref(),
            Some("First part. Second part.")
        );
        assert_eq!(
            channel
                .assemble_transcript("call/next", "Next turn.", true)
                .unwrap()
                .as_deref(),
            Some("Next turn.")
        );
        assert!(channel.inner.transcripts.lock().is_empty());
    }

    #[tokio::test]
    async fn private_messages_use_stable_contact_ids_and_live_denies() {
        let (channel, live) = fixture();
        let (tx, mut rx) = mpsc::channel(4);
        channel
            .event(
                json!({"type":"message", "contact":"18", "text":"ignored"}),
                &tx,
            )
            .await
            .unwrap();
        assert!(rx.try_recv().is_err());
        channel
            .event(
                json!({"type":"message", "id":"1", "contact":"17", "text":"Hello"}),
                &tx,
            )
            .await
            .unwrap();
        let message = rx.recv().await.unwrap();
        assert_eq!(message.platform_sender_id.as_deref(), Some("17"));
        assert!(channel.is_direct_message(&message));
        let mut config = live.snapshot();
        config
            .peer_groups
            .get_mut("owner")
            .unwrap()
            .ignore
            .push(PeerUsername("17".into()));
        live.publish(live.next_revision().unwrap(), config).unwrap();
        assert!(!channel.allowed("17", false));
        channel
            .event(
                json!({"type":"message", "contact":"17", "text":"denied"}),
                &tx,
            )
            .await
            .unwrap();
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn wildcards_do_not_authorize_calls_and_disable_is_live() {
        let (channel, live) = fixture();
        assert!(channel.allowed("17", true));
        let mut config = live.snapshot();
        config.peer_groups.get_mut("owner").unwrap().external_peers =
            vec![PeerUsername("*".into())];
        live.publish(live.next_revision().unwrap(), config).unwrap();
        assert!(!channel.allowed("17", true));
        for id in ["0", "-1", "17 ", "@17", "*", "someone"] {
            assert!(!channel.allowed(id, false));
        }
        let mut config = live.snapshot();
        config.peer_groups.get_mut("owner").unwrap().external_peers =
            vec![PeerUsername("17".into())];
        config
            .channels
            .simplex
            .get_mut("personal")
            .unwrap()
            .calls_enabled = false;
        live.publish(live.next_revision().unwrap(), config).unwrap();
        assert!(channel.allowed("17", false));
        assert!(!channel.allowed("17", true));
    }

    #[tokio::test]
    async fn notes_transcribe_with_owning_agent_without_call_scope_or_call_permission() {
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };
        use zeroclaw_config::schema::LocalWhisperTranscriptionProviderConfig;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/transcribe"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"text":"A spoken request"})),
            )
            .mount(&server)
            .await;
        let (channel, live) = fixture();
        let mut config = live.snapshot();
        config
            .channels
            .simplex
            .get_mut("personal")
            .unwrap()
            .calls_enabled = false;
        config.transcription.enabled = true;
        config.transcription.max_duration_secs = 600;
        config
            .agents
            .get_mut("assistant")
            .unwrap()
            .transcription_provider = "local_whisper.fixture".into();
        config.providers.transcription.local_whisper.insert(
            "fixture".into(),
            LocalWhisperTranscriptionProviderConfig {
                uri: format!("{}/transcribe", server.uri()),
                ..Default::default()
            },
        );
        live.publish(live.next_revision().unwrap(), config).unwrap();
        let (tx, mut rx) = mpsc::channel(4);
        let audio = base64::engine::general_purpose::STANDARD.encode(b"audio");
        channel.event(json!({"type":"voice_note","contact":"18","id":"42","audio":audio,"filename":"voice.m4a","duration":27}), &tx).await.unwrap();
        assert!(server.received_requests().await.unwrap().is_empty());
        channel.event(json!({"type":"voice_note","contact":"17","id":"42","audio":audio,"filename":"voice.m4a","duration":27}), &tx).await.unwrap();
        let message = rx.recv().await.unwrap();
        assert_eq!(message.id, "42");
        assert_eq!(message.content, "A spoken request");
        assert!(message.voice_origin);
        assert!(message.thread_ts.is_none());
        assert!(message.interruption_scope_id.is_none());
        assert_eq!(message.channel_alias.as_deref(), Some("personal"));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        assert!(channel.event(json!({"type":"voice_note","contact":"17","id":"43","audio":audio,"filename":"voice.m4a","duration":601}), &tx).await.is_err());
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn receive_admission_retains_platform_item_id_and_checks_limits_before_download() {
        let (channel, live) = fixture();
        let mut config = live.snapshot();
        config.transcription.enabled = true;
        config.transcription.max_duration_secs = 600;
        live.publish(live.next_revision().unwrap(), config).unwrap();
        let (writer, mut requests) = mpsc::channel::<Value>(4);
        *channel.inner.writer.lock() = Some(writer);
        let responder = channel.clone();
        let task = zeroclaw_spawn::spawn!(async move {
            let value = requests.recv().await.unwrap();
            assert_eq!(value["type"], "receive_voice");
            assert_eq!(value["message_id"], "42");
            assert_ne!(value["id"], "42");
            assert_eq!(value["max_duration"], 600);
            responder
                .inner
                .pending
                .lock()
                .remove(value["id"].as_str().unwrap())
                .unwrap()
                .send(json!({"ok":true}))
                .unwrap();
        });
        let (tx, _rx) = mpsc::channel(4);
        channel.event(json!({"type":"voice_invitation","contact":"18","id":"42","file_id":7,"filename":"voice.m4a","bytes":2000,"duration":27}),&tx).await.unwrap();
        channel.event(json!({"type":"voice_invitation","contact":"17","id":"42","file_id":7,"filename":"voice.m4a","bytes":2000,"duration":27}),&tx).await.unwrap();
        task.await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires ffmpeg with AAC; run explicitly for voice deployment"]
    async fn note_synthesis_starts_early_and_force_finalization_keeps_prepared_audio_and_text() {
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };
        use zeroclaw_config::schema::{OpenAITtsProviderConfig, TtsProviderConfig};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(crate::tts::speech::tests::silence_wav(2400)),
            )
            .mount(&server)
            .await;
        let (channel, live) = fixture();
        let mut config = live.snapshot();
        config
            .channels
            .simplex
            .get_mut("personal")
            .unwrap()
            .calls_enabled = false;
        config.tts.enabled = true;
        config.agents.get_mut("assistant").unwrap().tts_provider = "openai.fixture".into();
        config.providers.tts.openai.insert(
            "fixture".into(),
            OpenAITtsProviderConfig {
                base: TtsProviderConfig {
                    api_key: Some("fixture".into()),
                    uri: Some(format!("{}/speech", server.uri())),
                    response_format: Some("wav".into()),
                    ..Default::default()
                },
            },
        );
        live.publish(live.next_revision().unwrap(), config).unwrap();
        let (writer, mut requests) = mpsc::channel::<Value>(4);
        *channel.inner.writer.lock() = Some(writer);
        let responder = channel.clone();
        let task = zeroclaw_spawn::spawn!(async move {
            let value = requests.recv().await.unwrap();
            assert_eq!(value["type"], "send");
            assert!(
                value["text"]
                    .as_str()
                    .unwrap()
                    .contains("The remaining answer.")
            );
            let audio = base64::engine::general_purpose::STANDARD
                .decode(value["voice"].as_str().unwrap())
                .unwrap();
            assert_eq!(&audio[4..8], b"ftyp");
            responder
                .inner
                .pending
                .lock()
                .remove(value["id"].as_str().unwrap())
                .unwrap()
                .send(json!({"ok":true}))
                .unwrap();
        });
        let draft = channel
            .send_draft(&SendMessage::new("draft", "17").force_voice())
            .await
            .unwrap()
            .unwrap();
        let opener =
            "The opening sentence is long enough to synthesize before the response finishes.";
        channel
            .prepare_voice_draft("17", &draft, &format!("{opener} Still writing"))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while server.received_requests().await.unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        channel
            .finalize_draft_with_modality(
                "17",
                &draft,
                &format!("{opener} The remaining answer."),
                false,
                true,
            )
            .await
            .unwrap();
        task.await.unwrap();
        let requests = server.received_requests().await.unwrap();
        assert_eq!(
            requests.len(),
            2,
            "the opener must not be synthesized twice"
        );
        assert!(channel.inner.drafts.lock().is_empty());
    }

    #[tokio::test]
    async fn live_speech_is_sent_before_finalization_without_repeating_the_opener() {
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };
        use zeroclaw_config::schema::{OpenAITtsProviderConfig, TtsProviderConfig};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/speech"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"audio".to_vec()))
            .mount(&server)
            .await;
        let (channel, live) = fixture();
        let mut config = live.snapshot();
        config.tts.enabled = true;
        config.agents.get_mut("assistant").unwrap().tts_provider = "openai.fixture".into();
        config.providers.tts.openai.insert(
            "fixture".into(),
            OpenAITtsProviderConfig {
                base: TtsProviderConfig {
                    api_key: Some("fixture".into()),
                    uri: Some(format!("{}/speech", server.uri())),
                    ..Default::default()
                },
            },
        );
        live.publish(live.next_revision().unwrap(), config).unwrap();
        let (writer, mut requests) = mpsc::channel::<Value>(4);
        *channel.inner.writer.lock() = Some(writer);
        let observer = channel.clone();
        let (audio_tx, mut audio_rx) = mpsc::channel(8);
        let responder = zeroclaw_spawn::spawn!(async move {
            while let Some(value) = requests.recv().await {
                audio_tx.send(value.clone()).await.unwrap();
                if let Some(waiter) = observer
                    .inner
                    .pending
                    .lock()
                    .remove(value["id"].as_str().unwrap())
                {
                    waiter.send(json!({"ok": true})).ok();
                }
            }
        });
        let opener =
            "The opening sentence is long enough to synthesize before the response finishes.";
        let mut draft = VoiceDraft::new(channel, "17".into(), "call/generation".into());
        draft.update(0, &format!("{opener} Still writing"), false);
        let first = tokio::time::timeout(Duration::from_secs(3), audio_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first["type"], "audio");
        assert_eq!(first["scope"], "call/generation");
        draft
            .finish(&format!("{opener} The remaining answer."))
            .await
            .unwrap();
        assert_eq!(audio_rx.recv().await.unwrap()["type"], "audio");
        assert!(audio_rx.try_recv().is_err());
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
        responder.abort();
    }
}
