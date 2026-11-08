//! Opt-in IMAP draft creation. SMTP and mailbox deletion are never used.
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use async_imap::{Session, types::NameAttribute};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use lettre::Message;
use lettre::message::{
    Mailbox,
    header::{ContentType, HeaderName, HeaderValue},
};
use mail_parser::MessageParser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, AsyncWrite};
use zeroclaw_api::{
    attribution::ToolKind,
    tool::{Tool, ToolOutput, ToolResult},
};
use zeroclaw_config::{
    policy::{SecurityPolicy, ToolOperation},
    scattered_types::EmailConfig,
    schema::Config,
};

use crate::email_imap::imap_connect;

zeroclaw_api::tool_attribution!(EmailDraftTool, ToolKind::Plugin);

/// The owner resolves current config on each call, including immediately before
/// APPEND. This handle never snapshots draft grants or mailbox credentials.
pub type EmailDraftConfigResolver = Arc<dyn Fn(&str) -> Result<EmailConfig> + Send + Sync>;

pub struct EmailDraftTool {
    config: EmailDraftConfigResolver,
    auth_service: Option<Arc<zeroclaw_providers::auth::AuthService>>,
}

static DESCRIPTION: OnceLock<String> = OnceLock::new();
const HASH_HEADER: &str = "X-ZeroClaw-Draft-Hash";
const MAX_BODY_BYTES: usize = 256 * 1024;

fn text(key: &str) -> String {
    crate::i18n::get_required_tool_string(key)
}

#[derive(Deserialize, Serialize)]
struct DraftRequest {
    channel: String,
    request_id: String,
    to: Vec<String>,
    #[serde(default)]
    cc: Vec<String>,
    subject: String,
    body: String,
    #[serde(default)]
    reply_to_uid: Option<u32>,
    #[serde(default = "inbox")]
    reply_folder: String,
}

fn inbox() -> String {
    "INBOX".into()
}

fn valid_header(value: &str) -> bool {
    !value.contains(['\r', '\n', '\0']) && value.len() <= 2048
}

impl DraftRequest {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.channel.is_empty()
                && !self.request_id.is_empty()
                && self.request_id.len() <= 200
                && !self.request_id.chars().any(char::is_control)
                && (1..=20).contains(&self.to.len())
                && self.cc.len() <= 20
                && valid_header(&self.subject)
                && self.body.len() <= MAX_BODY_BYTES
                && self.reply_to_uid != Some(0)
                && valid_header(&self.reply_folder)
                && !self.reply_folder.is_empty(),
            text("tool-email-draft-error-input")
        );
        for address in self.to.iter().chain(&self.cc) {
            anyhow::ensure!(valid_header(address), text("tool-email-draft-error-input"));
            address
                .parse::<Mailbox>()
                .context(text("tool-email-draft-error-address"))?;
        }
        Ok(())
    }

    fn message_id(&self) -> String {
        let hash = Sha256::digest(format!("{}\0{}", self.channel, self.request_id).as_bytes());
        format!("<{hash:x}@draft.zeroclaw.invalid>")
    }

    fn content_hash(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }
}

impl EmailDraftTool {
    pub fn new(
        config: EmailDraftConfigResolver,
        auth_service: Option<Arc<zeroclaw_providers::auth::AuthService>>,
    ) -> Self {
        Self {
            config,
            auth_service,
        }
    }

    /// Resolve both account permission and the caller's current write policy
    /// from the same canonical config at the action boundary.
    pub fn resolve_config(config: &Config, agent: &str, mailbox: &str) -> Result<EmailConfig> {
        let security = SecurityPolicy::for_agent(config, agent)?;
        anyhow::ensure!(
            security.is_tool_allowed("email_draft"),
            text("tool-email-draft-error-disabled")
        );
        // The outer RateLimitedTool owns action accounting; this is a fresh
        // authorization check, not a second budget charge.
        security
            .authorize_tool_operation(ToolOperation::Act, "email_draft")
            .map_err(anyhow::Error::msg)?;
        let account = config
            .channels
            .email
            .get(mailbox)
            .context(text("tool-email-draft-error-disabled"))?;
        anyhow::ensure!(
            account.can_save_drafts(),
            text("tool-email-draft-error-disabled")
        );
        Ok(account.clone())
    }

    fn current_account(&self, alias: &str) -> Result<EmailConfig> {
        let config = (self.config)(alias)?;
        anyhow::ensure!(
            config.can_save_drafts(),
            text("tool-email-draft-error-disabled")
        );
        Ok(config)
    }

    async fn save<S>(
        &self,
        session: &mut Session<S>,
        account: &EmailConfig,
        request: &DraftRequest,
    ) -> Result<Value>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + std::fmt::Debug,
    {
        let folders: Vec<_> = session
            .list(Some(""), Some("\"*\""))
            .await?
            .try_collect()
            .await?;
        let drafts: Vec<_> = folders
            .iter()
            .filter(|folder| {
                folder.attributes().contains(&NameAttribute::Drafts)
                    && !folder.attributes().contains(&NameAttribute::NoSelect)
            })
            .collect();
        anyhow::ensure!(drafts.len() == 1, text("tool-email-draft-error-folder"));
        let folder = drafts[0].name();
        let id = request.message_id();
        let hash = request.content_hash()?;

        // EXAMINE + PEEK preserve user flags, including on the source message.
        session.examine(folder).await?;
        let existing = session
            .uid_search(format!("HEADER Message-ID \"{id}\""))
            .await?;
        if !existing.is_empty() {
            anyhow::ensure!(existing.len() == 1, text("tool-email-draft-error-conflict"));
            let uid = existing
                .iter()
                .next()
                .context(text("tool-email-draft-error-conflict"))?;
            let rows: Vec<_> = session
                .uid_fetch(uid.to_string(), "BODY.PEEK[HEADER]")
                .await?
                .try_collect()
                .await?;
            let message = rows
                .first()
                .and_then(|row| row.header())
                .and_then(|body| MessageParser::default().parse(body))
                .context(text("tool-email-draft-error-conflict"))?;
            anyhow::ensure!(
                message
                    .header(HASH_HEADER)
                    .and_then(|value| value.as_text())
                    == Some(hash.as_str()),
                text("tool-email-draft-error-conflict")
            );
            return Ok(
                json!({"saved":true,"already_existed":true,"channel":request.channel,"folder":folder,"message_id":id,"uid":uid}),
            );
        }

        let mut message = Message::builder()
            .from(
                account
                    .username
                    .parse::<Mailbox>()
                    .context(text("tool-email-draft-error-address"))?,
            )
            .subject(&request.subject)
            .message_id(Some(id.clone()))
            .header(ContentType::TEXT_PLAIN)
            .raw_header(HeaderValue::new(
                HeaderName::new_from_ascii(HASH_HEADER.into())?,
                hash,
            ));
        for to in &request.to {
            message = message.to(to.parse::<Mailbox>()?);
        }
        for cc in &request.cc {
            message = message.cc(cc.parse::<Mailbox>()?);
        }
        if let Some(uid) = request.reply_to_uid {
            session.examine(&request.reply_folder).await?;
            let rows: Vec<_> = session
                .uid_fetch(uid.to_string(), "BODY.PEEK[HEADER]")
                .await?
                .try_collect()
                .await?;
            let parent = rows
                .first()
                .and_then(|row| row.header())
                .and_then(|body| MessageParser::default().parse(body))
                .context(text("tool-email-draft-error-parent"))?;
            let parent_id = parent
                .message_id()
                .filter(|id| valid_message_id(id))
                .context(text("tool-email-draft-error-parent"))?;
            let mut references = parent
                .references()
                .as_text_list()
                .unwrap_or_default()
                .iter()
                .filter(|id| valid_message_id(id))
                .take(20)
                .map(|id| format!("<{id}>"))
                .collect::<Vec<_>>();
            references.push(format!("<{parent_id}>"));
            message = message
                .in_reply_to(format!("<{parent_id}>"))
                .references(references.join(" "));
        }
        let bytes = message.body(request.body.clone())?.formatted();
        // Revocation or rotation during any network awaits must take effect
        // before the only write. Do not send credentials to a stale connection.
        let current = self.current_account(&request.channel)?;
        anyhow::ensure!(
            serde_json::to_value(&current)? == serde_json::to_value(account)?,
            text("tool-email-draft-error-changed")
        );
        session
            .append(folder, Some(r"(\Draft)"), None, bytes)
            .await
            .context(text("tool-email-draft-error-uncertain"))?;
        Ok(
            json!({"saved":true,"already_existed":false,"channel":request.channel,"folder":folder,"message_id":id}),
        )
    }
}

fn valid_message_id(id: &str) -> bool {
    id.len() <= 512
        && id.contains('@')
        && !id
            .chars()
            .any(|c| c.is_control() || c.is_whitespace() || matches!(c, '<' | '>'))
}

#[async_trait]
impl Tool for EmailDraftTool {
    fn name(&self) -> &str {
        "email_draft"
    }
    fn description(&self) -> &str {
        DESCRIPTION
            .get_or_init(|| text("tool-email-draft"))
            .as_str()
    }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object","required":["channel","request_id","to","subject","body"],"properties":{
            "channel":{"type":"string","description":text("tool-email-draft-param-channel")},
            "request_id":{"type":"string","maxLength":200,"description":text("tool-email-draft-param-request")},
            "to":{"type":"array","minItems":1,"maxItems":20,"items":{"type":"string"}},
            "cc":{"type":"array","maxItems":20,"items":{"type":"string"}},
            "subject":{"type":"string","maxLength":2048},
            "body":{"type":"string","maxLength":262144,"description":text("tool-email-draft-param-body")},
            "reply_to_uid":{"type":"integer","minimum":1,"maximum":4294967295u64,"description":text("tool-email-draft-param-reply")},
            "reply_folder":{"type":"string","default":"INBOX"}
        }})
    }
    async fn execute(&self, args: Value) -> Result<ToolResult> {
        let request: DraftRequest =
            serde_json::from_value(args).context(text("tool-email-draft-error-input"))?;
        request.validate()?;
        let account = self.current_account(&request.channel)?;
        let mut session = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            imap_connect(&account, self.auth_service.as_ref(), &request.channel),
        )
        .await
        .context(text("tool-email-draft-error-connect"))?
        .context(text("tool-email-draft-error-connect"))?;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            self.save(&mut session, &account, &request),
        )
        .await
        .context(text("tool-email-draft-error-uncertain"))?;
        // APPEND acknowledgment owns the result; logout failure must never
        // turn a saved draft into a reported failure and invite a duplicate.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), session.logout()).await;
        Ok(ToolResult::ok(ToolOutput::json(result?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, DuplexStream};

    #[derive(Default)]
    struct MailboxState {
        drafts: Vec<Vec<u8>>,
        commands: Vec<String>,
        parent: Vec<u8>,
        no_drafts: bool,
        drop_append_ack: bool,
        revoke_on_list: Option<Arc<Mutex<EmailConfig>>>,
    }

    fn account() -> EmailConfig {
        EmailConfig {
            enabled: true,
            allow_drafts: true,
            imap_host: "imap.example.test".into(),
            username: "owner@example.test".into(),
            ..Default::default()
        }
    }
    fn request() -> DraftRequest {
        DraftRequest {
            channel: "personal".into(),
            request_id: "draft-test-1".into(),
            to: vec!["Recipient <recipient@example.test>".into()],
            cc: Vec::new(),
            subject: "Café appointment".into(),
            body: "Hello!\n\nÇa va?\n".into(),
            reply_to_uid: None,
            reply_folder: inbox(),
        }
    }
    fn tool(config: Arc<Mutex<EmailConfig>>) -> EmailDraftTool {
        EmailDraftTool::new(Arc::new(move |_| Ok(config.lock().unwrap().clone())), None)
    }

    // Exercises the real async-imap encoder/parser against a duplex server.
    // The server accepts only read-only discovery/fetches and Drafts APPEND;
    // any STORE, DELETE, EXPUNGE or unrelated command fails this fixture.
    async fn session(state: Arc<Mutex<MailboxState>>) -> Session<DuplexStream> {
        let (client, server) = tokio::io::duplex(1024 * 1024);
        zeroclaw_spawn::spawn!(async move {
            let mut stream = BufReader::new(server);
            let mut selected = String::new();
            loop {
                let mut line = String::new();
                if stream.read_line(&mut line).await.unwrap() == 0 {
                    break;
                }
                let line = line.trim_end_matches(['\r', '\n']).to_string();
                state.lock().unwrap().commands.push(line.clone());
                let (tag, command) = line.split_once(' ').unwrap();
                let mut response = String::new();
                if command.starts_with("LOGIN ") {
                    response = format!("{tag} OK logged in\r\n");
                } else if command.starts_with("LIST ") {
                    let (no_drafts, revoke) = {
                        let s = state.lock().unwrap();
                        (s.no_drafts, s.revoke_on_list.clone())
                    };
                    if let Some(revoke) = revoke {
                        revoke.lock().unwrap().allow_drafts = false;
                    }
                    if !no_drafts {
                        response.push_str("* LIST (\\Drafts) \"/\" \"[Gmail]/Brouillons\"\r\n");
                    }
                    response.push_str(&format!("* LIST () \"/\" \"INBOX\"\r\n{tag} OK list\r\n"));
                } else if command.starts_with("EXAMINE ") {
                    selected = command.to_string();
                    response = format!(
                        "* 1 EXISTS\r\n* OK [UIDVALIDITY 1] stable\r\n{tag} OK [READ-ONLY] examined\r\n"
                    );
                } else if command.starts_with("UID SEARCH ") {
                    let s = state.lock().unwrap();
                    let found = s.drafts.iter().position(|body| {
                        let parsed = MessageParser::default().parse(body).unwrap();
                        command.contains(&format!("<{}>", parsed.message_id().unwrap()))
                    });
                    response = match found {
                        Some(index) => format!("* SEARCH {}\r\n{tag} OK search\r\n", index + 1),
                        None => format!("* SEARCH\r\n{tag} OK search\r\n"),
                    };
                } else if command.starts_with("UID FETCH ") {
                    assert!(
                        command.ends_with("BODY.PEEK[HEADER]"),
                        "Original flags must be preserved: {command}"
                    );
                    let body = {
                        let s = state.lock().unwrap();
                        if selected.contains("Brouillons") {
                            s.drafts[0].clone()
                        } else {
                            s.parent.clone()
                        }
                    };
                    let prefix = format!("* 1 FETCH (UID 1 BODY[HEADER] {{{}}}\r\n", body.len());
                    stream.get_mut().write_all(prefix.as_bytes()).await.unwrap();
                    stream.get_mut().write_all(&body).await.unwrap();
                    response = format!(")\r\n{tag} OK fetched\r\n");
                } else if command.starts_with("APPEND ") {
                    assert!(
                        command.starts_with("APPEND \"[Gmail]/Brouillons\" (\\Draft) {"),
                        "Only Drafts may be written: {command}"
                    );
                    let length: usize = command
                        .rsplit_once('{')
                        .unwrap()
                        .1
                        .trim_end_matches('}')
                        .parse()
                        .unwrap();
                    stream
                        .get_mut()
                        .write_all(b"+ send literal\r\n")
                        .await
                        .unwrap();
                    stream.get_mut().flush().await.unwrap();
                    let mut bytes = vec![0; length + 2];
                    stream.read_exact(&mut bytes).await.unwrap();
                    assert_eq!(&bytes[length..], b"\r\n");
                    bytes.truncate(length);
                    let drop_ack = {
                        let mut s = state.lock().unwrap();
                        s.drafts.push(bytes);
                        std::mem::take(&mut s.drop_append_ack)
                    };
                    if drop_ack {
                        break;
                    }
                    response = format!("{tag} OK appended\r\n");
                } else if command == "LOGOUT" {
                    response = format!("* BYE logout\r\n{tag} OK logout\r\n");
                } else {
                    panic!("Unexpected IMAP operation: {command}");
                }
                stream
                    .get_mut()
                    .write_all(response.as_bytes())
                    .await
                    .unwrap();
                stream.get_mut().flush().await.unwrap();
                if command == "LOGOUT" {
                    break;
                }
            }
        });
        async_imap::Client::new(client)
            .login("owner@example.test", "synthetic-password")
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn writes_utf8_draft_to_server_designated_folder_and_reuses_receipt() {
        let cfg = account();
        let tool = tool(Arc::new(Mutex::new(cfg.clone())));
        let request = request();
        let state = Arc::new(Mutex::new(MailboxState::default()));
        let mut imap = session(state.clone()).await;
        let saved = tool.save(&mut imap, &cfg, &request).await.unwrap();
        assert_eq!(saved["already_existed"], false);
        let again = tool.save(&mut imap, &cfg, &request).await.unwrap();
        assert_eq!(again["already_existed"], true);
        let state = state.lock().unwrap();
        assert_eq!(state.drafts.len(), 1);
        let parsed = MessageParser::default().parse(&state.drafts[0]).unwrap();
        assert_eq!(parsed.subject(), Some("Café appointment"));
        assert!(parsed.body_text(0).unwrap().contains("Ça va?"));
        assert_eq!(
            parsed.from().unwrap().first().unwrap().address(),
            Some("owner@example.test")
        );
        assert_eq!(
            parsed.to().unwrap().first().unwrap().address(),
            Some("recipient@example.test")
        );
        assert_eq!(
            state
                .commands
                .iter()
                .filter(|c| c.contains(" APPEND "))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn reply_preserves_original_flags_and_links_thread() {
        let cfg = account();
        let tool = tool(Arc::new(Mutex::new(cfg.clone())));
        let mut request = request();
        request.reply_to_uid = Some(1);
        let state = Arc::new(Mutex::new(MailboxState {
            parent: b"Message-ID: <parent@example.test>\r\nReferences: <root@example.test>\r\n\r\n"
                .to_vec(),
            ..Default::default()
        }));
        tool.save(&mut session(state.clone()).await, &cfg, &request)
            .await
            .unwrap();
        let state = state.lock().unwrap();
        let parsed = MessageParser::default().parse(&state.drafts[0]).unwrap();
        assert_eq!(parsed.in_reply_to().as_text(), Some("parent@example.test"));
        let refs = parsed.references().as_text_list().unwrap();
        assert_eq!(
            refs.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
            ["root@example.test", "parent@example.test"]
        );
        assert!(
            state
                .commands
                .iter()
                .any(|c| c.ends_with("EXAMINE \"INBOX\""))
        );
        assert!(
            state
                .commands
                .iter()
                .any(|c| c.ends_with("UID FETCH 1 BODY.PEEK[HEADER]"))
        );
    }

    #[tokio::test]
    async fn conflicting_retry_never_overwrites_saved_draft() {
        let cfg = account();
        let tool = tool(Arc::new(Mutex::new(cfg.clone())));
        let mut request = request();
        let state = Arc::new(Mutex::new(MailboxState::default()));
        let mut imap = session(state.clone()).await;
        tool.save(&mut imap, &cfg, &request).await.unwrap();
        request.body = "Changed".into();
        assert!(tool.save(&mut imap, &cfg, &request).await.is_err());
        assert_eq!(state.lock().unwrap().drafts.len(), 1);
    }

    #[tokio::test]
    async fn lost_append_ack_reconciles_on_new_connection_without_duplicate() {
        let cfg = account();
        let tool = tool(Arc::new(Mutex::new(cfg.clone())));
        let request = request();
        let state = Arc::new(Mutex::new(MailboxState {
            drop_append_ack: true,
            ..Default::default()
        }));
        let error = tool
            .save(&mut session(state.clone()).await, &cfg, &request)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("may have saved"));
        let result = tool
            .save(&mut session(state.clone()).await, &cfg, &request)
            .await
            .unwrap();
        assert_eq!(result["already_existed"], true);
        assert_eq!(state.lock().unwrap().drafts.len(), 1);
    }

    #[tokio::test]
    async fn grant_revoked_during_discovery_prevents_append() {
        let cfg = account();
        let live = Arc::new(Mutex::new(cfg.clone()));
        let tool = tool(live.clone());
        let state = Arc::new(Mutex::new(MailboxState {
            revoke_on_list: Some(live),
            ..Default::default()
        }));
        assert!(
            tool.save(&mut session(state.clone()).await, &cfg, &request())
                .await
                .is_err()
        );
        assert!(state.lock().unwrap().drafts.is_empty());
    }

    #[tokio::test]
    async fn absent_special_use_folder_does_not_guess_or_create_a_folder() {
        let cfg = account();
        let tool = tool(Arc::new(Mutex::new(cfg.clone())));
        let state = Arc::new(Mutex::new(MailboxState {
            no_drafts: true,
            ..Default::default()
        }));
        assert!(
            tool.save(&mut session(state.clone()).await, &cfg, &request())
                .await
                .is_err()
        );
        assert!(state.lock().unwrap().drafts.is_empty());
    }

    #[tokio::test]
    async fn disabled_or_observer_mailbox_is_refused_before_connecting() {
        for config in [
            EmailConfig::default(),
            EmailConfig {
                observer_mode: true,
                ..account()
            },
            EmailConfig {
                allow_drafts: false,
                ..account()
            },
        ] {
            let tool = tool(Arc::new(Mutex::new(config)));
            let error = tool
                .execute(serde_json::to_value(request()).unwrap())
                .await
                .unwrap_err();
            assert!(error.to_string().contains("not permitted"));
        }
    }

    #[test]
    fn bounds_and_header_injection_are_checked_before_io() {
        let mut draft = request();
        draft.subject = "Subject\r\nBcc: unintended@example.test".into();
        assert!(draft.validate().is_err());
        let mut draft = request();
        draft.to = vec!["recipient@example.test\r\nBcc: other@example.test".into()];
        assert!(draft.validate().is_err());
        let mut draft = request();
        draft.body = "x".repeat(MAX_BODY_BYTES + 1);
        assert!(draft.validate().is_err());
        let mut draft = request();
        draft.to.clear();
        assert!(draft.validate().is_err());
        let mut draft = request();
        draft.reply_to_uid = Some(0);
        assert!(draft.validate().is_err());
        assert!(serde_json::from_value::<DraftRequest>(json!({"channel":"personal","request_id":"one","to":["a@example.test"],"subject":"s","body":"b","reply_to_uid":4294967296u64})).is_err());
    }

    #[test]
    fn defaults_keep_drafts_closed_and_observer_mode_wins() {
        assert!(!EmailConfig::default().allow_drafts);
        assert!(account().can_save_drafts());
        assert!(
            !EmailConfig {
                observer_mode: true,
                ..account()
            }
            .can_save_drafts()
        );
        assert!(
            !EmailConfig {
                enabled: false,
                ..account()
            }
            .can_save_drafts()
        );
    }
}
