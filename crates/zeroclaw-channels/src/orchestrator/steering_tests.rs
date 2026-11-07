// Included in the existing test module to reuse its real dispatcher fixtures.
struct SteeringGateTool {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    completed: Arc<AtomicBool>,
}
impl zeroclaw_api::attribution::Attributable for SteeringGateTool {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        MockPriceTool.role()
    }
    fn alias(&self) -> &str {
        "mock_price"
    }
}
#[async_trait::async_trait]
impl Tool for SteeringGateTool {
    fn name(&self) -> &str {
        "mock_price"
    }
    fn description(&self) -> &str {
        "Wait at a controlled tool boundary"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        MockPriceTool.parameters_schema()
    }
    async fn execute(
        &self,
        _: serde_json::Value,
    ) -> anyhow::Result<zeroclaw_api::tool::ToolResult> {
        self.started.notify_one();
        self.release.notified().await;
        self.completed.store(true, Ordering::Release);
        Ok(zeroclaw_api::tool::ToolResult::ok("completed once"))
    }
}

#[derive(Default)]
struct SteeringFinalPause {
    entered: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

struct SteeringBoundaryProvider {
    requests: Mutex<Vec<Vec<ChatMessage>>>,
    final_pause: Option<Arc<SteeringFinalPause>>,
    completed: Arc<AtomicBool>,
}
impl zeroclaw_api::attribution::Attributable for SteeringBoundaryProvider {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        ToolCallingModelProvider.role()
    }
    fn alias(&self) -> &str {
        "steering-fixture"
    }
}
#[async_trait::async_trait]
impl ModelProvider for SteeringBoundaryProvider {
    async fn chat_with_system(
        &self,
        _: Option<&str>,
        _: &str,
        _: &str,
        _: Option<f64>,
    ) -> anyhow::Result<String> {
        anyhow::bail!("fixture expects history")
    }
    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        _: &str,
        _: Option<f64>,
    ) -> anyhow::Result<String> {
        let call = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(messages.to_vec());
            requests.len()
        };
        if call == 1 {
            return Ok(tool_call_payload());
        }
        if call == 2 {
            assert!(
                messages.iter().any(|m| m.tool_result.is_some()),
                "completed tool result must reach the next model round"
            );
        }
        assert!(
            self.completed.load(Ordering::Acquire),
            "tool must finish before steering changes the model request"
        );
        if call == 2
            && let Some(pause) = &self.final_pause
        {
            pause.entered.notify_one();
            pause.release.notified().await;
        }
        Ok(format!(
            "Finished: {}",
            messages
                .iter()
                .rev()
                .find(|m| m.role == "user")
                .unwrap()
                .content
        ))
    }
}

struct SteeringReceivedHook(Arc<tokio::sync::Notify>);
#[async_trait::async_trait]
impl zeroclaw_runtime::hooks::HookHandler for SteeringReceivedHook {
    fn name(&self) -> &str {
        "steering-receive-fixture"
    }
    async fn on_message_received(
        &self,
        msg: ChannelMessage,
    ) -> zeroclaw_runtime::hooks::HookResult<ChannelMessage> {
        if msg.id == "follow-up" || msg.id == "newest" {
            self.0.notify_one();
        }
        zeroclaw_runtime::hooks::HookResult::Continue(msg)
    }
}

#[derive(Clone, Copy)]
enum SteeringCase {
    Steer,
    Queue,
    FinalRace,
    FinalRaceOrdered,
    Stop,
    Revoked,
    Debounced,
    OtherSender,
    ModeChange,
    Flood,
    Duplicate,
    EnableMidTurn,
}

async fn run_channel_steering_case(native: bool, case: SteeringCase) {
    use zeroclaw_config::message_arrival::MessageArrivalMode;
    let tmp = tempfile::TempDir::new().unwrap();
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let received = Arc::new(tokio::sync::Notify::new());
    let completed = Arc::new(AtomicBool::new(false));
    let final_pause = matches!(
        case,
        SteeringCase::FinalRace | SteeringCase::FinalRaceOrdered
    )
    .then(|| Arc::new(SteeringFinalPause::default()));
    let provider = Arc::new(SteeringBoundaryProvider {
        requests: Mutex::new(Vec::new()),
        final_pause: final_pause.clone(),
        completed: Arc::clone(&completed),
    });
    let channel = Arc::new(SendMessageRecordingChannel {
        channel_name: "discord",
        ..Default::default()
    });
    let mut config = Config {
        config_path: tmp.path().join("config.toml"),
        data_dir: tmp.path().join("data"),
        ..Default::default()
    };
    config.channels.discord.insert(
        "fixture".into(),
        zeroclaw_config::schema::DiscordConfig {
            enabled: true,
            message_arrival: Some(
                if matches!(case, SteeringCase::Queue | SteeringCase::EnableMidTurn) {
                    MessageArrivalMode::Queue
                } else {
                    MessageArrivalMode::Steer
                },
            ),
            ..Default::default()
        },
    );
    let mut agent = zeroclaw_config::schema::AliasedAgentConfig {
        enabled: true,
        model_provider: "openai.primary".into(),
        risk_profile: "fixture".into(),
        channels: vec!["discord.fixture".into()],
        ..Default::default()
    };
    agent.precheck.enabled = false;
    config.agents.insert("test-agent".into(), agent.clone());
    config.risk_profiles.insert(
        "fixture".into(),
        zeroclaw_config::schema::RiskProfileConfig {
            level: AutonomyLevel::Full,
            auto_approve: vec!["mock_price".into()],
            ..Default::default()
        },
    );
    config.peer_groups.insert(
        "fixture".into(),
        peer_group("discord.fixture", &["alice", "bob"], false),
    );
    let profile = config.providers.models.ensure("openai", "primary").unwrap();
    profile.model = Some("test-model".into());
    profile.api_key = Some("fixture-key".into());
    let live = zeroclaw_config::live::LiveConfig::new(config.clone());
    let mut hooks = zeroclaw_runtime::hooks::HookRunner::new();
    hooks.register(Box::new(SteeringReceivedHook(Arc::clone(&received))));
    let mut ctx = test_runtime_ctx_with_observer_and_tools(
        channel.clone(),
        provider.clone(),
        config,
        agent,
        "openai.primary",
        Some(Arc::new(hooks)),
        Arc::new(NoopObserver),
        vec![Box::new(SteeringGateTool {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
            completed: Arc::clone(&completed),
        })],
    );
    let inner = Arc::get_mut(&mut ctx).unwrap();
    inner.live_config = live.handle();
    if matches!(case, SteeringCase::Debounced) {
        inner.debouncer = Arc::new(zeroclaw_infra::debounce::MessageDebouncer::new(
            Duration::from_millis(10),
        ));
    }
    inner.workspace_dir = Arc::new(tmp.path().to_owned());
    inner.provider_runtime_options.zeroclaw_dir = Some(tmp.path().to_owned());
    if native {
        inner.native_sessions = Some(Arc::new(zeroclaw_session_host::NativeSessionHub::new(
            inner.live_config.clone(),
        )));
    }
    let make_message = |id: &str, text: &str, timestamp| ChannelMessage {
        id: id.into(),
        content: text.into(),
        sender: "alice".into(),
        platform_sender_id: Some("alice".into()),
        reply_target: "alice".into(),
        channel: "discord".into(),
        channel_alias: Some("fixture".into()),
        explicitly_addressed: true,
        conversation_scope: if matches!(case, SteeringCase::OtherSender) {
            zeroclaw_api::channel::ChannelConversationScope::ReplyTarget
        } else {
            zeroclaw_api::channel::ChannelConversationScope::Sender
        },
        timestamp,
        ..Default::default()
    };
    let first = make_message("initial", "Do the work", 1);
    let history_key = runtime_conversation_history_key(&ctx, &first);
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let router = AgentRouter::single(Arc::clone(&ctx));
    let dispatcher = zeroclaw_spawn::spawn!(run_message_dispatch_loop(rx, router, 1));
    tx.send(first).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), started.notified())
        .await
        .unwrap();
    if matches!(case, SteeringCase::EnableMidTurn) {
        let mut updated = live.handle().read().clone();
        updated
            .channels
            .discord
            .get_mut("fixture")
            .unwrap()
            .message_arrival = Some(MessageArrivalMode::Steer);
        live.publish(live.next_revision().unwrap(), updated)
            .unwrap();
    }
    if let Some(pause) = &final_pause {
        release.notify_one();
        tokio::time::timeout(Duration::from_secs(10), pause.entered.notified())
            .await
            .unwrap();
    }
    let mut followup = make_message("follow-up", "Also include the final detail", 2);
    followup.voice_origin = true;
    if matches!(case, SteeringCase::OtherSender) {
        followup.sender = "bob".into();
        followup.platform_sender_id = Some("bob".into());
    }
    tx.send(followup.clone()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), received.notified())
        .await
        .unwrap();
    if matches!(case, SteeringCase::Flood) {
        for i in 0..CONVERSATION_LANE_BACKLOG_LIMIT + 2 {
            tx.send(make_message(
                &format!("flood-{i}"),
                &format!("extra detail {i}"),
                3 + i as u64,
            ))
            .await
            .unwrap();
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    if matches!(case, SteeringCase::Duplicate) {
        tx.send(followup).await.unwrap();
    }
    if matches!(case, SteeringCase::FinalRaceOrdered) {
        // An unknown slash input uses normal admission, blocking steering of
        // newer inputs until that older message reaches its ordinary turn.
        tx.send(make_message("older", "/ordinary older detail", 3))
            .await
            .unwrap();
        tx.send(make_message("newest", "Newest detail", 4))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(10), received.notified())
            .await
            .unwrap();
    }
    // Let the runnable admission/queue tasks reach the parked tool.
    for _ in 0..16 {
        tokio::task::yield_now().await;
    }
    if matches!(case, SteeringCase::ModeChange) {
        let mut updated = live.handle().read().clone();
        updated
            .channels
            .discord
            .get_mut("fixture")
            .unwrap()
            .message_arrival = Some(MessageArrivalMode::Queue);
        live.publish(live.next_revision().unwrap(), updated)
            .unwrap();
    }
    if matches!(case, SteeringCase::Revoked) {
        let mut updated = live.handle().read().clone();
        updated.peer_groups.clear();
        live.publish(live.next_revision().unwrap(), updated)
            .unwrap();
    }
    if matches!(case, SteeringCase::Stop) {
        tx.send(make_message("stop", "/stop", 3)).await.unwrap();
        tokio::time::timeout(Duration::from_secs(10), async {
            while channel.sent_messages.lock().await.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
    if let Some(pause) = &final_pause {
        pause.release.notify_one();
    } else {
        release.notify_one();
    }
    drop(tx);
    tokio::time::timeout(Duration::from_secs(20), dispatcher)
        .await
        .unwrap()
        .unwrap();
    let requests = provider.requests.lock().unwrap().clone();
    if matches!(case, SteeringCase::FinalRaceOrdered) {
        let first_seen = |text: &str| {
            requests
                .iter()
                .enumerate()
                .find_map(|(round, messages)| {
                    messages
                        .iter()
                        .position(|message| {
                            message.role == "user" && message.content.ends_with(text)
                        })
                        .map(|index| (round, index))
                })
                .unwrap()
        };
        assert!(first_seen("Also include the final detail") < first_seen("/ordinary older detail"));
        assert!(
            first_seen("/ordinary older detail") < first_seen("Newest detail"),
            "a newer input must not steer ahead of an older completion-race fallback"
        );
        return;
    }
    if matches!(case, SteeringCase::Flood) {
        assert_eq!(requests.len(), 2);
        let users = requests[1]
            .iter()
            .filter(|message| message.role == "user")
            .count();
        assert!(
            users <= CONVERSATION_LANE_BACKLOG_LIMIT + 1,
            "steering must not expand the lane backlog: {users}"
        );
        let busy =
            zeroclaw_runtime::i18n::get_required_cli_string("channel-runtime-conversation-busy");
        assert!(
            channel
                .sent_messages
                .lock()
                .await
                .iter()
                .any(|message| message.content == busy)
        );
        return;
    }

    let contains_followup = |messages: &[ChatMessage]| {
        messages
            .iter()
            .any(|m| m.role == "user" && m.content.ends_with("Also include the final detail"))
    };
    if matches!(case, SteeringCase::Stop | SteeringCase::Revoked) {
        assert!(
            requests.iter().all(|messages| !contains_followup(messages)),
            "revoked/cancelled input must never reach the provider"
        );
        return;
    }
    let expected = if matches!(
        case,
        SteeringCase::Steer
            | SteeringCase::Debounced
            | SteeringCase::Duplicate
            | SteeringCase::EnableMidTurn
    ) {
        2
    } else {
        3
    };
    assert_eq!(
        requests.len(),
        expected,
        "steering must not restart a turn; native={native}"
    );
    assert!(contains_followup(requests.last().unwrap()));
    let cache = ctx
        .conversation_histories
        .lock()
        .unwrap()
        .peek(&history_key)
        .unwrap()
        .clone();
    assert_eq!(
        cache
            .iter()
            .filter(|m| m.role == "user" && m.content.ends_with("Also include the final detail"))
            .count(),
        1
    );
    assert!(
        channel
            .sent_messages
            .lock()
            .await
            .iter()
            .any(|m| m.content.contains("Also include the final detail"))
    );
    if native {
        let port = session_history::open(
            Arc::clone(&ctx),
            &make_message("inspect", "", 3),
            &history_key,
        )
        .await
        .unwrap()
        .unwrap();
        let archived = port
            .history_reader()
            .unwrap()
            .history(&history_key, 100)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            archived
                .messages
                .iter()
                .filter(|message| message.role == "user"
                    && message.text.ends_with("Also include the final detail"))
                .count(),
            1,
            "permanent history must contain the steered input independently of the rolling projection"
        );
        let restored = port
            .restore_context()
            .await
            .unwrap()
            .unwrap()
            .chat_projection();
        assert_eq!(
                restored
                    .iter()
                    .filter(|m| m.role == "user" && m.content.ends_with("Also include the final detail"))
                    .count(),
                1,
                "steered input must survive native SQLite hydration"
            );
    }
}

#[tokio::test]
async fn channel_steering_finishes_tool_and_delivers_voice_followup_in_same_turn() {
    for native in [false, true] {
        run_channel_steering_case(native, SteeringCase::Steer).await;
    }
}
#[tokio::test]
async fn channel_steering_queue_mode_keeps_followup_as_a_separate_turn() {
    run_channel_steering_case(false, SteeringCase::Queue).await;
}
#[tokio::test]
async fn channel_steering_final_response_race_preserves_unconsumed_message() {
    for native in [false, true] {
        run_channel_steering_case(native, SteeringCase::FinalRace).await;
    }
}

#[tokio::test]
async fn channel_steering_final_response_fallback_preserves_fifo_before_newer_input() {
    for native in [false, true] {
        run_channel_steering_case(native, SteeringCase::FinalRaceOrdered).await;
    }
}
#[tokio::test]
async fn channel_steering_stop_cancels_both_active_and_offered_followup() {
    run_channel_steering_case(false, SteeringCase::Stop).await;
}
#[tokio::test]
async fn channel_steering_rechecks_live_sender_authority_without_retrying_refused_input() {
    run_channel_steering_case(false, SteeringCase::Revoked).await;
}

#[tokio::test]
async fn channel_steering_preserves_debounce_before_consumption() {
    run_channel_steering_case(false, SteeringCase::Debounced).await;
}
#[tokio::test]
async fn channel_steering_does_not_merge_another_sender_into_active_turn() {
    run_channel_steering_case(false, SteeringCase::OtherSender).await;
}
#[tokio::test]
async fn channel_steering_mode_change_falls_back_to_queue_without_losing_input() {
    run_channel_steering_case(false, SteeringCase::ModeChange).await;
}

#[tokio::test]
async fn channel_steering_retains_lane_backlog_limit_while_tool_is_running() {
    run_channel_steering_case(false, SteeringCase::Flood).await;
}

#[tokio::test]
async fn channel_steering_repeated_transport_identity_is_archived_and_injected_once() {
    run_channel_steering_case(true, SteeringCase::Duplicate).await;
}

#[tokio::test]
async fn channel_steering_resolves_mode_from_live_config_during_active_turn() {
    run_channel_steering_case(false, SteeringCase::EnableMidTurn).await;
}

#[test]
fn channel_steering_cache_resync_keeps_followup_when_trimming_preserves_message_count() {
    let ctx = test_channel_ctx();
    let key = "steering-resync";
    let original = ChatMessage::user("original text").with_timestamp(1000);
    let old = ChatMessage::user("old input").with_timestamp(0);
    let known = vec![old, original.clone()];
    ctx.conversation_histories
        .lock()
        .unwrap()
        .put(key.into(), known.clone());
    let rendered = "fixture context\noriginal text";
    let followup = ChatMessage::user("also do this before original text").with_timestamp(1000);
    let history = vec![
        ChatMessage::system("fixture"),
        ChatMessage::user(rendered).with_timestamp(1000),
        followup.clone(),
    ];
    resync_history_after_trim(
        &ctx,
        key,
        &history,
        false,
        false,
        2,
        &known,
        Some("original text"),
        Some(rendered),
    );
    let cache = ctx
        .conversation_histories
        .lock()
        .unwrap()
        .peek(key)
        .unwrap()
        .clone();
    assert_eq!(cache.len(), 2);
    assert_eq!(cache[0].content, original.content);
    assert_eq!(cache[1].content, followup.content);
}
