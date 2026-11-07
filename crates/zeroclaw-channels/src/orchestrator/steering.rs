//! The conversation lane owns follow-ups until the runtime admits them. A
//! closed receiver or a turn finishing without another tool round leaves the
//! original pending turn in FIFO order; enqueue success alone never loses it.
use super::*;
use std::collections::VecDeque;
use zeroclaw_config::message_arrival::MessageArrivalMode;
use zeroclaw_runtime::agent::{SteeringAdmission, SteeringInput, SteeringPosture};

pub(super) fn arrival_mode(
    ctx: &ChannelRuntimeContext,
    msg: &ChannelMessage,
) -> MessageArrivalMode {
    let Some(alias) = msg
        .channel_alias
        .as_deref()
        .filter(|alias| !alias.is_empty())
    else {
        return MessageArrivalMode::resolve(
            None,
            ctx.interrupt_on_new_message
                .enabled_for_channel(&msg.channel),
        );
    };
    let config = ctx.live_config.read();
    macro_rules! mode {
        ($channels:expr) => {
            $channels.get(alias).map_or(MessageArrivalMode::Queue, |c| {
                MessageArrivalMode::resolve(c.message_arrival, c.interrupt_on_new_message)
            })
        };
    }
    match msg.channel.as_str() {
        "simplex" => mode!(config.channels.simplex),
        "telegram" => mode!(config.channels.telegram),
        "slack" => mode!(config.channels.slack),
        "discord" => mode!(config.channels.discord),
        "mattermost" => mode!(config.channels.mattermost),
        "matrix" => mode!(config.channels.matrix),
        "whatsapp" => mode!(config.channels.whatsapp),
        _ => MessageArrivalMode::Queue,
    }
}

pub(super) async fn run_turn(
    registry: Arc<ConversationLaneRegistry>,
    turn: Box<PendingTurn>,
    queue: &mut tokio::sync::mpsc::Receiver<Box<PendingTurn>>,
) -> VecDeque<Box<PendingTurn>> {
    let (tx, rx) = tokio::sync::mpsc::channel(CONVERSATION_LANE_BACKLOG_LIMIT);
    // Only routing identity is needed after the active turn moves into its
    // worker. Policy is always resolved from live config at use time.
    let active_ctx = Arc::clone(&turn.ctx);
    let active_msg = turn.msg.clone();
    let active_generation = turn.agent_generation;
    // This freezes execution identity, not authorization. Consumption checks
    // that the current binding still selects this same conversation.
    let active_history = runtime_conversation_history_key(&active_ctx, &active_msg);
    let mut retained = Vec::new();
    let mut deferred = VecDeque::new();
    let mut receiving = true;
    let mut running = Box::pin(
        std::panic::AssertUnwindSafe(Arc::clone(&registry).process_turn(turn, Some(rx)))
            .catch_unwind(),
    );
    loop {
        tokio::select! {
            biased;
            next = queue.recv(), if receiving && retained.len() < CONVERSATION_LANE_BACKLOG_LIMIT => {
                let Some(next) = next else { receiving = false; continue; };
                // The matcher borrows only the active routing facts, not its
                // permits or lifecycle registration.
                if !compatible_message(&active_ctx, &active_msg, active_generation, &next) {
                    deferred.push_back(next);
                    receiving = false;
                    continue;
                }
                let (input, taken) = input_for_turn(&next, &active_history);
                match tx.try_send(input) {
                    Ok(()) => retained.push((next, taken)),
                    Err(_) => { deferred.push_back(next); receiving = false; }
                }
            }
            result = &mut running => {
                if result.is_err() {
                    ::zeroclaw_log::record!(ERROR, ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail).with_outcome(::zeroclaw_log::EventOutcome::Failure), "Channel message worker crashed");
                }
                break;
            }
        }
    }
    // Untaken inputs precede the first incompatible follow-up. Their original
    // registrations and global pending permits survive until actual execution.
    let mut pending = VecDeque::new();
    for (turn, taken) in retained {
        if !taken.load(Ordering::Acquire) {
            pending.push_back(turn);
        }
    }
    pending.extend(deferred);
    pending
}

fn compatible_message(
    ctx: &Arc<ChannelRuntimeContext>,
    msg: &ChannelMessage,
    generation: u64,
    next: &PendingTurn,
) -> bool {
    // Match borrowed routing facts without constructing a second PendingTurn
    // or copying its lifecycle state.
    next.agent_generation == generation
        && ctx.agent_alias == next.ctx.agent_alias
        && compatible_route(ctx, msg, next)
}

fn compatible_route(ctx: &ChannelRuntimeContext, a: &ChannelMessage, next: &PendingTurn) -> bool {
    let b = &next.msg;
    let enrich_links = next.ctx.live_config.read().link_enricher.enabled;
    arrival_mode(&next.ctx, b) == MessageArrivalMode::Steer
        && interruption_scope_key(a) == interruption_scope_key(b)
        && a.platform_sender_id == b.platform_sender_id
        && a.reply_target == b.reply_target
        && a.thread_ts == b.thread_ts
        && a.interruption_scope_id.is_none()
        && b.interruption_scope_id.is_none()
        && !a.passive_context
        && !b.passive_context
        && b.attachments.is_empty()
        && !b.content.trim().is_empty()
        && !b.content.trim_start().starts_with('/')
        && !(enrich_links && (b.content.contains("http://") || b.content.contains("https://")))
        && ctx.sop_engine.is_none()
        && ctx.sop_audit.is_none()
        && find_channel_for_message(&next.ctx.channels_by_name, b).is_some_and(|channel| {
            should_bypass_reply_intent_precheck(a, channel.is_direct_message(a))
                && should_bypass_reply_intent_precheck(b, channel.is_direct_message(b))
                && !channel.drop_self_messages(b)
                && !zeroclaw_runtime::peers::should_drop_self_loop(
                    &b.sender,
                    channel.self_handle().as_deref(),
                )
        })
}

/// The acknowledgement creates the consumption fact. It is deliberately
/// separate from channel acceptance so terminal races retain the original.
fn input_for_turn(next: &PendingTurn, history_key: &str) -> (SteeringInput, Arc<AtomicBool>) {
    let history_key = history_key.to_owned();
    let taken = Arc::new(AtomicBool::new(false));
    let consumed = Arc::clone(&taken);
    let ctx = Arc::clone(&next.ctx);
    let message = next.msg.clone();
    let cancellation = next.registration.as_ref().map(|r| r.cancellation.clone());
    let ingress = channel_ingress_context(&message);
    let mut chat = channel_user_history_message(&message, WHATSAPP_CURRENT_GROUP_MESSAGE_LABEL);
    // The enclosing turn already identifies this sender. Keep the actual
    // input literal; its independent ingress envelope owns attribution.
    chat.content = message.content.clone();
    let input = SteeringInput::with_admission(
        chat,
        Box::new(move || {
            if cancellation
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return SteeringAdmission::Refused("channel_steering_cancelled".into());
            }
            // Current authority refusal settles this input. Retrying it through
            // another turn must never revive the transport's original grant.
            if session_history::authorize_steering(&ctx, &message, &history_key).is_err() {
                consumed.store(true, Ordering::Release);
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject),
                    "channel_steering_authority_changed"
                );
                return SteeringAdmission::Refused("channel_steering_authority_changed".into());
            }
            if arrival_mode(&ctx, &message) != MessageArrivalMode::Steer {
                return SteeringAdmission::Refused("channel_steering_mode_changed".into());
            }
            SteeringAdmission::Admitted(SteeringPosture::default())
        }),
    )
    .with_ingress(ingress)
    .with_settlement(Box::new({
        let settled = Arc::clone(&taken);
        move || settled.store(true, Ordering::Release)
    }));
    (input, taken)
}
