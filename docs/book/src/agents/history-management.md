# History management

## Optional continuous main conversation

An agent can use one durable main conversation across authenticated dashboard
connections and explicitly bound Telegram direct chats. Other agents and
unbound connections retain their normal session isolation. Enable this
separately from the summary deck:

```toml
[agents.assistant.main_session]
enabled = true
web = true

[agents.assistant.main_session.telegram]
# Bot alias = owner's numeric Telegram user ID (illustrative value).
personal = "10001"
```

Use your configured agent and bot aliases. The agent must already own
`telegram.personal`, and its existing peer allowlist must admit that user.
The binding uses Telegram's adapter-stamped numeric identity and only joins
that user's direct chat; groups, topics, other bots and display usernames do
not join. Dashboard connections must authenticate as a paired operator or
administrator. Joining a conversation grants no additional tool permissions.
All opted-in dashboard operator connections share this agent's main history;
use a separate agent for a separate person's conversation.

Gateway `require_pairing` must remain enabled for the Web binding. Both gateway
and channel session persistence must be enabled for their respective bindings.
They use the existing native session store under the
same data directory, with the ID `main-session-<agent-alias>`. The daemon
shares its execution coordinator across transports, so each turn refreshes
the latest committed context after acquiring its execution lane. Concurrent
messages are serialized; a stale browser or channel cache cannot replace a
newer connection's conversation. The dashboard adopts the server's canonical
session ID and loads its permanent history on reconnect. A Telegram exchange
is visible on dashboard history reload; this option does not add live
cross-channel transcript broadcasting.

The main conversation is exempt from automatic idle expiry. Telegram `/new`
and `/clear`, and dashboard deletion, refuse to reset it while the option is
enabled. New dashboard session IDs resolve back to the same conversation.
Summary rotation bounds the model's active context while original history
continues to grow on disk. This is a continuous logical conversation, not an
unlimited provider context window.

Configure this on a dedicated personal agent. Existing isolated sessions are
not merged automatically. Restart the daemon and reconnect clients after
changing bindings. Disabling `main_session.enabled` restores ordinary routing;
the main history stays retained and can be rejoined by enabling it again.
Explicit history erasure and installation backups remain separate operator
actions. The default is disabled, so existing installations keep their current
session routing.

## Optional asynchronous summary deck

Native sessions can retain recent verbatim turns plus a rotating deck of older
summaries. This policy is independent of permanent history: original messages,
tool exchanges, identities and timestamps remain in the append-only session
store when either raw context or an old summary leaves the provider request.

Enable it on the runtime profile used by your agent:

```toml
[runtime_profiles.personal.summary_deck]
enabled = true
summary_share = 0.15
trigger_ratio = 0.8
max_summary_tokens = 1000
timeout_secs = 60
# Optional configured model-provider alias; otherwise use the session route.
# provider = "openai.summarizer"
```

The default is disabled. Existing `max_context_tokens` and
`context_compact_ratio` remain the overall budget controls. An explicit zero
token budget disables automatic deck maintenance as well. Keep
`max_history_messages` at four or more when enabling the deck.

The implementation reserves 15% of the resolved input target for request
overhead, then subtracts the current system messages. `summary_share` divides
the remaining estimated token allowance between summaries and recent history.
Summarization starts at `trigger_ratio` of the recent-history allowance, or
the corresponding fraction of the whole-turn limit. Estimates use roughly
four characters per token; provider capacity enforcement remains authoritative.

One background job per session branch freezes an older range of completed
turns, preserving the two newest turns and their complete tool exchanges.
Inference runs without holding a SQLite transaction or the foreground turn
lane. Oversized source ranges are split into bounded inference chunks without
discarding their original messages. The generated summary is bounded and
checked against its captured provider result before publication. The oldest
cards leave the deck when its token allocation fills. Cards do not recursively
fold the entire old deck into every new summary.

Normal requests continue while inference runs. At the recent-context or turn
limit, context preparation waits for the pending job. A failed, empty,
oversized, unavailable or unauthorized summary never authorizes deleting the
raw source. A protected turn that cannot fit still fails the normal capacity
guard; an unlimited logical conversation does not enlarge a model's physical
context window. Summaries are explicitly marked as generated historical data,
carried in the protected user-role context slot for provider compatibility.
They are never system instructions, approvals or long-term-memory writes.

Source identities, captured inference output and the installed deck are
durable session facts. A process crash cancels an unfinished local transform
during normal session recovery; originals remain available and a subsequent
request can schedule fresh summarization. It does not replay a provider or
tool effect. A completed deck survives restart without regenerating it.
Live policy and read/effect permissions are rechecked before publication.

Summaries incur additional model calls. Their provider attempts are captured
in session history and ordinary completed/failed attempt usage is recorded
against the agent and session in the existing cost ledger. The foreground
response's context meter still describes that response, not background usage.

Turning the policy off stops new automatic summaries and retains the currently
committed working context. Original history remains readable. Roll back the
feature on the same native history foundation; reverting the foundation's
one-way storage migration requires restoring its pre-migration backup, as
described in [session events](../architecture/session-events.md).

The runtime keeps conversation history for each agent session and sends a
provider-facing working history to the model. Two complementary limits operate
on different representations:

1. **Token-budget trimming** acts on the provider-facing `ChatMessage` working
   history and drops oldest whole turns until the estimated context fits the
   token budget.
2. **Whole-turn retention** applies the runtime profile's configured history
   limit to structured `Agent::history` (`ConversationMessage`) and the
   provider-facing `ChatMessage` history used by the legacy agent loop.

Both limits retain turns atomically. A turn starts at a real user message and
includes the assistant response and any tool calls and tool results before the
next user message. Trimming therefore does not split a tool call from its
result. A user message whose text happens to begin with `[Tool results]` is
treated as part of the turn before it, not as a new turn.

## Whole-turn retention

`history_trim::trim_to_recent_turns` enforces the token budget.
`history_trim::trim_to_recent_turn_count` and
`history_trim::trim_conversation_to_recent_turns` enforce the configured turn
limit for the two history representations. Each keeps the newest complete turn
even when the configured value is `0`. This is intentional: preserving a
complete current turn is safer than dropping its newest messages or breaking a
tool exchange.

Leading system messages are retained. When no trim is needed, message order and
shape are left unchanged.

## Token budget

The token budget comes from `ResolvedRuntime::effective_context_budget()`:

- Existing profiles retain the historical 32,000-token budget when neither
  `max_context_tokens` nor `context_compact_ratio` is set, capped by the
  selected model's configured capacity when that capacity is smaller.
- `max_context_tokens` remains an absolute budget. An explicit value of `0`
  disables proactive token-budget trimming.
- Setting `context_compact_ratio` opts into a model-relative budget: the
  selected provider alias/model's `context_window` (or a 32,000 fallback when
  its capacity is unknown) multiplied by the ratio. Values outside `(0.0, 1.0]`
  are treated as unset. When both settings are present, `max_context_tokens`
  is a downward cap on the ratio-derived budget.
- When `history_pruning.enabled` is set with a positive
  `history_pruning.max_tokens`, that value pulls the budget down (never up),
  letting operators trim earlier.
- Every positive effective budget is capped by the selected model's
  *configured* capacity. The 32,000 fallback is a compatibility stub, not
  model truth: when the provider profile declares no `context_window` and the
  runtime profile sets a positive `max_context_tokens`, that budget is honored
  and also becomes the window operand (so a ratio applies to it) instead of
  being clamped down to 32,000. The explicit zero sentinel remains zero and
  continues to disable proactive trimming.

The effective budget is a proactive trimming target, not a hard request limit.
After dropping all eligible older turns, the runtime retains the newest complete
turn even if it remains above that target. It sends the request when the prepared
messages, images, hooks, and tool schemas fit the resolved model context window.
A request that still exceeds that capacity fails before provider dispatch; raising
the proactive target alone cannot make it fit. This capacity check also applies
when proactive trimming is disabled (`max_context_tokens = 0`) and to the final
summary request after the tool iteration limit is reached.

Capacity and budget are resolved together for the active provider/model route.
Classifier hints and explicit session switches use the same route selection as
provider dispatch, so the next model call, proactive trim, overflow diagnostic,
cost attribution, and client context meter use the selected provider/model and
its resolved pair. If the selected model does not match the model configured on
that provider profile, capacity is treated as unknown instead of borrowing the
profile's metadata for a different model. The internal 32,000 compatibility
fallback remains available for safety calculations, but wire clients receive no
`model_context_window` value for an unknown capacity.

Token counts are estimated by `history::estimate_history_tokens`: roughly four
characters per token plus four framing tokens per message. This is a heuristic,
not a provider tokenizer. Loadable image attachments are charged a fixed
per-image cost only in messages whose images are dispatched: user turns and the
tool results of the current user turn. A tool result's images are the
attachments the tool declared, not marker text in its output. Older tool
results are priced as their message bytes, and system and
assistant text is priced as text.

Proactive token-budget trimming runs before the first provider call of a turn
when history already exceeds the effective budget and at provider-call
boundaries between tool-loop iterations. If a provider reports a context-window
overflow, the existing reactive recovery path retries after trimming to two
thirds of the current estimated history size. It does not derive a new recovery
target from model capacity. Both paths retain whole turns, so neither splits a
tool exchange.

## Whole-turn limit

`max_history_messages` is the legacy name of the history setting in an agent's
runtime profile. In structured agents and the legacy agent loop, its unit is a
complete user turn rather than an individual message row. Tool calls and tool
results remain part of their surrounding turn and do not consume independent
slots.

The default is `50` turns. An explicitly configured value is authoritative,
including `0`; the newest complete turn is still retained. The field name is
kept for schema compatibility and can be renamed in a future schema version.

Channel sender caches remain a separate exception: they count individual
message rows and preserve their existing `0`-means-default behavior.

## Visible trimming

Whenever token-budget trimming or the whole-turn limit drops older turns, the
runtime:

1. Inserts a breadcrumb before the first retained turn so the model knows that
   earlier context was omitted.
2. Emits `HistoryTrimmed` with the number of dropped messages, retained turns,
   and a reason identifying the token budget or turn limit.

The event is surfaced through the active client transport and through the
observer path used by dashboards and event subscribers. Trimming is therefore
not log-only and is not silent to either the model or connected clients.

The legacy `agent::run` path in `loop_.rs` reports count-based trimming through
logs only, without the breadcrumb or `HistoryTrimmed` event. This path serves
interactive use as well as one-shot and non-interactive daemon, cron, subagent,
and SOP callers.

## Tool context retention

`keep_tool_context_turns` (runtime profile, default `2`) is the number of
completed turns before the running one whose tool-call and tool-result rows are
still sent to the model. Every older turn is sent as its user prompt and its
closing assistant reply, so the model still sees what was asked and what was
answered without the raw tool output, and user and assistant rows keep
alternating on every provider. A turn that no reply closed (it ended on a call
or a result) is sent as its prompt and one
`[Tool exchange: N tool call(s) — results collapsed]` summary row, so the model
sees that calls were made rather than a prompt answered by nothing. The running
turn is always sent whole because the model needs its own results, so `0` keeps
tool context for the running turn only; the local-model preset's `1` keeps it
for one prior turn as well, and the `unbounded` preset uses `8`.

This is a request-time bound, not a transcript edit. The tool loop builds each
provider request from a collapsed copy; the working history and every store an
owner persists from it (channel sender cache, ACP transcript, CLI session file)
keep all their rows, so raising the value later shows older tool rows again. The
summary row is the pruning marker the providers already recognise, and it only
ever follows a user prompt, so no route sees two assistant rows in a row.

A turn collapses as a unit, in both native (`role=tool`) and prompt-mode
(`[Tool results]` carrier) shapes, so no orphan call or result is created. A
turn that ended on a call or a result keeps only its prompt and the summary. An
assistant row counts as a tool-call carrier when it names a tool the loop can
execute, the same test the loop applies before it runs a call, so an interrupted
exchange (a call whose results never arrived) is still summarised; a reply whose
JSON merely contains a `tool_calls` array naming no such tool, such as a quoted
API trace, executed nothing and is kept whole as the turn's reply.
Leading system messages and the trim breadcrumb are copied through. Rows the
loop itself appends inside a turn belong to the turn they interrupt and do not
open a new one for retention: its mid-turn feedback row (`[Tool call parse
error]`, user-role because no tool call id exists to attach it to) and the
prompt of each live SOP step (`[SOP: …]`), so a caller that resumes after its
steps still sends the tool round that opened its turn. Two loops that run on the
caller's own history as part of the caller's turn never collapse: live SOP steps
and the skill-review fork, since each step or review prompt is a user row and
the caller's turn would otherwise look older from inside them. The first request of a turn that collapses anything logs a `Tool
context collapsed` record with the row and turn counts; no `HistoryTrimmed`
event is emitted because no turn is lost.

On channels the same key also decides whether the current turn's tool rows are
persisted into the sender cache at all (`0` skips them).

## Pairing safety

Whole-turn retention is the primary tool-pairing guarantee: a tool call and its
result belong to the same turn and are retained or dropped together. The orphan
sweep remains a final safety net for histories that were already inconsistent,
such as restored or externally modified sessions.

Tool-result length limits are separate. `max_tool_result_chars` bounds an
individual result when it is recorded; it does not trim conversation history.
Provider-side context enforcement is also separate, though a provider overflow
can trigger the runtime's reactive token-budget trim.
