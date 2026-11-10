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

## Scheduled turns in the main conversation

For an agent with `main_session.enabled = true`, new agent cron jobs default to
`session_target = "main"` when created through the cron tool, dashboard/API, or
a declarative schedule without an explicit target. They use the same canonical
conversation as bound web and messaging connections: the scheduled prompt and
response enter permanent history and participate in ordinary summary rotation.
The next interactive turn can refer to the scheduled response naturally.

Main-target scheduled turns use the shared session's execution lane. A running
conversation turn finishes before a scheduled turn executes; the scheduled turn
refreshes committed context before requesting a model response. Delivery remains
separate: configure the job's announcement channel and recipient to receive it
in a messaging app. This does not add live transcript broadcasting to other
connected clients.

An explicit `session_target = "isolated"` retains a separate per-run history.
Existing imperative jobs retain their stored targets. Declarative jobs without
an explicit target follow this default when resynchronized from configuration.
Agents without the shared main conversation enabled keep isolated defaults and
the existing legacy main target. `uses_memory` separately controls long-term
memory access; setting it to false does not remove shared conversation context.
Use delegated workers for lengthy background work from the main conversation.
A synchronous `cron_run` call inside the same conversation refuses to wait on
its own active turn; perform the task directly or trigger the job from the
dashboard instead.
Previously isolated run histories are not automatically merged.

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
# Optional: prefill a dedicated loopback llama.cpp primary after publication.
warm_cache = false
max_summary_tokens = 1000
timeout_secs = 600
# Optional configured model-provider alias; otherwise inherit summary_provider,
# then context_compression.summary_provider, then the session route.
# provider = "openai.summarizer"
```

The default is disabled. Existing `max_context_tokens` and
`context_compact_ratio` remain the overall budget controls. An explicit zero
token budget disables automatic deck maintenance as well. Keep
`max_history_messages` at four or more when enabling the deck.

`timeout_secs` is an inactivity allowance, not a total generation deadline.
Summaries use streaming inference; nonempty text or reasoning refreshes the
allowance, while transport keepalives and empty frames do not. Cold prefill is
silent, so choose an allowance long enough for the slowest expected prefill
(for example, `1800` for a local model). The default is 600 seconds. The same
setting bounds waiting to publish a completed card while a foreground turn is
busy. The deck's explicit `provider` takes precedence over the agent's
`summary_provider`, then the runtime profile's compression summary provider;
with none configured, the current session route is used.

The implementation reserves 15% of the resolved input target for request
overhead, then subtracts the current system messages. `summary_share` divides
the remaining estimated token allowance between summaries and recent history.
Summarization starts at `trigger_ratio` of the recent-history allowance, or
the corresponding fraction of the whole-turn limit. Estimates use roughly
four UTF-8 bytes per token, including structured tool arguments and reasoning;
provider capacity enforcement remains authoritative. Source chunks, generated
responses and the retained deck use the same byte-based fallback.
Summary prompts state the corresponding UTF-8 byte ceiling and target 80% of it,
rather than asking the model to guess a tokenizer-independent token count.
The hard output check still rejects oversized cards without discarding sources.

One background job per session branch freezes an older range of completed
turns, preserving the two newest turns and their complete tool exchanges.
Inference runs without holding a SQLite transaction or the foreground turn
lane. Oversized source ranges are split into bounded inference chunks without
discarding their original messages. The generated summary is bounded and
checked against its captured provider result before publication. The oldest
cards leave the deck when its token allocation fills. Cards do not recursively
fold the entire old deck into every new summary.

New summaries are written from the conversational assistant's first-person
perspective, preserving attribution: “I checked the logs” for its own actions,
and “Alex asked me…” when the supplied context establishes the user's preferred
name. Otherwise the summary says “the user.” Plans, tool observations and other
agents' work must remain distinguishable from completed assistant actions.
Existing cards and original messages are not rewritten.

Background deck inference also receives current excerpts of the owning agent's
`IDENTITY.md`, `USER.md` and `SOUL.md`, capped at 1024 UTF-8 bytes per file.
These optional Markdown references guide names and voice, not claims about
conversation events or instructions to execute. They are read again for each
inference call; there is no second editable profile or live identity cache.
Normal captured provider requests retain the supplied briefing for audit.
Links outside
the agent workspace are rejected. Tool instructions, config, and the wider
memory store are not part of this briefing. AIEOS-only identity is not separately
rendered into this Markdown briefing; names present in the source conversation
remain available.

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

When normal session recovery proves that an execution process was lost, an
unfinished main-provider response with recorded visible text becomes an
assistant message in the active context, followed by an explicit
`[Interrupted: ...]` marker. The next turn can see what the assistant had
partially said. The marker does not claim that planned actions completed or
that every recorded fragment reached the user. Recovery does not replay tools,
resume generation, or turn an unknown outcome into a successful completion.
Reasoning, tool arguments and maintenance-model output are excluded. The
recovered message references the original text payloads, retaining their
erasure dependencies; repeated recovery does not add another copy. Existing
summary cards and prior context references remain in place.

SimpleX also finalizes orphaned live messages in its bounded recent-chat scan
after native contact authorization. A live message older than an hour is still
eligible, so a long turn does not leave a permanent live indicator after a
restart. This display cleanup is separate from restoring model context.

Summaries incur additional model calls. Their provider attempts are captured
in session history and ordinary completed/failed attempt usage is recorded
against the agent and session in the existing cost ledger. The foreground
response's context meter still describes that response, not background usage.

### Warming a local model after summary publication

Set `summary_deck.warm_cache = true` to opt into background prompt-cache warming
for a dedicated, single-slot (`--parallel 1`) loopback llama.cpp primary provider
using Chat Completions. Other provider families, a different session model
override, and previous fallback routes are skipped. This is independent of the
provider used to generate summaries.

After a card is committed, warming waits for the session's active turn to finish,
then takes its existing foreground lane. A later arriving turn waits for the
prefill to finish and can append to its warmed prefix. The warmer reads the latest
committed context and resolves the system prefix and tool definitions from the
last admitted main request in the same live authority/config epoch. It does not
maintain another editable prompt, tool registry or conversation history. If no
such request exists yet, that warmup is skipped; a future publication can retry.
Configuration, model-selection, context and authority changes cancel or skip
warming. No SQLite transaction is held over inference.

The ordinary provider request builder supplies the wire format. A bounded
one-token request fills the resident KV cache; its generated token is discarded,
never becomes an assistant message and never executes a tool. Some llama.cpp
builds sample one token even when zero are requested, so zero-token semantics
are not assumed. Actual prompt-processing progress renews the summary policy's
`timeout_secs` inactivity allowance; empty frames and keepalives do not. A failed
warmup releases the lane without invalidating an already committed summary.

Warming stops before the first image-bearing turn rather than independently
loading media paths or remote images. That text prefix remains reusable;
foreground multimodal preparation still authorizes and processes images normally.

For the primary profile, `provider_extra = { cache_prompt = true, id_slot = 0 }`
can explicitly select the resident slot. llama.cpp's default
`--sleep-idle-seconds -1` disables idle unloading. `--cache-ram 0` disables the
additional CPU prompt-cache pool, not the live slot's KV cache. No dummy chat
keepalives are needed: they could evict useful context. The cache stays resident
while the server remains running and another request does not replace its prefix.
A restart, model reload, changing system/tool prompts or another client can still
invalidate it; this does not add disk snapshots or guarantee cache residency
across processes. Separate sessions and other API clients do not share the
session's foreground lock.

Diagnostics record `llamacpp_cache_warm_started`,
`llamacpp_cache_warm_completed` (prompt/cache/processed token counts and duration),
and `summary_cache_warm_failed`, without prompt content. Warming is ephemeral and
does not replay on restart. Disable `warm_cache` and reload to cancel active work.

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

## Retained memory recall

Recalled memories remain in the working context for all model providers. Each
turn stores its selected snippets as a structured observation, separate from
the user's literal message. Provider dispatch renders that observation as the
existing `[Memory context]` block. Reconnects and restarts preserve it; normal
turn trimming and summary-deck rotation determine when it leaves active context.

Recall still runs against the new input using the existing scope, relevance,
filtering and per-turn budgets. An unchanged observation already in active
context is skipped before consuming the new-entry budget. Comparison uses the
latest retained observation for each key, including its selected text and date.
A changed value is appended with a later turn; earlier observations are never
refreshed in place. Once its annotated turn leaves active context, a memory can
be recalled again. Summaries may retain its meaning, but do not act as a second
deduplication index.

Between intentional context changes, adding a turn preserves earlier rendered
recall blocks. This avoids memory-recall prefix edits that undermine provider
prompt caching; cache availability and hits still depend on the provider.
Compaction, tool-context collapse, system-prompt changes and other request
changes can still invalidate a prefix. Retaining notes increases input tokens
and can bring rotation forward. Their rendered bytes count toward context and
summary budgets.

The original transcript and user-message memory autosaves contain the raw user
input, without injected recall. Current memory records and historical recall
observations have different owners: editing or deleting a memory record does
not rewrite a conversation that already received it. Native payload erasure
can hide a stored observation, and summaries depending on it become unavailable.
Erasing the original user source cannot resurrect it through a recall snapshot.
There is no provider-specific toggle or additional configuration for this policy.

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
UTF-8 bytes per token plus four framing tokens per message. Tool arguments,
reasoning and structured message metadata contribute to this estimate. This is a heuristic,
not a provider tokenizer. Loadable image attachments are charged a fixed
per-image cost only in messages whose images are dispatched: user turns and the
tool results of the current user turn. A tool result's images are the
attachments the tool declared, not marker text in its output. Older tool
results are priced as their message bytes, and system and
assistant text is priced as text.

Reported provider input-token usage calibrates the foreground request estimate
conservatively upward for the same provider and model. Switching either route
component resets that calibration. Usage describes the complete rendered request,
including system text, tool schemas and protocol framing; it is not an immutable
per-message token count. Different models can use different tokenizers, so stored
messages and summary budgets still need an estimate before dispatch. Background
summary usage remains in the cost ledger rather than calibrating a different
foreground request.

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
