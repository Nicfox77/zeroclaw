# Hindsight long-term memory

Hindsight is an opt-in, exclusive long-term-memory backend. The SQLite native
conversation archive remains the canonical transcript. The rolling summary deck
continues to manage active context independently. Enabling Hindsight does not
convert, rewrite, or delete either of those stores.

This integration is fork-specific. It deliberately selects Hindsight as the
long-term-memory authority; it is not an enrichment connector that also writes
SQLite memories. Upstream adoption needs review against
[ADR-010](../architecture/decisions/ADR-010-memory-authority-boundaries.md).

## Data flow and ownership

1. Native session admission registers a delivery cursor before execution starts.
2. The existing native engine commits the user input and final assistant response.
3. A background worker advances bounded, indexed event pages and durably enqueues
   transcript references. It reads original inputs, admitted steering messages,
   and committed final output,
   not the provider prompt or its injected memories.
4. Hindsight extracts facts and performs its native observation consolidation.
5. Before a normal conversational response, the shared memory-injection path
   retrieves evidence using the question and bounded immediate chat context.
   Adaptive reflection routes deeper historical questions through Hindsight.
6. Retrieved evidence uses the existing structured, retained memory blocks. It is
   not ingested again as new source material.

### Automatic recall during live voice

Live SimpleX calls select a compact automatic recall policy from the incoming
call scope. Each turn asks Hindsight for its `low` retrieval budget and up to
256 tokens, honoring a smaller configured `recall_max_tokens` value. The prompt
renderer allows up to two memories within 1024 characters including entry labels.
Character limits are approximate token budgets because conversational models use
different tokenizers.

Live calls query the authorized Hindsight bank once. The legacy sender and
conversation session IDs resolve to that same bank, so they share one lookup.

Compact automatic recall uses native retrieval for historical questions too.
The agent can call its explicit memory and reflection tools when deeper evidence
is needed. Voice notes and text messages use the standard automatic policy.
This selection is resolved for each turn and adds no persistent configuration.

### Growing conversation documents

Each agent/principal/conversation/branch has one stable Hindsight document. The
background worker sends only the next completed turn with `update_mode: append`.
Its content is a JSON array of messages with `role`, `speaker`, `content`, original
RFC 3339 `timestamp`, `event_id`, and `turn_id`. Hindsight merges these arrays into
one valid conversation array. It does not receive a growing full transcript from
ZeroClaw on every request. Delivery UUIDs remain per turn and are reused after
uncertain acknowledgments. Appends are serialized per stream; backoff and manual
retry cannot let a newer turn overtake an older one. Historical imports enqueue
before live capture resumes for that stream.

Each message's original text is redacted and screened before JSON encoding.
Screening does not join message boundaries or treat JSON-escaped newlines as
ordinary characters. Disallowed source text still blocks delivery; a rejected
turn remains in the durable queue and prevents later appends from overtaking it.

Speaker labels resolve at delivery time from dedicated structured configuration:

```toml
[operator]
display_name = "Alex"

[agents.assistant]
display_name = "Nova"
```

These settings create display metadata only. They never grant access, join
accounts, or change principal IDs, bank IDs, or session ownership. The operator
name applies only to the implicit installation operator; unrelated principals
keep `User`. An empty agent display name falls back to the selected AIEOS
`identity.names` (nickname, full name, first name, last name), then the agent
alias. Runtime identity prompts and speaker labels share the same AIEOS name
normalization. Explicit display names override AIEOS naming for these transcripts;
AIEOS remains canonical when there is no override. Missing operator names use
`User`. No name is inferred from Markdown or prose, and no name cache is kept.
Changing or reformatting `IDENTITY.md` / `USER.md` cannot change these labels.

The summary engine receives the same resolved names and its existing bounded
identity references. Original transcript speakers keep names and roles; only
newly generated memories and summaries use the assistant's first-person voice.
AIEOS describes the agent, not its human operator; its `metadata.username` is
also an agent identifier. Adopting AIEOS does not fill the operator setting.

Hindsight requires source-text storage for append mode (upstream defaults to
`HINDSIGHT_API_STORE_DOCUMENT_TEXT=true`). Its stored conversation is a derived
memory source; SQLite remains canonical. Hindsight chunks the combined document
and skips unchanged chunks. Extraction sees the chunks being processed, **not the
entire endless conversation on every append**. JSON message boundaries provide
neighboring context within those chunks; they do not guarantee cross-chunk
reasoning.

Prompt caching and delta extraction are separate optimizations. In tested
Hindsight 1.2.0, extraction supplies a stable instruction/schema prefix to its
provider caching interface; changing bank missions and source content remain in
the request body after that prefix. Gemini can use explicit cache handles,
Anthropic uses cache markers, and compatible automatic-cache providers can reuse
matching leading tokens. MiniMax uses the OpenAI-compatible adapter: actual cache
hits and billing depend on MiniMax, and no cache-hit rate is promised. Hindsight
records cached-token usage when the provider reports it. The local MiniMax
validation reported nonzero cached tokens for `retain_extract_facts`; this
confirms cache use in that run, not a guaranteed rate for future requests. This is not one ongoing
provider conversation or a guaranteed conversation-wide KV cache.

Source erasure fences retrieval for the bank, drains in-flight appends, deletes
the affected conversation document, and replays surviving canonical turn
references in order under a new durable operation generation. It then clears and
refreshes mental models before unfencing retrieval. Rebuild intent, phase, and
UUIDs survive restarts; failed replay does not expose stale evidence or switch
backends.

**Upgrade from the earlier per-turn document format:** do not merely remove the
turn suffix from an existing delivery receipt. Back up the Hindsight PostgreSQL
store and delivery ledger, stop ingestion, delete only the explicitly selected
agent/principal bank through Hindsight's native bank API, and reset that bank's
conversation delivery receipts for an explicit chronological backfill. Preserve
the agent/bank identity mapping, canonical archive, and summary data. This
operator-approved rebuild intentionally discards derived memories in that bank,
including manual notes/models; export anything that cannot be reconstructed
before using it. An old pending receipt whose document ID no longer matches
fails with `hindsight_document_format_requires_explicit_reimport` rather than
silently replacing a different document.

`<data_dir>/hindsight/delivery.db` holds agent/bank identities, transcript cursors,
operation UUIDs, ingestion status, erasure frontiers, and backfill intents. It has
no vectors, memory-search implementation, or substitute recall path. Standalone
explicit notes additionally spool their screened delivery body until extraction
completes, because they need not have a corresponding conversation record. That
body is removed on completion. Ordinary conversational jobs store references only. Keyed
note edits are delivered in their accepted order, so an older outage retry cannot
overtake a newer edit. Repeating a pending latest value reuses its operation. A fresh write after
completion, or a change back to an earlier value, creates a new delivery intent;
a completed receipt is not treated as proof that UI-managed content still exists.

The worker commits its operation UUID before HTTP submission. Uncertain responses
are reconciled against that UUID; a missing operation can be resubmitted with the
same ID. HTTP acceptance is not counted as completed retention. Failed jobs retain
backoff and diagnostics. At capacity, the durable source cursor stays behind the
archive instead of discarding turns or delaying a completed reply.

A bank is scoped by a persistent agent UUID, principal, namespace, and tenant.
Session IDs are provenance, not observation-isolation tags. Explicit personal
connection bindings in `agent.main_session` can join the operator's bank across
Web, Telegram, and SimpleX. Other admitted channel peers get separate identities.
Removing a binding or disabling an agent is checked through live policy. Tools
cannot select arbitrary banks. RPC callers use their own private plane; Hindsight
has no shared-plane override. Paired dashboard routes use the installation operator.

Tool output, system prompts, summaries, and hidden reasoning are excluded from
automatic retention. ACP/Code sessions remain isolated from automatic memory capture and injection.
Internal delegation, cron, and daemon turns are not treated
as new user-authored conversations. Explicit memory notes remain possible.
Recognized credentials are redacted using the existing secret redactor; ordinary
memory scanning, namespace policy, and optional audit also apply.

A changed native erasure frontier blocks retrieval until retained source
references have been checked again. A removed source document is deleted through
Hindsight, including its derived memory links/observations. An in-flight retain is
allowed to reach its terminal state before document deletion, avoiding the race
where a cooperatively cancelled worker writes facts after deletion. The bank stays
blocked during that reconciliation and until native full refreshes of its mental
models complete, including manually refreshed models. Refresh operation receipts
survive a restart; no mental-model definitions are copied into the delivery ledger.
Erasure does not fall back to SQLite recall. Same-frontier validated appends extend
the receipt atomically, and background frontier checks use bounded indexed
queries; ordinary turns do not rescan completed delivery history. The canonical
catalog's temporary-session discovery also uses indexed metadata events rather
than materializing provider progress, and its physical-cleanup probe uses a
partial index of redacted payloads. Neither index changes transcript contents or
session lifetime rules.

The dashboard's Memories tab reports the selected backend and delivery health.
This example uses synthetic queue values and no private memory content:

![Hindsight status in the ZeroClaw dashboard](./images/hindsight-dashboard.png)

## Explicit provisioning

Requirements: Python 3.12, `uv`, the Hugging Face `hf` CLI, a supported Linux
runtime for pg0, and sufficient RAM/disk for the model caches. No GPU is required
for the verified CPU configuration.

Run the explicit provisioner:

```sh
./dev/setup-hindsight.sh /srv/zeroclaw-hindsight
```

Choose a directory writable by your service user. The default, when no argument
is supplied, is `$XDG_DATA_HOME/zeroclaw/hindsight` or
`$HOME/.local/share/zeroclaw/hindsight`.

The script pins `hindsight-embed==1.2.0`, `hindsight-api==1.2.0`, pg0 0.16.1,
CPU PyTorch 2.15.0, and tested inference-library versions in separate API and
decision virtual environments. It downloads these immutable model revisions:

| Component | Model | Revision |
| --- | --- | --- |
| Embeddings | `hindsight-ai/hs-embed-1` | `3164caa4db95a03faac8a327f914f7c8f42d2f98` |
| Ordinary reranking | `hindsight-ai/hs-rerank-1` | `927f5be31280298a51cf29af8a29cb111ed812c5` |
| Decisions | `hindsight-ai/hs-decide-1-flash` | `e2c789026dfd1f7608cafb2ef6a9bb31397af802` |

Runtime startup uses cached models and offline mode. Missing packages or weights
are an actionable startup failure, not an implicit download or hosted fallback.

## Optional Hindsight web interface

The official Control Plane can inspect banks, facts, source documents,
observations, mental models, and retrieval results. Provision its pinned package
alongside the API, without replacing embedded PostgreSQL:

```sh
ZEROCLAW_HINDSIGHT_WITH_UI=1 ./dev/setup-hindsight.sh /srv/zeroclaw-hindsight
/srv/zeroclaw-hindsight/control-plane/node_modules/.bin/hindsight-control-plane \
  --hostname 127.0.0.1 --port 9999 --api-url http://127.0.0.1:9077
```

The UI is a separately running upstream component. Bind it to loopback and use
an authenticated private tunnel or SSH forwarding for remote access. It exposes
memory-management authority over the connected local API, not ZeroClaw's
per-principal tool interface. Do not expose it as a public dashboard or give
unrelated users access. `HINDSIGHT_CP_ACCESS_KEY` can additionally protect the UI;
store it in a protected service environment file, not command arguments.

ZeroClaw owns backend selection, principal/bank bindings, ingestion policy, and
managed-process provider credentials. Hindsight owns bank missions and stored
memory content. The integration does not overwrite bank missions on every turn.
Change bank retain/observation/reflection missions in Hindsight's Configuration
tab; change process/provider settings through ZeroClaw and reload its daemon.
Explicit mental models declared in ZeroClaw remain configuration-managed; create
additional UI-owned models outside those declared IDs.

If Hindsight's own UI/manager should own process-wide provider settings as well,
choose `mode = "external"` and let that manager supervise the API. Automatic
retention, recall, reflection, and ZeroClaw bank authorization still operate.
Avoid having ZeroClaw and the Control Center both supervise or rewrite the same
managed profile. Managed mode remains the recommended initial setup here.

Retain, observation, and reflection missions inherit the shared first-person
guidance by default; explicit bank missions edited in Hindsight override those
process defaults. With `identity_context = true`, the installation operator's
bank also receives current bounded `IDENTITY.md`, `USER.md`, and `SOUL.md`
excerpts from its owning agent's workspace (1024 bytes per file). The shared
workspace briefing reader rejects escaping links and invalid files. Credentials
are redacted before delivery. Other principals never receive this profile.

The reference guides naming and voice, separately from source conversation
evidence. ZeroClaw updates only its marked reference block in each mission;
operator-authored mission text is preserved. File edits are resolved again at use
time. Disabling the setting removes the block on the next operation on the
operator's bank. The full system prompt, stored memories, and hidden reasoning
are excluded. Speaker attribution remains explicit.

## Separate decision server

The decision model is not an in-process feature of `hindsight-embed`. The
provisioner installs the official model repository's serving package and writes a
small launcher that reads its key from a mode-0600 file. The key is not placed in
the operating-system command line. It binds to `127.0.0.1:8000` and provides the
required `/v1/systemone` protocol.

For a user service, save the following as
`~/.config/systemd/user/zeroclaw-hindsight-decision.service`, adapting both absolute
paths to the provisioned directory:

```ini
[Unit]
Description=Hindsight local CPU decision model

[Service]
Type=simple
ExecStart=/srv/zeroclaw-hindsight/decision-runtime/bin/python /srv/zeroclaw-hindsight/serve-decision.py
Environment=HF_HUB_OFFLINE=1
Environment=CUDA_VISIBLE_DEVICES=
Environment=HIP_VISIBLE_DEVICES=
Environment=OMP_NUM_THREADS=4
Environment=MKL_NUM_THREADS=4
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```sh
systemctl --user daemon-reload
systemctl --user enable --now zeroclaw-hindsight-decision.service
```

The CPU Transformers server is an official upstream implementation of the decision
protocol. For vLLM, use the serving plugin documented in the
[model repository](https://huggingface.co/hindsight-ai/hs-decide-1-flash).
Stock vLLM's OpenAI endpoint does not implement `/v1/systemone`. This fork has not
validated the vLLM plugin on ROCm or the host AMD GPU. Do not change drivers,
reallocate another service's GPU, or claim GPU support based on the CPU tests.

### Verified Intel integrated GPU option

On Linux x86-64 with an accessible Intel Arc integrated GPU, provision the opt-in
XPU runtime after the ordinary setup:

```sh
./dev/setup-hindsight-xpu.sh /srv/zeroclaw-hindsight
```

It installs a separate PyTorch 2.15.0 XPU environment and verifies pinned Intel
compute-runtime 26.40.40196.8, IGC 2.42.6, GMM 22.10.2, and Level Zero 1.34.0
packages before extracting them privately. No system packages or kernel drivers
are replaced. The official Hindsight serving package still owns inference and
`/v1/systemone`. The XPU launcher uses float32 and math attention: the default
bf16/optimized path failed on the tested Meteor Lake GPU.

In the decision service above, replace `ExecStart` and add its environment file:

```ini
ExecStart=/srv/zeroclaw-hindsight/decision-xpu-runtime/bin/python /srv/zeroclaw-hindsight/serve-decision-xpu.py
EnvironmentFile=/srv/zeroclaw-hindsight/decision-xpu.env
```

Do not retain a second `ExecStart` in the same service. Reload systemd and restart
that service. An unavailable configured XPU is an explicit failure, not an
implicit CPU fallback. Check the authenticated `/version` and `/v1/systemone`
endpoints before connecting Hindsight.

Measured warm synthetic decision latency on the tested Intel Arc integrated GPU:

| Input tokens | CPU, four threads | Intel XPU |
| --- | --- | --- |
| 422 | 1.36 s | 0.35–0.37 s |
| 3662 | 14.6 s | 2.9–3.0 s |
| 7262 | Not compared | 6.5 s |

Decision probabilities matched the CPU fixtures within floating-point tolerance.
These measurements exclude first-call kernel warmup and are not a quality
benchmark. Other devices, distributions, and vLLM/ROCm remain unverified.

## Configuration

Adapt this example to existing agent/provider aliases. Do not duplicate credentials
from an existing provider; `credential_provider` resolves its decrypted API key at
use time and does not change the conversational model, routes, or summary provider.

```toml
[memory]
backend = "hindsight"
rerank_enabled = false
consolidation_extract_facts = false

[memory.types]
enabled = false

[knowledge]
enabled = false

[channels]
session_persistence = true
session_backend = "sqlite"

[agents.assistant.memory]
backend = "hindsight"

[memory.hindsight]
mode = "managed"
api_url = "http://127.0.0.1:9077"
bank_template = "zeroclaw-{agent}"
startup_timeout_secs = 120
request_timeout_secs = 15
reflect_timeout_secs = 120
retain_async = true
retry_failed_retains = true
queue_capacity = 10000
auto_retain = true
auto_recall = true
auto_reflect = true
reflect_mode = "adaptive"
recall_budget = "mid"
recall_max_tokens = 2048
reflect_budget = "mid"
prefer_observations = true
identity_context = true

[memory.hindsight.runtime]
binary_path = "/srv/zeroclaw-hindsight/runtime/bin/hindsight-embed"
data_dir = "/srv/zeroclaw-hindsight"
package_version = "1.2.0"

[memory.hindsight.models]
embedding = "hindsight-ai/hs-embed-1"
reranker = "hindsight-ai/hs-rerank-1"
decision = "hs-decide-1-flash"
decision_enabled = true
decision_endpoint = "http://127.0.0.1:8000"
decision_recall_rerank = false

[memory.hindsight.llm]
provider = "minimax"
model = "MiniMax-M3"
credential_provider = "minimax.default"
```

Every configured agent must explicitly select `hindsight` or `none`, including
worker profiles. Validation rejects competing SQLite memory, local extraction,
local reranking, built-in knowledge-graph processing, and cross-agent shared-memory
grants. These restrictions do not disable the SQLite transcript or summary deck.

`llm.provider` follows upstream Hindsight provider names. For a custom
OpenAI-compatible server use `provider = "openai"`, its model, and an explicit
`base_url`. A direct `api_key` uses ZeroClaw's existing encrypted-secret handling.
Alternatively, `runtime.env_file` can identify an owner-only protected file of
supported `HINDSIGHT_API_*_LLM_{PROVIDER,MODEL,BASE_URL,API_KEY}` settings. This file
cannot override listeners, executables, or the database. Do not combine a direct
key and a credential-provider reference.

Per-operation overrides use the same LLM structure:

```toml
[memory.hindsight.llm_overrides.reflect]
provider = "openai"
model = "your-reflection-model"
credential_provider = "openai.default"
```

Supported operation names are `retain`, `reflect`, `consolidation`, and
`mental_model_refresh`. Newly authored extraction uses the shared first-person
memory guidance; original quoted source material is unchanged.

Run `zeroclaw memory setup` to write the official protected embed profile. Reload
the ZeroClaw daemon to apply backend changes. Managed mode verifies the installed
Python distribution versions, locks dedicated ownership, uses the official daemon
manager, checks health, and supervises subsequent restarts through that manager.
An initial startup failure leaves memory degraded and requires a daemon reload
after its reported cause is corrected. It records
which environment was actually started. Changed settings cannot silently continue
using an old managed configuration. Managed shutdown stops only its own profile.

For an independently operated Hindsight service, select `mode = "external"`.
ZeroClaw then neither starts nor stops that service. External TLS endpoints are
supported. Managed API binding is literal loopback HTTP; expose a proxy only as an
explicit operator decision. A failed Hindsight service never selects another
long-term-memory backend.

## Reflection and mental models

`reflect_mode` supports `off`, `adaptive`, and `always`. Adaptive routing uses the
current question, not phrases in unrelated retrieved context. It recognizes
historical decisions, reasons, evolution, earlier discussion, and cross-conversation
synthesis. Trivial conversation uses ordinary recall. Explicit `hindsight_recall`,
`hindsight_reflect`, and `hindsight_retain` use the same scoped, screened binding.
Disabling automatic recall does not disable those tools.

Only explicitly configured mental models are initialized:

```toml
[memory.hindsight.mental_models.project-architecture]
name = "Project architecture"
source_query = "What durable architectural decisions and constraints have we established?"
max_tokens = 2048
min_refresh_interval_seconds = 3600
```

Their native trigger uses `refresh_after_consolidation = true`. These are durable
knowledge pages, not rolling context summaries. Do not create a mental model per
turn or session. `decision_recall_rerank = true` changes upstream recall ordering to
the decision model; it does not add another Rust reranker after Hindsight.

## Deferred retention on a shared local model

Retention can wait for a daily window while chat continues to recall existing
memories. Keep `auto_retain = true`: that registers conversations and collects
bounded pages of canonical transcript references. Turning it off disables
capture rather than merely postponing extraction.

```toml
[memory.hindsight.retention]
deferred = true
window = "04:00-08:00"
timezone = "America/Los_Angeles"
```

The window is local wall-clock time in the named IANA timezone, including DST.
Its start is inclusive and end exclusive. Overnight windows such as
`"23:00-02:00"` are supported. Omit `window` for manual-only retention. The default
`deferred = false` preserves immediate/background delivery without a time gate.
These are live admission settings; each new submission reads current policy.

The existing delivery ledger remains the queue authority. Conversation bodies
stay in the native transcript; standalone explicit notes use the existing
private, screened note spool until delivery completes. `memory_store` and
`hindsight_retain` acknowledge the durable queue write in deferred mode rather
than waiting for extraction. Queued information becomes searchable only after
Hindsight finishes retaining it. Capacity limits leave the transcript cursor
behind rather than dropping conversation turns or blocking chat.

The daemon's existing delivery worker must be running. Local operator commands
can inspect or temporarily override its admission policy without starting a
second worker or making an LLM call:

```sh
zeroclaw memory retention status
zeroclaw memory retention drain --minutes 240
zeroclaw memory retention pause
zeroclaw memory retention resume
```

`drain` grants a bounded permission (default 240 minutes, maximum 1440), including
outside the daily window. The worker drains eligible work asynchronously and
can accept newly queued work during that permission. The grant expires by wall
clock and survives a process restart. It does not reset delivery failures or
bypass retry policy: `memory retry-pending` requests retry separately. `pause`
persists a stop on new retention admissions, even during the daily window;
`resume` clears the manual override and restores the configured schedule.
`retention status` is a local JSON view of the current policy, override,
admission state, queue counts and in-flight operation. It does not contact the
LLM or need a healthy Hindsight service. These are local administrative CLI
commands, not new remotely exposed slash commands. An authorized operator can
also ask their agent to run the command through its existing shell tool.

Deferred mode admits only one new retention operation at a time across notes
and conversations. Its claim is durable before submission. A lost HTTP
acknowledgement keeps that claim until reconciliation observes an upstream
terminal state; a local transport failure cannot release it and flood the
server with more work. Retries are gated too. Polling previously admitted
operations continues while paused/outside the window. Closing the window or
pausing does not preempt an inference already admitted to Hindsight, reclaim its
KV cache, cancel work queued by other clients, or disable Hindsight's own
consolidation. Allow time for the last admitted operation to finish.

Recall is independent of the gate. Reflection and privacy-required erasure/
mental-model cleanup also retain their existing behavior; those may invoke a
model outside the retention window. Redaction checks and erasure fences still
apply, and source text erased while queued is never sent later. A queued replay
needed after erasure waits for retention admission, so the affected bank can
remain fenced until its safe rebuild completes. To share one accelerated model,
point Hindsight's generative LLM at the same compatible endpoint/model as chat;
this policy does not load or choose another model.

Roll back by first pausing/stopping the delivery worker, then restoring the
previous binary/config. An older binary ignores the new admission settings and
control tables and will resume its original immediate delivery behavior. Keep
the ledger and native transcripts; never delete them to clear a backlog.

## Status, recovery, and historical import

```sh
zeroclaw memory status
zeroclaw memory health
zeroclaw memory retry-pending
zeroclaw memory list --agent assistant
zeroclaw memory get MEMORY_ID --agent assistant
zeroclaw memory backfill --agent assistant --session CONVERSATION_ID
zeroclaw memory backfill --agent assistant --since 2026-01-01T00:00:00Z --until 2026-02-01T00:00:00Z
```

Status reports API/database readiness, configured models, decision-server health,
managed-configuration application, banks, queue counts, backfill backlog, last
successful retention/recall/reflection, recent stable error keys, and durations.
It excludes credentials and memory text. The paired dashboard uses the same
backend for memory browsing/search and exposes `/api/memory/status`.

Historical backfill is explicit and resumable. It uses canonical native completed
turns, original timestamps, stable document IDs, and independent import cursors.
Date bounds select turn-completion timestamps. Replaying the same import does not
rewind live capture or create another retention operation for an existing source.
An unattributed source requires an explicit operator principal and still cannot
bypass source ownership checks. No historical import starts merely by enabling
Hindsight.

Native source erasure deletes retained documents. Forgetting a fact by its memory
UUID instead uses Hindsight's reversible invalidation, which prunes its derived
observations and excludes it from recall. A standalone keyed note can be removed
by deleting its deterministic source document.

If Hindsight is down, chat continues from session context and rolling summaries;
recall failure is recorded, and retained turns remain pending in the delivery
ledger. Restore the service, inspect health, then retry pending work. Do not remove
the ledger to clear an error: it owns idempotency and bank identity.

## Backups and re-embedding

Back up the canonical SQLite archive separately using its existing backup
procedure. Also preserve the Hindsight pg0 state directory, delivery ledger, and
protected profile/decision credentials. For a cold filesystem backup, stop the
ZeroClaw daemon, stop its dedicated API through `hindsight-embed -p PROFILE daemon
stop`, then stop that dedicated database with the official
`PG0_HOME=/srv/zeroclaw-hindsight/pg0 /srv/zeroclaw-hindsight/runtime/bin/pg0 stop --name PROFILE`
(adapt the configured runtime directory). Copy closed data files, not a live PostgreSQL data directory. Restore
PostgreSQL and the delivery ledger from a consistent point. Keep all backups
private. A restored archive can explicitly backfill missing memory sources.

Hindsight 1.2 has no public bank-wide re-embedding endpoint. `zeroclaw memory
reindex` therefore reports an explicit unsupported operation instead of claiming
that zero entries were successfully rebuilt. For an existing document, the
supported upstream operation is:

```text
POST /v1/default/banks/{bank_id}/documents/{document_id}/reprocess
```

It re-runs extraction with current engine settings and replaces that document's
facts. It is not an atomic embedding-dimension migration. For an embedding model
or dimension change, back up and stop the old service, provision a fresh dedicated
Hindsight data directory, archive the closed delivery ledger, and explicitly
backfill the canonical SQLite archive into the new installation. Never delete or
rewrite the conversation archive. Do not mix old/new embedding dimensions in a
populated bank or silently reuse completed-delivery bookkeeping against an empty
new database.

## Validation and remaining limits

The integration includes mock transport/security tests, durable ledger tests, and
opt-in tests against a real local service:

```sh
cargo test -p zeroclaw-config --lib hindsight
cargo test -p zeroclaw-memory --lib hindsight
cargo test -p zeroclaw-session-host --lib hindsight
ZEROCLAW_TEST_HINDSIGHT_URL=http://127.0.0.1:9077 cargo test -p zeroclaw-memory --lib real_local_retain -- --ignored
ZEROCLAW_TEST_HINDSIGHT_URL=http://127.0.0.1:9077 cargo test -p zeroclaw-session-host --lib real_completed_turn -- --ignored
```

The real tests require all provisioned components and an independently configured
generative LLM. They create isolated fixture banks and exercise native turn
commit, automatic retention, observation consolidation, recall, reflection,
principal isolation, and source-erasure propagation. They must not be reported as
passed when those services are absent.

Known limitations requiring further work or operator handling:

- Backfill currently handles native completed turns. Older imported legacy
  snapshots without recorded turn/owner provenance are not guessed into a bank.
- Bulk re-embedding/dimension migration requires the explicit upstream procedure
  above; the CLI does not invent an unsupported Hindsight API.
- Per-agent alias renaming and bulk principal/namespace purge are not implemented
  for Hindsight; unsupported operations fail explicitly.
- Backfill merges source event-time frontiers across sessions and resumes after
  backpressure. Worker fault tests cover lost acknowledgments, failed extraction,
  reopened ledgers, retry, in-flight erasure, and mental-model cleanup receipts.
- CPU decision serving and decision-reranked recall are verified. A three-query
  fixture measured ordinary recall at 32–90 ms and decision recall at roughly
  1.5 seconds. This is a latency smoke test, not a retrieval-quality evaluation;
  ordinary reranking remains the default. Intel XPU is separately verified above;
  vLLM/ROCm serving is unverified.

See upstream [installation](https://hindsight.vectorize.io/developer/installation),
[embed runtime](https://hindsight.vectorize.io/sdks/embed),
[models](https://hindsight.vectorize.io/developer/models), and
[decision models](https://hindsight.vectorize.io/developer/decision-models).

## Identity context in memory and summaries

With `identity_context = true`, Hindsight retention, observation consolidation
and reflection receive the selected agent identity as a reference for naming
and voice. When the agent selects `identity.format = "aieos"`, this uses the
same structured identity parser and renderer as chat. Bounded excerpts from
`IDENTITY.md`, `USER.md` and `SOUL.md` remain additive. Background session
summaries use this identity reference too.

The current agent configuration and identity files are read when preparing each
operation; a separate cached persona is not created. Identity reference text is
not conversation evidence, and existing first-person and speaker-attribution
rules still apply. Ordinary recall performs retrieval rather than a separate
LLM reflection call; its bank identity reference is refreshed before recall.
