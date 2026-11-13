# Hindsight memory without a session-history dependency

Hindsight is an exclusive long-term memory backend. Its LLM extracts and consolidates memories; the main agent receives automatic retrieval and can request deeper synthesis with `hindsight_reflect`. The agent does not receive `memory_store`, `memory_recall`, `hindsight_retain`, or `hindsight_recall`. Native deletion tools retain their existing policy checks.

This feature does not install the proposed native session-event history subsystem. Upstream SQLite sessions remain the authority for session ownership and deletion. Hindsight owns extracted memories. A delivery outbox in the session SQLite database temporarily owns pending extraction payloads; it is not searchable memory, a resume transcript, or a history export.

## Configuration

Start with an independently provisioned Hindsight service and a normal configured agent. Select `hindsight` both globally and for that agent, enable SQLite session persistence, and disable the local memory processors rejected by configuration validation. Other enabled agents must select Hindsight or `none`; cross-agent memory grants are unsupported.

```toml
[memory]
backend = "hindsight"

[memory.hindsight]
mode = "external"
api_url = "http://127.0.0.1:9077"
auto_retain = true
auto_recall = true

[channels]
session_persistence = true
session_backend = "sqlite"

[agents.assistant.memory]
backend = "hindsight"
```

Use the normal secret configuration mechanism for service credentials. Never commit credentials or a managed runtime profile. Managed mode additionally requires provisioning the pinned runtime/models and running `zeroclaw memory setup`; setup writes a protected profile, not a runtime installation. Start the daemon to run the background delivery worker. `zeroclaw memory status` reports service and delivery diagnostics; `zeroclaw memory retry-pending` schedules unfinished operations for reconciliation.

Hindsight reflection is an optional built-in: full tool selection includes it; minimal selection can add `hindsight_reflect` to `[tools].optional`. Permission policy still controls its availability. Turning off automatic recall does not restore manual recall or write tools.

## Recall dates

Automatic recall date stamps use the host system timezone, matching live message
timestamps. The backend converts the event timestamp (`occurred_start`), or the
mention timestamp (`mentioned_at`) when no event timestamp is supplied, before
rendering its calendar date. Timezone rules apply at the event's date, including
daylight saving time. Hindsight's stored timestamps and original transcripts are
unchanged; existing memories are localized when read. Missing timestamps remain
unstamped, and non-RFC3339 values retain their existing rendering behavior.

## Public source adapter

The public adapter accepts completed, successful turns from authenticated RPC Chat sessions and sender-scoped channels. Capture checks the admitted agent/principal against current session ownership and live channel admission policy. Anonymous legacy WebSocket sessions, unowned RPC sessions, ACP/Code, cron, nested work, and shared-room channel sessions are not automatic retention sources. Shared-room metadata currently records the latest sender, which is insufficient authority for a private memory bank.

The handoff stages the original admitted request and accepted final response. Tool output, hidden reasoning, injected retrieval context, and intermediate provider messages are excluded. Capture timestamps are explicitly identified as capture times, not original speech or message timestamps. Native history remains necessary for full-fidelity steering/intermediate-message replay.

A stable turn identity makes repeated capture idempotent. The SQLite outbox survives restart and context trimming, including deferred extraction windows. The HTTP delivery ledger reconciles uncertain acknowledgments and remote asynchronous operations before retrying. Payloads are released after acknowledgment; identity receipts remain for deletion reconciliation. Queue capacity is bounded. A failed queue write rolls back the final session mutation too. RPC logs `hindsight_turn_commit_failed`; channels report a session persistence failure and do not advance the reply cache. The existing response delivery path can still deliver the generated reply, so operators must investigate persistence failures; this is not a guarantee that every displayed or streamed response was saved. There is no silent historical backfill.

The accepted final session mutation and its outbox entry share one SQLite transaction in `sessions/sessions.db`, with `synchronous=FULL`. A process crash before commit leaves neither write; after commit both survive and the worker can resume delivery. RPC commits the final conversation snapshot and trim provenance together with the payload. Sender-scoped channels commit the final assistant message together with the payload before advancing their reply cache. Earlier input or intermediate messages can already exist independently; the atomic unit is the final completed turn and its extraction payload, not network response delivery.

The earlier standalone prototype used `hindsight/transcript-outbox.db`. On startup its pending payloads and identity receipts are copied transactionally into the session database before the old rows are retired. Repeating an interrupted migration preserves identities and does not restore acknowledged text. Stop the previous process before upgrading; the native downstream adapter does not use this migration.

Deleting a whole source session invalidates source visibility and schedules remote document erasure. Recall fails closed until the affected source audit completes. Changing owner, revoking admission, or an unreadable source also fails closed. Transcript trimming is not memory erasure. Public upstream does not supply individual-turn erasure/replay authority, and acknowledged outbox payloads cannot be reconstructed for partial replay; such a request fails explicitly. Use Hindsight memory deletion for individual extracted memories.

## Backend and host boundary

`hindsight::source::TranscriptAuthority` supplies admitted execution scope and synchronous source-visibility checks. Explicitly principal-bound memory calls remain the preferred API. The unbound authority rejects reads of banks with transcript sources. `hindsight::delivery::TranscriptSource` supplies resumable source references, payload loading, and visibility checks. Separating visibility from payload loading lets the public outbox release acknowledged text without pretending that text is a permanent archive.

A host with native events can keep its existing event-backed delivery source and provide its own authority through `create_memory_for_agent_live_with_authority`. No native event types, installation guards, session catalog, history schema, or history migration are required by the shared Hindsight backend. Stable bank, document, stream, and operation identities are unchanged for existing native hosts.

The private downstream uses this seam to preserve its canonical event capture, original timestamps, steering records, historical imports, and erasure/replay support. It does not enable the public adapter in parallel and does not re-ingest existing conversations.

## Prompts and migration

Hindsight extraction and identity orientation are supplied to Hindsight's own service configuration and bank mission. The main agent's generated prompt does not assign backend memory-storage duties when no memory-write tool is registered. File-based `MEMORY.md` remains available in permitted persistent-memory contexts as an editable curated notepad. It complements automatic transcript memory; use file tools to maintain concise important notes rather than manual Hindsight retain calls. Isolated contexts continue to exclude the file. Untouched stock workspace instructions are rendered without manual memory duties; customized workspace files remain user-owned and must be reviewed explicitly for old `memory_store`, `memory_recall`, daily-note, and `MEMORY.md` instructions.

Selecting Hindsight does not import or delete an existing local memory database. There is no public `memory backfill` command because mutable transcripts cannot prove historical completeness. Keep the previous backend and configuration for rollback. Disabling this feature stops new capture; preserve the protected Hindsight directory if outstanding deliveries need reconciliation.

## Review boundaries

This is a backend proposal, not an implementation or acceptance of the permanent-history RFC. ADR-010 remains proposed; this exclusive remote backend and its explicit completed-turn ingestion need review against that proposal's enrichment and transcript-authority requirements. Runtime changes are composition and existing-agent/tool wiring; the implementation resides in `zeroclaw-memory`, configuration in `zeroclaw-config`, and the strict ownership read in `zeroclaw-infra`. Upstream reviewers must resolve any holding-crate exception needed before merging.

### Curated context alongside automatic memory

Conversation memories form automatically through Hindsight; the main agent does
not need to issue storage calls. It can still maintain the workspace files for
particularly important facts that should be available in prompt context without
a retrieval match. Identity, personality and user facts belong in their selected
files; `MEMORY.md` holds other concise durable notes, decisions and open loops.
The file is loaded when persistent-memory prompt context is constructed, within
normal prompt budgets. Explicitly isolated contexts continue to omit it.
