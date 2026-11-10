# SimpleX

The optional `channel-simplex` adapter supports direct text, incoming files and images, voice notes, and incoming
encrypted audio calls from the SimpleX Android app. It uses the official
SimpleX CLI for local identities, encrypted messaging, and call signaling. A
separate Node.js/Chromium helper supplies virtual audio using SimpleX's
unmodified WebRTC implementation. No phone number, registered account, physical
microphone, or telephony provider is required.

## Install

Build ZeroClaw with `cargo build --release --features embedded-web,channel-simplex`.
Install Node.js 20 or newer and ffmpeg/ffprobe with AAC support, then prepare the helper from the repository root:

```sh
cd contrib/simplex
npm ci --ignore-scripts
node setup.mjs
npx playwright-core install chromium
```

The setup command verifies pinned upstream artifact digests and preserves the
SimpleX license. It downloads a Linux x86-64 CLI binary on that platform; other
platforms need their own compatible official CLI. SimpleX and its WebRTC assets
remain separately installed external software, with their upstream licenses.

Set `CHROMIUM_PATH` in the daemon's environment to the installed Chromium
executable. `SIMPLEX_CALL_ASSETS` can select the directory containing `call.js`
and `lz-string.min.js`; it defaults to `~/.local/share/zeroclaw-simplex`.
The headless helper requires Chromium's sandbox to work in the host environment.

Create a private directory for the SimpleX databases and run the CLI with a
loopback WebSocket API (for example, port 5225). Use a service with `UMask=0077`
and do not expose that API: the official CLI API has no authentication.

```sh
mkdir -m 700 -p "$HOME/.zeroclaw/data/simplex"
umask 077
"$HOME/.local/share/zeroclaw-simplex/simplex-chat" \
  -d "$HOME/.zeroclaw/data/simplex/agent" -p 5225 \
  --create-bot-display-name Assistant
```

Create a one-time invitation with the CLI's `/_connect <userId>` API, connect
from Android, and retrieve the resulting numeric contact ID with
`/_contacts <userId>`. Keep the invitation private. Use the actual local contact
ID in the configuration below; display names never authorize access.

## Configuration

```toml
[channels.simplex.personal]
enabled = true
websocket_url = "ws://127.0.0.1:5225"
bridge_command = ["node", "/absolute/path/to/contrib/simplex/bridge.mjs"]
calls_enabled = true
message_arrival = "steer"
max_attachment_bytes = 1073741824 # 1 GiB; optional, this is the default

[peer_groups.simplex_owner]
channel = "simplex.personal"
external_peers = ["17"] # example contact ID, replace after pairing
output_modality = "voice" # text plus a voice note for every reply

# Add "simplex.personal" to the owning agent's channels list.
# To share its existing permanent conversation across Web and other bindings:
[agents.assistant.main_session.simplex]
personal = "17"
```

The main-session option requires that agent's `main_session.enabled` and channel
session persistence. Back up the SimpleX database with the configuration: contact
IDs belong to that database. Restoring a different database requires rebinding
the owner. Membership does not replace transport authorization.

Channels and calls default disabled. Only explicit numeric peer grants are
accepted; wildcard grants do not authorize this adapter. Existing `ignore`
rules still override grants. Authorization, agent ownership, speech activation,
and speech-provider selection resolve from live config at use time. Restart or
reload the daemon after changing the bridge executable or CLI endpoint.

## Incoming files and images

Direct attachments from authorized contacts are downloaded through the official
CLI, with a configurable limit of 1 GiB per file by default. The bridge checks advertised and received sizes,
transfer identity, and the private destination before handing off a local path. Unauthorized
contacts cannot initiate downloads. Filenames cannot contain path separators or
control characters. Files are stored with private permissions in the owning
agent's workspace under `simplex_files/`; these durable files remain until the
operator removes them. Captions and image-only messages are supported.

`max_attachment_bytes` accepts 1 byte through 5 GiB. SimpleX's own app, CLI,
relay and account-tier limits still apply. Documents, archives and large files
stay on disk: only a small path reference crosses the bridge, and native code
streams a bounded copy into the owning workspace. It checks the live limit again
at completion. The bridge deletes its temporary copy after native acknowledgement;
a failed or stalled transfer is cleaned up after 30 minutes without progress.
Voice notes keep their separate transcription limits. Archives are transferred
intact; receiving a ZIP does not automatically extract or execute its contents.

SimpleX must also permit files for that contact. Keep the profile-wide file
preference closed when pairing a bot. Use the private CLI API to inspect the
contact's preferences, merge the file override into any existing overrides, then
apply them with `/_set prefs @<contactId> <preferences-json>`. For a contact with
no other overrides, the preferences JSON is `{"files":{"allow":"yes"}}`.
This controls the phone's attachment UI; it does not replace ZeroClaw's peer grants.

Supported images are detected from their bytes and enter the shared native
multimodal pipeline, rather than being replaced with an OCR description. The
selected provider must support images. A text-only model needs a configured
`multimodal.vision_model_provider` fallback. The fallback can receive subsequent
turns while images remain in active context, as well as the initial image turn.
Other attachments become local document references available to the agent's file
and document tools; this does not add automatic parsing of every document format
or execute uploaded files. Attachment transfer errors produce a retry notice.

## Live text replies

Live messages are the default for SimpleX. An admitted turn starts with
`Processing…` and keeps it while awaiting provider output, including prompt
prefill. It shows `Thinking…` only after the provider emits reasoning progress.
Providers that do not expose reasoning stay at `Processing…` until response
text arrives. Safe response text replaces the label in the same message as it streams. Updates are
coalesced to at most two per second, and the message leaves live mode when the
response completes. Completed narration before tools becomes a finished message;
the next response phase starts another live message. `channels.show_tool_calls`
continues to control tool details independently of these status labels.

These labels are transport status only: they never enter the conversation or
speech synthesis. Provider reasoning is not exposed. If outbound hooks must
approve content, or the reply is explicitly routed elsewhere, speculative text
is withheld until the shared delivery path approves it. Text finalizes before
waiting for a voice note, so text and speech do not delay one another.

Cancellation deletes an unused placeholder or finalizes visible partial text
with `[Interrupted]`. On reconnect, the helper discovers only recent sent live
items in bounded direct-chat history; native authorization must approve each
cleanup. It does not replay an answer after an uncertain final edit. Old items
outside that recovery window can remain unfinished. Update the helper and the
native binary together when upgrading this adapter.

## Voice notes

Incoming direct voice notes are downloaded only after checking the live peer
grant and transcription limits. They use the owning agent’s transcription
provider and enter the same conversation as text. `transcription.max_duration_secs`
sets the duration limit; the transport also caps each note at 5 MiB (or the smaller
`transcription.max_audio_bytes`). For ten-minute input, set the duration to 600.
The helper checks the actual file duration before transcription.

`peer_groups.<name>.output_modality` controls replies: `voice` sends both text and
a native M4A voice note, `text` sends text, and `mirror` follows the incoming
message type. This does not require `calls_enabled`. Speech is prepared from
approved streaming text and joined with one AAC encode when the final response
is ready. Before upload, the helper losslessly remuxes AAC into a finalized M4A
with a populated duration and sample index for Android MediaPlayer; fragmented
pipeline output is never uploaded directly. Final rewrites invalidate speculative
audio. No partial voice notes
are uploaded. Outbound hooks defer preparation until their final decision.

The helper stages audio in private temporary directories (0700, files 0600).
Incoming files are removed after transcription handoff; outgoing source files
remain until encrypted upload completes. Transfer errors, cancellation, timeout,
and helper shutdown clean up owned files. The temporary STT copy is plaintext;
the SimpleX transfer remains encrypted. A failed transfer produces a text notice.
On startup, the helper checks at most 20 recent direct chats and 20 items per
chat, retrying only the newest pending voice invitation created locally within
the last hour. Completed notes and older history are not replayed. Recovery
only emits metadata; native live authorization still precedes every download.
Generic attachments are ignored.

Completed assistant text alongside tool calls is sent immediately as an
intermediate text message, after the shared outbound hooks and leak checks.
Repeated flushes do not repeat accepted text. Tool status and private reasoning
are excluded. Optional tool-debug notifications use the existing global
`channels.show_tool_calls` setting and are text-only. Set it to `false` to hide
tool notifications while keeping assistant acknowledgments and updates.
The final answer is still delivered with its configured voice
note; intermediate text does not create additional voice notes. During a call,
that same narration already uses the existing live speech stream.

The CLI’s configured server list owns relay approval. Configure the official
file-relay presets used by your phone (including Flux when enabled there).
A note can reference several relay replicas; all must satisfy the CLI’s policy.
The adapter does not bypass this with `approved_relays=on`. A policy rejection
after a multipart file description arrives produces an immediate text notice.

Ordinary follow-ups use the shared [message arrival modes](./overview.md#messages-during-an-active-turn).

## Calls and speech

Configure the owning agent's existing `transcription_provider` and `tts_provider`.
The adapter uses those providers; it does not create a separate model loop or
conversation store. Both global speech subsystems must also be enabled.

Only encrypted audio invitations are accepted. Video invitations and invitations
without a shared encryption key are rejected. The phone's call screen should
show encryption. Public SimpleX messaging and TURN relays are used by default;
neither needs access to message or encrypted call content. A configured external
model provider still receives agent prompts after decryption on your server.

Call playback uses a 48 kHz mono audio graph and a 64 kbps Opus sender ceiling.
A fidelity-oriented content hint is applied to the synthesized track. This
preserves the source model’s bandwidth without first reducing playback to
16 kHz. Opus can still adapt to network conditions and receiver limits.
Recognition uses a separate 16 kHz audio graph, keeping the existing PCM and
voice-detection timing contract. Voice-note encoding is independent of this path.

For Parakeet Unified live recognition, configure the owning agent's selected
`local_whisper` provider with an explicit loopback WebSocket endpoint:

```toml
[providers.transcription.local_whisper.local]
uri = "http://127.0.0.1:18162/v1/audio/transcriptions" # files and voice notes
stream_uri = "ws://127.0.0.1:18161/v1/audio/stream" # live PCM
# bearer_token uses the provider's existing secret configuration
```

`stream_uri` defaults absent. The endpoint must be loopback and implement the
Parakeet companion protocol: authenticated WebSocket, `ready` with `sampleRate`
16000 and `streaming` true, binary mono PCM16 input, and `speech_start`, `partial`,
and `final` JSON events carrying `utteranceId`. The companion owns streaming ASR,
voice activity detection, noise reduction, and endpoint timing. Configure its
buffered attention profile there. Parakeet Unified's balanced 560 ms chunk plus
560 ms right context provides 1.12 s of algorithmic lookahead; that is separate
from end-of-speech silence and processing/network latency.

With this endpoint selected, microphone frames stream continuously in bounded,
memory-only queues. Partials stay ephemeral; only the finalized utterance enters
the existing conversation and model loop. Speech-start stops queued playback,
pending decoding, and synthesis, and, with `interrupt_on_new_message`, cancels the
preceding model turn immediately without waiting for final transcription. A
stream failure ends the call rather than silently dropping PCM or switching to
batch recognition. Only one companion connection is owned by an admitted call;
ending it releases recognition resources. Peer grants, speech activation, and
endpoint selection are checked against live configuration during the call.

Without `stream_uri`, the adapter retains its bounded batch fallback: an energy
detector ends utterances after approximately 700 ms of silence, transcribes
continuous speech in 30-second chunks, and combines them into one user turn.
Model cancellation in that fallback still waits for a recognized utterance.

Playback receipts outlive model generation. Interrupting a clip records completed
segments, the partly played segment's text and elapsed/duration milliseconds, and
discarded queued audio. A passive delivery observation precedes the next spoken
user turn in shared history, including when generation had already finished.
The complete assistant reply remains in the append-only archive and context;
no historical text is rewritten and synthetic output is not transcribed again.
The elapsed time is measured at the sender, so network and phone buffering make
it an estimate of playout, not an exact word boundary or proof of what was heard.
The agent is explicitly told that only part of the current segment was played.
Cancellation does not undo completed tool effects. The streaming detector does
not provide semantic turn-taking or guarantee that noise will never interrupt.

Outbound speech uses the existing sentence/paragraph segmenter and the agent's
TTS provider. Sanitized, leak-checked assistant text may become audible while the
model is streaming. Reasoning, tool protocol, and progress are excluded. When
outbound hooks are active, speech waits for their final decision. Already spoken
audio cannot be withdrawn; a rewrite of spoken content stops that speech worker
instead of replaying it. Text replies are also delivered to the SimpleX chat.

Group chats and agent-initiated calls are not handled.
Android screen-off, Bluetooth routing, TURN reachability, and battery restrictions
must be verified on the actual phone.

Streamed audio chunks use the current turn's native delivery capture. Cancellation
stops synthesis while allowing an already admitted delivery to settle its receipt.
The bridge distinguishes a proven pre-playback rejection from an uncertain
transport failure; interrupted speech must not be retried as a fresh full answer.

## Text formatting

Approved outbound text is converted from standard Markdown to SimpleX's inline
formatting at the helper's send boundary. This covers ordinary messages,
intermediate narration and the text delivered alongside voice notes. Conversation
history and speech synthesis retain their original text.

Bold (`**text**` or `__text__`) becomes `*text*`; italics become `_text_` and
strikethrough becomes `~text~`. Headings become bold lines; lists and tables use
readable text layouts. Links retain a visible destination. SimpleX cannot nest
styles, so nested emphasis is flattened. Intraword emphasis becomes plain text
to preserve identifiers and words. Inline and fenced code, escaped source and
raw HTML are passed through verbatim; this does not add a full code-block or
HTML renderer to SimpleX.

The helper uses a pinned Markdown lexer and never evaluates HTML or fetches
linked resources. Text is split at the existing 12,000-character limit without
splitting Unicode characters; JSON-encoded chunks also respect a byte budget; each converted emphasis span closes inside its
message. There is no formatting setting to synchronize with the model.

## Verification and rollback

Run `npm test` in `contrib/simplex` (ffmpeg and ffprobe are required for note tests). With `CHROMIUM_PATH` and assets configured,
`SIMPLEX_RTC_TEST=1 npm test` additionally checks encrypted audio in both directions
through public TURN and rejects mismatched encryption keys. `SIMPLEX_CLI_TEST=1 npm test` additionally
creates two temporary CLI profiles and verifies live message creation, edits,
finalization, parsed formatting, encrypted file/image transfers, and voice-note transfers
in both directions through public relays; `SIMPLEX_CLI_PATH` selects the CLI binary.
`SIMPLEX_DAEMON_TEST=1 node --test live-daemon.test.mjs` drives an isolated real
daemon and helper with mock model and SimpleX WebSocket endpoints. Build the
native binary first; `ZEROCLAW_TEST_BINARY` can select its path. This checks
status, streaming edits, tool narration, finalization, and conversation continuity
without contacting existing users or model services. Rust adapter tests exercise
peer denial, live policy changes, routing, and speech before finalization.
`SIMPLEX_RELAY_TEST=1 npm test` additionally exercises the real helper through
public TURN relays with simulated CLI signaling, including speech detection and
interruption. It does not contact phone users or invoke model providers.

For an end-to-end check, send text and a voice note from Android. Confirm
transcription and text-plus-voice delivery with `output_modality = "voice"`.
Then start an audio call from Android. Confirm the encryption indicator, speak, hear the reply, interrupt it,
and repeat with headphones and the screen locked. Confirm that the same main
conversation is visible from the Web dashboard.

Disable `calls_enabled` to stop admitting calls, or disable this channel and
reload the daemon to disconnect the helper. Stop the CLI service if desired.
Preserve the SimpleX databases and session archive; removing the adapter does
not require erasing either. `SimpleX event processing failed` and
`Live call speech update failed` identify adapter failures in the daemon log.
