# SimpleX

The optional `channel-simplex` adapter supports direct text, voice notes, and incoming
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
interrupt_on_new_message = true

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
is ready. Final rewrites invalidate speculative audio. No partial voice notes
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

The CLI’s configured server list owns relay approval. Configure the official
file-relay presets used by your phone (including Flux when enabled there).
A note can reference several relay replicas; all must satisfy the CLI’s policy.
The adapter does not bypass this with `approved_relays=on`. A policy rejection
after a multipart file description arrives produces an immediate text notice.

## Calls and speech

Configure the owning agent's existing `transcription_provider` and `tts_provider`.
The adapter uses those providers; it does not create a separate model loop or
conversation store. Both global speech subsystems must also be enabled.

Only encrypted audio invitations are accepted. Video invitations and invitations
without a shared encryption key are rejected. The phone's call screen should
show encryption. Public SimpleX messaging and TURN relays are used by default;
neither needs access to message or encrypted call content. A configured external
model provider still receives agent prompts after decryption on your server.

Incoming speech is collected in bounded, memory-only utterances using an energy
threshold and approximately 700 ms of silence. Continuous speech is transcribed
in 30-second chunks in the background; one combined turn enters dispatch only
when silence closes it. At speech onset, queued playback stops and stale-generation audio
is refused. A recognized utterance enters the shared dispatcher; with
`interrupt_on_new_message`, it also cancels the preceding model turn. Recognition
latency therefore still affects when model work is cancelled.

Outbound speech uses the existing sentence/paragraph segmenter and the agent's
TTS provider. Sanitized, leak-checked assistant text may become audible while the
model is streaming. Reasoning, tool protocol, and progress are excluded. When
outbound hooks are active, speech waits for their final decision. Already spoken
audio cannot be withdrawn; a rewrite of spoken content stops that speech worker
instead of replaying it. Text replies are also delivered to the SimpleX chat.

Group chats, generic file attachments, and agent-initiated calls are not handled.
Android screen-off, Bluetooth routing, TURN reachability, and battery restrictions
must be verified on the actual phone.

Streamed audio chunks use the current turn's native delivery capture. Cancellation
stops synthesis while allowing an already admitted delivery to settle its receipt.
The bridge distinguishes a proven pre-playback rejection from an uncertain
transport failure; interrupted speech must not be retried as a fresh full answer.

## Verification and rollback

Run `npm test` in `contrib/simplex` (ffmpeg and ffprobe are required for note tests). With `CHROMIUM_PATH` and assets configured,
`SIMPLEX_RTC_TEST=1 npm test` additionally checks encrypted audio in both directions
and rejects mismatched encryption keys. `SIMPLEX_CLI_TEST=1 npm test` additionally
creates two temporary CLI profiles and verifies encrypted voice-note transfers
in both directions through public relays; `SIMPLEX_CLI_PATH` selects the CLI binary.
Rust adapter tests exercise peer denial,
live policy changes, routing, and speech before finalization.
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
