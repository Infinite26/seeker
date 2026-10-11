# AGENTS.md — Seeker architecture & developer notes

Hands-free voice assistant. `mic → VAD → ASR → wake word → brain → TTS → speaker`,
plus a live Omarchy Quickshell panel.

## Pipeline
```
sounddevice mic ──► UtteranceSegmenter (energy VAD) ──► faster-whisper (small.en)
   │                                                         │
   │ wake.detect() "seeker"                                  ▼
   └──────────────► AutoBrain ──┬─ chat  (LLMClient, SSE) ────► Status (status.json)
                                └─ agent (OpenCodeBrain)           │
                                     │  GET /event SSE             ▼
                                     │  (tool actions + text     Omarchy panel
                                     │   token deltas)
                                     ▼
                          SentenceEmitter ──► speech queue ──► Speaker
                          (sentence by sentence, pipelined Kokoro TTS)
```

## Files (`seeker_voice/`)
| File | Responsibility |
| --- | --- |
| `__main__.py` | CLI + main loop, wake/session/stop intents, the turn orchestrator (one speech queue per turn), status updates |
| `audio.py` | `MicStream`, `UtteranceSegmenter` (VAD), `Speaker` (`speak_queue`, interruptible), tones |
| `asr.py` | `faster-whisper` (default) and `audio8` (remote) backends |
| `wake.py` | wake-word detection (first-2-tokens, exact aliases, fuzzy core, leading mis-hears) |
| `brain.py` | `AutoBrain` router, `OpenCodeBrain` (sessions, `GET /event` monitor, abort), `ChatBrain` |
| `tts.py` | Kokoro sidecar client, `SentenceEmitter` (streamed text → sentences), `split_for_speech()` |
| `status.py` | atomic `status.json` writer the panel watches |
| `memory.py` | per-session conversation history (keyed by session id, persisted) |
| `config.py` | defaults + deep-merge of `config.json` |

## Key design points
- **Everything streams.** `brain.stream_reply(command, on_sentence=...)` hands
  each completed sentence to the turn the moment the model finishes it;
  `SentenceEmitter` grows from live `message.part.delta` events (opencode) or
  OpenAI SSE deltas (chat) and re-syncs to the authoritative final parts. The
  turn's `emit()` pushes `split_for_speech()` chunks into one `queue.Queue` fed
  to `Speaker.speak_queue()` — ack first, then reply sentences, one synthesised
  while the next is generated.
- **End-of-speech is adaptive** (`UtteranceSegmenter._silence_needed`): short
  commands close after `silence_ms` (800 ms), longer speech grows toward
  `silence_max_ms` (1500 ms) so a think-pause doesn't split a question. If a
  transcript *still* ends on a dangling word/phrase (`_looks_incomplete`), the
  loop holds it for `audio.stitch_s` and merges the continuation
  (`_merge_fragments` drops a repeated wake word) before dispatching.
- **Only addressed turns get answered.** Wakeless utterances go through
  `_looks_addressed()` (session/stop commands always count; grunts, laughs,
  first-person remarks and stray fragments are panel-only) and `_is_echo()`
  (near-verbatim of what we just said). The chat brain has the same `SKIP`
  convention as the agent ("reply SKIP when not addressed"), suppressed in
  `emit()`. `feedback.max_speak_sentences` (3) aborts a rambling reply;
  explicit "in N sentences" / "in detail" requests are exempt.
- **`is_playing()` means audible**: the mic loop must keep segmenting (and accept
  "stop") while the brain thinks but nothing is coming out of the speaker yet.
- **Requests run in a worker thread** so the mic loop keeps running. Cancel paths
  (`_cancel_turn`): a "stop" utterance, barge-in (talk-over aborts the task too),
  and a wake-word request that arrives mid-task (replaces it). All set the turn's
  `cancel` event + `brain.abort()` (`POST /session/{id}/abort`).
- **`AutoBrain`** routes conversational turns to `ChatBrain` (no tool loop, ~1-3 s)
  and everything else to the full agent; on failure it falls through to the other
  brain. The router lives in `brain.looks_conversational()` (chitchat/question
  patterns minus tool/action hints).
- **Sessions are directory-scoped**: every session request passes
  `?directory=<session.directory>`; the current id+dir persist in
  `~/.cache/seeker/sessions.json`. Sessions are created **lazily on the first
  turn** with a title from that request (`_title_from`) — never eagerly on
  "new session" (which just clears the current id) — so there are no empty
  "New session - ..." shells. `AutoBrain` ensures the session before routing, so
  chat-only conversations are listable/resumable too.
- **Chat memory is keyed by session id** (`memory.py`): history is fetched with
  `brain.session_key()` before a turn and stored under it after (the id may be
  created mid-turn). `messages(None)` returns nothing — a fresh conversation
  must never see the previous one's history.
- **Live activity** comes from `GET /event?directory=…` SSE; `message.part.updated`
  → tool parts → `Status.add_action()`, and text-part deltas → the emitter. The
  user's own question is excluded via a client-chosen `messageID` on the prompt.
- **Status** (`~/.cache/seeker/status.json`) is written atomically; the panel
  polls it. `add_heard()` records *every* transcript (even ignored/media).
- **Panel**: `~/.config/omarchy/plugins/seeker.status/` (kind `panel`,
  `keepLoaded`). Third-party keep-loaded panels are not auto-mounted, so Seeker
  summons it on start and a watcher remounts it after a shell restart (but never
  reopens it if the user pressed ×). IPC: `quickshell ipc -p /usr/share/omarchy/shell call seeker.status panelState`.
- **Wake/ASR hardening**: wake only in the first 2 tokens; `_HALLUCINATIONS`
  blocklist + `vad_filter`/thresholds suppress Whisper noise hallucinations.
- **SKIP**: the agent replies `SKIP` when input isn't addressed to it; the emitter
  never speaks a sentence that normalises to "skip".

## Services (systemd --user, linger on)
- `seeker.service` — the assistant
- `seeker-opencode.service` — `opencode serve` on `127.0.0.1:4097`
  (`OPENCODE_CONFIG_DIR=~/.config/seeker`)
- `seeker-tts.service` — Kokoro sidecar on `127.0.0.1:8770`

## Agent config
`~/.config/seeker/opencode.json`: provider `opencode-go`, `permission` all
`allow` (**full machine access**), and the `seeker` agent (one-sentence,
spoken-style persona; replies `SKIP` when not addressed).

## Gotchas
- **`omarchy-opencode-go-proxy` must forward SSE incrementally.** Its `forward()`
  loop originally used `response.read(65536)`, which blocks for the full 64 KB
  and killed all token streaming (every reply arrived in one burst at the end).
  It now uses `response.read1(65536)`. If an Omarchy update replaces
  `~/.local/bin/omarchy-opencode-go-proxy`, re-apply that change — streaming
  latency depends on it.
- opencode `GET /event` delivers live text via `message.part.delta`
  (`field: "text"`); `message.part.updated` carries the *final* part text only.
  Reasoning parts also emit `field: "text"` deltas — only parts registered by a
  `message.part.updated` with `type: "text"` are treated as speech.
- **SSE streams are Latin-1 by default.** `requests` sets `ISO-8859-1` on
  `text/event-stream` (no charset), which turns every UTF-8 dash/quote into
  mojibake ("â€”"). Always set `resp.encoding = "utf-8"` on streaming reads;
  `clean_for_speech()` also repairs mojibake and maps unicode punctuation to
  speakable ASCII as a second line of defence.
- **ASR name-biasing**: `initial_prompt` keeps "Seeker" transcribed as "Seeker"
  (Whisper's small.en otherwise hears "See, Kurt" / "Secret"). Multi-word
  `leading_mishears` ("see kurt") catch the rest — the leading check matches a
  1- or 2-token prefix at the very start, with a command after it.
- **ASR normalization is transient-safe**: `_normalize` scales by the
  98th-percentile level (gain clamped 0.5–12x), so one loud knock can't leave
  the speech itself too quiet to decode. Beam search (`beam_size_long`) is used
  only for utterances over `long_s` seconds — greedy stays fast for commands.
- The chat LLM is a reasoning model: `llm.max_tokens` must leave room for the
  hidden reasoning tokens or the visible answer comes back empty (120 was too
  small; 800 works). An empty fast-path reply falls back to the agent.
- opencode `GET /event` can replay parts on reconnect → `Status.add_action` dedupes.
- Whisper hallucinations on silence are common; keep the blocklist current.
- `chroma`/mic: if PipeWire drops the USB mic, `ensure_input_source()` restarts
  WirePlumber and waits.
- Don't `pkill -f "opencode serve ... 4097"` from a shell whose command line
  contains that string — it kills the shell.
