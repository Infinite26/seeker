# Seeker

A local voice wake assistant. Say **"Seeker"**, and it answers out loud.

```
mic ──► VAD segmentation ──► ASR (faster-whisper) ──► wake word "seeker"
                                                        │
        speaker ◄── Kokoro TTS ◄── LLM reply ◄──────────┘
```

- **Ears:** `faster-whisper` (CTranslate2, CPU int8). Default model `small.en`
  (English, RTF ≈ 0.3 on a Ryzen Z1 Extreme). Peak-normalizes audio so quiet
  speech still transcribes. `large-v3-turbo` is too slow on this CPU
  (≈ 11 s for a 3 s clip); see *Switching ASR* below.
- **Wake word:** tolerant matching for "seeker"/"seek her"/"seeka"/"sika"/etc.
  plus known leading mis-hearings ("secret", "speaker") — see `leading_mishears`.
- **Continuous conversation:** after the first wake word, follow-up turns need no
  wake word for `conversation.timeout_s` (default 30 s) — and every turn refreshes
  the window, so a real back-and-forth never drops back to wake-word mode. Use
  `--conversation 0` for endless, or `--no-wake` to never require it.
- **Barge-in (interrupt):** Seeker keeps the mic open while speaking; talk over it
  and it stops mid-sentence and listens — and aborts whatever it was doing. While
  it is thinking, saying "Seeker, ..." replaces the running request; "stop" /
  "cancel" aborts it. Tunable via `barge_in`.
- **Conversation memory:** the last `memory.history_turns` exchanges are replayed
  to the LLM, so "what did we talk about?" works — and they persist to disk across
  restarts (`~/.cache/seeker/history.json`).
- **Brain:** three selectable backends (`brain.backend`):
  - **`auto`** (default) — routes each turn to whichever brain answers it fastest:
    plain chat for conversation (questions, greetings, reactions), the full agent
    for anything that needs the machine or the outside world. Falls back to the
    agent if chat fails (and vice versa).
  - **`opencode`** — always the full agent: drives a headless `opencode serve`
    over its HTTP API, so a voice command runs through the **full opencode
    agent**: shell, files, tools, MCP. Uses the dedicated config
    `~/.config/seeker/opencode.json` and the `seeker` agent. **Full access**
    (permissions all `allow`).
  - **`chat`** — plain OpenAI-compatible chat only (the local opencode gateway at
    `http://127.0.0.1:8317/zen/go/v1` + Ollama fallback).
- **Voice:** Kokoro-82M sidecar, reusing `/home/infinite/Projects/sentient/tts_server`
  (voices like `af_heart`; RTF ≈ 0.3 on CPU). Autostarted on demand.
- **Streaming replies:** the brain's reply is streamed sentence by sentence
  (live token deltas from the model), and each sentence is synthesised while the
  next one is still being generated — so speech starts as soon as the first
  sentence exists instead of after the whole reply. Fully interruptible.
- **Fast end-of-speech:** the VAD closes an utterance after 0.8 s of silence
  (`audio.silence_ms`), so there is no dead air between you finishing and Seeker
  answering. Raise it if you pause mid-thought a lot.
- **Audio8-ASR-Infinite** is wired as a pluggable backend for a GPU box / the
  24/7 docker stack — set `asr.backend` to `audio8`.

## Quick start

```bash
cd /home/infinite/Projects/seeker
./setup.sh                 # once: create .venv + install deps
./run.sh                   # listen on the mic (Ctrl-C to stop)
```

Then say: **"Seeker, what's a fun fact about the ocean?"**

If you just say "Seeker", it replies "Yes?" and listens for a follow-up for 8 s.

First run loads Whisper (~20-40 s on this CPU); watch for
`Seeker is listening`. The opencode brain server is started automatically (or by
`seeker-opencode.service`) — no API keys needed beyond what's in
`~/.config/seeker/opencode.json`. With the `chat` backend, start the
`openai-oauth` proxy or `ollama serve` instead. If the chosen brain is down,
Seeker still hears and transcribes — it just answers "my brain is offline".

## Commands

```bash
./run.sh                                # wake once, then keep talking (60s window)
./run.sh --conversation 0               # endless conversation (never need wake again)
./run.sh --no-wake                      # always listening, wake word never required
./run.sh --list-devices                 # show capture devices
./run.sh --calibrate                    # live RMS / noise-floor / speech flags
./run.sh --transcribe clip.wav          # test the ASR backend on a file
./run.sh --say "hello"                  # test TTS playback
./run.sh --text "seeker hi there"       # simulate a transcript (no mic)
./run.sh --once                         # handle one utterance then exit
./run.sh --brain chat                   # bypass opencode, use the plain gateway
./run.sh --no-llm --no-tts              # hear-only / print-only modes
```

Say **"Seeker, start over"** (or "new conversation" / "reset") to start a fresh
opencode session.

## Configuration (`config.json`)

| Key | Meaning |
| --- | --- |
| `aliases` | Wake phrases/`fuzzy` variants. Add mis-hearings here. |
| `leading_mishears` | First-word mis-hearings that count as wake (e.g. "secret"). |
| `require_wake` | `true` (default) needs the wake word once; `false` = always listening. |
| `conversation.timeout_s` | No-wake follow-up window in seconds (refreshed every turn); `0` = forever. |
| `barge_in.ratio` / `min_rms` | How loud you must be to interrupt Seeker's speech. |
| `feedback.*` | Spoken confirmations: `ack_actions` (natural "On it." for action requests; questions get none), `error_text`, `startup_text`, and `work_fillers` / `work_filler_after_s` ("Let me see." when the agent is busy with tools and silent too long). Set a text to `null` to fall back to a beep; `enabled`/per-kind flags toggle them. |
| `memory.enabled` / `history_turns` | Replay recent turns to the LLM for recall. |
| `memory.persist` / `path` | Save the transcript to disk across restarts. |
| `audio.input_device` | sounddevice index/name; `null` = default. |
| `audio.output_sink` | PipeWire sink name for TTS playback (e.g. the analog speakers); `null` = system default. |
| `audio.silence_ms` | End-of-speech pause for short commands (800 ms — snappy). |
| `audio.silence_max_ms` | Cap for long speech: trailing silence grows toward this as an utterance gets longer (1500 ms), so a pause mid-question doesn't split it. |
| `audio.stitch_s` | If a transcript still ends mid-sentence ("...what is the capital of"), wait this long for the rest and merge before answering (3 s). |
| `audio.*` | Other VAD timings and energy gate (tune with `--calibrate`). |
| `asr.backend` | `faster-whisper` or `audio8`. |
| `brain.backend` | `auto` (fast chat for conversation, agent for work), `opencode`, or `chat`. |
| `llm.base_url` / `model` / `fallbacks[]` | Ordered OpenAI-compatible chain (streamed). |
| `tts.voice` / `tts.endpoint` | Kokoro voice and sidecar URL. |
| `system_prompt` | Seeker's persona and brevity rules. |

### Switching ASR

```jsonc
// fastest / English-only (default on this machine)
"faster_whisper": { "model": "small.en", "cpu_threads": 8, "language": "en" }

// bilingual (zh + en) with auto language detection
"faster_whisper": { "model": "small", "language": null }

// most accurate but ~11 s/clip here — only with a GPU
"faster_whisper": { "model": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                    "device": "cpu", "compute_type": "int8" }
```

### Audio8 (remote / GPU)

Run the Audio8 stack from `github.com/Edge0-AI/Audio8-ASR-Infinite` (or its
`docker compose`) on a CUDA machine, front it with an OpenAI-compatible
`/v1/audio/transcriptions` endpoint, then:

```jsonc
"asr": {
  "backend": "audio8",
  "audio8": { "transcribe_url": "http://GPU_HOST:18190/v1/audio/transcriptions",
              "model": "audio8-asr-infinite", "language": "en" }
}
```

## Run as a service

```bash
systemctl --user daemon-reload
systemctl --user enable --now seeker-tts.service seeker-opencode.service seeker.service
journalctl --user -u seeker -f
```

- `seeker-opencode.service` — headless opencode server on `127.0.0.1:4097`
  (loopback only, no password). Seeker attaches to it; it is started
  automatically if absent.
- `seeker-tts.service` — Kokoro TTS sidecar (optional; autostarted if absent).

## Brain (opencode)

The `seeker` agent is defined in `~/.config/seeker/opencode.json` with a
spoken-reply persona, `maxSteps: 20`, and `permission` set to `allow` for
`bash`/`edit`/`webfetch`/`external_directory` (**full access**). It runs in the
working directory `~/.cache/seeker/workspace`.

- Server: `GET http://127.0.0.1:4097/session` (health), `POST /session`,
  `POST /session/{id}/message`.
- The current session id + directory + turn count persist at
  `~/.cache/seeker/sessions.json`, so context survives restarts.
- Sessions are scoped to their own `directory` on every request, so resuming a
  session runs the agent in the right folder.
- Change model/agent in `config.json` under `brain.opencode`.

## Sessions (voice control)

Every conversation is a session — chat turns and tool turns share it, it gets a
title from your first request ("What is the capital of france", not "New session
- ..."), and it appears in the list even if you only talked (no agent work).
Your chat history is stored **per session**, so "start over" really forgets and
"continue" brings the conversation back with its memory.

Say these (they never reach the agent):

| Say | Effect |
| --- | --- |
| "new session" / "new conversation" / "start fresh" / "start over" | start a new session |
| "we're done" / "that's all" / "end session" | end the session; stay listening, next wake starts fresh |
| "continue" / "resume" / "go on" / "where were we" | resume the most recent existing session |
| "list sessions" | browse every session that already exists (panel switches to the list) |
| "next" / "previous" | move the selection through the list |
| "open it" / "that one" / "continue that one" | resume the selected session |
| "continue the session about <name>" | resume by title (fuzzy match) |
| "stop" / "be quiet" / "shut up" | stop talking immediately and abort the running task |
| "cancel" / "never mind" / "abort" | abort the running task without speaking |

## Omarchy status panel

A floating card (`~/.config/omarchy/plugins/seeker.status/`, kind `panel`,
`keepLoaded`) sits top-left under the bar and shows **what Seeker is doing on
the machine** — the live tool activity from opencode's event stream
(`bash: git status`, `read: config.json`, `edit: main.py`, …), a rolling list of
recent actions, the state, the last reply, and the current session.

- **−** collapses it to a one-line pill; **×** hides it, and it reappears on your
  next interaction.
- While browsing sessions it lists the existing sessions with the selection
  highlighted.
- Feed: `~/.cache/seeker/status.json` (written by Seeker, watched by the panel).
- Toggle from a keybind: `omarchy-shell shell toggle seeker.status '{}'`
  (add e.g. `o.bind("SUPER", "S", "omarchy-shell shell toggle seeker.status ''")`
  in `~/.config/hypr/bindings.lua`).

## Notes

- Peak RAM with the default setup is roughly: Whisper `small` ≈ 0.5 GB +
  Kokoro ≈ 0.6 GB + the opencode server ≈ 0.3–0.6 GB. Close other heavy apps on
  this 9 GB handheld if you hit swap.
- **The opencode brain has full machine access.** A misheard voice command can
  run shell or edit files. Reduce risk via `brain.opencode.agent` (use a
  restricted agent), `permission` in `~/.config/seeker/opencode.json`, or
  `--brain chat` for a tool-less brain.
- If wake detection is unreliable, run `--calibrate` and adjust
  `audio.energy_ratio` / `audio.min_rms`, or add the mis-heard spelling to
  `aliases` / `leading_mishears`.
- **Boot / mic self-heal:** at startup Seeker picks a real capture source (never a
  `.monitor`), sets it as the PipeWire default, and — if the USB mic has vanished
  because WirePlumber dropped it — restarts WirePlumber and waits for it to
  reappear. It then retries opening the mic (up to 5 min) instead of crash-looping.
- **TTS reliability:** replies are posted straight to the sidecar (no gating on a
  separate health check); on failure Seeker re-ensures the sidecar and retries
  once, so a transient blip can't make it go silent.
- **Self-echo guard:** after Seeker speaks it flushes the mic and ignores a
  repeated identical request for 8 s, so it cannot loop on its own voice. If you
  hear speech that the journal doesn't credit to Seeker
  (`journalctl --user -u seeker.service -f`), the source is not Seeker — check
  the mic's hardware monitoring / other apps.
