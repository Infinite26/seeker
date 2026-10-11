# Seeker — Features & Usage

Say **"Seeker"**, then talk. Seeker hears you, thinks with the opencode agent,
and answers out loud. It can also run things on this machine.

## Talking to it
- Wake it with **"Seeker"** (or "hey seeker", "ok seeker").
- After waking, you can keep talking for a short window (30 s, refreshed every
  turn) without repeating the name.
- Replies start as soon as the first sentence is ready — they're streamed, so
  it talks while it's still thinking of the rest.
- It won't answer half a question: a pause mid-sentence just keeps it listening
  (longer speech gets more pause tolerance, and an unfinished-looking transcript
  is held for the rest of the sentence).
- **It only answers things meant for it.** Grunts, laughs, remarks to other
  people, "I didn't talk to you", the TV — shown in the panel as *heard:* but
  never answered. Wakeless turns must sound like a request or question.
- Replies are capped at 3 sentences (ask "in five sentences" to override).
- Everything else it hears (TV, music, noise) is shown in the panel as
  *heard:* but ignored.

### Session commands (no wake word needed for the explicit ones)
| Say | Effect |
| --- | --- |
| "new session" / "start fresh" / "start over" | start a fresh session |
| "we're done" / "that's all" / "end session" | end the session; next wake starts new |
| "continue" / "resume" / "where were we" | resume the most recent session |
| "list sessions" | browse every existing session in the panel |
| "next" / "previous" | move the selection |
| "open it" / "that one" / "continue that one" | resume the selected session |
| "continue the session about `<name>`" | resume by title |

### Control
| Say | Effect |
| --- | --- |
| "stop" / "be quiet" / "shut up" | stop talking and abort the running task |
| "cancel" / "never mind" | abort the running task silently |
| talk over its reply | barge-in: it stops mid-sentence and listens |
| "Seeker, ..." while it's working | replaces the request it was working on |

## The panel (top-left)
A floating card under the bar shows:
- **heard:** the latest thing the mic transcribed (so you can see it listening),
  plus a short history.
- **▸** the live action the agent is doing (`bash: git status`, `read: …`, `edit: …`).
- the phase (listening / heard / thinking / speaking), the last reply, and the
  current session title + turn count.
- **−** collapses it to a pill; **×** hides it — it comes back on your next
  interaction. It auto-mounts at login and after a shell restart.

## Voice feedback
- On startup it says *"I'm online and listening."*
- When it hears a request to *do* something it says *"On it."* (or similar) —
  questions get no filler, the answer just starts.
- While the agent is busy with tools and quiet for a few seconds it says
  *"Let me see."* once.
- On a brain failure it says *"Sorry, my brain is not responding."*

## Configuration (`config.json`)
| Key | Meaning |
| --- | --- |
| `conversation.timeout_s` | seconds you can keep talking without the wake word (30) |
| `require_wake` | `false` = always listening |
| `brain.backend` | `auto` (chat for talk, agent for work), `opencode`, `chat` |
| `aliases` / `leading_mishears` | wake-word phrases and tolerated mis-hearings |
| `feedback.*` | spoken confirmations (`ack_actions`, `work_fillers`; set `*_text` to `null` for beeps) |
| `barge_in.*` | how loud you must be to interrupt |
| `status.path` | status file the panel reads |
| `brain.opencode.*` | model, agent, workdir, session files |

## Running it
```bash
cd /home/infinite/Projects/seeker
./run.sh                 # foreground
systemctl --user status  # seeker, seeker-tts, seeker-opencode
```
- `--text "seeker hi"` simulate a request (no mic)
- `--no-wake` always listening, `--conversation 0` endless window
- `--say "hi"` test TTS, `--transcribe clip.wav` test ASR, `--calibrate` mic levels

## Troubleshooting
- **Doesn't hear me** → `./run.sh --calibrate`; check the panel's *heard:* line.
- **Wakes on TV** → raise `wake` cutoff / remove aliases; see `wake.py`.
- **No voice** → `systemctl --user status seeker-tts`; the sidecar must be healthy.
- **Panel gone** → `omarchy-shell shell summon seeker.status '{}'`.
