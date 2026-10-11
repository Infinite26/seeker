"""Seeker voice loop CLI.

Examples:
    ./run.sh                          # listen; say "Seeker" once, then keep talking
    ./run.sh --once                   # handle a single utterance then exit
    ./run.sh --no-wake                # never require the wake word ("talk constantly")
    ./run.sh --text "seeker what time is it"   # simulate a transcript, no mic
    ./run.sh --transcribe clip.wav    # test the ASR backend on a file
    ./run.sh --say "hello"            # test the TTS backend
    ./run.sh --list-devices           # show audio devices
    ./run.sh --calibrate              # print live RMS/noise-floor levels
"""
from __future__ import annotations

import argparse
import difflib
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import wave

import numpy as np

from . import config as config_mod
from .asr import build as build_asr
from .audio import (
    MicStream,
    Speaker,
    UtteranceSegmenter,
    chirp_wav,
    ensure_input_source,
    list_devices,
    play_wav_bytes,
    tone_wav,
)
from .brain import build_brain
from .memory import ConversationMemory
from .status import Status
from .tts import TTSClient, clean_for_speech, split_for_speech
from .wake import detect

FOLLOWUP_WINDOW_S = 12.0

# Session-management intents (matched against the whole command, so normal
# sentences like "continue reading the file" are still sent to the agent).
_NEW_SESSION = {
    "new session", "new conversation", "new chat", "start fresh", "start over",
    "start a new session", "begin a new session", "fresh session", "reset",
    "reset the conversation", "forget this conversation", "clear the conversation",
    "clear our conversation",
}
_DONE_SESSION = {
    "we're done", "were done", "we are done", "that's all", "thats all",
    "end session", "end the session", "done for now", "we're done for now",
    "all done", "i'm done", "im done", "stop the session",
}
_CONTINUE = {
    "continue", "resume", "go on", "go on with it", "carry on", "keep going",
    "where were we", "pick up where we left off", "continue the conversation",
    "continue that", "resume that",
}
_LIST_SESSIONS = {
    "list sessions", "what sessions do we have", "show sessions",
    "show me the sessions", "list the sessions", "what sessions are there",
    "list my sessions", "show sessions list",
}
_NEXT = {"next", "next session", "next one", "go to the next session", "next please"}
_PREV = {"previous", "previous session", "previous one", "go back", "back",
         "last one", "the previous one"}
_OPEN = {
    "open it", "open that", "open that one", "that one", "select it",
    "select that one", "use that one", "open the session", "continue that one",
    "resume that one", "open the one",
}
_SESSION_NAME_RE = re.compile(
    r"^(?:continue|resume|open|go to|switch to)\s+"
    r"(?:the\s+)?(?:session|conversation|one|chat)\s+"
    r"(?:about|named|called|on|with)?\s*(.+)$", re.I)

# Stop / cancel: halt speech and abort a running agent task.
_STOP = {
    "stop", "stop it", "stop that", "stop talking", "stop now", "shut up",
    "be quiet", "quiet", "silence", "enough", "cease", "halt",
}
_CANCEL = {
    "cancel", "cancel that", "abort", "abort that", "never mind", "nevermind",
    "forget it", "stop the task", "stop working", "stop the request",
}
_WORD_RE = re.compile(r"[a-z0-9']+")

# Tolerant phrase fragments: a session command matches if the normalized command
# contains any of these (so "start a new session please" still works).
_NEW_CONTAINS = ("new session", "new conversation", "new chat", "start fresh",
                 "start over", "start a new", "begin a new", "another session",
                 "reset the conversation", "forget this conversation",
                 "forget the conversation", "clear the conversation",
                 "clear our conversation")
_DONE_CONTAINS = ("end session", "end the session", "done for now", "we're done",
                  "were done", "we are done", "that's all", "thats all",
                  "all done", "i'm done", "im done", "stop the session",
                  "finish the session", "close the session")
_CONTINUE_CONTAINS = ("continue session", "continue the session", "continue our session",
                      "continue conversation", "continue the conversation",
                      "resume session", "resume the session", "go on with it",
                      "pick up where", "where were we", "carry on")
_LIST_CONTAINS = ("list sessions", "list the sessions", "list my sessions",
                  "show sessions", "show me the sessions", "show the sessions",
                  "what sessions", "sessions list")
_NEXT_CONTAINS = ("next session", "next one", "go to the next", "next please")
_PREV_CONTAINS = ("previous session", "previous one", "go back", "the previous one",
                  "last one", "previous one")
_OPEN_CONTAINS = ("open the session", "open that one", "open that session",
                  "select that one", "use that one", "continue that one",
                  "resume that one", "that session")

# Strong, session-explicit phrases accepted even without the wake word (generic
# phrases like "that's all" still require the wake word to avoid media triggers).
_WAKE_OPTIONAL = (
    "new session", "new conversation", "new chat", "start a new session",
    "begin a new session", "start fresh", "start over", "another session",
    "end session", "end the session", "stop the session", "close the session",
    "finish the session", "list sessions", "list the sessions", "list my sessions",
    "show sessions", "show me the sessions", "what sessions", "resume session",
    "resume the session", "continue session", "continue the session",
    "continue our session", "where were we",
)


def _normalize_key(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _intent_match(key: str, exact: set, contains=()) -> bool:
    if not key:
        return False
    if key in exact:
        return True
    return any(fragment in key for fragment in contains)


def _looks_like_session_command(key: str) -> bool:
    return bool(key) and any(fragment in key for fragment in _WAKE_OPTIONAL)


# Classic Whisper hallucinations on silence / media noise. Recorded for display
# but never treated as a command.
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks", "thanks for watching",
    "thanks for watching everyone", "thank you for watching", "you", "ahem",
    "mm-hmm", "mm hmm", "mhm", "hmm", "hm", "uh", "um", "ah", "oh", "eh", "huh",
    "okay", "ok", "bye", "goodbye", "please subscribe", "subscribe",
    "subtitles by", "subtitles", "music", "applause", "foreign",
    "yeah", "yep", "nope", "ha ha", "oh my", "whew",
}


def _is_hallucination(text: str) -> bool:
    key = _normalize_key(text)
    return len(key) < 2 or key in _HALLUCINATIONS


# Grunts, laughs and stray reactions: real speech, but never a request.
_IGNORED_UTTERANCES = {
    "ugh", "ughh", "ughhh", "ah", "ahh", "oh", "ohh", "ow", "oof", "hmm", "hm",
    "huh", "eh", "mhm", "mm", "mmhmm", "ha", "haha", "hahaha", "hahahaha",
    "heh", "hehe", "lol", "lmao", "rofl", "yay", "whoa", "woah", "wow",
    "damn", "dang", "crap", "bruh", "bro", "dude", "man", "shit", "fuck",
    "okay", "ok", "k", "cool", "nice", "great", "awesome", "sweet", "neat",
    "right", "true", "same", "exactly", "sure", "yeah", "yep", "yup", "nah",
    "nope", "no", "yes", "y", "n", "whatever", "anyway", "anyways", "so",
    "like", "wait", "huh what", "what what", "oh okay", "oh ok", "oh cool",
    "oh nice", "my bad", "no way", "for real", "i guess", "i suppose",
    "thanks", "thank you", "thank you seeker", "thx", "ty", "cheers",
    "bye", "goodbye", "goodnight", "good night", "see ya", "see you",
    "okay cool", "okay nice", "thats cool", "thats nice", "thats great",
    "that's cool", "that's nice", "that's great", "noted", "gotcha",
}
# First-person remarks about oneself are not requests for the assistant.
_SELF_TALK_PREFIXES = (
    "i am ", "i'm ", "im ", "i was ", "i were ", "i will ", "i'll ", "ill ",
    "i just ", "i think ", "i thought ", "i mean ", "i feel ", "i felt ",
    "i gotta ", "i got ta ", "i got to ", "i have to ", "i need to ",
    "i can't ", "i cant ", "i don't ", "i dont ", "i didnt ", "i didn't ",
    "i wanna ", "i want to ", "i'm going to ", "im going to ", "i'm gonna ",
    "im gonna ", "i bet ", "i guess ", "i hope ", "i remember ",
    "we are ", "we're ", "we were ", "we should ", "we need to ", "let me ",
    "he is ", "he's ", "she is ", "she's ", "they are ", "they're ",
    "it is ", "it's ", "that is ", "that's ", "there is ", "there's ",
    "this is ", "if ", "when ", "because ", "since ", "okay i ", "ok i ",
)
# What a request to an assistant looks like (question or imperative).
_ADDRESSED_START = re.compile(
    r"^(?:please |hey |yo |ok |okay |so |um |uh )?"
    r"(?:what|who|where|when|why|how|which|whose|whats|what's|tell|explain|"
    r"define|describe|remind|remember|note|find|search|google|look|show|give|"
    r"open|close|run|start|stop|play|pause|set|make|create|write|read|check|"
    r"turn|add|remove|delete|install|update|download|send|call|text|launch|"
    r"kill|reboot|shutdown|build|fix|test|deploy|calculate|compute|convert|"
    r"can|could|would|will|should|shall|may|do|does|did|is|are|was|were|"
    r"have|has|had|let's|lets|never|dont|don't|"
    r"i need you|i want you|i'd like you|id like you|i would like you|"
    r"i'd love you|id love you|could you|would you)\b")


def _is_session_command(key: str, raw: str = "") -> bool:
    """True for session-management phrasing (which always counts as addressed)."""
    return (_intent_match(key, _NEW_SESSION, _NEW_CONTAINS)
            or _intent_match(key, _DONE_SESSION, _DONE_CONTAINS)
            or _intent_match(key, _CONTINUE, _CONTINUE_CONTAINS)
            or _intent_match(key, _LIST_SESSIONS, _LIST_CONTAINS)
            or _intent_match(key, _NEXT, _NEXT_CONTAINS)
            or _intent_match(key, _PREV, _PREV_CONTAINS)
            or _intent_match(key, _OPEN, _OPEN_CONTAINS)
            or bool(_SESSION_NAME_RE.match((raw or key).strip())))


def _looks_addressed(text: str) -> bool:
    """True if a wakeless utterance sounds like it is meant for the assistant.

    Everything else (grunts, laughs, remarks to someone else, first-person
    musing) is shown in the panel but never answered — that is what stops
    Seeker from joining conversations it was not part of."""
    key = _normalize_key(text)
    if not key:
        return False
    if _is_session_command(key, text) or _is_stop_key(key):
        return True
    if key in _HALLUCINATIONS or key in _IGNORED_UTTERANCES:
        return False
    words = key.split()
    if len(words) <= 2 and words[-1] in _IGNORED_UTTERANCES:
        return False
    if key.endswith("..."):
        return False
    if key.startswith(_SELF_TALK_PREFIXES):
        return False
    if (text or "").strip().endswith(("?", "!")):
        return True
    return bool(_ADDRESSED_START.match(key))


def _is_echo(text: str, state: dict, within_s: float = 5.0) -> bool:
    """True if the mic most likely heard Seeker's own voice.

    Compares against what was spoken in the last few seconds — a transcript
    that is nearly verbatim our own reply is an echo, not a request."""
    key = _normalize_key(text)
    if not key:
        return False
    now = time.time()
    for ts, spoken_key in state.get("spoken", []):
        if now - ts > within_s:
            continue
        if difflib.SequenceMatcher(None, key, spoken_key).ratio() >= 0.8:
            return True
    return False


def _note_spoken(state: dict, text: str) -> None:
    """Record something Seeker said, so its echo can be recognised."""
    key = _normalize_key(text)
    if not key:
        return
    recent = [s for s in state.get("spoken", []) if time.time() - s[0] < 30]
    recent.append((time.time(), key))
    state["spoken"] = recent[-12:]


# Words / phrase-endings that mean the speaker hasn't finished their sentence
# yet.  If a transcript ends on one of these, hold the answer briefly so the
# rest of the sentence can arrive instead of answering a half-question.
_DANGLING = {
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "with", "from", "by",
    "about", "and", "or", "but", "so", "as", "than", "then", "if", "my", "your",
    "his", "her", "our", "their", "what", "who", "where", "when", "why", "how",
    "which", "whose", "is", "are", "was", "were", "am", "can", "could", "will",
    "would", "should", "shall", "may", "might", "do", "does", "did", "have",
    "has", "had", "its", "it's", "i'm", "you're", "we're", "they're",
    "that's", "there's", "lets", "let's",
}
# Two-word endings that dangle even though neither word does alone ("can you"
# vs. the complete "who are you" / "thank you").
_DANGLING_BIGRAMS = {
    ("can", "you"), ("could", "you"), ("will", "you"), ("would", "you"),
    ("can", "we"), ("could", "we"), ("shall", "we"), ("should", "we"),
    ("can", "i"), ("could", "i"), ("do", "you"), ("did", "you"),
    ("does", "it"), ("will", "it"), ("would", "it"), ("is", "there"),
    ("are", "there"), ("tell", "me"), ("look", "up"), ("tell", "us"),
}


def _looks_incomplete(text: str) -> bool:
    """True if the transcript sounds like the speaker was cut off mid-sentence."""
    raw = (text or "").strip()
    if not raw:
        return True
    if raw.endswith(("...", "…", ",", ";", ":", "-", "—")):
        return True
    words = _normalize_key(raw).split()
    if not words:
        return True
    if words[-1] in _DANGLING:
        return True
    return len(words) >= 2 and (words[-2], words[-1]) in _DANGLING_BIGRAMS


def _merge_fragments(held: str, new: str, cfg: dict) -> str:
    """Join a held utterance with its continuation.

    A repeated wake word in the continuation is dropped (people re-address
    mid-sentence: 'seeker can you' + 'seeker check the weather'), so the merged
    text keeps exactly one leading wake word for the detector."""
    matched, command, _alias = detect(
        new, cfg.get("aliases"), leading_mishears=cfg.get("leading_mishears"))
    if matched and command.strip():
        new = command
    return f"{held} {new}".strip()


def _is_stop_key(key: str) -> bool:
    if not key:
        return False
    if key in _STOP or key in _CANCEL:
        return True
    last = key.split()[-1]
    return (last in ("stop", "quiet", "silence", "enough", "halt")
            or key.endswith("shut up") or key.endswith("be quiet")
            or key.endswith("never mind") or key.endswith("cancel that")
            or key.endswith("stop it") or key.endswith("stop that"))


def _is_stop(text: str, cfg: dict, state: dict) -> bool:
    matched, command, _alias = detect(
        text, cfg.get("aliases"), leading_mishears=cfg.get("leading_mishears"))
    if matched:
        key = _normalize_key(command)
    elif time.time() < state.get("conv_until", 0.0) or not cfg.get("require_wake", True):
        key = _normalize_key(text)
    else:
        return False
    return _is_stop_key(key)


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.float32)
    return pcm, sr


_ACK_ACTIONS = ["On it.", "Will do.", "Got it.", "Sure thing.", "Doing that now."]
_ACTION_RE = re.compile(
    r"\b(get|open|search|find|play|set|turn|make|start|stop|close|send|add|"
    r"remove|delete|download|install|run|check|show|give|google|write|create|"
    r"remind|call|launch|kill|reboot|shutdown|update|look up|put on|switch)\b",
    re.IGNORECASE,
)
_ack_i = 0


def _pick_ack(command: str, fb: dict) -> str | None:
    """Pick a short, natural acknowledgement for an action request, immediately.

    Questions and plain statements get no ack — their answer is the reply, so
    they start speaking as soon as it's ready (e.g. "The weather today is...").
    """
    text = (command or "").strip()
    if not text or not _ACTION_RE.search(text):
        return None
    actions = fb.get("ack_actions") or _ACK_ACTIONS
    if not actions:
        return None
    global _ack_i
    phrase = actions[_ack_i % len(actions)]
    _ack_i += 1
    return phrase


def _ack_phrase(command: str, fb: dict) -> str | None:
    """The spoken 'I heard you' line for a request (None = silence)."""
    if not fb.get("enabled", True) or not fb.get("ack", True):
        return None
    return _pick_ack(command, fb)


def _feedback(speaker: Speaker, tts: TTSClient, cfg: dict, kind: str,
              no_tts: bool = False, command: str | None = None) -> None:
    """Short spoken confirmation so the user knows Seeker heard them / is alive.

    Uses a voiced phrase (`<kind>_text` in config); falls back to a beep only if
    no phrase is set.  For the ``ack`` kind, a natural request-aware phrase is
    chosen when ``command`` is given."""
    if no_tts:
        return
    fb = cfg.get("feedback", {}) or {}
    if not fb.get("enabled", True) or not fb.get(kind, True):
        return
    phrase = fb.get(f"{kind}_text")
    if kind == "ack" and command is not None:
        phrase = _pick_ack(command, fb)
    if phrase:
        _speak(speaker, tts, phrase, cfg, no_tts, announce=False)
        return
    if kind == "ack" and command is not None:
        # Natural-ack mode: questions/statements get no ack at all.
        return
    try:
        if kind == "ack":
            speaker.play(tone_wav(880.0, 110))
        elif kind == "error":
            speaker.play(tone_wav(300.0, 240))
        elif kind == "startup":
            speaker.play(chirp_wav())
    except Exception:
        pass


def _speak(speaker: Speaker, tts: TTSClient, text: str, cfg: dict, no_tts: bool,
           announce: bool = True) -> None:
    if announce:
        print(f"{cfg['name']}: {text}")
    if no_tts:
        return
    # Speak sentence by sentence; the next is synthesised while the current one
    # plays, so audio starts after the first sentence rather than the full reply.
    chunks = split_for_speech(text)
    if chunks:
        speaker.speak_texts(chunks, tts.synthesize)


def _conversation_timeout(cfg: dict) -> float:
    conv = cfg.get("conversation", {}) or {}
    if not conv.get("enabled", True):
        return 0.0  # 0 == conversation mode off (wake word every time)
    return float(conv.get("timeout_s", 60))


# ---------------------------------------------------------------------------
# session management
# ---------------------------------------------------------------------------

def _refresh_sessions(brain, state: dict, status: Status | None) -> list[dict]:
    sessions = [s for s in brain.list_sessions() if s.get("id")]
    current = getattr(brain, "_session_id", None)
    sessions = [s for s in sessions if s["id"] != current]
    state["browse"] = {"sessions": sessions, "index": 0}
    if status:
        status.update(
            mode="sessions",
            sessions=[{"title": s["title"], "id": (s["id"] or "")[:12]} for s in sessions],
            selected=0 if sessions else -1,
        )
    return sessions


def _browse_sessions(state: dict) -> list[dict]:
    return (state.get("browse") or {}).get("sessions") or []


def _session_intent(key: str, command: str, brain, status: Status | None,
                    speaker: Speaker, tts: TTSClient, cfg: dict, state: dict,
                    no_tts: bool) -> bool:
    """Handle a session-management command. Returns True if it was one."""
    def say(text: str, mode: str | None = None, reset_selected: bool = False) -> None:
        if status:
            fields: dict = {}
            if mode:
                fields["mode"] = mode
            if reset_selected:
                fields["selected"] = -1
            if fields:
                status.update(**fields)
        _note_spoken(state, text)
        _speak(speaker, tts, text, cfg, no_tts)

    if _intent_match(key, _NEW_SESSION, _NEW_CONTAINS):
        try:
            # Lazy: the next request creates the session, titled from what you
            # actually ask — no empty "New session - ..." shells.
            brain.end_current()
            say("Starting a new session.", mode="activity", reset_selected=True)
        except Exception:
            say("I couldn't start a new session.", mode="activity", reset_selected=True)
        return True

    if _intent_match(key, _DONE_SESSION, _DONE_CONTAINS):
        brain.end_current()
        say("Okay, session ended. Say my name when you want a new one.",
            mode="activity", reset_selected=True)
        return True

    if _intent_match(key, _LIST_SESSIONS, _LIST_CONTAINS):
        sessions = _refresh_sessions(brain, state, status)
        if not sessions:
            say("I couldn't find any other sessions.", mode="sessions")
        else:
            say(f"You have {len(sessions)} sessions. First up, {sessions[0]['title']}.",
                mode="sessions")
        return True

    is_next = _intent_match(key, _NEXT, _NEXT_CONTAINS)
    is_prev = _intent_match(key, _PREV, _PREV_CONTAINS)
    if is_next or is_prev:
        sessions = _browse_sessions(state) or _refresh_sessions(brain, state, status)
        if not sessions:
            say("I couldn't find any sessions.", mode="sessions")
            return True
        step = 1 if is_next else -1
        index = (state["browse"].get("index", 0) + step) % len(sessions)
        state["browse"]["index"] = index
        if status:
            status.update(mode="sessions", selected=index)
        say(sessions[index]["title"], mode="sessions")
        return True

    if _intent_match(key, _CONTINUE, _CONTINUE_CONTAINS) or \
            _intent_match(key, _OPEN, _OPEN_CONTAINS):
        sessions = _browse_sessions(state) or _refresh_sessions(brain, state, status)
        if not sessions:
            say("I couldn't find another session to continue.",
                mode="activity", reset_selected=True)
            return True
        index = state["browse"].get("index", 0)
        session = sessions[index]
        try:
            brain.resume_session(session)
            say(f"Continuing {session['title']}.", mode="activity", reset_selected=True)
        except Exception:
            say("I couldn't open that session.", mode="activity", reset_selected=True)
        return True

    match = _SESSION_NAME_RE.match(command.strip())
    if match:
        wanted = match.group(1).strip().lower()
        all_sessions = [s for s in brain.list_sessions(50) if s.get("id")]
        best = None
        for s in all_sessions:
            title = (s.get("title") or "").lower()
            if wanted and (wanted in title or title in wanted):
                best = s
                break
        if not best:
            import difflib
            scored = sorted(all_sessions,
                            key=lambda s: difflib.SequenceMatcher(
                                None, wanted, (s.get("title") or "").lower()).ratio(),
                            reverse=True)
            if scored and difflib.SequenceMatcher(
                    None, wanted, (scored[0].get("title") or "").lower()).ratio() >= 0.6:
                best = scored[0]
        if best:
            try:
                brain.resume_session(best)
                say(f"Continuing {best['title']}.", mode="activity", reset_selected=True)
            except Exception:
                say("I couldn't open that session.", mode="activity", reset_selected=True)
        else:
            say(f"I couldn't find a session about {wanted}.",
                mode="activity", reset_selected=True)
        return True

    return False


def _handle_transcript(text: str, cfg: dict, brain, tts: TTSClient,
                       speaker: Speaker, state: dict, status: Status | None,
                       no_tts: bool, no_llm: bool) -> bool:
    """Process one transcript.  Returns True if Seeker spoke.

    The whole turn speaks through a single queue: the ack (if any) first, then
    reply sentences as the brain streams them out — so speech starts the moment
    the first sentence exists instead of after the full reply."""
    now = time.time()
    if not text.strip():
        return False

    # My own voice coming back through the mic is not a request.
    if _is_echo(text, state):
        print(f"[heard] {text}  (echo of my own voice)")
        return False

    matched, command, alias = detect(
        text, cfg.get("aliases"),
        leading_mishears=cfg.get("leading_mishears"),
    )
    in_conversation = now < state.get("conv_until", 0.0)
    require_wake = bool(cfg.get("require_wake", True))

    def _refresh_conversation() -> None:
        timeout = _conversation_timeout(cfg)
        if timeout > 0:
            state["conv_until"] = now + timeout    # each turn keeps it alive
        elif (cfg.get("conversation", {}) or {}).get("enabled", True):
            state["conv_until"] = float("inf")     # "talk constantly" until stopped

    if matched:
        _refresh_conversation()
        print(f"[wake] matched {alias!r}")
        if (cfg.get("panel", {}) or {}).get("enabled", True):
            # Show the status panel as soon as we're addressed, so the user can
            # watch what Seeker is doing.
            threading.Thread(target=_maybe_reopen_panel, daemon=True).start()
    elif in_conversation or not require_wake:
        command = text
        # Only things that sound like they are meant for Seeker get answered;
        # the rest of the room's chatter is displayed but ignored.
        if not _looks_addressed(text):
            print(f"[heard] {text}  (not addressed to me)")
            return False
        _refresh_conversation()
    elif _looks_like_session_command(_normalize_key(text)):
        # Session management works without the wake word.
        command = text
        print(f"[session] (no wake) {text!r}")
    else:
        if text:
            print(f"[heard] {text}  (no wake word)")
        return False

    state["awaiting_until"] = 0
    key = command.strip().lower()

    if not command:
        state["awaiting_until"] = now + FOLLOWUP_WINDOW_S
        _note_spoken(state, cfg.get("wake_reply", "Yes?"))
        _speak(speaker, tts, cfg.get("wake_reply", "Yes?"), cfg, no_tts)
        return True

    if status:
        status.bump_seq()
        status.update(state="heard", heard=text)
        if (cfg.get("panel", {}) or {}).get("enabled", True):
            threading.Thread(target=_maybe_reopen_panel, daemon=True).start()

    # Session-management commands never reach the agent (and may repeat freely).
    if _session_intent(key, command, brain, status, speaker, tts, cfg, state, no_tts):
        return True

    # Stop / cancel: halt speech and abort whatever the agent is doing.
    if _is_stop_key(key):
        print(f"[stop] {command!r}")
        speaker.stop()
        if hasattr(brain, "abort"):
            brain.abort()
        if status:
            status.update(state="listening")
        return True

    # Ignore an identical agent request repeated within a few seconds (self-echo).
    if key and key == state.get("last_cmd") and now - state.get("last_cmd_t", 0) < 8:
        print(f"[heard] ignoring repeated request: {command!r}")
        return False
    state["last_cmd"] = key
    state["last_cmd_t"] = now

    tag = f" | wake: {alias}" if alias else " | conversation"
    print(f"[heard] {text}  (command: {command}{tag})")

    cancel = state.setdefault("cancel", threading.Event())
    memory = state.get("memory")
    brain_key = getattr(brain, "session_key", lambda: None)()
    history = memory.messages(brain_key) if memory else None

    # One speech queue for the whole turn (ack + streamed reply sentences).
    sq: queue.Queue = queue.Queue()
    started = False
    spoken: list[str] = []
    speak_lock = threading.Lock()
    max_speak = int((cfg.get("feedback", {}) or {}).get("max_speak_sentences", 3) or 0)
    if max_speak and re.search(
            r"\b(?:in|into|with|using)\s+(?:three|four|five|six|seven|eight|nine|"
            r"ten|eleven|twelve|\d+)\s+(?:sentences?|points?|lines?|steps?|things?)\b"
            r"|\bin detail\b|\bstep by step\b|\bfull (?:answer|explanation)\b",
            command, re.I):
        max_speak = 0   # you asked for length; don't cut it short
    capped = threading.Event()

    def emit(sentence: str) -> None:
        """Speak a completed sentence (called by the brain as it generates)."""
        abort_now = False
        with speak_lock:
            if cancel.is_set() or _normalize_key(sentence) == "skip":
                return   # "SKIP" = the brain says this wasn't addressed
            if max_speak and len(spoken) >= max_speak:
                abort_now = not capped.is_set()
                capped.set()
            else:
                spoken.append(sentence)
        if abort_now:
            print(f"[cap] reply longer than {max_speak} sentences; stopping it")
            if hasattr(brain, "abort"):
                try:
                    brain.abort()
                except Exception:
                    pass
            return
        if capped.is_set():
            return
        _note_spoken(state, sentence)
        if status:
            status.update(state="speaking", reply=" ".join(spoken))
        print(f"{cfg['name']}: {sentence}")
        if not no_tts:
            for chunk in split_for_speech(sentence):
                sq.put(chunk)

    def close_speech() -> None:
        nonlocal started
        if started:
            sq.put(None)
            started = False

    if not no_tts:
        speaker.speak_queue(sq, tts.synthesize)
        started = True
        ack = _ack_phrase(command, cfg.get("feedback", {}) or {})
        if ack:
            _note_spoken(state, ack)
            sq.put(ack)
        _start_work_filler(sq, cfg, status, cancel, spoken, speak_lock, state)

    if status:
        status.update(state="thinking", reply="")

    if no_llm or not getattr(brain, "enabled", True):
        reply = command
        emit(reply)
    else:
        try:
            reply = brain.stream_reply(command, on_sentence=emit, history=history)
        except Exception as exc:
            if "abort" in str(exc).lower():
                print("[stop] request aborted")
                close_speech()
                return True
            print(f"[brain] unavailable: {exc}", file=sys.stderr)
            fb = cfg.get("feedback", {}) or {}
            reply = cfg.get("offline_reply", "My brain is offline.")
            print(f"{cfg['name']}: {reply}")
            if not no_tts:
                if fb.get("enabled", True) and fb.get("error", True) \
                        and fb.get("error_text"):
                    _note_spoken(state, fb["error_text"])
                    sq.put(fb["error_text"])
                _note_spoken(state, reply)
                sq.put(reply)
            if status:
                status.update(state="error", reply=reply)
            close_speech()
            return True

    if reply.strip().upper() == "SKIP":
        print("[skip] agent says not addressed")
        close_speech()
        return True
    if cancel.is_set():
        print("[stop] request cancelled before speaking")
        close_speech()
        return True
    if memory is not None:
        # The session may have been created during this turn; store under its key.
        memory.add(command, reply,
                   key=getattr(brain, "session_key", lambda: None)())
    if reply and not spoken:
        # Reply arrived without streaming (defensive): speak it now.
        emit(clean_for_speech(reply))
    if status:
        status.update(state="speaking", reply=reply or " ".join(spoken))
    close_speech()
    return True


def _start_work_filler(sq: queue.Queue, cfg: dict, status: Status | None,
                       cancel: threading.Event, spoken: list, lock,
                       state: dict | None = None) -> None:
    """Say a short 'let me see' if the agent is working away with tools and the
    user has heard nothing for a while.  One per turn, only while still silent."""
    fb = cfg.get("feedback", {}) or {}
    fillers = fb.get("work_fillers") or []
    after = float(fb.get("work_filler_after_s", 0) or 0)
    if not fillers or after <= 0:
        return

    def run() -> None:
        deadline = time.time() + after
        while time.time() < deadline:
            if cancel.is_set() or spoken:
                return
            time.sleep(0.25)
        with lock:
            if cancel.is_set() or spoken:
                return
            actions = (status.snapshot().get("actions") if status else None) or []
            if not actions:
                return   # still just thinking; silence is fine
            phrase = random.choice(fillers)
            if state is not None:
                _note_spoken(state, phrase)
            sq.put(phrase)

    threading.Thread(target=run, daemon=True).start()


def _make_memory(cfg: dict) -> ConversationMemory | None:
    mem_cfg = cfg.get("memory", {}) or {}
    if not mem_cfg.get("enabled", True):
        return None
    path = mem_cfg.get("path") if mem_cfg.get("persist", True) else None
    return ConversationMemory(path=path, max_turns=int(mem_cfg.get("history_turns", 8)))


def _memory_for(cfg: dict, brain) -> ConversationMemory | None:
    """Conversation memory, with the pre-session-format history adopted into
    the current session so an upgrade doesn't forget the conversation."""
    memory = _make_memory(cfg)
    if memory is not None:
        memory.adopt("chat", getattr(brain, "session_key", lambda: None)())
    return memory


def _make_status(cfg: dict) -> Status:
    st_cfg = cfg.get("status", {}) or {}
    return Status(st_cfg.get("path", "/home/infinite/.cache/seeker/status.json"),
                  enabled=st_cfg.get("enabled", True))


# -- Omarchy panel lifecycle ------------------------------------------------
_OMARCHY_SHELL_DIR = "/usr/share/omarchy/shell"
_PANEL_ID = "seeker.status"


def _qs_ipc(args: list[str], timeout: float = 8.0):
    env = os.environ.copy()
    if not env.get("WAYLAND_DISPLAY"):
        runtime = env.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        try:
            socks = sorted(f for f in os.listdir(runtime)
                           if f.startswith("wayland-") and not f.endswith(".lock"))
            if socks:
                env["WAYLAND_DISPLAY"] = socks[-1]
        except OSError:
            pass
    try:
        return subprocess.run(
            ["quickshell", "ipc", "-p", _OMARCHY_SHELL_DIR, "call", *args],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except Exception:
        return None


def _panel_state() -> str | None:
    """Return "open"/"closed" if the panel is mounted, else None (shell gone)."""
    r = _qs_ipc([_PANEL_ID, "panelState"])
    if r is None or r.returncode != 0:
        return None
    out = (r.stdout or "").strip()
    return out if out in ("open", "closed") else None


def _summon_panel() -> None:
    _qs_ipc(["shell", "summon", _PANEL_ID, "{}"])


def _maybe_reopen_panel() -> None:
    """Show the panel on an interaction: mount it if the shell hasn't yet, and
    tell a loaded panel to open (so it reappears even after the user pressed ×)."""
    _summon_panel()


def _watch_panel(stop_event: threading.Event) -> None:
    """Keep the panel mounted across shell restarts, without reopening it if the
    user closed it with ×."""
    if _panel_state() is None:          # not mounted -> summon now
        _summon_panel()
    while not stop_event.wait(30):
        if _panel_state() is None:      # shell restarted -> remount
            _summon_panel()


def _open_mic(a: dict, timeout_s: float = 300.0) -> MicStream:
    """Pick a real capture source and open the input stream, retrying until the
    device appears.  At boot the USB mic / PipeWire may not be ready yet, so we
    wait instead of crash-looping."""
    # Make sure the USB mic (not a monitor) is present and is the default source.
    source = ensure_input_source(timeout_s=120.0)
    if source:
        os.environ["PULSE_SOURCE"] = source
    device = a.get("input_device") or ("pulse" if source else "default")

    deadline = time.time() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        mic = MicStream(a["sample_rate"], a["frame_ms"], device)
        try:
            return mic.open()
        except Exception as exc:
            if time.time() >= deadline:
                raise
            print(f"[audio] input device not ready (attempt {attempt}): {exc}; "
                  f"retrying in 2s", file=sys.stderr, flush=True)
            if attempt % 10 == 0:
                ensure_input_source(timeout_s=10.0)
            time.sleep(2.0)


def cmd_text(args, cfg: dict) -> int:
    status = _make_status(cfg)
    brain = build_brain(cfg, status=status)
    tts = TTSClient(cfg["tts"])
    if not args.no_tts:
        tts.ensure_running()
    speaker = Speaker(sink=cfg["audio"].get("output_sink"),
                      device=cfg["audio"].get("output_device"))
    _handle_transcript(args.text, cfg, brain, tts, speaker,
                       {"memory": _memory_for(cfg, brain)}, status, args.no_tts, args.no_llm)
    speaker.wait()
    status.update(state="offline")
    return 0


def cmd_transcribe(args, cfg: dict) -> int:
    asr = build_asr(cfg["asr"])
    pcm, sr = _load_wav(args.transcribe)
    print(asr.transcribe(pcm, sr))
    return 0


def cmd_say(args, cfg: dict) -> int:
    tts = TTSClient(cfg["tts"])
    if not tts.ensure_running():
        print("TTS sidecar is not available", file=sys.stderr)
        return 1
    data = tts.synthesize(args.say)
    if not data:
        print("synthesis failed", file=sys.stderr)
        return 1
    speaker = Speaker(sink=cfg["audio"].get("output_sink"),
                      device=cfg["audio"].get("output_device"))
    speaker.play(data)
    speaker.wait()
    return 0


def cmd_calibrate(args, cfg: dict) -> int:
    a = cfg["audio"]
    seg = UtteranceSegmenter(
        a["sample_rate"], a["frame_ms"], silence_ms=a["silence_ms"],
        min_speech_ms=a["min_speech_ms"], preroll_ms=a["preroll_ms"],
        max_utterance_s=a["max_utterance_s"], energy_ratio=a["energy_ratio"],
        noise_floor_alpha=a["noise_floor_alpha"], min_rms=a["min_rms"],
        silence_max_ms=a.get("silence_max_ms", 1500),
    )
    print("Calibrating ~10s. Stay quiet, then say 'seeker'.")
    deadline = time.time() + 10
    with MicStream(a["sample_rate"], a["frame_ms"], a["input_device"]) as mic:
        while time.time() < deadline:
            frame = mic.read()
            rms = UtteranceSegmenter.frame_rms(frame)
            thr = max(seg._floor * seg.energy_ratio, seg.min_rms)
            utt = seg.feed(frame)
            flag = "SPEECH" if rms > thr else "      "
            print(f"\rrms={rms:.4f} floor={seg._floor:.4f} thr={thr:.4f} {flag}",
                  end="", flush=True)
            if utt is not None:
                print(f"\n-> utterance {utt.size / a['sample_rate']:.2f}s")
    print()
    return 0


def _cancel_turn(state: dict, brain, speaker: Speaker,
                 mic: MicStream | None = None, seg: UtteranceSegmenter | None = None,
                 status: Status | None = None) -> None:
    """Abort an in-flight turn: silence speech, kill the generation, reset the VAD."""
    cancel = state.get("cancel")
    if cancel is not None:
        cancel.set()
    if hasattr(brain, "abort"):
        brain.abort()
    speaker.stop()
    if mic is not None:
        mic.flush()
    if seg is not None:
        seg.reset()
    if status is not None:
        status.update(state="listening")


def cmd_loop(args, cfg: dict) -> int:
    a = cfg["audio"]
    asr = build_asr(cfg["asr"])
    status = _make_status(cfg)
    brain = build_brain(cfg, status=status)
    tts = TTSClient(cfg["tts"])
    if not args.no_tts and tts.ensure_running():
        # warm the ONNX session + the short stock phrases so the first real
        # sentence isn't a cold start
        def _warm_tts() -> None:
            fb = cfg.get("feedback", {}) or {}
            phrases = ["Ready.", cfg.get("wake_reply", "Yes?"),
                       *(fb.get("ack_actions") or _ACK_ACTIONS)]
            for phrase in fb.get("startup_text"), fb.get("error_text"):
                if phrase:
                    phrases.append(phrase)
            for phrase in dict.fromkeys(phrases):
                tts.synthesize(phrase)
        threading.Thread(target=_warm_tts, daemon=True).start()

    print(f"[setup] loading ASR ({asr.name}) ...", flush=True)
    try:
        asr.warmup()
    except Exception as exc:
        print(f"[setup] ASR warmup failed: {exc}", file=sys.stderr)

    if getattr(brain, "name", "") in ("opencode", "auto"):
        print("[setup] starting opencode brain ...", flush=True)
        try:
            brain.ensure_server()
        except Exception as exc:
            print(f"[setup] opencode brain unavailable: {exc}", file=sys.stderr)

    panel_stop = threading.Event()
    if (cfg.get("panel", {}) or {}).get("enabled", True):
        threading.Thread(target=_watch_panel, args=(panel_stop,), daemon=True).start()

    seg = UtteranceSegmenter(
        a["sample_rate"], a["frame_ms"], silence_ms=a["silence_ms"],
        min_speech_ms=a["min_speech_ms"], preroll_ms=a["preroll_ms"],
        max_utterance_s=a["max_utterance_s"], energy_ratio=a["energy_ratio"],
        noise_floor_alpha=a["noise_floor_alpha"], min_rms=a["min_rms"],
        silence_max_ms=a.get("silence_max_ms", 1500),
    )
    speaker = Speaker(sink=a.get("output_sink"), device=a.get("output_device"))
    barge = cfg.get("barge_in", {}) or {}
    barge_ratio = float(barge.get("ratio", 6.0))
    barge_min = float(barge.get("min_rms", 0.05))

    mode = "always listening (no wake word)" if not cfg.get("require_wake", True) \
        else f'say "{cfg["name"]}" to start, then just keep talking'

    mic = _open_mic(a)
    print(f"{cfg['name']} is listening. {mode} (Ctrl-C to stop)")
    status.update(state="listening", **getattr(brain, "session_info", lambda: {})())
    _feedback(speaker, tts, cfg, "startup", args.no_tts)  # spoken "I'm up"

    state: dict = {"memory": _memory_for(cfg, brain)}
    was_playing = False
    try:
        while True:
            frame = mic.read()
            now = time.time()

            if speaker.is_playing():
                # While talking, only react to a loud voice = barge-in.
                if now < state.get("ignore_until", 0.0):
                    continue
                if UtteranceSegmenter.frame_rms(frame) > seg.barge_threshold(
                        barge_ratio, barge_min):
                    print("[barge-in] interrupted, listening ...")
                    _cancel_turn(state, brain, speaker, mic, seg)
                    was_playing = False
                else:
                    was_playing = True
                    continue
            else:
                if was_playing:
                    # Playback just finished: drop the speaker tail and resume.
                    mic.flush()
                    seg.reset()
                    state["ignore_until"] = now + 0.25
                    was_playing = False
                    status.update(state="listening")
                    continue
                if now < state.get("ignore_until", 0.0):
                    continue

            # Fragment stitching: if the last transcript sounded unfinished,
            # hold the answer until the rest of the sentence arrives (or the
            # continuation window expires).
            stitch_s = float(a.get("stitch_s", 3.0))
            text = ""
            held = state.get("held")
            if held is not None and now >= state.get("stitch_until", 0.0) \
                    and not seg.speaking:
                state["held"] = None
                text = held
                print(f"[stitch] no continuation after {stitch_s:.1f}s: {text!r}")
            else:
                utt = seg.feed(frame)
                if utt is None:
                    continue

                t0 = time.time()
                text = asr.transcribe(utt, a["sample_rate"])
                dt = time.time() - t0
                if not text:
                    continue
                print(f"[asr {dt:.2f}s] {text}")
                status.add_heard(text)          # always show what the mic hears
                if _is_hallucination(text):
                    continue
                if state.get("held") is not None:
                    text = _merge_fragments(state["held"], text, cfg)
                    state["held"] = None
                    print(f"[stitch] merged -> {text!r}")
                if _looks_incomplete(text):
                    state["held"] = text
                    state["stitch_until"] = time.time() + stitch_s
                    status.update(state="heard", heard=text)
                    continue

            handler = state.get("handler")
            running = handler is not None and handler.is_alive()
            if running:
                # A request is in flight (thinking/acting): only "stop" and a
                # fresh wake-word request get through.
                if _is_stop(text, cfg, state):
                    print(f"[stop] {text!r}")
                    _cancel_turn(state, brain, speaker, mic, seg, status)
                    continue
                elif text:
                    matched, _cmd, _alias = detect(
                        text, cfg.get("aliases"),
                        leading_mishears=cfg.get("leading_mishears"))
                    if matched:
                        # Addressing Seeker again mid-task replaces the request.
                        print(f"[replace] new request while busy: {text!r}")
                        _cancel_turn(state, brain, speaker, mic, seg)
                        handler.join(timeout=2.0)
                    else:
                        print(f"[busy] ignoring while working: {text}")
                        continue
                else:
                    continue

            state["cancel"] = threading.Event()
            state["handler"] = threading.Thread(
                target=_handle_transcript,
                args=(text, cfg, brain, tts, speaker, state, status,
                      args.no_tts, args.no_llm),
                daemon=True,
            )
            state["handler"].start()
            if args.once:
                state["handler"].join()
                speaker.wait()
                break
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        mic.close()
        speaker.stop()
        if hasattr(brain, "stop_monitor"):
            brain.stop_monitor()
        panel_stop.set()
        status.update(state="offline")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seeker", description="Seeker voice wake assistant")
    ap.add_argument("--config", default=None, help="path to config.json")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--text", help="simulate a transcript (no mic)")
    ap.add_argument("--transcribe", metavar="WAV", help="transcribe a wav file and exit")
    ap.add_argument("--say", metavar="TEXT", help="synthesize text and play it")
    ap.add_argument("--calibrate", action="store_true", help="print live audio levels")
    ap.add_argument("--once", action="store_true", help="handle one utterance then exit")
    ap.add_argument("--no-wake", action="store_true",
                    help="never require the wake word (talk constantly)")
    ap.add_argument("--conversation", type=float, metavar="SECONDS",
                    help="seconds to keep the no-wake conversation open (0 = forever)")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--brain", choices=["opencode", "chat"],
                    help="override the brain backend for this run")
    args = ap.parse_args(argv)

    if args.list_devices:
        print(list_devices())
        return 0

    cfg = config_mod.load(args.config)
    if args.no_wake:
        cfg["require_wake"] = False
    if args.conversation is not None:
        cfg.setdefault("conversation", {})["timeout_s"] = args.conversation
        cfg["conversation"]["enabled"] = True
    if args.brain:
        cfg.setdefault("brain", {})["backend"] = args.brain

    if args.text:
        return cmd_text(args, cfg)
    if args.transcribe:
        return cmd_transcribe(args, cfg)
    if args.say:
        return cmd_say(args, cfg)
    if args.calibrate:
        return cmd_calibrate(args, cfg)
    return cmd_loop(args, cfg)


if __name__ == "__main__":
    sys.exit(main())