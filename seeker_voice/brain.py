"""Brains for Seeker.

OpenCodeBrain drives a persistent headless ``opencode serve`` over its HTTP API
so voice commands run through the full opencode agent (tools, shell, files)
instead of a plain chat completion.  Both brains stream: replies are handed back
sentence by sentence as they are generated, so TTS starts while the model is
still writing.

API used:
    POST /session                      -> create a session
    POST /session/{id}/message         -> send a message, return the reply
    GET  /session                      -> list existing sessions
    GET  /event                        -> SSE stream (live tool + text deltas)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import requests

from .tts import SentenceEmitter, clean_for_speech

_DEFAULT_BINARY_CANDIDATES = (
    "opencode",
    "/home/infinite/.local/share/mise/shims/opencode",
    "/home/infinite/.local/bin/opencode",
    "/usr/bin/opencode",
)

_DETAIL_KEYS = ("command", "filePath", "path", "pattern", "url", "description",
                "query", "prompt", "message")


def _clean_for_speech(text: str) -> str:
    return clean_for_speech(text)


def _title_from(command: str, words: int = 7) -> str:
    """A short human title for a new conversation, from its first request."""
    parts = re.findall(r"[A-Za-z0-9'’\-]+", command or "")
    title = " ".join(parts[:words]).strip(" ,.")
    return (title[:1].upper() + title[1:]) if title else ""


def _tool_detail(tool: str, inp) -> str:
    if isinstance(inp, dict):
        for key in _DETAIL_KEYS:
            value = inp.get(key)
            if value:
                return str(value).replace("\n", " ")[:90]
    return ""


class BrainError(RuntimeError):
    pass


class OpenCodeBrain:
    name = "opencode"

    def __init__(self, cfg: dict, fallback=None, persona: str | None = None,
                 status=None):
        self.cfg = dict(cfg or {})
        self.fallback = fallback
        self.persona = persona
        self.status = status
        self.enabled = True
        self.url = (self.cfg.get("server_url") or "http://127.0.0.1:4097").rstrip("/")
        self.workdir = self.cfg.get("workdir") or "/home/infinite/.cache/seeker/workspace"
        self.config_dir = self.cfg.get("config_dir")
        self.binary = self._resolve_binary(self.cfg.get("binary", "opencode"))
        self.autostart = bool(self.cfg.get("autostart", True))
        self.timeout_s = float(self.cfg.get("timeout_s", 180))
        self.model = self.cfg.get("model", "opencode-go/deepseek-v4-flash")
        self.agent = self.cfg.get("agent", "seeker")
        self.session_file = self.cfg.get("session_file",
                                         "/home/infinite/.cache/seeker/opencode_session.json")
        self.sessions_file = self.cfg.get("sessions_file",
                                          "/home/infinite/.cache/seeker/sessions.json")
        self.sessions_limit = int(self.cfg.get("sessions_limit", 12))
        self.pinned_session = self.cfg.get("session_id")
        # opencode scopes sessions by directory; every session-scoped request must
        # carry it or the server cannot find the session after a restart.
        self.params = {"directory": self.workdir}
        self._session_id: str | None = None
        self._session_title: str = ""
        self._turns: int = 0
        self._proc: subprocess.Popen | None = None
        self._busy = False
        self._stream: SentenceEmitter | None = None
        self._user_mid: str | None = None
        self._turn_lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._load_state()

    # -- process / server -------------------------------------------------
    @staticmethod
    def _resolve_binary(name: str) -> str:
        found = shutil.which(name)
        if found:
            return found
        for cand in _DEFAULT_BINARY_CANDIDATES:
            if os.path.isabs(cand) and os.path.exists(cand):
                return cand
        return name

    def healthy(self) -> bool:
        try:
            r = requests.get(self.url + "/session", timeout=4)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def ensure_server(self, wait_s: float = 60.0) -> bool:
        if self.healthy():
            self.start_monitor()
            return True
        if not self.autostart:
            return False

        os.makedirs(self.workdir, exist_ok=True)
        port = self.url.rsplit(":", 1)[-1]
        env = os.environ.copy()
        if self.config_dir:
            env["OPENCODE_CONFIG_DIR"] = self.config_dir
        env.setdefault("OPENCODE_DISABLE_AUTOUPDATE", "1")
        cmd = [self.binary, "serve", "--hostname", "127.0.0.1", "--port", str(port)]
        print(f"[brain] starting: {' '.join(cmd)}", file=sys.stderr, flush=True)
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=self.workdir, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            print(f"[brain] cannot start opencode server: {exc}", file=sys.stderr)
            return False

        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.healthy():
                print("[brain] opencode server ready", file=sys.stderr, flush=True)
                self.start_monitor()
                return True
            if self._proc.poll() is not None:
                print(f"[brain] opencode server exited ({self._proc.returncode})",
                      file=sys.stderr)
                return False
            time.sleep(1.0)
        return False

    # -- persisted session state -----------------------------------------
    def _load_state(self) -> None:
        try:
            with open(self.sessions_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            current = data.get("current") or {}
            if current.get("id"):
                self._session_id = current["id"]
                self._session_title = current.get("title", "")
                if current.get("directory"):
                    self.params["directory"] = current["directory"]
                self._turns = int(data.get("turns", 0))
            return
        except (OSError, ValueError):
            pass
        # migrate from the old single-session file
        try:
            with open(self.session_file, "r", encoding="utf-8") as fh:
                sid = json.load(fh).get("session_id")
            if sid:
                self._session_id = sid
        except (OSError, ValueError):
            pass

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.sessions_file), exist_ok=True)
            payload = {
                "current": {
                    "id": self._session_id,
                    "directory": self.params.get("directory"),
                    "title": self._session_title,
                },
                "turns": self._turns,
            }
            tmp = f"{self.sessions_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.sessions_file)
        except OSError:
            pass

    # -- sessions ---------------------------------------------------------
    def _session_exists(self, sid: str, directory: str | None = None) -> bool:
        """True if reachable. On a network error assume it exists so a transient
        hiccup never forks a new conversation."""
        params = {"directory": directory} if directory else self.params
        try:
            r = requests.get(f"{self.url}/session/{sid}", params=params, timeout=6)
            if r.status_code == 200:
                return True
            if r.status_code == 404:
                return False
            return True
        except requests.RequestException:
            return True

    def ensure_session(self, title_hint: str | None = None) -> str:
        if self.pinned_session:
            self._session_id = self.pinned_session
            return self._session_id
        if self._session_id and self._session_exists(self._session_id):
            return self._session_id
        self._session_id = None
        return self.new_session(title=_title_from(title_hint) if title_hint else None)

    def session_key(self) -> str | None:
        """Identity of the current conversation (used to key chat memory)."""
        return self._session_id

    def note_turn(self) -> None:
        """Count a turn that happened outside the agent (chat fast path)."""
        self._turns += 1
        self._save_state()
        self._sync_status()

    def new_session(self, directory: str | None = None, title: str | None = None) -> str:
        directory = directory or self.workdir
        # Let opencode auto-title from the first request unless a title is given.
        body = {"title": title} if title else {}
        r = requests.post(self.url + "/session", json=body,
                          params={"directory": directory}, timeout=30)
        r.raise_for_status()
        sid = r.json().get("id")
        if not sid:
            raise BrainError(f"session create returned no id: {r.text[:200]}")
        self._session_id = sid
        self._session_title = title or ""
        self.params["directory"] = directory
        self._turns = 0
        self._save_state()
        self._sync_status()
        print(f"[brain] new session {sid} in {directory}", file=sys.stderr, flush=True)
        return sid

    def _refresh_title(self) -> None:
        sid = self._session_id
        if not sid:
            return
        try:
            r = requests.get(f"{self.url}/session/{sid}", params=self.params, timeout=6)
            if r.status_code != 200:
                return
            title = (r.json() or {}).get("title")
            if title and title != self._session_title:
                self._session_title = title
                self._save_state()
                self._sync_status()
        except requests.RequestException:
            pass

    def resume_session(self, session: dict) -> str:
        sid = session.get("id")
        directory = session.get("directory") or self.workdir
        title = session.get("title") or ""
        if not sid or not self._session_exists(sid, directory):
            raise BrainError("session not found")
        self._session_id = sid
        self._session_title = title
        self.params["directory"] = directory
        self._turns = 0
        self._save_state()
        self._sync_status()
        print(f"[brain] resumed session {sid} ({title})", file=sys.stderr, flush=True)
        return sid

    def end_current(self) -> None:
        self._session_id = None
        self._session_title = ""
        self._turns = 0
        self._save_state()
        self._sync_status()

    def abort(self) -> None:
        """Abort the in-flight agent run for the current session."""
        sid = self._session_id
        if not sid:
            return
        try:
            requests.post(f"{self.url}/session/{sid}/abort",
                          params=self.params, timeout=5)
            print(f"[brain] aborted {sid[:12]}", file=sys.stderr, flush=True)
        except requests.RequestException:
            pass

    def reset(self) -> None:  # "start over" -> brand new session
        try:
            self.new_session()
        except Exception as exc:
            print(f"[brain] reset failed: {exc}", file=sys.stderr)

    def list_sessions(self, limit: int | None = None) -> list[dict]:
        limit = limit or self.sessions_limit
        try:
            r = requests.get(self.url + "/session", timeout=10)
            r.raise_for_status()
            sessions = r.json()
        except Exception as exc:
            print(f"[brain] list sessions failed: {exc}", file=sys.stderr)
            return []
        out = []
        for s in sessions:
            if not isinstance(s, dict):
                continue
            out.append({
                "id": s.get("id"),
                "title": s.get("title") or "(untitled)",
                "directory": s.get("directory") or "",
                "updated": (s.get("time") or {}).get("updated", 0),
            })
        out.sort(key=lambda s: s.get("updated", 0), reverse=True)
        return out[:limit]

    def session_info(self) -> dict:
        return {
            "session": (self._session_id or "")[:12],
            "session_title": self._session_title,
            "turns": self._turns,
        }

    def _sync_status(self) -> None:
        if self.status:
            self.status.update(**self.session_info())

    # -- reply ------------------------------------------------------------
    def stream_reply(self, command: str, on_sentence=None, history=None) -> str:
        """Run one agent turn.  `on_sentence` gets each spoken sentence as soon
        as the model finishes it (before the whole reply exists)."""
        try:
            return self._reply_opencode(command, on_sentence)
        except Exception as exc:
            # An abort (user said "stop" / barge-in) is intentional: never fall back.
            if "abort" in str(exc).lower():
                print(f"[brain] aborted: {exc}", file=sys.stderr)
                raise BrainError(str(exc))
            print(f"[brain] opencode failed: {exc}", file=sys.stderr)
            if self.fallback is not None:
                fb_stream = getattr(self.fallback, "stream_reply", None)
                if fb_stream is not None:
                    return fb_stream(command, history=history, on_sentence=on_sentence)
                return self.fallback.reply(command, history=history)
            raise BrainError(str(exc))

    def reply(self, command: str, history=None) -> str:
        return self.stream_reply(command, history=history)

    def _reply_opencode(self, command: str, on_sentence=None) -> str:
        with self._turn_lock:  # one turn at a time (an aborted turn must unwind first)
            return self._reply_locked(command, on_sentence)

    def _reply_locked(self, command: str, on_sentence=None) -> str:
        if not self.ensure_server():
            raise BrainError("opencode server not reachable")
        sid = self.ensure_session(title_hint=command)
        self._turns += 1
        self._save_state()
        self._sync_status()

        provider_id, _, model_id = self.model.partition("/")
        # Give the user message a known id so the event stream can tell our
        # question apart from the assistant's answer.
        self._user_mid = "msg" + uuid.uuid4().hex[:20]
        emitter = SentenceEmitter(on_sentence) if on_sentence else None
        self._stream = emitter
        body = {
            "messageID": self._user_mid,
            "model": {"providerID": provider_id, "modelID": model_id or self.model},
            "agent": self.agent,
            "parts": [{"type": "text", "text": command}],
        }
        self._busy = True
        try:
            r = requests.post(f"{self.url}/session/{sid}/message", json=body,
                              params=self.params, timeout=self.timeout_s)
            if r.status_code == 404:
                # stale/deleted session; recreate once and retry on the new one
                sid = self.new_session()
                r = requests.post(f"{self.url}/session/{sid}/message", json=body,
                                  params=self.params, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        finally:
            self._busy = False
            self._stream = None
            if emitter is not None:
                emitter.seal()

        info = data.get("info") or {}
        if info.get("error"):
            raise BrainError(str(info["error"])[:300])

        chunks = []
        for part in data.get("parts", []):
            if part.get("type") == "text" and not part.get("synthetic") \
                    and not part.get("ignored"):
                chunks.append(part.get("text") or "")
                if emitter is not None:
                    emitter.feed(part.get("id") or "part", part.get("text") or "")
        text = _clean_for_speech("".join(chunks))
        if not text:
            raise BrainError("empty reply from opencode")
        if emitter is not None:
            emitter.flush()
        self._refresh_title()   # opencode may have auto-titled from this turn
        self._sync_status()
        return text

    # -- live activity monitor (GET /event SSE) ---------------------------
    def start_monitor(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        self._monitor_stop.set()

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.is_set():
            directory = self.params.get("directory")
            try:
                with requests.get(self.url + "/event",
                                  params={"directory": directory},
                                  stream=True, timeout=(5, 30)) as resp:
                    if resp.status_code != 200:
                        time.sleep(2.0)
                        continue
                    resp.encoding = "utf-8"   # SSE has no charset; never Latin-1
                    for line in resp.iter_lines(decode_unicode=True):
                        if self._monitor_stop.is_set():
                            return
                        if self.params.get("directory") != directory:
                            break  # session moved; reconnect with the new directory
                        if not line or not line.startswith("data:"):
                            continue
                        try:
                            self._handle_event(line[5:].strip())
                        except Exception as exc:  # never let one event kill the stream
                            print(f"[brain] event handler error: {exc}",
                                  file=sys.stderr, flush=True)
            except Exception:
                if self._monitor_stop.is_set():
                    return
                time.sleep(1.5)

    def _handle_event(self, raw: str) -> None:
        try:
            event = json.loads(raw)
        except ValueError:
            return
        etype = event.get("type")
        props = event.get("properties") or {}

        if etype == "message.part.delta":
            # Live token deltas: grow the sentence emitter as the model writes.
            stream = self._stream
            if stream is None or props.get("sessionID") != self._session_id:
                return
            if props.get("messageID") == self._user_mid or props.get("field") != "text":
                return
            stream.feed_delta(props.get("partID"), props.get("delta") or "")
            return

        if etype != "message.part.updated":
            return
        part = props.get("part") or {}
        if part.get("sessionID") != self._session_id:
            return
        if part.get("type") == "tool":
            if not self._busy:
                return
            text = self._describe_tool(part)
            if text and self.status:
                self.status.add_action(text)
            return
        # Assistant text parts: register so their deltas count as speech (the
        # user's own question carries our messageID and is skipped).
        stream = self._stream
        if stream is None or part.get("type") != "text":
            return
        if part.get("messageID") == self._user_mid:
            return
        if part.get("synthetic") or part.get("ignored"):
            return
        stream.feed(part.get("id"), part.get("text") or "")

    @staticmethod
    def _describe_tool(part: dict) -> str | None:
        tool = part.get("tool") or "tool"
        state = part.get("state") or {}
        status = state.get("status")
        if status == "running":
            detail = state.get("title") or _tool_detail(tool, state.get("input"))
            return f"{tool}: {detail}" if detail else tool
        if status == "error":
            return f"{tool} failed"
        return None


class ChatBrain:
    """Thin adapter so the plain chat client matches the brain interface."""

    name = "chat"

    def __init__(self, llm, status=None):
        self.llm = llm
        self.status = status
        self.enabled = llm.enabled
        self._turns = 0

    def stream_reply(self, command: str, on_sentence=None, history=None) -> str:
        self._turns += 1
        return self.llm.stream_reply(command, history=history, on_sentence=on_sentence)

    def reply(self, command: str, history=None) -> str:
        return self.stream_reply(command, history=history)

    def reset(self) -> None:  # nothing persistent to reset
        pass

    def end_current(self) -> None:
        self._turns = 0

    def abort(self) -> None:
        pass

    def new_session(self, directory=None, title=None) -> str:
        self._turns = 0
        return ""

    def resume_session(self, session) -> str:
        return ""

    def list_sessions(self, limit=None) -> list:
        return []

    def session_info(self) -> dict:
        return {"session": "", "session_title": "", "turns": self._turns}

    def session_key(self) -> str | None:
        return "chat"

    def ensure_server(self) -> bool:
        return True

    def start_monitor(self) -> None:
        pass

    def stop_monitor(self) -> None:
        pass


# -- auto routing -----------------------------------------------------------
# Conversational turns (questions, greetings, reactions) go to the plain chat
# brain: no tool loop, so the answer starts in ~1-2 s.  Anything that smells
# like work on the machine or the outside world goes to the full agent.

_CHITCHAT_PREFIX = re.compile(
    r"^(?:hi|hey|hello|yo|good (?:morning|afternoon|evening|night)|thanks|thank you|"
    r"thx|cool|nice|awesome|great|wow|okay|ok|bye|goodbye|see you|cheers|"
    r"nice one|well done|appreciate it|that's (?:cool|nice|great|awesome)|"
    r"how are you|how's it going|hows it going|what's up|whats up|who are you|"
    r"what(?:'s| is) your name|what can you do|what do you do|"
    r"tell me (?:a joke|something|about yourself)|say something)\b")

_QUESTION = re.compile(
    r"^(?:what|who|where|when|why|how|which|whose|whats|what's|whos|who's|"
    r"tell me|explain|define|describe|do you|are you|is it|is that|is there|"
    r"can you|could you|would you|will you|should i|should we|does|did|"
    r"have you|are we|do we|may i)\b")

_TOOL_HINTS = re.compile(
    r"\b(?:weather|news|current(?:ly)?|latest|right now|search|google|look up|"
    r"find out|browse|open|close|run|execute|launch|restart|reboot|shutdown|"
    r"install|update|upgrade|download|upload|delete|remove|create|write|edit|"
    r"fix|build|compile|deploy|send|email|remind|timer|alarm|schedule|calendar|"
    r"play|pause|volume|brightness|wifi|bluetooth|screenshot|battery|cpu|gpu|"
    r"ram|disk|storage|files?|folder|directory|project|computer|machine|phone|"
    r"screen|terminal|shell|script|git|docker|server|process|kill|chmod|sudo|"
    r"package|config|system)\b")

_ACTION_VERBS = re.compile(
    r"\b(?:get|open|search|find|play|set|turn|make|start|stop|close|send|add|"
    r"remove|delete|download|install|run|check|show|give|google|write|create|"
    r"remind|call|launch|kill|reboot|shutdown|update|switch)\b")


def looks_conversational(command: str) -> bool:
    """True for turns that are chat, not work (safe to answer without tools)."""
    key = " ".join(re.findall(r"[a-z0-9']+", (command or "").lower()))
    if not key:
        return False
    if _CHITCHAT_PREFIX.match(key):
        return True
    if _TOOL_HINTS.search(key):
        return False
    if _QUESTION.match(key) or key.startswith(("tell me", "explain", "define")):
        return True
    if _ACTION_VERBS.search(key):
        return False
    return len(key.split()) <= 10


class AutoBrain:
    """Routes each turn to the brain that answers it fastest.

    chat  -> plain LLM (fast, conversational, history-aware)
    agent -> full opencode agent (tools, shell, files, sessions)
    """

    name = "auto"

    def __init__(self, agent, chat):
        self.agent = agent
        self.chat = chat
        self.enabled = getattr(agent, "enabled", True) or getattr(chat, "enabled", True)

    def stream_reply(self, command: str, on_sentence=None, history=None) -> str:
        # Make sure the conversation exists as a real (titled) session before
        # routing, so chat-only conversations are listable and resumable too.
        try:
            self.agent.ensure_session(title_hint=command)
        except Exception as exc:
            print(f"[brain] session ensure failed: {exc}", file=sys.stderr)
        if looks_conversational(command) and getattr(self.chat, "enabled", True):
            try:
                reply = self.chat.stream_reply(command, on_sentence=on_sentence,
                                               history=history)
                if hasattr(self.agent, "note_turn"):
                    self.agent.note_turn()
                return reply
            except Exception as exc:
                print(f"[brain] fast path failed ({exc}); using the agent",
                      file=sys.stderr)
        return self.agent.stream_reply(command, on_sentence=on_sentence)

    def reply(self, command: str, history=None) -> str:
        return self.stream_reply(command, history=history)

    # sessions / lifecycle delegate to the agent (it owns the conversation store)
    def abort(self) -> None:
        self.agent.abort()
        self.chat.abort()

    def reset(self) -> None:
        self.agent.reset()

    def new_session(self, directory=None, title=None) -> str:
        return self.agent.new_session(directory, title)

    def resume_session(self, session) -> str:
        return self.agent.resume_session(session)

    def end_current(self) -> None:
        self.agent.end_current()

    def list_sessions(self, limit=None) -> list:
        return self.agent.list_sessions(limit)

    def session_info(self) -> dict:
        return self.agent.session_info()

    def session_key(self) -> str | None:
        return self.agent.session_key()

    def ensure_session(self, title_hint: str | None = None) -> str:
        return self.agent.ensure_session(title_hint=title_hint)

    def ensure_server(self) -> bool:
        return self.agent.ensure_server()

    def start_monitor(self) -> None:
        self.agent.start_monitor()

    def stop_monitor(self) -> None:
        self.agent.stop_monitor()


def build_brain(cfg: dict, status=None):
    """Return the configured brain (auto, opencode or chat)."""
    from .llm import LLMClient

    llm = LLMClient(cfg.get("llm", {}))
    brain_cfg = cfg.get("brain", {}) or {}
    backend = (brain_cfg.get("backend") or "opencode").lower()
    chat = ChatBrain(llm, status=status)
    if backend == "chat":
        return chat
    agent = OpenCodeBrain(brain_cfg.get("opencode", {}), fallback=llm,
                          persona=(cfg.get("llm", {}) or {}).get("system_prompt"),
                          status=status)
    if backend == "auto":
        return AutoBrain(agent, chat)
    return agent