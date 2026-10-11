"""Short-term conversation memory for Seeker (per session, optionally persisted).

Each conversation (opencode session) gets its own rolling transcript, so
"new session" really starts fresh and "continue" brings the old chat history
back with it.
"""
from __future__ import annotations

import json
import os

_UNSET = object()


class ConversationMemory:
    """Rolling transcript of recent turns fed back to the LLM as chat history,
    kept separately per session key."""

    def __init__(self, path: str | None = None, max_turns: int = 8,
                 max_sessions: int = 40):
        self.path = path
        self.max_turns = max(1, int(max_turns))
        self.max_sessions = max(1, int(max_sessions))
        self.key: str | None = None
        self.turns: list[dict] = []
        self._sessions: dict[str, list[dict]] = {}
        if self.path:
            self.load()

    # -- storage ---------------------------------------------------------
    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and isinstance(data.get("sessions"), dict):
            self._sessions = {
                k: [t for t in (v or []) if isinstance(t, dict)
                    and t.get("role") in ("user", "assistant")][-2 * self.max_turns:]
                for k, v in data["sessions"].items() if isinstance(v, list)
            }
        elif isinstance(data, dict) and isinstance(data.get("turns"), list):
            # migrate the old single-transcript format
            self._sessions = {"chat": data["turns"][-2 * self.max_turns:]}
        elif isinstance(data, list):
            self._sessions = {"chat": data[-2 * self.max_turns:]}

    def save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            # keep the newest sessions only
            for old in list(self._sessions)[:-self.max_sessions]:
                if old != self.key:
                    self._sessions.pop(old, None)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"sessions": self._sessions}, fh, indent=2)
        except OSError:
            pass

    # -- per-session transcript -----------------------------------------
    def messages(self, key: str | None | object = _UNSET) -> list[dict]:
        """History for `key` (omitted = the current session).  An explicit
        ``None`` means "no session yet" and returns nothing — a fresh
        conversation must not see the previous one's history."""
        if key is _UNSET:
            key = self.key
        if not key:
            return []
        return list(self._sessions.get(key, []))

    def add(self, user: str, assistant: str, key: str | None | object = _UNSET) -> None:
        if key is _UNSET:
            key = self.key
        if not key:
            key = "chat"
        turns = self._sessions.get(key, [])
        turns = turns + [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]
        self._sessions[key] = turns[-2 * self.max_turns:]
        self.key = key
        self.turns = self._sessions[key]
        self.save()

    def clear(self, key: str | None = None) -> None:
        if key is None:
            key = self.key
        if key:
            self._sessions.pop(key, None)
        self.turns = []
        self.save()

    def switch(self, key: str | None) -> None:
        """Make `key` the current session (used by session-management commands)."""
        self.key = key or None
        self.turns = list(self._sessions.get(self.key, [])) if self.key else []

    def adopt(self, src_key: str, key: str | None) -> None:
        """Move one bucket's turns onto `key` (old-format migration)."""
        turns = self._sessions.pop(src_key, None)
        if turns and key and not self._sessions.get(key):
            self._sessions[key] = turns
            self.save()
