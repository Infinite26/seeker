"""Live status feed for the Seeker Omarchy panel.

Writes a small JSON document (~/.cache/seeker/status.json) that the Quickshell
panel watches: what Seeker is doing, the last heard/reply, the current session,
and (while browsing) the list of existing opencode sessions.
"""
from __future__ import annotations

import json
import os
import threading
import time


class Status:
    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self._data = {
            "state": "starting",     # starting|listening|heard|thinking|speaking|error|offline
            "action": "",            # current tool activity, e.g. "bash: git status"
            "actions": [],           # rolling list of recent activities
            "heard": "",
            "transcript": [],       # rolling list of everything ASR heard
            "reply": "",
            "session": "",
            "session_title": "",
            "turns": 0,
            "seq": 0,                # increments once per accepted request
            "mode": "activity",      # activity|sessions
            "sessions": [],
            "selected": -1,
            "ts": time.time(),
        }
        self._flush()

    def update(self, **fields) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data.update(fields)
            self._data["ts"] = time.time()
            self._flush()

    def bump_seq(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data["seq"] = int(self._data.get("seq", 0)) + 1
            self._data["actions"] = []
            self._data["action"] = ""
            self._data["ts"] = time.time()
            self._flush()

    def add_heard(self, text: str, keep: int = 6) -> None:
        """Record every ASR transcript (even when not addressed) so the panel can
        show what the mic is hearing."""
        if not self.enabled or not text:
            return
        with self._lock:
            self._data["heard"] = text
            history = list(self._data.get("transcript", []))
            if not history or history[-1] != text:
                history.append(text)
            self._data["transcript"] = history[-keep:]
            self._data["ts"] = time.time()
            self._flush()

    def add_action(self, text: str, keep: int = 5) -> None:
        if not self.enabled or not text:
            return
        with self._lock:
            actions = list(self._data.get("actions", []))
            # The event stream can replay recent parts on reconnect; skip a
            # repeat that is already in the visible history.
            if any(a.get("text") == text for a in actions[-4:]):
                return
            actions.append({"t": time.time(), "text": text})
            self._data["actions"] = actions[-keep:]
            self._data["action"] = text
            self._data["ts"] = time.time()
            self._flush()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def _flush(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            # Write in place (single write) rather than rename: replacing the
            # file changes its inode and breaks file watchers (the panel).
            payload = json.dumps(self._data)
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
        except OSError:
            pass
