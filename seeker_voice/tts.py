"""Kokoro TTS sidecar client (reuses the Sentient tts_server)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time

import requests

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_MD = re.compile(r"[*_`#>]")
# One complete sentence: runs of text ending in terminal punctuation (plus any
# closing quotes/brackets) followed by whitespace or end-of-buffer.
_SENTENCE_END = re.compile(r".*?[.!?…]+[\"'”’)\]]*(?=\s|$)", re.S)
# Unicode punctuation -> speakable ASCII (TTS reads the rest literally).
_PUNCT = str.maketrans({
    "\u2014": ", ", "\u2013": ", ", "\u2212": "-",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u00a0": " ", "\u2022": " ",
})


def _repair_mojibake(text: str) -> str:
    """Fix UTF-8 read as Latin-1 ('â€”' -> '—'), a common stream-decoding bug."""
    if "â" not in text and "Ã" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        for bad, good in (("â€”", ", "), ("â€“", ", "), ("â€™", "'"),
                          ("â€˜", "'"), ("â€œ", '"'), ("â€\x9d", '"'),
                          ("â€¦", "...")):
            text = text.replace(bad, good)
        return text


def clean_for_speech(text: str) -> str:
    """Strip markdown / code fences / bullets so TTS reads words, not markup."""
    text = _repair_mojibake(text or "")
    text = text.translate(_PUNCT)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)   # drop code fences
    text = _MD.sub("", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)        # markdown links -> text
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)   # bullets
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SentenceEmitter:
    """Turns streamed reply text into spoken sentences.

    Feed it the reply text as it grows (token deltas or part snapshots); every
    time a sentence completes it is cleaned and handed to ``on_sentence`` so the
    TTS pipeline can start speaking while the model is still generating.
    ``flush()`` emits whatever is left when the reply ends.
    """

    def __init__(self, on_sentence=None):
        self.on_sentence = on_sentence
        self._order: list[str] = []
        self._parts: dict[str, str] = {}
        self._consumed = ""
        self._pending = ""
        self._sealed = False
        self._lock = threading.Lock()

    @property
    def parts(self) -> dict[str, str]:
        return self._parts

    def seal(self) -> None:
        """Stop accepting token deltas (the turn is over; final text only)."""
        with self._lock:
            self._sealed = True

    def feed(self, part_id: str, text: str) -> None:
        """Set the authoritative text of one part (delta-accumulated or final)."""
        with self._lock:
            if part_id not in self._parts:
                self._order.append(part_id)
            self._parts[part_id] = text or ""
            self._process()

    def feed_delta(self, part_id: str, delta: str) -> None:
        """Append a token delta to a part (ignored for unknown/finished parts)."""
        with self._lock:
            if self._sealed or part_id not in self._parts or not delta:
                return
            self._parts[part_id] += delta
            self._process()

    def _joined(self) -> str:
        return "".join(self._parts[p] for p in self._order)

    def _process(self) -> None:
        joined = self._joined()
        if not joined.startswith(self._consumed):
            i = 0
            for i, (a, b) in enumerate(zip(self._consumed, joined)):
                if a != b:
                    break
            else:
                i = min(len(self._consumed), len(joined))
            self._consumed = joined[:i]
        self._pending += joined[len(self._consumed):]
        self._consumed = joined
        while True:
            m = _SENTENCE_END.match(self._pending)
            if not m:
                break
            sentence = m.group(0)
            self._pending = self._pending[len(sentence):]
            self._emit(sentence)

    def _emit(self, raw: str) -> None:
        text = clean_for_speech(raw)
        if text and self.on_sentence is not None:
            self.on_sentence(text)

    def flush(self) -> None:
        """Emit the trailing fragment (end of the reply)."""
        with self._lock:
            self._process()
            if self._pending.strip():
                self._emit(self._pending)
            self._pending = ""

    @property
    def text(self) -> str:
        return clean_for_speech(self._joined())


def split_for_speech(text: str, max_len: int = 180, first_max: int = 90) -> list[str]:
    """Split a reply into speakable chunks so TTS can start on the first
    sentence instead of waiting for the whole thing.

    The very first chunk is kept short (``first_max``) by cutting at its first
    clause, so audio starts as soon as possible; later sentences are grouped up
    to ``max_len`` and long ones are broken at clause punctuation."""
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks: list[str] = []
    for sentence in _SENTENCE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        parts = re.split(r"(?<=[;:,])\s+", sentence)
        # Cut a short opening chunk off the first (possibly long) sentence.
        if not chunks and len(sentence) > first_max and len(parts) > 1:
            chunks.append(parts.pop(0))
        buf = ""
        for part in parts:
            candidate = (buf + " " + part).strip()
            if buf and len(candidate) > max_len:
                chunks.append(buf)
                buf = part
            else:
                buf = candidate
        if buf:
            chunks.append(buf)
    return chunks or [text]


class TTSClient:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled", True))
        self.endpoint = (self.cfg.get("endpoint") or "").rstrip("/")
        self.voice = self.cfg.get("voice", "af_heart")
        self.speed = float(self.cfg.get("speed", 1.0))
        self.autostart = bool(self.cfg.get("autostart", True))
        self.sidecar_run = self.cfg.get("sidecar_run") or ""
        self._proc: subprocess.Popen | None = None

    def healthy(self) -> bool:
        if not self.endpoint:
            return False
        try:
            r = requests.get(self.endpoint + "/health", timeout=2)
            return r.status_code == 200 and r.json().get("ok", False)
        except Exception:
            return False

    def ensure_running(self, wait_s: float = 60.0, grace_s: float = 8.0) -> bool:
        if self.healthy():
            return True
        # Give a systemd-managed sidecar (seeker-tts.service) a moment to come up
        # before we spawn a second copy and race it for the port.
        grace_deadline = time.time() + grace_s
        while time.time() < grace_deadline:
            if self.healthy():
                return True
            time.sleep(0.5)
        if not (self.autostart and self.sidecar_run and os.path.exists(self.sidecar_run)):
            print("[tts] sidecar unavailable and autostart disabled", file=sys.stderr)
            return False
        print(f"[tts] starting sidecar: {self.sidecar_run}", file=sys.stderr, flush=True)
        self._proc = subprocess.Popen(
            [self.sidecar_run],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.healthy():
                print("[tts] sidecar ready", file=sys.stderr, flush=True)
                return True
            time.sleep(1.0)
        return False

    def synthesize(self, text: str, _retry: bool = True) -> bytes | None:
        if not (self.enabled and self.endpoint and text.strip()):
            return None
        payload = {"text": text, "voice": self.voice, "speed": self.speed, "format": "wav"}
        try:
            # Post directly rather than gating on a separate /health call, so a
            # transient health blip can never silently drop the speech.
            r = requests.post(self.endpoint + "/tts", json=payload, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as exc:
            if _retry:
                print(f"[tts] synth failed ({exc}); ensuring sidecar, retrying",
                      file=sys.stderr)
                if self.ensure_running():
                    return self.synthesize(text, _retry=False)
            print(f"[tts] synthesis failed: {exc}", file=sys.stderr)
            return None
