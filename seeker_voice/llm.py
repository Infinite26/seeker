"""OpenAI-compatible LLM client with an ordered fallback chain."""
from __future__ import annotations

import json
import sys
from datetime import datetime

import requests

from .tts import SentenceEmitter


class LLMError(RuntimeError):
    pass


def _messages_for(command: str, system_prompt: str,
                  history: list[dict] | None) -> list[dict]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Current local time: {now}."},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": command})
    return messages


def _chat_stream(base_url: str, api_key: str, model: str, messages: list[dict],
                 max_tokens: int, temperature: float, timeout_s: float,
                 extra_headers: dict | None, on_sentence) -> str:
    """Stream a chat completion, emitting each sentence as soon as it completes.

    Returns the full reply text.  Falls back to plain buffering if the endpoint
    cannot stream."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    resp = requests.post(url, json=body, headers=headers, timeout=timeout_s,
                         stream=True)
    resp.raise_for_status()
    resp.encoding = "utf-8"   # text/event-stream has no charset; never Latin-1
    emitter = SentenceEmitter(on_sentence)
    content: list[str] = []
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            delta = ((payload.get("choices") or [{}])[0].get("delta") or {})
            piece = delta.get("content") or ""
            if piece:
                content.append(piece)
                emitter.feed("reply", "".join(content))
    except Exception:
        if not content:
            raise
        # half a reply is better than falling back and repeating it
    finally:
        try:
            resp.close()
        except Exception:
            pass
    emitter.flush()
    text = "".join(content).strip()
    if not text:
        raise LLMError(f"{base_url}: empty response")
    return text


class LLMClient:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.system_prompt = self.cfg.get("system_prompt", "You are Seeker.")
        self.max_tokens = int(self.cfg.get("max_tokens", 200))
        self.temperature = float(self.cfg.get("temperature", 0.6))
        self.timeout_s = float(self.cfg.get("timeout_s", 30))
        self.enabled = bool(self.cfg.get("enabled", True))

        global_headers = dict(self.cfg.get("extra_headers", {}) or {})
        self.endpoints: list[dict] = []
        primary = {
            "base_url": self.cfg.get("base_url", ""),
            "api_key": self.cfg.get("api_key", ""),
            "model": self.cfg.get("model", ""),
        }
        if primary["base_url"] and primary["model"]:
            self.endpoints.append(primary)
        for fb in self.cfg.get("fallbacks", []) or []:
            if fb.get("base_url") and fb.get("model"):
                self.endpoints.append(fb)
        for ep in self.endpoints:
            merged = dict(global_headers)
            merged.update(ep.get("extra_headers", {}) or {})
            ep["extra_headers"] = merged

    def stream_reply(self, command: str, history: list[dict] | None = None,
                     on_sentence=None) -> str:
        """Reply to `command`, calling on_sentence with each spoken sentence as
        soon as it is complete (so speech starts while the model still writes)."""
        if not self.enabled or not self.endpoints:
            raise LLMError("LLM disabled or no endpoints configured")

        messages = _messages_for(command, self.system_prompt, history)
        errors: list[str] = []
        for ep in self.endpoints:
            try:
                text = _chat_stream(
                    ep["base_url"], ep.get("api_key", ""), ep["model"],
                    messages, self.max_tokens, self.temperature, self.timeout_s,
                    ep.get("extra_headers"), on_sentence,
                )
                return text
            except Exception as exc:  # try next endpoint
                errors.append(f"{ep['base_url']}: {exc}")
                print(f"[llm] {ep['base_url']} failed: {exc}", file=sys.stderr)
        raise LLMError("; ".join(errors))

    def reply(self, command: str, history: list[dict] | None = None) -> str:
        return self.stream_reply(command, history=history)
