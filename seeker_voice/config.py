"""Configuration loading for Seeker."""
from __future__ import annotations

import copy
import json
import os

DEFAULTS = {
    "name": "Seeker",
    "aliases": ["seeker"],
    "wake_reply": "Yes?",
    "leading_mishears": ["secret"],
    "conversation": {"enabled": True, "timeout_s": 30},
    "unrecognized_reply": "I didn't catch that.",
    "offline_reply": "I heard my name, but my brain is offline right now.",
    "feedback": {
        "enabled": True,
        "ack": True,
        "error": True,
        "startup": True,
        "ack_text": "Okay.",
        "error_text": "Sorry, my brain is not responding.",
        "startup_text": "I'm online and listening.",
        "work_fillers": ["Let me see.", "One moment.", "Looking into it."],
        "work_filler_after_s": 4.0,
        "max_speak_sentences": 3,
    },
    "status": {
        "enabled": True,
        "path": "/home/infinite/.cache/seeker/status.json",
    },
    "panel": {
        "enabled": True,
    },
    "audio": {
        "input_device": None,
        "output_device": None,
        "sample_rate": 16000,
        "frame_ms": 30,
        "silence_ms": 800,
        "silence_max_ms": 1500,
        "stitch_s": 3.0,
        "min_speech_ms": 250,
        "preroll_ms": 300,
        "max_utterance_s": 30,
        "energy_ratio": 2.5,
        "noise_floor_alpha": 0.95,
        "min_rms": 0.006,
    },
    "asr": {
        "backend": "faster-whisper",
        "faster_whisper": {
            "model": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": 4,
            "beam_size": 1,
            "beam_size_long": 2,
            "long_s": 2.0,
            "language": None,
            "vad_filter": False,
            "initial_prompt": "A spoken conversation with a voice assistant named Seeker. Hey Seeker.",
        },
        "audio8": {
            "transcribe_url": "http://127.0.0.1:18190/v1/audio/transcriptions",
            "model": "audio8-asr-infinite",
            "language": "en",
        },
    },
    "tts": {
        "enabled": True,
        "endpoint": "http://127.0.0.1:8770",
        "voice": "af_heart",
        "speed": 1.05,
        "autostart": True,
        "sidecar_run": "",
    },
    "llm": {
        "enabled": True,
        "base_url": "http://127.0.0.1:8317/zen/go/v1",
        "api_key": "sk-local",
        "model": "deepseek-v4.1-flash",
        "max_tokens": 800,
        "temperature": 0.6,
        "timeout_s": 30,
        "system_prompt": "You are Seeker, Infinite's voice assistant.",
        "fallbacks": [],
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path: str | None = None) -> dict:
    """Load config.json (deep-merged over defaults)."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    cfg = copy.deepcopy(DEFAULTS)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _merge(cfg, json.load(fh))
    return cfg
