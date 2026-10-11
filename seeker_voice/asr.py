"""ASR backends: faster-whisper (local, default) and Audio8 (remote, pluggable)."""
from __future__ import annotations

import io
import sys
import wave

import numpy as np

TARGET_SR = 16000


def _resample(pcm: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    if sr == target or pcm.size == 0:
        return pcm
    duration = pcm.size / float(sr)
    n_out = int(duration * target)
    src_x = np.linspace(0.0, duration, num=pcm.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(dst_x, src_x, pcm).astype(np.float32)


def _normalize(pcm: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Bring quiet speech into Whisper's sweet spot.

    Uses the 98th-percentile level rather than the absolute peak so a single
    transient (a knock, a lip smack) can't set the scale and leave the speech
    itself too quiet to transcribe, and caps the gain so ambient noise isn't
    blown up either."""
    if pcm.size == 0:
        return pcm
    level = float(np.percentile(np.abs(pcm), 98))
    if level < 1e-4:
        return pcm
    gain = min(max(target_peak / level, 0.5), 12.0)
    return (pcm * gain).astype(np.float32)


def _wav_bytes(pcm: np.ndarray, sr: int = TARGET_SR) -> bytes:
    pcm_i16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


class ASRBackend:
    name = "base"

    def warmup(self) -> None:
        """Load models / open connections ahead of the first utterance."""

    def transcribe(self, pcm: np.ndarray, sr: int = TARGET_SR) -> str:
        raise NotImplementedError


class FasterWhisperASR(ASRBackend):
    """Local CTranslate2 Whisper via faster-whisper. CPU int8 by default."""

    name = "faster-whisper"

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.model_id = self.cfg.get("model", "mobiuslabsgmbh/faster-whisper-large-v3-turbo")
        self.device = self.cfg.get("device", "cpu")
        self.compute_type = self.cfg.get("compute_type", "int8")
        self.cpu_threads = int(self.cfg.get("cpu_threads", 4))
        self.beam_size = int(self.cfg.get("beam_size", 1))
        self.beam_size_long = int(self.cfg.get("beam_size_long", max(2, self.beam_size)))
        self.long_s = float(self.cfg.get("long_s", 2.0))
        self.language = self.cfg.get("language")
        self.vad_filter = bool(self.cfg.get("vad_filter", False))
        # Style/name bias: keeps "Seeker" transcribed as "Seeker" (not "See,
        # Kurt") and normalises punctuation on unclear audio.
        self.initial_prompt = self.cfg.get(
            "initial_prompt",
            "A spoken conversation with a voice assistant named Seeker. Hey Seeker.")
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        print(f"[asr] loading {self.model_id} ({self.device}/{self.compute_type}, "
              f"{self.cpu_threads} threads)", file=sys.stderr, flush=True)
        self._model = WhisperModel(
            self.model_id,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

    def warmup(self) -> None:
        self._load()

    def transcribe(self, pcm: np.ndarray, sr: int = TARGET_SR) -> str:
        if pcm.size == 0:
            return ""
        self._load()
        audio = _normalize(_resample(np.asarray(pcm, dtype=np.float32), sr))
        # Long, rambling speech is where greedy decoding slips up; short
        # commands are already accurate and beam search would only add latency.
        beam = self.beam_size_long if audio.size > self.long_s * TARGET_SR else self.beam_size
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=beam,
            initial_prompt=self.initial_prompt or None,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


class Audio8ASR(ASRBackend):
    """Remote Audio8-ASR-Infinite backend (Edge0/Audio8-ASR-Infinite).

    The canonical Audio8 deployment is the docker compose stack from
    github.com/Edge0-AI/Audio8-ASR-Infinite (a vLLM realtime websocket on
    18191).  This backend talks to whatever OpenAI-compatible transcription
    endpoint the Audio8 service is fronted with, so the same voice loop can use
    Audio8 on a GPU box or the 24/7 docker service without code changes.
    """

    name = "audio8"

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.url = self.cfg.get("transcribe_url")
        self.model = self.cfg.get("model", "audio8-asr-infinite")
        self.language = self.cfg.get("language", "en")

    def transcribe(self, pcm: np.ndarray, sr: int = TARGET_SR) -> str:
        if pcm.size == 0:
            return ""
        if not self.url:
            raise RuntimeError("audio8 backend: transcribe_url is not configured")

        import requests

        audio = _normalize(_resample(np.asarray(pcm, dtype=np.float32), sr))
        files = {"file": ("utterance.wav", _wav_bytes(audio), "audio/wav")}
        data = {"model": self.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        resp = requests.post(self.url, files=files, data=data, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        return (payload.get("text") or "").strip()


def build(cfg: dict) -> ASRBackend:
    backend = (cfg or {}).get("backend", "faster-whisper").lower()
    if backend in ("faster-whisper", "faster_whisper", "whisper"):
        return FasterWhisperASR(cfg.get("faster_whisper", {}))
    if backend in ("audio8", "audio8-asr-infinite"):
        return Audio8ASR(cfg.get("audio8", {}))
    raise ValueError(f"unknown ASR backend: {backend!r}")
