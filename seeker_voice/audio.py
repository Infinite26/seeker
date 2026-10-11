"""Microphone capture, energy-based VAD segmentation, and playback."""
from __future__ import annotations

import io
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - import error surfaced at runtime
    sd = None


def list_devices() -> str:
    if sd is None:
        return "sounddevice is not installed"
    return str(sd.query_devices())


def _run(cmd: list[str]):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    except Exception:
        return None


def pipewire_sources() -> list[str]:
    r = _run(["pactl", "list", "short", "sources"])
    if not r or r.returncode != 0:
        return []
    sources = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            sources.append(parts[1])
    return sources


def _default_source() -> str:
    r = _run(["pactl", "get-default-source"])
    return r.stdout.strip() if r and r.returncode == 0 else ""


def ensure_input_source(timeout_s: float = 120.0, log=None) -> str:
    """Return a real capture source (never a `.monitor`) and make it the default.

    WirePlumber occasionally loses the USB mic (especially around boot), leaving
    a monitor as the default source — which would make Seeker hear its own output
    or fail to open the mic.  If no real source exists, nudge WirePlumber once and
    wait for it to reappear."""
    if log is None:
        log = lambda m: print(m, file=sys.stderr, flush=True)
    deadline = time.time() + timeout_s
    nudged = False
    while time.time() < deadline:
        sources = pipewire_sources()
        real = [s for s in sources if ".monitor" not in s]
        preferred = [s for s in real if s.startswith("alsa_input.")] or real
        if preferred:
            src = preferred[0]
            if _default_source() != src:
                if _run(["pactl", "set-default-source", src]):
                    log(f"[audio] default capture source -> {src}")
            return src
        if not nudged and shutil.which("systemctl"):
            log("[audio] no capture source found; restarting wireplumber to re-detect the mic")
            _run(["systemctl", "--user", "restart", "wireplumber"])
            nudged = True
            time.sleep(4)
        else:
            time.sleep(2)
    return ""


class MicStream:
    """Open the default (or configured) input device and yield mono float32 frames."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, device=None):
        if sd is None:
            raise RuntimeError("sounddevice is not installed; run ./setup.sh")
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.device = device
        self._stream = None

    def open(self) -> "MicStream":
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            device=self.device,
        )
        self._stream.start()
        return self

    def __enter__(self) -> "MicStream":
        return self.open()

    def read(self) -> np.ndarray:
        data, overflowed = self._stream.read(self.frame_samples)
        if overflowed:
            print("[audio] input overflow", file=sys.stderr)
        return np.asarray(data, dtype=np.float32).reshape(-1)

    def flush(self) -> None:
        """Discard any buffered input (used after speaking to drop our own voice)."""
        if self._stream is None:
            return
        try:
            available = int(getattr(self._stream, "read_available", 0) or 0)
            if available > 0:
                self._stream.read(available)
        except Exception:
            pass

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def __exit__(self, *exc):
        self.close()


class UtteranceSegmenter:
    """Adaptive energy gate that turns a stream of frames into speech utterances.

    feed(frame) returns a complete float32 utterance array when speech ends,
    otherwise None. Short blips are discarded.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int,
        silence_ms: int = 1500,
        min_speech_ms: int = 250,
        preroll_ms: int = 300,
        max_utterance_s: float = 30.0,
        energy_ratio: float = 2.5,
        noise_floor_alpha: float = 0.95,
        min_rms: float = 0.006,
        silence_max_ms: int = 1500,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_frames = max(1, int(silence_ms / frame_ms))
        self.silence_max_frames = max(self.silence_frames, int(silence_max_ms / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.max_frames = max(1, int(max_utterance_s * 1000 / frame_ms))
        self.energy_ratio = energy_ratio
        self.alpha = noise_floor_alpha
        self.min_rms = min_rms

        self._preroll = deque(maxlen=max(1, int(preroll_ms / frame_ms)))
        # Noise floor is an EMA of non-speech frames only, seeded at min_rms so
        # an utterance that starts on the very first frame still triggers.
        # Raise audio.min_rms (see --calibrate) in a noisy room.
        self._floor = min_rms
        self._threshold = max(min_rms * energy_ratio, min_rms)
        self._start_hits = 0
        self._start_needed = 2
        self._speaking = False
        self._utt: list[np.ndarray] = []
        self._silence = 0
        self._speech = 0

    @staticmethod
    def frame_rms(frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        return float(math.sqrt(float(np.mean(np.square(frame, dtype=np.float64))) + 1e-12))

    def _rms(self, frame: np.ndarray) -> float:
        return self.frame_rms(frame)

    def barge_threshold(self, ratio: float = 6.0, floor_min: float = 0.05) -> float:
        """Loud threshold used to detect the user talking over our own TTS."""
        return max(self._floor * ratio, floor_min)

    def _silence_needed(self) -> int:
        """Trailing silence (frames) required to close the utterance.

        Short commands end as soon as `silence_ms` passes (snappy), but the
        longer someone has been talking, the more slack they get — a pause while
        thinking mid-question must not split the question in half."""
        speech_s = self._speech * self.frame_ms / 1000.0
        extra_ms = max(0.0, speech_s - 1.0) * 350.0   # +350ms per extra second
        frames = int((self.silence_frames * self.frame_ms + extra_ms) / self.frame_ms)
        return min(max(frames, self.silence_frames), self.silence_max_frames)

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        rms = self._rms(frame)

        if not self._speaking:
            if rms <= self._threshold:
                # non-speech: adapt the floor toward the ambient level
                self._floor += (1.0 - self.alpha) * (rms - self._floor)
                self._threshold = max(self._floor * self.energy_ratio, self.min_rms)
                self._start_hits = 0
                self._preroll.append(frame)
                return None
            self._start_hits += 1
            self._preroll.append(frame)
            if self._start_hits < self._start_needed:
                return None
            # speech confirmed: seed the utterance with the pre-roll
            self._speaking = True
            self._speech = self._start_hits
            self._silence = 0
            self._start_hits = 0
            self._utt = list(self._preroll)
            self._preroll.clear()
            return None

        # currently speaking: threshold is frozen so a long utterance cannot
        # cut itself off mid-sentence
        threshold = self._threshold
        self._utt.append(frame)
        if rms < threshold:
            self._silence += 1
        else:
            self._silence = 0
            self._speech += 1

        if self._silence >= self._silence_needed() or len(self._utt) >= self.max_frames:
            utterance = np.concatenate(self._utt) if self._utt else np.zeros(0, np.float32)
            self._speaking = False
            self._utt = []
            ended_ok = self._speech >= self.min_speech_frames
            self._speech = 0
            self._silence = 0
            if ended_ok and utterance.size:
                return utterance
        return None

    def reset(self) -> None:
        """Drop buffered audio but keep the learned noise floor."""
        self._preroll.clear()
        self._utt = []
        self._speaking = False
        self._silence = 0
        self._speech = 0
        self._start_hits = 0

    @property
    def speaking(self) -> bool:
        """True while an utterance is being captured (speech in progress)."""
        return self._speaking


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _beep(freq_hz: float, ms: int, sample_rate: int, volume: float) -> np.ndarray:
    n = int(sample_rate * ms / 1000)
    t = np.linspace(0.0, ms / 1000.0, n, endpoint=False)
    sig = (np.sin(2 * np.pi * freq_hz * t) * volume).astype(np.float32)
    fade = min(n // 2, int(sample_rate * 0.012))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        sig[:fade] *= ramp
        sig[-fade:] *= ramp[::-1]
    return sig


def tone_wav(freq_hz: float = 880.0, ms: int = 120, sample_rate: int = 24000,
             volume: float = 0.35) -> bytes:
    """A short, click-free sine beep as WAV bytes (for audio feedback)."""
    return to_wav_bytes(_beep(freq_hz, ms, sample_rate, volume), sample_rate)


def chirp_wav(sample_rate: int = 24000, volume: float = 0.35) -> bytes:
    """Two rising beeps: 'I'm up and listening.'"""
    gap = np.zeros(int(sample_rate * 0.05), dtype=np.float32)
    sig = np.concatenate([
        _beep(660.0, 90, sample_rate, volume),
        gap,
        _beep(990.0, 110, sample_rate, volume),
    ])
    return to_wav_bytes(sig, sample_rate)


def from_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return pcm, sr


class Speaker:
    """Non-blocking, interruptible playback with a synthesis pipeline.

    Speech is queued as WAV chunks.  A producer synthesises chunk n+1 while the
    player is still speaking chunk n, so playback starts after the *first*
    sentence instead of after the whole reply.  The mic stays live throughout so
    barge-in can interrupt mid-sentence.  Routes to a PipeWire sink via pw-play.

    ``speak_queue()`` feeds the pipeline from an external queue of texts, so a
    reply can be streamed in sentence by sentence while it is still being
    generated.
    """

    def __init__(self, sink: Optional[str] = None, device=None):
        self.sink = sink
        self.device = device
        self._proc: Optional[subprocess.Popen] = None
        self._path: Optional[str] = None
        self._stop = threading.Event()
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=3)
        self._producer: Optional[threading.Thread] = None
        self._player: Optional[threading.Thread] = None
        self._sounding = False

    # -- public API -------------------------------------------------------
    def play(self, data: bytes) -> None:
        """Speak a single already-synthesised WAV (non-blocking)."""
        self._start(lambda: iter((data,)))

    def speak_texts(self, texts, synth) -> None:
        """Speak texts in order, synthesising each with synth(text) -> wav bytes,
        one sentence ahead of playback (non-blocking)."""
        def wavs():
            for text in texts:
                if self._stop.is_set():
                    return
                try:
                    wav = synth(text)
                except Exception as exc:
                    print(f"[audio] synthesis failed: {exc}", file=sys.stderr)
                    continue
                if wav:
                    yield wav
        self._start(wavs)

    def speak_queue(self, texts: "queue.Queue[Optional[str]]", synth) -> None:
        """Speak chunks pulled from `texts` as they arrive (None = end of speech).

        Synthesis runs one chunk ahead of playback, so the reply starts playing
        as soon as its first chunk exists while later chunks are still streaming
        in from the brain."""
        def wavs():
            while not self._stop.is_set():
                try:
                    text = texts.get(timeout=0.1)
                except queue.Empty:
                    continue
                if text is None:
                    return
                try:
                    wav = synth(text)
                except Exception as exc:
                    print(f"[audio] synthesis failed: {exc}", file=sys.stderr)
                    continue
                if wav:
                    yield wav
        self._start(wavs)

    def is_playing(self) -> bool:
        """True only while audio is actually coming out of the speaker.

        A queued-but-not-yet-synthesised turn counts as *not* playing, so the
        mic loop keeps segmenting (and can accept "stop") while the brain thinks."""
        return self._sounding and not self._stop.is_set()

    def wait(self) -> None:
        for th in (self._producer, self._player):
            if th is not None:
                try:
                    th.join()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop.set()
        self._sounding = False
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        for th in (self._producer, self._player):
            if th is not None:
                try:
                    th.join(timeout=1.0)
                except Exception:
                    pass
        self._producer = self._player = None
        self._drain_queue()
        self._cleanup_path()
        self._proc = None

    # -- internals --------------------------------------------------------
    def _start(self, wavs) -> None:
        self.stop()
        self._stop.clear()
        q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=3)
        self._q = q
        self._producer = threading.Thread(target=self._run_producer, args=(wavs, q),
                                          daemon=True)
        self._player = threading.Thread(target=self._run_player, args=(q,), daemon=True)
        self._producer.start()
        self._player.start()

    def _run_producer(self, wavs, q) -> None:
        try:
            for wav in wavs():
                if self._stop.is_set():
                    break
                q.put(wav)
        except Exception as exc:
            print(f"[audio] producer error: {exc}", file=sys.stderr)
        finally:
            try:
                q.put(None)  # sentinel: no more audio
            except Exception:
                pass

    def _run_player(self, q) -> None:
        try:
            while True:
                try:
                    wav = q.get()
                except Exception:
                    break
                if wav is None or self._stop.is_set():
                    break
                self._sounding = True
                self._play_chunk(wav)
        finally:
            self._sounding = False

    def _play_chunk(self, data: bytes) -> None:
        if self.sink:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(data)
                    self._path = tf.name
                self._proc = subprocess.Popen(
                    ["pw-play", "--target", self.sink, self._path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                while self._proc.poll() is None:
                    if self._stop.is_set():
                        self._proc.terminate()
                        break
                    time.sleep(0.02)
                try:
                    self._proc.wait(timeout=1)
                except Exception:
                    pass
                self._proc = None
                self._cleanup_path()
                return
            except Exception as exc:
                print(f"[audio] pw-play to {self.sink} failed ({exc}); using default",
                      file=sys.stderr)
                self._cleanup_path()
        if sd is None:
            return
        pcm, sr = from_wav_bytes(data)
        sd.play(pcm, sr, device=self.device)
        while True:
            if self._stop.is_set():
                sd.stop()
                break
            try:
                stream = sd.get_stream()
            except Exception:
                break
            if not (stream is not None and stream.active):
                break
            time.sleep(0.02)

    def _drain_queue(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass

    def _cleanup_path(self) -> None:
        if self._path:
            try:
                os.remove(self._path)
            except OSError:
                pass
            self._path = None


def play_wav_bytes(data: bytes, device=None, sink: str | None = None) -> None:
    """Play WAV bytes.  If `sink` is given, route to that PipeWire/Pulse sink via
    pw-play (useful when the default sink is HDMI); else use the sounddevice
    default output."""
    if sink:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(data)
                path = tf.name
            try:
                subprocess.run(["pw-play", "--target", sink, path], check=True)
                return
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        except Exception as exc:
            print(f"[audio] pw-play to {sink} failed ({exc}); using default", file=sys.stderr)

    if sd is None:
        raise RuntimeError("sounddevice is not installed; run ./setup.sh")
    pcm, sr = from_wav_bytes(data)
    sd.play(pcm, sr, device=device)
    sd.wait()
