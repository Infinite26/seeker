"""Seeker — a local voice wake assistant named Seeker.

Pipeline: microphone -> VAD segmentation -> ASR -> wake-word ("seeker") ->
LLM reply -> Kokoro TTS -> speaker.
"""

__version__ = "0.1.0"
