"""
Whisper transcription engine for Warden.

Uses MLX Whisper (Apple Silicon optimized) for real-time speech-to-text.
"""

import logging

import numpy as np
from scipy.signal import resample_poly

import mlx_whisper
from config import (
    AUDIO_RATE, WHISPER_MODEL, MIN_AUDIO_DURATION, NO_SPEECH_THRESHOLD,
    WHISPER_INITIAL_PROMPT,
)
from dispatch.preamble import check_preamble

log = logging.getLogger("warden.whisper")

WHISPER_RATE = 16000


def transcribe_audio(audio_48k: np.ndarray):
    """Transcribe audio and check for dispatch callsigns."""
    duration = len(audio_48k) / AUDIO_RATE

    if duration < MIN_AUDIO_DURATION:
        log.debug("Audio too short (%.2fs), skipping", duration)
        return

    audio_16k = resample_poly(audio_48k, WHISPER_RATE, AUDIO_RATE).astype(np.float32)
    audio_16k = np.clip(audio_16k, -1.0, 1.0)

    result = mlx_whisper.transcribe(
        audio_16k,
        path_or_hf_repo=WHISPER_MODEL,
        language="en",
        initial_prompt=WHISPER_INITIAL_PROMPT,
    )

    segments = result.get("segments", [])
    if segments:
        avg_no_speech = np.mean([s.get("no_speech_prob", 0) for s in segments])
        if avg_no_speech > NO_SPEECH_THRESHOLD:
            log.debug("No speech (prob=%.2f)", avg_no_speech)
            return

    text = result.get("text", "").strip()
    if text:
        log.info("Transcribed: %s", text)
        check_preamble(text)
    else:
        log.debug("No speech detected")
