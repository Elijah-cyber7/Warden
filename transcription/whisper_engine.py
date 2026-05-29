"""
Whisper transcription engine for Warden.

Uses MLX Whisper (Apple Silicon optimized) for real-time speech-to-text.
"""

import logging
import time

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


def preload_model():
    """
    Pre-load Whisper weights and warm the MLX compile cache.

    mlx_whisper caches the loaded model in a module-level ModelHolder, so a
    single transcribe() call with a tiny silence buffer is enough to (a) pull
    the weights from disk/HF into memory and (b) trigger the @mx.compile JIT
    for the decode kernels. Subsequent real transcribes skip both costs.

    Safe to call from a background thread on startup.
    """
    log.info("Preloading Whisper model: %s", WHISPER_MODEL)
    t0 = time.monotonic()
    try:
        silent = np.zeros(WHISPER_RATE, dtype=np.float32)
        mlx_whisper.transcribe(
            silent,
            path_or_hf_repo=WHISPER_MODEL,
            language="en",
            no_speech_threshold=1.0,
            condition_on_previous_text=False,
        )
        log.info("Whisper ready in %.1fs", time.monotonic() - t0)
    except Exception as e:
        log.error("Whisper preload failed (%.1fs in): %s",
                  time.monotonic() - t0, e)


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
