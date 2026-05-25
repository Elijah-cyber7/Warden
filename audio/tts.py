"""
Text-to-Speech for Warden dispatch responses.

Uses Piper TTS for local synthesis, then routes output to
radio TX, laptop speakers, or both.
"""

import io
import logging
import threading
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice
from scipy.signal import resample_poly

from config import AUDIO_RATE, PIPER_VOICE, PIPER_VOICES_DIR, TTS_OUTPUT
from audio.player import audio_queue

log = logging.getLogger("warden.tts")

_voice: PiperVoice | None = None
_radio_controller = None
_tx_lock = threading.Lock()


def set_radio_controller(controller):
    """Register the radio controller (called once from main.py)."""
    global _radio_controller
    _radio_controller = controller


def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        model_path = PIPER_VOICES_DIR / f"{PIPER_VOICE}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {model_path}. "
                f"Run: python3 -m piper.download_voices {PIPER_VOICE} --download-dir {PIPER_VOICES_DIR}"
            )
        _voice = PiperVoice.load(str(model_path))
        log.info("Loaded voice: %s", PIPER_VOICE)
    return _voice


def synthesize_speech(text: str) -> np.ndarray:
    """Synthesize speech locally via Piper. Returns float32 audio at AUDIO_RATE."""
    voice = _get_voice()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    buf.seek(0)

    with wave.open(buf, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if sample_rate != AUDIO_RATE:
        audio = resample_poly(audio, AUDIO_RATE, sample_rate).astype(np.float32)

    return np.clip(audio, -1.0, 1.0)


def speak(text: str):
    """Synthesize speech and route to radio TX and/or laptop speakers."""
    log.info("Speaking: %s", text)
    audio = synthesize_speech(text)

    if TTS_OUTPUT in ("transmit", "both"):
        if _radio_controller is None:
            log.warning("No radio controller — cannot transmit")
        else:
            with _tx_lock:
                _radio_controller.transmit(audio)

    if TTS_OUTPUT in ("speakers", "both"):
        audio_queue.put(audio)
