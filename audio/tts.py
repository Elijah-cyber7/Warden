import io
import threading
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice
from scipy.signal import resample_poly

from config import AUDIO_RATE, PIPER_VOICE, PIPER_VOICES_DIR, TTS_OUTPUT
from audio.player import audio_queue

_voice = None
_tx_processor = None
_tx_lock = threading.Lock()


def set_tx_processor(processor):
    """Register the TX pipeline (called once from main.py)."""
    global _tx_processor
    _tx_processor = processor


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
        print(f"[TTS] Loaded Piper voice: {PIPER_VOICE}")
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
    print(f"[TTS] {text}")
    audio = synthesize_speech(text)

    if TTS_OUTPUT in ("transmit", "both"):
        if _tx_processor is None:
            print("[TTS] Warning: no TX processor registered — cannot transmit")
        else:
            with _tx_lock:
                _tx_processor.transmit(audio)

    if TTS_OUTPUT in ("speakers", "both"):
        audio_queue.put(audio)
