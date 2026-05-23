import mlx_whisper
import numpy as np
from scipy.signal import resample_poly
from config import AUDIO_RATE
from dispatch.preamble import check_preamble

WHISPER_RATE = 16000
MODEL_PATH = "mlx-community/whisper-large-v3-turbo"


def transcribe_audio(audio_48k: np.ndarray):
    duration = len(audio_48k) / AUDIO_RATE
    print(f"[TRANSCRIBE] samples={len(audio_48k)} duration={duration:.2f}s")

    if duration < 0.5:
        print("[WHISPER] Audio too short, skipping")
        return

    audio_16k = resample_poly(audio_48k, WHISPER_RATE, AUDIO_RATE).astype(np.float32)
    audio_16k = np.clip(audio_16k, -1.0, 1.0)

    result = mlx_whisper.transcribe(
        audio_16k,
        path_or_hf_repo=MODEL_PATH,
        language="en",
    )

    segments = result.get("segments", [])
    if segments:
        avg_no_speech = np.mean([s.get("no_speech_prob", 0) for s in segments])
        if avg_no_speech > 0.6:
            print(f"[WHISPER] No speech (no_speech_prob={avg_no_speech:.2f})")
            return

    text = result.get("text", "").strip()
    if text:
        print(f"\n[WHISPER] {text}")
        check_preamble(text)
    else:
        print("[WHISPER] No speech detected")