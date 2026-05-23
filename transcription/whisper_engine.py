import whisper
import numpy as np
from config import AUDIO_RATE
from dispatch.preamble import check_preamble

# Whisper expects 16kHz audio
WHISPER_RATE = 16000

# Load model once at startup (base model - good balance of speed/accuracy)
print("[WHISPER] Loading model...")
model = whisper.load_model("base")
print("[WHISPER] Model loaded")


def transcribe_audio(audio_48k: np.ndarray):
    """Transcribe audio using OpenAI Whisper."""
    duration = len(audio_48k) / AUDIO_RATE
    print(f"[TRANSCRIBE] samples={len(audio_48k)} duration={duration:.2f}s")
    
    # Skip very short clips (< 0.5s) - likely just noise
    if duration < 0.5:
        print("[WHISPER] Audio too short, skipping")
        return
    
    # Whisper expects float32 audio at 16kHz
    # Resample from 48kHz to 16kHz
    from scipy.signal import resample_poly
    audio_16k = resample_poly(audio_48k, WHISPER_RATE, AUDIO_RATE).astype(np.float32)
    
    # Normalize to [-1, 1] range
    audio_16k = np.clip(audio_16k, -1.0, 1.0)
    
    # Transcribe with Whisper
    # fp16=False for CPU, set to True if using CUDA GPU
    result = model.transcribe(
        audio_16k,
        language="en",
        fp16=False,
        condition_on_previous_text=False,
    )
    
    text = result.get("text", "").strip()
    
    if text:
        print(f"\n[WHISPER] {text}")
        check_preamble(text)
    else:
        print("[WHISPER] No speech detected")