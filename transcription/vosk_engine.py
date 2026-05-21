from vosk import Model, KaldiRecognizer
from scipy.signal import resample_poly
from config import AUDIO_RATE
from dispatch.preamble import check_preamble
import json
import numpy as np
# vosk requires 16kHz — resample from 48kHz
VOSK_RATE = 16000

model = Model(lang="en-us")
recognizer = KaldiRecognizer(model, VOSK_RATE)
recognizer.SetWords(True)


def transcribe_audio(audio_48k: np.ndarray):
    audio_16k = resample_poly(audio_48k, VOSK_RATE, AUDIO_RATE)
    pcm = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    recognizer.AcceptWaveform(pcm)
    final = json.loads(recognizer.FinalResult())
    recognizer.Reset()
    text = final.get("text", "").strip()
    if text:
        print(f"\n[FINAL] {text}")
        check_preamble(text)