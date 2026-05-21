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
    print(f"[TRANSCRIBE] samples={len(audio_48k)} duration={len(audio_48k) / AUDIO_RATE:.2f}s")
    audio_16k = resample_poly(audio_48k, VOSK_RATE, AUDIO_RATE)
    pcm = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    accepted = recognizer.AcceptWaveform(pcm)
    print(f"[VOSK] accepted={accepted}")
    if accepted:
        result = json.loads(recognizer.Result())
        text = result.get("text", "").strip()
        if text:
            print(f"\n[FINAL] {text}")
            check_preamble(text)
    else:
        partial = json.loads(recognizer.PartialResult())
        p = partial.get("partial", "").strip()
        if p:
            print(f"\r[LIVE] {p}    ", end="", flush=True)

    # always drain final result after buffer flush
    final = json.loads(recognizer.FinalResult())
    print(f"[FINAL RESULT] {final}")
    text = final.get("text", "").strip()
    if text:
        print(f"\n[FINAL] {text}")
        check_preamble(text)