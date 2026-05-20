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
    # resample to vosk rate
    audio_16k = resample_poly(audio_48k, VOSK_RATE, AUDIO_RATE)

    # convert to 16-bit PCM
    pcm = (audio_16k * 32767).astype(np.int16).tobytes()
    print(f"Accepted: {recognizer.AcceptWaveform(pcm)}")

    if recognizer.AcceptWaveform(pcm):
        # full utterance complete
        result = json.loads(recognizer.Result())
        text = result.get("text", "").strip()
        if text:
            print(f"\n[FINAL] {text}")
            check_preamble(text)
    else:
        # partial — print in place without newline so it updates on same line
        partial = json.loads(recognizer.PartialResult())
        p = partial.get("partial", "").strip()
        if p:
            print(f"\r[LIVE] {p}    ", end="", flush=True)