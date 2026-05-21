import queue
import sounddevice as sd
from config import AUDIO_RATE

audio_queue = queue.Queue()


def audio_worker():
    while True:
        try:
            with sd.OutputStream(samplerate=AUDIO_RATE, channels=1, dtype='float32') as stream:
                while True:
                    audio = audio_queue.get()
                    if audio is None:
                        return
                    stream.write(audio.reshape(-1, 1))
        except sd.PortAudioError as e:
            print(f"[AUDIO] stream error: {e} — restarting")