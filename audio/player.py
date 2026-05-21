import queue
import sounddevice as sd
from config import AUDIO_RATE

audio_queue = queue.Queue()


def audio_worker():
    with sd.OutputStream(samplerate=AUDIO_RATE, channels=1, dtype='float32',blocksize=2500) as stream:
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            stream.write(audio.reshape(-1, 1))