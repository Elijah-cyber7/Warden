"""Audio playback worker for Warden."""

import queue

import numpy as np
import sounddevice as sd
from config import AUDIO_RATE

audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()


def audio_worker():
    """Blocking worker that plays audio chunks from the queue."""
    with sd.OutputStream(samplerate=AUDIO_RATE, channels=1, dtype='float32', blocksize=0) as stream:
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            stream.write(audio.reshape(-1, 1))
