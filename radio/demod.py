import numpy as np
from scipy.signal import firwin, lfilter, resample_poly, butter, sosfilt
from config import SAMPLE_RATE, AUDIO_RATE, CHANNEL_BW, SQUELCH
from audio.player import audio_queue
from transcription.vosk_engine import transcribe_audio

# filters built once at startup
channel_filter = firwin(128, cutoff=CHANNEL_BW / SAMPLE_RATE)
bandpass = butter(1, [300 / (AUDIO_RATE / 2), 3400 / (AUDIO_RATE / 2)], btype='band', output='sos')

# accumulation buffer for transcription
_audio_buffer = []
_BUFFER_FLUSH_SIZE = 10  # flush to transcription every N chunks


def process_iq(iq):
    global _audio_buffer

    # filter IQ to channel bandwidth
    I = lfilter(channel_filter, 1.0, iq.real)
    Q = lfilter(channel_filter, 1.0, iq.imag)
    iq_filtered = I + 1j * Q

    # FM demod
    conj = iq_filtered[:-1] * np.conj(iq_filtered[1:])
    demodulated = np.angle(conj)

    # resample to audio rate
    audio = resample_poly(demodulated, 1, int(SAMPLE_RATE / AUDIO_RATE))

    # squelch gate
    power = np.sqrt(np.mean(audio ** 2))
    if power < SQUELCH:
        # flush accumulated buffer to transcription on silence
        if _audio_buffer:
            full_audio = np.concatenate(_audio_buffer)
            transcribe_audio(full_audio)
            _audio_buffer = []
        return

    # bandpass filter
    audio = sosfilt(bandpass, audio)

    # normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # send to audio playback
    audio_queue.put(audio.astype(np.float32))

    # accumulate for transcription
    _audio_buffer.append(audio)