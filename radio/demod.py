import numpy as np
from scipy.signal import lfilter, lfilter_zi, resample_poly, butter, sosfilt, sosfilt_zi, firwin, iirnotch
from config import SAMPLE_RATE, AUDIO_RATE, CHANNEL_BW, SQUELCH
from audio.player import audio_queue
from transcription.vosk_engine import transcribe_audio
import scipy.io.wavfile as wav

DECIMATE_1 = 50
INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATE_1

# channel filter with state
_ch_taps = firwin(128, CHANNEL_BW / 2 / (INTERMEDIATE_RATE / 2))
_ch_zi_I = lfilter_zi(_ch_taps, 1.0)
_ch_zi_Q = lfilter_zi(_ch_taps, 1.0)

# CTCSS notch with state
_notch_b, _notch_a = iirnotch(123.7 / (AUDIO_RATE / 2), Q=5)
_notch_zi = lfilter_zi(_notch_b, _notch_a)

# voice bandpass with state
_vp_sos = butter(6, [300 / (AUDIO_RATE / 2), 4000 / (AUDIO_RATE / 2)], btype='band', output='sos')
_vp_zi = sosfilt_zi(_vp_sos)

_audio_buffer = []
_last_sample = np.complex64(1 + 0j)


def process_iq(iq):
    global _audio_buffer, _vp_zi, _notch_zi, _ch_zi_I, _ch_zi_Q, _last_sample

    iq_power = np.mean(np.abs(iq) ** 2)
    #print(f"iq_power: {iq_power:.6f}")
    if iq_power < SQUELCH:
        if _audio_buffer:
            full_audio = np.concatenate(_audio_buffer)
            wav.write('debug.wav', AUDIO_RATE, (full_audio * 32767).astype(np.int16))
            transcribe_audio(full_audio)
            _audio_buffer = []
        return

    # decimate
    iq_d = iq[::DECIMATE_1]

    # channel filter with state
    I, _ch_zi_I = lfilter(_ch_taps, 1.0, iq_d.real, zi=_ch_zi_I)
    Q, _ch_zi_Q = lfilter(_ch_taps, 1.0, iq_d.imag, zi=_ch_zi_Q)
    iq_filtered = (I + 1j * Q).astype(np.complex64)

    # FM demod with IQ context
    iq_ext = np.concatenate(([_last_sample], iq_filtered))
    _last_sample = iq_filtered[-1]
    conj = iq_ext[:-1] * np.conj(iq_ext[1:])
    demodulated = np.angle(conj).astype(np.float32)

    # resample
    audio = resample_poly(demodulated, int(AUDIO_RATE), INTERMEDIATE_RATE)

    # CTCSS notch with state
    audio, _notch_zi = lfilter(_notch_b, _notch_a, audio, zi=_notch_zi)

    # voice bandpass with state
    audio, _vp_zi = sosfilt(_vp_sos, audio, zi=_vp_zi)

    audio = (audio * 5.0).astype(np.float32)

    audio_queue.put(audio)
    _audio_buffer.append(audio)