import numpy as np
from scipy.signal import lfilter, lfilter_zi, resample_poly, sosfilt, sosfilt_zi, firwin, iirnotch
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

# voice bandpass: FIR highpass (300 Hz) + FIR lowpass (4000 Hz) cascade
_hp_taps = firwin(801, 400, fs=AUDIO_RATE, pass_zero=False)
_lp_taps = firwin(401, 3400, fs=AUDIO_RATE, pass_zero=True)
_hp_zi = lfilter_zi(_hp_taps, 1.0)
_lp_zi = lfilter_zi(_lp_taps, 1.0)

_audio_buffer = []
_last_sample = np.complex64(1 + 0j)


def process_iq(iq):
    global _audio_buffer, _hp_zi, _lp_zi, _ch_zi_I, _ch_zi_Q, _last_sample

    iq_power = np.mean(np.abs(iq) ** 2)
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

    # FIR highpass (300 Hz) + FIR lowpass (4000 Hz) cascade
    audio, _hp_zi = lfilter(_hp_taps, 1.0, audio, zi=_hp_zi)
    audio, _lp_zi = lfilter(_lp_taps, 1.0, audio, zi=_lp_zi)

    audio = (audio * 5.0).astype(np.float32)

    audio_queue.put(audio)
    _audio_buffer.append(audio)