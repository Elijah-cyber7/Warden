import numpy as np
from scipy.signal import lfilter, lfilter_zi, resample_poly, butter, sosfilt, sosfilt_zi, firwin, iirnotch, tf2sos
from config import SAMPLE_RATE, AUDIO_RATE, CHANNEL_BW, SQUELCH
from audio.player import audio_queue
from transcription.vosk_engine import transcribe_audio
import scipy.io.wavfile as wav

DECIMATE_1 = 50
INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATE_1

# channel filter with state
_ch_taps = firwin(256, CHANNEL_BW / 2 / (INTERMEDIATE_RATE / 2))
_ch_zi_I = lfilter_zi(_ch_taps, 1.0)
_ch_zi_Q = lfilter_zi(_ch_taps, 1.0)

# CTCSS notch — convert to SOS for numerical stability
_notch_b, _notch_a = iirnotch(123.7 / (AUDIO_RATE / 2), Q=10)
_notch_sos = tf2sos(_notch_b, _notch_a)
_notch_zi = sosfilt_zi(_notch_sos)

# voice bandpass with state
_vp_sos = butter(2, [300 / (AUDIO_RATE / 2), 3200 / (AUDIO_RATE / 2)], btype='band', output='sos')
_vp_zi = sosfilt_zi(_vp_sos)

_audio_buffer = []
_last_sample = np.complex64(1 + 0j)


def _flush_buffer():
    global _audio_buffer
    if not _audio_buffer:
        return
    full_audio = np.concatenate(_audio_buffer)
    _audio_buffer = []
    wav.write('debug.wav', AUDIO_RATE, (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16))
    transcribe_audio(full_audio)


def process_iq(iq):
    global _audio_buffer, _vp_zi, _notch_zi, _ch_zi_I, _ch_zi_Q, _last_sample

    # decimate first
    iq_d = iq[::DECIMATE_1]

    # channel filter with state
    I, _ch_zi_I = lfilter(_ch_taps, 1.0, iq_d.real, zi=_ch_zi_I)
    Q, _ch_zi_Q = lfilter(_ch_taps, 1.0, iq_d.imag, zi=_ch_zi_Q)
    iq_filtered = (I + 1j * Q).astype(np.complex64)

    # squelch on channel power, not wideband IQ
    channel_power = np.mean(np.abs(iq_filtered) ** 2)
    print(f"[POWER] {channel_power:.6f}")
    if channel_power < SQUELCH:
        _flush_buffer()
        return

    # FM demod with IQ context
    iq_ext = np.concatenate(([_last_sample], iq_filtered))
    _last_sample = iq_filtered[-1]
    conj = iq_ext[:-1] * np.conj(iq_ext[1:])
    demodulated = np.angle(conj).astype(np.float32)

    # resample to audio rate
    audio = resample_poly(demodulated, int(AUDIO_RATE), INTERMEDIATE_RATE)

    # CTCSS notch (SOS, stable)
    audio, _notch_zi = sosfilt(_notch_sos, audio, zi=_notch_zi)

    # voice bandpass with state
    audio, _vp_zi = sosfilt(_vp_sos, audio, zi=_vp_zi)

    # normalize to [-1, 1] before accumulation and playback
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    audio = np.clip((audio / rms) * 0.01, -1.0, 1.0).astype(np.float32)

    audio_queue.put(audio)
    _audio_buffer.append(audio)