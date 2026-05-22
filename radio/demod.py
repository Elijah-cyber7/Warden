import numpy as np
from scipy.signal import lfilter, lfilter_zi, resample_poly, butter, sosfilt, sosfilt_zi, firwin, iirnotch, tf2sos, decimate
from config import SAMPLE_RATE, AUDIO_RATE, CHANNEL_BW, SQUELCH
from audio.player import audio_queue
from transcription.vosk_engine import transcribe_audio
import scipy.io.wavfile as wav

DECIMATE_1 = 50
INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATE_1

# channel filter at FULL sample rate (before decimation)
_ch_taps = firwin(512, CHANNEL_BW / (SAMPLE_RATE / 2))
_ch_zi_I = lfilter_zi(_ch_taps, 1.0) * 0
_ch_zi_Q = lfilter_zi(_ch_taps, 1.0) * 0

# CTCSS notch — convert to SOS for numerical stability
_notch_b, _notch_a = iirnotch(123.7 / (AUDIO_RATE / 2), Q=10)
_notch_sos = tf2sos(_notch_b, _notch_a)
_notch_zi = sosfilt_zi(_notch_sos) * 0

# de-emphasis filter (750µs time constant for land mobile radio)
_deemph_tau = 750e-6
_deemph_alpha = 1.0 / (1.0 + (INTERMEDIATE_RATE * _deemph_tau))
_deemph_b = np.array([_deemph_alpha])
_deemph_a = np.array([1.0, -(1.0 - _deemph_alpha)])
_deemph_zi = lfilter_zi(_deemph_b, _deemph_a) * 0

# voice bandpass with state
_vp_sos = butter(2, [300 / (AUDIO_RATE / 2), 3200 / (AUDIO_RATE / 2)], btype='band', output='sos')
_vp_zi = sosfilt_zi(_vp_sos) * 0

_audio_buffer = []
_last_sample = np.complex64(1 + 0j)

# soft squelch state
_squelch_gain = 0.0
_squelch_attack = 0.05
_squelch_release = 0.002


def _flush_buffer():
    global _audio_buffer
    if not _audio_buffer:
        return
    full_audio = np.concatenate(_audio_buffer)
    _audio_buffer = []
    wav.write('debug.wav', AUDIO_RATE, (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16))
    transcribe_audio(full_audio)


def process_iq(iq):
    global _audio_buffer, _vp_zi, _notch_zi, _ch_zi_I, _ch_zi_Q, _last_sample, _deemph_zi, _squelch_gain

    # 1. Channel filter at FULL sample rate (before decimation to avoid aliasing)
    I, _ch_zi_I = lfilter(_ch_taps, 1.0, iq.real, zi=_ch_zi_I)
    Q, _ch_zi_Q = lfilter(_ch_taps, 1.0, iq.imag, zi=_ch_zi_Q)
    iq_filtered = (I + 1j * Q).astype(np.complex64)

    # 2. Decimate AFTER filtering
    iq_d = iq_filtered[::DECIMATE_1]

    # 3. Measure channel power for squelch
    channel_power = np.mean(np.abs(iq_d) ** 2)
    squelch_open = channel_power >= SQUELCH

    # soft squelch: ramp gain up/down to avoid clicks
    if squelch_open:
        _squelch_gain = min(1.0, _squelch_gain + _squelch_attack)
    else:
        _squelch_gain = max(0.0, _squelch_gain - _squelch_release)

    # if fully closed and buffer exists, flush it
    if _squelch_gain < 0.001:
        if _audio_buffer:
            _flush_buffer()
        return

    # 4. FM demod with correct phase difference direction
    iq_ext = np.concatenate(([_last_sample], iq_d))
    _last_sample = iq_d[-1]
    # correct order: current * conj(previous) gives positive frequency for positive deviation
    phase_diff = iq_ext[1:] * np.conj(iq_ext[:-1])
    demodulated = np.angle(phase_diff).astype(np.float32)

    # 5. De-emphasis filter (750µs for land mobile radio)
    demodulated, _deemph_zi = lfilter(_deemph_b, _deemph_a, demodulated, zi=_deemph_zi)

    # 6. Resample to audio rate
    audio = resample_poly(demodulated, int(AUDIO_RATE), INTERMEDIATE_RATE)

    # 7. CTCSS notch (SOS, stable)
    audio, _notch_zi = sosfilt(_notch_sos, audio, zi=_notch_zi)

    # 8. Voice bandpass with state
    audio, _vp_zi = sosfilt(_vp_sos, audio, zi=_vp_zi)

    # 9. Gentle normalization with fixed gain (not per-block RMS)
    # FM deviation for NBFM is ~2.5kHz, which gives max phase change of ~0.98 rad/sample
    # Scale to reasonable audio level
    audio = audio * 0.5

    # apply soft squelch gain
    audio = (audio * _squelch_gain).astype(np.float32)

    # clip to valid range
    audio = np.clip(audio, -1.0, 1.0)

    audio_queue.put(audio)
    _audio_buffer.append(audio)