import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly, butter, sosfilt, sosfilt_zi, iirnotch
from config import SAMPLE_RATE, AUDIO_RATE, CHANNEL_BW, SQUELCH, CTCSS_FREQ, CTCSS_THRESHOLD
from audio.player import audio_queue
from transcription.vosk_engine import transcribe_audio

# --- Cascade decimation ---
# 2 MHz → 200 kHz (decimate 10) → channel filter → 48 kHz (resample 6/25)
# Channel filter at 200 kHz: cutoff = 12500/200000 = 0.0625
# Compare to previous 12500/2400000 = 0.0052 — filter now actually works
INTERMEDIATE_RATE = 200000
DECIMATE_1 = 10  # 2 MHz → 200 kHz

# --- NFM channel filter with state ---
_ch_b = firwin(1024, cutoff=CHANNEL_BW / INTERMEDIATE_RATE, window=('kaiser', 8.0))
_ch_a = np.array([1.0])
_ch_zi_I = lfilter_zi(_ch_b, _ch_a)
_ch_zi_Q = lfilter_zi(_ch_b, _ch_a)

# --- De-emphasis 300 µs with state ---
_tau = 300e-6
_alpha = np.exp(-1.0 / (_tau * AUDIO_RATE))
_de_b = np.array([1.0 - _alpha])
_de_a = np.array([1.0, -_alpha])
_de_zi = lfilter_zi(_de_b, _de_a)

# --- Voice bandpass 300 Hz–3400 Hz with state ---
_bp_sos = butter(2, [300 / (AUDIO_RATE / 2), 3400 / (AUDIO_RATE / 2)], btype='band', output='sos')
_bp_zi = sosfilt_zi(_bp_sos)

# --- CTCSS notch 123 Hz with state ---
_notch_b, _notch_a = iirnotch(123.0 / (AUDIO_RATE / 2), Q=30)
_notch_zi = lfilter_zi(_notch_b, _notch_a)

# accumulation buffer
_audio_buffer = []


def _goertzel(samples: np.ndarray, target_freq: float, fs: float) -> float:
    N = len(samples)
    k = int(0.5 + N * target_freq / fs)
    omega = 2.0 * np.pi * k / N
    coeff = 2.0 * np.cos(omega)
    b = np.array([1.0, 0.0, 0.0])
    a = np.array([1.0, -coeff, 1.0])
    y = lfilter(b, a, samples)
    return (y[-1] ** 2 + y[-2] ** 2 - coeff * y[-1] * y[-2]) / (N * N)


def detect_ctcss(audio: np.ndarray) -> bool:
    power = _goertzel(audio, CTCSS_FREQ, AUDIO_RATE)
    print(f"[CTCSS] power: {power:.8f}")
    return power > CTCSS_THRESHOLD


def process_iq(iq: np.ndarray):
    global _audio_buffer
    global _ch_zi_I, _ch_zi_Q
    global _de_zi, _bp_zi, _notch_zi

    # stage 1: 2 MHz → 200 kHz
    iq_d = resample_poly(iq.real, 1, DECIMATE_1) + 1j * resample_poly(iq.imag, 1, DECIMATE_1)

    # NFM channel filter — zi preserves state across chunk boundaries
    I, _ch_zi_I = lfilter(_ch_b, _ch_a, iq_d.real, zi=_ch_zi_I)
    Q, _ch_zi_Q = lfilter(_ch_b, _ch_a, iq_d.imag, zi=_ch_zi_Q)
    iq_filtered = I + 1j * Q

    # FM demod
    conj = iq_filtered[:-1] * np.conj(iq_filtered[1:])
    demodulated = np.angle(conj)

    # stage 2: 200 kHz → 48 kHz
    audio = resample_poly(demodulated, 6, 25)

    # squelch
    power = np.sqrt(np.mean(audio ** 2))
    print(f"[SQUELCH] power: {power:.8f}")
    if power < SQUELCH:
        if _audio_buffer:
            full_audio = np.concatenate(_audio_buffer)
            transcribe_audio(full_audio)
            _audio_buffer = []
        return

    # de-emphasis — state preserved
    audio, _de_zi = lfilter(_de_b, _de_a, audio, zi=_de_zi)

    # CTCSS notch — state preserved
    audio, _notch_zi = lfilter(_notch_b, _notch_a, audio, zi=_notch_zi)

    # voice bandpass — state preserved
    audio, _bp_zi = sosfilt(_bp_sos, audio, zi=_bp_zi)

    # convert to int16 for playback only
    audio_int16 = (np.clip(audio * 200.0, -1.0, 1.0) * 32767).astype(np.int16)

    # keep float for transcription — vosk_engine expects float
    _audio_buffer.append(audio)

    audio_queue.put(audio_int16)
    _audio_buffer.append(audio_int16)