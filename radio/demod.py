import numpy as np
from scipy.signal import lfilter, lfilter_zi, resample_poly, firwin
from config import SAMPLE_RATE, AUDIO_RATE, CHANNEL_BW
from audio.player import audio_queue
from transcription.vosk_engine import transcribe_audio
import scipy.io.wavfile as wav

DECIMATE_1 = 50
INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATE_1  # 2MHz/50 = 40kHz

# NBFM deviation is ±2.5kHz, max phase change per sample:
# 2π * 2500 / INTERMEDIATE_RATE ≈ 0.39 rad/sample at 40kHz
NBFM_DEVIATION = 2500.0
DEMOD_SCALE = INTERMEDIATE_RATE / (2.0 * np.pi * NBFM_DEVIATION)

# overlap-save parameters for resample_poly
_resample_overlap = 1024
_resample_buffer = np.array([], dtype=np.float32)
_resample_buffer_ready = False

# channel filter with state (lowpass at CHANNEL_BW/2 = 6.25kHz)
_ch_taps = firwin(128, CHANNEL_BW / 2 / (INTERMEDIATE_RATE / 2))
_ch_zi_I = lfilter_zi(_ch_taps, 1.0)
_ch_zi_Q = lfilter_zi(_ch_taps, 1.0)

# de-emphasis filter (750µs time constant for land mobile radio)
# This compensates for pre-emphasis used in transmission
_deemph_tau = 750e-6
_deemph_alpha = np.exp(-1.0 / (INTERMEDIATE_RATE * _deemph_tau))
_deemph_b = np.array([1.0 - _deemph_alpha])
_deemph_a = np.array([1.0, -_deemph_alpha])
_deemph_zi = lfilter_zi(_deemph_b, _deemph_a) * 0

# voice bandpass: FIR highpass (300 Hz) + FIR lowpass (3400 Hz) cascade
_hp_taps = firwin(801, 300, fs=AUDIO_RATE, pass_zero=False)
_lp_taps = firwin(401, 3400, fs=AUDIO_RATE, pass_zero=True)
_hp_zi = lfilter_zi(_hp_taps, 1.0)
_lp_zi = lfilter_zi(_lp_taps, 1.0)

_audio_buffer = []
_last_I = 0.0
_last_Q = 0.0

# AGC state
_agc_gain = 1.0
_agc_target = 0.3  # target RMS level
_agc_attack = 0.01  # fast attack for loud signals
_agc_decay = 0.0001  # slow decay to keep gain stable

# Discriminator squelch with hysteresis
# Measures high-frequency noise in demod output - high noise = no signal
_sq_hf_taps = firwin(65, 5000, fs=INTERMEDIATE_RATE, pass_zero=False)  # highpass at 5kHz
_sq_hf_zi = lfilter_zi(_sq_hf_taps, 1.0)
_sq_open = False
_sq_open_thresh = 0.8  # noise level to open squelch (lower = more sensitive)
_sq_close_thresh = 1.2  # noise level to close squelch (hysteresis)
_sq_gain = 0.0  # soft squelch gain for smooth transitions
_sq_attack = 0.15  # fast open
_sq_release = 0.02  # slower close


def _reset_filter_states():
    """Reset all filter states for clean start on new transmission."""
    global _ch_zi_I, _ch_zi_Q, _hp_zi, _lp_zi, _sq_hf_zi, _deemph_zi, _agc_gain, _last_I, _last_Q
    _ch_zi_I = lfilter_zi(_ch_taps, 1.0)
    _ch_zi_Q = lfilter_zi(_ch_taps, 1.0)
    _hp_zi = lfilter_zi(_hp_taps, 1.0)
    _lp_zi = lfilter_zi(_lp_taps, 1.0)
    _sq_hf_zi = lfilter_zi(_sq_hf_taps, 1.0)
    _deemph_zi = lfilter_zi(_deemph_b, _deemph_a) * 0
    _agc_gain = 1.0
    _last_I = 0.0
    _last_Q = 0.0


def process_iq(iq):
    global _audio_buffer, _hp_zi, _lp_zi, _ch_zi_I, _ch_zi_Q, _last_I, _last_Q
    global _resample_buffer, _resample_buffer_ready, _agc_gain, _deemph_zi
    global _sq_hf_zi, _sq_open, _sq_gain

    # decimate
    iq_d = iq[::DECIMATE_1]

    # channel filter with state
    I, _ch_zi_I = lfilter(_ch_taps, 1.0, iq_d.real, zi=_ch_zi_I)
    Q, _ch_zi_Q = lfilter(_ch_taps, 1.0, iq_d.imag, zi=_ch_zi_Q)

    # === HARD LIMITER ===
    # Critical for FM: removes AM noise (ignition, power lines, etc.)
    # FM carries info in frequency, not amplitude - limiting rejects AM interference
    magnitude = np.sqrt(I**2 + Q**2) + 1e-10
    I_limited = I / magnitude
    Q_limited = Q / magnitude

    # === DACM (Differentiate And Cross Multiply) FM DEMODULATOR ===
    # Avoids atan() discontinuities, more robust than angle-based demod
    # Formula: audio = Q*dI - I*dQ (since we limited, I²+Q² ≈ 1)
    I_ext = np.concatenate(([_last_I], I_limited))
    Q_ext = np.concatenate(([_last_Q], Q_limited))
    _last_I = I_limited[-1]
    _last_Q = Q_limited[-1]
    
    dI = np.diff(I_ext)
    dQ = np.diff(Q_ext)
    
    # Cross multiply - no division needed since signal is limited to unit magnitude
    demodulated = (Q_limited * dI - I_limited * dQ).astype(np.float32)
    
    # Scale to normalize for NBFM deviation
    demodulated = demodulated * DEMOD_SCALE

    # === DE-EMPHASIS FILTER ===
    # Compensates for pre-emphasis used in transmission (750µs for land mobile)
    demodulated, _deemph_zi = lfilter(_deemph_b, _deemph_a, demodulated, zi=_deemph_zi)

    # Discriminator squelch - measure high-frequency noise content
    # After limiting, noise floor in demod output indicates signal presence
    hf_noise, _sq_hf_zi = lfilter(_sq_hf_taps, 1.0, demodulated, zi=_sq_hf_zi)
    noise_level = np.sqrt(np.mean(hf_noise ** 2))
    
    # Hysteresis: different thresholds for opening vs closing
    if _sq_open:
        if noise_level > _sq_close_thresh:
            _sq_open = False
    else:
        if noise_level < _sq_open_thresh:
            _sq_open = True
    
    # Soft squelch transition
    if _sq_open:
        _sq_gain = min(1.0, _sq_gain + _sq_attack)
    else:
        _sq_gain = max(0.0, _sq_gain - _sq_release)
    
    # If squelch fully closed, flush buffer and skip
    if _sq_gain < 0.01:
        if _audio_buffer:
            full_audio = np.concatenate(_audio_buffer)
            wav.write('debug.wav', AUDIO_RATE, (full_audio * 32767).astype(np.int16))
            transcribe_audio(full_audio)
            _audio_buffer = []
        _resample_buffer = np.array([], dtype=np.float32)
        _resample_buffer_ready = False
        _reset_filter_states()  # clean start for next transmission
        return

    # overlap-save resample: 40 kHz -> 48 kHz
    if len(_resample_buffer) > 0:
        demodulated = np.concatenate([_resample_buffer, demodulated])
    _resample_buffer = demodulated[-_resample_overlap:].copy()
    audio = resample_poly(demodulated, int(AUDIO_RATE), INTERMEDIATE_RATE)
    discard = int(_resample_overlap * AUDIO_RATE / INTERMEDIATE_RATE)
    if _resample_buffer_ready and discard > 0 and len(audio) > discard:
        audio = audio[discard:]
    _resample_buffer_ready = True

    # voice bandpass: FIR highpass (300 Hz) + FIR lowpass (3400 Hz)
    audio, _hp_zi = lfilter(_hp_taps, 1.0, audio, zi=_hp_zi)
    audio, _lp_zi = lfilter(_lp_taps, 1.0, audio, zi=_lp_zi)

    # AGC - only adjust when squelch fully open to avoid pumping during transitions
    if _sq_gain >= 0.99:
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
        if rms * _agc_gain > _agc_target:
            _agc_gain = max(0.1, _agc_gain - _agc_attack)
        else:
            _agc_gain = min(50.0, _agc_gain + _agc_decay)
    
    audio = (audio * _agc_gain).astype(np.float32)
    
    # Apply soft squelch gain
    audio = audio * _sq_gain
    
    # Final clipping
    audio = np.clip(audio, -1.0, 1.0)

    audio_queue.put(audio)
    _audio_buffer.append(audio)