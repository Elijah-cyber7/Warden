import numpy as np
from scipy.signal import lfilter, lfilter_zi, resample_poly, sosfilt, sosfilt_zi, firwin, iirnotch
from config import SAMPLE_RATE, AUDIO_RATE, CHANNEL_BW, SQUELCH
from audio.player import audio_queue
from transcription.vosk_engine import transcribe_audio
import scipy.io.wavfile as wav

DECIMATE_1 = 50
INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATE_1

# overlap-save parameters for resample_poly
_resample_overlap = 1024
_resample_buffer = np.array([], dtype=np.float32)
_resample_buffer_ready = False

# channel filter with state
_ch_taps = firwin(128, CHANNEL_BW / 2 / (INTERMEDIATE_RATE / 2))
_ch_zi_I = lfilter_zi(_ch_taps, 1.0)
_ch_zi_Q = lfilter_zi(_ch_taps, 1.0)

# voice bandpass: FIR highpass (400 Hz) + FIR lowpass (3400 Hz) cascade
_hp_taps = firwin(801, 400, fs=AUDIO_RATE, pass_zero=False)
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

# Noise gate state
_gate_threshold = 0.02  # RMS threshold to open gate
_gate_open = False
_gate_attack = 0.005  # fast open
_gate_release = 0.0005  # slow close to avoid cutting words
_gate_gain = 0.0


def process_iq(iq):
    global _audio_buffer, _hp_zi, _lp_zi, _ch_zi_I, _ch_zi_Q, _last_I, _last_Q
    global _resample_buffer, _resample_buffer_ready, _agc_gain, _gate_gain

    iq_power = np.mean(np.abs(iq) ** 2)
    if iq_power < SQUELCH:
        if _audio_buffer:
            full_audio = np.concatenate(_audio_buffer)
            wav.write('debug.wav', AUDIO_RATE, (full_audio * 32767).astype(np.int16))
            transcribe_audio(full_audio)
            _audio_buffer = []
        _resample_buffer = np.array([], dtype=np.float32)
        _resample_buffer_ready = False
        return

    # decimate
    iq_d = iq[::DECIMATE_1]

    # channel filter with state
    I, _ch_zi_I = lfilter(_ch_taps, 1.0, iq_d.real, zi=_ch_zi_I)
    Q, _ch_zi_Q = lfilter(_ch_taps, 1.0, iq_d.imag, zi=_ch_zi_Q)

    # Quadrature FM demod: (I * dQ - Q * dI) / (I² + Q²)
    # Compute derivatives
    I_ext = np.concatenate(([_last_I], I))
    Q_ext = np.concatenate(([_last_Q], Q))
    _last_I = I[-1]
    _last_Q = Q[-1]
    
    dI = np.diff(I_ext)
    dQ = np.diff(Q_ext)
    
    # Quadrature demod formula
    mag_sq = I**2 + Q**2 + 1e-10  # avoid division by zero
    demodulated = ((I * dQ - Q * dI) / mag_sq).astype(np.float32)

    # overlap-save resample: 160 kHz -> 48 kHz
    if len(_resample_buffer) > 0:
        demodulated = np.concatenate([_resample_buffer, demodulated])
    _resample_buffer = demodulated[-_resample_overlap:].copy()
    audio = resample_poly(demodulated, int(AUDIO_RATE), INTERMEDIATE_RATE)
    discard = int(_resample_overlap * AUDIO_RATE / INTERMEDIATE_RATE)
    if _resample_buffer_ready and discard > 0 and len(audio) > discard:
        audio = audio[discard:]
    _resample_buffer_ready = True

    # FIR highpass (200 Hz) + FIR lowpass (4500 Hz) cascade
    audio, _hp_zi = lfilter(_hp_taps, 1.0, audio, zi=_hp_zi)
    audio, _lp_zi = lfilter(_lp_taps, 1.0, audio, zi=_lp_zi)

    # AGC - adjust gain to maintain target level
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
    if rms * _agc_gain > _agc_target:
        # too loud, reduce gain quickly
        _agc_gain = max(0.1, _agc_gain - _agc_attack)
    else:
        # too quiet, increase gain slowly
        _agc_gain = min(50.0, _agc_gain + _agc_decay)
    
    audio = (audio * _agc_gain).astype(np.float32)
    
    # Noise gate - mute when signal is below threshold
    rms_post = np.sqrt(np.mean(audio ** 2)) + 1e-10
    if rms_post > _gate_threshold:
        _gate_gain = min(1.0, _gate_gain + _gate_attack)
    else:
        _gate_gain = max(0.0, _gate_gain - _gate_release)
    audio = audio * _gate_gain
    
    # Output gain boost
    audio = audio * 3.0
    audio = np.clip(audio, -1.0, 1.0)

    audio_queue.put(audio)
    _audio_buffer.append(audio)