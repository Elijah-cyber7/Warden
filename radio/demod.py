import numpy as np
from scipy.signal import lfilter, lfilter_zi, sosfilt, sosfilt_zi, firwin, iirnotch, tf2sos, resample_poly
from config import SAMPLE_RATE, AUDIO_RATE, CHANNEL_BW, SQUELCH, CTCSS_FREQ
from audio.player import audio_queue
from transcription.vosk_engine import transcribe_audio
import scipy.io.wavfile as wav

DECIMATE_1 = 50
INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATE_1  # 160 kHz

# overlap-save parameters
_resample_overlap = 512
_resample_buffer = np.array([], dtype=np.float32)
_resample_buffer_ready = False

# channel filter at FULL sample rate (before decimation to prevent aliasing)
_ch_taps = firwin(256, CHANNEL_BW / 2, fs=SAMPLE_RATE)
_ch_zi_I = lfilter_zi(_ch_taps, 1.0) * 0
_ch_zi_Q = lfilter_zi(_ch_taps, 1.0) * 0

# CTCSS notch at INTERMEDIATE_RATE — applied before resample_poly
_notch_b, _notch_a = iirnotch(CTCSS_FREQ, Q=10, fs=INTERMEDIATE_RATE)
_notch_sos = tf2sos(_notch_b, _notch_a)
_notch_zi = sosfilt_zi(_notch_sos)

# DC block — removes DC bias before voice filtering
_dc_taps = firwin(101, 50, fs=AUDIO_RATE, pass_zero=False)
_dc_zi = lfilter_zi(_dc_taps, 1.0)

# voice bandpass: FIR highpass (200 Hz) + FIR lowpass (2000 Hz) cascade
_hp_taps = firwin(201, 200, fs=AUDIO_RATE, pass_zero=False)
_lp_taps = firwin(401, 2000, fs=AUDIO_RATE, pass_zero=True)
_hp_zi = lfilter_zi(_hp_taps, 1.0)
_lp_zi = lfilter_zi(_lp_taps, 1.0)

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
    global _audio_buffer, _dc_zi, _hp_zi, _lp_zi, _notch_zi, _ch_zi_I, _ch_zi_Q, _last_sample
    global _resample_buffer, _resample_buffer_ready

    # 1. Channel filter at FULL sample rate
    I, _ch_zi_I = lfilter(_ch_taps, 1.0, iq.real, zi=_ch_zi_I)
    Q, _ch_zi_Q = lfilter(_ch_taps, 1.0, iq.imag, zi=_ch_zi_Q)
    iq_filtered = (I + 1j * Q).astype(np.complex64)

    # 2. Decimate to 160 kHz
    iq_d = iq_filtered[::DECIMATE_1]

    # squelch on channel power
    channel_power = np.mean(np.abs(iq_d) ** 2)
    if channel_power < SQUELCH:
        _flush_buffer()
        _resample_buffer = np.array([], dtype=np.float32)
        _resample_buffer_ready = False
        return

    # 3. FM demod with IQ context
    iq_ext = np.concatenate(([_last_sample], iq_d))
    _last_sample = iq_d[-1]
    conj = iq_ext[:-1] * np.conj(iq_ext[1:])
    demodulated = np.angle(conj).astype(np.float32)

    # 4. CTCSS notch at intermediate rate — before resample
    demodulated, _notch_zi = sosfilt(_notch_sos, demodulated, zi=_notch_zi)

    # 5. Overlap-save resample: 160 kHz -> 48 kHz
    if len(_resample_buffer) > 0:
        demodulated = np.concatenate([_resample_buffer, demodulated])
    _resample_buffer = demodulated[-_resample_overlap:].copy()
    audio = resample_poly(demodulated, int(AUDIO_RATE), INTERMEDIATE_RATE)
    discard = int(_resample_overlap * AUDIO_RATE / INTERMEDIATE_RATE)
    if _resample_buffer_ready and discard > 0 and len(audio) > discard:
        audio = audio[discard:]
    _resample_buffer_ready = True

    # 6. DC block
    audio, _dc_zi = lfilter(_dc_taps, 1.0, audio, zi=_dc_zi)

    # 7. FIR highpass (200 Hz) + FIR lowpass (2000 Hz) cascade
    audio, _hp_zi = lfilter(_hp_taps, 1.0, audio, zi=_hp_zi)
    audio, _lp_zi = lfilter(_lp_taps, 1.0, audio, zi=_lp_zi)

    # 8. Fixed gain
    audio = np.clip(audio * 8.0, -1.0, 1.0).astype(np.float32)

    audio_queue.put(audio)
    _audio_buffer.append(audio)