import numpy as np
from scipy.signal import lfilter, lfilter_zi, butter, sosfilt, sosfilt_zi, firwin, iirnotch, tf2sos, resample_poly
from config import SAMPLE_RATE, AUDIO_RATE, CHANNEL_BW, SQUELCH, CTCSS_FREQ
from audio.player import audio_queue
from transcription.vosk_engine import transcribe_audio
import scipy.io.wavfile as wav

DECIMATE_1 = 50
INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATE_1  # 160 kHz

# Overlap-save parameters for resample_poly
# resample_poly uses a FIR filter internally; we need to overlap by the filter length
_resample_overlap = 256  # samples to carry over between chunks
_resample_buffer = np.array([], dtype=np.float32)

# channel filter at FULL sample rate (before decimation to prevent aliasing)
_ch_taps = firwin(256, CHANNEL_BW / 2 / (SAMPLE_RATE / 2))
_ch_zi_I = lfilter_zi(_ch_taps, 1.0) * 0
_ch_zi_Q = lfilter_zi(_ch_taps, 1.0) * 0

# CTCSS notch — convert to SOS for numerical stability
_notch_b, _notch_a = iirnotch(CTCSS_FREQ / (AUDIO_RATE / 2), Q=35)
_notch_sos = tf2sos(_notch_b, _notch_a)
_notch_zi = sosfilt_zi(_notch_sos) * 0

# voice bandpass with state
_vp_sos = butter(6, [200 / (AUDIO_RATE / 2), 2500 / (AUDIO_RATE / 2)], btype='band', output='sos')
_vp_zi = sosfilt_zi(_vp_sos) * 0

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
    global _audio_buffer, _vp_zi, _notch_zi, _ch_zi_I, _ch_zi_Q, _last_sample, _resample_buffer

    # 1. Channel filter at FULL sample rate (before decimation to prevent aliasing)
    I, _ch_zi_I = lfilter(_ch_taps, 1.0, iq.real, zi=_ch_zi_I)
    Q, _ch_zi_Q = lfilter(_ch_taps, 1.0, iq.imag, zi=_ch_zi_Q)
    iq_filtered = (I + 1j * Q).astype(np.complex64)

    # 2. Decimate to 160 kHz
    iq_d = iq_filtered[::DECIMATE_1]

    # squelch on channel power
    channel_power = np.mean(np.abs(iq_d) ** 2)
    if channel_power < SQUELCH:
        _flush_buffer()
        _resample_buffer = np.array([], dtype=np.float32)  # reset on squelch
        return

    # 3. FM demod with IQ context
    iq_ext = np.concatenate(([_last_sample], iq_d))
    _last_sample = iq_d[-1]
    conj = iq_ext[:-1] * np.conj(iq_ext[1:])
    demodulated = np.angle(conj).astype(np.float32)

    # 4. Overlap-save resampling: 160 kHz -> 48 kHz
    # Prepend overlap from previous chunk
    if len(_resample_buffer) > 0:
        demodulated = np.concatenate([_resample_buffer, demodulated])
    
    # Save overlap for next chunk
    _resample_buffer = demodulated[-_resample_overlap:].copy()
    
    # Resample the full buffer
    audio = resample_poly(demodulated, int(AUDIO_RATE), INTERMEDIATE_RATE)
    
    # Discard the output corresponding to the overlap region
    # Output samples to discard = overlap * (AUDIO_RATE / INTERMEDIATE_RATE)
    discard = int(_resample_overlap * AUDIO_RATE / INTERMEDIATE_RATE)
    if len(_resample_buffer) == _resample_overlap and discard > 0:
        audio = audio[discard:]

    # 5. CTCSS notch (SOS, stable)
    audio, _notch_zi = sosfilt(_notch_sos, audio, zi=_notch_zi)

    # 6. Voice bandpass with state
    audio, _vp_zi = sosfilt(_vp_sos, audio, zi=_vp_zi)

    # 7. Fixed gain
    audio = np.clip(audio * 8.0, -1.0, 1.0).astype(np.float32)

    audio_queue.put(audio)
    _audio_buffer.append(audio)