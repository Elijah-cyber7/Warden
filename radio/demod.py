"""
FM Demodulator for Warden SDR pipeline.

Pure DSP module: IQ samples in -> audio samples out.
"""

import numpy as np
from scipy.signal import resample_poly
from config import SAMPLE_RATE, AUDIO_RATE, DECIMATION, AUDIO_OUTPUT_GAIN
from audio.filters import ChannelFilter, VoiceBandpassFilter, DeemphasisFilter


INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATION


class FMDemodulator:
    """
    NBFM demodulator using quadrature detection.

    Signal chain:
    1. Decimate IQ samples
    2. Channel filter (lowpass)
    3. Quadrature FM demod: (I*dQ - Q*dI) / (I² + Q²)
    4. Resample to audio rate
    5. Voice bandpass filter
    6. De-emphasis
    """

    def __init__(self, output_gain: float = AUDIO_OUTPUT_GAIN):
        self._output_gain = output_gain

        self._channel_filter = ChannelFilter()
        self._voice_filter = VoiceBandpassFilter()
        self._deemphasis = DeemphasisFilter()

        self._last_I = 0.0
        self._last_Q = 0.0

        self._resample_overlap = 1024
        self._resample_buffer = np.array([], dtype=np.float32)
        self._resample_buffer_ready = False

    def process(self, iq: np.ndarray) -> np.ndarray:
        """
        Demodulate IQ samples to audio.

        Args:
            iq: Complex64 IQ samples at SAMPLE_RATE.

        Returns:
            Float32 audio samples at AUDIO_RATE.
        """
        iq_d = iq[::DECIMATION]

        I, Q = self._channel_filter.process(iq_d.real, iq_d.imag)

        I_ext = np.concatenate(([self._last_I], I))
        Q_ext = np.concatenate(([self._last_Q], Q))
        self._last_I = float(I[-1])
        self._last_Q = float(Q[-1])

        dI = np.diff(I_ext)
        dQ = np.diff(Q_ext)

        mag_sq = I**2 + Q**2 + 1e-10
        demodulated = ((I * dQ - Q * dI) / mag_sq).astype(np.float32)

        if len(self._resample_buffer) > 0:
            demodulated = np.concatenate([self._resample_buffer, demodulated])
        self._resample_buffer = demodulated[-self._resample_overlap:].copy()

        audio = resample_poly(demodulated, int(AUDIO_RATE), INTERMEDIATE_RATE)

        discard = int(self._resample_overlap * AUDIO_RATE / INTERMEDIATE_RATE)
        if self._resample_buffer_ready and discard > 0 and len(audio) > discard:
            audio = audio[discard:]
        self._resample_buffer_ready = True

        audio = self._voice_filter.process(audio)
        audio = self._deemphasis.process(audio)
        audio = np.clip(audio * self._output_gain, -1.0, 1.0).astype(np.float32)

        return audio

    def reset(self):
        """Reset all state for a new transmission."""
        self._channel_filter.reset()
        self._voice_filter.reset()
        self._deemphasis.reset()
        self._last_I = 0.0
        self._last_Q = 0.0
        self._resample_buffer = np.array([], dtype=np.float32)
        self._resample_buffer_ready = False
