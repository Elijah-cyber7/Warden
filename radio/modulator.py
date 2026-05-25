"""
FM Modulator for Warden SDR pipeline.

Pure DSP module: audio samples in -> IQ samples out.
Proper NBFM modulation with bandwidth limiting and CTCSS.
"""

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly
from config import (
    SAMPLE_RATE, AUDIO_RATE, NBFM_DEVIATION, DECIMATION,
    CTCSS_FREQ, CTCSS_LEVEL, VOICE_HP_CUTOFF, VOICE_LP_CUTOFF,
    TX_VOICE_GAIN
)
from audio.filters import PreemphasisFilter


INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATION


class FMModulator:
    """
    NBFM modulator for transmission.

    Signal chain:
    1. Voice bandpass filter (300-4000 Hz)
    2. Pre-emphasis
    3. Add CTCSS tone (post-filter so it bypasses voice HP)
    4. Interpolate to intermediate rate
    5. FM modulate (audio -> phase -> IQ)
    6. Channel filter (Carson's rule BW)
    7. Interpolate to SDR sample rate
    """

    def __init__(self, ctcss_enabled: bool = True):
        self._voice_hp_taps = firwin(101, VOICE_HP_CUTOFF, fs=AUDIO_RATE, pass_zero=False)
        self._voice_hp_zi = lfilter_zi(self._voice_hp_taps, 1.0)
        self._voice_lp_taps = firwin(101, VOICE_LP_CUTOFF, fs=AUDIO_RATE, pass_zero=True)
        self._voice_lp_zi = lfilter_zi(self._voice_lp_taps, 1.0)

        self._preemphasis = PreemphasisFilter()

        self._ctcss_enabled = ctcss_enabled
        self._ctcss_phase = 0.0
        self._ctcss_phase_inc = 2.0 * np.pi * CTCSS_FREQ / AUDIO_RATE

        carson_bw = 2 * (NBFM_DEVIATION + VOICE_LP_CUTOFF)
        channel_cutoff = min(carson_bw / 2, INTERMEDIATE_RATE / 2 * 0.9)
        self._channel_taps = firwin(65, channel_cutoff, fs=INTERMEDIATE_RATE)
        self._channel_zi_I = lfilter_zi(self._channel_taps, 1.0)
        self._channel_zi_Q = lfilter_zi(self._channel_taps, 1.0)

        self._phase = 0.0
        self._phase_sensitivity = 2.0 * np.pi * NBFM_DEVIATION / INTERMEDIATE_RATE

    def modulate(self, audio: np.ndarray, with_ctcss: bool = True) -> np.ndarray:
        """
        Modulate audio to FM IQ samples.

        Args:
            audio: Float32 audio samples at AUDIO_RATE, range [-1, 1].
            with_ctcss: Whether to add CTCSS tone.

        Returns:
            Complex64 IQ samples at SAMPLE_RATE.
        """
        audio, self._voice_hp_zi = lfilter(self._voice_hp_taps, 1.0, audio, zi=self._voice_hp_zi)
        audio, self._voice_lp_zi = lfilter(self._voice_lp_taps, 1.0, audio, zi=self._voice_lp_zi)

        audio = self._preemphasis.process(audio)

        # Apply voice gain before mixing with CTCSS
        audio = audio * TX_VOICE_GAIN

        if self._ctcss_enabled and with_ctcss:
            audio = self._scale_voice_for_ctcss(audio)
            ctcss = self._generate_ctcss(len(audio))
            audio = audio + ctcss
        else:
            audio = np.clip(audio, -1.0, 1.0)

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        audio_interp = resample_poly(audio, INTERMEDIATE_RATE, AUDIO_RATE).astype(np.float32)

        phase_delta = self._phase_sensitivity * audio_interp
        phase = self._phase + np.cumsum(phase_delta)
        self._phase = phase[-1] % (2.0 * np.pi)

        I = np.cos(phase).astype(np.float32)
        Q = np.sin(phase).astype(np.float32)

        I, self._channel_zi_I = lfilter(self._channel_taps, 1.0, I, zi=self._channel_zi_I)
        Q, self._channel_zi_Q = lfilter(self._channel_taps, 1.0, Q, zi=self._channel_zi_Q)

        I_out = resample_poly(I, DECIMATION, 1).astype(np.float32)
        Q_out = resample_poly(Q, DECIMATION, 1).astype(np.float32)

        return (I_out + 1j * Q_out).astype(np.complex64)

    def _generate_ctcss(self, num_samples: int) -> np.ndarray:
        """Generate phase-continuous CTCSS tone samples at audio rate."""
        if num_samples <= 0:
            return np.array([], dtype=np.float32)
        phases = self._ctcss_phase + self._ctcss_phase_inc * np.arange(num_samples)
        self._ctcss_phase = (phases[-1] + self._ctcss_phase_inc) % (2.0 * np.pi)
        return (CTCSS_LEVEL * np.sin(phases)).astype(np.float32)

    def _scale_voice_for_ctcss(self, audio: np.ndarray) -> np.ndarray:
        """Scale voice amplitude to leave headroom for the CTCSS tone."""
        if len(audio) == 0:
            return audio.astype(np.float32)

        headroom = max(0.0, 1.0 - CTCSS_LEVEL)
        peak = float(np.max(np.abs(audio)))
        if peak > headroom and peak > 0.0:
            audio = audio * (headroom / peak)
        return audio.astype(np.float32)

    def reset(self):
        """Reset all filter state (use between unrelated transmissions)."""
        self._voice_hp_zi = lfilter_zi(self._voice_hp_taps, 1.0)
        self._voice_lp_zi = lfilter_zi(self._voice_lp_taps, 1.0)
        self._preemphasis.reset()
        self._ctcss_phase = 0.0
        self._channel_zi_I = lfilter_zi(self._channel_taps, 1.0)
        self._channel_zi_Q = lfilter_zi(self._channel_taps, 1.0)
        self._phase = 0.0

    @property
    def ctcss_enabled(self) -> bool:
        return self._ctcss_enabled

    @ctcss_enabled.setter
    def ctcss_enabled(self, enabled: bool):
        self._ctcss_enabled = enabled
