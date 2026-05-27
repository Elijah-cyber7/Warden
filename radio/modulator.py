"""
FM Modulator for Warden SDR pipeline.

Pure DSP module: audio samples in -> IQ samples out.
Proper NBFM modulation with bandwidth limiting and CTCSS.
"""

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly
import config
from config import (
    SAMPLE_RATE, AUDIO_RATE, NBFM_DEVIATION, DECIMATION,
    VOICE_HP_CUTOFF, VOICE_LP_CUTOFF,
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
    6. Interpolate to SDR sample rate

    Note: there is intentionally NO real-valued baseband channel filter on I/Q.
    Filtering I and Q independently with a real lowpass destroys FM's
    constant-envelope property and grafts amplitude modulation onto the
    output (peaks well past unity → DAC clipping → splatter and a low-freq
    AM wobble that wrecks the receiver's CTCSS decoder). Carson's-rule
    bandwidth is already enforced by the choice of NBFM_DEVIATION and the
    voice low-pass; the resample_poly stages handle anti-imaging.
    """

    def __init__(self, ctcss_enabled: bool = True):
        self._voice_hp_taps = firwin(101, VOICE_HP_CUTOFF, fs=AUDIO_RATE, pass_zero=False)
        self._voice_hp_zi = lfilter_zi(self._voice_hp_taps, 1.0)
        self._voice_lp_taps = firwin(101, VOICE_LP_CUTOFF, fs=AUDIO_RATE, pass_zero=True)
        self._voice_lp_zi = lfilter_zi(self._voice_lp_taps, 1.0)

        self._preemphasis = PreemphasisFilter()

        self._ctcss_enabled = ctcss_enabled
        self._ctcss_phase = 0.0

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

        # Generate complex-baseband FM signal. We work on the complex signal
        # directly so the resample_poly anti-imaging filter sees I and Q as
        # one entity and preserves the constant envelope.
        iq_inter = np.exp(1j * phase).astype(np.complex64)

        iq_out = resample_poly(iq_inter, DECIMATION, 1).astype(np.complex64)

        return iq_out

    def _generate_ctcss(self, num_samples: int) -> np.ndarray:
        """Generate phase-continuous CTCSS tone samples at audio rate."""
        if num_samples <= 0:
            return np.array([], dtype=np.float32)
        phase_inc = 2.0 * np.pi * config.CTCSS_FREQ / AUDIO_RATE
        phases = self._ctcss_phase + phase_inc * np.arange(num_samples)
        self._ctcss_phase = (phases[-1] + phase_inc) % (2.0 * np.pi)
        return (config.CTCSS_LEVEL * np.sin(phases)).astype(np.float32)

    def _scale_voice_for_ctcss(self, audio: np.ndarray) -> np.ndarray:
        """Scale voice amplitude to leave headroom for the CTCSS tone."""
        if len(audio) == 0:
            return audio.astype(np.float32)

        headroom = max(0.0, 1.0 - config.CTCSS_LEVEL)
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
        self._phase = 0.0

    @property
    def ctcss_enabled(self) -> bool:
        return self._ctcss_enabled

    @ctcss_enabled.setter
    def ctcss_enabled(self, enabled: bool):
        self._ctcss_enabled = enabled
