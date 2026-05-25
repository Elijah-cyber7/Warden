"""
FM Modulator for Warden SDR pipeline.

Pure DSP module: audio samples in -> IQ samples out.
NBFM modulation with CTCSS.
"""

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly
from config import (
    SAMPLE_RATE, AUDIO_RATE, NBFM_DEVIATION,
    CTCSS_FREQ, CTCSS_LEVEL, VOICE_HP_CUTOFF, VOICE_LP_CUTOFF
)
from audio.filters import PreemphasisFilter


class FMModulator:
    """
    NBFM modulator for transmission.
    
    Signal chain:
    1. Voice bandpass filter (300-4000 Hz)
    2. Pre-emphasis
    3. Add CTCSS tone (post-pre-emphasis, bypasses voice filter)
    4. FM modulate (audio -> phase -> IQ) directly at SDR sample rate
    """
    
    def __init__(self, ctcss_enabled: bool = True):
        # Voice bandpass
        self._voice_hp_taps = firwin(101, VOICE_HP_CUTOFF, fs=AUDIO_RATE, pass_zero=False)
        self._voice_hp_zi = lfilter_zi(self._voice_hp_taps, 1.0)
        self._voice_lp_taps = firwin(101, VOICE_LP_CUTOFF, fs=AUDIO_RATE, pass_zero=True)
        self._voice_lp_zi = lfilter_zi(self._voice_lp_taps, 1.0)
        
        # Pre-emphasis
        self._preemphasis = PreemphasisFilter()
        
        # CTCSS
        self._ctcss_enabled = ctcss_enabled
        self._ctcss_phase = 0.0
        self._ctcss_phase_inc = 2.0 * np.pi * CTCSS_FREQ / AUDIO_RATE
        
        # FM modulator state
        self._phase = 0.0
        self._phase_sensitivity = 2.0 * np.pi * NBFM_DEVIATION / SAMPLE_RATE
    
    def modulate(self, audio: np.ndarray, with_ctcss: bool = True) -> np.ndarray:
        """
        Modulate audio to FM IQ samples.
        
        Args:
            audio: Float32 audio samples at AUDIO_RATE, range [-1, 1].
            with_ctcss: Whether to add CTCSS tone (default True).
            
        Returns:
            Complex64 IQ samples at SAMPLE_RATE.
        """
        # 1. Voice bandpass
        audio, self._voice_hp_zi = lfilter(self._voice_hp_taps, 1.0, audio, zi=self._voice_hp_zi)
        audio, self._voice_lp_zi = lfilter(self._voice_lp_taps, 1.0, audio, zi=self._voice_lp_zi)
        
        # 2. Pre-emphasis
        audio = self._preemphasis.process(audio)
        
        # 3. Add CTCSS tone
        if self._ctcss_enabled and with_ctcss:
            ctcss = self._generate_ctcss(len(audio))
            audio = audio + ctcss
        
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        
        # 4. Resample to SDR rate
        audio_up = resample_poly(audio, int(SAMPLE_RATE), AUDIO_RATE).astype(np.float32)
        
        # 5. FM modulate: integrate audio for phase, convert to IQ
        phase_delta = self._phase_sensitivity * audio_up
        phase = self._phase + np.cumsum(phase_delta)
        self._phase = phase[-1] % (2.0 * np.pi)
        
        iq = np.exp(1j * phase).astype(np.complex64)
        
        return iq
    
    def _generate_ctcss(self, num_samples: int) -> np.ndarray:
        """Generate CTCSS tone samples at audio rate."""
        phases = self._ctcss_phase + self._ctcss_phase_inc * np.arange(num_samples)
        self._ctcss_phase = (phases[-1] + self._ctcss_phase_inc) % (2.0 * np.pi)
        return (CTCSS_LEVEL * np.sin(phases)).astype(np.float32)
    
    def reset(self):
        """Reset modulator state."""
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
