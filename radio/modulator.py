"""
FM Modulator for Warden SDR pipeline.

Pure DSP module: audio samples in -> IQ samples out.
Proper NBFM modulation with bandwidth limiting and CTCSS.
"""

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly
from config import (
    SAMPLE_RATE, AUDIO_RATE, NBFM_DEVIATION, CHANNEL_BW, DECIMATION,
    CTCSS_FREQ, CTCSS_LEVEL, VOICE_HP_CUTOFF, VOICE_LP_CUTOFF,
    CHANNEL_FILTER_TAPS
)
from audio.filters import PreemphasisFilter


INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATION


class FMModulator:
    """
    NBFM modulator for transmission.
    
    Signal chain:
    1. Voice bandpass filter (300-4000 Hz)
    2. Pre-emphasis
    3. Add CTCSS tone (post-pre-emphasis)
    4. Interpolate to intermediate rate
    5. FM modulate (audio -> phase -> IQ)
    6. Channel filter (bandwidth limiting)
    7. Interpolate to SDR sample rate
    """
    
    def __init__(self, ctcss_enabled: bool = True):
        # Voice bandpass (same as RX for symmetry)
        self._voice_hp_taps = firwin(401, VOICE_HP_CUTOFF, fs=AUDIO_RATE, pass_zero=False)
        self._voice_hp_zi = lfilter_zi(self._voice_hp_taps, 1.0)
        self._voice_lp_taps = firwin(401, VOICE_LP_CUTOFF, fs=AUDIO_RATE, pass_zero=True)
        self._voice_lp_zi = lfilter_zi(self._voice_lp_taps, 1.0)
        
        # Pre-emphasis
        self._preemphasis = PreemphasisFilter()
        
        # CTCSS
        self._ctcss_enabled = ctcss_enabled
        self._ctcss_phase = 0.0
        self._ctcss_phase_inc = 2.0 * np.pi * CTCSS_FREQ / AUDIO_RATE
        
        # Channel filter at intermediate rate (bandwidth limiting)
        self._channel_taps = firwin(CHANNEL_FILTER_TAPS, CHANNEL_BW / 2, fs=INTERMEDIATE_RATE)
        self._channel_zi_I = lfilter_zi(self._channel_taps, 1.0)
        self._channel_zi_Q = lfilter_zi(self._channel_taps, 1.0)
        
        # FM modulator state
        self._phase = 0.0
        self._phase_sensitivity = 2.0 * np.pi * NBFM_DEVIATION / INTERMEDIATE_RATE
    
    def modulate(self, audio: np.ndarray, with_ctcss: bool = True) -> np.ndarray:
        """
        Modulate audio to FM IQ samples.
        
        Args:
            audio: Float32 audio samples at AUDIO_RATE, range [-1, 1].
            with_ctcss: Whether to add CTCSS tone (default True).
            
        Returns:
            Complex64 IQ samples at SAMPLE_RATE.
        """
        # 1. Voice bandpass filter
        audio, self._voice_hp_zi = lfilter(self._voice_hp_taps, 1.0, audio, zi=self._voice_hp_zi)
        audio, self._voice_lp_zi = lfilter(self._voice_lp_taps, 1.0, audio, zi=self._voice_lp_zi)
        
        # 2. Pre-emphasis
        audio = self._preemphasis.process(audio)
        
        # 3. Add CTCSS tone (post-pre-emphasis so it's not attenuated)
        if self._ctcss_enabled and with_ctcss:
            ctcss = self._generate_ctcss(len(audio))
            audio = np.clip(audio + ctcss, -1.0, 1.0)
        
        # 4. Interpolate to intermediate rate
        audio_interp = resample_poly(audio, INTERMEDIATE_RATE, AUDIO_RATE).astype(np.float32)
        
        # 5. FM modulate: audio -> instantaneous phase -> IQ
        phase_delta = self._phase_sensitivity * audio_interp
        phase = self._phase + np.cumsum(phase_delta)
        self._phase = phase[-1] % (2.0 * np.pi)
        
        I = np.cos(phase).astype(np.float32)
        Q = np.sin(phase).astype(np.float32)
        
        # 6. Channel filter (bandwidth limiting)
        I, self._channel_zi_I = lfilter(self._channel_taps, 1.0, I, zi=self._channel_zi_I)
        Q, self._channel_zi_Q = lfilter(self._channel_taps, 1.0, Q, zi=self._channel_zi_Q)
        
        # 7. Interpolate to SDR sample rate
        I_out = resample_poly(I, DECIMATION, 1).astype(np.float32)
        Q_out = resample_poly(Q, DECIMATION, 1).astype(np.float32)
        
        return (I_out + 1j * Q_out).astype(np.complex64)
    
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
        self._channel_zi_I = lfilter_zi(self._channel_taps, 1.0)
        self._channel_zi_Q = lfilter_zi(self._channel_taps, 1.0)
        self._phase = 0.0
    
    @property
    def ctcss_enabled(self) -> bool:
        return self._ctcss_enabled
    
    @ctcss_enabled.setter
    def ctcss_enabled(self, enabled: bool):
        self._ctcss_enabled = enabled
