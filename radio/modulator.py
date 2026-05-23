"""
FM Modulator for Warden SDR pipeline.

Pure DSP module: audio samples in -> IQ samples out.
"""

import numpy as np
from scipy.signal import resample_poly
from config import SAMPLE_RATE, AUDIO_RATE, NBFM_DEVIATION
from audio.filters import PreemphasisFilter


class FMModulator:
    """
    NBFM modulator for transmission.
    
    Signal chain:
    1. Pre-emphasis (boost high frequencies)
    2. Resample audio to SDR sample rate
    3. Integrate audio to get instantaneous phase
    4. Convert phase to IQ: exp(j * phase)
    """
    
    def __init__(self):
        self._preemphasis = PreemphasisFilter()
        self._phase = 0.0
        self._phase_sensitivity = 2.0 * np.pi * NBFM_DEVIATION / SAMPLE_RATE
    
    def modulate(self, audio: np.ndarray) -> np.ndarray:
        """
        Modulate audio to FM IQ samples.
        
        Args:
            audio: Float32 audio samples at AUDIO_RATE, range [-1, 1].
            
        Returns:
            Complex64 IQ samples at SAMPLE_RATE.
        """
        # 1. Pre-emphasis
        audio = self._preemphasis.process(audio)
        
        # 2. Resample from audio rate to SDR sample rate
        audio_upsampled = resample_poly(audio, int(SAMPLE_RATE), int(AUDIO_RATE))
        
        # 3. FM modulation: integrate audio to get phase
        phase_delta = self._phase_sensitivity * audio_upsampled
        phase = self._phase + np.cumsum(phase_delta)
        
        self._phase = phase[-1] % (2.0 * np.pi)
        
        # 4. Convert phase to IQ
        iq = np.exp(1j * phase).astype(np.complex64)
        
        return iq
    
    def reset(self):
        """Reset modulator state."""
        self._preemphasis.reset()
        self._phase = 0.0


def generate_silence_iq(duration_seconds: float) -> np.ndarray:
    """
    Generate IQ samples for silence (carrier only, no modulation).
    
    Args:
        duration_seconds: Duration of silence in seconds.
        
    Returns:
        Complex64 IQ samples (unmodulated carrier).
    """
    num_samples = int(SAMPLE_RATE * duration_seconds)
    return np.ones(num_samples, dtype=np.complex64)
