"""
Tone generators for Warden SDR pipeline.

Provides CTCSS (Continuous Tone-Coded Squelch System) tone generation
for TX to open squelch on receiving radios.
"""

import numpy as np
from config import CTCSS_FREQ, CTCSS_LEVEL, AUDIO_RATE


class CTCSSGenerator:
    """
    Generates continuous CTCSS tone for FM transmission.
    
    CTCSS tones are sub-audible (67-254 Hz) and must be present
    throughout the transmission for the receiving radio to unmute.
    """
    
    def __init__(self, freq: float = CTCSS_FREQ, level: float = CTCSS_LEVEL, 
                 sample_rate: int = AUDIO_RATE):
        self._freq = freq
        self._level = level
        self._sample_rate = sample_rate
        self._phase = 0.0
        self._phase_increment = 2.0 * np.pi * freq / sample_rate
    
    def generate(self, num_samples: int) -> np.ndarray:
        """
        Generate CTCSS tone samples.
        
        Args:
            num_samples: Number of samples to generate.
            
        Returns:
            Float32 array of tone samples.
        """
        phases = self._phase + self._phase_increment * np.arange(num_samples)
        self._phase = (phases[-1] + self._phase_increment) % (2.0 * np.pi)
        tone = (self._level * np.sin(phases)).astype(np.float32)
        return tone
    
    def reset(self):
        """Reset phase to zero."""
        self._phase = 0.0
    
    @property
    def frequency(self) -> float:
        """Current CTCSS frequency in Hz."""
        return self._freq
    
    @frequency.setter
    def frequency(self, freq: float):
        """Set CTCSS frequency in Hz."""
        self._freq = freq
        self._phase_increment = 2.0 * np.pi * freq / self._sample_rate


def mix_audio_with_ctcss(audio: np.ndarray, ctcss: CTCSSGenerator) -> np.ndarray:
    """
    Mix voice audio with CTCSS tone.
    
    Args:
        audio: Voice audio samples.
        ctcss: CTCSS generator instance.
        
    Returns:
        Mixed audio with CTCSS tone added.
    """
    tone = ctcss.generate(len(audio))
    mixed = audio + tone
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)
