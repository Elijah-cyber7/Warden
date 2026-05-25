"""
Audio filters for Warden SDR pipeline.

Provides stateful FIR/IIR filters for voice processing:
- Channel filter for IQ processing
- Voice bandpass (highpass + lowpass cascade)
- De-emphasis (RX) and pre-emphasis (TX)
"""

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi
from config import (
    AUDIO_RATE, SAMPLE_RATE, DECIMATION, CHANNEL_BW,
    VOICE_HP_CUTOFF, VOICE_LP_CUTOFF, EMPHASIS_TAU, CHANNEL_FILTER_TAPS
)


# Derived constants
INTERMEDIATE_RATE = int(SAMPLE_RATE) // DECIMATION


class ChannelFilter:
    """Lowpass filter for channel selection after decimation."""
    
    def __init__(self):
        self._taps = firwin(CHANNEL_FILTER_TAPS, CHANNEL_BW / 2 / (INTERMEDIATE_RATE / 2))
        self._zi_I = lfilter_zi(self._taps, 1.0)
        self._zi_Q = lfilter_zi(self._taps, 1.0)
    
    def process(self, I: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter I and Q components separately with state preservation."""
        I_out, self._zi_I = lfilter(self._taps, 1.0, I, zi=self._zi_I)
        Q_out, self._zi_Q = lfilter(self._taps, 1.0, Q, zi=self._zi_Q)
        return I_out, Q_out
    
    def reset(self):
        """Reset filter state for new transmission."""
        self._zi_I = lfilter_zi(self._taps, 1.0)
        self._zi_Q = lfilter_zi(self._taps, 1.0)


class VoiceBandpassFilter:
    """Cascaded highpass + lowpass FIR filter for voice frequencies."""
    
    def __init__(self):
        self._hp_taps = firwin(401, VOICE_HP_CUTOFF, fs=AUDIO_RATE, pass_zero=False)
        self._hp_zi = lfilter_zi(self._hp_taps, 1.0)
        
        self._lp_taps = firwin(401, VOICE_LP_CUTOFF, fs=AUDIO_RATE, pass_zero=True)
        self._lp_zi = lfilter_zi(self._lp_taps, 1.0)
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter with state preservation."""
        audio, self._hp_zi = lfilter(self._hp_taps, 1.0, audio, zi=self._hp_zi)
        audio, self._lp_zi = lfilter(self._lp_taps, 1.0, audio, zi=self._lp_zi)
        return audio
    
    def reset(self):
        """Reset filter state for new transmission."""
        self._hp_zi = lfilter_zi(self._hp_taps, 1.0)
        self._lp_zi = lfilter_zi(self._lp_taps, 1.0)


class DeemphasisFilter:
    """
    Single-pole IIR lowpass for de-emphasis.
    Compensates for pre-emphasis applied during transmission.
    """
    
    def __init__(self, sample_rate: int = AUDIO_RATE):
        self._alpha = np.exp(-1.0 / (sample_rate * EMPHASIS_TAU))
        self._b = np.array([1.0 - self._alpha])
        self._a = np.array([1.0, -self._alpha])
        self._zi = lfilter_zi(self._b, self._a)
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply de-emphasis with state preservation."""
        audio, self._zi = lfilter(self._b, self._a, audio, zi=self._zi)
        return audio
    
    def reset(self):
        """Reset filter state for new transmission."""
        self._zi = lfilter_zi(self._b, self._a)


class PreemphasisFilter:
    """
    Single-zero FIR pre-emphasis filter.
    Boosts high frequencies before transmission without adding low-frequency gain.
    """
    
    def __init__(self, sample_rate: int = AUDIO_RATE):
        self._alpha = np.exp(-1.0 / (sample_rate * EMPHASIS_TAU))
        self._b = np.array([1.0, -self._alpha])
        self._a = np.array([1.0])
        self._zi = np.zeros(len(self._b) - 1, dtype=np.float64)
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis with state preservation."""
        audio, self._zi = lfilter(self._b, self._a, audio, zi=self._zi)
        return audio
    
    def reset(self):
        """Reset filter state for new transmission."""
        self._zi = np.zeros(len(self._b) - 1, dtype=np.float64)
