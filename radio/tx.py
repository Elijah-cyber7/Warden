"""
TX processing for Warden.

Handles the transmit pipeline: audio -> modulator -> SDR.
Includes CTCSS tone mixing for squelch compatibility.
"""

import numpy as np
from config import AUDIO_RATE, SAMPLE_RATE
from radio.sdr import SDRDevice
from radio.modulator import FMModulator
from audio.tones import CTCSSGenerator, mix_audio_with_ctcss


class TXProcessor:
    """
    Transmit processor that handles the full TX pipeline.
    
    Mixes audio with CTCSS tone, modulates to FM, and transmits.
    """
    
    def __init__(self, sdr: SDRDevice):
        self._sdr = sdr
        self._modulator = FMModulator()
        self._ctcss = CTCSSGenerator()
    
    def transmit(self, audio: np.ndarray, lead_in: float = 0.1, lead_out: float = 0.1):
        """
        Transmit audio with CTCSS tone.
        
        Args:
            audio: Float32 audio samples at AUDIO_RATE.
            lead_in: Seconds of CTCSS-only carrier before voice (default 100ms).
            lead_out: Seconds of CTCSS-only carrier after voice (default 100ms).
        """
        print(f"[TX] Transmitting {len(audio)/AUDIO_RATE:.2f}s of audio")
        
        self._sdr.start_tx()
        
        try:
            if lead_in > 0:
                self._transmit_ctcss_only(lead_in)
            
            self._transmit_audio(audio)
            
            if lead_out > 0:
                self._transmit_ctcss_only(lead_out)
                
        finally:
            self._sdr.stop_tx()
            self._modulator.reset()
            self._ctcss.reset()
            print("[TX] Transmission complete")
    
    def _transmit_ctcss_only(self, duration: float):
        """Transmit CTCSS tone only (no voice)."""
        num_samples = int(AUDIO_RATE * duration)
        silence = np.zeros(num_samples, dtype=np.float32)
        audio_with_ctcss = mix_audio_with_ctcss(silence, self._ctcss)
        
        iq = self._modulator.modulate(audio_with_ctcss)
        self._sdr.write_tx(iq)
    
    def _transmit_audio(self, audio: np.ndarray):
        """Transmit voice audio with CTCSS."""
        audio_with_ctcss = mix_audio_with_ctcss(audio, self._ctcss)
        iq = self._modulator.modulate(audio_with_ctcss)
        
        chunk_size = int(SAMPLE_RATE * 0.1)
        for i in range(0, len(iq), chunk_size):
            chunk = iq[i:i + chunk_size]
            self._sdr.write_tx(chunk)
