"""
TX processing for Warden.

Handles the transmit pipeline: audio -> modulator -> SDR.
CTCSS tone is integrated in the modulator (post-pre-emphasis).
"""

import time

import numpy as np
from config import AUDIO_RATE, SAMPLE_RATE, TX_SETTLE_SEC
from radio.sdr import SDRDevice
from radio.modulator import FMModulator


# Small chunk size forces blocking backpressure from SDR hardware
TX_CHUNK_SAMPLES = int(SAMPLE_RATE * 0.02)  # 20ms chunks


class TXProcessor:
    """
    Transmit processor that handles the full TX pipeline.
    
    CTCSS is handled inside FMModulator to ensure proper signal chain.
    Writes in small chunks so hardware backpressure paces the transmission.
    """
    
    def __init__(self, sdr: SDRDevice):
        self._sdr = sdr
        self._modulator = FMModulator(ctcss_enabled=True)
    
    def transmit(self, audio: np.ndarray, lead_in: float = 0.2, lead_out: float = 0.2):
        """
        Transmit audio with CTCSS tone.
        
        Args:
            audio: Float32 audio samples at AUDIO_RATE.
            lead_in: Seconds of CTCSS-only carrier before voice.
            lead_out: Seconds of CTCSS-only carrier after voice.
        """
        print(f"[TX] Transmitting {len(audio)/AUDIO_RATE:.2f}s of audio")
        
        self._sdr.start_tx()
        time.sleep(TX_SETTLE_SEC)
        
        try:
            if lead_in > 0:
                self._transmit_tone_only(lead_in)
            
            self._transmit_audio(audio)
            
            if lead_out > 0:
                self._transmit_tone_only(lead_out)
                
        finally:
            self._sdr.stop_tx()
            self._modulator.reset()
            print("[TX] Transmission complete")
    
    def _transmit_tone_only(self, duration: float):
        """Transmit CTCSS tone only (no voice) for lead-in/out."""
        num_samples = int(AUDIO_RATE * duration)
        silence = np.zeros(num_samples, dtype=np.float32)
        iq = self._modulator.modulate(silence, with_ctcss=True)
        self._write_chunks(iq)
    
    def _transmit_audio(self, audio: np.ndarray):
        """Transmit voice audio with CTCSS."""
        iq = self._modulator.modulate(audio, with_ctcss=True)
        self._write_chunks(iq)
    
    def _write_chunks(self, iq: np.ndarray):
        """Write IQ data in small chunks. SDR write_tx blocks until accepted."""
        for i in range(0, len(iq), TX_CHUNK_SAMPLES):
            chunk = iq[i:i + TX_CHUNK_SAMPLES]
            self._sdr.write_tx(chunk)
