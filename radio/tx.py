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


# Write in 50ms chunks, paced to real-time
TX_CHUNK_DURATION = 0.05
TX_CHUNK_SAMPLES = int(SAMPLE_RATE * TX_CHUNK_DURATION)


class TXProcessor:
    """
    Transmit processor that handles the full TX pipeline.
    
    CTCSS is handled inside FMModulator to ensure proper signal chain.
    Writes are paced to real-time to prevent buffer overflow.
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
        self._write_paced(iq)
    
    def _transmit_audio(self, audio: np.ndarray):
        """Transmit voice audio with CTCSS, paced to real-time."""
        iq = self._modulator.modulate(audio, with_ctcss=True)
        expected_duration = len(iq) / SAMPLE_RATE
        print(f"[TX] Modulated: {len(iq)} IQ samples = {expected_duration:.2f}s at {int(SAMPLE_RATE)} SPS")
        self._write_paced(iq)
    
    def _write_paced(self, iq: np.ndarray):
        """Write IQ data to SDR paced to real-time to prevent buffer overflow."""
        start_time = time.monotonic()
        samples_written = 0
        total_written = 0

        for i in range(0, len(iq), TX_CHUNK_SAMPLES):
            chunk = iq[i:i + TX_CHUNK_SAMPLES]
            written = self._sdr.write_tx(chunk)
            samples_written += len(chunk)
            total_written += written

            # Pace to real-time: sleep until the SDR should have consumed what we've written
            elapsed = time.monotonic() - start_time
            expected_elapsed = samples_written / SAMPLE_RATE
            sleep_time = expected_elapsed - elapsed - 0.01  # 10ms headroom
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"[TX] Wrote {total_written}/{len(iq)} IQ samples")
