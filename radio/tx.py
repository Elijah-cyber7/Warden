"""
TX processing for Warden.

Handles the transmit pipeline: audio -> modulator -> SDR.
CTCSS tone is integrated in the modulator (post-pre-emphasis).
"""

import time

import numpy as np
from config import AUDIO_RATE, SAMPLE_RATE, TX_SETTLE_SEC, TX_VOICE_GAIN
from radio.sdr import SDRDevice
from radio.modulator import FMModulator


# Write in 50ms chunks, paced to real-time
TX_CHUNK_DURATION = 0.05
TX_CHUNK_SAMPLES = int(SAMPLE_RATE * TX_CHUNK_DURATION)
TX_RAMP_DOWN_SEC = 0.02


class TXProcessor:
    """
    Transmit processor that handles the full TX pipeline.
    
    CTCSS is handled inside FMModulator to ensure proper signal chain.
    Writes are paced to real-time to prevent buffer overflow.
    """
    
    def __init__(self, sdr: SDRDevice):
        self._sdr = sdr
        self._modulator = FMModulator(ctcss_enabled=True)
    
    def transmit(self, audio: np.ndarray, lead_in: float = 0.2, lead_out: float = 0.5):
        """
        Transmit audio with CTCSS tone.
        
        Builds the entire audio as one continuous block before modulating
        to avoid resample_poly boundary artifacts that break CTCSS continuity.
        
        Args:
            audio: Float32 audio samples at AUDIO_RATE.
            lead_in: Seconds of CTCSS-only carrier before voice.
            lead_out: Seconds of CTCSS-only carrier after voice.
        """
        # Build one continuous audio block: [silence | voice | silence]
        segments = []
        if lead_in > 0:
            segments.append(np.zeros(int(AUDIO_RATE * lead_in), dtype=np.float32))
        voice_audio = self._apply_voice_gain(audio)
        segments.append(voice_audio)
        if lead_out > 0:
            segments.append(np.zeros(int(AUDIO_RATE * lead_out), dtype=np.float32))
        
        full_audio = np.concatenate(segments)
        total_duration = len(full_audio) / AUDIO_RATE
        print(f"[TX] Transmitting {total_duration:.2f}s ({lead_in:.1f}s lead-in + {len(audio)/AUDIO_RATE:.2f}s voice + {lead_out:.1f}s lead-out)")
        
        # Single modulate call = no boundary artifacts
        iq = self._modulator.modulate(full_audio, with_ctcss=True)
        iq = self._apply_ramp_down(iq)
        print(f"[TX] Modulated: {len(iq)} IQ samples = {len(iq)/SAMPLE_RATE:.2f}s at {int(SAMPLE_RATE)} SPS")
        
        self._sdr.start_tx()
        time.sleep(TX_SETTLE_SEC)
        
        try:
            self._write_paced(iq)
        finally:
            self._sdr.stop_tx()
            self._modulator.reset()
            print("[TX] Transmission complete")
    
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

    def _apply_voice_gain(self, audio: np.ndarray) -> np.ndarray:
        """Boost voice before modulation without changing CTCSS-only segments."""
        return np.clip(audio.astype(np.float32) * TX_VOICE_GAIN, -1.0, 1.0)

    def _apply_ramp_down(self, iq: np.ndarray) -> np.ndarray:
        """Softly reduce RF amplitude before stopping TX to avoid squelch pops."""
        ramp_samples = min(len(iq), int(SAMPLE_RATE * TX_RAMP_DOWN_SEC))
        if ramp_samples <= 1:
            return iq

        iq = iq.copy()
        ramp = np.linspace(1.0, 0.0, ramp_samples, dtype=np.float32)
        iq[-ramp_samples:] *= ramp
        return iq
