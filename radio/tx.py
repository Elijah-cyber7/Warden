"""
TX processing for Warden.

Handles the transmit pipeline: audio -> modulator -> SDR.
CTCSS tone is integrated in the modulator (post-pre-emphasis).
"""

import logging
import time

import numpy as np
from config import AUDIO_RATE, SAMPLE_RATE, TX_SETTLE_SEC
from radio.sdr import SDRDevice
from radio.modulator import FMModulator

log = logging.getLogger("warden.tx")

TX_CHUNK_DURATION = 0.05
TX_CHUNK_SAMPLES = int(SAMPLE_RATE * TX_CHUNK_DURATION)


class TXProcessor:
    """
    Transmit processor: builds a single continuous IQ block and streams it.

    The entire transmission (lead-in + voice + lead-out) is modulated in one
    call to avoid resample_poly boundary artifacts that break CTCSS continuity.
    """

    def __init__(self, sdr: SDRDevice):
        self._sdr = sdr
        self._modulator = FMModulator(ctcss_enabled=True)

    def transmit(self, audio: np.ndarray, lead_in: float = 0.2, lead_out: float = 0.5):
        """
        Transmit audio with CTCSS tone.

        Args:
            audio: Float32 audio samples at AUDIO_RATE.
            lead_in: Seconds of CTCSS-only carrier before voice.
            lead_out: Seconds of CTCSS-only carrier after voice.
        """
        segments = []
        if lead_in > 0:
            segments.append(np.zeros(int(AUDIO_RATE * lead_in), dtype=np.float32))
        segments.append(audio.astype(np.float32))
        if lead_out > 0:
            segments.append(np.zeros(int(AUDIO_RATE * lead_out), dtype=np.float32))

        full_audio = np.concatenate(segments)
        voice_dur = len(audio) / AUDIO_RATE
        total_dur = len(full_audio) / AUDIO_RATE
        log.info("TX %.2fs (%.1fs lead-in + %.2fs voice + %.1fs lead-out)",
                 total_dur, lead_in, voice_dur, lead_out)

        iq = self._modulator.modulate(full_audio, with_ctcss=True)
        log.debug("Modulated %d IQ samples (%.2fs)", len(iq), len(iq) / SAMPLE_RATE)

        self._sdr.start_tx()
        time.sleep(TX_SETTLE_SEC)

        try:
            self._write_paced(iq)
        finally:
            self._sdr.stop_tx()
            log.info("Transmission complete")

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

            elapsed = time.monotonic() - start_time
            expected_elapsed = samples_written / SAMPLE_RATE
            sleep_time = expected_elapsed - elapsed - 0.01
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.debug("Wrote %d/%d IQ samples", total_written, len(iq))
