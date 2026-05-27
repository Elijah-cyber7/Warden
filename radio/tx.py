"""
TX processing for Warden.

Handles the transmit pipeline: audio -> modulator -> SDR.
CTCSS tone is integrated in the modulator (post-pre-emphasis).
"""

import logging
import time

import numpy as np
from config import (
    AUDIO_RATE, SAMPLE_RATE, TX_SETTLE_SEC,
    TX_LEAD_IN_SEC, TX_LEAD_OUT_SEC,
)
from radio.sdr import SDRDevice
from radio.modulator import FMModulator

log = logging.getLogger("warden.tx")

# Large chunks so we keep the HackRF TX FIFO full. Soapy blocks on writeStream
# when the FIFO is full, so this naturally rate-limits without us calling sleep.
TX_CHUNK_SAMPLES = 1 << 18  # 262_144 samples (~131 ms at 2 MSPS)


class TXProcessor:
    """
    Transmit processor: builds a single continuous IQ block and streams it.

    The entire transmission (lead-in + voice + lead-out) is modulated in one
    call to avoid resample_poly boundary artifacts that break CTCSS continuity.
    """

    def __init__(self, sdr: SDRDevice):
        self._sdr = sdr
        self._modulator = FMModulator(ctcss_enabled=True)

    def transmit(self, audio: np.ndarray,
                 lead_in: float = TX_LEAD_IN_SEC,
                 lead_out: float = TX_LEAD_OUT_SEC):
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

        write_start = time.monotonic()
        try:
            self._write_blocking(iq)
        finally:
            # SoapyHackRF's deactivateStream halts TX immediately rather than
            # draining. If we stop right after the last writeStream returns,
            # whatever is still queued in the HackRF FIFO (typ. a few hundred
            # ms at 2 MSPS) is silently truncated — including the CTCSS
            # lead-out. Sleep for the remaining signal duration so the FIFO
            # has actually played out into RF before we cut the carrier.
            target_dur = len(iq) / SAMPLE_RATE
            elapsed = time.monotonic() - write_start
            drain = target_dur - elapsed
            if drain > 0:
                time.sleep(drain + 0.05)
                log.debug("Drained TX FIFO for %.3fs", drain + 0.05)
            self._sdr.stop_tx()
            log.info("Transmission complete")

    def _write_blocking(self, iq: np.ndarray):
        """
        Stream IQ data to the SDR.

        Soapy's writeStream blocks once the HackRF TX FIFO is full, which is the
        correct backpressure mechanism. Avoid sleeping between writes — that can
        let the FIFO drain and produce a brief carrier dropout, which the
        receiver hears as a CTCSS / phase glitch.
        """
        total_written = 0
        for i in range(0, len(iq), TX_CHUNK_SAMPLES):
            chunk = iq[i:i + TX_CHUNK_SAMPLES]
            total_written += self._sdr.write_tx(chunk)
        log.debug("Wrote %d/%d IQ samples", total_written, len(iq))
