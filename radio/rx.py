"""
RX processing loop for Warden.

Handles the receive pipeline: SDR -> demod -> squelch -> transcription.
Optionally emits IQ/squelch data to a GUI bridge.
"""

import logging
import threading

import numpy as np
from config import SQUELCH_THRESHOLD
from radio.sdr import SDRDevice
from radio.demod import FMDemodulator
from transcription.whisper_engine import transcribe_audio

log = logging.getLogger("warden.rx")


class RXProcessor:
    """
    Receive processor managing the full RX pipeline.

    Handles squelch gating, audio buffering, and transcription triggering.
    Supports pause/resume for half-duplex TX operation.
    """

    def __init__(self, sdr: SDRDevice, bridge=None):
        self._sdr = sdr
        self._bridge = bridge
        self._demod = FMDemodulator()
        self._audio_buffer: list[np.ndarray] = []
        self._running = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()

    def start(self):
        """Start the RX processing loop (blocks until stopped)."""
        self._sdr.start_rx()
        self._running = True
        log.info("RX started (squelch=%.2f)", SQUELCH_THRESHOLD)

        while self._running:
            self._pause_event.wait()

            if not self._running:
                break

            iq = self._sdr.read_rx()
            if iq is not None and not self._paused:
                self._process_block(iq)

    def stop(self):
        """Stop the RX processing loop."""
        self._running = False
        self._pause_event.set()
        self._flush_buffer()
        self._sdr.stop_rx()
        log.info("RX stopped")

    def pause(self):
        """Pause RX processing (for TX)."""
        if self._paused:
            return
        self._paused = True
        self._pause_event.clear()
        self._flush_buffer()
        self._sdr.stop_rx()
        log.debug("RX paused for TX")

    def resume(self):
        """Resume RX processing after TX."""
        if not self._paused:
            return
        self._sdr.start_rx()
        self._demod.reset()
        self._paused = False
        self._pause_event.set()
        log.debug("RX resumed")

    def _process_block(self, iq: np.ndarray):
        """Process a block of IQ samples through squelch and demod."""
        iq_power = np.mean(np.abs(iq) ** 2)
        squelch_open = bool(iq_power >= SQUELCH_THRESHOLD)

        if self._bridge:
            self._bridge.emit_iq(iq)
            self._bridge.emit_squelch(squelch_open, float(iq_power))

        if not squelch_open:
            self._flush_buffer()
            return

        audio = self._demod.process(iq)
        self._audio_buffer.append(audio)

    def _flush_buffer(self):
        """Flush audio buffer to transcription."""
        if not self._audio_buffer:
            return

        full_audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []
        self._demod.reset()

        transcribe_audio(full_audio)
