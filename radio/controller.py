"""
Radio controller for half-duplex operation.

Manages RX/TX switching to prevent simultaneous receive and transmit.
"""

import logging
import threading

import numpy as np
from config import AUDIO_RATE
from radio.sdr import SDRDevice
from radio.rx import RXProcessor
from radio.tx import TXProcessor

log = logging.getLogger("warden.radio")


class RadioController:
    """
    Coordinates RX and TX for half-duplex operation.

    Ensures RX is paused while transmitting and resumed after.
    """

    def __init__(self, sdr: SDRDevice):
        self._sdr = sdr
        self._rx = RXProcessor(sdr)
        self._tx = TXProcessor(sdr)
        self._tx_lock = threading.Lock()

    @property
    def rx(self) -> RXProcessor:
        return self._rx

    @property
    def tx(self) -> TXProcessor:
        return self._tx

    def transmit(self, audio: np.ndarray, lead_in: float = 0.1, lead_out: float = 0.5):
        """
        Transmit audio, pausing RX for the duration.

        Args:
            audio: Float32 audio samples at AUDIO_RATE.
            lead_in: Seconds of CTCSS-only before voice.
            lead_out: Seconds of CTCSS-only after voice.
        """
        with self._tx_lock:
            log.info("Switching to TX")
            self._rx.pause()

            try:
                self._tx.transmit(audio, lead_in=lead_in, lead_out=lead_out)
            finally:
                log.info("Switching to RX")
                self._rx.resume()

    def start_rx(self):
        """Start the RX processing loop (blocks until stopped)."""
        self._rx.start()

    def stop(self):
        """Stop all radio operations."""
        self._rx.stop()
