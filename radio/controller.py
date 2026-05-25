"""
Radio controller for half-duplex operation.

Manages RX/TX switching to prevent simultaneous receive and transmit.
"""

import threading
import numpy as np
from config import AUDIO_RATE
from radio.sdr import SDRDevice
from radio.rx import RXProcessor
from radio.tx import TXProcessor


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
        self._rx_paused = threading.Event()
        self._rx_paused.set()  # Start unpaused
    
    @property
    def rx(self) -> RXProcessor:
        return self._rx
    
    @property
    def tx(self) -> TXProcessor:
        return self._tx
    
    def transmit(self, audio: np.ndarray, lead_in: float = 0.1, lead_out: float = 0.1):
        """
        Transmit audio, pausing RX for the duration.
        
        Args:
            audio: Float32 audio samples at AUDIO_RATE.
            lead_in: Seconds of CTCSS-only before voice.
            lead_out: Seconds of CTCSS-only after voice.
        """
        with self._tx_lock:
            print("[RADIO] Switching to TX mode")
            
            # Pause RX
            self._rx_paused.clear()
            self._rx.pause()
            
            try:
                self._tx.transmit(audio, lead_in=lead_in, lead_out=lead_out)
            finally:
                # Resume RX
                print("[RADIO] Switching to RX mode")
                self._rx.resume()
                self._rx_paused.set()
    
    def start_rx(self):
        """Start the RX processing loop."""
        self._rx.start()
    
    def stop(self):
        """Stop all radio operations."""
        self._rx.stop()
