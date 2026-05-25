"""
RX processing loop for Warden.

Handles the receive pipeline: SDR -> demod -> audio -> transcription.
"""

import threading
import numpy as np
from config import SQUELCH_THRESHOLD
from radio.sdr import SDRDevice
from radio.demod import FMDemodulator
from transcription.whisper_engine import transcribe_audio


class RXProcessor:
    """
    Receive processor that handles the full RX pipeline.
    
    Manages squelch, audio buffering, and transcription triggering.
    Supports pause/resume for half-duplex TX operation.
    """
    
    def __init__(self, sdr: SDRDevice):
        self._sdr = sdr
        self._demod = FMDemodulator()
        self._audio_buffer = []
        self._running = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused
    
    def start(self):
        """Start the RX processing loop."""
        self._sdr.start_rx()
        self._running = True
        print(f"[RX] Started, squelch threshold={SQUELCH_THRESHOLD}")
        
        while self._running:
            # Wait if paused (for TX)
            self._pause_event.wait()
            
            if not self._running:
                break
                
            iq = self._sdr.read_rx()
            if iq is not None and not self._paused:
                self._process_block(iq)
    
    def stop(self):
        """Stop the RX processing loop."""
        self._running = False
        self._pause_event.set()  # Unblock if paused
        self._flush_buffer()
        self._sdr.stop_rx()
        print("[RX] Stopped")
    
    def pause(self):
        """Pause RX processing (for TX)."""
        if self._paused:
            return
        self._paused = True
        self._pause_event.clear()
        self._flush_buffer()
        self._sdr.stop_rx()
        print("[RX] Paused for TX")
    
    def resume(self):
        """Resume RX processing after TX."""
        if not self._paused:
            return
        self._sdr.start_rx()
        self._demod.reset()
        self._paused = False
        self._pause_event.set()
        print("[RX] Resumed")
    
    def _process_block(self, iq: np.ndarray):
        """Process a block of IQ samples."""
        iq_power = np.mean(np.abs(iq) ** 2)
        
        if iq_power < SQUELCH_THRESHOLD:
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
