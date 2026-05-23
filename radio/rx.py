"""
RX processing loop for Warden.

Handles the receive pipeline: SDR -> demod -> audio -> transcription.
"""

import numpy as np
import scipy.io.wavfile as wav
from config import AUDIO_RATE, SQUELCH_THRESHOLD
from radio.sdr import SDRDevice
from radio.demod import FMDemodulator
from audio.player import audio_queue
from transcription.whisper_engine import transcribe_audio


class RXProcessor:
    """
    Receive processor that handles the full RX pipeline.
    
    Manages squelch, audio buffering, and transcription triggering.
    """
    
    def __init__(self, sdr: SDRDevice):
        self._sdr = sdr
        self._demod = FMDemodulator()
        self._audio_buffer = []
        self._running = False
    
    def start(self):
        """Start the RX processing loop."""
        self._sdr.start_rx()
        self._running = True
        print(f"[RX] Started, squelch threshold={SQUELCH_THRESHOLD}")
        
        while self._running:
            iq = self._sdr.read_rx()
            if iq is not None:
                self._process_block(iq)
    
    def stop(self):
        """Stop the RX processing loop."""
        self._running = False
        self._flush_buffer()
        self._sdr.stop_rx()
        print("[RX] Stopped")
    
    def _process_block(self, iq: np.ndarray):
        """Process a block of IQ samples."""
        iq_power = np.mean(np.abs(iq) ** 2)
        
        if iq_power < SQUELCH_THRESHOLD:
            self._flush_buffer()
            return
        
        audio = self._demod.process(iq)
        audio_queue.put(audio)
        self._audio_buffer.append(audio)
    
    def _flush_buffer(self):
        """Flush audio buffer to transcription."""
        if not self._audio_buffer:
            return
        
        full_audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []
        
        self._demod.reset()
        
        wav.write('debug.wav', AUDIO_RATE, (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16))
        transcribe_audio(full_audio)
