"""
State bridge between radio backend threads and the Qt GUI.

All cross-thread communication goes through this object's Qt signals,
which are safe to connect to GUI widgets from any thread.
"""

import numpy as np
from PySide6.QtCore import QObject, Signal


class StateBridge(QObject):
    """
    Thread-safe bridge that emits Qt signals when radio state changes.

    Backend threads call the emit_* methods; GUI widgets connect to the signals.
    """

    iq_ready = Signal(np.ndarray)
    squelch_changed = Signal(bool, float)
    tx_state_changed = Signal(bool)
    transcription_ready = Signal(str, bool)
    log_ready = Signal(str)
    bit_result_ready = Signal(str, str, str, str)  # test_name, status, measurement, detail

    def __init__(self, parent=None):
        super().__init__(parent)

    def emit_iq(self, iq: np.ndarray):
        """Called from RX thread with raw IQ samples for spectrum display."""
        self.iq_ready.emit(iq)

    def emit_squelch(self, is_open: bool, power: float):
        """Called from RX thread when squelch state or power level changes."""
        self.squelch_changed.emit(is_open, power)

    def emit_tx_state(self, is_transmitting: bool):
        """Called from TX thread on start/stop."""
        self.tx_state_changed.emit(is_transmitting)

    def emit_transcription(self, text: str, matched: bool):
        """Called from whisper thread with transcription results."""
        self.transcription_ready.emit(text, matched)

    def emit_log(self, text: str):
        """Called when a log record should be shown in the GUI."""
        self.log_ready.emit(text)

    def emit_bit_result(self, test_name: str, status: str, measurement: str, detail: str):
        """Called from BIT runner thread with test results."""
        self.bit_result_ready.emit(test_name, status, measurement, detail)
