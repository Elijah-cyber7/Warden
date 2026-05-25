"""
Status bar widget showing radio state at a glance.

Displays RX/TX mode, center frequency, and squelch power level.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtGui import QFont
from config import CENTER_FREQ


class StatusBar(QWidget):
    """Horizontal status bar with radio state indicators."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        mono = QFont("Menlo", 12)

        self._mode_label = QLabel("RX")
        self._mode_label.setFont(mono)
        self._mode_label.setStyleSheet("color: #00ff88; font-weight: bold;")
        layout.addWidget(self._mode_label)

        layout.addSpacing(16)

        self._freq_label = QLabel(f"{CENTER_FREQ / 1e6:.5f} MHz")
        self._freq_label.setFont(mono)
        self._freq_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self._freq_label)

        layout.addStretch()

        self._squelch_label = QLabel("Sq: CLOSED")
        self._squelch_label.setFont(mono)
        self._squelch_label.setStyleSheet("color: #888888;")
        layout.addWidget(self._squelch_label)

        layout.addSpacing(16)

        self._power_label = QLabel("Pwr: —")
        self._power_label.setFont(mono)
        self._power_label.setStyleSheet("color: #888888;")
        layout.addWidget(self._power_label)

    def set_mode(self, is_transmitting: bool):
        """Update RX/TX indicator."""
        if is_transmitting:
            self._mode_label.setText("TX")
            self._mode_label.setStyleSheet("color: #ff4444; font-weight: bold;")
        else:
            self._mode_label.setText("RX")
            self._mode_label.setStyleSheet("color: #00ff88; font-weight: bold;")

    def set_squelch(self, is_open: bool, power: float):
        """Update squelch state and power display."""
        if is_open:
            self._squelch_label.setText("Sq: OPEN")
            self._squelch_label.setStyleSheet("color: #ffcc00;")
        else:
            self._squelch_label.setText("Sq: CLOSED")
            self._squelch_label.setStyleSheet("color: #888888;")

        self._power_label.setText(f"Pwr: {power:.2f}")

    def set_frequency(self, freq_hz: float):
        """Update displayed frequency."""
        self._freq_label.setText(f"{freq_hz / 1e6:.5f} MHz")
