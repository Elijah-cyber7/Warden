"""
Main application window for Warden GUI.

Assembles all widgets into a cohesive layout and wires up the StateBridge signals.
"""

import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter
)
from PySide6.QtGui import QPalette, QColor

from gui.bridge import StateBridge
from gui.spectrum import SpectrumWidget
from gui.waterfall import WaterfallWidget
from gui.status_bar import StatusBar
from gui.transcript_panel import TranscriptPanel
from gui.config_panel import ConfigPanel

log = logging.getLogger("warden.gui")


class WardenWindow(QMainWindow):
    """Main application window."""

    def __init__(self, bridge: StateBridge, sdr=None, parent=None):
        super().__init__(parent)
        self._bridge = bridge

        self.setWindowTitle("Warden — SDR Dispatch")
        self.setMinimumSize(1100, 750)
        self._apply_dark_theme()

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # Status bar at top
        self._status_bar = StatusBar()
        root_layout.addWidget(self._status_bar)

        # Main vertical splitter: spectrum area / bottom panels
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: spectrum + waterfall stacked
        spectrum_container = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_container)
        spectrum_layout.setContentsMargins(0, 0, 0, 0)
        spectrum_layout.setSpacing(2)

        self._spectrum = SpectrumWidget()
        self._waterfall = WaterfallWidget()
        spectrum_layout.addWidget(self._spectrum, stretch=2)
        spectrum_layout.addWidget(self._waterfall, stretch=3)

        main_splitter.addWidget(spectrum_container)

        # Bottom: transcript log + config panel side by side
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._transcript = TranscriptPanel()
        self._config_panel = ConfigPanel(sdr=sdr)

        bottom_splitter.addWidget(self._transcript)
        bottom_splitter.addWidget(self._config_panel)
        bottom_splitter.setSizes([600, 300])

        main_splitter.addWidget(bottom_splitter)
        main_splitter.setSizes([450, 250])

        root_layout.addWidget(main_splitter)

        # Connect bridge signals to widgets
        self._bridge.iq_ready.connect(self._on_iq)
        self._bridge.squelch_changed.connect(self._status_bar.set_squelch)
        self._bridge.tx_state_changed.connect(self._status_bar.set_mode)
        self._bridge.transcription_ready.connect(self._transcript.add_entry)

    def _on_iq(self, iq):
        """Route IQ data to both spectrum and waterfall."""
        self._spectrum.update_spectrum(iq)
        self._waterfall.update_waterfall(iq)

    def _apply_dark_theme(self):
        """Apply a dark color palette to the application."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(204, 204, 204))
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.Text, QColor(204, 204, 204))
        palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(204, 204, 204))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
