"""
Configuration panel for live parameter adjustment.

Provides sliders and spinboxes to adjust SDR gains, frequency,
CTCSS settings, and squelch threshold at runtime.
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel,
    QSlider, QDoubleSpinBox, QSpinBox, QGroupBox
)
from PySide6.QtGui import QFont

import config

log = logging.getLogger("warden.gui.config")


class ConfigPanel(QWidget):
    """Live configuration panel with gain/frequency/squelch controls."""

    def __init__(self, sdr=None, parent=None):
        super().__init__(parent)
        self._sdr = sdr

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Configuration")
        header.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        header.setStyleSheet("color: #cccccc;")
        layout.addWidget(header)

        # --- RX Gains ---
        rx_group = QGroupBox("RX Gains")
        rx_form = QFormLayout(rx_group)

        self._lna_slider = self._make_slider(0, 40, 8, int(config.RX_LNA_GAIN))
        self._lna_slider.valueChanged.connect(self._on_lna_changed)
        rx_form.addRow("LNA (dB):", self._lna_slider)

        self._vga_slider = self._make_slider(0, 62, 2, int(config.RX_VGA_GAIN))
        self._vga_slider.valueChanged.connect(self._on_vga_changed)
        rx_form.addRow("VGA (dB):", self._vga_slider)

        layout.addWidget(rx_group)

        # --- TX Gains ---
        tx_group = QGroupBox("TX Gain")
        tx_form = QFormLayout(tx_group)

        self._tx_vga_slider = self._make_slider(0, 47, 1, int(config.TX_VGA_GAIN))
        self._tx_vga_slider.valueChanged.connect(self._on_tx_vga_changed)
        tx_form.addRow("VGA (dB):", self._tx_vga_slider)

        layout.addWidget(tx_group)

        # --- Frequency ---
        freq_group = QGroupBox("Frequency")
        freq_form = QFormLayout(freq_group)

        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(400.0, 6000.0)
        self._freq_spin.setDecimals(5)
        self._freq_spin.setSuffix(" MHz")
        self._freq_spin.setValue(config.CENTER_FREQ / 1e6)
        self._freq_spin.valueChanged.connect(self._on_freq_changed)
        freq_form.addRow("Center:", self._freq_spin)

        layout.addWidget(freq_group)

        # --- CTCSS ---
        ctcss_group = QGroupBox("CTCSS")
        ctcss_form = QFormLayout(ctcss_group)

        self._ctcss_freq_spin = QDoubleSpinBox()
        self._ctcss_freq_spin.setRange(67.0, 254.0)
        self._ctcss_freq_spin.setDecimals(1)
        self._ctcss_freq_spin.setSuffix(" Hz")
        self._ctcss_freq_spin.setValue(config.CTCSS_FREQ)
        self._ctcss_freq_spin.valueChanged.connect(self._on_ctcss_freq_changed)
        ctcss_form.addRow("Tone:", self._ctcss_freq_spin)

        self._ctcss_level_spin = QDoubleSpinBox()
        self._ctcss_level_spin.setRange(0.01, 0.50)
        self._ctcss_level_spin.setDecimals(3)
        self._ctcss_level_spin.setSingleStep(0.01)
        self._ctcss_level_spin.setValue(config.CTCSS_LEVEL)
        self._ctcss_level_spin.valueChanged.connect(self._on_ctcss_level_changed)
        ctcss_form.addRow("Level:", self._ctcss_level_spin)

        layout.addWidget(ctcss_group)

        # --- Squelch ---
        sq_group = QGroupBox("Squelch")
        sq_form = QFormLayout(sq_group)

        self._squelch_spin = QDoubleSpinBox()
        self._squelch_spin.setRange(0.0, 50.0)
        self._squelch_spin.setDecimals(2)
        self._squelch_spin.setSingleStep(0.1)
        self._squelch_spin.setValue(config.SQUELCH_THRESHOLD)
        self._squelch_spin.valueChanged.connect(self._on_squelch_changed)
        sq_form.addRow("Threshold:", self._squelch_spin)

        layout.addWidget(sq_group)

        layout.addStretch()

    def _make_slider(self, min_val: int, max_val: int, step: int, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setSingleStep(step)
        slider.setValue(value)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(step * 2)
        return slider

    def _on_lna_changed(self, value: int):
        config.RX_LNA_GAIN = value
        log.info("RX LNA gain → %d dB", value)
        if self._sdr:
            self._sdr._set_rx_gains()

    def _on_vga_changed(self, value: int):
        config.RX_VGA_GAIN = value
        log.info("RX VGA gain → %d dB", value)
        if self._sdr:
            self._sdr._set_rx_gains()

    def _on_tx_vga_changed(self, value: int):
        config.TX_VGA_GAIN = value
        log.info("TX VGA gain → %d dB", value)
        if self._sdr:
            self._sdr._set_tx_gains()

    def _on_freq_changed(self, value: float):
        config.CENTER_FREQ = value * 1e6
        log.info("Center freq → %.5f MHz", value)

    def _on_ctcss_freq_changed(self, value: float):
        config.CTCSS_FREQ = value
        log.info("CTCSS freq → %.1f Hz", value)

    def _on_ctcss_level_changed(self, value: float):
        config.CTCSS_LEVEL = value
        log.info("CTCSS level → %.3f", value)

    def _on_squelch_changed(self, value: float):
        config.SQUELCH_THRESHOLD = value
        log.info("Squelch threshold → %.2f", value)
