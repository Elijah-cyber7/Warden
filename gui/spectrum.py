"""
Real-time PSD (Power Spectral Density) plot widget.

Displays frequency vs. power as a live-updating line graph using pyqtgraph.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from config import SAMPLE_RATE


class SpectrumWidget(QWidget):
    """Real-time power spectral density line plot."""

    FFT_SIZE = 2048

    def __init__(self, parent=None):
        super().__init__(parent)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("bottom", "Sample bandwidth", units="kHz")
        self._plot_widget.setLabel("left", "Power", units="dB")
        self._plot_widget.setTitle("Spectrum")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setYRange(-80, 0)

        self._curve = self._plot_widget.plot(pen=pg.mkPen("c", width=1))

        freq_axis = np.linspace(
            0.0,
            SAMPLE_RATE / 1e3,
            self.FFT_SIZE
        )
        self._freq_axis = freq_axis

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget)

    def update_spectrum(self, iq: np.ndarray):
        """Compute PSD from IQ samples and update the plot."""
        if len(iq) < self.FFT_SIZE:
            return

        segment = iq[:self.FFT_SIZE]
        window = np.hanning(self.FFT_SIZE)
        fft = np.fft.fftshift(np.fft.fft(segment * window))
        psd = 20 * np.log10(np.abs(fft) + 1e-12)

        self._curve.setData(self._freq_axis, psd)
