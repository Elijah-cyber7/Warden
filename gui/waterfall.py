"""
Scrolling waterfall display widget.

Displays a time-vs-frequency intensity plot that scrolls downward as new
IQ data arrives. Uses pyqtgraph ImageItem for efficient rendering.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from config import SAMPLE_RATE, CENTER_FREQ


class WaterfallWidget(QWidget):
    """Scrolling spectrogram / waterfall display."""

    FFT_SIZE = 2048
    HISTORY_LINES = 200

    def __init__(self, parent=None):
        super().__init__(parent)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("bottom", "Frequency", units="MHz")
        self._plot_widget.setLabel("left", "Time")
        self._plot_widget.setTitle("Waterfall")

        self._image = pg.ImageItem()
        self._plot_widget.addItem(self._image)

        colormap = pg.colormap.get("viridis")
        self._image.setLookupTable(colormap.getLookupTable())

        self._data = np.full((self.HISTORY_LINES, self.FFT_SIZE), -80.0, dtype=np.float32)

        freq_min = (CENTER_FREQ - SAMPLE_RATE / 2) / 1e6
        freq_max = (CENTER_FREQ + SAMPLE_RATE / 2) / 1e6
        self._image.setRect(freq_min, 0, freq_max - freq_min, self.HISTORY_LINES)

        self._plot_widget.setXRange(freq_min, freq_max)
        self._plot_widget.setYRange(0, self.HISTORY_LINES)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget)

    def update_waterfall(self, iq: np.ndarray):
        """Compute one PSD line and scroll it into the waterfall."""
        if len(iq) < self.FFT_SIZE:
            return

        segment = iq[:self.FFT_SIZE]
        window = np.hanning(self.FFT_SIZE)
        fft = np.fft.fftshift(np.fft.fft(segment * window))
        psd = 20 * np.log10(np.abs(fft) + 1e-12)

        self._data = np.roll(self._data, -1, axis=0)
        self._data[-1, :] = psd

        self._image.setImage(self._data.T, autoLevels=False, levels=(-80, 0))
