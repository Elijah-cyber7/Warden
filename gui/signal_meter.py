"""
Horizontal dB signal-level gauge.

Shows the current RX power as a colored bar with a numeric dB readout and a
marker line for the squelch threshold. Wired to the StateBridge's
squelch_changed signal — same data the StatusBar uses, just visualised.
"""

import math

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QLinearGradient
from PySide6.QtWidgets import QWidget

from config import SQUELCH_THRESHOLD


def _linear_to_db(power_linear: float) -> float:
    if power_linear <= 0.0:
        return -120.0
    return 10.0 * math.log10(power_linear)


class SignalMeter(QWidget):
    """Horizontal dB gauge with squelch threshold marker."""

    DB_MIN = -20.0
    DB_MAX = 40.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setMaximumHeight(56)

        self._level_db: float = self.DB_MIN
        self._peak_db: float = self.DB_MIN
        self._squelch_db: float = _linear_to_db(SQUELCH_THRESHOLD)
        self._squelch_open: bool = False
        self._transmitting: bool = False

    def set_level(self, is_open: bool, power_linear: float):
        """Update from the bridge's squelch_changed signal (is_open, power)."""
        self._squelch_open = is_open
        self._level_db = _linear_to_db(power_linear)
        # Slow-decay peak hold (~30 dB/s at the default RX block rate)
        decay = 0.5
        self._peak_db = max(self._level_db, self._peak_db - decay)
        # Re-read squelch threshold each call so config-panel edits propagate
        import config
        self._squelch_db = _linear_to_db(config.SQUELCH_THRESHOLD)
        self.update()

    def set_transmitting(self, is_tx: bool):
        """Switch to TX color scheme while transmitting."""
        self._transmitting = is_tx
        self.update()

    def _db_to_x(self, db: float, x0: int, w: int) -> int:
        frac = (db - self.DB_MIN) / (self.DB_MAX - self.DB_MIN)
        frac = max(0.0, min(1.0, frac))
        return int(x0 + w * frac)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        readout_w = 76
        margin_x = 8
        margin_y = 8
        bar_x = margin_x + readout_w
        bar_y = margin_y
        bar_w = max(0, w - bar_x - margin_x)
        bar_h = h - 2 * margin_y

        # --- numeric readout ---
        painter.setPen(QColor(204, 204, 204))
        painter.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
        readout_rect = QRectF(margin_x, 0, readout_w - 6, h)
        painter.drawText(
            readout_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            f"{self._level_db:5.1f} dB",
        )

        # --- bar background ---
        bg_rect = QRectF(bar_x, bar_y, bar_w, bar_h)
        painter.fillRect(bg_rect, QColor(20, 20, 20))
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(bg_rect)

        # --- gradient fill up to current level ---
        level_x = self._db_to_x(self._level_db, bar_x, bar_w)
        fill_w = level_x - bar_x
        if fill_w > 0:
            grad = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
            if self._transmitting:
                grad.setColorAt(0.0, QColor(180, 60, 60))
                grad.setColorAt(1.0, QColor(255, 80, 80))
            elif self._squelch_open:
                grad.setColorAt(0.0, QColor(0, 160, 90))
                grad.setColorAt(1.0, QColor(0, 255, 136))
            else:
                grad.setColorAt(0.0, QColor(40, 90, 160))
                grad.setColorAt(1.0, QColor(70, 140, 220))
            painter.fillRect(QRectF(bar_x, bar_y, fill_w, bar_h), grad)

        # --- peak-hold tick ---
        if self._peak_db > self.DB_MIN:
            peak_x = self._db_to_x(self._peak_db, bar_x, bar_w)
            painter.setPen(QPen(QColor(255, 255, 255, 160), 2))
            painter.drawLine(peak_x, bar_y, peak_x, bar_y + bar_h)

        # --- squelch threshold marker ---
        sq_x = self._db_to_x(self._squelch_db, bar_x, bar_w)
        painter.setPen(QPen(QColor(255, 200, 0), 2, Qt.PenStyle.DashLine))
        painter.drawLine(sq_x, bar_y - 2, sq_x, bar_y + bar_h + 2)

        # --- dB scale ticks every 10 dB ---
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.setFont(QFont("Menlo", 8))
        tick_db = int(math.ceil(self.DB_MIN / 10.0) * 10)
        while tick_db <= self.DB_MAX:
            x = self._db_to_x(tick_db, bar_x, bar_w)
            painter.drawLine(x, bar_y + bar_h, x, bar_y + bar_h + 3)
            label_rect = QRectF(x - 16, bar_y + bar_h + 2, 32, 12)
            # Only draw labels if they'd be inside the widget
            if label_rect.bottom() <= h:
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    f"{tick_db}",
                )
            tick_db += 10
