"""
Scrolling transcription log panel.

Shows timestamped transcription results with visual distinction
between preamble-matched and unmatched entries.
"""

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtGui import QFont, QTextCursor, QColor


class TranscriptPanel(QWidget):
    """Scrolling log of transcription results."""

    MAX_ENTRIES = 200

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Radio Log")
        header.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        header.setStyleSheet("color: #cccccc;")
        layout.addWidget(header)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Menlo", 10))
        self._log.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #cccccc; border: none; }"
        )
        layout.addWidget(self._log)

        self._entry_count = 0

    def add_entry(self, text: str, matched: bool):
        """Add a timestamped transcription entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if text.startswith("Warden:"):
            color = "#66ccff"
            prefix = "<<<"
        elif matched:
            color = "#00ff88"
            prefix = ">>>"
        else:
            color = "#888888"
            prefix = "   "

        safe_text = text.replace("\n", " ").strip()
        line = f"{timestamp} {prefix} {safe_text}"

        self._log.moveCursor(QTextCursor.MoveOperation.End)
        fmt = self._log.currentCharFormat()
        fmt.setForeground(QColor(color))
        self._log.setCurrentCharFormat(fmt)
        self._log.insertPlainText(line + "\n")
        self._log.moveCursor(QTextCursor.MoveOperation.End)

        self._entry_count += 1
        if self._entry_count > self.MAX_ENTRIES:
            self._trim_old_entries()

    def _trim_old_entries(self):
        """Remove oldest entries to keep memory bounded."""
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor, 50)
        cursor.removeSelectedText()
        self._entry_count -= 50
