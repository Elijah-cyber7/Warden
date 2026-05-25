"""
Scrolling application log panel.

Shows backend log messages inside the GUI without requiring the terminal.
"""

from html import escape

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtGui import QFont, QTextCursor


class LogPanel(QWidget):
    """Scrolling log of application events."""

    MAX_ENTRIES = 300

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Logs")
        header.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        header.setStyleSheet("color: #cccccc;")
        layout.addWidget(header)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Menlo", 9))
        self._log.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #cccccc; border: none; }"
        )
        layout.addWidget(self._log)

        self._entry_count = 0

    def add_log(self, text: str):
        """Add a formatted log message."""
        html = f'<span style="color:#bbbbbb">{escape(text)}</span><br>'
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.insertHtml(html)
        self._log.moveCursor(QTextCursor.MoveOperation.End)

        self._entry_count += 1
        if self._entry_count > self.MAX_ENTRIES:
            self._trim_old_entries()

    def _trim_old_entries(self):
        """Remove oldest entries to keep memory bounded."""
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor, 75)
        cursor.removeSelectedText()
        self._entry_count -= 75
