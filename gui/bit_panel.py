"""
BIT (Built-In Test) panel for the Warden GUI.

Provides a "Run BIT" button, test selection checkboxes, and a color-coded
results table showing pass/fail status, measurements, and thresholds.
"""

import logging
from datetime import datetime

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtGui import QFont, QColor

from bit.runner import BITRunner

log = logging.getLogger("warden.gui.bit")


TESTS = [
    ("usb_health", "USB Health (transmits)"),
    ("fifo_integrity", "FIFO Integrity (transmits)"),
]

STATUS_COLORS = {
    "PASS": QColor(0, 200, 80),
    "FAIL": QColor(255, 60, 60),
    "running": QColor(255, 200, 0),
    "pending": QColor(136, 136, 136),
}

THRESHOLDS = {
    "usb_health": ">= 3.5 MB/s, < 1% err",
    "fifo_integrity": "0 stalls, 0 sample loss",
}


class BITPanel(QWidget):
    """Built-In Test panel with run controls and results display."""

    def __init__(self, sdr=None, radio=None, bridge=None, parent=None):
        super().__init__(parent)
        self._sdr = sdr
        self._radio = radio
        self._bridge = bridge
        self._runner = BITRunner(sdr, radio=radio, bridge=bridge) if sdr else None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Built-In Test")
        header.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        header.setStyleSheet("color: #cccccc;")
        layout.addWidget(header)

        # Test selection checkboxes
        select_group = QGroupBox("Tests")
        select_layout = QVBoxLayout(select_group)

        self._checkboxes: dict[str, QCheckBox] = {}
        for test_id, test_label in TESTS:
            cb = QCheckBox(test_label)
            cb.setChecked(False)
            cb.setStyleSheet("color: #cccccc;")
            self._checkboxes[test_id] = cb
            select_layout.addWidget(cb)

        layout.addWidget(select_group)

        # Run button
        btn_layout = QHBoxLayout()
        self._run_btn = QPushButton("Run BIT")
        self._run_btn.setStyleSheet(
            "QPushButton { background-color: #2a82da; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 3px; }"
            "QPushButton:disabled { background-color: #555555; color: #888888; }"
        )
        self._run_btn.clicked.connect(self._on_run_clicked)
        btn_layout.addWidget(self._run_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Last run timestamp
        self._timestamp_label = QLabel("")
        self._timestamp_label.setFont(QFont("Menlo", 9))
        self._timestamp_label.setStyleSheet("color: #888888;")
        layout.addWidget(self._timestamp_label)

        # Results table
        self._table = QTableWidget(len(TESTS), 4)
        self._table.setHorizontalHeaderLabels(["Test", "Status", "Measurement", "Threshold"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setFont(QFont("Menlo", 10))
        self._table.setStyleSheet(
            "QTableWidget { background-color: #1e1e1e; color: #cccccc; "
            "gridline-color: #333333; border: none; }"
            "QHeaderView::section { background-color: #2a2a2a; color: #cccccc; "
            "border: 1px solid #333333; padding: 4px; }"
        )

        for row, (test_id, test_label) in enumerate(TESTS):
            self._table.setItem(row, 0, QTableWidgetItem(test_label))
            self._table.setItem(row, 1, QTableWidgetItem("—"))
            self._table.setItem(row, 2, QTableWidgetItem("—"))
            self._table.setItem(row, 3, QTableWidgetItem(THRESHOLDS[test_id]))

        layout.addWidget(self._table)
        layout.addStretch()

        # Connect bridge signal if available
        if self._bridge:
            self._bridge.bit_result_ready.connect(self._on_result)

    def _on_run_clicked(self):
        if self._runner is None:
            log.warning("BIT: no SDR device available")
            return

        if self._runner.is_running:
            return

        selected = [tid for tid, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            log.warning("BIT: no tests selected")
            return

        self._run_btn.setEnabled(False)
        self._timestamp_label.setText(f"Started: {datetime.now().strftime('%H:%M:%S')}")

        for row, (test_id, _) in enumerate(TESTS):
            if test_id in selected:
                self._set_row_status(row, "pending", "—", "")
            else:
                self._set_row_status(row, "pending", "skipped", "")

        self._runner.run_tests(selected, on_complete=self._on_bit_complete)

    def _on_bit_complete(self):
        self._run_btn.setEnabled(True)
        self._timestamp_label.setText(
            f"Completed: {datetime.now().strftime('%H:%M:%S')}"
        )

    @Slot(str, str, str, str)
    def _on_result(self, test_name: str, status: str, measurement: str, detail: str):
        """Handle a BIT result signal from the bridge."""
        for row, (test_id, _) in enumerate(TESTS):
            if test_id == test_name:
                self._set_row_status(row, status, measurement, detail)
                break

    def _set_row_status(self, row: int, status: str, measurement: str, detail: str):
        """Update a table row with status coloring."""
        color = STATUS_COLORS.get(status, QColor(136, 136, 136))

        status_item = QTableWidgetItem(status)
        status_item.setForeground(color)
        self._table.setItem(row, 1, status_item)

        meas_item = QTableWidgetItem(measurement)
        meas_item.setForeground(color)
        self._table.setItem(row, 2, meas_item)

        if detail:
            self._table.setItem(row, 3, QTableWidgetItem(detail))
