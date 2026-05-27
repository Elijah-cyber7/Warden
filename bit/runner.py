"""
BIT Runner — orchestrates Built-In Test execution.

Runs selected tests sequentially on a background thread, emitting results
via the StateBridge so the GUI can update in real time.
"""

import logging
import threading
from typing import Callable

from bit import usb_health, fifo_integrity

log = logging.getLogger("warden.bit")


class BITRunner:
    """
    Orchestrates BIT execution on a background thread.

    Pauses RX before tests (some require TX access) and resumes after.
    """

    def __init__(self, sdr_device, radio=None, bridge=None):
        self._device = sdr_device
        self._radio = radio
        self._bridge = bridge
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def run_tests(self, tests: list[str], on_complete: Callable | None = None):
        """
        Launch BIT tests on a background thread.

        Args:
            tests: List of test names to run ("usb_health", "fifo_integrity").
            on_complete: Optional callback when all tests finish.
        """
        if self._running:
            log.warning("BIT already running")
            return

        thread = threading.Thread(
            target=self._run, args=(tests, on_complete),
            daemon=True, name="bit-runner",
        )
        thread.start()

    def _run(self, tests: list[str], on_complete: Callable | None):
        self._running = True
        log.info("BIT starting: %s", ", ".join(tests))

        if self._radio:
            self._radio._rx.pause()

        try:
            for test_name in tests:
                self._emit("running", test_name, "Running...", "")
                try:
                    result = self._execute_test(test_name)
                    status = "PASS" if result.passed else "FAIL"
                    self._emit(status, test_name, self._measurement(test_name, result), result.detail)
                except Exception as e:
                    log.exception("BIT test '%s' raised an exception", test_name)
                    self._emit("FAIL", test_name, "ERROR", str(e))
        finally:
            if self._radio:
                self._radio._rx.resume()
            self._running = False
            log.info("BIT complete")
            if on_complete:
                on_complete()

    def _execute_test(self, test_name: str):
        raw_device = self._device._device
        if test_name == "usb_health":
            return usb_health.run(raw_device)
        elif test_name == "fifo_integrity":
            return fifo_integrity.run(raw_device)
        else:
            raise ValueError(f"Unknown test: {test_name}")

    def _measurement(self, test_name: str, result) -> str:
        if test_name == "usb_health":
            return f"{result.throughput_mbs:.1f} MB/s"
        elif test_name == "fifo_integrity":
            return f"{result.stall_count} stalls"
        return ""

    def _emit(self, status: str, test_name: str, measurement: str, detail: str):
        if self._bridge:
            self._bridge.emit_bit_result(test_name, status, measurement, detail)
