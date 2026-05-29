"""
Warden — SDR dispatch pipeline.

Receives FM transmissions via HackRF, demodulates and transcribes in real-time,
and dispatches responses when a recognized callsign is detected.

Launch with --gui flag for the desktop interface, or without for headless mode.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import logging
import sys
import threading

from config import OPENAI_API_KEY
from audio.player import audio_worker, audio_queue
from radio.sdr import SDRDevice
from radio.controller import RadioController
from audio.tts import (
    set_bridge as set_tts_bridge,
    set_radio_controller,
    preload_voice,
)
from transcription import preload_model as preload_whisper

log = logging.getLogger("warden")


def start_model_preload():
    """Warm up Whisper + Piper on background threads so the GUI is responsive
    immediately and the first transcription / TTS doesn't pay the load cost."""
    threading.Thread(
        target=preload_whisper, daemon=True, name="preload-whisper"
    ).start()
    threading.Thread(
        target=preload_voice, daemon=True, name="preload-tts"
    ).start()


class GuiLogHandler(logging.Handler):
    """Forward Python log records into the Qt GUI bridge."""

    def __init__(self, bridge):
        super().__init__()
        self._bridge = bridge

    def emit(self, record):
        try:
            self._bridge.emit_log(self.format(record))
        except Exception:
            self.handleError(record)


def run_headless(sdr: SDRDevice):
    """Run Warden in headless (no GUI) mode."""
    radio = RadioController(sdr)
    set_radio_controller(radio)

    try:
        radio.start_rx()
    except KeyboardInterrupt:
        log.info("Shutdown requested")
        radio.stop()


def run_gui(sdr: SDRDevice):
    """Run Warden with the PySide6 desktop GUI."""
    from PySide6.QtWidgets import QApplication
    from gui.bridge import StateBridge
    from gui.app import WardenWindow
    from dispatch.preamble import set_bridge

    app = QApplication(sys.argv)
    app.setApplicationName("Warden")

    bridge = StateBridge()
    set_bridge(bridge)
    set_tts_bridge(bridge)

    gui_log_handler = GuiLogHandler(bridge)
    gui_log_handler.setLevel(logging.INFO)
    gui_log_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(gui_log_handler)

    radio = RadioController(sdr, bridge=bridge)
    set_radio_controller(radio)

    window = WardenWindow(bridge, sdr=sdr, radio=radio)
    window.show()

    rx_thread = threading.Thread(target=radio.start_rx, daemon=True)
    rx_thread.start()

    exit_code = app.exec()

    radio.stop()
    return exit_code


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    use_gui = "--gui" in sys.argv

    log.info("Starting Warden%s...", " (GUI)" if use_gui else "")
    if not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY is empty — check .env in the project root")

    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    start_model_preload()

    sdr = SDRDevice()
    if not sdr.open():
        log.error("Failed to open SDR device")
        return 1

    try:
        if use_gui:
            return run_gui(sdr)
        else:
            run_headless(sdr)
            return 0
    finally:
        sdr.close()
        log.info("Exited")


if __name__ == "__main__":
    raise SystemExit(main())
