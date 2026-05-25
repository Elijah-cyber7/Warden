"""
Warden — SDR dispatch pipeline.

Receives FM transmissions via HackRF, demodulates and transcribes in real-time,
and dispatches responses when a recognized callsign is detected.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import logging
import threading

from config import OPENAI_API_KEY
from audio.player import audio_worker, audio_queue
from radio.sdr import SDRDevice
from radio.controller import RadioController
from audio.tts import set_radio_controller

log = logging.getLogger("warden")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Starting Warden...")
    if not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY is empty — check .env in the project root")

    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    sdr = SDRDevice()
    if not sdr.open():
        log.error("Failed to open SDR device")
        return 1

    radio = RadioController(sdr)
    set_radio_controller(radio)

    try:
        radio.start_rx()
    except KeyboardInterrupt:
        log.info("Shutdown requested")
        radio.stop()
    finally:
        sdr.close()

    log.info("Exited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
