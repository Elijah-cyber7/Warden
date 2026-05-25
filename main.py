import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import threading
from config import OPENAI_API_KEY
from audio.player import audio_worker, audio_queue
from radio.sdr import SDRDevice
from radio.controller import RadioController
from audio.tts import set_radio_controller


def main():
    print("[WARDEN] Starting...")
    if not OPENAI_API_KEY:
        print("[WARDEN] Warning: OPENAI_API_KEY is empty — check .env in the project root")
    
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()
    
    sdr = SDRDevice()
    if not sdr.open():
        print("[WARDEN] Failed to open SDR device")
        return 1

    radio = RadioController(sdr)
    set_radio_controller(radio)
    
    try:
        radio.start_rx()
    except KeyboardInterrupt:
        print("\n[WARDEN] Shutdown requested...")
        radio.stop()
    finally:
        sdr.close()
    
    print("[WARDEN] Exited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())