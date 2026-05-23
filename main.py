import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import threading
from audio.player import audio_worker, audio_queue
from radio.sdr import SDRDevice
from radio.rx import RXProcessor


def main():
    print("[WARDEN] Starting...")
    
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()
    
    sdr = SDRDevice()
    if not sdr.open():
        print("[WARDEN] Failed to open SDR device")
        return 1
    
    rx = RXProcessor(sdr)
    
    try:
        rx.start()
    except KeyboardInterrupt:
        print("\n[WARDEN] Shutdown requested...")
        rx.stop()
    finally:
        sdr.close()
    
    print("[WARDEN] Exited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())