import threading
import signal
import sys
from audio.player import audio_worker, audio_queue
from radio.sdr import init_sdr, rx_loop

_shutdown = False


def signal_handler(sig, frame):
    global _shutdown
    print("\n[SHUTDOWN] Ctrl+C received, cleaning up...")
    _shutdown = True
    audio_queue.put(None)  # signal audio thread to exit
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    # start audio output thread
    t = threading.Thread(target=audio_worker, daemon=True)
    t.start()

    # init hardware and start receiving
    sdr = init_sdr()
    rx_loop(sdr)