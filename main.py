import threading
from audio.player import audio_worker
from radio.sdr import init_sdr, rx_loop

if __name__ == "__main__":
    # start audio output thread
    t = threading.Thread(target=audio_worker, daemon=True)
    t.start()

    # init hardware and start receiving
    sdr = init_sdr()
    rx_loop(sdr)