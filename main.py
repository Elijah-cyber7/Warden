import threading
import queue
import numpy as np
import sounddevice as sd
import SoapySDR
import vosk
from scipy.signal import firwin, lfilter, resample_poly, butter, sosfilt


# config
SAMPLE_RATE = 2.4e6
CENTER_FREQ = 89.300e6   #462.550e6
AUDIO_RATE = 48000
GAIN = 78
BUFF_SIZE = 1024 * 256
CHANNEL_BW = 200000 #12500

# filters built once at startup
channel_filter = firwin(128, cutoff=CHANNEL_BW / SAMPLE_RATE)
audio_filter = firwin(64, cutoff=0.02)

#highpass = butter(5, 300 / (AUDIO_RATE / 2), btype='high', output='sos')

bandpass = butter(1, [300 / (AUDIO_RATE / 2), 3400 / (AUDIO_RATE / 2)], btype='band', output='sos')

# squelch threshold — adjust this value based on your noise floor
SQUELCH = 0.01

# audio queue between RX thread and audio thread
audio_queue = queue.Queue()


def init_sdr():
    results = SoapySDR.Device.enumerate()
    print(f"Found devices: {results}")
    sdr = SoapySDR.Device(results[0])
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, CENTER_FREQ)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, GAIN)
    return sdr


def process_iq(iq):
    # filter IQ to channel bandwidth
    I = lfilter(channel_filter, 1.0, iq.real)
    Q = lfilter(channel_filter, 1.0, iq.imag)
    iq_filtered = I + 1j * Q

    # FM demod
    conj = iq_filtered[:-1] * np.conj(iq_filtered[1:])
    demodulated = np.angle(conj)

    # resample to audio rate
    audio = resample_poly(demodulated, 1, int(SAMPLE_RATE / AUDIO_RATE))

    # squelch — measure signal power, gate if below threshold
    power = np.sqrt(np.mean(audio ** 2))
    if power < SQUELCH:
        return  # silence, don't play

    # strip CTCSS tone with highpass filter
    #audio = sosfilt(highpass, audio)
    audio = sosfilt(bandpass, audio)

    # normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    audio_queue.put(audio.astype(np.float32))

def rx_loop(sdr):
    rx_stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(rx_stream)
    buf = np.zeros(BUFF_SIZE, dtype=np.complex64)

    print("Listening on 462.550 MHz channel 10...")
    while True:
        sr = sdr.readStream(rx_stream, [buf], len(buf))
        if sr.ret > 0:
            process_iq(buf[:sr.ret].copy())


def audio_worker():
    with sd.OutputStream(samplerate=AUDIO_RATE, channels=1, dtype='float32') as stream:
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            stream.write(audio.reshape(-1, 1))


if __name__ == "__main__":
    # start audio output thread
    t = threading.Thread(target=audio_worker, daemon=True)
    t.start()

    # init hardware and start receiving
    sdr = init_sdr()
    rx_loop(sdr)