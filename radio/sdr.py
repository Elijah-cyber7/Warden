import numpy as np
import SoapySDR
from config import SAMPLE_RATE, CENTER_FREQ, GAIN, BUFF_SIZE
from radio.demod import process_iq


def init_sdr():
    results = SoapySDR.Device.enumerate()
    print(f"Found devices: {results}")
    sdr = SoapySDR.Device(results[0])
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, CENTER_FREQ)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, GAIN)
    return sdr


def rx_loop(sdr):
    rx_stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(rx_stream)
    buf = np.zeros(BUFF_SIZE, dtype=np.complex64)

    print(f"Listening on {CENTER_FREQ / 1e6:.4f} MHz...")
    while True:
        sr = sdr.readStream(rx_stream, [buf], len(buf))
        if sr.ret > 0:
            process_iq(buf[:sr.ret].copy())