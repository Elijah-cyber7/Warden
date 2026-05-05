
import SoapySDR
import numpy as np

SAMPLE_RATE = 2.4e6
CENTER_FREQ = 462.550e6
GAIN = 40
BUFF_SIZE = 1024 * 16

def init_sdr():
    sdr = SoapySDR.Device({"driver": "hackrf"})
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0 , CENTER_FREQ)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, GAIN)
    return sdr

def rx_loop(sdr):
    rx_stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(rx_stream)
    buf = np.zeros(BUFF_SIZE, dtype=np.complex64)

    print("Listening on 462.550 MHZ...")
    while True:
        sr = sdr.readStream(rx_stream, [buf], len(buf))
        if sr.ret > 0:
            process_iq(buf[:sr.ret])

def process_iq(data):
    #placeholder for demod
    print(f"Got {len(data)} samples")

if __name__ == "__main__":
    sdr = init_sdr()
    rx_loop(sdr)


