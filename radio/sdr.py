import numpy as np
import SoapySDR
from config import SAMPLE_RATE, CENTER_FREQ, LNA_GAIN, VGA_GAIN, AMP_ENABLE, BUFF_SIZE
from radio.demod import process_iq


def init_sdr():
    results = SoapySDR.Device.enumerate()
    if not results:
        raise RuntimeError("No SDR devices found - check USB connection")
    print(f"Found devices: {results}")
    sdr = SoapySDR.Device(results[0])
    
    # Basic config
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, CENTER_FREQ)
    
    # Set individual gain stages for HackRF
    # LNA = RF amplifier (0-40 dB)
    # VGA = IF/baseband amplifier (0-62 dB) 
    # AMP = external 14 dB RF amp on/off
    try:
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "LNA", LNA_GAIN)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "VGA", VGA_GAIN)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "AMP", 14.0 if AMP_ENABLE else 0.0)
        print(f"[SDR] LNA={LNA_GAIN}dB, VGA={VGA_GAIN}dB, AMP={'ON' if AMP_ENABLE else 'OFF'}")
    except Exception as e:
        # Fallback for SDRs that don't support named gain stages
        total_gain = LNA_GAIN + VGA_GAIN + (14 if AMP_ENABLE else 0)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, total_gain)
        print(f"[SDR] Using combined gain={total_gain}dB (named stages not supported: {e})")
    
    # Print available gain range for reference
    try:
        gain_range = sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
        print(f"[SDR] Total gain range: {gain_range}")
    except:
        pass
    
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