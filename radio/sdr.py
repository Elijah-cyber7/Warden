"""
SDR device management for Warden.

Handles HackRF initialization and configuration for both RX and TX.
"""

import numpy as np
import SoapySDR
from config import (
    SAMPLE_RATE, CENTER_FREQ, BUFF_SIZE,
    RX_LNA_GAIN, RX_VGA_GAIN, RX_AMP_ENABLE,
    TX_VGA_GAIN, TX_AMP_ENABLE
)


class SDRDevice:
    """
    Manages HackRF SDR device via SoapySDR.
    
    Supports both RX and TX with separate gain configurations.
    """
    
    def __init__(self):
        self._device = None
        self._rx_stream = None
        self._tx_stream = None
        self._rx_buffer = None
    
    def open(self) -> bool:
        """
        Open and configure the SDR device.
        
        Returns:
            True if device opened successfully, False otherwise.
        """
        results = SoapySDR.Device.enumerate()
        if not results:
            print("[SDR] No devices found - check USB connection")
            return False
        
        print(f"[SDR] Found devices: {results}")
        self._device = SoapySDR.Device(results[0])
        
        self._device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, SAMPLE_RATE)
        self._device.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, SAMPLE_RATE)
        self._device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, CENTER_FREQ)
        self._device.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, CENTER_FREQ)
        
        self._set_rx_gains()
        self._set_tx_gains()
        
        self._rx_buffer = np.zeros(BUFF_SIZE, dtype=np.complex64)
        
        print(f"[SDR] Configured for {CENTER_FREQ / 1e6:.4f} MHz @ {SAMPLE_RATE / 1e6:.1f} MSPS")
        return True
    
    def _set_rx_gains(self):
        """Configure RX gain stages."""
        try:
            self._device.setGain(SoapySDR.SOAPY_SDR_RX, 0, "LNA", RX_LNA_GAIN)
            self._device.setGain(SoapySDR.SOAPY_SDR_RX, 0, "VGA", RX_VGA_GAIN)
            self._device.setGain(SoapySDR.SOAPY_SDR_RX, 0, "AMP", 14.0 if RX_AMP_ENABLE else 0.0)
            print(f"[SDR] RX: LNA={RX_LNA_GAIN}dB, VGA={RX_VGA_GAIN}dB, AMP={'ON' if RX_AMP_ENABLE else 'OFF'}")
        except Exception as e:
            total = RX_LNA_GAIN + RX_VGA_GAIN + (14 if RX_AMP_ENABLE else 0)
            self._device.setGain(SoapySDR.SOAPY_SDR_RX, 0, total)
            print(f"[SDR] RX: Combined gain={total}dB (named stages not supported)")
    
    def _set_tx_gains(self):
        """Configure TX gain stages."""
        try:
            self._device.setGain(SoapySDR.SOAPY_SDR_TX, 0, "VGA", TX_VGA_GAIN)
            self._device.setGain(SoapySDR.SOAPY_SDR_TX, 0, "AMP", 14.0 if TX_AMP_ENABLE else 0.0)
            print(f"[SDR] TX: VGA={TX_VGA_GAIN}dB, AMP={'ON' if TX_AMP_ENABLE else 'OFF'}")
        except Exception as e:
            total = TX_VGA_GAIN + (14 if TX_AMP_ENABLE else 0)
            self._device.setGain(SoapySDR.SOAPY_SDR_TX, 0, total)
            print(f"[SDR] TX: Combined gain={total}dB (named stages not supported)")
    
    def start_rx(self):
        """Start RX stream."""
        if self._rx_stream is not None:
            return
        self._rx_stream = self._device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self._device.activateStream(self._rx_stream)
        print("[SDR] RX stream started")
    
    def stop_rx(self):
        """Stop RX stream."""
        if self._rx_stream is None:
            return
        self._device.deactivateStream(self._rx_stream)
        self._device.closeStream(self._rx_stream)
        self._rx_stream = None
        print("[SDR] RX stream stopped")
    
    def read_rx(self) -> np.ndarray | None:
        """
        Read IQ samples from RX stream.
        
        Returns:
            Complex64 array of IQ samples, or None if no data.
        """
        if self._rx_stream is None:
            return None
        
        sr = self._device.readStream(self._rx_stream, [self._rx_buffer], len(self._rx_buffer))
        if sr.ret > 0:
            return self._rx_buffer[:sr.ret].copy()
        return None
    
    def start_tx(self):
        """Start TX stream."""
        if self._tx_stream is not None:
            return
        self._tx_stream = self._device.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32)
        self._device.activateStream(self._tx_stream)
        print("[SDR] TX stream started")
    
    def stop_tx(self):
        """Stop TX stream."""
        if self._tx_stream is None:
            return
        self._device.deactivateStream(self._tx_stream)
        self._device.closeStream(self._tx_stream)
        self._tx_stream = None
        print("[SDR] TX stream stopped")
    
    def write_tx(self, iq: np.ndarray) -> int:
        """
        Write IQ samples to TX stream. Blocks until all samples are accepted.
        
        Args:
            iq: Complex64 array of IQ samples to transmit.
            
        Returns:
            Total number of samples written.
        """
        if self._tx_stream is None:
            return 0
        
        total_written = 0
        timeout_us = 500000  # 500ms timeout per write attempt
        
        while total_written < len(iq):
            remaining = iq[total_written:]
            sr = self._device.writeStream(
                self._tx_stream, [remaining], len(remaining), timeoutUs=timeout_us
            )
            if sr.ret > 0:
                total_written += sr.ret
            elif sr.ret == 0:
                # Buffer full, wait briefly for hardware to drain
                import time
                time.sleep(0.005)
            else:
                # Error
                print(f"[SDR] writeStream error: {sr.ret}")
                break
        
        return total_written
    
    def close(self):
        """Close device and all streams."""
        self.stop_rx()
        self.stop_tx()
        self._device = None
        print("[SDR] Device closed")
