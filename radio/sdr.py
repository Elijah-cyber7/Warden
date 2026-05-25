"""
SDR device management for Warden.

Handles HackRF initialization and configuration for both RX and TX via SoapySDR.
"""

import logging
import time

import numpy as np
import SoapySDR
from config import (
    SAMPLE_RATE, CENTER_FREQ, BUFF_SIZE,
    RX_LNA_GAIN, RX_VGA_GAIN, RX_AMP_ENABLE,
    TX_VGA_GAIN, TX_AMP_ENABLE
)

log = logging.getLogger("warden.sdr")


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
        """Open and configure the SDR device."""
        results = SoapySDR.Device.enumerate()
        if not results:
            log.error("No devices found — check USB connection")
            return False

        log.info("Found device: %s", results[0].get("label", results[0]))
        self._device = SoapySDR.Device(results[0])

        self._device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, SAMPLE_RATE)
        self._device.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, SAMPLE_RATE)
        self._device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, CENTER_FREQ)
        self._device.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, CENTER_FREQ)

        self._set_rx_gains()
        self._set_tx_gains()

        self._rx_buffer = np.zeros(BUFF_SIZE, dtype=np.complex64)

        log.info("Configured: %.4f MHz @ %.1f MSPS", CENTER_FREQ / 1e6, SAMPLE_RATE / 1e6)
        return True

    def _set_rx_gains(self):
        """Configure RX gain stages."""
        try:
            self._device.setGain(SoapySDR.SOAPY_SDR_RX, 0, "LNA", RX_LNA_GAIN)
            self._device.setGain(SoapySDR.SOAPY_SDR_RX, 0, "VGA", RX_VGA_GAIN)
            self._device.setGain(SoapySDR.SOAPY_SDR_RX, 0, "AMP", 14.0 if RX_AMP_ENABLE else 0.0)
            log.info("RX gains: LNA=%ddB VGA=%ddB AMP=%s", RX_LNA_GAIN, RX_VGA_GAIN,
                     "ON" if RX_AMP_ENABLE else "OFF")
        except Exception:
            total = RX_LNA_GAIN + RX_VGA_GAIN + (14 if RX_AMP_ENABLE else 0)
            self._device.setGain(SoapySDR.SOAPY_SDR_RX, 0, total)
            log.info("RX gain: %ddB (combined — named stages not supported)", total)

    def _set_tx_gains(self):
        """Configure TX gain stages."""
        try:
            self._device.setGain(SoapySDR.SOAPY_SDR_TX, 0, "VGA", TX_VGA_GAIN)
            self._device.setGain(SoapySDR.SOAPY_SDR_TX, 0, "AMP", 14.0 if TX_AMP_ENABLE else 0.0)
            log.info("TX gains: VGA=%ddB AMP=%s", TX_VGA_GAIN, "ON" if TX_AMP_ENABLE else "OFF")
        except Exception:
            total = TX_VGA_GAIN + (14 if TX_AMP_ENABLE else 0)
            self._device.setGain(SoapySDR.SOAPY_SDR_TX, 0, total)
            log.info("TX gain: %ddB (combined — named stages not supported)", total)

    def start_rx(self):
        """Start RX stream."""
        if self._rx_stream is not None:
            return
        self._rx_stream = self._device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self._device.activateStream(self._rx_stream)
        log.debug("RX stream started")

    def stop_rx(self):
        """Stop RX stream."""
        if self._rx_stream is None:
            return
        self._device.deactivateStream(self._rx_stream)
        self._device.closeStream(self._rx_stream)
        self._rx_stream = None
        log.debug("RX stream stopped")

    def read_rx(self) -> np.ndarray | None:
        """
        Read IQ samples from RX stream.

        Returns:
            Complex64 array of IQ samples, or None on timeout/error.
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
        log.debug("TX stream started")

    def stop_tx(self):
        """Stop TX stream."""
        if self._tx_stream is None:
            return
        self._device.deactivateStream(self._tx_stream)
        self._device.closeStream(self._tx_stream)
        self._tx_stream = None
        log.debug("TX stream stopped")

    def write_tx(self, iq: np.ndarray) -> int:
        """
        Write IQ samples to TX stream with backpressure handling.

        Returns:
            Number of samples actually written.
        """
        if self._tx_stream is None:
            return 0

        total_written = 0
        max_chunk = 65536
        retries = 0

        while total_written < len(iq):
            end = min(total_written + max_chunk, len(iq))
            chunk = iq[total_written:end]

            sr = self._device.writeStream(
                self._tx_stream, [chunk], len(chunk), timeoutUs=1000000
            )

            if sr.ret > 0:
                total_written += sr.ret
                retries = 0
            else:
                retries += 1
                if retries > 200:
                    log.warning("write_tx stalled at %d/%d samples (ret=%d)",
                                total_written, len(iq), sr.ret)
                    break
                time.sleep(0.01)

        return total_written

    def close(self):
        """Close device and all streams."""
        self.stop_rx()
        self.stop_tx()
        self._device = None
        log.info("Device closed")
