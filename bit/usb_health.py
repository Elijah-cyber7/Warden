"""
USB Health BIT — measures write throughput and error rate.

Streams silence IQ to the HackRF and checks that the USB pipe can sustain
the required data rate without excessive errors.
"""

import time
from dataclasses import dataclass

import numpy as np
import SoapySDR

from config import SAMPLE_RATE


EXPECTED_THROUGHPUT_MBS = 3.5
MAX_ERROR_RATE = 0.01
TEST_DURATION_SEC = 1.0
CHUNK_SAMPLES = 65536


@dataclass
class USBHealthResult:
    passed: bool
    throughput_mbs: float
    error_rate: float
    total_writes: int
    failed_writes: int
    detail: str


def run(device) -> USBHealthResult:
    """
    Measure USB write throughput and error rate.

    Args:
        device: SoapySDR device instance (already opened).

    Returns:
        USBHealthResult with pass/fail and measurements.
    """
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.complex64)
    bytes_per_sample = 8  # complex64 = 4 bytes I + 4 bytes Q

    stream = device.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32)
    device.activateStream(stream)
    time.sleep(0.3)

    total_writes = 0
    failed_writes = 0
    total_samples_written = 0

    try:
        start = time.monotonic()
        deadline = start + TEST_DURATION_SEC

        while time.monotonic() < deadline:
            sr = device.writeStream(stream, [silence], len(silence), timeoutUs=500000)
            total_writes += 1
            if sr.ret > 0:
                total_samples_written += sr.ret
            else:
                failed_writes += 1

        elapsed = time.monotonic() - start
    finally:
        device.deactivateStream(stream)
        device.closeStream(stream)

    total_bytes = total_samples_written * bytes_per_sample
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
    error_rate = failed_writes / total_writes if total_writes > 0 else 1.0

    passed = throughput_mbs >= EXPECTED_THROUGHPUT_MBS and error_rate < MAX_ERROR_RATE

    if passed:
        detail = f"{throughput_mbs:.1f} MB/s, {error_rate*100:.1f}% errors"
    else:
        problems = []
        if throughput_mbs < EXPECTED_THROUGHPUT_MBS:
            problems.append(f"low throughput ({throughput_mbs:.1f} < {EXPECTED_THROUGHPUT_MBS} MB/s)")
        if error_rate >= MAX_ERROR_RATE:
            problems.append(f"high error rate ({error_rate*100:.1f}%)")
        detail = "; ".join(problems)

    return USBHealthResult(
        passed=passed,
        throughput_mbs=throughput_mbs,
        error_rate=error_rate,
        total_writes=total_writes,
        failed_writes=failed_writes,
        detail=detail,
    )
