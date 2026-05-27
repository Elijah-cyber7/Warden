"""
TX FIFO Integrity BIT — checks for underruns during sustained streaming.

Streams 2 seconds of constant-envelope CW and monitors for writeStream
stalls that would cause carrier dropouts (breaking CTCSS continuity).
Also verifies total streaming duration matches expected wall-clock time,
confirming backpressure rate-limiting is functioning correctly.
"""

import time
from dataclasses import dataclass

import numpy as np
import SoapySDR

from config import SAMPLE_RATE


TEST_DURATION_SEC = 2.0
CHUNK_SAMPLES = 65536
TIMING_TOLERANCE = 0.05  # 5% tolerance on total write duration


@dataclass
class FIFOIntegrityResult:
    passed: bool
    stall_count: int
    total_stall_ms: float
    timing_error_pct: float
    detail: str


def run(device) -> FIFOIntegrityResult:
    """
    Stream continuous IQ and check for FIFO underruns.

    Args:
        device: SoapySDR device instance (already opened).

    Returns:
        FIFOIntegrityResult with pass/fail and measurements.
    """
    total_samples = int(SAMPLE_RATE * TEST_DURATION_SEC)
    expected_duration = TEST_DURATION_SEC

    # Constant-envelope CW (same as real TX output structure)
    phase = np.linspace(0, 2 * np.pi * 1000 * TEST_DURATION_SEC,
                        total_samples, endpoint=False)
    iq_data = np.exp(1j * phase).astype(np.complex64)

    stream = device.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32)
    device.activateStream(stream)
    time.sleep(0.3)

    stall_count = 0
    total_stall_time = 0.0
    samples_written = 0

    try:
        write_start = time.monotonic()

        for offset in range(0, total_samples, CHUNK_SAMPLES):
            chunk = iq_data[offset:offset + CHUNK_SAMPLES]

            sr = device.writeStream(stream, [chunk], len(chunk), timeoutUs=1000000)

            if sr.ret <= 0:
                stall_start = time.monotonic()
                retries = 0
                while sr.ret <= 0 and retries < 100:
                    time.sleep(0.005)
                    sr = device.writeStream(stream, [chunk], len(chunk), timeoutUs=500000)
                    retries += 1
                stall_duration = time.monotonic() - stall_start
                stall_count += 1
                total_stall_time += stall_duration

            if sr.ret > 0:
                samples_written += sr.ret

        write_elapsed = time.monotonic() - write_start
    finally:
        device.deactivateStream(stream)
        device.closeStream(stream)

    timing_error_pct = abs(write_elapsed - expected_duration) / expected_duration
    total_stall_ms = total_stall_time * 1000.0

    passed = stall_count == 0 and timing_error_pct <= TIMING_TOLERANCE

    if passed:
        detail = f"Clean stream, {write_elapsed:.2f}s elapsed (expected {expected_duration:.1f}s)"
    else:
        problems = []
        if stall_count > 0:
            problems.append(f"{stall_count} stalls ({total_stall_ms:.0f}ms total)")
        if timing_error_pct > TIMING_TOLERANCE:
            problems.append(f"timing drift {timing_error_pct*100:.1f}% (>{TIMING_TOLERANCE*100:.0f}%)")
        detail = "; ".join(problems)

    return FIFOIntegrityResult(
        passed=passed,
        stall_count=stall_count,
        total_stall_ms=total_stall_ms,
        timing_error_pct=timing_error_pct,
        detail=detail,
    )
