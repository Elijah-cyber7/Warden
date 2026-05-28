"""
TX FIFO Integrity BIT — checks for underruns during sustained streaming.

Streams 2 seconds of constant-envelope CW and monitors for writeStream
stalls that would cause carrier dropouts (breaking CTCSS continuity).
Also verifies total streaming duration matches expected wall-clock time,
confirming backpressure rate-limiting is functioning correctly.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import SoapySDR

from config import SAMPLE_RATE


log = logging.getLogger("warden.bit.fifo")

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
    total_chunks = (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES

    log.info("FIFO Integrity: streaming %d samples in %d chunks (%.1fs expected)",
             total_samples, total_chunks, expected_duration)
    log.info("FIFO Integrity: sample rate %.1f MSPS, chunk size %d",
             SAMPLE_RATE / 1e6, CHUNK_SAMPLES)

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
    chunk_index = 0

    try:
        write_start = time.monotonic()

        for offset in range(0, total_samples, CHUNK_SAMPLES):
            chunk = iq_data[offset:offset + CHUNK_SAMPLES]
            chunk_index += 1

            sr = device.writeStream(stream, [chunk], len(chunk), timeoutUs=1000000)

            if sr.ret <= 0:
                stall_start = time.monotonic()
                elapsed_at_stall = stall_start - write_start
                retries = 0
                while sr.ret <= 0 and retries < 100:
                    time.sleep(0.005)
                    sr = device.writeStream(stream, [chunk], len(chunk), timeoutUs=500000)
                    retries += 1
                stall_duration = time.monotonic() - stall_start
                stall_count += 1
                total_stall_time += stall_duration

                log.warning(
                    "FIFO Integrity: STALL #%d at chunk %d/%d (%.3fs into stream) — "
                    "ret=%d, recovered after %d retries (%.1fms stall). "
                    "Cause: USB write blocked or FIFO overflow",
                    stall_count, chunk_index, total_chunks, elapsed_at_stall,
                    sr.ret, retries, stall_duration * 1000
                )

            if sr.ret > 0:
                samples_written += sr.ret

        write_elapsed = time.monotonic() - write_start
    finally:
        device.deactivateStream(stream)
        device.closeStream(stream)

    timing_error_pct = abs(write_elapsed - expected_duration) / expected_duration
    total_stall_ms = total_stall_time * 1000.0

    passed = stall_count == 0 and timing_error_pct <= TIMING_TOLERANCE

    log.info("FIFO Integrity: streamed %d/%d samples in %.3fs (expected %.1fs)",
             samples_written, total_samples, write_elapsed, expected_duration)
    log.info("FIFO Integrity: timing error %.1f%% (tolerance %.0f%%)",
             timing_error_pct * 100, TIMING_TOLERANCE * 100)
    log.info("FIFO Integrity: %d stalls, %.1fms total stall time",
             stall_count, total_stall_ms)

    if not passed:
        if stall_count > 0 and timing_error_pct > TIMING_TOLERANCE:
            log.error("FIFO Integrity: FAIL — stalls AND timing drift. "
                      "USB cannot sustain continuous TX. Check: USB hub, "
                      "cable quality, bus contention from other devices")
        elif stall_count > 0:
            log.error("FIFO Integrity: FAIL — %d stalls detected. "
                      "Carrier dropped for %.1fms total. This WILL break CTCSS. "
                      "Check: USB power, hub contention, Python GC pressure",
                      stall_count, total_stall_ms)
        else:
            log.error("FIFO Integrity: FAIL — timing drift %.1f%%. "
                      "Stream took %.3fs instead of %.1fs. "
                      "Possible cause: USB backpressure inconsistent, "
                      "device clock mismatch, or writeStream blocking too long",
                      timing_error_pct * 100, write_elapsed, expected_duration)
    else:
        log.info("FIFO Integrity: PASS")

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
