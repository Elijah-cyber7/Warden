"""
TX FIFO Integrity BIT — checks for write failures during sustained streaming.

Streams 2 seconds of constant-envelope CW and monitors for writeStream
failures (ret <= 0) that indicate the USB pipe cannot sustain the required
throughput. Also verifies all samples were accepted by the device.

NOTE: SoapyHackRF buffers internally and does NOT rate-limit writeStream
to the sample clock. Writes completing faster than real-time is normal —
the production tx.py compensates with a drain sleep. This test does NOT
check wall-clock timing; it checks data delivery integrity.
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


@dataclass
class FIFOIntegrityResult:
    passed: bool
    stall_count: int
    total_stall_ms: float
    samples_written: int
    samples_expected: int
    write_duration: float
    detail: str


def run(device) -> FIFOIntegrityResult:
    """
    Stream continuous IQ and check for write failures.

    A "stall" is any writeStream call that returns ret <= 0, meaning the
    USB/driver layer rejected the write. In real TX operation, this would
    cause a gap in the carrier that breaks CTCSS.

    Args:
        device: SoapySDR device instance (already opened).

    Returns:
        FIFOIntegrityResult with pass/fail and measurements.
    """
    total_samples = int(SAMPLE_RATE * TEST_DURATION_SEC)
    total_chunks = (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES

    log.info("FIFO Integrity: streaming %d samples in %d chunks",
             total_samples, total_chunks)
    log.info("FIFO Integrity: sample rate %.1f MSPS, chunk size %d",
             SAMPLE_RATE / 1e6, CHUNK_SAMPLES)

    # Constant-envelope CW (same structure as real TX output)
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
                    "FIFO Integrity: STALL #%d at chunk %d/%d (%.3fs into writes) — "
                    "writeStream returned %d, recovered after %d retries (%.1fms). "
                    "This means the USB/driver layer rejected a write — in real TX "
                    "this would be a carrier dropout that breaks CTCSS.",
                    stall_count, chunk_index, total_chunks, elapsed_at_stall,
                    sr.ret, retries, stall_duration * 1000
                )

            if sr.ret > 0:
                samples_written += sr.ret

        write_elapsed = time.monotonic() - write_start
    finally:
        device.deactivateStream(stream)
        device.closeStream(stream)

    total_stall_ms = total_stall_time * 1000.0
    all_samples_accepted = samples_written == total_samples
    passed = stall_count == 0 and all_samples_accepted

    log.info("FIFO Integrity: wrote %d/%d samples in %.3fs",
             samples_written, total_samples, write_elapsed)
    log.info("FIFO Integrity: %d stalls, %.1fms total stall time",
             stall_count, total_stall_ms)

    if not passed:
        if stall_count > 0 and not all_samples_accepted:
            log.error("FIFO Integrity: FAIL — %d stalls AND sample loss (%d/%d). "
                      "USB pipe is unreliable. Check: USB hub, cable, bus contention",
                      stall_count, samples_written, total_samples)
        elif stall_count > 0:
            log.error("FIFO Integrity: FAIL — %d stalls (%.1fms total). "
                      "Carrier would drop %d times during a real transmission. "
                      "Check: USB power, hub contention, other USB devices",
                      stall_count, total_stall_ms, stall_count)
        else:
            log.error("FIFO Integrity: FAIL — sample loss: only %d/%d accepted. "
                      "Device rejected %d samples.",
                      samples_written, total_samples, total_samples - samples_written)
    else:
        log.info("FIFO Integrity: PASS — all samples accepted, zero stalls")

    if passed:
        detail = f"All {total_samples} samples accepted, 0 stalls, {write_elapsed:.2f}s write time"
    else:
        problems = []
        if stall_count > 0:
            problems.append(f"{stall_count} stalls ({total_stall_ms:.0f}ms total)")
        if not all_samples_accepted:
            problems.append(f"sample loss: {samples_written}/{total_samples}")
        detail = "; ".join(problems)

    return FIFOIntegrityResult(
        passed=passed,
        stall_count=stall_count,
        total_stall_ms=total_stall_ms,
        samples_written=samples_written,
        samples_expected=total_samples,
        write_duration=write_elapsed,
        detail=detail,
    )
