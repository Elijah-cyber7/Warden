#!/usr/bin/env python3
"""
Test script for TX pipeline.

Generates a test tone, plays a WAV file, or uses TTS and transmits it.
Run with: python test_tx.py [tone|wav|tts|modulator] [options]
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import logging
import sys

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
from config import AUDIO_RATE, CTCSS_FREQ, CENTER_FREQ, NBFM_DEVIATION, SAMPLE_RATE
from radio.sdr import SDRDevice
from radio.tx import TXProcessor

log = logging.getLogger("warden.test")


def generate_test_tone(freq: float = 1000.0, duration: float = 2.0) -> np.ndarray:
    """Generate a simple sine wave test tone."""
    t = np.arange(int(AUDIO_RATE * duration)) / AUDIO_RATE
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone.astype(np.float32)


def load_wav_file(path: str) -> np.ndarray:
    """Load a WAV file and resample to AUDIO_RATE."""
    sample_rate, audio = wav.read(path)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sample_rate != AUDIO_RATE:
        audio = resample_poly(audio, AUDIO_RATE, sample_rate).astype(np.float32)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def print_tx_info():
    """Print TX configuration summary."""
    log.info("Freq: %.5f MHz | Deviation: ±%d Hz | CTCSS: %.1f Hz | Audio: %d Hz",
             CENTER_FREQ / 1e6, NBFM_DEVIATION, CTCSS_FREQ, AUDIO_RATE)


def test_tone_tx(freq: float = 1000.0, duration: float = 2.0):
    """Test TX with a simple tone."""
    log.info("Generating %gHz test tone (%.1fs)", freq, duration)
    audio = generate_test_tone(freq, duration)
    print_tx_info()

    sdr = SDRDevice()
    if not sdr.open():
        return 1

    tx = TXProcessor(sdr)
    try:
        tx.transmit(audio, lead_in=0.4, lead_out=0.5)
    finally:
        sdr.close()
    return 0


def test_wav_tx(wav_path: str):
    """Test TX with a WAV file."""
    if not os.path.exists(wav_path):
        log.error("File not found: %s", wav_path)
        return 1

    log.info("Loading: %s", wav_path)
    audio = load_wav_file(wav_path)
    log.info("Audio: %d samples (%.2fs)", len(audio), len(audio) / AUDIO_RATE)
    print_tx_info()

    sdr = SDRDevice()
    if not sdr.open():
        return 1

    tx = TXProcessor(sdr)
    try:
        tx.transmit(audio, lead_in=0.2, lead_out=0.5)
    finally:
        sdr.close()
    return 0


def test_tts_tx(text: str = None):
    """Test TX with TTS audio."""
    try:
        from audio.tts import synthesize_speech
    except ImportError as e:
        log.error("TTS not available: %s", e)
        return 1

    if text is None:
        text = "Alpha X-Ray 3-1, this is Warden. Radio check, how copy?"

    log.info("Synthesizing: '%s'", text)

    try:
        audio = synthesize_speech(text)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1

    log.info("Audio: %d samples (%.2fs)", len(audio), len(audio) / AUDIO_RATE)
    print_tx_info()

    sdr = SDRDevice()
    if not sdr.open():
        return 1

    tx = TXProcessor(sdr)
    try:
        tx.transmit(audio, lead_in=0.2, lead_out=0.5)
    finally:
        sdr.close()
    return 0


def test_modulator_only(source: str = "tone"):
    """Test modulator without SDR — saves IQ to WAV for inspection."""
    from radio.modulator import FMModulator

    if source == "tone":
        audio = generate_test_tone(1000.0, 1.0)
        log.info("Source: 1kHz test tone (1s)")
    elif os.path.exists(source):
        audio = load_wav_file(source)
        log.info("Source: %s (%.2fs)", source, len(audio) / AUDIO_RATE)
    else:
        log.error("Unknown source: %s", source)
        return 1

    print_tx_info()

    mod = FMModulator(ctcss_enabled=True)
    iq = mod.modulate(audio)

    log.info("Output: %d IQ samples | magnitude: %.3f–%.3f",
             len(iq), np.abs(iq).min(), np.abs(iq).max())

    iq_stereo = np.column_stack([iq.real, iq.imag])
    iq_int16 = (iq_stereo * 32767).astype(np.int16)
    wav.write('test_iq.wav', int(SAMPLE_RATE), iq_int16)
    log.info("Saved test_iq.wav (%d Hz stereo I/Q)", int(SAMPLE_RATE))

    return 0


USAGE = """\
Usage: python test_tx.py <mode> [options]

Modes:
  tone [freq] [duration]  - Transmit test tone (default: 1000Hz, 2s)
  wav <file.wav>          - Transmit WAV file
  tts [text]              - Transmit TTS speech
  modulator [source]      - Test modulator only, save IQ file

Examples:
  python test_tx.py tone
  python test_tx.py tone 800 3
  python test_tx.py wav voice.wav
  python test_tx.py tts 'Hello world'
  python test_tx.py modulator
"""


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "tone":
        freq = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
        duration = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        sys.exit(test_tone_tx(freq, duration))

    elif mode == "wav":
        if len(sys.argv) < 3:
            print("Error: WAV file path required")
            sys.exit(1)
        sys.exit(test_wav_tx(sys.argv[2]))

    elif mode == "tts":
        text = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else None
        sys.exit(test_tts_tx(text))

    elif mode == "modulator":
        source = sys.argv[2] if len(sys.argv) > 2 else "tone"
        sys.exit(test_modulator_only(source))

    else:
        print(f"Unknown mode: {mode}")
        print(USAGE)
        sys.exit(1)
