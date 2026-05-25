#!/usr/bin/env python3
"""
Test script for TX pipeline.

Generates a test tone, plays a WAV file, or uses TTS and transmits it.
Run with: python test_tx.py [tone|wav|tts|modulator] [options]
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
from config import AUDIO_RATE, CTCSS_FREQ, CENTER_FREQ, NBFM_DEVIATION
from radio.sdr import SDRDevice
from radio.tx import TXProcessor


def generate_test_tone(freq: float = 1000.0, duration: float = 2.0) -> np.ndarray:
    """Generate a simple sine wave test tone."""
    t = np.arange(int(AUDIO_RATE * duration)) / AUDIO_RATE
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone.astype(np.float32)


def load_wav_file(path: str) -> np.ndarray:
    """Load a WAV file and resample to AUDIO_RATE."""
    sample_rate, audio = wav.read(path)
    
    # Convert to float32 [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    
    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sample_rate != AUDIO_RATE:
        audio = resample_poly(audio, AUDIO_RATE, sample_rate).astype(np.float32)
    
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def print_tx_info():
    """Print TX configuration info."""
    print(f"[TEST] Center frequency: {CENTER_FREQ/1e6:.5f} MHz")
    print(f"[TEST] FM deviation: ±{NBFM_DEVIATION} Hz")
    print(f"[TEST] CTCSS tone: {CTCSS_FREQ} Hz")
    print(f"[TEST] Audio rate: {AUDIO_RATE} Hz")


def test_tone_tx(freq: float = 1000.0, duration: float = 2.0):
    """Test TX with a simple tone."""
    print(f"[TEST] Generating {freq}Hz test tone ({duration} seconds)...")
    audio = generate_test_tone(freq, duration)
    
    print(f"[TEST] Audio: {len(audio)} samples, {len(audio)/AUDIO_RATE:.2f}s")
    print_tx_info()
    
    sdr = SDRDevice()
    if not sdr.open():
        print("[TEST] Failed to open SDR")
        return 1
    
    tx = TXProcessor(sdr)
    
    try:
        print("[TEST] Transmitting...")
        tx.transmit(audio, lead_in=0.4, lead_out=0.5)
        print("[TEST] Done!")
    finally:
        sdr.close()
    
    return 0


def test_wav_tx(wav_path: str):
    """Test TX with a WAV file."""
    print(f"[TEST] Loading WAV file: {wav_path}")
    
    if not os.path.exists(wav_path):
        print(f"[TEST] File not found: {wav_path}")
        return 1
    
    audio = load_wav_file(wav_path)
    
    print(f"[TEST] Audio: {len(audio)} samples, {len(audio)/AUDIO_RATE:.2f}s")
    print_tx_info()
    
    sdr = SDRDevice()
    if not sdr.open():
        print("[TEST] Failed to open SDR")
        return 1
    
    tx = TXProcessor(sdr)
    
    try:
        print("[TEST] Transmitting...")
        tx.transmit(audio, lead_in=0.2, lead_out=0.5)
        print("[TEST] Done!")
    finally:
        sdr.close()
    
    return 0


def test_tts_tx(text: str = None):
    """Test TX with TTS audio."""
    try:
        from audio.tts import synthesize_speech
    except ImportError as e:
        print(f"[TEST] TTS not available: {e}")
        print("[TEST] Make sure piper-tts is installed and voice model is downloaded")
        return 1
    
    if text is None:
        text = "Alpha X-Ray 3-1, this is Warden. Radio check, how copy?"
    
    print(f"[TEST] Synthesizing: '{text}'")
    
    try:
        audio = synthesize_speech(text)
    except FileNotFoundError as e:
        print(f"[TEST] {e}")
        return 1
    
    print(f"[TEST] Audio: {len(audio)} samples, {len(audio)/AUDIO_RATE:.2f}s")
    print_tx_info()
    
    sdr = SDRDevice()
    if not sdr.open():
        print("[TEST] Failed to open SDR")
        return 1
    
    tx = TXProcessor(sdr)
    
    try:
        print("[TEST] Transmitting...")
        tx.transmit(audio, lead_in=0.2, lead_out=0.5)
        print("[TEST] Done!")
    finally:
        sdr.close()
    
    return 0


def test_modulator_only(source: str = "tone"):
    """Test modulator without SDR (generates IQ file for inspection)."""
    from radio.modulator import FMModulator
    
    print("[TEST] Testing modulator (no SDR)...")
    
    if source == "tone":
        audio = generate_test_tone(1000.0, 1.0)
        print("[TEST] Source: 1kHz test tone")
    elif os.path.exists(source):
        audio = load_wav_file(source)
        print(f"[TEST] Source: {source}")
    else:
        print(f"[TEST] Unknown source: {source}")
        return 1
    
    print_tx_info()
    
    mod = FMModulator(ctcss_enabled=True)
    iq = mod.modulate(audio)
    
    print(f"[TEST] Input: {len(audio)} audio samples ({len(audio)/AUDIO_RATE:.2f}s)")
    print(f"[TEST] Output: {len(iq)} IQ samples")
    print(f"[TEST] IQ magnitude range: {np.abs(iq).min():.3f} - {np.abs(iq).max():.3f}")
    
    # Save IQ as stereo WAV for inspection
    iq_stereo = np.column_stack([iq.real, iq.imag])
    iq_int16 = (iq_stereo * 32767).astype(np.int16)
    
    from config import SAMPLE_RATE
    wav.write('test_iq.wav', int(SAMPLE_RATE), iq_int16)
    print(f"[TEST] Saved test_iq.wav at {int(SAMPLE_RATE)} Hz sample rate")
    print("[TEST] Open in Audacity (import as raw stereo) or SDR software")
    
    return 0


def print_usage():
    print("Usage: python test_tx.py <mode> [options]")
    print()
    print("Modes:")
    print("  tone [freq] [duration]  - Transmit test tone (default: 1000Hz, 2s)")
    print("  wav <file.wav>          - Transmit WAV file")
    print("  tts [text]              - Transmit TTS speech")
    print("  modulator [source]      - Test modulator only, save IQ file")
    print("                            source: 'tone' or path to WAV file")
    print()
    print("Examples:")
    print("  python test_tx.py tone")
    print("  python test_tx.py tone 800 3")
    print("  python test_tx.py wav voice.wav")
    print("  python test_tx.py tts 'Hello world'")
    print("  python test_tx.py modulator")
    print("  python test_tx.py modulator voice.wav")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "tone":
        freq = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
        duration = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        sys.exit(test_tone_tx(freq, duration))
    
    elif mode == "wav":
        if len(sys.argv) < 3:
            print("Error: WAV file path required")
            print_usage()
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
        print_usage()
        sys.exit(1)
