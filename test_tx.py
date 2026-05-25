#!/usr/bin/env python3
"""
Test script for TX pipeline.

Generates a test tone or TTS audio and transmits it.
Run with: python test_tx.py [tone|tts]
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np
from config import AUDIO_RATE, CTCSS_FREQ
from radio.sdr import SDRDevice
from radio.tx import TXProcessor


def generate_test_tone(freq: float = 1000.0, duration: float = 2.0) -> np.ndarray:
    """Generate a simple sine wave test tone."""
    t = np.arange(int(AUDIO_RATE * duration)) / AUDIO_RATE
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone.astype(np.float32)


def generate_two_tone(freq1: float = 1000.0, freq2: float = 1800.0, duration: float = 2.0) -> np.ndarray:
    """Generate a two-tone test signal (common for FM testing)."""
    t = np.arange(int(AUDIO_RATE * duration)) / AUDIO_RATE
    tone = 0.3 * np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(2 * np.pi * freq2 * t)
    return tone.astype(np.float32)


def test_tone_tx():
    """Test TX with a simple tone."""
    print("[TEST] Generating 1kHz test tone (2 seconds)...")
    audio = generate_test_tone(1000.0, 2.0)
    
    print(f"[TEST] Audio: {len(audio)} samples, {len(audio)/AUDIO_RATE:.2f}s")
    print(f"[TEST] CTCSS tone: {CTCSS_FREQ} Hz")
    
    sdr = SDRDevice()
    if not sdr.open():
        print("[TEST] Failed to open SDR")
        return 1
    
    tx = TXProcessor(sdr)
    
    try:
        print("[TEST] Transmitting...")
        tx.transmit(audio, lead_in=0.2, lead_out=0.2)
        print("[TEST] Done!")
    finally:
        sdr.close()
    
    return 0


def test_tts_tx():
    """Test TX with TTS audio."""
    try:
        from audio.tts import synthesize_speech
    except ImportError as e:
        print(f"[TEST] TTS not available: {e}")
        print("[TEST] Make sure piper-tts is installed and voice model is downloaded")
        return 1
    
    text = "Alpha X-Ray 3-1, this is Warden. Radio check, how copy?"
    print(f"[TEST] Synthesizing: '{text}'")
    
    try:
        audio = synthesize_speech(text)
    except FileNotFoundError as e:
        print(f"[TEST] {e}")
        return 1
    
    print(f"[TEST] Audio: {len(audio)} samples, {len(audio)/AUDIO_RATE:.2f}s")
    print(f"[TEST] CTCSS tone: {CTCSS_FREQ} Hz")
    
    sdr = SDRDevice()
    if not sdr.open():
        print("[TEST] Failed to open SDR")
        return 1
    
    tx = TXProcessor(sdr)
    
    try:
        print("[TEST] Transmitting...")
        tx.transmit(audio, lead_in=0.2, lead_out=0.2)
        print("[TEST] Done!")
    finally:
        sdr.close()
    
    return 0


def test_modulator_only():
    """Test modulator without SDR (generates IQ file for inspection)."""
    from radio.modulator import FMModulator
    import scipy.io.wavfile as wav
    
    print("[TEST] Testing modulator (no SDR)...")
    
    audio = generate_test_tone(1000.0, 1.0)
    mod = FMModulator(ctcss_enabled=True)
    
    iq = mod.modulate(audio)
    
    print(f"[TEST] Input: {len(audio)} audio samples")
    print(f"[TEST] Output: {len(iq)} IQ samples")
    print(f"[TEST] IQ magnitude range: {np.abs(iq).min():.3f} - {np.abs(iq).max():.3f}")
    
    # Save IQ as stereo WAV for inspection in SDR software
    iq_stereo = np.column_stack([iq.real, iq.imag])
    iq_int16 = (iq_stereo * 32767).astype(np.int16)
    wav.write('test_iq.wav', int(2e6 // 13 * 13), iq_int16)  # Intermediate rate
    print("[TEST] Saved test_iq.wav (open in Audacity or SDR software)")
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_tx.py [tone|tts|modulator]")
        print("  tone      - Transmit 1kHz test tone")
        print("  tts       - Transmit TTS speech")
        print("  modulator - Test modulator only (no SDR, saves IQ file)")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "tone":
        sys.exit(test_tone_tx())
    elif mode == "tts":
        sys.exit(test_tts_tx())
    elif mode == "modulator":
        sys.exit(test_modulator_only())
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
