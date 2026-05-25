# =============================================================================
# WARDEN CONFIGURATION
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# -----------------------------------------------------------------------------
# SDR Hardware
# -----------------------------------------------------------------------------
SAMPLE_RATE = 2e6           # SDR sample rate (Hz)
BUFF_SIZE = 1024 * 256      # IQ buffer size (samples)

# -----------------------------------------------------------------------------
# Frequency
# -----------------------------------------------------------------------------
CENTER_FREQ = 462.61255e6   # Tune frequency (Hz) - FRS channel
CHANNEL_BW = 12500          # NBFM channel bandwidth (Hz)

# -----------------------------------------------------------------------------
# RX Gain (HackRF)
# LNA: 0-40 dB in 8 dB steps (RF amplifier)
# VGA: 0-62 dB in 2 dB steps (IF/baseband amplifier)
# AMP: 0 or 14 dB (external RF amp enable)
# -----------------------------------------------------------------------------
RX_LNA_GAIN = 24            # RF gain - moderate to avoid intermod
RX_VGA_GAIN = 30            # Baseband gain - adjust for signal level
RX_AMP_ENABLE = False       # External 14dB amp - usually not needed

# -----------------------------------------------------------------------------
# TX Gain (HackRF)
# VGA: 0-47 dB (TX IF gain)
# AMP: 0 or 14 dB (TX RF amp enable)
# -----------------------------------------------------------------------------
TX_VGA_GAIN = 30            # TX IF gain
TX_AMP_ENABLE = False        # TX RF amp - BE CAREFUL with power levels
TX_SETTLE_SEC = 0.1         # Pause after start_tx before first IQ write

# -----------------------------------------------------------------------------
# CTCSS (Continuous Tone-Coded Squelch System)
# Required for walkie-talkies to open their squelch
# -----------------------------------------------------------------------------
CTCSS_FREQ = 127.3          # CTCSS tone frequency (Hz) - must match radio
CTCSS_LEVEL = 0.03      # CTCSS tone amplitude (0.0-1.0, typically 0.1-0.2)

# -----------------------------------------------------------------------------
# FM Modulation/Demodulation
# -----------------------------------------------------------------------------
NBFM_DEVIATION = 2500       # NBFM frequency deviation (Hz) - standard ±2.5kHz
DECIMATION = 13             # Decimation factor for RX processing

# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------
AUDIO_RATE = 48000          # Audio sample rate (Hz)
AUDIO_OUTPUT_GAIN = 25.0    # Output gain multiplier for speaker

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
# Voice bandpass
VOICE_HP_CUTOFF = 300       # Highpass cutoff (Hz) - remove sub-bass
VOICE_LP_CUTOFF = 4000      # Lowpass cutoff (Hz) - voice bandwidth

# De-emphasis (RX) / Pre-emphasis (TX)
# 750µs is standard for land mobile radio (RDA1846 chip default)
EMPHASIS_TAU = 750e-6       # Time constant (seconds)

# Channel filter
CHANNEL_FILTER_TAPS = 128   # Number of FIR taps for channel filter

# -----------------------------------------------------------------------------
# Squelch
# -----------------------------------------------------------------------------
SQUELCH_THRESHOLD = 1.5     # IQ power threshold for squelch

# -----------------------------------------------------------------------------
# Transcription (MLX Whisper for Apple Silicon)
# -----------------------------------------------------------------------------
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
MIN_AUDIO_DURATION = 0.5    # Minimum audio duration to transcribe (seconds)
NO_SPEECH_THRESHOLD = 0.6   # Reject if no_speech_prob exceeds this

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
CALLSIGNS = ["Alpha X-Ray 3-1", "Bravo 7", "dispatch"]  # Preamble trigger words
WHISPER_INITIAL_PROMPT = ", ".join(CALLSIGNS) + ". Two-way radio dispatch."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -----------------------------------------------------------------------------
# TTS (Piper — local)
# Download voice: python3 -m piper.download_voices en_US-lessac-medium --download-dir voices
# -----------------------------------------------------------------------------
PIPER_VOICES_DIR = Path(__file__).parent / "voices"
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
TTS_OUTPUT = os.getenv("TTS_OUTPUT", "speakers")  # transmit | speakers | both