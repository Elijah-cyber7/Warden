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
CENTER_FREQ = 462.6125e6   # Tune frequency (Hz) — FRS channel
CHANNEL_BW = 12500          # NBFM channel bandwidth (Hz)

# -----------------------------------------------------------------------------
# RX Gain (HackRF)
# LNA: 0-40 dB in 8 dB steps (RF amplifier)
# VGA: 0-62 dB in 2 dB steps (IF/baseband amplifier)
# AMP: 0 or 14 dB (external RF amp enable)
# -----------------------------------------------------------------------------
RX_LNA_GAIN = 24            # RF gain — moderate to avoid intermod
RX_VGA_GAIN = 30            # Baseband gain — adjust for signal level
RX_AMP_ENABLE = False       # External 14dB amp — usually not needed

# -----------------------------------------------------------------------------
# TX Gain (HackRF)
# VGA: 0-47 dB (TX IF gain)
# AMP: 0 or 14 dB (TX RF amp enable)
# -----------------------------------------------------------------------------
TX_VGA_GAIN = 0             # TX IF gain (0 dB default — bench testing with
                            # a nearby walkie-talkie; raise only if needed)
TX_AMP_ENABLE = False       # TX RF amp — BE CAREFUL with power levels
TX_SETTLE_SEC = 0.6         # Pause after start_tx before first IQ write
                            # (PLL lock time varies — needs headroom)
TX_LEAD_IN_SEC = 0.8        # CTCSS-only carrier before voice
                            # (first ~300ms may be unstable post-PLL lock)
TX_LEAD_OUT_SEC = 0.5       # CTCSS-only carrier after voice

# -----------------------------------------------------------------------------
# CTCSS (Continuous Tone-Coded Squelch System)
# Required for walkie-talkies to open their squelch.
#
# Effective FM deviation contributed by the tone:
#     CTCSS deviation (Hz) = NBFM_DEVIATION * CTCSS_LEVEL
#
# The "standard" commercial spec is 300–750 Hz, but many consumer FRS/GMRS
# handhelds have very narrow sub-audible bandpasses and actually want a much
# smaller CTCSS injection — pushing it too hard makes the decoder treat the
# tone as voice and reject it. 0.03 * 2500 = 75 Hz, which matches what this
# radio responds to. If you change radios, tune this empirically.
# -----------------------------------------------------------------------------
CTCSS_FREQ = 127.3          # CTCSS tone frequency (Hz) — must match radio
CTCSS_LEVEL = 0.03          # Tone amplitude relative to full deviation

# -----------------------------------------------------------------------------
# FM Modulation/Demodulation
# -----------------------------------------------------------------------------
NBFM_DEVIATION = 2500       # NBFM frequency deviation (Hz) — standard ±2.5kHz
DECIMATION = 13             # Decimation factor for RX processing

# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------
AUDIO_RATE = 48000          # Audio sample rate (Hz)
AUDIO_OUTPUT_GAIN = 25.0    # Output gain multiplier for speaker

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
VOICE_HP_CUTOFF = 300       # Highpass cutoff (Hz) — remove sub-bass
VOICE_LP_CUTOFF = 4000      # Lowpass cutoff (Hz) — voice bandwidth
EMPHASIS_TAU = 750e-6       # De/pre-emphasis time constant (seconds)
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
ASSISTANT_NAME = "Jarvis"   # How the assistant identifies itself in replies
CALLSIGNS = ["Alpha X-Ray 3-1", "Bravo 7", "dispatch", ASSISTANT_NAME]
WHISPER_INITIAL_PROMPT = ", ".join(CALLSIGNS)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -----------------------------------------------------------------------------
# TTS (Piper — local)
# Download voice: python3 -m piper.download_voices en_US-amy-medium --download-dir voices
# -----------------------------------------------------------------------------
PIPER_VOICES_DIR = Path(__file__).parent / "voices"
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-amy-medium")
TTS_OUTPUT = os.getenv("TTS_OUTPUT", "transmit")  # transmit | speakers | both


# =============================================================================
# Validation — fail fast on bad config
# =============================================================================
def _validate():
    errors = []
    if not (400e6 <= CENTER_FREQ <= 6000e6):
        errors.append(f"CENTER_FREQ={CENTER_FREQ/1e6:.1f}MHz — outside 400-6000 MHz range")
    if not (0 <= RX_LNA_GAIN <= 40):
        errors.append(f"RX_LNA_GAIN={RX_LNA_GAIN} — must be 0-40")
    if not (0 <= RX_VGA_GAIN <= 62):
        errors.append(f"RX_VGA_GAIN={RX_VGA_GAIN} — must be 0-62")
    if not (0 <= TX_VGA_GAIN <= 47):
        errors.append(f"TX_VGA_GAIN={TX_VGA_GAIN} — must be 0-47")
    if not (0.01 <= CTCSS_LEVEL <= 0.5):
        errors.append(f"CTCSS_LEVEL={CTCSS_LEVEL} — should be 0.01-0.5")
    if not (67 <= CTCSS_FREQ <= 254):
        errors.append(f"CTCSS_FREQ={CTCSS_FREQ} — standard range is 67-254 Hz")
    # Note: no upper-bound check on CTCSS deviation here. Different radios
    # accept very different injection levels; the right value is empirical.
    if TX_LEAD_IN_SEC < 0:
        errors.append(f"TX_LEAD_IN_SEC={TX_LEAD_IN_SEC} — must be >= 0")
    if TX_LEAD_OUT_SEC < 0:
        errors.append(f"TX_LEAD_OUT_SEC={TX_LEAD_OUT_SEC} — must be >= 0")
    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))


_validate()
