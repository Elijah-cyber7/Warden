# Warden

An AI-powered radio dispatch system built on software-defined radio. Warden listens on a configured frequency, transcribes incoming voice transmissions in real time, detects operator callsigns, queries an LLM for a response, synthesizes speech locally, and transmits the reply back over the air — all hands-free.

Think of it as a voice-activated AI assistant that lives on a radio channel.

## How It Works

```
Operator keys up radio        Warden (HackRF One)
        |                            |
        |  ── FM voice + CTCSS ──▶   |  Receive & demodulate
        |                            |  Transcribe (Whisper)
        |                            |  Detect callsign
        |                            |  Query OpenAI
        |                            |  Synthesize reply (Piper TTS)
        |  ◀── FM voice + CTCSS ──   |  Transmit response
        |                            |
```

The system operates half-duplex: it pauses receiving while transmitting, then resumes listening.

## Features

- **Full-duplex DSP** — Narrowband FM modulation/demodulation with proper channel filtering, de/pre-emphasis, and CTCSS tone coding
- **Real-time transcription** — MLX Whisper on Apple Silicon for low-latency speech-to-text
- **Callsign matching** — Configurable trigger words/phrases with spoken-number normalization
- **AI dispatch** — OpenAI chat completions with a radio-operator persona
- **Local TTS** — Piper voice synthesis, routable to radio TX, speakers, or both
- **Desktop GUI** — PySide6 interface with live spectrum plot, signal meter, transcript log, and configuration controls
- **Built-In Test (BIT)** — Automated hardware diagnostics for USB throughput and TX FIFO integrity
- **Headless mode** — Runs without a display for embedded/remote deployments

## Hardware

| Component | Role |
|---|---|
| HackRF One | SDR transceiver (RX + TX) |
| UHF handheld radio | Operator's radio — any NBFM radio with matching frequency and CTCSS tone |

A basic FRS/GMRS walkie-talkie (Retevis H777, Baofeng, etc.) works for bench testing. For legal operation on amateur bands, use a radio that covers your licensed frequencies.

## Quick Start

### Prerequisites

- Python 3.11+
- macOS (Apple Silicon recommended for MLX Whisper) or Linux
- HackRF One connected via USB
- An OpenAI API key

### Installation

```bash
cd Warden

# Virtual environment (--system-site-packages required for SoapySDR bindings)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Python dependencies
pip install -r requirements.txt

# System packages — macOS
brew install soapysdr soapyhackrf hackrf

# System packages — Debian/Ubuntu
# sudo apt install python3-soapysdr soapysdr-module-hackrf hackrf
```

### TTS Voice

```bash
python3 -m piper.download_voices en_US-amy-medium --download-dir voices
```

### Environment

Create a `.env` file in the `Warden/` directory:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
PIPER_VOICE=en_US-amy-medium
TTS_OUTPUT=transmit
```

`TTS_OUTPUT` controls where synthesized speech goes: `transmit` (over the air), `speakers` (local playback), or `both`.

### Run

```bash
# Headless
python3 main.py

# With GUI
python3 main.py --gui
```

### Verify Hardware

```bash
hackrf_info          # Confirm HackRF is detected
SoapySDRUtil --find  # Confirm SoapySDR sees the device
```

## Configuration

All parameters live in `config.py` with validation at import time.

### Radio

| Parameter | Default | Description |
|---|---|---|
| `CENTER_FREQ` | 462.6125 MHz | Tune frequency |
| `CHANNEL_BW` | 12,500 Hz | Channel filter bandwidth |
| `NBFM_DEVIATION` | ±2,500 Hz | FM deviation |
| `CTCSS_FREQ` | 127.3 Hz | Sub-audible tone (must match operator radio) |
| `CTCSS_LEVEL` | 0.03 | Tone amplitude relative to full deviation |

### Gains

| Parameter | Default | Range | Description |
|---|---|---|---|
| `RX_LNA_GAIN` | 24 dB | 0–40 (8 dB steps) | RF front-end gain |
| `RX_VGA_GAIN` | 30 dB | 0–62 (2 dB steps) | Baseband gain |
| `TX_VGA_GAIN` | 0 dB | 0–47 | TX IF gain |
| `RX_AMP_ENABLE` | Off | — | 14 dB external RF amp |
| `TX_AMP_ENABLE` | Off | — | 14 dB TX RF amp |

### Timing

| Parameter | Default | Description |
|---|---|---|
| `TX_SETTLE_SEC` | 0.3 s | Delay after activating TX stream before writing IQ |
| `TX_LEAD_IN_SEC` | 0.4 s | CTCSS-only carrier before voice (opens receiver squelch) |
| `TX_LEAD_OUT_SEC` | 0.5 s | CTCSS-only carrier after voice (clean tail) |

### Dispatch

| Parameter | Default | Description |
|---|---|---|
| `CALLSIGNS` | `["Alpha X-Ray 3-1", "Bravo 7", "dispatch", "Jarvis"]` | Phrases that trigger AI dispatch |
| `WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | Whisper model (auto-downloads) |
| `ASSISTANT_NAME` | `Jarvis` | How the AI identifies itself on air |

## Architecture

```
Warden/
├── main.py                  Entry point — headless or GUI mode
├── config.py                All settings with validation
├── test_tx.py               TX test utility
├── radio/
│   ├── sdr.py               HackRF device management (SoapySDR)
│   ├── controller.py        Half-duplex RX/TX coordinator
│   ├── rx.py                Receive loop — squelch, buffer, transcribe
│   ├── tx.py                Transmit pipeline — modulate and stream
│   ├── demod.py             FM demodulation (IQ → audio)
│   └── modulator.py         FM modulation (audio → IQ) with CTCSS
├── audio/
│   ├── player.py            Audio output queue and playback
│   ├── filters.py           Channel, voice bandpass, de/pre-emphasis filters
│   └── tts.py               Piper TTS synthesis and routing
├── transcription/
│   └── whisper_engine.py    MLX Whisper speech-to-text
├── dispatch/
│   ├── preamble.py          Callsign matching and dispatch trigger
│   └── openai_client.py     OpenAI chat API wrapper
├── bit/
│   ├── runner.py            BIT orchestration
│   ├── usb_health.py        USB throughput test
│   └── fifo_integrity.py    TX FIFO write-integrity test
└── gui/
    ├── app.py               Main window layout
    ├── bridge.py            Thread-safe Qt signal bridge
    ├── spectrum.py          Live PSD plot (pyqtgraph)
    ├── signal_meter.py      RF signal level indicator
    ├── status_bar.py        RX/TX mode display
    ├── transcript_panel.py  Radio transcript log
    ├── log_panel.py         Application log viewer
    ├── config_panel.py      Live gain/parameter controls
    └── bit_panel.py         Built-In Test controls and results
```

## Signal Chain

### Receive

```
HackRF RX → IQ (2 MSPS complex64)
  → Decimate ÷13 → 153.8 kSPS
  → Channel lowpass (FIR, 128 taps)
  → Quadrature FM demod
  → Resample → 48 kHz
  → Voice bandpass (300–4000 Hz)
  → De-emphasis (750 µs)
  → Power squelch gate
  → Buffer → flush on silence → Whisper (16 kHz)
  → Callsign match → OpenAI → TTS → TX
```

### Transmit

```
Audio (48 kHz)
  → Voice bandpass (300–4000 Hz)
  → Pre-emphasis (750 µs)
  → Scale + inject CTCSS tone
  → Resample → 153.8 kHz intermediate
  → FM modulate (phase accumulation, ±2.5 kHz)
  → Resample → 2 MSPS
  → Stream to HackRF TX (backpressure-paced)
```

The full transmission (lead-in + voice + lead-out) is modulated as a single continuous block to maintain CTCSS phase continuity.

## Testing TX

```bash
python3 test_tx.py tone              # 1 kHz test tone, 2 seconds
python3 test_tx.py tone 800 3        # 800 Hz tone, 3 seconds
python3 test_tx.py wav voice.wav     # Transmit a WAV file
python3 test_tx.py tts "Hello world" # Synthesize and transmit
python3 test_tx.py modulator         # Save modulated IQ to file (no SDR needed)
```

## Built-In Tests

The BIT system runs hardware diagnostics from the GUI or programmatically:

| Test | What it checks |
|---|---|
| **USB Health** | Sustained write throughput (expects >= 3.5 MB/s) and error rate |
| **FIFO Integrity** | 2-second continuous stream with zero write failures (stalls = carrier dropouts) |

## Legal Notice

Transmitting on any frequency requires appropriate authorization. In the US:

- **FRS channels** (462/467 MHz) are limited to type-accepted radios — the HackRF is not type-accepted
- **Amateur (ham) bands** require an FCC license but allow homebrew transmitters
- **For bench testing**, keep TX power at minimum and use a dummy load or keep the antenna off

This project is intended for licensed amateur radio operators and educational use. Know and follow your local regulations.

## Dependencies

- **numpy / scipy** — DSP (filtering, resampling, modulation)
- **SoapySDR** — Hardware abstraction for HackRF
- **mlx-whisper** — Speech recognition (Apple Silicon optimized)
- **openai** — LLM API client
- **piper-tts** — Local neural text-to-speech
- **PySide6 / pyqtgraph** — Desktop GUI and spectrum visualization
- **sounddevice** — Audio playback
- **python-dotenv** — Environment configuration
