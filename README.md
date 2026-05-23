# Warden

Warden is a software-defined radio (SDR) dispatch pipeline that receives FM transmissions via a HackRF, demodulates and transcribes audio in real time, and filters transmissions based on a configurable preamble before dispatching to an AI backend.

## What it does

- Receives FM radio transmissions via HackRF One
- Demodulates narrowband FM (NBFM) or wideband FM (WBFM)
- Gates audio via squelch to suppress noise between transmissions
- Transcribes audio using MLX Whisper (Apple Silicon)
- Filters transcriptions against a configurable callsign/preamble list
- Dispatches matched transmissions to an AI API (OpenAI, in progress)

## Hardware

- HackRF One
- Retevis H777 FRS walkie-talkie (channel 10 — 462.5500 MHz, CTCSS 123.0 Hz)

## Project Structure

```
Warden/
├── main.py                  # entry point — starts audio thread and RX loop
├── config.py                # all configuration constants
├── requirements.txt         # Python dependencies
├── radio/
│   ├── sdr.py               # HackRF init, RX/TX streams via SoapySDR
│   ├── rx.py                # RX loop — squelch, buffering, transcription trigger
│   ├── tx.py                # TX pipeline — CTCSS mixing and transmission
│   ├── demod.py             # NBFM demodulation (IQ → audio)
│   └── modulator.py         # NBFM modulation (audio → IQ)
├── audio/
│   ├── player.py            # audio output queue and playback thread
│   ├── filters.py           # channel, voice bandpass, de/pre-emphasis filters
│   └── tones.py             # CTCSS tone generation for TX
├── transcription/
│   └── whisper_engine.py    # MLX Whisper transcription
└── dispatch/
    └── preamble.py          # callsign filter and dispatch stub
```

## Dependencies

```bash
# Create virtual environment with system site packages (required for SoapySDR)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install SoapySDR and HackRF support (macOS)
brew install soapysdr soapyhackrf hackrf ffmpeg
```

For Linux (Debian/Ubuntu):
```bash
sudo apt install python3-soapysdr soapysdr-module-hackrf hackrf ffmpeg
```

Transcription uses [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) and requires Apple Silicon. Install with:

```bash
pip install mlx-whisper
```

## Configuration

All settings live in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `CENTER_FREQ` | 462.61255e6 | Tune frequency in Hz |
| `SAMPLE_RATE` | 2e6 | HackRF sample rate |
| `RX_LNA_GAIN` | 24 | RF gain in dB (0–40, 8 dB steps) |
| `RX_VGA_GAIN` | 30 | Baseband gain in dB (0–62, 2 dB steps) |
| `CHANNEL_BW` | 12500 | Channel filter bandwidth in Hz |
| `SQUELCH_THRESHOLD` | 1.5 | IQ power threshold for squelch |
| `AUDIO_RATE` | 48000 | Audio sample rate |
| `CTCSS_FREQ` | 123.0 | CTCSS tone for TX (Hz) |
| `WHISPER_MODEL` | mlx-community/whisper-large-v3-turbo | Hugging Face model ID |
| `CALLSIGNS` | ["Alpha X-Ray 3-1", "dispatch"] | Preamble trigger words |

### Swapping between broadcast FM and PTT (walkie-talkie)

**Broadcast FM (testing):**
```python
CENTER_FREQ = 89.300e6
CHANNEL_BW = 200000
```

**PTT / Retevis H777 channel 10:**
```python
CENTER_FREQ = 462.550e6
CHANNEL_BW = 12500
```

## Signal chain

### RX (active in `main.py`)

```
HackRF RX
    → IQ samples (2 MSPS, complex64)
    → Decimate by 13 → 153.8 kHz
    → Channel lowpass filter (FIR, 128 taps)
    → FM demodulation (quadrature: I*dQ - Q*dI)
    → Resample 153.8 kHz → 48 kHz (overlap-save)
    → Voice bandpass filter (FIR, 300 Hz – 4000 Hz)
    → De-emphasis (750µs time constant)
    → Squelch gate (IQ power threshold)
    → Audio playback (sounddevice OutputStream, separate thread)
    → Accumulation buffer
    → On silence: flush to Whisper (resample 48 kHz → 16 kHz)
    → Transcription → preamble filter → dispatch (in progress)
```

### TX (implemented, not wired to `main.py`)

```
Audio (48 kHz)
    → Mix with CTCSS tone (123.0 Hz)
    → Pre-emphasis (750µs time constant)
    → Resample 48 kHz → 2 MSPS
    → FM modulation (phase integration)
    → HackRF TX
```

## Running

```bash
source .venv/bin/activate
python3 main.py
```

Plug in HackRF before running. Verify it is detected:

```bash
hackrf_info
SoapySDRUtil --find
```

## Status

| Component | Status |
|---|---|
| HackRF RX | Working |
| FM demodulation | Working |
| Audio playback | Working |
| Squelch | Working |
| MLX Whisper transcription | Working |
| Preamble filter | Stub |
| OpenAI dispatch | Not started |
| TX pipeline (`TXProcessor`) | Implemented, not integrated |
| Voice response | Not started |

## Notes

- Whisper model downloads automatically on first run from Hugging Face
- Virtual environment requires `--system-site-packages` flag for SoapySDR bindings
- USB cable quality affects HackRF detection — use a known good data cable
- Each transcription flush writes `debug.wav` in the project root for debugging
- Transcription runs synchronously on the RX thread when squelch closes a transmission
