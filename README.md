# Warden

Warden is a software-defined radio (SDR) dispatch pipeline that receives FM transmissions via a HackRF, demodulates and transcribes audio in real time, and filters transmissions based on a configurable preamble before dispatching to an AI backend.

## What it does

- Receives FM radio transmissions via HackRF One
- Demodulates narrowband FM (NBFM) or wideband FM (WBFM)
- Gates audio via squelch to suppress noise between transmissions
- Transcribes audio using OpenAI Whisper (base model)
- Filters transcriptions against a configurable callsign/preamble list
- Dispatches matched transmissions to an AI API (OpenAI, in progress)

## Hardware

- HackRF One
- Retevis H777 FRS walkie-talkie (channel 10 — 462.5500 MHz, CTCSS 123.0 Hz)

## Project Structure

```
Warden/
├── main.py                  # entry point
├── config.py                # all configuration constants
├── requirements.txt         # Python dependencies
├── radio/
│   ├── sdr.py               # HackRF init and RX loop via SoapySDR
│   └── demod.py             # FM demodulation, filtering, audio accumulation
├── audio/
│   └── player.py            # audio output queue and playback thread
├── transcription/
│   └── whisper_engine.py    # OpenAI Whisper transcription
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

## Configuration

All settings live in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `CENTER_FREQ` | 462.61255e6 | Tune frequency in Hz |
| `SAMPLE_RATE` | 2e6 | HackRF sample rate |
| `GAIN` | 78 | Total RX gain in dB |
| `CHANNEL_BW` | 12500 | Channel filter bandwidth in Hz |
| `SQUELCH` | 1.5 | Signal power threshold |
| `AUDIO_RATE` | 48000 | Audio playback sample rate |
| `CALLSIGNS` | ["warden", "dispatch"] | Preamble trigger words |

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

```
HackRF RX
    → IQ samples (2 MSPS, complex64)
    → Decimate by 13 → 153.8 kHz
    → Channel lowpass filter (FIR, 128 taps)
    → FM demodulation (quadrature: I*dQ - Q*dI)
    → Resample 153.8 kHz → 48 kHz (overlap-save)
    → Voice bandpass filter (FIR, 400 Hz – 4000 Hz)
    → De-emphasis (750µs time constant)
    → Squelch gate (IQ power threshold)
    → Audio playback (sounddevice OutputStream, separate thread)
    → Accumulation buffer
    → On silence: flush to Whisper (resample 48 kHz → 16 kHz)
    → Transcription → preamble filter → dispatch (in progress)
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
| Whisper transcription | Working |
| Preamble filter | Stub |
| OpenAI dispatch | Not started |
| TX / voice response | Not started |

## Notes

- Whisper model downloads automatically on first run (~140MB for base model)
- Virtual environment requires `--system-site-packages` flag for SoapySDR bindings
- USB cable quality affects HackRF detection — use a known good data cable
- For faster transcription with NVIDIA GPU, set `fp16=True` in `whisper_engine.py`
