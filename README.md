# Warden

Warden is a software-defined radio (SDR) dispatch pipeline that receives FM transmissions via a HackRF, demodulates and transcribes audio in real time, and filters transmissions based on a configurable preamble before dispatching to an AI backend.

## What it does

- Receives FM radio transmissions via HackRF One
- Demodulates narrowband FM (NBFM) or wideband FM (WBFM)
- Gates audio via squelch to suppress noise between transmissions
- Streams real-time transcription to the terminal using Vosk
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
├── radio/
│   ├── sdr.py               # HackRF init and RX loop via SoapySDR
│   └── demod.py             # FM demodulation, filtering, audio accumulation
├── audio/
│   └── player.py            # audio output queue and playback thread
├── transcription/
│   └── vosk_engine.py       # real-time Vosk STT engine
└── dispatch/
    └── preamble.py          # callsign filter and dispatch stub
```

## Dependencies

```bash
pip3 install vosk sounddevice numpy scipy
brew install soapysdr soapyhackrf hackrf
```

SoapySDR is installed via Homebrew and requires Python bindings to be accessible from your virtual environment:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

## Configuration

All settings live in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `CENTER_FREQ` | 89.300e6 | Tune frequency in Hz |
| `SAMPLE_RATE` | 2.4e6 | HackRF sample rate |
| `GAIN` | 78 | Total RX gain in dB |
| `CHANNEL_BW` | 200000 | Channel filter bandwidth in Hz |
| `SQUELCH` | 0.01 | Signal power threshold |
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
    → IQ samples (2.4 MSPS, complex64)
    → Channel bandpass filter (FIR, 128 taps)
    → FM demodulation (conjugate multiply + angle)
    → Resample 2.4MHz → 48kHz (resample_poly)
    → Squelch gate (RMS power threshold)
    → Voice bandpass filter (Butterworth order 1, 300Hz–3400Hz)
    → Normalize
    → Audio playback (sounddevice OutputStream, separate thread)
    → Accumulation buffer
    → On silence: flush to Vosk (resample 48kHz → 16kHz)
    → Real-time partial transcription to terminal
    → Final transcription → preamble filter → dispatch (in progress)
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
| Real-time Vosk transcription | Working |
| Preamble filter | Stub |
| OpenAI dispatch | Not started |
| TX / voice response | Not started |

## Notes

- HackRF firmware: 2023.01.1 — update pending
- Vosk model: `vosk-model-small-en-us-0.15` loaded via `Model(lang="en-us")`
- Virtual environment requires `--system-site-packages` flag for SoapySDR bindings
- USB cable quality affects HackRF detection — use a known good data cable
