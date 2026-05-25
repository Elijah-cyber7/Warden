# Warden

Software-defined radio dispatch pipeline. Receives FM transmissions via HackRF, demodulates and transcribes audio in real time, matches callsigns, queries an AI backend, and transmits the response back over the air.

## What it does

- Receives narrowband FM via HackRF One
- Demodulates NBFM with proper channel filtering and de-emphasis
- Gates audio via power-squelch to suppress noise between transmissions
- Transcribes speech using MLX Whisper (Apple Silicon)
- Matches configurable callsigns/preambles in transcriptions
- Dispatches matched speech to OpenAI and speaks the reply via Piper TTS
- Transmits responses over FM with CTCSS tone (half-duplex)

## Hardware

- HackRF One (TX + RX)
- FRS walkie-talkie (Retevis H777 or similar) tuned to matching channel/CTCSS

## Project Structure

```
Warden/
├── main.py                  # Entry point — starts audio thread and RX loop
├── config.py                # All configuration constants with validation
├── test_tx.py               # TX test tool (tone, WAV, TTS, modulator)
├── requirements.txt         # Python dependencies
├── radio/
│   ├── sdr.py               # HackRF device management via SoapySDR
│   ├── controller.py        # Half-duplex RX/TX coordinator
│   ├── rx.py                # RX loop — squelch, buffering, transcription
│   ├── tx.py                # TX pipeline — modulate and stream IQ
│   ├── demod.py             # FM demodulation (IQ → audio)
│   └── modulator.py         # FM modulation (audio → IQ) with CTCSS
├── audio/
│   ├── player.py            # Audio output queue and playback thread
│   ├── filters.py           # Channel, voice bandpass, de/pre-emphasis filters
│   └── tts.py               # Piper TTS synthesis and routing
├── transcription/
│   └── whisper_engine.py    # MLX Whisper transcription
└── dispatch/
    ├── preamble.py          # Callsign matching and dispatch trigger
    └── openai_client.py     # OpenAI chat API (sync + async)
```

## Setup

```bash
# Create virtual environment with system site packages (required for SoapySDR)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install SoapySDR and HackRF support (macOS)
brew install soapysdr soapyhackrf hackrf
```

For Linux (Debian/Ubuntu):
```bash
sudo apt install python3-soapysdr soapysdr-module-hackrf hackrf
```

### TTS Voice

Download a Piper voice model:
```bash
python3 -m piper.download_voices en_US-amy-medium --download-dir voices
```

### Environment Variables

Create a `.env` file in the `Warden/` directory:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
PIPER_VOICE=en_US-amy-medium
TTS_OUTPUT=transmit
```

`TTS_OUTPUT` options: `transmit` | `speakers` | `both`

## Configuration

All settings live in `config.py` with validation at import time.

| Variable | Default | Description |
|---|---|---|
| `CENTER_FREQ` | 462.61255e6 | Tune frequency (Hz) |
| `SAMPLE_RATE` | 2e6 | SDR sample rate |
| `RX_LNA_GAIN` | 24 | RF gain (0–40 dB, 8 dB steps) |
| `RX_VGA_GAIN` | 30 | Baseband gain (0–62 dB, 2 dB steps) |
| `TX_VGA_GAIN` | 30 | TX IF gain (0–47 dB) |
| `TX_SETTLE_SEC` | 0.3 | Delay after TX stream start before writing IQ |
| `CHANNEL_BW` | 12500 | Channel filter bandwidth (Hz) |
| `SQUELCH_THRESHOLD` | 1.5 | IQ power threshold for squelch |
| `CTCSS_FREQ` | 127.3 | CTCSS tone frequency (Hz) |
| `CTCSS_LEVEL` | 0.03 | CTCSS tone amplitude |
| `NBFM_DEVIATION` | 2500 | FM deviation ±Hz |
| `AUDIO_RATE` | 48000 | Audio sample rate |
| `WHISPER_MODEL` | mlx-community/whisper-large-v3-turbo | Whisper model |
| `CALLSIGNS` | ["Alpha X-Ray 3-1", "Bravo 7", "dispatch"] | Preamble triggers |

## Signal Chain

### RX

```
HackRF RX → IQ (2 MSPS complex64)
  → Decimate ÷13 → 153.8 kSPS
  → Channel lowpass (FIR, 128 taps)
  → Quadrature FM demod: (I·dQ − Q·dI) / (I² + Q²)
  → Resample → 48 kHz (overlap-save)
  → Voice bandpass (300–4000 Hz)
  → De-emphasis (750µs)
  → Squelch gate
  → Buffer → on silence: flush to Whisper (resample → 16 kHz)
  → Callsign match → OpenAI → TTS → TX
```

### TX

```
Audio (48 kHz)
  → Voice bandpass (300–4000 Hz)
  → Pre-emphasis (750µs)
  → Scale voice + add CTCSS tone (127.3 Hz)
  → Resample → intermediate rate (153.8 kHz)
  → FM modulate (phase accumulation, ±2.5 kHz deviation)
  → Channel filter (Carson's rule BW)
  → Interpolate → 2 MSPS
  → Stream to HackRF TX (paced to real-time)
```

The entire transmission (lead-in silence + voice + lead-out silence) is modulated as a single continuous block to prevent CTCSS phase discontinuities.

## Running

```bash
source .venv/bin/activate
python3 main.py
```

### Testing TX

```bash
python3 test_tx.py tone              # 1kHz test tone, 2s
python3 test_tx.py tone 800 3        # 800Hz, 3s
python3 test_tx.py wav voice.wav     # Transmit a WAV file
python3 test_tx.py tts 'Hello world' # Synthesize and transmit
python3 test_tx.py modulator         # Save IQ to file (no SDR needed)
```

### Verify Hardware

```bash
hackrf_info
SoapySDRUtil --find
```

## Status

| Component | Status |
|---|---|
| HackRF RX/TX | Working |
| FM demodulation | Working |
| FM modulation + CTCSS | Working |
| Half-duplex controller | Working |
| Audio playback | Working |
| Squelch | Working |
| MLX Whisper transcription | Working |
| Callsign/preamble matching | Working |
| OpenAI dispatch | Working |
| Piper TTS | Working |
| TX voice response | Working |

## Notes

- Whisper model downloads automatically on first run from Hugging Face
- Virtual environment requires `--system-site-packages` for SoapySDR bindings
- USB cable quality matters — use a known good data cable
- Transcription runs on the RX thread when squelch closes
- Half-duplex: RX pauses during TX and resumes after
- Config validation runs at import time — bad values fail fast with clear messages
