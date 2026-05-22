# radio
SAMPLE_RATE = 2e6
CENTER_FREQ = 462.61255e6  # swap to 462.550e6 for PTT
GAIN = 78
BUFF_SIZE = 1024 * 256
CHANNEL_BW = 12500  # swap to 12500 for NBFM PTT

# audio
AUDIO_RATE = 48000
SQUELCH = 1.5

# preamble — callsigns that authorize dispatch
CALLSIGNS = ["warden", "dispatch"]  # replace with your actual callsigns