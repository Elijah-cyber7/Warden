# radio
SAMPLE_RATE = 2.4e6
CENTER_FREQ = 89.300e6  # swap to 462.550e6 for PTT
GAIN = 78
BUFF_SIZE = 1024 * 256
CHANNEL_BW = 200000  # swap to 12500 for NBFM PTT

# audio
AUDIO_RATE = 48000
SQUELCH = 0.01

# preamble — callsigns that authorize dispatch
CALLSIGNS = ["warden", "dispatch"]  # replace with your actual callsigns