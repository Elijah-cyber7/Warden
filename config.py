# radio
SAMPLE_RATE = 2.4e6
CENTER_FREQ = 462.550e6  # swap to 462.550e6 for PTT
GAIN = 60
BUFF_SIZE = 1024 * 256
CHANNEL_BW = 12500  # swap to 12500 for NBFM PTT

# audio
AUDIO_RATE = 48000
SQUELCH = 0.01

#CTCSS
CTCSS_FREQ = 123.0
CTCSS_THRESHOLD = 0.0001

# preamble — callsigns that authorize dispatch
CALLSIGNS = ["warden", "dispatch"]  # replace with your actual callsigns