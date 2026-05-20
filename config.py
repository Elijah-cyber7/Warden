# radio
SAMPLE_RATE = 2e6
CENTER_FREQ = 462.550e6
GAIN = 60
BUFF_SIZE = 1024 * 128
CHANNEL_BW = 12500

# audio
AUDIO_RATE = 48000
SQUELCH = 0.01  # calibrate after confirming clean audio

# CTCSS
CTCSS_FREQ = 123.0
CTCSS_THRESHOLD = 0.00000001  # calibrate from [CTCSS] terminal output

# preamble
CALLSIGNS = ["warden", "dispatch"]