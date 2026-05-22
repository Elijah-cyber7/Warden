# radio
SAMPLE_RATE = 2e6
CENTER_FREQ = 462.61255e6  # swap to 462.550e6 for PTT
BUFF_SIZE = 1024 * 256
CHANNEL_BW = 12500  # NBFM channel bandwidth

# HackRF gain stages (all in dB)
# LNA (RF/antenna amp): 0-40 dB in 8 dB steps
# VGA (IF/baseband amp): 0-62 dB in 2 dB steps
# AMP (14 dB RF amp): True/False
LNA_GAIN = 24   # moderate RF gain - too high causes intermod
VGA_GAIN = 30   # baseband gain - adjust for signal level
AMP_ENABLE = False  # external 14dB amp - usually not needed

# audio
AUDIO_RATE = 48000

# preamble — callsigns that authorize dispatch
CALLSIGNS = ["warden", "dispatch"]  # replace with your actual callsigns