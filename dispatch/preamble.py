"""
Dispatch module for Warden.

Handles callsign detection (preamble matching) and OpenAI API interaction.
"""

import logging
import re

from config import CALLSIGNS
from dispatch.openai_client import chat_async

log = logging.getLogger("warden.dispatch")

_bridge = None


def set_bridge(bridge):
    """Register the GUI bridge for transcription notifications."""
    global _bridge
    _bridge = bridge

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

_pattern = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in CALLSIGNS) + r')\s*[,:\-]?',
    re.IGNORECASE,
)


def normalize_transcript(text: str) -> str:
    """Normalize spoken numbers so 'Bravo seven' matches 'Bravo 7'."""
    normalized = text
    for word, digit in _NUMBER_WORDS.items():
        normalized = re.sub(rf'\b{word}\b', digit, normalized, flags=re.IGNORECASE)
    return normalized


def check_preamble(text: str):
    """Check if transcript contains a callsign and dispatch if so."""
    normalized = normalize_transcript(text)
    matched = bool(_pattern.search(normalized))

    if _bridge:
        _bridge.emit_transcription(text, matched)

    if matched:
        log.info("Preamble matched: '%s'", text)
        dispatch(normalized)
    else:
        log.debug("No preamble: '%s'", text)


def extract_message(text: str) -> str | None:
    """Return speech after the matched callsign, or None if not found."""
    match = _pattern.search(text)
    if not match:
        return None
    remainder = text[match.end():].strip()
    remainder = re.sub(r'^[,:\-\s]+', '', remainder)
    return remainder or None


def dispatch(text: str):
    """Extract message from transcript and send to OpenAI."""
    message = extract_message(text)
    if not message:
        log.warning("No speech after callsign in: '%s'", text)
        return

    log.info("Dispatching: '%s'", message)
    chat_async(message)
