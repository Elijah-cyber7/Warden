import re

from config import CALLSIGNS
from dispatch.openai_client import chat_async

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

# Match callsign followed by optional comma/colon (common in radio speech)
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
    normalized = normalize_transcript(text)
    if _pattern.search(normalized):
        print(f"[PREAMBLE MATCHED] '{text}' — dispatching to API")
        dispatch(normalized)
    else:
        print(f"[PREAMBLE NOT MATCHED] '{text}' — discarding")


def extract_message(text: str) -> str | None:
    """Return speech after the matched callsign, or None if no callsign found."""
    match = _pattern.search(text)
    if not match:
        return None
    remainder = text[match.end():].strip()
    remainder = re.sub(r'^[,:\-\s]+', '', remainder)
    return remainder or None


def dispatch(text: str):
    message = extract_message(text)
    if not message:
        print("[DISPATCH] No speech after callsign")
        return

    print(f"[DISPATCH] Sending: '{message}'")
    chat_async(message)
