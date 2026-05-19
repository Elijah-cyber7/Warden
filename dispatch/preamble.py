import re
from config import CALLSIGNS

# build pattern from configured callsigns
_pattern = re.compile(
    r'\b(' + '|'.join(CALLSIGNS) + r')\b',
    re.IGNORECASE
)


def check_preamble(text: str):
    if _pattern.search(text):
        print(f"[PREAMBLE MATCHED] '{text}' — dispatching to API")
        dispatch(text)
    else:
        print(f"[PREAMBLE NOT MATCHED] '{text}' — discarding")


def dispatch(text: str):
    # OpenAI call goes here next
    print(f"[DISPATCH] {text}")