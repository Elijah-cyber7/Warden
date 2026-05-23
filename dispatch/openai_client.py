import threading

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL

_client = None

DEFAULT_SYSTEM_PROMPT = (
    "You are Warden, a radio dispatch assistant. "
    "The operator has addressed you by callsign over two-way radio. "
    "Respond concisely and clearly, as if replying over the air."
)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set — add it to .env")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def chat(message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Send a message to OpenAI and return the assistant reply."""
    response = get_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content.strip()


def chat_async(message: str, on_reply=None, on_error=None, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """Call OpenAI in a background thread so the RX loop is not blocked."""
    def _run():
        try:
            reply = chat(message, system_prompt=system_prompt)
            if on_reply:
                on_reply(reply)
            else:
                print(f"\n[OPENAI] {reply}\n")
                from audio.tts import speak
                speak(reply)
        except Exception as e:
            if on_error:
                on_error(e)
            else:
                print(f"[OPENAI] API error: {e}")

    threading.Thread(target=_run, daemon=True).start()
