"""Dispatch module for Warden — callsign detection and AI response."""
from dispatch.preamble import check_preamble, extract_message, dispatch
from dispatch.openai_client import chat, chat_async
