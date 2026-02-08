import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_TEXT = os.getenv("GEMINI_MODEL_TEXT", "gemini-1.5-flash")
GEMINI_MODEL_VISION = os.getenv("GEMINI_MODEL_VISION", "gemini-1.5-flash")
GEMINI_MODEL_EMBED = os.getenv("GEMINI_MODEL_EMBED", "models/text-embedding-004")

USE_VISION_DEFAULT = os.getenv("USE_VISION", "false").lower() in ("1", "true", "yes")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))


def require_api_key() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
