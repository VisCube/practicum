from os import environ
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "assistant.md"
).read_text(encoding="utf-8")

FM_API_KEY = environ["FM_API_KEY"]
FM_BASE_URL = environ["FM_BASE_URL"]
LLM_MODEL = environ["LLM_MODEL"]
STT_MODEL = environ["STT_MODEL"]

RAG_KEY_ID = environ["RAG_KEY_ID"]
RAG_KEY_SECRET = environ["RAG_KEY_SECRET"]
RAG_BASE_URL = environ["RAG_BASE_URL"]

IAM_TOKEN_URL = environ["IAM_TOKEN_URL"]
