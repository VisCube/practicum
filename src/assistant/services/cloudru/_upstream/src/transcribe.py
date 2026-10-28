from openai import OpenAI

from src.config import FM_API_KEY, FM_BASE_URL, STT_MODEL


def transcribe(data: bytes) -> str:
    client = OpenAI(api_key=FM_API_KEY, base_url=FM_BASE_URL)
    response = client.audio.transcriptions.create(
        model=STT_MODEL,
        file=("audio.wav", data),
        language="ru",
    )
    return response.text
