from __future__ import annotations

import asyncio
import base64
import binascii
import contextvars
import io
import json
import re
import wave
from urllib.parse import quote, urlsplit, urlunsplit

from websockets.asyncio.client import connect

from assistant.core.models import Audio
from assistant.services.tts.base import TtsError

_MAX_AUDIO = 24_000 * 2 * 300  # до 5 минут моно PCM16


def _realtime_url(base_url: str, model: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in ("https", "wss")
        or not parsed.hostname
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "FM_BASE_URL должен быть HTTPS/WSS URL без query, fragment и учётных данных"
        )
    return urlunsplit(
        (
            "wss",
            parsed.netloc,
            parsed.path.rstrip("/") + "/realtime",
            "model=" + quote(model, safe="/"),
            "",
        )
    )


_PAREN_TEXT = re.compile(r"\(([^()\n]+)\)")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")


def _clean_cloudru_text(text: str) -> str:
    """Лёгкая чистка для речи; не пересказывает и не выбрасывает содержимое скобок."""
    out = _BOLD.sub(r"\1", text)

    def unpack(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if not re.search(r"[А-Яа-яЁё]", inner) or inner[-1] in ".!?;:":
            return match.group(0)
        return ", " + inner + ", "

    if not re.search(r"\([^()]*\(", out):
        out = _PAREN_TEXT.sub(unpack, out)
        out = re.sub(r"\s+,", ",", out)
        out = re.sub(r",\s*,", ",", out)
        out = re.sub(r",\s*([.!?;:])", r"\1", out)
        out = re.sub(r"^\s*,\s*", "", out)
        out = re.sub(r",\s*$", "", out)
    return re.sub(r"\s+", " ", out).strip()


class CloudRuTts:
    """Изолированные соединения и лёгкая очистка текста для речи."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "GigaChat/SaluteSpeech",
        speaker: str = "Bab-Freespeech",
        language: str = "auto",
        timeout: float = 10.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Для TTS_ADAPTER=cloudru требуется FM_API_KEY")
        if timeout <= 0:
            raise ValueError("SPEECH_TIMEOUT должен быть положительным")
        self._api_key = api_key.strip()
        self._url = _realtime_url(base_url, model)
        self.speaker = speaker
        self.speakers = [speaker]
        self.language = language
        self.timeout = timeout
        self._last_text: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
            "cloudru_tts_last_text", default=None
        )

    def _prepare(self, text: str) -> str:
        return _clean_cloudru_text(text)

    def last_spoken(self, text: str) -> str:
        result = self._last_text.get()
        return result[1] if result and result[0] == text.strip() else text

    async def synthesize(self, text: str, *, speaker: str | None = None) -> Audio:
        text = text.strip()
        if not text:
            return Audio()
        voice = speaker or self.speaker
        try:
            async with asyncio.timeout(self.timeout):
                prepared = self._prepare(text)
                if not prepared:
                    return Audio()
                audio = await self._synthesize(prepared, voice)
                self._last_text.set((text, prepared))
                return audio
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise TtsError("Cloud.ru TTS: превышен таймаут") from exc
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(
                f"Cloud.ru TTS: ошибка соединения или протокола ({type(exc).__name__})"
            ) from exc

    async def _synthesize(self, text: str, voice: str) -> Audio:
        async with connect(
            self._url,
            additional_headers={
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            subprotocols=["realtime"],
            max_size=4 * 1024 * 1024,
            close_timeout=1,
        ) as ws:
            await self._wait_for(ws, "session.created")
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "voice": voice,
                            "output_audio_format": "pcm16",
                            "gigatts": {
                                "voice": voice,
                                "lang": self.language,
                                "content_type": "text",
                            },
                        },
                    },
                    ensure_ascii=False,
                )
            )
            updated = await self._wait_for(ws, "session.updated")
            session = updated.get("session") or {}
            if session.get("output_audio_format", "pcm16") != "pcm16":
                raise TtsError("Cloud.ru TTS: сервер вернул неожиданный формат аудио")
            await ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio"],
                            "instructions": text,
                        },
                    },
                    ensure_ascii=False,
                )
            )
            pcm = bytearray()
            while True:
                message = await ws.recv()
                if isinstance(message, bytes):
                    chunk = message
                else:
                    event = self._event(message)
                    kind = event.get("type")
                    if kind == "response.done":
                        response = event.get("response") or {}
                        if response.get("status") != "completed":
                            raise TtsError("Cloud.ru TTS: синтез не завершён успешно")
                        break
                    if kind != "response.audio.delta":
                        continue
                    try:
                        chunk = base64.b64decode(event.get("delta", ""), validate=True)
                    except (ValueError, TypeError, binascii.Error) as exc:
                        raise TtsError("Cloud.ru TTS: некорректный аудиочанк") from exc
                if len(pcm) + len(chunk) > _MAX_AUDIO:
                    raise TtsError("Cloud.ru TTS: превышен лимит размера аудио")
                pcm.extend(chunk)
            if not pcm or len(pcm) % 2:
                raise TtsError("Cloud.ru TTS: пустое или повреждённое PCM аудио")
            output = io.BytesIO()
            with wave.open(output, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(pcm)
            return Audio(data=output.getvalue(), mime="audio/wav")

    @staticmethod
    def _event(message: str) -> dict:
        event = json.loads(message)
        if not isinstance(event, dict):
            raise TtsError("Cloud.ru TTS: некорректное событие")
        if event.get("type") == "error":
            raise TtsError("Cloud.ru TTS: ошибка провайдера")
        return event

    async def _wait_for(self, ws, wanted: str) -> dict:
        while True:
            message = await ws.recv()
            if not isinstance(message, str):
                raise TtsError("Cloud.ru TTS: неожиданный бинарный кадр")
            event = self._event(message)
            if event.get("type") == wanted:
                return event
