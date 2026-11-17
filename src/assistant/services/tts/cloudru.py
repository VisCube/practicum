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

from websockets.asyncio.client import ClientConnection, connect
from websockets.protocol import State

from assistant.core.models import Audio
from assistant.services.tts.base import TtsError

_MAX_AUDIO = 24_000 * 2 * 300  # до 5 минут моно PCM16
_MAX_CONNECTIONS = 16
_IDLE_TIMEOUT = 60.0


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
    """Пул изолированных по звонку соединений и лёгкая очистка текста."""

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
        self._slots = asyncio.Semaphore(_MAX_CONNECTIONS)
        self._idle: dict[ClientConnection, tuple[str, str, asyncio.TimerHandle]] = {}
        self._busy: dict[ClientConnection, str | None] = {}
        self._invalid: set[ClientConnection] = set()
        self._connecting: dict[asyncio.Task, str | None] = {}
        self._invalid_connects: set[asyncio.Task] = set()
        self._closing: set[asyncio.Task[None]] = set()
        self._closed = False
        self._last_text: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
            "cloudru_tts_last_text", default=None
        )

    def _prepare(self, text: str) -> str:
        return _clean_cloudru_text(text)

    def last_spoken(self, text: str) -> str:
        result = self._last_text.get()
        return result[1] if result and result[0] == text.strip() else text

    async def synthesize(self, text: str, *, speaker: str | None = None) -> Audio:
        # Пробы и прогрев не имеют идентификатора звонка — их не переиспользуем.
        return await self._synthesize_request(text, speaker=speaker, call_id=None)

    async def synthesize_for_call(
        self, text: str, *, call_id: str, speaker: str | None = None
    ) -> Audio:
        return await self._synthesize_request(text, speaker=speaker, call_id=call_id or None)

    async def _synthesize_request(
        self, text: str, *, speaker: str | None, call_id: str | None
    ) -> Audio:
        text = text.strip()
        if not text:
            return Audio()
        voice = speaker or self.speaker
        try:
            async with asyncio.timeout(self.timeout):
                prepared = self._prepare(text)
                if not prepared:
                    return Audio()
                audio = await self._synthesize(prepared, voice, call_id)
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

    async def _connect(self, voice: str) -> ClientConnection:
        ws = await connect(
            self._url,
            additional_headers={
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            subprotocols=["realtime"],
            max_size=4 * 1024 * 1024,
            close_timeout=1,
        )
        try:
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
            return ws
        except BaseException:
            await ws.close()
            raise

    def _close_later(self, ws: ClientConnection) -> None:
        task = asyncio.create_task(ws.close())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    def _expire(self, ws: ClientConnection) -> None:
        item = self._idle.pop(ws, None)
        if item is not None:
            item[2].cancel()
            self._close_later(ws)

    def drop_call(self, call_id: str) -> None:
        """Не возвращать в пул соединения завершённого звонка."""
        for ws, (owner, _, timer) in list(self._idle.items()):
            if owner == call_id:
                self._idle.pop(ws)
                timer.cancel()
                self._close_later(ws)
        self._invalid.update(ws for ws, owner in self._busy.items() if owner == call_id)
        self._invalid_connects.update(
            task for task, owner in self._connecting.items() if owner == call_id
        )

    async def aclose(self) -> None:
        self._closed = True
        sockets = list(self._idle) + list(self._busy)
        for _, _, timer in self._idle.values():
            timer.cancel()
        self._idle.clear()
        if sockets:
            await asyncio.gather(*(ws.close() for ws in sockets), return_exceptions=True)
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)

    async def _synthesize(self, text: str, voice: str, call_id: str | None) -> Audio:
        if self._closed:
            raise TtsError("Cloud.ru TTS: адаптер закрыт")
        async with self._slots:
            if self._closed:
                raise TtsError("Cloud.ru TTS: адаптер закрыт")
            ws: ClientConnection | None = None
            if call_id:
                for candidate, (owner, old_voice, timer) in list(self._idle.items()):
                    if candidate.state is not State.OPEN:
                        self._idle.pop(candidate)
                        timer.cancel()
                        self._close_later(candidate)
                    elif owner == call_id and old_voice == voice:
                        self._idle.pop(candidate)
                        timer.cancel()
                        ws = candidate
                        break
            if ws is None:
                task = asyncio.current_task()
                assert task is not None
                self._connecting[task] = call_id
                try:
                    ws = await self._connect(voice)
                    if self._closed or task in self._invalid_connects:
                        await ws.close()
                        raise TtsError("Cloud.ru TTS: звонок или адаптер закрыт")
                finally:
                    self._connecting.pop(task, None)
                    self._invalid_connects.discard(task)
            self._busy[ws] = call_id
            success = False
            try:
                audio = await self._receive_audio(ws, text)
                success = True
                return audio
            finally:
                self._busy.pop(ws, None)
                invalid = ws in self._invalid
                self._invalid.discard(ws)
                if (
                    success
                    and call_id
                    and not invalid
                    and not self._closed
                    and ws.state is State.OPEN
                ):
                    if len(self._idle) >= _MAX_CONNECTIONS:
                        oldest = next(iter(self._idle))
                        self._expire(oldest)
                    timer = asyncio.get_running_loop().call_later(_IDLE_TIMEOUT, self._expire, ws)
                    self._idle[ws] = (call_id, voice, timer)
                else:
                    await ws.close()

    async def _receive_audio(self, ws: ClientConnection, text: str) -> Audio:
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
            # Текст ошибки провайдера не логируем: он может содержать запрос.
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
