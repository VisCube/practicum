"""Эмуляция телефонии из браузера. Монтируется только при `DEBUG_UI=true`.

Позволяет отлаживать бота вручную: ввод реплик текстом или аудио,
просмотр состояния сессии, содержимого outbox и проба TTS/ASR.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from assistant.core.models import Audio, CallState
from assistant.services.asr.base import AsrError
from assistant.services.asr.fake import FakeAsr
from assistant.services.crm.outbox import OutboxRecord
from assistant.services.tts.base import TtsError
from assistant.services.tts.fake import FakeTts

from ..dependencies import Container, get_container, tts_voice_info
from ..schemas.dto import (
    AsrResponse,
    DebugConfig,
    SessionView,
    TtsRequest,
    TtsResponse,
    TurnRequest,
    TurnResponse,
)
from .telephony import execute_turn

router = APIRouter(prefix="/debug", tags=["debug"])
log = logging.getLogger(__name__)

_MAX_AUDIO_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 МБ (~4+ минуты LPCM 16 kHz)
_HTML_CACHE: tuple[int, int, str, str, str] | None = None  # (mtime_ns, size, html, build, mark)
ContainerDep = Annotated[Container, Depends(get_container)]


def _static_index() -> Path:
    """Какой файл стенда раздавать."""
    source = Path.cwd() / "src" / "assistant" / "api" / "static" / "index.html"
    if source.is_file():
        return source
    return Path(__file__).resolve().parent.parent / "static" / "index.html"


_HTML = _static_index()
_PACKAGED_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


def client_mark(html: str) -> str:
    """Метка версии клиентской логики из самого файла (`const UI_BUILD = '...'`)."""
    marker = "const UI_BUILD = '"
    start = html.find(marker)
    if start < 0:
        return "?"
    start += len(marker)
    end = html.find("'", start)
    return html[start:end] if end > start else "?"


def _load_static_bundle() -> tuple[str, str, str]:
    """Читает `index.html` с кэшем по `(mtime_ns, size)` без повторного чтения диска."""
    global _HTML_CACHE
    st = _HTML.stat()
    if (
        _HTML_CACHE is not None
        and _HTML_CACHE[0] == st.st_mtime_ns
        and _HTML_CACHE[1] == st.st_size
    ):
        return _HTML_CACHE[2], _HTML_CACHE[3], _HTML_CACHE[4]
    raw = _HTML.read_bytes()
    build = hashlib.sha1(raw).hexdigest()[:8]
    html = raw.decode("utf-8")
    mark = client_mark(html)
    _HTML_CACHE = (st.st_mtime_ns, st.st_size, html, build, mark)
    return html, build, mark


def ui_build() -> str:
    """Короткий отпечаток статики: по нему видно, какая сборка открыта в браузере."""
    try:
        _, build, _ = _load_static_bundle()
        return build
    except OSError:
        return "?"


def log_ui_build() -> None:
    """Пишет в лог сервера, какую именно статику он раздаёт."""
    try:
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_HTML.stat().st_mtime))
        _, build, mark = _load_static_bundle()
        log.info(
            "стенд: %s | сборка %s | клиент %s | изменён %s",
            _HTML,
            build,
            mark,
            mtime,
        )
        if mark == "?":
            log.warning(
                "стенд: в раздаваемом файле нет метки версии клиента — "
                "это старая ревизия, пересоберите образ или обновите файл",
            )
    except OSError:
        log.warning("стенд: файл статики не найден — %s", _HTML)
        return
    if _PACKAGED_HTML != _HTML and _PACKAGED_HTML.is_file():
        try:
            packaged = hashlib.sha1(_PACKAGED_HTML.read_bytes()).hexdigest()[:8]
            if packaged != ui_build():
                log.warning(
                    "стенд: копия в пакете (%s) отличается от раздаваемой — "
                    "образ собран из другой ревизии",
                    _PACKAGED_HTML,
                )
        except OSError:
            pass


async def _read_bounded_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        with_int = content_length.strip()
        if with_int.isdigit() and int(with_int) > _MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(413, "аудиофайл слишком большой (лимит 8 МБ)")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(413, "аудиофайл слишком большой (лимит 8 МБ)")
        chunks.append(chunk)
    return b"".join(chunks)


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    # no-store: иначе браузер (и прокси) показывают вчерашний стенд после деплоя
    html, build, mark = _load_static_bundle()
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, max-age=0, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Ui-Build": build,
            "X-Ui-Client": mark,
        },
    )


@router.get("/config", response_model=DebugConfig)
async def config(container: ContainerDep) -> DebugConfig:
    settings = container.settings
    speakers, speaker = tts_voice_info(container.tts)
    llm_name = (
        "cloudru"
        if settings.dialog_adapter == "cloudru"
        else (settings.llm_adapter if container.orchestrator.llm else "fake")
    )
    return DebugConfig(
        dialog=settings.dialog_adapter,
        asr="fake" if isinstance(container.asr, FakeAsr) else settings.asr_adapter,
        tts="fake" if isinstance(container.tts, FakeTts) else settings.tts_adapter,
        llm=llm_name,
        crm=settings.crm_adapter,
        demo_phones=container.demo_phones,
        tts_speakers=speakers,
        tts_speaker=speaker,
        ui_build=ui_build(),
    )


@router.post("/call/{call_id}/say", response_model=TurnResponse)
async def say_text(
    call_id: str,
    payload: TurnRequest,
    request: Request,
    container: ContainerDep,
) -> TurnResponse:
    """Реплика текстом, минуя ASR — для быстрой отладки логики диалога."""
    turn = container.orchestrator.handle_turn(call_id, payload.phone, payload.text)
    return await execute_turn(container, call_id, turn, request, heard=payload.text)


@router.post("/call/{call_id}/say-audio", response_model=TurnResponse)
async def say_audio(
    call_id: str,
    request: Request,
    container: ContainerDep,
) -> TurnResponse:
    """Принимает аудио-реплику, распознаёт через ASR и возвращает ответ бота."""
    body = await _read_bounded_body(request)
    mime = request.headers.get("content-type", "audio/pcm")
    phone = request.headers.get("x-phone", "")

    asr_started = time.perf_counter()
    try:
        heard = await container.asr.transcribe(Audio(data=body, mime=mime))
    except AsrError as exc:
        raise HTTPException(502, f"ASR: {exc}") from exc
    timings = {"asr": int((time.perf_counter() - asr_started) * 1000)}

    if not heard:
        call = container.orchestrator.store.get(call_id)
        state = call.state if call else CallState.GREETING
        return TurnResponse(call_id=call_id, text="", heard="", state=state, timings=timings)

    turn = container.orchestrator.handle_turn(call_id, phone, heard)
    return await execute_turn(container, call_id, turn, request, heard=heard, timings=timings)


@router.get("/call/{call_id}", response_model=SessionView)
async def session(call_id: str, container: ContainerDep) -> SessionView:
    """Возвращает полный снимок сессии звонка для просмотра в отладочном UI."""
    call = container.orchestrator.store.get(call_id)
    if call is None:
        raise HTTPException(404, "сессия не найдена")
    return SessionView(
        call_id=call.call_id,
        phone=call.phone,
        state=call.state,
        owner=call.owner.full_name if call.owner else None,
        topic=call.topic.name if call.topic else None,
        slots=call.slots,
        escalated=call.escalated,
        escalation_markers=call.escalation_markers,
        appeals=call.appeals,
        transcript=[entry.model_dump(mode="json") for entry in call.transcript],
    )


@router.delete("/call/{call_id}", status_code=204)
async def hangup(call_id: str, container: ContainerDep) -> None:
    """Завершает звонок — удаляет сессию из хранилища."""
    container.journal(call_id, "end", reason="stop")
    container.delete_call(call_id)


@router.get("/outbox", response_model=list[OutboxRecord])
async def outbox(container: ContainerDep) -> list[OutboxRecord]:
    """Возвращает все записи из outbox — исходящие задачи и сообщения для CRM."""
    return container.outbox.store.all()


@router.post("/tts", response_model=TtsResponse)
async def tts(payload: TtsRequest, container: ContainerDep) -> TtsResponse:
    try:
        audio = await container.tts.synthesize(payload.text, speaker=payload.speaker)
    except TtsError as exception:
        raise HTTPException(502, f"TTS: {exception}") from exception

    used = payload.speaker or getattr(container.tts, "speaker", None)
    last_spoken = getattr(container.tts, "last_spoken", None)
    spoken = last_spoken(payload.text) if callable(last_spoken) else payload.text
    return TtsResponse(
        spoken=spoken,
        audio_b64=None if audio.empty else base64.b64encode(audio.data).decode(),
        audio_mime=audio.mime,
        speaker=used if isinstance(used, str) else None,
    )


@router.post("/asr", response_model=AsrResponse)
async def asr_probe(
    request: Request,
    container: ContainerDep,
) -> AsrResponse:
    """Проба ASR: lpcm → текст текущим адаптером. Без сессии и бота."""
    body = await _read_bounded_body(request)
    if not body:
        raise HTTPException(400, "empty body")
    mime = request.headers.get("content-type", "audio/pcm")
    started = time.perf_counter()
    try:
        text = await container.asr.transcribe(Audio(data=body, mime=mime))
    except AsrError as exc:
        raise HTTPException(502, f"ASR: {exc}") from exc
    return AsrResponse(text=text or "", ms=int((time.perf_counter() - started) * 1000))


@router.post("/warmup")
async def warmup(container: ContainerDep) -> dict[str, Any]:
    """Прогревает адаптеры: первый ход не должен платить за холодный старт."""
    return await container.warm()
