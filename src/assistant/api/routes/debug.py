"""Эмуляция телефонии из браузера. Монтируется только при DEBUG_UI=true

Позволяет отлаживать бота вручную: ввод реплик текстом или аудио,
просмотр состояния сессии, содержимого outbox и проба TTS"""

import base64
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from assistant.core.models import Audio, CallState
from assistant.services.asr.base import AsrError
from assistant.services.asr.fake import FakeAsr
from assistant.services.crm.outbox import OutboxRecord
from assistant.services.tts.base import TtsError
from assistant.services.tts.fake import FakeTts

from ..dependencies import Container, get_container
from ..schemas.dto import DebugConfig, SessionView, TtsRequest, TtsResponse, TurnRequest, TurnResponse
from .telephony import execute_turn

router = APIRouter(prefix="/debug", tags=["debug"])

_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> str:
    return _HTML.read_text(encoding="utf-8")


@router.get("/config", response_model=DebugConfig)
async def config(container: Container = Depends(get_container)) -> DebugConfig:
    """Информация о выбранной конфигурации - подключенных адаптерах"""
    settings = container.settings
    return DebugConfig(
        asr="fake" if isinstance(container.asr, FakeAsr) else settings.asr_adapter,
        tts="fake" if isinstance(container.tts, FakeTts) else settings.tts_adapter,
        llm=settings.llm_adapter if container.orchestrator.llm else "fake",
        crm=settings.crm_adapter,
        demo_phones=container.demo_phones,
    )


@router.post("/call/{call_id}/say", response_model=TurnResponse)
async def say_text(call_id: str, payload: TurnRequest, request: Request,
                   container: Container = Depends(get_container)) -> TurnResponse:
    """Реплика текстом, минуя ASR - для быстрой отладки логики диалога"""
    turn = container.orchestrator.handle_turn(call_id, payload.phone, payload.text)
    return await execute_turn(container, call_id, turn, request, heard=payload.text)


@router.post("/call/{call_id}/say-audio", response_model=TurnResponse)
async def say_audio(call_id: str, request: Request, container: Container = Depends(get_container)) -> TurnResponse:
    """Принимает аудио-реплику, распознаёт через ASR и возвращает ответ бота"""
    body = await request.body()
    mime = request.headers.get("content-type", "audio/pcm")
    phone = request.headers.get("x-phone", "")

    try:
        heard = await container.asr.transcribe(Audio(data=body, mime=mime))
    except AsrError as exc:
        raise HTTPException(502, f"ASR: {exc}") from exc

    if not heard:
        call = container.orchestrator.store.get(call_id)
        state = call.state if call else CallState.GREETING
        return TurnResponse(call_id=call_id, text="", heard="", state=state)

    turn = container.orchestrator.handle_turn(call_id, phone, heard)
    return await execute_turn(container, call_id, turn, request, heard=heard)


@router.get("/call/{call_id}", response_model=SessionView)
async def session(call_id: str, container: Container = Depends(get_container)) -> SessionView:
    """Возвращает полный снимок сессии звонка для просмотра в отладочном UI"""
    call = container.orchestrator.store.get(call_id)
    if call is None:
        raise HTTPException(404, "сессия не найдена")
    return SessionView(
        call_id=call.call_id, phone=call.phone, state=call.state,
        owner=call.owner.full_name if call.owner else None,
        topic=call.topic.name if call.topic else None, slots=call.slots,
        escalated=call.escalated, escalation_markers=call.escalation_markers,
        appeals=call.appeals,
        transcript=[entry.model_dump(mode="json") for entry in call.transcript],
    )


@router.delete("/call/{call_id}", status_code=204)
async def hangup(call_id: str, container: Container = Depends(get_container)) -> None:
    """Завершает звонок - удаляет сессию из хранилища"""
    container.orchestrator.store.delete(call_id)


@router.get("/outbox", response_model=list[OutboxRecord])
async def outbox(container: Container = Depends(get_container)) -> list[OutboxRecord]:
    """Возвращает все записи из outbox - исходящие задачи и сообщения для CRM"""
    return container.outbox.store.all()


@router.post("/tts", response_model=TtsResponse)
async def tts(payload: TtsRequest, container: Container = Depends(get_container)) -> TtsResponse:
    """Озвучивает произвольный текст текущим TTS-адаптером — для подбора голоса и проверки фраз"""
    try:
        audio = await container.tts.synthesize(payload.text)
    except TtsError as exception:
        raise HTTPException(502, f"TTS: {exception}") from exception
    return TtsResponse(
        spoken=payload.text,
        audio_b64=None if audio.empty else base64.b64encode(audio.data).decode(),
        audio_mime=audio.mime,
    )