"""Управление телефонией: старт звонка, реплики клиента"""

import logging
from collections.abc import Awaitable
from uuid import uuid4

from fastapi import APIRouter, Depends, Request

from assistant.core.models import CallState, Event, Reply
from assistant.templates.phrases import say

from ..dependencies import Container, get_container, want_audio, want_tts_speaker
from ..schemas.dto import StartRequest, TurnRequest, TurnResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/call", tags=["telephony"])


async def execute_turn(container: Container, call_id: str, turn: Awaitable[Reply], request: Request,
                       heard: str | None = None) -> TurnResponse:
    """Общая обёртка для /start и /turn: запросы одного звонка идут по очереди, при любом сбое - перевод на оператора"""
    store = container.orchestrator.store
    try:
        async with store.lock(call_id):
            reply = await turn
    except Exception:  # noqa: BLE001 — граница системы, здесь ловим всё
        log.exception("turn failed for call %s", call_id)
        reply = Reply(text=say("tech_failure"), transfer=True, state=CallState.TRANSFERRED,
                      events=[Event(type="transfer", payload={"reason": "техническая ошибка"})])
        if session := store.get(call_id):
            session.state = CallState.TRANSFERRED
            session.say(reply.text)
            store.save(session)
    if reply.end_call:
        store.delete(call_id)
    return await container.to_response(
        call_id, reply, heard=heard,
        with_audio=want_audio(request),
        tts_speaker=want_tts_speaker(request),
    )


@router.post("/start", response_model=TurnResponse)
async def start(payload: StartRequest, request: Request,
                container: Container = Depends(get_container)) -> TurnResponse:
    """Начало звонка: заводим сессию, робот приветствует пользователя"""
    call_id = payload.call_id or uuid4().hex[:12]
    return await execute_turn(container, call_id, container.orchestrator.start(call_id, payload.phone), request)


@router.post("/turn", response_model=TurnResponse)
async def turn(payload: TurnRequest, request: Request,
               container: Container = Depends(get_container)) -> TurnResponse:
    """Обработка роботом реплики клиента"""
    return await execute_turn(container, payload.call_id,
                              container.orchestrator.handle_turn(payload.call_id, payload.phone, payload.text), request)