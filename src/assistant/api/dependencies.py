"""Сборка приложения из адаптеров по Settings — единственное место с импортами адаптеров."""

import base64
import contextlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fastapi import Request

from assistant.core.config import Settings
from assistant.core.models import CallState, Event, Reply, TopicKind
from assistant.core.orchestrator import Orchestrator
from assistant.core.session_store import InMemorySessionStore, SessionStore
from assistant.handlers.consultation import ConsultationHandler
from assistant.handlers.intake import IntakeHandler
from assistant.nlu.classifier import TopicClassifier
from assistant.nlu.escalation import EscalationDetector
from assistant.nlu.llm_tasks import LlmTasks
from assistant.services.asr.base import SpeechRecognizer
from assistant.services.asr.fake import FakeAsr
from assistant.services.crm.base import CrmClient
from assistant.services.crm.fake import DEMO_OWNERS, InMemoryCrm
from assistant.services.crm.outbox import InMemoryOutboxStore, Outbox
from assistant.services.knowledge_base.retriever import MarkdownRetriever
from assistant.services.llm.openai_compat import OpenAICompatLlm
from assistant.services.telephony.fake import FakeTelephony
from assistant.services.tts.base import SpeechSynthesizer, TtsError
from assistant.services.tts.fake import FakeTts

from .schemas.dto import TurnResponse

log = logging.getLogger(__name__)


def _ms(started: float) -> int:
    """Миллисекунды от perf_counter — для таймингов по этапам хода."""
    return int((time.perf_counter() - started) * 1000)


class DialogRunner(Protocol):
    """Контракт обработчика звонка для API (`Orchestrator` или `CloudRuDialogAdapter`)."""

    store: SessionStore

    @property
    def llm(self) -> object: ...

    async def start(self, call_id: str, phone: str) -> Reply: ...
    async def handle_turn(self, call_id: str, phone: str, text: str) -> Reply: ...


def want_audio(request: Request) -> bool:
    """Стенд шлёт X-Audio: off, когда озвучка выключена — не гоняем TTS впустую."""
    return request.headers.get("x-audio", "on").lower() != "off"


def want_tts_speaker(request: Request) -> str | None:
    """X-Tts-Speaker — выбор голоса со стенда, дебаг онли."""
    value = request.headers.get("x-tts-speaker", "").strip()
    return value or None


def tts_voice_info(tts: SpeechSynthesizer) -> tuple[list[str], str | None]:
    """Список голосов и текущий дефолт, если поддерживает адаптер."""
    speakers = list(getattr(tts, "speakers", None) or [])
    speaker = getattr(tts, "speaker", None)
    return speakers, speaker if isinstance(speaker, str) else None


@dataclass
class Container:
    """Собранные зависимости приложения: адаптеры, оркестратор и outbox."""

    settings: Settings
    orchestrator: DialogRunner
    outbox: Outbox
    crm: CrmClient
    asr: SpeechRecognizer
    tts: SpeechSynthesizer
    closeables: list = field(default_factory=list)

    async def to_response(
        self,
        call_id: str,
        reply: Reply,
        heard: str | None = None,
        *,
        with_audio: bool = True,
        tts_speaker: str | None = None,
        timings: dict[str, int] | None = None,
    ) -> TurnResponse:
        audio_b64, mime = None, "audio/ogg"
        stages = dict(timings or {})
        if with_audio and reply.text:
            started = time.perf_counter()
            try:
                audio = await self.tts.synthesize(reply.text, speaker=tts_speaker)
                if not audio.empty:
                    audio_b64, mime = base64.b64encode(audio.data).decode(), audio.mime
            except TtsError as error:
                log.warning("tts failed, fallback to text: %s", error)
            stages["tts"] = _ms(started)
        return TurnResponse(
            call_id=call_id,
            text=reply.text,
            audio_b64=audio_b64,
            audio_mime=mime,
            heard=heard,
            transfer=reply.transfer,
            end_call=reply.end_call,
            state=reply.state,
            events=reply.events,
            timings=stages or None,
        )

    def build_start_init(
        self,
        *,
        call_id: str,
        phone: str,
        state: CallState,
        events: list[Event],
        timings: dict[str, int] | None,
    ) -> dict[str, object]:
        """Собирает прозрачную сводку инициализации звонка для отображения в UI на `/call/start`."""
        from assistant.services.tts.fake import FakeTts

        settings = self.settings
        clean_phone = phone.strip()
        session = self.orchestrator.store.get(call_id)
        owner = session.owner if session else None

        pipeline: dict[str, str] = {
            "dialog": settings.dialog_adapter,
            "asr": "fake" if isinstance(self.asr, FakeAsr) else settings.asr_adapter,
            "tts": "fake" if isinstance(self.tts, FakeTts) else settings.tts_adapter,
            "crm": settings.crm_adapter,
        }
        if settings.dialog_adapter == "cloudru" and settings.llm_model:
            pipeline["llm_model"] = settings.llm_model
        elif getattr(self.orchestrator, "llm", None) and settings.llm_model:
            pipeline["llm_model"] = settings.llm_model

        steps: list[str] = []
        context: dict[str, object]
        if settings.dialog_adapter == "cloudru":
            start_cue = str(getattr(self.orchestrator, "START_CUE", "Алло"))
            tool_names = [
                str(ev.payload.get("name"))
                for ev in events
                if ev.type == "tool_call" and ev.payload.get("name")
            ]
            steps.append(f"Создан агент Assistant(phone={clean_phone!r})")
            steps.append(f"Передан системный промпт + «Звонок с номера {clean_phone}»")
            steps.append(f"Выполнен стартовый ход приветствия (сигнал: {start_cue!r})")
            if tool_names:
                steps.append(f"Вызваны инструменты на старте: {', '.join(tool_names)}")
            elif owner:
                steps.append("Идентификация выполнена через локальный фолбэк CRM-заглушки")
            context = {
                "runner": "CloudRuDialogAdapter → Assistant",
                "system_messages": ["prompts/Assistant.md", f"Звонок с номера {clean_phone}"],
                "start_cue": start_cue,
                "tools_called": tool_names,
            }
        else:
            steps.append(f"Создана сессия звонка (phone={clean_phone!r})")
            steps.append("Поиск карточки собственника: crm.find_owner_by_phone")
            outage_count = sum(1 for ev in events if ev.type == "outage_notice")
            if owner:
                steps.append(
                    f"Проверка аварий по адресу ({owner.address}): найдено {outage_count}",
                )
            context = {
                "runner": "Orchestrator (rules)",
                "crm_lookup": "find_owner_by_phone" + (" + active_outages" if owner else ""),
                "outages_found": outage_count,
            }

        stages = dict(timings or {})
        if "tts" in stages:
            steps.append(f"Синтезировано приветствие ({pipeline['tts']}, {stages['tts']} мс)")

        caller: dict[str, str] | None = None
        if owner:
            caller = {
                "full_name": owner.full_name,
                "address": owner.address,
                "apartment": owner.apartment,
            }

        return {
            "call_id": call_id,
            "phone": clean_phone,
            "state": str(state),
            "identified": owner is not None,
            "caller": caller,
            "pipeline": pipeline,
            "steps": steps,
            "context": context,
            "timings_ms": stages,
        }

    async def warm(self) -> dict[str, int]:
        """Прогревает адаптеры, чтобы первый ход не платил за холодный старт.

        TTS: короткая фраза — первый синтез дороже последующих (ленивая сборка
        графа, кеш нормализатора). LLM: один дешёвый запрос — установка
        соединения и JIT у провайдера. Ошибки не пробрасываем: прогрев
        необязателен, а его сбой не должен ломать звонок.
        """
        stages: dict[str, int] = {}

        started = time.perf_counter()
        try:
            await self.tts.synthesize("Проверка связи")
        except Exception as error:  # noqa: BLE001 — прогрев не должен падать громко
            log.info("warmup tts skipped: %s", error)
        stages["tts"] = _ms(started)

        llm = getattr(self.orchestrator, "llm", None)
        if llm is not None:
            started = time.perf_counter()
            try:
                await llm.complete("Отвечай одним словом.", "скажи: ок")
            except Exception as error:  # noqa: BLE001
                log.info("warmup llm skipped: %s", error)
            stages["llm"] = _ms(started)

        log.info("warmup: %s", stages)
        return stages

    async def aclose(self) -> None:
        """Закрывает адаптеры из closeables — ошибки при завершении не мешают штатному выходу."""
        for closeable in self.closeables:
            with contextlib.suppress(Exception):
                await closeable.aclose()

    def delete_call(self, call_id: str) -> None:
        """Удаляет сессию звонка и связанного с ней диалогового агента."""
        self.orchestrator.store.delete(call_id)
        agents = getattr(self.orchestrator, "_agents", None)
        if isinstance(agents, dict):
            agents.pop(call_id, None)

    @property
    def demo_phones(self) -> list[str]:
        """Список демо-телефонов из активной CRM-заглушки."""
        if self.settings.dialog_adapter == "cloudru":
            from assistant.services.cloudru.bridge import load_cloudru_demo_phones

            return load_cloudru_demo_phones()
        crm = getattr(self.crm, "inner", self.crm)  # TelegramCrm оборачивает InMemoryCrm
        return [o.phone for o in DEMO_OWNERS] if isinstance(crm, InMemoryCrm) else []


def build_container(settings: Settings) -> Container:
    """Собирает контейнер зависимостей по настройкам."""
    closeables: list = []

    use_cloudru_dialog = settings.dialog_adapter == "cloudru"
    classifier = TopicClassifier.from_yaml(
        settings.classifier_path,
        model_path=None if use_cloudru_dialog else Path(settings.classifier_model_path),
        sub_confidence_threshold=settings.classifier_sub_threshold,
        min_confidence=settings.classifier_min_confidence,
    )

    demo_crm = InMemoryCrm()
    crm: CrmClient
    if settings.crm_adapter == "fake":
        crm = demo_crm
    elif settings.crm_adapter == "telegram":
        if not (settings.telegram_bot_token and settings.telegram_chat_id):
            raise RuntimeError("CRM_ADAPTER=telegram: нужны TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID")
        from assistant.services.cloudru.bridge import load_cloudru_slot_labels
        from assistant.services.crm.telegram import TelegramCrm

        slot_labels = {
            **load_cloudru_slot_labels(),
            **{s.id: s.label for s in classifier.slots.values()},
        }
        crm = TelegramCrm(
            demo_crm,
            settings.telegram_bot_token,
            settings.telegram_chat_id,
            channel_username=settings.telegram_channel_username,
            slot_labels=slot_labels,
            timeout=settings.outbox_crm_timeout,
        )
        closeables.append(crm)
    else:
        raise NotImplementedError("crm_adapter=bitrix: клиент Битрикс24 ещё не реализован")

    asr = _build_asr(settings)
    tts = _build_tts(settings)

    outbox = Outbox(
        crm,
        InMemoryOutboxStore(),
        max_attempts=settings.outbox_max_attempts,
        backoff_sec=settings.outbox_backoff_sec,
        crm_timeout=settings.outbox_crm_timeout,
    )
    store = InMemorySessionStore(ttl_sec=settings.session_ttl_sec)
    telephony = FakeTelephony()

    orchestrator: DialogRunner
    if use_cloudru_dialog:
        from assistant.services.cloudru.bridge import export_cloudru_env
        from assistant.services.cloudru.dialog import CloudRuDialogAdapter

        export_cloudru_env(settings, require_rag=True)
        orchestrator = CloudRuDialogAdapter(
            store=store,
            crm=crm,
            outbox=outbox,
            telephony=telephony,
            crm_timeout=settings.crm_timeout,
            agent_timeout=settings.agent_timeout,
            session_ttl_sec=settings.session_ttl_sec,
        )
    else:
        llm_tasks = _build_llm_tasks(settings, closeables)
        escalation = EscalationDetector.from_yaml(settings.escalation_path)
        knowledge_base = MarkdownRetriever.from_dir(settings.knowledge_dir)
        orchestrator = Orchestrator(
            store=store,
            crm=crm,
            classifier=classifier,
            escalation=escalation,
            handlers={
                TopicKind.APPEAL: IntakeHandler(outbox, classifier.slots, llm=llm_tasks),
                TopicKind.CONSULT: ConsultationHandler(knowledge_base, llm_tasks),
            },
            telephony=telephony,
            llm=llm_tasks,
            crm_timeout=settings.crm_timeout,
        )

    return Container(settings, orchestrator, outbox, crm, asr, tts, closeables)


def _build_llm_tasks(settings: Settings, closeables: list) -> LlmTasks | None:
    if settings.llm_adapter in ("openai", "cloudru"):
        base_url = settings.llm_base_url or (
            settings.fm_base_url if settings.llm_adapter == "cloudru" else ""
        )
        api_key = settings.llm_api_key or (
            settings.fm_api_key if settings.llm_adapter == "cloudru" else ""
        )
        if api_key and settings.llm_model:
            client = OpenAICompatLlm(
                base_url, api_key, settings.llm_model, timeout=settings.llm_timeout,
            )
            closeables.append(client)
            return LlmTasks(client)
        log.warning(
            "llm_adapter=%s, но ключ/модель не заданы — работаем без LLM",
            settings.llm_adapter,
        )
    return None


def _build_asr(settings: Settings) -> SpeechRecognizer:
    if settings.asr_adapter == "vosk":
        from assistant.services.asr.vosk import VoskAsr

        return VoskAsr(settings.vosk_model_path, sample_rate=settings.asr_sample_rate)
    if settings.asr_adapter == "t-one":
        from assistant.services.asr.tone import ToneAsr

        return ToneAsr(
            settings.tone_model_dir,
            input_sample_rate=settings.asr_sample_rate,
            num_threads=settings.tone_num_threads,
        )
    if settings.asr_adapter == "cloudru":
        from assistant.services.asr.cloudru import CloudRuAsr
        from assistant.services.cloudru.bridge import export_cloudru_env

        export_cloudru_env(settings, require_rag=settings.dialog_adapter == "cloudru")
        return CloudRuAsr(
            sample_rate=settings.asr_sample_rate,
            timeout=settings.speech_timeout,
            use_itn=settings.asr_itn,
        )
    return FakeAsr()


def _build_tts(settings: Settings) -> SpeechSynthesizer:
    if settings.tts_adapter == "piper":
        from assistant.services.tts.piper import PiperTts

        return PiperTts(settings.piper_model_path, None)
    if settings.tts_adapter == "silero":
        from assistant.services.tts.silero import SileroTts

        return SileroTts(
            settings.silero_model_path,
            speaker=settings.silero_speaker or None,
            sample_rate=settings.silero_sample_rate,
            device=settings.silero_device,
            put_accent=settings.silero_put_accent,
            put_yo=settings.silero_put_yo,
            normalize=settings.tts_normalize,
            stress=settings.tts_stress,
            use_ssml=settings.silero_ssml,
            rate=settings.silero_rate,
            pitch=settings.silero_pitch,
        )
    return FakeTts()


def get_container(request: Request) -> Container:
    return request.app.state.container
