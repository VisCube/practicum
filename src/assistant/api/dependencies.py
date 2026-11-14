"""Сборка приложения из адаптеров по Settings — единственное место с импортами адаптеров."""

from __future__ import annotations

import base64
import contextlib
import logging
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from fastapi import Request

from assistant.core.config import Settings
from assistant.core.models import Audio, CallState, Event, Reply, TopicKind
from assistant.core.orchestrator import Orchestrator
from assistant.core.session_store import InMemorySessionStore, SessionStore
from assistant.handlers.consultation import ConsultationHandler
from assistant.handlers.intake import IntakeHandler
from assistant.nlu.classifier import TopicClassifier
from assistant.nlu.escalation import EscalationDetector
from assistant.nlu.itn import warm_itn
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

# 100 мс тишины lpcm16 16 кГц: проходит все проверки длины и греет модель,
# но не тратит время на реальное распознавание.
_WARMUP_AUDIO = b"\x00" * 3200


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
    closeables: list[Any] = field(default_factory=list)

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
            except Exception as error:
                log.exception("unexpected tts error, fallback to text: %s", error)
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
        has_llm = settings.dialog_adapter == "cloudru" or bool(
            getattr(self.orchestrator, "llm", None)
        )
        if has_llm and settings.llm_model:
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

        Греем и ASR: первый `transcribe` всегда дороже последующих — ленивый импорт
        `text_to_num`, первый `KaldiRecognizer`/`sherpa-onnx`, а у облачного адаптера
        ещё и DNS + TCP + TLS до провайдера. Без этого первая реплика звонка
        приходит заметно позже следующих.
        """
        stages: dict[str, int] = {}

        warm_itn()

        started = time.perf_counter()
        try:
            await self.tts.synthesize("Проверка связи")
        except Exception as error:
            log.info("warmup tts skipped: %s", error)
        stages["tts"] = _ms(started)

        started = time.perf_counter()
        try:
            asr_warm = getattr(self.asr, "warm", None)
            if callable(asr_warm):
                await asr_warm()
            await self.asr.transcribe(Audio(data=_WARMUP_AUDIO, mime="audio/pcm"))
        except Exception as error:
            log.info("warmup asr skipped: %s", error)
        stages["asr"] = _ms(started)

        llm = getattr(self.orchestrator, "llm", None)
        if llm is not None:
            started = time.perf_counter()
            try:
                await llm.complete("Отвечай одним словом.", "скажи: ок")
            except Exception as error:
                log.info("warmup llm skipped: %s", error)
            stages["llm"] = _ms(started)
        elif self.settings.dialog_adapter == "cloudru":
            # CloudRuDialogAdapter объявляет `llm = None`, но его первый ход — самый
            # дорогой: клиент OpenAI создаётся лениво в `Assistant.__init__`, плюс
            # не открыто ни одного TCP-соединения до провайдера. Греем настоящим
            # стартовым ходом агента и сразу же отпускаем его.
            started = time.perf_counter()
            try:
                await self._warm_cloudru_dialog()
            except Exception as error:
                log.info("warmup cloudru dialog skipped: %s", error)
            stages["dialog"] = _ms(started)

        log.info("warmup: %s", stages)
        return stages

    async def _warm_cloudru_dialog(self) -> None:
        """Один стартовый ход диалогового агента Cloud.ru на служебной сессии."""
        call_id = f"__warmup__{uuid4().hex[:8]}"
        try:
            await self.orchestrator.start(call_id, "")
        finally:
            # delete_call сам убирает и сессию, и диалогового агента звонка
            self.delete_call(call_id)

    def sweep(self) -> None:
        """Периодическая очистка просроченных сессий, блокировок и диалоговых агентов."""
        store_sweep = getattr(self.orchestrator.store, "sweep", None)
        if callable(store_sweep):
            store_sweep()
        orch_sweep = getattr(self.orchestrator, "sweep", None)
        if callable(orch_sweep):
            orch_sweep()

    async def aclose(self) -> None:
        """Закрывает адаптеры из closeables — ошибки при завершении не мешают штатному выходу."""
        for closeable in self.closeables:
            with contextlib.suppress(Exception):
                await closeable.aclose()

    def delete_call(self, call_id: str) -> None:
        """Удаляет сессию звонка и связанного с ней диалогового агента."""
        self.orchestrator.store.delete(call_id)
        drop_call = getattr(self.orchestrator, "drop_call", None)
        if callable(drop_call):
            drop_call(call_id)
            return
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


class _ExecutorCloseable:
    """Адаптер пула потоков под закрытие в Container.aclose()."""

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self._executor = executor

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def build_container(settings: Settings) -> Container:
    """Собирает контейнер зависимостей по настройкам."""
    closeables: list[Any] = []

    # Разделяем пулы потоков локального ASR и TTS, чтобы длинный синтез на CPU
    # никогда не задерживал распознавание параллельной реплики звонящего.
    asr_executor: Executor | None = None
    if settings.asr_adapter in ("vosk", "t-one"):
        asr_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="speech-asr")
        closeables.append(_ExecutorCloseable(asr_pool))
        asr_executor = asr_pool

    tts_executor: Executor | None = None
    if settings.tts_adapter in ("piper", "silero"):
        tts_workers = max(1, settings.silero_workers) if settings.tts_adapter == "silero" else 1
        tts_pool = ThreadPoolExecutor(max_workers=tts_workers, thread_name_prefix="speech-tts")
        closeables.append(_ExecutorCloseable(tts_pool))
        tts_executor = tts_pool

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

    asr = _build_asr(settings, executor=asr_executor)
    if hasattr(asr, "aclose"):
        closeables.append(asr)

    tts = _build_tts(settings, executor=tts_executor)
    if hasattr(tts, "aclose"):
        closeables.append(tts)

    outbox = Outbox(
        crm,
        InMemoryOutboxStore(),
        max_attempts=settings.outbox_max_attempts,
        backoff_sec=settings.outbox_backoff_sec,
        crm_timeout=settings.outbox_crm_timeout,
    )
    store = InMemorySessionStore(ttl_sec=settings.session_ttl_sec)
    telephony = FakeTelephony()
    escalation = EscalationDetector.from_yaml(settings.escalation_path)

    orchestrator: DialogRunner
    if use_cloudru_dialog:
        from assistant.services.cloudru.bridge import export_cloudru_env
        from assistant.services.cloudru.dialog import CloudRuDialogAdapter

        export_cloudru_env(settings, require_rag=True)
        cloudru_runner = CloudRuDialogAdapter(
            store=store,
            crm=crm,
            outbox=outbox,
            telephony=telephony,
            escalation=escalation,
            crm_timeout=settings.crm_timeout,
            agent_timeout=settings.agent_timeout,
            session_ttl_sec=settings.session_ttl_sec,
            max_concurrency=settings.cloudru_max_concurrency,
        )
        closeables.append(cloudru_runner)
        orchestrator = cloudru_runner
    else:
        llm_tasks = _build_llm_tasks(settings, closeables)
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


def _build_llm_tasks(settings: Settings, closeables: list[Any]) -> LlmTasks | None:
    if settings.llm_adapter in ("openai", "cloudru"):
        base_url = settings.llm_base_url or (
            settings.fm_base_url if settings.llm_adapter == "cloudru" else ""
        )
        api_key = settings.llm_api_key or (
            settings.fm_api_key if settings.llm_adapter == "cloudru" else ""
        )
        if api_key and settings.llm_model:
            client = OpenAICompatLlm(
                base_url,
                api_key,
                settings.llm_model,
                timeout=settings.llm_timeout,
            )
            closeables.append(client)
            return LlmTasks(client)
        log.warning(
            "llm_adapter=%s, но ключ/модель не заданы — работаем без LLM",
            settings.llm_adapter,
        )
    return None


def _build_asr(
    settings: Settings,
    *,
    executor: Executor | None = None,
) -> SpeechRecognizer:
    if settings.asr_adapter == "vosk":
        from assistant.services.asr.vosk import VoskAsr

        return VoskAsr(
            settings.vosk_model_path,
            sample_rate=settings.asr_sample_rate,
            use_itn=settings.asr_itn,
            timeout=settings.speech_timeout,
            executor=executor,
        )
    if settings.asr_adapter == "t-one":
        from assistant.services.asr.tone import ToneAsr

        return ToneAsr(
            settings.tone_model_dir,
            input_sample_rate=settings.asr_sample_rate,
            num_threads=settings.tone_num_threads,
            use_itn=settings.asr_itn,
            timeout=settings.speech_timeout,
            executor=executor,
        )
    if settings.asr_adapter == "cloudru":
        from assistant.services.asr.cloudru import CloudRuAsr
        from assistant.services.cloudru.bridge import export_cloudru_env

        export_cloudru_env(settings, require_rag=settings.dialog_adapter == "cloudru")
        return CloudRuAsr(
            sample_rate=settings.asr_sample_rate,
            timeout=settings.speech_timeout,
            use_itn=settings.asr_itn,
            api_key=settings.fm_api_key,
            base_url=settings.fm_base_url,
            model=settings.stt_model,
        )
    return FakeAsr()


def _build_tts(
    settings: Settings,
    *,
    executor: Executor | None = None,
) -> SpeechSynthesizer:
    if settings.tts_adapter == "piper":
        from assistant.services.tts.piper import PiperTts

        return PiperTts(
            settings.piper_model_path,
            None,
            normalize=settings.tts_normalize,
            timeout=settings.speech_timeout,
            executor=executor,
        )
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
            num_threads=settings.silero_num_threads,
            workers=settings.silero_workers,
            timeout=settings.speech_timeout,
            executor=executor,
        )
    return FakeTts()


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container
