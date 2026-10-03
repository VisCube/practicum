"""Сборка приложения из адаптеров по Settings. Единственное место, где импортируются fake и реальные клиенты."""

import base64
import logging
from dataclasses import dataclass, field

from fastapi import Request

from assistant.core.config import Settings
from assistant.core.models import Reply, TopicKind
from assistant.core.orchestrator import Orchestrator
from assistant.core.session_store import InMemorySessionStore
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


def want_audio(request: Request) -> bool:
    """Стенд шлёт X-Audio: off, когда озвучка выключена — не гоняем TTS впустую. Телефония заголовок не шлёт."""
    return request.headers.get("x-audio", "on").lower() != "off"

def want_tts_speaker(request: Request) -> str | None:
    """X-Tts-Speaker — выбор голоса со стенда, дебаг онли"""
    value = request.headers.get("x-tts-speaker", "").strip()
    return value or None


def tts_voice_info(tts: SpeechSynthesizer) -> tuple[list[str], str | None]:
    """Список голосов и текущий дефолт, если поддерживает адаптер"""
    speakers = list(getattr(tts, "speakers", None) or [])
    speaker = getattr(tts, "speaker", None)
    return speakers, speaker if isinstance(speaker, str) else None


@dataclass
class Container:
    """Собранные зависимости приложения: адаптеры, оркестратор и outbox"""
    settings: Settings
    orchestrator: Orchestrator
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
    ) -> TurnResponse:
        audio_b64, mime = None, "audio/ogg"
        if with_audio and reply.text:
            try:
                audio = await self.tts.synthesize(reply.text, speaker=tts_speaker)
                if not audio.empty:
                    audio_b64, mime = base64.b64encode(audio.data).decode(), audio.mime
            except TtsError as error:
                log.warning("tts failed, fallback to text: %s", error)
        return TurnResponse(
            call_id=call_id, text=reply.text, audio_b64=audio_b64, audio_mime=mime, heard=heard,
            transfer=reply.transfer, end_call=reply.end_call, state=reply.state, events=reply.events,
        )

    async def aclose(self) -> None:
        """Закрывает все адаптеры из closeables — ошибки при завершении глотает, чтобы не мешать штатному выходу"""
        for closeable in self.closeables:
            try:
                await closeable.aclose()
            except Exception:  # noqa: BLE001 — на выходе не падаем
                pass

    @property
    def demo_phones(self) -> list[str]:
        """Список демо-телефонов из InMemoryCrm - пуст, если подключена реальная CRM"""
        crm = getattr(self.crm, "inner", self.crm)      # TelegramCrm оборачивает InMemoryCrm
        return [o.phone for o in DEMO_OWNERS] if isinstance(crm, InMemoryCrm) else []


def build_container(settings: Settings) -> Container:
    """Собирает контейнер зависимостей по настройкам - выбирает адаптеры, инициализирует оркестратор"""
    closeables: list = []
    classifier = TopicClassifier.from_yaml(settings.classifier_path)

    demo_crm = InMemoryCrm()
    crm: CrmClient
    if settings.crm_adapter == "fake":
        crm = demo_crm
    elif settings.crm_adapter == "telegram":
        if not (settings.telegram_bot_token and settings.telegram_chat_id):
            raise RuntimeError("CRM_ADAPTER=telegram: нужны TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID")
        from assistant.services.crm.telegram import TelegramCrm
        crm = TelegramCrm(demo_crm, settings.telegram_bot_token, settings.telegram_chat_id,
                          channel_username=settings.telegram_channel_username,
                          slot_labels={s.id: s.label for s in classifier.slots.values()},
                          timeout=settings.outbox_crm_timeout)
        closeables.append(crm)
    else:
        raise NotImplementedError("crm_adapter=bitrix: клиент Битрикс24 ещё не реализован")

    llm_tasks: LlmTasks | None = None
    if settings.llm_adapter == "openai" and settings.llm_api_key and settings.llm_model:
        client = OpenAICompatLlm(settings.llm_base_url, settings.llm_api_key, settings.llm_model,
                                 timeout=settings.llm_timeout)
        closeables.append(client)
        llm_tasks = LlmTasks(client)
    elif settings.llm_adapter != "fake":
        log.warning("llm_adapter=%s, но ключ/модель не заданы — работаем без LLM", settings.llm_adapter)

    asr: SpeechRecognizer
    tts: SpeechSynthesizer
    if settings.asr_adapter == "vosk":
        from assistant.services.asr.vosk import VoskAsr
        asr = VoskAsr(settings.vosk_model_path, sample_rate=settings.asr_sample_rate)
    elif settings.asr_adapter == "t-one":
        from assistant.services.asr.tone import ToneAsr
        asr = ToneAsr(
            settings.tone_model_dir,
            input_sample_rate=settings.asr_sample_rate,
            num_threads=settings.tone_num_threads,
        )
    else:
        asr = FakeAsr()

    if settings.tts_adapter == "piper":
        from assistant.services.tts.piper import PiperTts
        tts = PiperTts(settings.piper_model_path, None)
    elif settings.tts_adapter == "silero":
        from assistant.services.tts.silero import SileroTts
        tts = SileroTts(
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
    else:
        tts = FakeTts()

    escalation = EscalationDetector.from_yaml(settings.escalation_path)
    knowledge_base = MarkdownRetriever.from_dir(settings.knowledge_dir)
    outbox = Outbox(crm, InMemoryOutboxStore(), max_attempts=settings.outbox_max_attempts,
                    backoff_sec=settings.outbox_backoff_sec, crm_timeout=settings.outbox_crm_timeout)
    orchestrator = Orchestrator(
        store=InMemorySessionStore(ttl_sec=settings.session_ttl_sec), crm=crm, classifier=classifier,
        escalation=escalation,
        handlers={
            TopicKind.APPEAL: IntakeHandler(outbox, classifier.slots, llm=llm_tasks),
            TopicKind.CONSULT: ConsultationHandler(knowledge_base, llm_tasks),
        },
        telephony=FakeTelephony(), llm=llm_tasks, crm_timeout=settings.crm_timeout,
    )
    return Container(settings, orchestrator, outbox, crm, asr, tts, closeables)


def get_container(request: Request) -> Container:
    return request.app.state.container