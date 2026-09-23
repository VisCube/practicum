from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex[:8].upper()


class CallState(StrEnum):
    GREETING = "greeting"          # звонок начат, приветствие произнесено
    TOPIC = "topic"                # ждём формулировку темы
    IDENTIFY = "identify"          # собственник не найден, спрашиваем ФИО/адрес
    SLOTS = "slots"                # собираем реквизиты по теме
    CONFIRM = "confirm"            # прочитали сводку, ждём да/нет
    DONE = "done"                  # заявка создана / консультация дана
    TRANSFERRED = "transferred"    # переведено на оператора


class RiskLevel(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    CRITICAL = "critical"


class TopicKind(StrEnum):
    APPEAL = "appeal"    # создаём заявку (Сделку)
    CONSULT = "consult"  # отвечаем из базы знаний


class SlotSpec(BaseModel):
    """Описание одного реквизита: как спросить и как распознать."""

    id: str
    label: str
    question: str
    kind: Literal["number", "text", "address", "name", "choice"] = "text"
    choices: list[str] = Field(default_factory=list)
    cues: list[str] = Field(default_factory=list)

class Owner(BaseModel):
    id: str
    full_name: str
    phone: str
    address: str
    apartment: str
    bitrix_contact_id: str | None = None


class Topic(BaseModel):
    id: str
    name: str
    kind: TopicKind = TopicKind.APPEAL
    bitrix_id: str
    risk_level: RiskLevel = RiskLevel.NORMAL
    keywords: list[str] = Field(default_factory=list)
    required_slots: list[str] = Field(default_factory=list)


class Outage(BaseModel):
    """Активная авария по адресу - уведомляем собственника при совпадении с его темой"""
    id: str
    address: str
    description: str
    eta: datetime | None = None


class Turn(BaseModel):
    """Одна реплика в транскрипте звонка - либо от пользователя, либо от бота"""
    role: Literal["user", "bot"]
    text: str
    ts: datetime = Field(default_factory=_now)


class Event(BaseModel):
    """Событие для внешнего мира (телефония, дев-стенд): что случилось на этом ходе

    escalated - прозвучала угроза жалобы, заявка уйдёт с флагом; разговор продолжается
    transfer  - звонок переводится на оператора; payload — контекст для него"""

    type: Literal["outage_notice", "appeal_created", "escalated", "transfer", "call_ended"]
    payload: dict[str, Any] = Field(default_factory=dict)


class Appeal(BaseModel):
    """Заявка, созданная по итогам звонка и переданная в CRM"""
    id: str = Field(default_factory=_new_id)
    call_id: str
    phone: str
    owner: Owner | None
    topic: Topic
    slots: dict[str, str]
    risk_level: RiskLevel
    escalated: bool = False
    escalation_markers: list[str] = Field(default_factory=list)
    transcript: list[Turn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    bitrix_deal_id: str | None = None


class CallSession(BaseModel):
    """Состояние активного звонка: шаг диалога, собранные данные и история реплик"""
    call_id: str
    phone: str
    state: CallState = CallState.GREETING
    owner: Owner | None = None
    owner_lookup_done: bool = False
    topic: Topic | None = None
    slots: dict[str, str] = Field(default_factory=dict)
    pending_slot: str | None = None      # какой слот сейчас спрашиваем
    transcript: list[Turn] = Field(default_factory=list)
    escalated: bool = False
    escalation_markers: list[str] = Field(default_factory=list)   # что прозвучало: ГЖИ, суд…
    appeals: list[str] = Field(default_factory=list)   # id созданных заявок за звонок
    attempts: int = 0        # счётчик переспросов на текущем шаге

    def say(self, text: str) -> None:
        self.transcript.append(Turn(role="bot", text=text))

    def hear(self, text: str) -> None:
        self.transcript.append(Turn(role="user", text=text))


class Reply(BaseModel):
    """Результат одного хода: что сказать, что произошло."""

    text: str
    transfer: bool = False
    end_call: bool = False
    events: list[Event] = Field(default_factory=list)
    state: CallState


class Passage(BaseModel):
    """Фрагмент из базы знаний с оценкой релевантности - результат поиска по запросу"""
    source: str
    heading: str
    text: str
    score: float = 0.0


class Audio(BaseModel):
    """Аудиофрагмент: байты + MIME, пустая data - показывает текст"""

    data: bytes = b""
    mime: str = "audio/ogg"

    @property
    def empty(self) -> bool:
        return not self.data