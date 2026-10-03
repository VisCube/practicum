from pydantic import BaseModel, Field

from assistant.core.models import CallState, Event


class StartRequest(BaseModel):
    phone: str
    call_id: str | None = None      # id телефонии, для стенда не используется


class TurnRequest(BaseModel):
    call_id: str
    phone: str
    text: str


class TurnResponse(BaseModel):
    call_id: str
    text: str                       # реплика бота
    audio_b64: str | None = None    # готовая озвучка от TTS; None - адаптер аудио не дал, клиент озвучивает text сам
    audio_mime: str = "audio/ogg"
    heard: str | None = None        # что распознал ASR
    transfer: bool = False          # попросить платформу переключить на оператора
    end_call: bool = False          # попросить платформу завершить звонок
    state: CallState
    events: list[Event] = Field(default_factory=list)


class SessionView(BaseModel):
    call_id: str
    phone: str
    state: CallState
    owner: str | None
    topic: str | None
    slots: dict[str, str]
    escalated: bool
    escalation_markers: list[str]
    appeals: list[str]
    transcript: list[dict]


class DebugConfig(BaseModel):
    asr: str
    tts: str
    llm: str
    crm: str
    demo_phones: list[str]
    tts_speakers: list[str] = Field(default_factory=list)
    tts_speaker: str | None = None


class TtsRequest(BaseModel):
    text: str
    speaker: str | None = None


class TtsResponse(BaseModel):
    spoken: str                     # текст, отправленный в TTS (зарезервировано для нормализации — пока равен запросу)
    audio_b64: str | None = None
    audio_mime: str = "audio/ogg"
    speaker: str | None = None