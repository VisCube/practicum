"""Настройки приложения. Читаются из окружения и .env. Неверное имя адаптера — ошибка на старте."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

DialogAdapter = Literal["rules", "cloudru"]
AsrAdapter = Literal["fake", "vosk", "t-one", "cloudru"]
TtsAdapter = Literal["fake", "piper", "silero"]
LlmAdapter = Literal["fake", "openai", "cloudru"]
CrmAdapter = Literal["fake", "telegram", "bitrix"]
TelephonyAdapter = Literal["fake"]


class Settings(BaseSettings):
    """Все параметры приложения одним объектом — адаптеры, таймауты, пути к моделям и данным."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Общее
    app_env: str = "dev"
    debug_ui: bool = False          # монтировать /debug и HTML-стенд (только dev)
    log_level: str = "INFO"

    # Данные
    data_dir: Path = Path("data")   # корень для classifier.yaml, escalation.yaml и knowledge/

    # Адаптеры
    dialog_adapter: DialogAdapter = "rules"
    asr_adapter: AsrAdapter = "fake"
    tts_adapter: TtsAdapter = "fake"
    llm_adapter: LlmAdapter = "fake"
    crm_adapter: CrmAdapter = "fake"
    telephony_adapter: TelephonyAdapter = "fake"

    # Таймауты горячего пути (внутри звонка), сек
    crm_timeout: float = 2.0
    llm_timeout: float = 4.0        # максимум ждём ответа от LLM в rules-режиме
    speech_timeout: float = 10.0    # максимум ждём ответа от ASR/TTS
    agent_timeout: float = 45.0     # максимум ждём хода диалогового агента Cloud.ru (LLM + RAG)

    # Сессии
    session_ttl_sec: float = 1800   # брошенный звонок выметается из памяти через 30 мин

    # Речь (локальные модели ASR)
    asr_sample_rate: int = 16000    # lpcm на входе
    vosk_model_path: str = "models/vosk-model-small-ru-0.22"
    tone_model_dir: str = "models/t-one"
    tone_num_threads: int = 2
    asr_itn: bool = True

    # LLM — любой OpenAI-совместимый endpoint
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # Cloud.ru: Foundation Models, Managed RAG и IAM
    fm_api_key: str = ""
    fm_base_url: str = "https://foundation-models.api.cloud.ru/v1"
    stt_model: str = "openai/whisper-large-v3"
    rag_key_id: str = ""
    rag_key_secret: str = ""
    rag_base_url: str = ""
    iam_token_url: str = "https://iam.api.cloud.ru/api/v1/auth/token"

    # Outbox: фоновая отправка заявок в CRM
    outbox_max_attempts: int = 5
    outbox_backoff_sec: float = 2.0
    outbox_crm_timeout: float = 10.0

    # Telegram (CRM_ADAPTER=telegram): заявки летят в чат диспетчера, справочники — из fake
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_channel_username: str = ""

    # Локальные модели TTS (TTS_ADAPTER=piper | silero)
    piper_model_path: str = "models/ru_RU-ruslan-medium.onnx"
    silero_model_path: str = "models/v5_cis_base.pt"
    silero_speaker: str = ""
    silero_sample_rate: int = 24000
    silero_device: str = "cpu"
    silero_put_accent: bool = True
    silero_put_yo: bool = True

    tts_normalize: bool = True
    tts_stress: bool = True
    silero_ssml: bool = True
    silero_rate: str = "medium"     # x-slow|slow|medium|fast|x-fast
    silero_pitch: str = "medium"    # x-low|low|medium|high|x-high

    classifier_model_path: str = "models/model.pkl"
    classifier_min_confidence: float = 0.35
    classifier_sub_threshold: float = 0.75

    @property
    def classifier_path(self) -> Path:
        return self.data_dir / "classifier.yaml"

    @property
    def escalation_path(self) -> Path:
        return self.data_dir / "escalation.yaml"

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"
