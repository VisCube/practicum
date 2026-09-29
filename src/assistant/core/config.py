"""Настройки приложения. Читаются из окружения и .env. Неверное имя адаптера — ошибка на старте"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

AsrAdapter = Literal["fake", "vosk"]
TtsAdapter = Literal["fake", "piper"]
LlmAdapter = Literal["fake", "openai"]
CrmAdapter = Literal["fake", "telegram", "bitrix"]
TelephonyAdapter = Literal["fake"]


class Settings(BaseSettings):
    """Все параметры приложения одним объектом — адаптеры, таймауты, пути к моделям и данным"""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Общее
    app_env: str = "dev"
    debug_ui: bool = False          # монтировать /debug и HTML-стенд (только dev)
    log_level: str = "INFO"

    # Данные
    data_dir: Path = Path("data")       # корень для classifier.yaml, escalation.yaml и knowledge/

    # Адаптеры
    asr_adapter: AsrAdapter = "fake"
    tts_adapter: TtsAdapter = "fake"
    llm_adapter: LlmAdapter = "fake"
    crm_adapter: CrmAdapter = "fake"
    telephony_adapter: TelephonyAdapter = "fake"

    # Таймауты горячего пути (внутри звонка), сек
    crm_timeout: float = 2.0
    llm_timeout: float = 4.0           # максимум ждём ответа от LLM
    speech_timeout: float = 10.0       # максимум ждём ответа от ASR/TTS

    # Сессии
    session_ttl_sec: float = 1800   # брошенный звонок выметается из памяти через 30 мин

    # Речь
    asr_sample_rate: int = 16000    # lpcm на входе ASR

    # LLM - любой OpenAI-совместимый endpoint
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # Outbox: фоновая отправка заявок в CRM
    outbox_max_attempts: int = 5
    outbox_backoff_sec: float = 2.0
    outbox_crm_timeout: float = 10.0   # фону можно ждать дольше, чем горячему пути

    # Telegram (CRM_ADAPTER=telegram): заявки летят в чат диспетчера, справочники — из fake
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_channel_username: str = ""   # публичный канал для заявок: username для ссылок на сообщения

    # Локальные модели (ASR_ADAPTER=vosk, TTS_ADAPTER=piper)
    vosk_model_path: str = "models/vosk-model-small-ru-0.22"
    piper_model_path: str = "models/ru_RU-ruslan-medium.onnx"

    @property
    def classifier_path(self) -> Path:
        return self.data_dir / "classifier.yaml"

    @property
    def escalation_path(self) -> Path:
        return self.data_dir / "escalation.yaml"

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"