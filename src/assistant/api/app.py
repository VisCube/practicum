import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from assistant.core.config import Settings

from .dependencies import build_container
from .routes import telephony


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собирает и возвращает FastAPI-приложение с учётом настроек

    Если настройки не переданы — читаются из переменных окружения
    Дебаг роуты монтируются только при DEBUG_UI=true
    """
    settings = settings or Settings()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=settings.log_level,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Инициализация: собираем зависимости и запускаем фоновую отправку outbox
        # Завершение: сигнализируем воркеру остановиться и дожидаемся его
        container = build_container(settings)
        app.state.container = container
        stop_event = asyncio.Event()
        worker = asyncio.create_task(container.outbox.run(stop_event, interval=1.0))
        try:
            yield
        finally:
            stop_event.set()
            await worker
            await container.aclose()

    app = FastAPI(title="Голосовой ассистент УК", version="0.1.0", lifespan=lifespan)
    app.include_router(telephony.router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "env": settings.app_env}

    if settings.debug_ui:
        from .routes import debug
        app.include_router(debug.router)

    return app