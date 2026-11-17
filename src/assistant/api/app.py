from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from assistant.core.config import Settings

from .dependencies import Container, build_container
from .routes import telephony

log = logging.getLogger(__name__)


async def _run_sweep_loop(
    container: Container,
    stop_event: asyncio.Event,
    interval: float = 60.0,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            try:
                container.sweep()
            except Exception:
                log.exception("sweep: ошибка фоновой очистки")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собирает и возвращает FastAPI-приложение с учётом настроек.

    Если настройки не переданы — читаются из переменных окружения.
    Дебаг-роуты монтируются только при `DEBUG_UI=true`.
    """
    settings = settings or Settings()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=settings.log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = build_container(settings)
        app.state.container = container
        stop_event = asyncio.Event()
        worker = asyncio.create_task(container.outbox.run(stop_event, interval=1.0))
        sweeper = asyncio.create_task(_run_sweep_loop(container, stop_event, interval=60.0))
        warmup = asyncio.create_task(container.warm())
        try:
            yield
        finally:
            stop_event.set()
            await asyncio.gather(worker, sweeper, return_exceptions=True)
            warmup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warmup
            await container.aclose()

    app = FastAPI(title="Голосовой ассистент УК", version="0.1.0", lifespan=lifespan)
    app.include_router(telephony.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env}

    if settings.debug_ui:
        from .routes import debug, voice_ws

        app.include_router(debug.router)
        debug.log_ui_build()
        if settings.voice_ws:
            app.include_router(voice_ws.router)

    return app
