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
                if container.call_journal:
                    store = container.orchestrator.store
                    container.call_journal.expire_http(getattr(store, "contains_or_locked", None))
                    container.call_journal.sweep()
            except Exception:
                log.exception("sweep: ошибка фоновой очистки")


def _protect_transport_logs() -> None:
    for name in (
        "uvicorn.error", "websockets", "websockets.client", "websockets.server",
        "httpcore", "httpcore.connection", "httpcore.http11",
        "httpcore2", "httpcore2.connection", "httpcore2.http11",
        "openai", "openai._base_client",
    ):
        logger = logging.getLogger(name)
        if logger.getEffectiveLevel() < logging.INFO:
            logger.setLevel(logging.INFO)


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
    _protect_transport_logs()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = build_container(settings)
        app.state.container = container
        if settings.call_log_enabled:
            from assistant.services.call_log.journal import CallJournal

            journal = CallJournal(
                settings.call_log_path,
                settings.call_log_retention_days,
                abandoned_sec=settings.session_ttl_sec,
                max_used_bytes=settings.call_log_max_used_mb * 1024 * 1024,
                min_free_bytes=settings.call_log_min_free_mb * 1024 * 1024,
                backup_dir=settings.call_log_backup_dir,
                backup_retention_days=settings.call_log_backup_retention_days,
                backup_interval_sec=settings.call_log_backup_interval_sec,
            )
            await journal.start()
            container.call_journal = journal
            container.closeables.append(journal)
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
