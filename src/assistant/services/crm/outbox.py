"""Outbox: заявка сохраняется локально сразу, в CRM уходит фоном с ретраями

Клиент получает внутренний номер (appeal.id) мгновенно, даже если CRM лежит
Время — UTC; naive datetime на входе считаем UTC"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field

from assistant.core.models import Appeal
from assistant.services.crm.base import CrmClient, CrmError

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class OutboxStatus(StrEnum):
    """Статус записи в outbox: ожидает отправки, отправлена или исчерпаны попытки"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"      # исчерпаны попытки - нужен человек


class OutboxRecord(BaseModel):
    """Одна запись в outbox: заявка, статус доставки и параметры следующей попытки"""
    appeal: Appeal
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = 0
    last_error: str | None = None
    next_try_at: datetime = Field(default_factory=_utcnow)


class OutboxStore(Protocol):
    """Контракт хранилища outbox — put/get и выборка записей, готовых к отправке"""
    def put(self, rec: OutboxRecord) -> None: ...
    def get(self, appeal_id: str) -> OutboxRecord | None: ...
    def due(self, now: datetime) -> list[OutboxRecord]: ...
    def all(self) -> list[OutboxRecord]: ...


class InMemoryOutboxStore:
    """Словарь в памяти - для тестов и однопроцессного режима"""
    def __init__(self) -> None:
        self._recs: dict[str, OutboxRecord] = {}

    def put(self, rec: OutboxRecord) -> None:
        self._recs[rec.appeal.id] = rec

    def get(self, appeal_id: str) -> OutboxRecord | None:
        return self._recs.get(appeal_id)

    def due(self, now: datetime) -> list[OutboxRecord]:
        now = _aware(now)
        return [record for record in self._recs.values()
                if record.status == OutboxStatus.PENDING and _aware(record.next_try_at) <= now]

    def all(self) -> list[OutboxRecord]:
        return list(self._recs.values())

class Outbox:
    """Надёжная доставка заявок в CRM: мгновенная запись локально, отправка фоном с ретраями"""
    def __init__(
            self,
            crm: CrmClient,
            store: OutboxStore,
            *,
            max_attempts: int = 5,
            backoff_sec: float = 2.0,
            crm_timeout: float = 10.0,
    ) -> None:
        self.crm = crm
        self.store = store
        self.max_attempts = max_attempts
        self.backoff_sec = backoff_sec
        self.crm_timeout = crm_timeout      # зависший запрос не должен останавливать очередь

    def enqueue(self, appeal: Appeal) -> str:
        """Синхронно и мгновенно: положить в очередь, вернуть внутренний номер"""
        if self.store.get(appeal.id) is None:
            self.store.put(OutboxRecord(appeal=appeal))
        return appeal.id

    async def process_once(self, now: datetime | None = None) -> int:
        """Одна итерация воркера. Возвращает число успешно отправленных"""
        now = _aware(now) if now else _utcnow()
        sent_count = 0
        for record in self.store.due(now):
            record.attempts += 1
            try:
                deal_id = await asyncio.wait_for(self.crm.create_deal(record.appeal), self.crm_timeout)
            except (CrmError, TimeoutError) as crm_error:
                record.last_error = str(crm_error) or "timeout"   # у TimeoutError пустое сообщение
                if record.attempts >= self.max_attempts:
                    record.status = OutboxStatus.FAILED
                    log.error("outbox: заявка %s не доставлена: %s", record.appeal.id, record.last_error)
                else:
                    retry_delay = self.backoff_sec * 2 ** (record.attempts - 1)
                    record.next_try_at = now + timedelta(seconds=retry_delay)
                    log.warning("outbox: %s попытка %d, повтор через %.0fс", record.appeal.id,
                                record.attempts, retry_delay)
            else:
                record.appeal.bitrix_deal_id = deal_id
                record.status = OutboxStatus.SENT
                record.last_error = None
                sent_count += 1
            self.store.put(record)
        return sent_count

    async def run(self, stop_event: asyncio.Event, interval: float = 1.0) -> None:
        """Фоновый цикл для lifespan приложения"""
        while not stop_event.is_set():
            try:
                await self.process_once()
            except Exception:  # noqa: BLE001 — воркер не должен умирать
                log.exception("outbox: ошибка итерации")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except TimeoutError:
                pass