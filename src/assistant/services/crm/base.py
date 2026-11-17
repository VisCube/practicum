"""Порт CRM"""

from typing import Protocol

from assistant.core.models import Appeal, Outage, Owner


class CrmError(Exception):
    """CRM недоступна или отказала. Оркестратор и outbox ловят именно это"""


class CrmClient(Protocol):
    async def find_owner_by_phone(self, phone: str) -> Owner | None:
        """Собственник по номеру телефона. None — не найден"""
        ...

    async def active_outages(self, owner: Owner) -> list[Outage]:
        """Активные аварии по дому собственника. Адаптер сам решает, по какому ключу искать"""
        ...

    async def create_deal(self, appeal: Appeal) -> str:
        """Создать Сделку, вернуть её id в CRM. Должно быть идемпотентно по appeal.id

        Handlers этот метод не вызывают — только Outbox"""
        ...