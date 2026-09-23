"""In-memory CRM с демо-данными. Умеет имитировать задержку и отказ"""

import asyncio

from assistant.core.models import Appeal, Outage, Owner
from assistant.services.crm.base import CrmError

DEMO_OWNERS = [
    Owner(id="o1", full_name="Иванов Иван Иванович", phone="+79990000001",
          address="ул. Ленина, д. 10", apartment="42", bitrix_contact_id="101"),
    Owner(id="o2", full_name="Петрова Мария Сергеевна", phone="+79990000002",
          address="ул. Ленина, д. 10", apartment="7", bitrix_contact_id="102"),
    Owner(id="o3", full_name="Сидоров Пётр Алексеевич", phone="+79990000003",
          address="пр. Мира, д. 5", apartment="15", bitrix_contact_id="103"),
]

DEMO_OUTAGES = [
    Outage(id="out1", address="ул. Ленина, д. 10",
           description="Отключение горячей воды из-за ремонта стояка. Ориентировочно до 18:00."),
]


class InMemoryCrm:
    """In-memory CRM с демо-данными - для тестов и локальной разработки"""
    def __init__(
            self,
            owners: list[Owner] | None = None,
            outages: list[Outage] | None = None,
            *,
            delay: float = 0.0,
            fail_times: int = 0,
    ) -> None:
        self.owners = {owner.phone: owner for owner in (owners if owners is not None else DEMO_OWNERS)}
        self.outages = list(outages if outages is not None else DEMO_OUTAGES)
        self.deals: dict[str, Appeal] = {}     # appeal.id -> appeal
        self.delay = delay
        self.fail_times = fail_times            # сколько первых вызовов create_deal уронить
        self.calls: list[str] = []

    async def _io(self, name: str) -> None:
        """Имитирует задержку и пишет имя метода в self.calls для проверки в тестах"""
        self.calls.append(name)
        if self.delay:
            await asyncio.sleep(self.delay)

    async def find_owner_by_phone(self, phone: str) -> Owner | None:
        await self._io("find_owner_by_phone")
        return self.owners.get(phone)

    async def active_outages(self, owner: Owner) -> list[Outage]:
        await self._io("active_outages")
        return [outage for outage in self.outages if outage.address.lower() == owner.address.lower()]

    async def create_deal(self, appeal: Appeal) -> str:
        await self._io("create_deal")
        if self.fail_times > 0:
            self.fail_times -= 1
            raise CrmError("CRM временно недоступна (fake)")
        self.deals.setdefault(appeal.id, appeal)   # идемпотентно
        return f"DEAL-{appeal.id}"
    