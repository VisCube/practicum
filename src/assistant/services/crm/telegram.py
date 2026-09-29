"""Telegram как выход для заявок: диспетчер получает их в чат.

Справочные запросы (собственник по телефону, аварии) делегируются вложенной CRM -
у Telegram этих данных нет. create_deal = сообщение в чат, id «сделки» — id сообщения
Идемпотентность — в памяти процесса: outbox повторяет только неудачные отправки,
так что дубль возможен лишь после рестарта с незавершённой записью"""

import html
import logging

import httpx

from assistant.core.models import Appeal, Outage, Owner, RiskLevel
from assistant.services.crm.base import CrmClient, CrmError

log = logging.getLogger(__name__)

_RISK = {RiskLevel.CRITICAL: "СРОЧНО", RiskLevel.NORMAL: "обычный", RiskLevel.LOW: "низкий"}
_IDENTITY = ("full_name", "address", "apartment")
_TRANSCRIPT_TURNS = 12
_MAX_LEN = 4000          # лимит Telegram — 4096


def render_appeal(appeal: Appeal, labels: dict[str, str]) -> str:
    """HTML-сообщение для диспетчера. Транскрипт свёрнут в раскрывающуюся цитату"""
    escape = html.escape
    lines = [
        f"<b>Заявка № {escape(appeal.id)}</b> · {escape(appeal.topic.name)}",
        f"Риск: <b>{_RISK[appeal.risk_level]}</b>"
        + (f" · угроза жалобы: {escape(', '.join(appeal.escalation_markers) or 'да')}" if appeal.escalated else ""),
        "",
        "Заявитель: " + (f"{escape(appeal.owner.full_name)} <i>(из CRM)</i>" if appeal.owner
                         else "<i>не идентифицирован, данные со слов</i>"),
        f"Телефон: {escape(appeal.phone)}",
        ]
    ordered = [slot_id for slot_id in _IDENTITY if slot_id in appeal.slots] + \
              [slot_id for slot_id in appeal.slots if slot_id not in _IDENTITY]
    for slot_id in ordered:
        if appeal.owner and slot_id in _IDENTITY and slot_id != "apartment":
            continue          # ФИО и адрес известного собственника уже сказаны строкой выше
        lines.append(f"{escape(labels.get(slot_id, slot_id))}: {escape(appeal.slots[slot_id])}")

    turns = appeal.transcript[-_TRANSCRIPT_TURNS:]
    if turns:
        quote = "\n".join(f"{'бот' if turn.role == 'bot' else 'житель'}: {escape(turn.text)}" for turn in turns)
        lines += ["", "<blockquote expandable>" + quote + "</blockquote>"]

    text = "\n".join(lines)
    return text if len(text) <= _MAX_LEN else text[:_MAX_LEN - 1] + "…"


class TelegramCrm:
    """CRM-адаптер поверх Telegram: заявка → сообщение в чат диспетчера
    Справочные запросы (собственник, аварии) делегируются вложенной внутренней CRM"""
    def __init__(
            self,
            inner: CrmClient,
            bot_token: str,
            chat_id: str,
            *,
            channel_username: str = "",
            slot_labels: dict[str, str] | None = None,
            client: httpx.AsyncClient | None = None,
            timeout: float = 10.0,
    ) -> None:
        self.inner = inner
        self.chat_id = chat_id
        self.channel_username = channel_username
        self.labels = slot_labels or {}
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._sent: dict[str, str] = {}      # appeal.id -> deal_id

    def _channel_link(self, message_id: int) -> str | None:
        """Ссылка на сообщение в публичном канале; None, если канал не настроен"""
        if not self.channel_username:
            return None
        return f"https://t.me/{self.channel_username}/{message_id}"

    async def find_owner_by_phone(self, phone: str) -> Owner | None:
        """Делегирует во вложенную CRM - у Telegram справочников нет"""
        return await self.inner.find_owner_by_phone(phone)

    async def active_outages(self, owner: Owner) -> list[Outage]:
        """Делегирует во вложенную CRM - у Telegram справочников нет"""
        return await self.inner.active_outages(owner)

    async def create_deal(self, appeal: Appeal) -> str:
        """Отправляет заявку в Telegram-чат и возвращает id сообщения как id сделки"""
        if existing_deal_id := self._sent.get(appeal.id):
            return existing_deal_id
        payload = {"chat_id": self.chat_id, "text": render_appeal(appeal, self.labels),
                   "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            response = await self._client.post(self._url, json=payload)
        except httpx.HTTPError as http_error:
            raise CrmError(f"telegram: {http_error}") from http_error
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code != 200 or not body.get("ok"):
            raise CrmError(f"telegram: {response.status_code} {body.get('description') or response.text[:200]}")
        deal_id = f"TG-{body['result']['message_id']}"
        self._sent[appeal.id] = deal_id
        appeal.channel_link = self._channel_link(body["result"]["message_id"])
        return deal_id

    async def aclose(self) -> None:
        """Закрывает HTTP-клиент"""
        await self._client.aclose()