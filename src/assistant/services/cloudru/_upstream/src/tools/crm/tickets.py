from datetime import date, timedelta
from random import randint

from src.tools.classifier import get_category


class Ticket:
    def __init__(
        self,
        id: int,
        status: str,
        deadline: str,
        category: str,
        phone: str,
        description: str,
        **fields: str | int,
    ):
        self.id = id
        self.status = status
        self.deadline = deadline
        self.category = category
        self.phone = phone
        self.description = description

        for name, value in fields.items():
            setattr(self, name, value)


TICKETS: list[Ticket] = [
    Ticket(
        id=1234,
        phone="+79990000001",
        category="Освещение, электричество / Неисправность освещения в подъезде / во дворе",
        street="Мира", house="19",
        resident="Иванов Иван Иванович", entry=1,
        description="Не работает освещение в подъезде",
        status="В работе",
        deadline="10.11.2026",
    ),
    Ticket(
        id=5678,
        phone="+79990000004",
        category="Водоснабжение / Нарушение качества услуги холодного, горячего водоснабжения",
        street="Мира", house="19", apartment=15,
        description="Слабый напор горячей воды",
        status="Зарегистрирована",
        deadline="12.11.2026",
    ),
    Ticket(
        id=3456,
        phone="+79990000002",
        category="Домофоны, шлагбаумы, ворота, видеонаблюдение / Неисправность домофона, переговорного устройства, считывателя, кнопки открывания",
        street="Мира", house="19",
        resident="Васильева Василиса Васильевна", entry=2,
        description="Не работает домофон",
        status="В работе",
        deadline="11.11.2026",
    ),
    Ticket(
        id=7890,
        phone="+79990000003",
        category="Отопление / Протечка отопления",
        street="Мира", house="32",
        resident="Петров Пётр Петрович", apartment=8,
        description="Протекает труба отопления",
        status="Выполнена",
        deadline="05.11.2026",
    ),
]


def create_ticket(
    category: str,
    phone: str,
    description: str,
    **fields: str | int,
) -> Ticket | dict:
    category_info = get_category(category)
    schema = category_info["parameters"]
    if missing := [f for f in schema["required"] if f not in fields]:
        return {"missing_fields": missing}

    data = Ticket(
        id=randint(1234, 7890),
        phone=phone,
        description=description,
        category=category_info["name"],
        status="Зарегистрирована",
        deadline=date.today().strftime("%d.%m.%Y"),
        **fields
    )

    TICKETS.append(data)
    return data


def get_ticket(number: str) -> Ticket | None:
    return next(
        (ticket for ticket in TICKETS if str(ticket.id) == number.strip()),
        None,
    )


def get_tickets_by_address(
    street: str,
    house: str,
    apartment: int | None = None,
) -> list[Ticket]:
    return [
        ticket for ticket in TICKETS
        if getattr(ticket, "street", "").strip().lower() == street.strip().lower()
        and getattr(ticket, "house", "").strip().lower() == house.strip().lower()
        and (apartment is None or getattr(ticket, "apartment", "") == apartment)
    ]
