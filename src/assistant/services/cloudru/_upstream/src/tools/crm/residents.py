from dataclasses import dataclass
from re import sub

from .addresses import Address, ADDRESSES


@dataclass(frozen=True)
class Phone:
    number: str


@dataclass
class Resident:
    name: str
    phone: Phone
    address: Address
    apartment: int


RESIDENTS: list[Resident] = [
    Resident(
        name="Иванов Иван Иванович",
        phone=Phone("+79990000001"),
        address=ADDRESSES[0],
        apartment=15,
    ),
    Resident(
        name="Васильева Василиса Васильевна",
        phone=Phone("+79990000002"),
        address=ADDRESSES[0],
        apartment=45,
    ),
    Resident(
        name="Петров Пётр Петрович",
        phone=Phone("+79990000003"),
        address=ADDRESSES[2],
        apartment=8,
    ),
]


def normalize_phone(phone: str) -> str:
    phone = sub(r"[\s()+-]", "", phone)
    if len(phone) == 11 and phone.startswith(("7", "8")):
        phone = phone[1:]
    return f"+7{phone}"


def get_resident_by_phone(phone: str) -> Resident | None:
    phone = Phone(normalize_phone(phone))
    return next((res for res in RESIDENTS if res.phone == phone), None)
