from json import dumps
from random import randint

from src.tools.classifier import classify


def create_ticket(
    full_name: str,
    address: str,
    details: str
) -> str:
    data = {
        "full_name": full_name,
        "address": address,
        "details": details,

        "category": classify(details),
        "id": randint(1234, 9876)
    }

    return dumps(data, ensure_ascii=False)
