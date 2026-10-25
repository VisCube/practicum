from json import dumps
from random import randint


def create_ticket(
    full_name: str,
    address: str,
    details: str,
    category: str
) -> str:
    data = {
        "full_name": full_name,
        "address": address,
        "details": details,
        "category": category,
        "id": randint(1234, 9876)
    }

    return dumps(data, ensure_ascii=False)
