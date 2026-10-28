from difflib import get_close_matches
from json import dumps, loads
from pathlib import Path
from random import randint

CATEGORIES_PATH = (
        Path(__file__).parent.parent.parent
        / "classifier" / "categories.json"
)
CATEGORIES: list[str] = loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
FALLBACK_CATEGORY = "Иное / Иное"


def create_ticket(
    full_name: str,
    address: str,
    details: str,
    category: str
) -> str:
    close_matches = get_close_matches(category, CATEGORIES, n=1, cutoff=0.6)
    category = close_matches[0] if close_matches else FALLBACK_CATEGORY

    data = {
        "full_name": full_name,
        "address": address,
        "details": details,
        "category": category,

        "id": randint(1234, 9876)
    }

    return dumps(data, ensure_ascii=False)
