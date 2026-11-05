from difflib import get_close_matches
from json import loads
from pathlib import Path
from random import randint

CATEGORIES_PATH = Path(__file__).parent.parent.parent / "classifier" / "categories.json"
CATEGORIES: list[dict] = loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
CATEGORIES_NAMES: list[str] = [category["name"] for category in CATEGORIES]
FALLBACK_CATEGORY = "Иное / Иное"


def get_category(category: str) -> dict:
    matches = get_close_matches(category, CATEGORIES_NAMES, n=1, cutoff=0.6)
    name = matches[0] if matches else FALLBACK_CATEGORY
    return next(item for item in CATEGORIES if item["name"] == name)


def create_ticket(
    full_name: str,
    address: str,
    details: str,
    category: str,
    **fields: str | int
) -> dict:
    category_info = get_category(category)
    schema = category_info["parameters"]
    missing = [f for f in schema["required"] if f not in fields]
    if missing:
        return {"missing_fields": missing}

    data = {
        "full_name": full_name,
        "address": address,
        "details": details,
        "category": category_info["name"],
        **{field: fields[field] for field in schema["properties"] if field in fields},

        "id": randint(1234, 9876)
    }

    return data
