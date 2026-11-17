from difflib import get_close_matches
from json import loads
from pathlib import Path


CATEGORIES_PATH = Path(
    __file__
).parent.parent.parent / "classifier" / "categories.json"
CATEGORIES: list[dict] = loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
CATEGORIES_NAMES: list[str] = [category["name"] for category in CATEGORIES]
FALLBACK_CATEGORY = "Иное / Иное"


def get_category(category: str) -> dict:
    matches = get_close_matches(category, CATEGORIES_NAMES, n=1, cutoff=0.6)
    name = matches[0] if matches else FALLBACK_CATEGORY
    return next(item for item in CATEGORIES if item["name"] == name)
