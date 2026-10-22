from difflib import get_close_matches
from pathlib import Path

from pypdf import PdfReader

KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"


def normalize(filename: str) -> str:
    return filename.strip().lower().removesuffix(".pdf").replace("ё", "е")


def list_knowledge_files() -> list[str]:
    return sorted(p.name for p in KNOWLEDGE_DIR.glob("*.pdf"))


# TODO заменить на RAG от Cloud.ru
def search_knowledge_base(filename: str) -> str:
    files = {p.name: p for p in KNOWLEDGE_DIR.glob("*.pdf")}
    normalized = {normalize(name): name for name in files}

    target = normalize(filename)
    matched_name = normalized.get(target)

    if matched_name is None:
        close = get_close_matches(target, normalized.keys(), n=1, cutoff=0.6)
        matched_name = normalized[close[0]] if close else None

    if matched_name is None:
        return "Ответа в базе знаний не найдено"

    path = files[matched_name]
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
