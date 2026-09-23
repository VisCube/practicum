"""Поиск по data/knowledge/*.md: документ режется на секции по «## », скоринг по совпадению основ слов
Для прототипа достаточно; на проде сюда встанет эмбеддинг-поиск с тем же интерфейсом"""

from pathlib import Path

from assistant.core.models import Passage
from assistant.nlu.text import normalize

_STOP = {
    "это", "что", "как", "для", "или", "при", "если", "мне", "меня", "нам", "вам", "надо",
    "нужно", "можно", "есть", "быть", "такое", "какой", "какая", "какие", "где", "когда",
    "почему", "зачем", "сколько", "куда", "откуда", "скажите", "подскажите", "пожалуйста",
    "хочу", "узнать", "спросить", "вопрос", "здравствуйте", "добрый", "день",
}


def stem(word: str) -> str:
    """Обрезает слово до первых 5 букв - грубая стемизация без морфологии"""
    return word[:5] if len(word) >= 6 else word


def tokens(text: str) -> set[str]:
    """Нормализует текст и возвращает множество стемов без стоп-слов и коротких слов"""
    return {stem(w) for w in normalize(text).split() if len(w) >= 3 and w not in _STOP}


class MarkdownRetriever:
    """Поиск по Markdown-файлам: секции по ## , скоринг по совпадению стемов слов"""
    def __init__(self, passages: list[Passage], *, min_score: float = 2.0) -> None:
        self.passages = passages
        self.min_score = min_score      # 2.0 = слово в заголовке либо два слова в тексте
        self._index = [(p, tokens(p.heading), tokens(p.text)) for p in passages]

    @classmethod
    def from_dir(cls, path: Path) -> "MarkdownRetriever":
        passages: list[Passage] = []
        for md_file in sorted(path.glob("*.md")):
            passages.extend(cls._parse(md_file))
        return cls(passages)

    @staticmethod
    def _parse(md: Path) -> list[Passage]:
        """Нарезает Markdown на секции по заголовкам ## , каждая секция - один Passage"""
        result: list[Passage] = []
        current_heading, line_buffer = None, []
        for line in md.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if current_heading and "".join(line_buffer).strip():
                    result.append(Passage(source=md.stem, heading=current_heading, text=" ".join(line_buffer).strip()))
                current_heading, line_buffer = line[3:].strip(), []
            elif line.startswith("# "):
                continue
            else:
                if line.strip():
                    line_buffer.append(line.strip())
        if current_heading and "".join(line_buffer).strip():
            result.append(Passage(source=md.stem, heading=current_heading, text=" ".join(line_buffer).strip()))
        return result

    async def search(self, query: str, *, sources: list[str] | None = None, limit: int = 3) -> list[Passage]:
        """Возвращает фрагменты, отсортированные по релевантности - совпадение в заголовке весит вдвое больше"""
        query_tokens = tokens(query)
        if not query_tokens:
            return []
        scored: list[Passage] = []
        for passage, heading_tokens, text_tokens in self._index:
            if sources and passage.source not in sources:
                continue
            score = 2.0 * len(query_tokens & heading_tokens) + 1.0 * len(query_tokens & text_tokens)
            if score >= self.min_score:
                scored.append(passage.model_copy(update={"score": score}))
        scored.sort(key=lambda passage: passage.score, reverse=True)
        return scored[:limit]