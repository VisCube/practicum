"""База знаний-заглушка с фиксированным набором фрагментов"""

from assistant.core.models import Passage


class FakeKnowledgeBase:
    """Заглушка базы знаний - возвращает фиксированный набор фрагментов"""
    def __init__(self, passages: list[Passage] | None = None) -> None:
        self.passages = passages or []

    async def search(self, query: str, *, sources: list[str] | None = None,
                     limit: int = 3) -> list[Passage]:
        results = self.passages
        if sources:
            results = [passage for passage in results if passage.source in sources]
        return results[:limit]