import requests

from src.config import RAG_BASE_URL
from src.iam import get_access_token


def search_knowledge_base(query: str) -> str:
    response = requests.post(
        f"{RAG_BASE_URL}/api/v2/retrieve",
        headers={"Authorization": f"Bearer {get_access_token()}"},
        json={
            "knowledge_base_version": "latest",
            "query": query,
            "retrieval_configuration": {
                "number_of_results": 3,
                "retrieval_type": "SEMANTIC",
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()["results"]

    if not results:
        return "Ответа в базе знаний не найдено"

    return "\n\n".join(result["content"] for result in results)
