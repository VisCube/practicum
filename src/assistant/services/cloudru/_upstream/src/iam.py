from time import monotonic

from requests import post

from src.config import IAM_TOKEN_URL, RAG_KEY_ID, RAG_KEY_SECRET

cached_token = None
cached_token_expires_at = 0.0


def get_access_token() -> str:
    global cached_token, cached_token_expires_at

    if cached_token and monotonic() < cached_token_expires_at:
        return cached_token

    response = post(
        IAM_TOKEN_URL,
        json={"keyId": RAG_KEY_ID, "secret": RAG_KEY_SECRET},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    cached_token = data["access_token"]
    cached_token_expires_at = monotonic() + 55 * 60
    return cached_token
