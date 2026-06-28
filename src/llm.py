"""LLM HTTP client. OpenAI-compatible chat completions."""
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import requests

load_dotenv()  # picks up .env at repo root if present


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("LLM_API_KEY", "")
        if not key:
            raise LLMError("LLM_API_KEY environment variable is not set or is empty")
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        return cls(base_url=base, api_key=key, model=model)


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    response_format: dict | None = None,
    timeout: int = 60,
    max_tokens: int | None = None,
) -> str:
    """Call an OpenAI-compatible chat endpoint. Return assistant text.

    Endpoint path is taken from ``LLM_ENDPOINT_PATH`` env var (default
    ``/chat/completions`` for OpenAI / Ollama / LM Studio; set to
    ``/text/chatcompletion_v2`` for MiniMax).
    """
    cfg = LLMConfig.from_env()
    path = os.environ.get("LLM_ENDPOINT_PATH", "/chat/completions")
    url = f"{cfg.base_url}{path}"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": model or cfg.model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if max_tokens is not None:
        # Some providers (OpenAI legacy, Typhoon) use max_tokens; newer ones
        # use max_completion_tokens. Send both — the provider ignores unknown.
        payload["max_tokens"] = max_tokens
        payload["max_completion_tokens"] = max_tokens
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.HTTPError as e:
        raise LLMError(f"LLM HTTP error: {e}") from e
    except (requests.RequestException, ValueError) as e:
        raise LLMError(f"LLM response error: {e}") from e
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Malformed LLM response: {type(e).__name__}: {e}") from e