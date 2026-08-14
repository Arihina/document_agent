from __future__ import annotations

from typing import Generator

import ollama

from app.core.config import settings


OLLAMA_MODEL = settings.OLLAMA_MODEL

_client = ollama.Client(host=settings.OLLAMA_HOST)


def stream_answer(
    prompt: str,
    options: dict | None = None,
) -> Generator[tuple[str, dict | None], None, None]:
    """Генерация по готовому промпту.

    Сборка промпта живёт в app/core/context.py: посчитать бюджет можно только
    до отправки, поэтому генерация не должна склеивать текст сама.
    """
    stream = _client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options=options or None,
    )
    prompt_tokens = 0
    completion_tokens = 0
    for chunk in stream:
        token: str = chunk["message"]["content"]
        if token:
            yield token, None
        if chunk.get("done"):
            prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
            completion_tokens = chunk.get("eval_count", 0) or 0

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    yield "", usage
