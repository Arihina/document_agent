from __future__ import annotations

from typing import Generator

import ollama

from app.core.config import settings


OLLAMA_MODEL = settings.OLLAMA_MODEL

_client = ollama.Client(host=settings.OLLAMA_HOST)


def _format_history(history: list[tuple[str, str]], max_turns: int = 10) -> str:
    recent = history[-(max_turns * 2):]
    lines = []
    for role, text in recent:
        label = "Пользователь" if role == "user" else "Ассистент"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _general_prompt(history: list[tuple[str, str]], question: str) -> str:
    return f"""Ты русскоязычный AI ассистент, который помогает разбираться с документами.
Отвечай ТОЛЬКО на русском языке.
Сейчас документ не прикреплён к этому вопросу — это либо общий вопрос,
либо вопрос до загрузки файла. Если по смыслу вопроса нужен документ,
которого нет, — вежливо попроси его прикрепить, не выдумывай ответ.

Предыдущий диалог:
{_format_history(history)}

Вопрос пользователя:
{question}

Ответ:""".strip()


def _document_prompt(
    document_markdown: str,
    history: list[tuple[str, str]],
    question: str,
) -> str:
    return f"""Ты русскоязычный AI ассистент, который отвечает на вопросы по содержимому документа.
Отвечай ТОЛЬКО на русском языке.
Используй ТОЛЬКО информацию из документа ниже. Если ответа в документе нет —
так и скажи, не выдумывай.

Документ:
{document_markdown}

Предыдущий диалог:
{_format_history(history)}

Вопрос пользователя:
{question}

Ответ:""".strip()


def stream_answer(
    question: str,
    document_markdown: str | None,
    history: list[tuple[str, str]],
) -> Generator[str, None, None]:
    if document_markdown:
        prompt = _document_prompt(document_markdown, history, question)
    else:
        prompt = _general_prompt(history, question)

    stream = _client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        token: str = chunk["message"]["content"]
        yield token
