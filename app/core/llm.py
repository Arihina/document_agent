from __future__ import annotations

from typing import Generator

import ollama

from app.core.config import settings


OLLAMA_MODEL = settings.OLLAMA_MODEL

_client = ollama.Client(host=settings.OLLAMA_HOST)


def _format_history(history: list, max_turns: int = 10) -> str:
    recent = history[-(max_turns * 2):]
    lines = []

    for item in recent:
        if isinstance(item, dict):
            role, text = item.get("role"), item.get("content", "")
        else:
            role, text = item

        label = "Пользователь" if role == "user" else "Ассистент"
        lines.append(f"{label}: {text}")

    return "\n".join(lines)


def _instructions_block(instructions: str | None) -> str:
    if not instructions or not instructions.strip():
        return ""
    return f"\nДополнительные инструкции пользователя:\n{instructions.strip()}\n"


def _general_prompt(
    history: list[tuple[str, str]], question: str, instructions: str | None = None,
) -> str:
    return f"""Ты русскоязычный AI ассистент, который помогает разбираться с документами.
Отвечай ТОЛЬКО на русском языке.
Сейчас документ не прикреплён к этому вопросу — это либо общий вопрос,
либо вопрос до загрузки файла. Если по смыслу вопроса нужен документ,
которого нет, — вежливо попроси его прикрепить, не выдумывай ответ.
{_instructions_block(instructions)}
Предыдущий диалог:
{_format_history(history)}

Вопрос пользователя:
{question}

Ответ:""".strip()


def _document_prompt(
    document_markdown: str,
    history: list[tuple[str, str]],
    question: str,
    instructions: str | None = None,
) -> str:
    return f"""Ты русскоязычный AI ассистент, который отвечает на вопросы по содержимому документа.
Отвечай ТОЛЬКО на русском языке.
Используй ТОЛЬКО информацию из документа ниже. Если ответа в документе нет —
так и скажи, не выдумывай.
{_instructions_block(instructions)}
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
    instructions: str | None = None,
    options: dict | None = None,
) -> Generator[tuple[str, dict | None], None, None]:
    if document_markdown:
        prompt = _document_prompt(
            document_markdown, history, question, instructions)
    else:
        prompt = _general_prompt(history, question, instructions)

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
