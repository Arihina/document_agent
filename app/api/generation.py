from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.context import (
    BuiltPrompt, ContextOverflow, Document, build_prompt,
)
from app.core.llm import stream_answer
from app.core.tokens import make_counter
from app.db.database import AsyncSessionLocal
from app.db import crud

log = logging.getLogger(__name__)

TITLE_MAX_LEN = 80

_counter = None


def get_counter():
    """Счётчик токенов создаётся лениво: загрузка токенайзера не должна
    блокировать старт приложения."""
    global _counter
    if _counter is None:
        _counter = make_counter(settings.TOKENIZER_REPO)
    return _counter


@dataclass
class Generation:
    answer: str = ""
    usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def collect(prompt: str, options: dict | None = None) -> Generation:
    out = Generation()

    for token, usage in stream_answer(prompt, options):
        if token:
            out.answer += token
        if usage is not None:
            out.usage = usage

    return out


async def resolve_files(
    db: AsyncSession, user_id: UUID, file_uuids: list[UUID],
) -> list[Document]:
    if not file_uuids:
        return []

    if len(file_uuids) > settings.MAX_ATTACHED_FILES:
        raise HTTPException(
            400,
            f"Прикреплено файлов: {len(file_uuids)}, максимум "
            f"{settings.MAX_ATTACHED_FILES}. Уменьшите число вложений или "
            "увеличьте MAX_ATTACHED_FILES.",
        )

    counter = get_counter()
    docs: list[Document] = []

    for file_uuid in file_uuids:
        f = await crud.get_file_for_user(db, file_uuid, user_id)
        if f is None:
            raise HTTPException(404, f"Файл {file_uuid} (file_id) не найден")
        if f.status != "done":
            raise HTTPException(
                400,
                f"Файл «{f.filename}» ещё не готов к использованию "
                f"(статус обработки: {f.status})",
            )

        tokens = f.markdown_tokens
        if tokens is None and f.markdown_content:
            tokens = counter.count(f.markdown_content)
            await crud.set_file_tokens(db, f, tokens)

        docs.append(Document(
            filename=f.filename,
            markdown=f.markdown_content or "",
            tokens=tokens,
        ))

    return docs


def build(
    question: str,
    documents: list[Document],
    history: list[tuple[str, str]],
    instructions: str | None,
    max_output_tokens: int | None,
) -> BuiltPrompt:
    """Транспортная граница: ContextOverflow становится 413."""
    try:
        built = build_prompt(
            question=question,
            documents=documents,
            history=history,
            instructions=instructions,
            counter=get_counter(),
            num_ctx=settings.CONTEXT_WINDOW,
            reserve_output=max_output_tokens or settings.RESERVE_OUTPUT_TOKENS,
            safety=settings.CONTEXT_SAFETY_TOKENS,
            history_min_tokens=settings.HISTORY_MIN_TOKENS,
            overflow=settings.DOCUMENT_OVERFLOW,
        )
    except ContextOverflow as e:
        raise HTTPException(413, str(e))

    log.info("окно контекста: %s", built.stats.as_log())
    return built


async def persist(
    user_id: UUID,
    question: str,
    answer: str,
    sources: list[str] | None,
    model: str,
    assistant_id: UUID,
    conversation_id: UUID | None,
    usage: dict | None,
) -> None:
    usage = usage or {}
    counter = get_counter()

    async with AsyncSessionLocal() as write_db:
        await crud.add_message(
            write_db, user_id=user_id, role="user", content=question,
            conversation_id=conversation_id,
            tokens=counter.count(question),
        )
        await crud.add_message(
            write_db, user_id=user_id, role="assistant", content=answer,
            sources=sources, model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            id=assistant_id, conversation_id=conversation_id,
            tokens=counter.count(answer),
        )
        if conversation_id is not None:
            title = question[:TITLE_MAX_LEN] + \
                ("…" if len(question) > TITLE_MAX_LEN else "")
            await crud.touch_conversation(write_db, conversation_id, title=title)
