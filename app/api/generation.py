from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import stream_answer
from app.db.database import AsyncSessionLocal
from app.db import crud

TITLE_MAX_LEN = 80


@dataclass
class Generation:
    answer: str = ""
    usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def collect(question, document_markdown, history, instructions=None, options=None) -> Generation:
    out = Generation()

    for token, usage in stream_answer(
        question, document_markdown, history,
        instructions=instructions, options=options,
    ):
        if token:
            out.answer += token
        if usage is not None:
            out.usage = usage

    return out


async def resolve_file(
    db: AsyncSession, user_id: UUID, file_uuid: UUID | None,
) -> tuple[str | None, str | None, UUID | None]:
    if file_uuid is None:
        return None, None, None

    f = await crud.get_file_for_user(db, file_uuid, user_id)
    if f is None:
        raise HTTPException(404, "Файл (file_id) не найден")

    if f.status != "done":
        raise HTTPException(
            400, f"Файл ещё не готов к использованию (статус обработки: {f.status})")

    return f.markdown_content, f.filename, f.id


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

    async with AsyncSessionLocal() as write_db:
        await crud.add_message(
            write_db, user_id=user_id, role="user", content=question,
            conversation_id=conversation_id,
        )
        await crud.add_message(
            write_db, user_id=user_id, role="assistant", content=answer,
            sources=sources, model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            id=assistant_id, conversation_id=conversation_id,
        )
        if conversation_id is not None:
            title = question[:TITLE_MAX_LEN] + \
                ("…" if len(question) > TITLE_MAX_LEN else "")
            await crud.touch_conversation(write_db, conversation_id, title=title)
