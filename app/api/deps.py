from uuid import UUID
from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db import crud
from app.core.auth import get_user_id


def parse_completion_id(completion_id: str) -> UUID:
    raw = completion_id.removeprefix("chatcmpl-")
    try:
        return UUID(raw)
    except ValueError:
        raise HTTPException(422, "Некорректный id completion'а")


def parse_file_id(file_id: str) -> UUID:
    raw = file_id.removeprefix("file-")
    try:
        return UUID(raw)
    except ValueError:
        raise HTTPException(422, "Некорректный id файла")


async def get_owned_completion(
    completion_id: str,
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    msg = await crud.get_message_for_user(db, parse_completion_id(completion_id), user_id)
    if msg is None:
        raise HTTPException(404, "Сообщение не найдено")
    return msg


async def get_owned_conversation(
    conversation_id: UUID,
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    c = await crud.get_conversation(db, conversation_id, user_id)
    if c is None:
        raise HTTPException(404, "Чат не найден")
    return c


async def get_owned_file(
    file_id: str,
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    f = await crud.get_file_for_user(db, parse_file_id(file_id), user_id)
    if f is None:
        raise HTTPException(404, "Файл не найден")
    return f
