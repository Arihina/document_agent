from uuid import UUID
from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db import crud
from app.core.auth import get_user_id


async def get_owned_session(
    session_id: UUID,
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    s = await crud.get_session(db, session_id, user_id)
    if s is None:
        raise HTTPException(404, "Чат не найден")
    return s


async def get_owned_message(
    message_id: UUID,
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    m = await crud.get_message_for_user(db, message_id, user_id)
    if m is None:
        raise HTTPException(404, "Сообщение не найдено")
    return m
