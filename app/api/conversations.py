from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db import crud
from app.core.auth import get_user_id

from app.api.deps import get_owned_conversation

router = APIRouter(prefix="/v1/platform/conversations", tags=["conversations"])


def _fmt_conversation(c) -> dict:
    return {
        "id": c.id, "title": c.title,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
    }


def _fmt_message(m) -> dict:
    fb = m.feedback
    return {
        "id": m.id, "role": m.role, "content": m.content,
        "sources": m.sources or [],
        "created_at": m.created_at.isoformat(),
        "feedback": {"vote": fb.vote, "comment": fb.comment} if fb else None,
    }


@router.post("", status_code=201)
async def create_conversation(
    body: dict = Body(default={}),
    user_id=Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    c = await crud.create_conversation(db, user_id=user_id, title=body.get("title"))
    return _fmt_conversation(c)


@router.get("")
async def list_conversations(
    user_id=Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    return [_fmt_conversation(c) for c in await crud.list_conversations(db, user_id)]


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    c=Depends(get_owned_conversation),
    db: AsyncSession = Depends(get_db),
):
    return [_fmt_message(m) for m in await crud.get_conversation_messages(db, c.id)]


@router.patch("/{conversation_id}")
async def rename_conversation(
    body: dict = Body(...),
    c=Depends(get_owned_conversation),
    db: AsyncSession = Depends(get_db),
):
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(422, "title не может быть пустым")
    return _fmt_conversation(await crud.rename_conversation(db, c, title))


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    c=Depends(get_owned_conversation),
    db: AsyncSession = Depends(get_db),
):
    await crud.delete_conversation(db, c)
