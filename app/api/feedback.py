from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db import crud

from app.api.deps import get_owned_message

router = APIRouter()
_MISSING = object()


def _feedback_out(fb) -> dict:
    return {
        "message_id": fb.message_id,
        "vote": fb.vote,
        "comment": fb.comment,
        "created_at": fb.created_at.isoformat(),
        "updated_at": fb.updated_at.isoformat(),
    }


@router.post("/messages/{message_id}/feedback", status_code=200)
async def set_feedback(
    body: dict = Body(default={}),
    msg=Depends(get_owned_message),
    db: AsyncSession = Depends(get_db),
):
    raw_vote = body.get("vote", _MISSING)
    comment: Optional[str] = body.get("comment", _MISSING)

    if raw_vote is not _MISSING and raw_vote not in (1, -1, None):
        raise HTTPException(422, "vote должен быть 1, -1 или null")
    
    if msg.role != "assistant":
        raise HTTPException(400, "Оценивать можно только ответы ассистента")
    
    fb = await crud.upsert_feedback(
        db, message_id=msg.id, vote=raw_vote, comment=comment, missing=_MISSING
    )
    if fb is None:
        raise HTTPException(500, "Не удалось сохранить оценку")
    
    return _feedback_out(fb)


@router.get("/messages/{message_id}/feedback")
async def get_feedback(
    msg=Depends(get_owned_message),
    db: AsyncSession = Depends(get_db),
):
    fb = await crud.get_feedback(db, msg.id)
    if fb is None:
        return {"message_id": msg.id, "vote": None, "comment": None}
    
    return _feedback_out(fb)


@router.delete("/messages/{message_id}/feedback", status_code=204)
async def delete_feedback(
    msg=Depends(get_owned_message),
    db: AsyncSession = Depends(get_db),
):
    fb = await crud.get_feedback(db, msg.id)
    if not fb:
        raise HTTPException(404, "Оценка не найдена")
    
    fb.vote = None
    fb.comment = None

    await db.commit()
