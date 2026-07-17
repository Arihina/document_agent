from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import iterate_in_threadpool

from app.db.database import get_db, AsyncSessionLocal
from app.db import crud
from app.models.models import Document

from app.core import llm, mineru
from app.core.auth import get_user_id

from app.api.deps import get_owned_session

router = APIRouter()

_histories: dict[UUID, list[tuple[str, str]]] = {}

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_CHAT_BODY_SCHEMA = {
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            }
        },
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "file": {"type": "string", "format": "binary"},
                },
                "required": ["message"],
            }
        },
    },
    "required": True,
}


async def _read_capped(upload: StarletteUploadFile, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Файл больше {limit // (1024 * 1024)} МБ")
    return data


def _fmt_session(s) -> dict:
    return {"id": s.id, "title": s.title,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat()}


def _fmt_message(m) -> dict:
    fb = m.feedback
    doc = m.document
    return {
        "id": m.id, "role": m.role, "content": m.content,
        "sources": m.sources if m.sources else [],
        "created_at": m.created_at.isoformat(),
        "feedback": {"vote": fb.vote, "comment": fb.comment} if fb else None,
        "document": (
            {
                "id": doc.id,
                "filename": doc.original_filename,
                "status": doc.status,
                "error": doc.error_message,
            }
            if doc else None
        ),
    }


@router.post("/sessions", status_code=201)
async def create_session(
    body: dict = Body(default={}),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    s = await crud.create_session(db, user_id=user_id, title=body.get("title"))
    return _fmt_session(s)


@router.get("/sessions")
async def list_sessions(
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db)
):
    return [_fmt_session(s) for s in await crud.list_sessions(db, user_id)]


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    s=Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    return [_fmt_message(m) for m in await crud.get_messages(db, s.id)]


@router.patch("/sessions/{session_id}")
async def rename_session(
    body: dict = Body(...),
    s=Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(422, "title не может быть пустым")
    return _fmt_session(await crud.rename_session(db, s, title))


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    s=Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    await crud.delete_session(db, s)
    _histories.pop(s.id, None)


@router.post("/sessions/{session_id}/chat", openapi_extra={"requestBody": _CHAT_BODY_SCHEMA})
async def chat(
    request: Request,
    s=Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    content_type = request.headers.get("content-type", "")
    file: StarletteUploadFile | None = None

    if content_type.startswith("multipart/form-data") or content_type.startswith(
        "application/x-www-form-urlencoded"
    ):
        form = await request.form()
        message = form.get("message", "")
        upload = form.get("file")
        if isinstance(upload, StarletteUploadFile):
            file = upload
    elif content_type.startswith("application/json"):
        body = await request.json()
        message = body.get("message", "")
    else:
        raise HTTPException(
            415, f"Неподдерживаемый Content-Type: {content_type or '<пусто>'}")

    if not isinstance(message, str):
        raise HTTPException(422, "message обязателен и должен быть строкой")

    question = message.strip()
    if not question:
        raise HTTPException(422, "Пустой вопрос")

    history = _histories.setdefault(
        s.id, await crud.build_history(db, s.id)
    )

    if s.title is None:
        await crud.rename_session(
            db, s,
            question[:80] + ("…" if len(question) > 80 else "")
        )

    user_msg = await crud.add_message(db, s.id, "user", question)

    active_document: Document | None = None

    if file is not None:
        content = await _read_capped(file)
        doc = await crud.create_document(
            db,
            session_id=s.id,
            message_id=user_msg.id,
            original_filename=file.filename or "документ",
            mime_type=file.content_type,
            size_bytes=len(content),
            content=content,
        )
        await crud.set_document_processing(db, doc)
        try:
            markdown = await mineru.parse_document(
                filename=doc.original_filename,
                content=content,
                mime_type=doc.mime_type,
            )
        except mineru.MinerUError as e:
            await crud.set_document_failed(db, doc, str(e))
            raise HTTPException(502, f"Не удалось обработать документ: {e}")

        doc = await crud.set_document_done(
            db, doc,
            markdown_content=markdown,
            ocr_backend=f"mineru:{mineru.settings.MINERU_BACKEND}:{mineru.settings.MINERU_LANG}",
        )
        active_document = doc
    else:
        active_document = await crud.get_latest_document(db, s.id)

    document_markdown = active_document.markdown_content if active_document else None

    async def _save(full_answer: str) -> UUID:
        async with AsyncSessionLocal() as write_db:
            sources = [
                active_document.original_filename] if active_document else None
            msg = await crud.add_message(write_db, s.id, "assistant", full_answer, sources)
            return msg.id

    state: dict = {"answer": ""}

    def _gen():
        if active_document:
            preview = (active_document.markdown_content or "")[:300]
            chunks_payload = [{
                "text": preview,
                "source": active_document.original_filename,
                "score": 1.0,
            }]
            yield f"data: {json.dumps({'chunks': chunks_payload}, ensure_ascii=False)}\n\n"

        full_answer = ""
        for token in llm.stream_answer(question, document_markdown, history):
            full_answer += token
            if token:
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        state["answer"] = full_answer

    async def _async_gen():
        async for chunk in iterate_in_threadpool(_gen()):
            yield chunk

        message_id = await _save(state["answer"])
        yield f"data: {json.dumps({'message_id': str(message_id)}, ensure_ascii=False)}\n\n"

        history.append(("user", question))
        history.append(("assistant", state["answer"]))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions/{session_id}/reset")
async def reset(s=Depends(get_owned_session)):
    _histories.pop(s.id, None)
    return {"status": "ok"}


@router.get("/sessions/clear")
async def clear_all_sessions(
    db: AsyncSession = Depends(get_db),
):
    await crud.delete_all_sessions(db)
    _histories.clear()
    return {"status": "ok", "message": "Все чаты удалены"}
