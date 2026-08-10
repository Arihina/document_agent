from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File as FastAPIFile, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db import crud
from app.models.models import File

from app.core import mineru
from app.core.auth import get_user_id
from app.api.deps import get_owned_file
from app.api.openai_format import file_object as _fmt

router = APIRouter(prefix="/v1/files", tags=["files"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.post("", status_code=201)
async def upload_file(
    file: UploadFile = FastAPIFile(...),
    conversation_id: str | None = Form(None),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    conv_uuid: UUID | None = None
    if conversation_id is not None:
        try:
            conv_uuid = UUID(conversation_id)
        except ValueError:
            raise HTTPException(400, "conversation_id должен быть UUID")
        if await crud.get_conversation(db, conv_uuid, user_id) is None:
            raise HTTPException(404, "Чат (conversation_id) не найден")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"Файл больше {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ")

    f = await crud.create_file(
        db, user_id=user_id,
        filename=file.filename or "документ",
        mime_type=file.content_type,
        size_bytes=len(content),
        content=content,
        conversation_id=conv_uuid,
    )
    f = await crud.set_file_processing(db, f)

    try:
        markdown = await mineru.parse_document(
            filename=f.filename, content=content, mime_type=f.mime_type,
        )
    except mineru.MinerUError as e:
        await crud.set_file_failed(db, f, str(e))
        raise HTTPException(502, f"Не удалось обработать документ: {e}")

    f = await crud.set_file_done(
        db, f, markdown_content=markdown,
        ocr_backend=f"mineru:{mineru.settings.MINERU_BACKEND}:{mineru.settings.MINERU_LANG}",
    )
    return _fmt(f)


@router.get("")
async def list_files(
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    return {"object": "list", "data": [_fmt(f) for f in await crud.list_files(db, user_id)]}


@router.get("/{file_id}")
async def get_file(f: File = Depends(get_owned_file)):
    return _fmt(f)


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    f: File = Depends(get_owned_file),
    db: AsyncSession = Depends(get_db),
):
    await crud.delete_file(db, f)
    return {"id": f"file-{f.id}", "object": "file", "deleted": True}
