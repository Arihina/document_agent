from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from app.models.models import ChatSession, ChatMessage, MessageFeedback, Document


async def create_session(db: AsyncSession, user_id, title: Optional[str] = None) -> ChatSession:
    s = ChatSession(user_id=user_id, title=title)
    db.add(s)
    
    await db.commit()
    await db.refresh(s)

    return s


async def get_session(db: AsyncSession, session_id: UUID, user_id) -> Optional[ChatSession]:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )

    return result.scalar_one_or_none()


async def list_sessions(db: AsyncSession, user_id) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )

    return result.scalars().all()


async def rename_session(db: AsyncSession, s: ChatSession, title: str) -> ChatSession:
    s.title = title
    await db.commit()
    await db.refresh(s)
    return s


async def delete_session(db: AsyncSession, s: ChatSession) -> None:
    await db.delete(s)
    await db.commit()


async def delete_all_sessions(db: AsyncSession) -> None:
    await db.execute(delete(ChatSession))
    await db.commit()


async def _touch(db: AsyncSession, session_id: UUID) -> None:
    await db.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(updated_at=datetime.now(timezone.utc))
    )
    await db.commit()


async def add_message(
    db: AsyncSession,
    session_id: UUID,
    role: str,
    content: str,
    sources: Optional[list] = None,
) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    await _touch(db, session_id)

    return msg


async def get_messages(db: AsyncSession, session_id: UUID) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )

    return result.scalars().all()


async def get_message(db: AsyncSession, message_id: UUID) -> Optional[ChatMessage]:
    result = await db.execute(select(ChatMessage).where(ChatMessage.id == message_id))
    return result.scalar_one_or_none()


async def get_message_for_user(db: AsyncSession, message_id: UUID, user_id) -> Optional[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .join(ChatSession, ChatMessage.session_id == ChatSession.id)
        .where(
            ChatMessage.id == message_id,
            ChatSession.user_id == user_id,
        )
    )

    return result.scalar_one_or_none()


async def build_history(db: AsyncSession, session_id: UUID) -> list[tuple[str, str]]:
    messages = await get_messages(db, session_id)
    return [(m.role, m.content) for m in messages]


async def upsert_feedback(
    db: AsyncSession,
    message_id: UUID,
    vote,
    comment,
    missing,
) -> Optional[MessageFeedback]:
    msg = await get_message(db, message_id)
    if not msg or msg.role != "assistant":
        return None

    result = await db.execute(
        select(MessageFeedback).where(MessageFeedback.message_id == message_id)
    )
    fb = result.scalar_one_or_none()

    if fb is None:
        fb = MessageFeedback(
            message_id=message_id,
            vote=None if vote is missing else vote,
            comment=None if comment is missing else comment,
        )
        db.add(fb)
    else:
        if vote is not missing:
            fb.vote = vote
        if comment is not missing:
            fb.comment = comment
        fb.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(fb)

    return fb


async def get_feedback(db: AsyncSession, message_id: UUID) -> Optional[MessageFeedback]:
    result = await db.execute(
        select(MessageFeedback).where(MessageFeedback.message_id == message_id)
    )
    return result.scalar_one_or_none()


async def create_document(
    db: AsyncSession,
    session_id: UUID,
    message_id: UUID,
    original_filename: str,
    mime_type: Optional[str],
    size_bytes: int,
    content: bytes,
) -> Document:
    doc = Document(
        session_id=session_id,
        message_id=message_id,
        original_filename=original_filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        content=content,
        status="pending",
    )
    db.add(doc)

    await db.commit()
    await db.refresh(doc)

    return doc


async def get_document(db: AsyncSession, document_id: UUID) -> Optional[Document]:
    """Метаданные + статус, без блоба файла (content отложен)."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    return result.scalar_one_or_none()


async def get_document_by_message(db: AsyncSession, message_id: UUID) -> Optional[Document]:
    result = await db.execute(select(Document).where(Document.message_id == message_id))
    return result.scalar_one_or_none()


async def get_document_content(db: AsyncSession, document_id: UUID) -> Optional[Document]:
    """То же самое, но с явной догрузкой content — для передачи в MinerU."""
    result = await db.execute(
        select(Document)
        .options(undefer(Document.content))
        .where(Document.id == document_id)
    )

    return result.scalar_one_or_none()


async def set_document_processing(db: AsyncSession, document: Document) -> Document:
    document.status = "processing"
    await db.commit()
    await db.refresh(document)

    return document


async def set_document_done(
    db: AsyncSession,
    document: Document,
    markdown_content: str,
    ocr_backend: str,
) -> Document:
    document.status = "done"
    document.markdown_content = markdown_content
    document.ocr_backend = ocr_backend
    document.error_message = None
    await db.commit()
    await db.refresh(document)

    return document


async def set_document_failed(db: AsyncSession, document: Document, error_message: str) -> Document:
    document.status = "failed"
    document.error_message = error_message
    await db.commit()
    await db.refresh(document)

    return document
