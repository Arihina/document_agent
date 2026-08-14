from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from app.models.models import Conversation, ChatMessage, MessageFeedback, File


async def create_conversation(db: AsyncSession, user_id, title: Optional[str] = None) -> Conversation:
    c = Conversation(user_id=user_id, title=title)
    db.add(c)

    await db.commit()
    await db.refresh(c)

    return c


async def get_conversation(db: AsyncSession, conversation_id: UUID, user_id) -> Optional[Conversation]:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )

    return result.scalar_one_or_none()


async def list_conversations(db: AsyncSession, user_id) -> list[Conversation]:
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )

    return result.scalars().all()


async def rename_conversation(db: AsyncSession, c: Conversation, title: str) -> Conversation:
    c.title = title

    await db.commit()
    await db.refresh(c)

    return c


async def delete_conversation(db: AsyncSession, c: Conversation) -> None:
    await db.delete(c)
    await db.commit()


async def get_conversation_messages(db: AsyncSession, conversation_id: UUID) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at)
    )

    return result.scalars().all()


async def get_recent_conversation_messages(
    db: AsyncSession, conversation_id: UUID, limit: int
) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))


async def touch_conversation(db: AsyncSession, conversation_id: UUID, title: Optional[str] = None) -> None:
    c = await db.get(Conversation, conversation_id)
    if c is None:
        return

    c.updated_at = datetime.now(timezone.utc)
    if title is not None and c.title is None:
        c.title = title

    await db.commit()


async def add_message(
    db: AsyncSession,
    user_id,
    role: str,
    content: str,
    sources: Optional[list] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    id: Optional[UUID] = None,
    conversation_id: Optional[UUID] = None,
    tokens: Optional[int] = None,
) -> ChatMessage:
    msg = ChatMessage(
        id=id or uuid4(),
        user_id=user_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens=tokens,
    )
    db.add(msg)

    await db.commit()
    await db.refresh(msg)

    return msg


async def get_message(db: AsyncSession, message_id: UUID) -> Optional[ChatMessage]:
    result = await db.execute(select(ChatMessage).where(ChatMessage.id == message_id))
    return result.scalar_one_or_none()


async def get_message_for_user(db: AsyncSession, message_id: UUID, user_id) -> Optional[ChatMessage]:
    result = await db.execute(
        select(ChatMessage).where(
            ChatMessage.id == message_id,
            ChatMessage.user_id == user_id,
        )
    )

    return result.scalar_one_or_none()


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


async def create_file(
    db: AsyncSession,
    user_id,
    filename: str,
    mime_type: Optional[str],
    size_bytes: int,
    content: bytes,
    conversation_id: Optional[UUID] = None,
) -> File:
    f = File(
        user_id=user_id,
        conversation_id=conversation_id,
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        content=content,
        status="pending",
    )
    db.add(f)

    await db.commit()
    await db.refresh(f)

    return f


async def get_file(db: AsyncSession, file_id: UUID) -> Optional[File]:
    result = await db.execute(select(File).where(File.id == file_id))
    return result.scalar_one_or_none()


async def get_file_for_user(db: AsyncSession, file_id: UUID, user_id) -> Optional[File]:
    result = await db.execute(
        select(File).where(File.id == file_id, File.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def get_latest_conversation_file(
    db: AsyncSession, conversation_id: UUID, user_id
) -> Optional[File]:
    result = await db.execute(
        select(File)
        .where(
            File.conversation_id == conversation_id,
            File.user_id == user_id,
            File.status == "done",
        )
        .order_by(File.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_file_content(db: AsyncSession, file_id: UUID) -> Optional[File]:
    result = await db.execute(
        select(File).options(undefer(File.content)).where(File.id == file_id)
    )

    return result.scalar_one_or_none()


async def list_files(db: AsyncSession, user_id) -> list[File]:
    result = await db.execute(
        select(File)
        .where(File.user_id == user_id)
        .order_by(File.created_at.desc())
    )

    return result.scalars().all()


async def delete_file(db: AsyncSession, f: File) -> None:
    await db.delete(f)
    await db.commit()


async def set_file_processing(db: AsyncSession, f: File) -> File:
    f.status = "processing"

    await db.commit()
    await db.refresh(f)

    return f


async def set_file_done(
    db: AsyncSession, f: File, markdown_content: str, ocr_backend: str,
    markdown_tokens: Optional[int] = None,
) -> File:
    f.status = "done"
    f.markdown_content = markdown_content
    f.markdown_tokens = markdown_tokens
    f.ocr_backend = ocr_backend
    f.error_message = None

    await db.commit()
    await db.refresh(f)

    return f


async def set_file_tokens(db: AsyncSession, f: File, tokens: int) -> File:
    f.markdown_tokens = tokens

    await db.commit()
    await db.refresh(f)

    return f


async def set_file_failed(db: AsyncSession, f: File, error_message: str) -> File:
    f.status = "failed"
    f.error_message = error_message

    await db.commit()
    await db.refresh(f)

    return f


async def delete_message(db: AsyncSession, msg: ChatMessage) -> None:
    await db.delete(msg)
    await db.commit()
