from sqlalchemy import (
    Integer, BigInteger, String, Text, LargeBinary, ForeignKey, CheckConstraint,
    TIMESTAMP,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, DeclarativeBase, mapped_column, Mapped

from typing import Optional
from datetime import datetime, timezone
from uuid import UUID as PyUUID, uuid4


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4,
    )
    user_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True,
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="selectin",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4,
    )
    user_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True,
    )
    conversation_id: Mapped[Optional[PyUUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True, index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True)
    tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    conversation: Mapped[Optional["Conversation"]] = relationship(
        back_populates="messages")
    feedback: Mapped[Optional["MessageFeedback"]] = relationship(
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class MessageFeedback(Base):
    __tablename__ = "message_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    vote: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        CheckConstraint("vote IN (-1, 1) OR vote IS NULL",
                        name="ck_vote_values"),
    )

    message: Mapped["ChatMessage"] = relationship(
        back_populates="feedback", lazy="selectin",
    )


class File(Base):
    __tablename__ = "files"

    id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4,
    )
    user_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True,
    )
    conversation_id: Mapped[Optional[PyUUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey(
            "conversations.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(
        String(127), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    content: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False, deferred=True)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending")
    markdown_content: Mapped[Optional[str]
                             ] = mapped_column(Text, nullable=True)
    
    markdown_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True)
    ocr_backend: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'done', 'failed')",
            name="ck_file_status",
        ),
    )
