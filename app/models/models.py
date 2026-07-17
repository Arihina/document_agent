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


class ChatSession(Base):
    __tablename__ = "chat_sessions"

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
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="selectin",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4,
    )
    session_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
    feedback: Mapped[Optional["MessageFeedback"]] = relationship(
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    document: Mapped[Optional["Document"]] = relationship(
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


class Document(Base):
    """
    Загруженный пользователем файл + результат его разбора MinerU.
    Само содержимое файла хранится в БД (BYTEA).

    session_id денормализован относительно message_id (источник истины —
    message_id) — добавлен ради дешёвых запросов "все документы сессии"
    без join через chat_messages.

    Удаление сессии/сообщения удаляет документ (и его содержимое) сразу
    двумя независимыми путями: ORM cascade="all, delete-orphan" на
    ChatMessage.document + ON DELETE CASCADE на обоих FK на уровне БД —
    второе подстраховывает на случай удаления в обход ORM.
    """
    __tablename__ = "documents"

    id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4,
    )
    session_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    message_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )

    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(
        String(127), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    content: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False, deferred=True)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending")
    markdown_content: Mapped[Optional[str]
                             ] = mapped_column(Text, nullable=True)
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
            name="ck_document_status",
        ),
    )

    session: Mapped["ChatSession"] = relationship()
    message: Mapped["ChatMessage"] = relationship(back_populates="document")
