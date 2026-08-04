from __future__ import annotations

import json
import time
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.db.database import get_db, AsyncSessionLocal
from app.db import crud

from app.core.llm import stream_answer
from app.core.auth import get_user_id
from app.api.deps import get_owned_completion, parse_file_id

router = APIRouter(prefix="/v1/chat/completions", tags=["chat"])

DEFAULT_MODEL = "document_chat"
TITLE_MAX_LEN = 80


def _parse_content(content) -> tuple[str, str | None]:
    if isinstance(content, str):
        return content.strip(), None
    if isinstance(content, list):
        texts: list[str] = []
        file_id: str | None = None
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                texts.append(str(part.get("text", "")))
            elif ptype == "file":
                file_id = (part.get("file") or {}).get("file_id")
        return "\n".join(texts).strip(), file_id
    return "", None


def _extract(body: dict) -> tuple[str, list[tuple[str, str]], str, str | None, bool, str | None]:
    model = body.get("model") or DEFAULT_MODEL
    messages = body.get("messages")
    stream = body.get("stream", False)
    conversation_id_raw = body.get("conversation_id")

    if not isinstance(messages, list) or not messages:
        raise HTTPException(422, "messages обязателен и не должен быть пустым")

    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise HTTPException(
            422, 'последнее сообщение должно иметь role="user"')

    question, file_id_raw = _parse_content(last.get("content"))
    if not question:
        raise HTTPException(422, "Пустой вопрос")

    history: list[tuple[str, str]] = []
    for m in messages[:-1]:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        text, _ = _parse_content(m.get("content"))
        history.append((m["role"], text))

    return model, history, question, file_id_raw, bool(stream), conversation_id_raw


async def _resolve_file(db: AsyncSession, user_id: UUID, file_id_raw: str | None) -> tuple[str | None, str | None]:
    if file_id_raw is None:
        return None, None
    file_uuid = parse_file_id(file_id_raw)
    f = await crud.get_file_for_user(db, file_uuid, user_id)
    if f is None:
        raise HTTPException(404, "Файл (file_id) не найден")
    if f.status != "done":
        raise HTTPException(
            422, f"Файл ещё не готов к использованию (status={f.status})")
    return f.markdown_content, f.filename


def _collect(question: str, document_markdown: str | None, history: list[tuple[str, str]]):
    full_answer = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for token, u in stream_answer(question, document_markdown, history):
        if token:
            full_answer += token
        if u is not None:
            usage = u
    return full_answer, usage


async def _persist(
    user_id: UUID,
    question: str,
    answer: str,
    sources: list[str] | None,
    model: str,
    assistant_id: UUID,
    conversation_id: UUID | None,
    usage: dict | None,
) -> None:
    usage = usage or {}
    async with AsyncSessionLocal() as write_db:
        await crud.add_message(
            write_db, user_id=user_id, role="user", content=question,
            conversation_id=conversation_id,
        )
        await crud.add_message(
            write_db, user_id=user_id, role="assistant", content=answer,
            sources=sources, model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            id=assistant_id, conversation_id=conversation_id,
        )
        if conversation_id is not None:
            title = question[:TITLE_MAX_LEN] + \
                ("…" if len(question) > TITLE_MAX_LEN else "")
            await crud.touch_conversation(write_db, conversation_id, title=title)


@router.post("")
async def chat_completions(
    body: dict = Body(...),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    model, history, question, file_id_raw, stream, conversation_id_raw = _extract(
        body)

    conversation_id: UUID | None = None
    if conversation_id_raw is not None:
        try:
            conversation_id = UUID(str(conversation_id_raw))
        except ValueError:
            raise HTTPException(422, "conversation_id должен быть UUID")
        if await crud.get_conversation(db, conversation_id, user_id) is None:
            raise HTTPException(404, "Чат (conversation_id) не найден")

    document_markdown, active_filename = await _resolve_file(db, user_id, file_id_raw)
    sources = [active_filename] if active_filename else None

    assistant_id = uuid4()
    completion_id = f"chatcmpl-{assistant_id}"
    created = int(time.time())
    conversation_id_str = str(conversation_id) if conversation_id else None

    if not stream:
        full_answer, usage = await run_in_threadpool(
            _collect, question, document_markdown, history
        )

        await _persist(user_id, question, full_answer, sources, model,
                       assistant_id, conversation_id, usage)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "conversation_id": conversation_id_str,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_answer},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }

    state = {"answer": "", "usage": None}

    def _gen():
        role_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model, "conversation_id": conversation_id_str,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

        for token, usage in stream_answer(question, document_markdown, history):
            if token:
                state["answer"] += token
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model, "conversation_id": conversation_id_str,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            if usage is not None:
                state["usage"] = usage

        final_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model, "conversation_id": conversation_id_str,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

    async def _async_gen():
        async for chunk in iterate_in_threadpool(_gen()):
            yield chunk
        yield "data: [DONE]\n\n"

        await _persist(user_id, question, state["answer"], sources, model,
                       assistant_id, conversation_id, state["usage"])

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{completion_id}")
async def get_completion(msg=Depends(get_owned_completion)):
    if msg.role != "assistant":
        raise HTTPException(404, "Completion не найден")

    prompt_tokens = msg.prompt_tokens or 0
    completion_tokens = msg.completion_tokens or 0

    return {
        "id": f"chatcmpl-{msg.id}",
        "object": "chat.completion",
        "created": int(msg.created_at.timestamp()),
        "model": msg.model,
        "conversation_id": str(msg.conversation_id) if msg.conversation_id else None,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": msg.content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
