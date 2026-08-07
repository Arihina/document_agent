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
from app.core.config import settings
from app.api.deps import get_owned_completion, parse_file_id

router = APIRouter(prefix="/v1/responses", tags=["responses"])

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
            if ptype in ("input_text", "output_text"):
                texts.append(str(part.get("text", "")))
            elif ptype == "input_file":
                file_id = part.get("file_id")
        return "\n".join(texts).strip(), file_id
    return "", None


def _extract(body: dict) -> tuple[str, list[dict], str, str | None, bool, str | None]:
    model = body.get("model") or DEFAULT_MODEL
    input_data = body.get("input")
    stream = body.get("stream", False)
    conversation_id_raw = body.get("conversation_id")

    if input_data is None:
        raise HTTPException(422, "input обязателен")

    if isinstance(input_data, str):
        question = input_data.strip()
        if not question:
            raise HTTPException(422, "Пустой input")
        return model, [], question, None, bool(stream), conversation_id_raw

    if not isinstance(input_data, list) or not input_data:
        raise HTTPException(
            422, "input должен быть строкой или непустым списком items")

    last = input_data[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise HTTPException(
            422, 'последний item в input должен иметь role="user"')

    question, file_id_raw = _parse_content(last.get("content"))
    if not question:
        raise HTTPException(422, "Пустой вопрос")

    history: list[dict] = []
    for m in input_data[:-1]:
        if not isinstance(m, dict) or m.get("type", "message") != "message":
            continue
        if m.get("role") not in ("user", "assistant"):
            continue
        text, _ = _parse_content(m.get("content"))
        history.append({"role": m["role"], "content": text})

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


def _collect(question: str, document_markdown: str | None, history: list[dict]):
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


def _usage_out(usage: dict) -> dict:
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _message_item(item_id: str, text: str, status: str) -> dict:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response_object(
    response_id: str, created: int, model: str, conversation_id_str: str | None,
    status: str, output: list[dict], usage: dict | None = None,
) -> dict:
    obj = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "conversation_id": conversation_id_str,
        "output": output,
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


def _sse_event(seq: int, event_type: str, **fields) -> str:
    payload = {"type": event_type, "sequence_number": seq, **fields}
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("")
async def create_response(
    body: dict = Body(...),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    model, req_history, question, file_id_raw, stream, conversation_id_raw = _extract(
        body)

    conversation_id: UUID | None = None
    if conversation_id_raw is not None:
        try:
            conversation_id = UUID(str(conversation_id_raw))
        except ValueError:
            raise HTTPException(422, "conversation_id должен быть UUID")
        if await crud.get_conversation(db, conversation_id, user_id) is None:
            raise HTTPException(404, "Чат (conversation_id) не найден")

        if req_history:
            raise HTTPException(
                422,
                "При переданном conversation_id input должен содержать "
                "только новый ход (без истории) — история собирается "
                "агентом из БД по conversation_id",
            )

    if file_id_raw is not None:
        document_markdown, active_filename = await _resolve_file(db, user_id, file_id_raw)
    elif conversation_id is not None:
        auto_file = await crud.get_latest_conversation_file(db, conversation_id, user_id)
        document_markdown = auto_file.markdown_content if auto_file else None
        active_filename = auto_file.filename if auto_file else None
    else:
        document_markdown, active_filename = None, None
    sources = [active_filename] if active_filename else None

    if conversation_id is not None:
        history = [
            {"role": m.role, "content": m.content}
            for m in await crud.get_recent_conversation_messages(
                db, conversation_id, settings.HISTORY_LIMIT)
        ]
    else:
        history = req_history

    assistant_id = uuid4()
    response_id = f"resp_{assistant_id}"
    item_id = f"msg_{assistant_id}"
    created = int(time.time())
    conversation_id_str = str(conversation_id) if conversation_id else None

    if not stream:
        full_answer, usage = await run_in_threadpool(
            _collect, question, document_markdown, history
        )

        await _persist(user_id, question, full_answer, sources, model,
                       assistant_id, conversation_id, usage)

        return _response_object(
            response_id, created, model, conversation_id_str, "completed",
            output=[_message_item(item_id, full_answer, "completed")],
            usage=_usage_out(usage),
        )

    state = {"answer": "", "usage": None}

    def _gen():
        seq = 0

        def _next():
            nonlocal seq
            seq += 1
            return seq

        yield _sse_event(
            _next(), "response.created",
            response=_response_object(
                response_id, created, model, conversation_id_str,
                "in_progress", output=[]),
        )
        yield _sse_event(
            _next(), "response.output_item.added",
            output_index=0, item=_message_item(item_id, "", "in_progress"),
        )
        yield _sse_event(
            _next(), "response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part={"type": "output_text", "text": "", "annotations": []},
        )

        for token, usage in stream_answer(question, document_markdown, history):
            if token:
                state["answer"] += token
                yield _sse_event(
                    _next(), "response.output_text.delta",
                    item_id=item_id, output_index=0, content_index=0,
                    delta=token,
                )
            if usage is not None:
                state["usage"] = usage

        yield _sse_event(
            _next(), "response.output_text.done",
            item_id=item_id, output_index=0, content_index=0,
            text=state["answer"],
        )
        yield _sse_event(
            _next(), "response.content_part.done",
            item_id=item_id, output_index=0, content_index=0,
            part={"type": "output_text",
                  "text": state["answer"], "annotations": []},
        )

        final_item = _message_item(item_id, state["answer"], "completed")
        yield _sse_event(_next(), "response.output_item.done",
                         output_index=0, item=final_item)

        yield _sse_event(
            _next(), "response.completed",
            response=_response_object(
                response_id, created, model, conversation_id_str,
                "completed", output=[final_item], usage=_usage_out(state["usage"] or {})),
        )

    async def _async_gen():
        try:
            async for chunk in iterate_in_threadpool(_gen()):
                yield chunk
        except Exception as e:
            yield _sse_event(9999, "error", message=str(e), code=None, param=None)
            return

        await _persist(
            user_id, question, state["answer"], sources, model,
            assistant_id, conversation_id, state["usage"],
        )

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{completion_id}")
async def get_response(msg=Depends(get_owned_completion)):
    if msg.role != "assistant":
        raise HTTPException(404, "Response не найден")

    return _response_object(
        f"resp_{msg.id}", int(msg.created_at.timestamp()), msg.model,
        str(msg.conversation_id) if msg.conversation_id else None,
        "completed",
        output=[_message_item(f"msg_{msg.id}", msg.content, "completed")],
        usage=_usage_out({
            "prompt_tokens": msg.prompt_tokens or 0,
            "completion_tokens": msg.completion_tokens or 0,
            "total_tokens": (msg.prompt_tokens or 0) + (msg.completion_tokens or 0),
        }),
    )
