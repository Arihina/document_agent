from __future__ import annotations

import time
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.db.database import get_db
from app.db import crud

from app.core.llm import stream_answer
from app.core.auth import get_user_id
from app.core.config import settings
from app.api.deps import get_owned_completion, parse_completion_id, parse_file_id
from app.api.generation import collect, persist, resolve_file
from app.api.openai_format import (
    message_item, response_object, responses_usage, sse_event, text_part,
)
from app.api.openai_request import (
    DIALOG_ROLES, collect_instructions, parse_responses_content,
    reject_inline_attachment, sampling_options,
)

router = APIRouter(prefix="/v1/responses", tags=["responses"])

DEFAULT_MODEL = "document_chat"


def _conversation_field(body: dict):
    conversation = body.get("conversation")

    if isinstance(conversation, dict):
        return conversation.get("id")
    if conversation is not None:
        return conversation

    return body.get("conversation_id")


def _extract(body: dict) -> dict:
    model = body.get("model") or DEFAULT_MODEL
    input_data = body.get("input")

    common = {
        "model": model,
        "stream": bool(body.get("stream", False)),
        "store": bool(body.get("store", True)),
        "conversation_raw": _conversation_field(body),
        "previous_response_id": body.get("previous_response_id"),
        "instructions": body.get("instructions"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_output_tokens": body.get("max_output_tokens"),
        "metadata": body.get("metadata") or {},
        "options": sampling_options(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_output_tokens"),
        ),
    }

    if input_data is None:
        raise HTTPException(400, "input обязателен")

    if isinstance(input_data, str):
        question = input_data.strip()
        if not question:
            raise HTTPException(400, "Пустой input")
        return {**common, "history": [], "question": question, "file_id_raw": None}

    if not isinstance(input_data, list) or not input_data:
        raise HTTPException(
            400, "input должен быть строкой или непустым списком items")

    last = input_data[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise HTTPException(
            400, 'последний item в input должен иметь role="user"')

    parsed = parse_responses_content(last.get("content"))
    reject_inline_attachment(parsed)
    if not parsed.text:
        raise HTTPException(400, "Пустой вопрос")

    items = [m for m in input_data[:-1]
             if isinstance(m, dict) and m.get("type", "message") == "message"]

    inline = collect_instructions(items, parse_responses_content)
    if inline:
        common["instructions"] = "\n".join(
            filter(None, [common["instructions"], inline]))

    history = [
        (m["role"], parse_responses_content(m.get("content")).text)
        for m in items if m.get("role") in DIALOG_ROLES
    ]

    return {**common, "history": history, "question": parsed.text,
            "file_id_raw": parsed.file_id}


async def _resolve_conversation(db, user_id, req) -> UUID | None:
    raw = req["conversation_raw"]

    if raw is None and req["previous_response_id"]:
        prev = await crud.get_message_for_user(
            db, parse_completion_id(req["previous_response_id"]), user_id)
        if prev is None:
            raise HTTPException(404, "previous_response_id не найден")
        return prev.conversation_id

    if raw is None:
        return None

    try:
        conversation_id = UUID(str(raw))
    except ValueError:
        raise HTTPException(400, "conversation должен быть UUID")

    if await crud.get_conversation(db, conversation_id, user_id) is None:
        raise HTTPException(404, "Чат (conversation) не найден")

    return conversation_id


@router.post("")
async def create_response(
    body: dict = Body(...),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    req = _extract(body)
    model = req["model"]

    conversation_id = await _resolve_conversation(db, user_id, req)
    history = req["history"]

    if conversation_id is not None:
        if history:
            raise HTTPException(
                400,
                "При переданном conversation input должен содержать только "
                "новый ход (без истории) — история собирается агентом из БД",
            )
        history = [
            (m.role, m.content)
            for m in await crud.get_recent_conversation_messages(
                db, conversation_id, settings.HISTORY_LIMIT)
        ]

    if req["file_id_raw"] is not None:
        document_markdown, active_filename, active_file_id = await resolve_file(
            db, user_id, parse_file_id(req["file_id_raw"]))
    elif conversation_id is not None:
        auto = await crud.get_latest_conversation_file(db, conversation_id, user_id)
        document_markdown = auto.markdown_content if auto else None
        active_filename = auto.filename if auto else None
        active_file_id = auto.id if auto else None
    else:
        document_markdown = active_filename = active_file_id = None

    sources = [active_filename] if active_filename else None
    file_id_out = f"file-{active_file_id}" if active_file_id else None

    assistant_id = uuid4()
    response_id = f"resp_{assistant_id}"
    item_id = f"msg_{assistant_id}"
    created = int(time.time())
    conversation_id_str = str(conversation_id) if conversation_id else None

    if not req["stream"]:
        gen = await run_in_threadpool(
            collect, req["question"], document_markdown, history,
            req["instructions"], req["options"],
        )

        if req["store"]:
            await persist(user_id, req["question"], gen.answer, sources, model,
                          assistant_id, conversation_id, gen.usage)

        return response_object(
            response_id, created, model, conversation_id_str, "completed",
            output=[message_item(item_id, gen.answer, "completed")],
            usage=responses_usage(gen.usage), req=req, file_id=file_id_out,
        )

    state = {"answer": "", "usage": None}

    def _gen():
        seq = 0

        def _next():
            nonlocal seq
            seq += 1
            return seq

        yield sse_event(
            _next(), "response.created",
            response=response_object(
                response_id, created, model, conversation_id_str,
                "in_progress", output=[], req=req, file_id=file_id_out),
        )
        yield sse_event(
            _next(), "response.in_progress",
            response=response_object(
                response_id, created, model, conversation_id_str,
                "in_progress", output=[], req=req, file_id=file_id_out),
        )
        yield sse_event(
            _next(), "response.output_item.added",
            output_index=0, item=message_item(item_id, "", "in_progress"),
        )
        yield sse_event(
            _next(), "response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part=text_part(""),
        )

        for token, usage in stream_answer(
            req["question"], document_markdown, history,
            instructions=req["instructions"], options=req["options"],
        ):
            if token:
                state["answer"] += token
                yield sse_event(
                    _next(), "response.output_text.delta",
                    item_id=item_id, output_index=0, content_index=0,
                    delta=token, logprobs=[],
                )
            if usage is not None:
                state["usage"] = usage

        yield sse_event(
            _next(), "response.output_text.done",
            item_id=item_id, output_index=0, content_index=0,
            text=state["answer"], logprobs=[],
        )
        yield sse_event(
            _next(), "response.content_part.done",
            item_id=item_id, output_index=0, content_index=0,
            part=text_part(state["answer"]),
        )

        final_item = message_item(item_id, state["answer"], "completed")
        yield sse_event(_next(), "response.output_item.done",
                        output_index=0, item=final_item)

        yield sse_event(
            _next(), "response.completed",
            response=response_object(
                response_id, created, model, conversation_id_str,
                "completed", output=[final_item],
                usage=responses_usage(state["usage"]), req=req,
                file_id=file_id_out),
        )

    async def _async_gen():
        seen = 0
        try:
            async for piece in iterate_in_threadpool(_gen()):
                seen += 1
                yield piece
        except Exception as e:
            yield sse_event(seen + 1, "error",
                            message=str(e), code=None, param=None)
            return

        if req["store"]:
            await persist(user_id, req["question"], state["answer"], sources,
                          model, assistant_id, conversation_id, state["usage"])

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{completion_id}")
async def get_response(msg=Depends(get_owned_completion)):
    if msg.role != "assistant":
        raise HTTPException(404, "Response не найден")

    return response_object(
        f"resp_{msg.id}", int(msg.created_at.timestamp()), msg.model,
        str(msg.conversation_id) if msg.conversation_id else None,
        "completed",
        output=[message_item(f"msg_{msg.id}", msg.content, "completed")],
        usage=responses_usage({"prompt_tokens": msg.prompt_tokens,
                               "completion_tokens": msg.completion_tokens}),
    )


@router.delete("/{completion_id}")
async def delete_response(
    msg=Depends(get_owned_completion),
    db: AsyncSession = Depends(get_db),
):
    if msg.role != "assistant":
        raise HTTPException(404, "Response не найден")

    response_id = f"resp_{msg.id}"
    await crud.delete_message(db, msg)

    return {"id": response_id, "object": "response.deleted", "deleted": True}
