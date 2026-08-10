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
from app.api.deps import get_owned_completion, parse_file_id
from app.api.generation import collect, persist, resolve_file
from app.api.openai_format import chat_usage, chunk, completion_object
from app.api.openai_request import (
    DIALOG_ROLES, collect_instructions, parse_chat_content,
    reject_inline_attachment, sampling_options,
)

router = APIRouter(prefix="/v1/chat/completions", tags=["chat"])

DEFAULT_MODEL = "document_chat"


def _extract(body: dict) -> dict:
    model = body.get("model") or DEFAULT_MODEL
    messages = body.get("messages")

    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages обязателен и не должен быть пустым")

    n = body.get("n")
    if n is not None and n != 1:
        raise HTTPException(
            400, "Поддерживается только n=1: сервис возвращает один вариант ответа")

    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise HTTPException(
            400, 'последнее сообщение должно иметь role="user"')

    parsed = parse_chat_content(last.get("content"))
    reject_inline_attachment(parsed)
    if not parsed.text:
        raise HTTPException(400, "Пустой вопрос")

    history: list[tuple[str, str]] = []
    for m in messages[:-1]:
        if not isinstance(m, dict) or m.get("role") not in DIALOG_ROLES:
            continue
        history.append((m["role"], parse_chat_content(m.get("content")).text))

    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise HTTPException(400, "stream_options должен быть объектом")

    return {
        "model": model,
        "history": history,
        "question": parsed.text,
        "file_id_raw": parsed.file_id,
        "instructions": collect_instructions(messages, parse_chat_content),
        "stream": bool(body.get("stream", False)),
        "include_usage": bool(stream_options.get("include_usage", False)),
        "store": bool(body.get("store", True)),
        "conversation_id_raw": body.get("conversation_id"),
        "options": sampling_options(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_completion_tokens",
                                body.get("max_tokens")),
        ),
    }


@router.post("")
async def chat_completions(
    body: dict = Body(...),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    req = _extract(body)
    model = req["model"]

    conversation_id: UUID | None = None
    if req["conversation_id_raw"] is not None:
        try:
            conversation_id = UUID(str(req["conversation_id_raw"]))
        except ValueError:
            raise HTTPException(400, "conversation_id должен быть UUID")
        if await crud.get_conversation(db, conversation_id, user_id) is None:
            raise HTTPException(404, "Чат (conversation_id) не найден")

    file_uuid = parse_file_id(
        req["file_id_raw"]) if req["file_id_raw"] else None
    document_markdown, active_filename, _ = await resolve_file(db, user_id, file_uuid)
    sources = [active_filename] if active_filename else None

    assistant_id = uuid4()
    completion_id = f"chatcmpl-{assistant_id}"
    created = int(time.time())
    conversation_id_str = str(conversation_id) if conversation_id else None

    if not req["stream"]:
        gen = await run_in_threadpool(
            collect, req["question"], document_markdown, req["history"],
            req["instructions"], req["options"],
        )

        if req["store"]:
            await persist(user_id, req["question"], gen.answer, sources, model,
                          assistant_id, conversation_id, gen.usage)

        return completion_object(completion_id, created, model,
                                 conversation_id_str, gen.answer,
                                 chat_usage(gen.usage))

    state = {"answer": "", "usage": None}

    def _gen():
        yield chunk(completion_id, created, model, conversation_id_str,
                    {"role": "assistant", "content": ""})

        for token, usage in stream_answer(
            req["question"], document_markdown, req["history"],
            instructions=req["instructions"], options=req["options"],
        ):
            if token:
                state["answer"] += token
                yield chunk(completion_id, created, model,
                            conversation_id_str, {"content": token})
            if usage is not None:
                state["usage"] = usage

        yield chunk(completion_id, created, model, conversation_id_str,
                    {}, finish_reason="stop")

        if req["include_usage"]:
            yield chunk(completion_id, created, model, conversation_id_str,
                        None, usage=chat_usage(state["usage"]))

    async def _async_gen():
        async for piece in iterate_in_threadpool(_gen()):
            yield piece
        yield "data: [DONE]\n\n"

        if req["store"]:
            await persist(user_id, req["question"], state["answer"], sources,
                          model, assistant_id, conversation_id, state["usage"])

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{completion_id}")
async def get_completion(msg=Depends(get_owned_completion)):
    if msg.role != "assistant":
        raise HTTPException(404, "Completion не найден")

    return completion_object(
        f"chatcmpl-{msg.id}", int(msg.created_at.timestamp()), msg.model,
        str(msg.conversation_id) if msg.conversation_id else None,
        msg.content,
        chat_usage({"prompt_tokens": msg.prompt_tokens,
                    "completion_tokens": msg.completion_tokens}),
    )


@router.delete("/{completion_id}")
async def delete_completion(
    msg=Depends(get_owned_completion),
    db: AsyncSession = Depends(get_db),
):
    if msg.role != "assistant":
        raise HTTPException(404, "Completion не найден")

    completion_id = f"chatcmpl-{msg.id}"
    await crud.delete_message(db, msg)

    return {"id": completion_id, "object": "chat.completion.deleted", "deleted": True}
