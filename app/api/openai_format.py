from __future__ import annotations

import json


def completion_object(
    completion_id: str, created: int, model: str, conversation_id_str: str | None,
    content: str, usage: dict,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "conversation_id": conversation_id_str,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "refusal": None,
                "annotations": [],
            },
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": usage,
    }


def chat_usage(usage: dict | None) -> dict:
    usage = usage or {}
    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": usage.get("total_tokens", pt + ct),
    }


def chunk(
    completion_id: str, created: int, model: str, conversation_id_str: str | None,
    delta: dict | None, finish_reason: str | None = None, usage: dict | None = None,
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "conversation_id": conversation_id_str,
        "choices": [] if delta is None else [{
            "index": 0,
            "delta": delta,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        payload["usage"] = usage

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def responses_usage(usage: dict | None) -> dict:
    """input_tokens_details / output_tokens_details обязательны в объекте
    usage — без них SDK не разбирает ответ."""
    usage = usage or {}
    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    return {
        "input_tokens": pt,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": ct,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.get("total_tokens", pt + ct),
    }


def text_part(text: str) -> dict:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def message_item(item_id: str, text: str, status: str) -> dict:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [text_part(text)],
    }


def response_object(
    response_id: str, created: int, model: str, conversation_id_str: str | None,
    status: str, output: list[dict], usage: dict | None = None,
    req: dict | None = None, error: dict | None = None,
    file_id: str | None = None,
) -> dict:
    req = req or {}

    obj = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "error": error,
        "incomplete_details": None,
        "instructions": req.get("instructions"),
        "metadata": req.get("metadata") or {},
        "temperature": req.get("temperature"),
        "top_p": req.get("top_p"),
        "max_output_tokens": req.get("max_output_tokens"),
        "previous_response_id": req.get("previous_response_id"),
        "store": req.get("store", True),
        "truncation": "disabled",
        "text": {"format": {"type": "text"}},
        # Расширения платформы
        "conversation_id": conversation_id_str,
        "file_id": file_id,
    }
    if usage is not None:
        obj["usage"] = usage

    return obj


def sse_event(seq: int, event_type: str, **fields) -> str:
    payload = {"type": event_type, "sequence_number": seq, **fields}
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


_FILE_STATUS_MAP = {
    "pending": "uploaded",
    "processing": "uploaded",
    "done": "processed",
    "failed": "error",
}


def file_status(internal_status: str) -> str:
    return _FILE_STATUS_MAP.get(internal_status, "error")


def file_object(f) -> dict:
    return {
        "id": f"file-{f.id}",
        "object": "file",
        "bytes": f.size_bytes,
        "created_at": int(f.created_at.timestamp()),
        "filename": f.filename,
        "purpose": "assistants",
        "status": file_status(f.status),
        "status_details": f.error_message,
        # Расширения платформы.
        "processing_status": f.status,
        "conversation_id": str(f.conversation_id) if f.conversation_id else None,
    }
