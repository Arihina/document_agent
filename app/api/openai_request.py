from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import HTTPException


INSTRUCTION_ROLES = ("system", "developer")
DIALOG_ROLES = ("user", "assistant")


@dataclass
class ParsedContent:
    text: str
    file_ids: list[str] = field(default_factory=list)
    inline_attachment: str | None = None


def parse_chat_content(content) -> ParsedContent:
    if isinstance(content, str):
        return ParsedContent(text=content.strip())

    if not isinstance(content, list):
        return ParsedContent(text="")

    texts: list[str] = []
    file_ids: list[str] = []
    inline: str | None = None

    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            texts.append(str(part.get("text", "")))
        elif ptype == "file":
            file_ref = part.get("file") or {}
            if file_ref.get("file_id"):
                file_ids.append(file_ref["file_id"])
            else:
                inline = inline or "file"
        elif ptype in ("image_url", "input_audio"):
            inline = inline or ptype

    return ParsedContent("\n".join(texts).strip(), _dedupe(file_ids), inline)


def parse_responses_content(content) -> ParsedContent:
    if isinstance(content, str):
        return ParsedContent(text=content.strip())

    if not isinstance(content, list):
        return ParsedContent(text="")

    texts: list[str] = []
    file_ids: list[str] = []
    inline: str | None = None

    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "output_text"):
            texts.append(str(part.get("text", "")))
        elif ptype == "input_file":
            if part.get("file_id"):
                file_ids.append(part["file_id"])
            else:
                inline = inline or "input_file"
        elif ptype in ("input_image", "input_audio"):
            inline = inline or ptype

    return ParsedContent("\n".join(texts).strip(), _dedupe(file_ids), inline)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def collect_file_ids(items: list, parse) -> list[str]:
    """Sticky файл: рабочий набор задаёт последнее сообщение с вложениями.

    Накопления по всей истории нет — иначе набор растёт бесконечно и упирается
    в MAX_ATTACHED_FILES на ровном месте. Приложил новый файл — набор сменился.
    """
    for m in reversed(items):
        if not isinstance(m, dict):
            continue
        if m.get("type", "message") != "message":
            continue
        if m.get("role") not in DIALOG_ROLES:
            continue
        parsed = parse(m.get("content"))
        if parsed.file_ids:
            return parsed.file_ids

    return []


def reject_inline_attachment(parsed: ParsedContent) -> None:
    if parsed.inline_attachment is None:
        return

    raise HTTPException(
        400,
        f"Вложение типа '{parsed.inline_attachment}' не поддерживается напрямую: "
        "загрузите документ через POST /v1/files и передайте полученный "
        "file_id в content",
    )


def collect_instructions(items: list, parse) -> str | None:
    texts = [
        parse(m.get("content")).text
        for m in items
        if isinstance(m, dict) and m.get("role") in INSTRUCTION_ROLES
    ]
    joined = "\n".join(t for t in texts if t.strip())
    return joined or None


def sampling_options(
    temperature=None, top_p=None, max_tokens=None, num_ctx=None,
) -> dict:
    options: dict = {}

    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    if temperature is not None:
        options["temperature"] = _number("temperature", temperature)
    if top_p is not None:
        options["top_p"] = _number("top_p", top_p)
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise HTTPException(
                400, "Ограничение на длину ответа должно быть целым числом >= 1")
        options["num_predict"] = max_tokens

    return options


def _number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(400, f"{name} должен быть числом")
    return float(value)
