from __future__ import annotations

from pathlib import Path

import httpx

from app.core.config import settings


class MinerUError(Exception):
    """MinerU недоступен или вернул ошибку/неожиданный формат ответа."""


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.MINERU_API_URL}/health")
        return resp.status_code == 200
    except httpx.RequestError:
        return False


async def parse_document(
    filename: str,
    content: bytes,
    mime_type: str | None = None,
) -> str:
    files = {"files": (filename, content,
                       mime_type or "application/octet-stream")}
    data = {
        "backend": settings.MINERU_BACKEND,
        "lang_list": settings.MINERU_LANG,
        "parse_method": "auto",
        "formula_enable": "true",
        "table_enable": "true",
        "return_md": "true",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.MINERU_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{settings.MINERU_API_URL}/file_parse",
                files=files,
                data=data,
            )
    except httpx.RequestError as e:
        raise MinerUError(
            f"MinerU недоступен по {settings.MINERU_API_URL}: {e}") from e

    if resp.status_code != 200:
        raise MinerUError(
            f"MinerU вернул {resp.status_code}: {resp.text[:500]}")

    try:
        payload = resp.json()
    except ValueError as e:
        raise MinerUError(f"MinerU вернул не-JSON: {resp.text[:500]}") from e

    results = payload.get("results") or {}
    if not results:
        raise MinerUError(f"MinerU не вернул results: {payload}")

    stem = Path(filename).stem
    entry = results.get(stem) or next(iter(results.values()))

    md_content = entry.get("md_content")
    if md_content is None:
        raise MinerUError(f"В ответе MinerU нет md_content: {entry}")

    return md_content
