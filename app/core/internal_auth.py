"""Внутренняя аутентификация master -> агент.

Это НЕ пользовательская аутентификация: здесь агент убеждается, что запрос
пришёл именно от master, а не от кого-то, кто просто подставил `X-User-Id`.
Пользователя проверяет master по JWT Keycloak; агенту он передаёт уже
проверенный `X-User-Id` (UUID) вместе с per-agent секретом `X-Internal-Token`.

Сознательно реализовано как чистый ASGI-middleware, а не BaseHTTPMiddleware:
BaseHTTPMiddleware оборачивает ответ в отдельную задачу и ломает SSE-стриминг
(/v1/chat/completions, /v1/responses). Здесь мы только читаем заголовок,
подменяем `http.response.start` (для X-Request-Id) и при ошибке возвращаем
401 в формате платформы — тело ответа не буферизуется.

"""

from __future__ import annotations

import hmac
import json
import logging
import re
import time
import uuid
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("document_agent.internal_auth")

INTERNAL_TOKEN_HEADER = "X-Internal-Token"
USER_ID_HEADER = "X-User-Id"
REQUEST_ID_HEADER = "X-Request-Id"
ASSERTION_HEADER = "X-Master-Assertion"


MIN_TOKEN_LENGTH = 32

# X-Request-Id принимается только в этом виде — защита логов от injection.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def new_request_id() -> str:
    return uuid.uuid4().hex


def normalize_request_id(value: str | None) -> str:
    """Принять валидный X-Request-Id или сгенерировать новый.

    Некорректный id — не повод отдавать 400: это диагностический заголовок,
    а не часть аутентификации.
    """
    if value and _REQUEST_ID_RE.match(value):
        return value
    return new_request_id()


def validate_startup(settings: Any) -> None:
    """Fail-closed проверка конфигурации внутренней аутентификации.

    Вызывается из lifespan ДО старта приёма запросов:

    * `token` + пустой/короткий MASTER_INTERNAL_TOKEN -> RuntimeError;
    * `assertion` -> RuntimeError (этап 2 не реализован);
    * `off` + ENV=prod -> RuntimeError; `off` в dev -> предупреждение;
    * `off` допустим только для локальной разработки/тестов.
    """
    mode = settings.INTERNAL_AUTH_MODE
    prod = getattr(settings, "ENV", "dev") == "prod"

    if mode == "off":
        if prod:
            raise RuntimeError(
                "INTERNAL_AUTH_MODE=off запрещён при ENV=prod: агент "
                "принимал бы запросы без проверки токена master"
            )
        logger.warning(
            "INTERNAL_AUTH_MODE=off: внутренняя аутентификация выключена — "
            "только для локальной разработки и тестов"
        )
        return

    if mode == "token":
        token = settings.MASTER_INTERNAL_TOKEN or ""
        if len(token) < MIN_TOKEN_LENGTH:
            raise RuntimeError(
                "Внутренняя аутентификация не настроена: "
                "INTERNAL_AUTH_MODE=token, но MASTER_INTERNAL_TOKEN пуст или "
                "короче 32 символов. Задайте токен из AGENT_INTERNAL_TOKENS "
                "master (ключ 'document_chat')."
            )
        return

    if mode == "assertion":
        raise RuntimeError(
            "INTERNAL_AUTH_MODE=assertion ещё не реализован в этом агенте "
            "(этап 2). Используйте 'token'."
        )

    raise RuntimeError(f"Неизвестный INTERNAL_AUTH_MODE={mode!r}")


def _safe_log(value: object, limit: int = 512) -> str:
    """Убрать управляющие символы из значения перед записью в лог."""
    text = str(value)[:limit]
    return "".join(ch if ch.isprintable() else "?" for ch in text)


class _AuthError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message


class InternalAuthMiddleware:
    """request_id + проверка X-Internal-Token / валидация X-User-Id."""

    def __init__(self, app: ASGIApp, settings: Any) -> None:
        self.app = app
        self.settings = settings
        self.public_paths = frozenset(
            p for p in getattr(settings, "public_paths", ()) if p
        )

    async def __call__(self, scope: Scope, receive: Receive,
                        send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming_id = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = normalize_request_id(incoming_id)

        # request.state читает scope["state"]; middleware выполняется до Router,
        # поэтому словарь создаём сами.
        state = scope.setdefault("state", {})
        state["request_id"] = request_id

        started = time.perf_counter()
        status = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        path = scope.get("path", "")

        try:
            if path in self.public_paths:
                await self.app(scope, receive, send_with_request_id)
                return

            user_id = self._authenticate(scope)
            state["user_id"] = user_id
            await self.app(scope, receive, send_with_request_id)
        except _AuthError as exc:
            status = exc.status
            await self._send_error(send, request_id, exc.status, exc.message)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            uid = state.get("user_id", "-")
            logger.info(
                "request_id=%s agent_id=%s method=%s path=%s status=%d "
                "user_id=%s duration_ms=%.1f",
                request_id,
                getattr(self.settings, "AGENT_ID", ""),
                _safe_log(scope.get("method", "")),
                _safe_log(path),
                status,
                str(uid) if uid != "-" else "-",
                duration_ms,
            )

    def _authenticate(self, scope: Scope) -> uuid.UUID:
        headers = Headers(scope=scope)
        mode = self.settings.INTERNAL_AUTH_MODE

        if mode == "token":
            received = headers.get(INTERNAL_TOKEN_HEADER)
            expected = self.settings.MASTER_INTERNAL_TOKEN or ""
            # hmac.compare_digest — константное время, без раннего выхода по
            # длине; не логируем предъявленное значение.
            if not received or not hmac.compare_digest(
                received.encode("utf-8"), expected.encode("utf-8")
            ):
                raise _AuthError(401, "Требуется внутренний токен master")
        elif mode == "off":
            # Только локальная разработка/тесты: токен не проверяем.
            pass
        else:  # assertion — блокируется ещё на validate_startup
            raise _AuthError(500, "INTERNAL_AUTH_MODE=assertion не реализован")

        raw = headers.get(USER_ID_HEADER)
        try:
            return uuid.UUID(raw)
        except (ValueError, TypeError):
            raise _AuthError(401, "Некорректный идентификатор пользователя")

    async def _send_error(self, send: Send, request_id: str, status: int,
                          message: str) -> None:
        body = json.dumps({
            "error": {
                "message": message,
                "type": "authentication_error",
                "param": None,
                "code": None,
            }
        }, ensure_ascii=False).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (REQUEST_ID_HEADER.lower().encode(), request_id.encode()),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def install_internal_auth(app: ASGIApp, settings: Any) -> None:
    """Подключить middleware внутренней аутентификации к приложению.

    Вызывать ДО включения CORS: CORSMiddleware должен остаться снаружи,
    чтобы OPTIONS-preflight не отсекался проверкой токена.
    """
    app.add_middleware(InternalAuthMiddleware, settings=settings)
