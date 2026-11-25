"""Контрактные тесты внутренней аутентификации master -> агент (этап 1).

Тесты построены на unittest (как в master_node) и не требуют БД: проверяют
только middleware, request.state и формат ошибок. Реальный секрет агента
сюда не кладём — используется тестовый токен.
"""

from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from app.core.auth import get_user_id
from app.core.internal_auth import install_internal_auth, validate_startup

TOKEN = "a" * 64
OTHER_TOKEN = "b" * 64
USER_ID = "0f5b1c9e-5a1e-4b1e-9c1e-111111111111"


@dataclass
class FakeSettings:
    INTERNAL_AUTH_MODE: str = "token"
    MASTER_INTERNAL_TOKEN: str | None = TOKEN
    PUBLIC_PATHS: str = "/health"
    AGENT_ID: str = "document_chat"
    ENV: str = "dev"
    MASTER_ASSERTION_PUBLIC_KEY: str | None = None

    @property
    def public_paths(self) -> tuple[str, ...]:
        return tuple(
            p.strip() for p in self.PUBLIC_PATHS.split(",") if p.strip())


def make_app(settings: FakeSettings) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/echo")
    async def echo(request: Request):
        return {"user_id": str(request.state.user_id)}

    @app.get("/v1/dep")
    async def dep(user_id=Depends(get_user_id)):
        return {"user_id": str(user_id)}

    install_internal_auth(app, settings)
    return app


def authed(token: str = TOKEN, user_id: str = USER_ID) -> dict:
    return {"X-Internal-Token": token, "X-User-Id": user_id}


class InternalAuthMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(make_app(FakeSettings()))

    def tearDown(self):
        self.client.close()

    def test_health_without_token_is_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_no_token_is_401(self):
        r = self.client.get("/v1/echo")
        self.assertEqual(r.status_code, 401)
        body = r.json()["error"]
        self.assertEqual(body["type"], "authentication_error")
        self.assertIn("токен", body["message"])

    def test_wrong_token_is_401(self):
        r = self.client.get("/v1/echo", headers=authed(token=OTHER_TOKEN))
        self.assertEqual(r.status_code, 401)

    def test_valid_token_and_user_is_200(self):
        r = self.client.get("/v1/echo", headers=authed())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["user_id"], USER_ID)

    def test_missing_user_id_is_401(self):
        r = self.client.get("/v1/echo", headers={"X-Internal-Token": TOKEN})
        self.assertEqual(r.status_code, 401)

    def test_bad_user_id_is_401(self):
        r = self.client.get(
            "/v1/echo", headers=authed(user_id="not-a-uuid"))
        self.assertEqual(r.status_code, 401)

    def test_public_path_is_exact_not_startswith(self):
        # /health публичен, но /healthz и будущие /health/* — нет.
        r = self.client.get("/healthz")
        self.assertEqual(r.status_code, 401)

    def test_error_format(self):
        r = self.client.get("/v1/echo")
        body = r.json()
        self.assertEqual(set(body.keys()), {"error"})
        self.assertEqual(
            set(body["error"].keys()), {"message", "type", "param", "code"})
        self.assertIsNone(body["error"]["param"])
        self.assertIsNone(body["error"]["code"])

    def test_request_id_roundtrip(self):
        r = self.client.get(
            "/v1/echo", headers={**authed(), "X-Request-Id": "req-42"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("x-request-id"), "req-42")

    def test_request_id_generated_when_missing(self):
        r = self.client.get("/health")
        self.assertRegex(r.headers.get("x-request-id"), r"^[0-9a-f]{32}$")

    def test_get_user_id_dependency_reads_state(self):
        r = self.client.get("/v1/dep", headers=authed())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["user_id"], USER_ID)

    def test_token_not_leaked_in_response(self):
        r = self.client.get("/v1/echo", headers=authed(token=OTHER_TOKEN))
        self.assertNotIn(OTHER_TOKEN, r.text)

    def test_off_mode_trusts_header_for_dev(self):
        client = TestClient(make_app(
            FakeSettings(INTERNAL_AUTH_MODE="off")))
        try:
            r = client.get("/v1/echo", headers={"X-User-Id": USER_ID})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["user_id"], USER_ID)
        finally:
            client.close()


class ValidateStartupTests(unittest.TestCase):
    def test_token_mode_with_short_token_fails_closed(self):
        with self.assertRaises(RuntimeError):
            validate_startup(FakeSettings(MASTER_INTERNAL_TOKEN="short"))

    def test_token_mode_with_missing_token_fails_closed(self):
        with self.assertRaises(RuntimeError):
            validate_startup(FakeSettings(MASTER_INTERNAL_TOKEN=None))

    def test_off_in_prod_fails_closed(self):
        with self.assertRaises(RuntimeError):
            validate_startup(FakeSettings(INTERNAL_AUTH_MODE="off", ENV="prod"))

    def test_assertion_not_implemented_fails_closed(self):
        with self.assertRaises(RuntimeError):
            validate_startup(FakeSettings(INTERNAL_AUTH_MODE="assertion"))

    def test_token_mode_valid_passes(self):
        validate_startup(FakeSettings())  # не бросает

    def test_off_in_dev_passes(self):
        validate_startup(FakeSettings(INTERNAL_AUTH_MODE="off"))


class LogRedactionTests(unittest.TestCase):
    def test_token_not_written_to_logs(self):
        records: list[str] = []

        class Sink(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        logger = logging.getLogger("document_agent.internal_auth")
        sink = Sink()
        logger.addHandler(sink)
        try:
            client = TestClient(make_app(FakeSettings()))
            try:
                client.get("/v1/echo", headers=authed(token=OTHER_TOKEN))
            finally:
                client.close()
        finally:
            logger.removeHandler(sink)

        joined = "\n".join(records)
        self.assertNotIn(OTHER_TOKEN, joined)
        self.assertNotIn(TOKEN, joined)


if __name__ == "__main__":
    unittest.main()
