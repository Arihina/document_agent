import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.chat import router as chat_router
from app.api.responses import router as responses_router
from app.api.feedback import router as feedback_router
from app.api.conversations import router as conversations_router
from app.api.files import router as files_router
from app.core import mineru
from app.core.config import settings

logger = logging.getLogger("document_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not await mineru.is_available():
        logger.warning(
            "MinerU недоступен по %s на старте — сервис поднимется, но "
            "обработка документов будет падать (502), пока MinerU не появится.",
            settings.MINERU_API_URL,
        )
    yield


app = FastAPI(
    title="Document OCR+LLM Agent API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)
app.include_router(responses_router)
app.include_router(feedback_router)
app.include_router(conversations_router)
app.include_router(files_router)


_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "not_found_error",
    413: "invalid_request_error",
    415: "invalid_request_error",
    422: "invalid_request_error",
}


_VALIDATION_STATUS = 400

_LOC_PREFIXES = {"body", "query", "path", "header", "cookie"}


def _param_from_loc(loc) -> str | None:
    """('body', 'messages', 0, 'role') -> 'messages.0.role'"""
    parts = [str(p) for p in loc if p not in _LOC_PREFIXES]
    return ".".join(parts) or None


def _error_body(status_code: int, message: str, param: str | None = None) -> dict:
    return {"error": {
        "message": message,
        "type": _ERROR_TYPES.get(status_code, "server_error"),
        "param": param,
        "code": None,
    }}


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.status_code, str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    message = first.get("msg", "Некорректный запрос")
    param = _param_from_loc(first.get("loc", ()))

    return JSONResponse(
        status_code=_VALIDATION_STATUS,
        content=_error_body(_VALIDATION_STATUS, message, param),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8006,
        reload=True,
        timeout_keep_alive=300,
    )
