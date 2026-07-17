import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.feedback import router as feedback_router
from app.core import mineru
from app.core.config import settings

logger = logging.getLogger("document_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # схему ведёт Alembic: alembic upgrade head перед стартом
    if not await mineru.is_available():
        logger.warning(
            "MinerU недоступен по %s на старте — сервис поднимется, но "
            "обработка документов будет падать (502), пока MinerU не появится.",
            settings.MINERU_API_URL,
        )
    yield


app = FastAPI(
    title="Document OCR+LLM Agent API",
    version="0.1.0",
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


app.include_router(chat_router, tags=["chat"])
app.include_router(feedback_router, tags=["feedback"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8006,
        reload=True,
        timeout_keep_alive=300,
    )
