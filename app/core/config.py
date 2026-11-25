from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    OLLAMA_HOST: str
    OLLAMA_MODEL: str

    MINERU_API_URL: str
    MINERU_BACKEND: str = "pipeline"
    MINERU_LANG: str = "cyrillic"
    MINERU_TIMEOUT_SECONDS: int = 600

    CONTEXT_WINDOW: int = 4096
    RESERVE_OUTPUT_TOKENS: int = 512
    CONTEXT_SAFETY_TOKENS: int = 96
    HISTORY_MIN_TOKENS: int = 384
    HISTORY_MAX_MESSAGES: int = 200
    TOKENIZER_REPO: str | None = None

    MAX_ATTACHED_FILES: int = 1
    DOCUMENT_OVERFLOW: str = "truncate"

    # --- Внутренняя аутентификация master -> агент ---
    # off | token | assertion. token (проверка X-Internal-Token);
    # assertion в этом агенте пока не реализован (старт с ним
    # завершится ошибкой — fail-closed).
    INTERNAL_AUTH_MODE: Literal["off", "token", "assertion"] = "token"
    # Собственный секрет агента. Выдаёт master в AGENT_INTERNAL_TOKENS под
    # ключом "document_chat". Только env/secret-файл.
    MASTER_INTERNAL_TOKEN: str | None = None
    # Пути без внутренней аутентификации. Только liveness, без данных.
    # Список через запятую; сравнение — точное, НЕ startswith.
    PUBLIC_PATHS: str = "/health"
    # Идентификатор агента: для aud в assertion и для логов.
    AGENT_ID: str = "document_chat"
    # dev | prod. В prod запрещены off и открытая документация (/docs, /openapi).
    ENV: Literal["dev", "prod"] = "dev"
    # (не используется, пока INTERNAL_AUTH_MODE != assertion).
    MASTER_ASSERTION_PUBLIC_KEY: str | None = None

    @property
    def public_paths(self) -> tuple[str, ...]:
        return tuple(
            p.strip() for p in self.PUBLIC_PATHS.split(",") if p.strip()
        )

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
