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

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
