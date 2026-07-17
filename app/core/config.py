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

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
