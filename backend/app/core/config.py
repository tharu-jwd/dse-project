from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    postgres_db: str
    postgres_user: str
    postgres_password: str
    postgres_port: int = 5432
    postgres_host: str = "127.0.0.1"

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    cors_origin: str = "http://localhost:5173"

    media_storage_dir: str = "storage/uploads"
    max_upload_size_bytes: int = 100 * 1024 * 1024

    transcriber_backend: str = "fake"

    @property
    def cors_origins_list(self) -> list[str]:
        return[
            origin.strip()
            for origin in self.cors_origin.split(",")
            if origin.strip()
        ]

    @property
    def media_storage_path(self) -> Path:
        path = Path(self.media_storage_dir)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    model_config = SettingsConfigDict(
        env_file = PROJECT_ROOT / ".env",
        env_file_encoding = "utf-8",
        case_sensitive=False,
        extra="ignore",
    )




settings = Settings()
