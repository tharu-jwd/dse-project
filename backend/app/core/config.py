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

    cors_origins: str = "http://localhost:5173"

    media_storage_dir: str = "storage/uploads"
    max_upload_size_bytes: int = 100 * 1024 * 1024

    transcriber_backend: str = "fake"

    # Complete Hugging Face Whisper checkpoint, local path or Hub model ID.
    whisper_model: str = "models/whisper-sinhala"
    whisper_language: str = "si"

    # used if .env is configured to whiper small + sinhala lora
    whisper_base_model: str = "openai/whisper-small"
    whisper_adapter_model: str = "SPEAK-ASR/whisper-si-exp-10"

    # Live streaming transcription (parallel path, off by default).
    streaming_enabled: bool = False
    streaming_max_sessions_per_user: int = 3


    @property
    def cors_origins_list(self) -> list[str]:
        return[
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def media_storage_path(self) -> Path:
        path = Path(self.media_storage_dir)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    @property
    def whisper_model_source(self) -> str:
        """Resolve local model paths from the repository root."""
        path = Path(self.whisper_model)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if path.exists():
            return str(path.resolve())

        return self.whisper_model

    model_config = SettingsConfigDict(
        env_file = PROJECT_ROOT / ".env",
        env_file_encoding = "utf-8",
        case_sensitive=False,
        extra="ignore",
    )




settings = Settings()
