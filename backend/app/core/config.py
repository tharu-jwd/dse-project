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

    # Step-1 data collection only (see scripts/validate_command_embeddings.py) -
    # not part of the real per-student enrollment bank built in a later step.
    voice_samples_dir: str = "storage/voice_samples"

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

    # Source HF checkpoint to convert, and where the converted
    # CTranslate2 model lives once `convert_to_ctranslate2.py` has run.
    streaming_source_model: str = "models/whisper-sinhala1"
    streaming_model_path: str = "models/whisper-sinhala1-ct2"
    streaming_compute_type: str = "int8"

    streaming_window_interval_seconds: float = 2.0
    streaming_max_buffer_seconds: float = 15.0
    streaming_overlap_seconds: float = 1.0
    streaming_vad_silence_ms: int = 500
    streaming_memory_ceiling_seconds: float = 60.0
    # COMMAND mode is event-driven (VAD checked on every incoming chunk,
    # no tick loop) and can afford to react to a much shorter pause than
    # dictation - a command is one short phrase, not a sentence someone
    # might pause mid-thought while composing.
    streaming_command_vad_silence_ms: int = 300

    # Voice commands: fuzzy-matched against a fixed phrase vocabulary in
    # COMMAND streaming mode only. Never applied to NOTE/dictation mode.
    voice_command_hotwords_enabled: bool = True
    voice_command_fuzzy_threshold: float = 80.0
    voice_command_destructive_threshold: float = 90.0
    # A near-exact skeleton match on a short, distinctive command phrase
    # (e.g. saying "next" and it transcribing to precisely "next") is
    # about as certain as this system gets. Above this score, the fuzzy
    # match is trusted outright even if the embedding path disagrees,
    # rather than downgrading to "confirm" - without this, adding more
    # short/similar-sounding commands (MCQ option numbers) made the
    # embedding path noisy enough to routinely override otherwise-exact
    # text matches for "next"/"previous", making them feel broken.
    voice_command_fuzzy_exact_override: float = 95.0
    # avg_logprob is a per-token log probability (typically in [-1, 0] for
    # confident speech); below this, a borderline fuzzy score is rejected.
    voice_command_logprob_floor: float = -0.5

    # Speaker-enrolled embedding matching: a second, sound-based opinion
    # alongside the fuzzy-text path above (app.streaming.embeddings).
    # `best_match` scores on Manhattan-distance-derived similarity, not
    # cosine - re-validation on the same 31 real recordings showed a
    # slightly larger same-vs-different-command separation (Cohen's d
    # 2.47 vs 2.27). This threshold is the max-margin suggestion from
    # that data (see scripts/validate_command_embeddings.py's `--csv`
    # output), not guessed - re-run it and update this if the checkpoint
    # or the recording conditions change materially.
    voice_embedding_similarity_threshold: float = 0.828
    voice_embedding_min_clip_seconds: float = 0.3
    # Stamped onto every enrolled sample so a future checkpoint swap can
    # be detected instead of silently comparing embeddings from two
    # different model spaces. Bump this by hand when the streaming
    # checkpoint is retrained/reconverted.
    voice_embedding_model_version: str = "whisper-sinhala1-ct2"

    # Enrollment: recording the samples that go into the bank above.
    voice_enrollment_samples_required: int = 5
    # Floor for "does this new take sound like this student's previous
    # takes of the same command" - deliberately looser than the runtime
    # match threshold above. Under Manhattan similarity, real same-phrase
    # pairs scored as low as 0.79 purely from natural variation, so a
    # stricter floor here would reject valid samples during enrollment,
    # not just genuinely mis-recorded ones.
    voice_enrollment_min_sample_similarity: float = 0.77

    # Runtime combination of the fuzzy-text and embedding paths
    # (app.streaming.command_resolution). Off by default so it can be
    # A/B tested - fuzzy-only is exactly today's behaviour either way,
    # and this only ever changes anything for a student who is both
    # flagged in and has an enrollment bank.
    voice_command_embedding_matching_enabled: bool = False
    # Higher bar for a destructive command (submit, delete) to count as
    # a *strong* embedding match, mirroring
    # voice_command_destructive_threshold on the fuzzy side. Set from
    # real data: delete/submit's own same-phrase pairs under Manhattan
    # similarity averaged 0.858 - this sits below that average (so
    # genuine repeats of the command still clear it) but comfortably
    # above the general threshold above.
    voice_embedding_destructive_threshold: float = 0.85
    # "Is there SOME weak candidate worth mentioning" floors, used only
    # to distinguish "confirm" (something maybe happened) from "none"
    # (ordinary dictation) when neither signal was strong. Below these,
    # nothing is reported at all - without a floor, ordinary fuzzy
    # string matching would loosely resemble *some* command in almost
    # every sentence and turn every dictated line into a confirmation
    # prompt.
    voice_command_fuzzy_borderline_floor: float = 60.0
    voice_embedding_borderline_floor: float = 0.74

    # A student unsure whether they were heard will often repeat a
    # command; without this, a repeated "next" would advance two
    # questions instead of one. Applies to any executed command
    # (COMMAND mode and NOTE mode's delete/stop alike).
    voice_command_debounce_seconds: float = 2.0

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
    def voice_samples_path(self) -> Path:
        path = Path(self.voice_samples_dir)

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

    @property
    def streaming_model_source_path(self) -> str:
        """Resolve the source HF checkpoint to convert, relative to repo root."""
        path = Path(self.streaming_source_model)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return str(path.resolve())

    @property
    def streaming_model_ct2_path(self) -> str:
        """Resolve the converted CTranslate2 model directory."""
        path = Path(self.streaming_model_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return str(path.resolve())

    model_config = SettingsConfigDict(
        env_file = PROJECT_ROOT / ".env",
        env_file_encoding = "utf-8",
        case_sensitive=False,
        extra="ignore",
    )




settings = Settings()
