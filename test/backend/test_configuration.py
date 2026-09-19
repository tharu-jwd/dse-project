"""Configuration tests (Master Test Plan §3.1.8) - the backend half.

Configuration testing asks: does the system behave correctly under the
configurations it will actually be run with? Browser/OS matrices need a
real browser and are covered separately in the report. What can be checked
here, and what has actually bitten this project, is the *server-side*
configuration surface: how settings are parsed, what they default to when
the environment says nothing, and what the CORS layer does with them.

Two real incidents motivate this file:
  - Two tests passed locally only because a developer's `.env` set
    VOICE_COMMAND_EMBEDDING_MATCHING_ENABLED=true while the code default is
    false - CI had no `.env`, so behaviour differed between environments.
  - A frontend built without VITE_API_BASE_URL looked healthy while failing
    every request; CORS_ORIGINS must likewise name the real frontend.

`Settings(_env_file=None)` is used throughout so these tests see only the
defaults and the explicit environment they set - never the developer's `.env`.
"""

import pytest
from pydantic import ValidationError

from app.core.config import PROJECT_ROOT, Settings

REQUIRED = {
    "POSTGRES_DB": "db",
    "POSTGRES_USER": "user",
    "POSTGRES_PASSWORD": "password",
    "JWT_SECRET_KEY": "secret",
}


@pytest.fixture
def clean_env(monkeypatch):
    """Removes every setting-shaped variable so a test controls the whole
    environment, then supplies only the required ones."""

    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- Required settings ---------------------------------------------------------


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_startup_fails_loudly_when_a_required_setting_is_missing(clean_env, missing):
    """The app must refuse to start rather than run with, say, no JWT secret."""

    clean_env.delenv(missing)
    with pytest.raises(ValidationError) as error:
        make_settings()
    assert missing.lower() in str(error.value).lower()


# --- Defaults: what happens when the environment says nothing ---------------------


def test_risky_features_default_to_off(clean_env):
    """Live transcription and voice-fingerprint matching are opt-in. This pins
    the defaults so a change is deliberate - and documents the trap: leaving
    these unset yields a server whose microphone features silently do nothing,
    with no error, which is why .env.example sets them explicitly."""

    settings = make_settings()
    assert settings.streaming_enabled is False
    assert settings.voice_command_embedding_matching_enabled is False


def test_transcriber_defaults_to_the_fake_backend(clean_env):
    """A fresh checkout must not try to load a 1GB model just to start."""

    assert make_settings().transcriber_backend == "fake"


def test_environment_variables_override_defaults_case_insensitively(clean_env):
    clean_env.setenv("streaming_enabled", "true")
    clean_env.setenv("VOICE_COMMAND_EMBEDDING_MATCHING_ENABLED", "1")

    settings = make_settings()
    assert settings.streaming_enabled is True
    assert settings.voice_command_embedding_matching_enabled is True


def test_unknown_environment_variables_are_ignored(clean_env):
    """`extra="ignore"` - an unrelated variable in the shell must not crash startup."""

    clean_env.setenv("SOME_UNRELATED_VARIABLE", "whatever")
    make_settings()


def test_numeric_and_boolean_settings_reject_garbage(clean_env):
    clean_env.setenv("POSTGRES_PORT", "not-a-number")
    with pytest.raises(ValidationError):
        make_settings()


# --- CORS origin parsing ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://a.example", ["http://a.example"]),
        ("http://a.example,http://b.example", ["http://a.example", "http://b.example"]),
        (" http://a.example , http://b.example ", ["http://a.example", "http://b.example"]),
        ("http://a.example,,", ["http://a.example"]),
        ("", []),
        (",", []),
    ],
)
def test_cors_origins_are_parsed_tolerantly(clean_env, raw, expected):
    """Whitespace and stray commas are common when editing .env by hand and
    must not produce a bogus origin like '' or ' http://b.example'."""

    assert make_settings(cors_origins=raw).cors_origins_list == expected


# --- Path resolution -----------------------------------------------------------------


@pytest.mark.parametrize(
    "setting, prop",
    [
        ("media_storage_dir", "media_storage_path"),
        ("voice_samples_dir", "voice_samples_path"),
    ],
)
def test_relative_paths_resolve_from_the_repo_root_not_the_cwd(clean_env, tmp_path, monkeypatch, setting, prop):
    """The same relative value must land in the same place whether the process
    is started from the repo root, backend/, or a Docker WORKDIR."""

    monkeypatch.chdir(tmp_path)
    resolved = getattr(make_settings(**{setting: "some/relative/dir"}), prop)
    assert resolved == (PROJECT_ROOT / "some/relative/dir").resolve()
    assert tmp_path not in resolved.parents


def test_absolute_paths_are_respected(clean_env, tmp_path):
    assert make_settings(media_storage_dir=str(tmp_path)).media_storage_path == tmp_path.resolve()


def test_model_paths_resolve_relative_to_the_repo_root(clean_env):
    settings = make_settings(streaming_model_path="models/x-ct2", streaming_source_model="models/x")
    assert settings.streaming_model_ct2_path == str((PROJECT_ROOT / "models/x-ct2").resolve())
    assert settings.streaming_model_source_path == str((PROJECT_ROOT / "models/x").resolve())


def test_a_missing_local_whisper_model_falls_back_to_the_raw_value(clean_env):
    """whisper_model may be a local path OR a hub id like 'small'. If no local
    directory exists it must be passed through untouched, not mangled into a
    nonexistent absolute path."""

    assert make_settings(whisper_model="small").whisper_model_source == "small"


# --- CORS behaviour on the live app -----------------------------------------------------


def _preflight(client, origin):
    return client.options(
        "/auth/login",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


def test_a_configured_origin_is_allowed(client):
    from app.core.config import settings

    origin = settings.cors_origins_list[0]
    response = _preflight(client, origin)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize(
    "origin",
    ["https://evil.example", "http://localhost:9999", "null", "https://sinhaspeech.vercel.app.evil.example"],
)
def test_an_unlisted_origin_is_not_allowed(client, origin):
    """The last case matters: a naive `endswith`/`startswith` check would let a
    lookalike domain through. Browsers only honour a response that names their
    exact origin, so the header must simply be absent."""

    from app.core.config import settings

    if origin in settings.cors_origins_list:
        pytest.skip("origin is legitimately configured in this environment")

    response = _preflight(client, origin)
    assert response.headers.get("access-control-allow-origin") is None, (
        f"origin {origin!r} was allowed by CORS"
    )


def test_the_wildcard_origin_is_never_configured_alongside_credentials(client):
    """`allow_credentials=True` combined with `*` would let any website make
    authenticated requests as the logged-in user. Starlette refuses to send
    that combination, but the configuration should never ask for it."""

    from app.core.config import settings

    assert "*" not in settings.cors_origins_list


# --- The worker's transcriber selection ----------------------------------------------------


def test_transcriber_factory_honours_the_configured_backend(monkeypatch):
    from app.core.config import settings
    from app.transcribers.factory import create_transcriber
    from app.transcribers.fake import FakeTranscriber

    monkeypatch.setattr(settings, "transcriber_backend", "  FAKE ")
    assert isinstance(create_transcriber(), FakeTranscriber)


def test_transcriber_factory_rejects_an_unknown_backend(monkeypatch):
    """A typo in TRANSCRIBER_BACKEND must fail at worker startup with a clear
    message, not silently fall back to something else."""

    from app.core.config import settings
    from app.transcribers.factory import create_transcriber

    monkeypatch.setattr(settings, "transcriber_backend", "wisper")
    with pytest.raises(RuntimeError, match="wisper"):
        create_transcriber()
