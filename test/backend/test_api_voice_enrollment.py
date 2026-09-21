"""API-level tests for the voice-samples and voice-enrollment routes
(Appendix A.6's coverage gap: the matching/storage logic underneath
these routes is fully covered by test_voice_enrollment.py, but no test
drove the routes themselves through TestClient - auth wiring, request
validation, and status codes were untested).

ffmpeg conversion runs for real (a tiny but valid WAV round-trips
through it fine, same as test_api_upload_pipeline.py's approach). The
one genuinely heavy step - StreamingTranscriber.embed(), which needs a
loaded model - is monkeypatched, exactly like test_streaming_commands_route.py
does for the same reason.
"""

import io
import struct
from types import SimpleNamespace

import numpy as np
import pytest

import app.api.routes.voice_enrollment as voice_enrollment_route
from app.core.config import settings
from app.services import voice_enrollment
from app.streaming.commands import COMMANDS


COMMAND_ID = COMMANDS[0].id


def _wav_bytes(seconds: float = 0.5) -> bytes:
    sample_rate = 16_000
    n_samples = int(sample_rate * seconds)
    data = b"\x00\x00" * n_samples
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(data),
    )
    return header + data


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch):
    async def fake_embed(audio):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        voice_enrollment_route, "get_streaming_transcriber", lambda: SimpleNamespace(embed=fake_embed)
    )


@pytest.fixture(autouse=True)
def _cleanup_enrollment(student):
    yield
    voice_enrollment.delete_samples(student.user_id, COMMAND_ID, "si")


@pytest.fixture
def _isolated_voice_samples_dir(tmp_path, monkeypatch):
    """/voice-samples writes straight to settings.voice_samples_dir, which
    in dev points at this repo's real training-data directory
    (storage/voice_samples) - never write test clips there, and never let
    a test's DELETE call touch real recorded samples."""

    monkeypatch.setattr(settings, "voice_samples_dir", str(tmp_path))


# --- GET /voice-enrollment ---------------------------------------------


def test_get_status_requires_authentication(client):
    response = client.get("/voice-enrollment")
    assert response.status_code == 401


def test_get_status_lists_every_command_for_the_caller(client, student):
    response = client.get("/voice-enrollment", headers=student.auth)
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "si"
    ids = {item["id"] for item in body["commands"]}
    assert COMMAND_ID in ids


def test_get_status_rejects_an_unknown_language(client, student):
    response = client.get("/voice-enrollment?language=fr", headers=student.auth)
    assert response.status_code == 422


# --- PATCH /voice-enrollment/active-language ----------------------------


def test_set_active_language_updates_the_caller(client, student):
    response = client.patch(
        "/voice-enrollment/active-language", headers=student.auth, json={"language": "en"}
    )
    assert response.status_code == 200
    assert response.json()["activeLanguage"] == "en"

    # restore, so the fixture's teardown/other tests see the default
    client.patch("/voice-enrollment/active-language", headers=student.auth, json={"language": "si"})


def test_set_active_language_rejects_an_invalid_value(client, student):
    response = client.patch(
        "/voice-enrollment/active-language", headers=student.auth, json={"language": "fr"}
    )
    assert response.status_code == 400


# --- POST /voice-enrollment/{command_id}/samples ------------------------


def test_submit_sample_requires_authentication(client):
    response = client.post(
        f"/voice-enrollment/{COMMAND_ID}/samples",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 401


def test_submit_sample_is_accepted_and_counted(client, student):
    response = client.post(
        f"/voice-enrollment/{COMMAND_ID}/samples",
        headers=student.auth,
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["accepted"] is True
    assert body["collected"] == 1


def test_submit_sample_rejects_an_empty_file(client, student):
    response = client.post(
        f"/voice-enrollment/{COMMAND_ID}/samples",
        headers=student.auth,
        files={"file": ("clip.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400


def test_submit_sample_rejects_an_unknown_command_id(client, student):
    response = client.post(
        "/voice-enrollment/not-a-real-command/samples",
        headers=student.auth,
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 404


# --- POST /voice-enrollment/{command_id}/practice ------------------------


def test_practice_requires_authentication(client):
    response = client.post(
        f"/voice-enrollment/{COMMAND_ID}/practice",
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 401


def test_practice_with_no_enrolled_samples_reports_nothing_to_compare(client, student):
    response = client.post(
        f"/voice-enrollment/{COMMAND_ID}/practice",
        headers=student.auth,
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enrolledSampleCount"] == 0
    assert body["passesThreshold"] is False


def test_practice_rejects_an_unknown_command_id(client, student):
    response = client.post(
        "/voice-enrollment/not-a-real-command/practice",
        headers=student.auth,
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 404


# --- DELETE /voice-enrollment/{command_id} -------------------------------


def test_delete_samples_requires_authentication(client):
    response = client.delete(f"/voice-enrollment/{COMMAND_ID}")
    assert response.status_code == 401


def test_delete_samples_resets_progress(client, student):
    client.post(
        f"/voice-enrollment/{COMMAND_ID}/samples",
        headers=student.auth,
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )

    response = client.delete(f"/voice-enrollment/{COMMAND_ID}", headers=student.auth)
    assert response.status_code == 200
    assert response.json()["collected"] == 0


def test_delete_samples_rejects_an_unknown_command_id(client, student):
    response = client.delete("/voice-enrollment/not-a-real-command", headers=student.auth)
    assert response.status_code == 404


# --- /voice-samples (dev data-collection endpoints) ----------------------


def test_voice_samples_list_requires_authentication(client):
    response = client.get("/voice-samples")
    assert response.status_code == 401


def test_voice_samples_list_reports_every_command(client, student, _isolated_voice_samples_dir):
    response = client.get("/voice-samples", headers=student.auth)
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["commands"]}
    assert COMMAND_ID in ids


def test_voice_samples_upload_rejects_an_unknown_command_id(client, student, _isolated_voice_samples_dir):
    response = client.post(
        "/voice-samples/not-a-real-command",
        headers=student.auth,
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert response.status_code == 404


def test_voice_samples_upload_rejects_an_empty_file(client, student, _isolated_voice_samples_dir):
    response = client.post(
        f"/voice-samples/{COMMAND_ID}",
        headers=student.auth,
        files={"file": ("clip.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400


def test_voice_samples_upload_then_delete_round_trips(client, student, _isolated_voice_samples_dir):
    upload = client.post(
        f"/voice-samples/{COMMAND_ID}",
        headers=student.auth,
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    assert upload.status_code == 200, upload.text
    assert upload.json()["counts"][COMMAND_ID] == 1

    delete = client.delete(f"/voice-samples/{COMMAND_ID}", headers=student.auth)
    assert delete.status_code == 200
    assert delete.json()["counts"][COMMAND_ID] == 0


def test_voice_samples_delete_rejects_an_unknown_command_id(client, student, _isolated_voice_samples_dir):
    response = client.delete("/voice-samples/not-a-real-command", headers=student.auth)
    assert response.status_code == 404
