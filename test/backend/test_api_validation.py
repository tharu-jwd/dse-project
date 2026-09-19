"""Systematic input validation across the API (Master Test Plan §3.1.2).

The property under test is narrow but important: **no malformed request may
produce a 5xx**. A 422/400/404 is the API telling the caller what was wrong;
a 500 is an unhandled exception and, in production, a stack trace in the logs
and a confusing failure for the user. Earlier coverage checked this for a
couple of hand-picked cases - this sweeps every write endpoint with a set of
deliberately bad bodies, plus malformed path ids.
"""

from uuid import uuid4

import pytest

# Bodies that are wrong in different ways: wrong types, missing fields,
# oversized values, injection-shaped strings, nesting where a scalar belongs.
JUNK_BODIES = [
    {},
    {"unexpected": "field only"},
    {"title": None, "questions": None},
    {"title": 12345, "questions": "not-a-list"},
    {"title": "x" * 5000, "questions": []},
    {"title": "'; DROP TABLE users; --", "questions": []},
    {"title": {"nested": {"object": True}}, "questions": [[]]},
    {"email": ["a"], "password": {"b": 1}},
    {"mark": "high", "feedback": 5},
    {"answers": "nope"},
    {"answers": [{"questionId": "not-a-uuid"}]},
    {"segments": "not-a-list", "title": ["x"]},
]


# --- Every JSON write endpoint, junk bodies, never a 5xx ---------------------

JSON_WRITE_ENDPOINTS = [
    ("post", "/auth/login", "anonymous"),
    ("post", "/quizzes", "teacher"),
    ("patch", f"/quizzes/{uuid4()}", "teacher"),
    ("post", f"/quizzes/{uuid4()}/submit", "student"),
    ("patch", f"/submissions/{uuid4()}/review", "teacher"),
    ("patch", f"/transcripts/{uuid4()}", "student"),
    ("patch", "/voice-enrollment/active-language", "student"),
]


@pytest.mark.parametrize("method, path, actor", JSON_WRITE_ENDPOINTS)
def test_junk_bodies_never_cause_a_server_error(client, student, teacher, method, path, actor):
    headers = {"student": student.auth, "teacher": teacher.auth, "anonymous": {}}[actor]

    for body in JUNK_BODIES:
        response = getattr(client, method)(path, headers=headers, json=body)
        assert response.status_code < 500, (
            f"{method.upper()} {path} returned {response.status_code} for body {body!r}: "
            f"{response.text[:200]}"
        )


# --- Specific validation rules -------------------------------------------------


def test_login_requires_both_fields(client):
    assert client.post("/auth/login", json={"email": "a@example.com"}).status_code == 422
    assert client.post("/auth/login", json={"password": "x"}).status_code == 422
    assert client.post("/auth/login", json={}).status_code == 422


def test_login_rejects_a_malformed_email(client):
    response = client.post("/auth/login", json={"email": "not-an-email", "password": "x"})
    assert response.status_code == 422


def test_quiz_title_is_required_and_length_limited(client, teacher):
    assert client.post("/quizzes", headers=teacher.auth, json={"questions": []}).status_code == 422
    too_long = {"title": "x" * 256, "questions": []}
    assert client.post("/quizzes", headers=teacher.auth, json=too_long).status_code == 422


def test_quiz_question_text_cannot_be_empty(client, teacher):
    body = {"title": "t", "questions": [{"text": "", "type": "SPOKEN"}]}
    assert client.post("/quizzes", headers=teacher.auth, json=body).status_code == 422


def test_mcq_needs_exactly_one_correct_option(client, teacher):
    two_correct = {
        "title": "t",
        "questions": [{
            "text": "q", "type": "MCQ",
            "options": [{"text": str(i), "isCorrect": i < 2} for i in range(4)],
        }],
    }
    assert client.post("/quizzes", headers=teacher.auth, json=two_correct).status_code == 422


def test_review_mark_must_be_between_0_and_100(client, teacher):
    for bad in (-1, 101, 1000):
        response = client.patch(
            f"/submissions/{uuid4()}/review", headers=teacher.auth, json={"mark": bad}
        )
        assert response.status_code == 422, f"mark={bad} was not rejected"


def test_transcription_upload_requires_a_file_and_a_title(client, student):
    import io

    no_file = client.post("/transcriptions", headers=student.auth, data={"title": "t", "type": "LECTURE"})
    assert no_file.status_code == 422

    no_title = client.post(
        "/transcriptions", headers=student.auth,
        files={"file": ("a.wav", io.BytesIO(b"x"), "audio/wav")}, data={"type": "LECTURE"},
    )
    assert no_title.status_code == 422


def test_transcription_upload_rejects_an_unknown_transcript_type(client, student):
    import io

    response = client.post(
        "/transcriptions", headers=student.auth,
        files={"file": ("a.wav", io.BytesIO(b"RIFFxxxx"), "audio/wav")},
        data={"title": "t", "type": "PODCAST"},
    )
    assert response.status_code in (400, 422)


# --- Malformed path ids --------------------------------------------------------

ID_PATHS = [
    "/quizzes/{}", "/transcripts/{}", "/submissions/{}", "/media/{}",
    "/transcriptions/jobs/{}", "/transcripts/{}/export", "/transcripts/{}/finalize",
]


@pytest.mark.parametrize("template", ID_PATHS)
@pytest.mark.parametrize("bad_id", ["not-a-uuid", "123", "%00", "../../etc/passwd", "a" * 300])
def test_a_malformed_id_is_a_client_error_not_a_crash(client, student, template, bad_id):
    method = "post" if template.endswith("/finalize") else "get"
    response = getattr(client, method)(template.format(bad_id), headers=student.auth)
    assert 400 <= response.status_code < 500, (
        f"{method.upper()} {template.format(bad_id)} returned {response.status_code}"
    )
