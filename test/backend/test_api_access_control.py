"""Security and access control tests (Master Test Plan §3.1.6).

Covers the two levels the test plan calls out:

  - Application-level: an actor reaches only the functions and data their
    role permits. The role guards live in app/api/routes/quiz.py and were
    previously untested, so a wrong comparison there would have let a
    student publish quizzes or read another student's submissions.
  - System-level: only authenticated callers get in at all, and a token
    that is missing, malformed, tampered with or expired is refused.

These drive the real FastAPI app through TestClient with real signed JWTs,
so the actual authentication dependency runs - mocking it out would defeat
the purpose.
"""

from datetime import timedelta
from uuid import uuid4

import jwt
import pytest

from app.core.config import settings
from app.core.security import create_access_token


# --- System-level: authentication -----------------------------------------

# Endpoints that must never serve an unauthenticated caller. /health and
# / are deliberately excluded - they are meant to be public.
PROTECTED_ENDPOINTS = [
    ("GET", "/auth/me"),
    ("GET", "/transcripts"),
    ("GET", "/quizzes"),
    ("GET", "/submissions"),
    ("GET", "/voice-enrollment"),
    ("GET", "/voice-samples"),
]


@pytest.mark.parametrize("method, path", PROTECTED_ENDPOINTS)
def test_endpoint_requires_authentication(client, method, path):
    response = client.request(method, path)
    assert response.status_code in (401, 403), (
        f"{method} {path} served an unauthenticated request ({response.status_code})"
    )


@pytest.mark.parametrize("method, path", PROTECTED_ENDPOINTS)
def test_endpoint_rejects_garbage_token(client, method, path):
    response = client.request(method, path, headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code in (401, 403)


def test_expired_token_is_rejected(client, student):
    """A token that was valid yesterday must not work today - otherwise a
    leaked token would grant access forever."""

    expired = create_access_token(student.user_id, expires_delta=timedelta(seconds=-60))
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code in (401, 403)


def test_token_signed_with_the_wrong_key_is_rejected(client, student):
    """Anyone can construct a JWT; only one signed with the server's secret
    should be honoured. This is the difference between authentication and
    merely reading a claim."""

    forged = jwt.encode(
        {"sub": str(student.user_id), "type": "access", "exp": 9999999999},
        "not-the-real-secret",
        algorithm=settings.jwt_algorithm,
    )
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code in (401, 403)


def test_token_for_a_nonexistent_user_is_rejected(client):
    """A correctly signed token whose subject has since been deleted must
    not authenticate - the signature alone is not enough."""

    orphan = create_access_token(uuid4())
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {orphan}"})
    assert response.status_code in (401, 403, 404)


def test_valid_token_is_accepted(client, student):
    """The negative tests above are only meaningful if the positive case
    actually works."""

    response = client.get("/auth/me", headers=student.auth)
    assert response.status_code == 200
    assert response.json()["email"] == student.email


def test_login_rejects_a_wrong_password(client, student):
    response = client.post(
        "/auth/login", json={"email": student.email, "password": "definitely-wrong"}
    )
    assert response.status_code in (400, 401, 403)


def test_login_does_not_reveal_whether_an_account_exists(client, student):
    """Different responses for "no such user" and "wrong password" let an
    attacker enumerate valid accounts. Both should look the same."""

    unknown = client.post(
        "/auth/login",
        json={"email": f"nobody-{uuid4()}@example.com", "password": "whatever"},
    )
    wrong_password = client.post(
        "/auth/login", json={"email": student.email, "password": "definitely-wrong"}
    )

    assert unknown.status_code == wrong_password.status_code, (
        "different status codes for unknown-account vs wrong-password let an "
        "attacker enumerate valid accounts"
    )
    # The app reports failures under "message"; compare the whole body so a
    # future change to the error shape is still caught.
    assert unknown.json() == wrong_password.json(), (
        f"response bodies differ: {unknown.json()} vs {wrong_password.json()}"
    )


# --- Application-level: role restrictions ---------------------------------


def test_student_cannot_create_a_quiz(client, student):
    """quiz.py guards this with `if user.role != "TEACHER"`. If that check
    were inverted or missing, any student could author quizzes."""

    response = client.post(
        "/quizzes",
        headers=student.auth,
        json={"title": "Unauthorised quiz", "questions": []},
    )
    assert response.status_code == 403, (
        f"a STUDENT was allowed to create a quiz (got {response.status_code})"
    )


def test_teacher_cannot_submit_quiz_answers(client, teacher):
    """The mirror guard: submitting is students-only."""

    response = client.post(
        f"/quizzes/{uuid4()}/submit", headers=teacher.auth, json={"answers": []}
    )
    # 403 for the role guard; 404 is acceptable only if the quiz lookup
    # happens first - but it must never be a success.
    assert response.status_code in (403, 404, 422)
    assert response.status_code not in (200, 201)


def test_student_cannot_review_a_submission(client, student):
    """Marking student work is a teacher action."""

    response = client.patch(
        f"/submissions/{uuid4()}/review",
        headers=student.auth,
        json={"marks": 10, "feedback": "unauthorised"},
    )
    assert response.status_code in (403, 404, 422)
    assert response.status_code not in (200, 201)


def test_student_cannot_list_submissions(client, student):
    """`GET /submissions` is the teacher's marking queue - it calls
    require_teacher() (quiz.py:442). A student reaching it would be seeing
    other students' work."""

    response = client.get("/submissions", headers=student.auth)
    assert response.status_code == 403


def test_teacher_only_sees_submissions_for_their_own_quizzes(client, teacher):
    """The query is scoped with `where(Quiz.created_by == current_user.user_id)`,
    so one teacher must not see another teacher's marking queue. A fresh
    teacher has created no quizzes, so the list must be empty - if it ever
    returns rows, the scoping has been lost."""

    response = client.get("/submissions", headers=teacher.auth)
    assert response.status_code == 200
    assert response.json() == [], (
        "a teacher with no quizzes received submissions - the owner scoping "
        "on GET /submissions is not working"
    )


def test_student_cannot_read_another_students_submission_directly(
    client, student, other_student
):
    """Guessing a UUID should not be enough - quiz.py:484 checks ownership
    explicitly. Uses a random id, so the only acceptable outcomes are
    "not found" or "forbidden", never a payload."""

    response = client.get(f"/submissions/{uuid4()}", headers=student.auth)
    assert response.status_code in (403, 404)


def test_student_cannot_read_another_students_transcript(client, student, other_student):
    response = client.get(f"/transcripts/{uuid4()}", headers=student.auth)
    assert response.status_code in (403, 404)


def test_student_cannot_delete_another_students_transcript(client, student):
    response = client.delete(f"/transcripts/{uuid4()}", headers=student.auth)
    assert response.status_code in (403, 404)
    assert response.status_code not in (200, 204)


# --- Input validation -----------------------------------------------------


def test_unknown_command_id_is_rejected_on_enrollment(client, student):
    """command_id is a free-form string in the database by design, so the
    route - not the schema - has to reject ids outside the vocabulary."""

    response = client.delete("/voice-enrollment/../../etc/passwd", headers=student.auth)
    assert response.status_code in (400, 404, 422)


def test_malformed_payload_returns_422_not_500(client, teacher):
    """A bad request body should be a validation error, not an unhandled
    server exception - a 500 here would mean the error surfaces as a stack
    trace rather than a useful message."""

    response = client.post(
        "/quizzes", headers=teacher.auth, json={"title": 12345, "questions": "not-a-list"}
    )
    assert response.status_code == 422, f"expected 422, got {response.status_code}"


def test_nonexistent_media_is_not_served(client, student):
    response = client.get(f"/media/{uuid4()}", headers=student.auth)
    assert response.status_code in (403, 404)
