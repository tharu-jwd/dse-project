import uuid
from dataclasses import dataclass

import pytest

from app.core.security import create_access_token, hash_password
from app.db.session import SessionLocal
from app.models.user import User


@pytest.fixture
def db_user():
    """A throwaway user in the real dev database, for tests that need a
    genuine user_id to satisfy command_enrollments' foreign key.

    There's no isolated test-database setup in this project yet, so this
    talks to the same Postgres instance the app runs against and cleans
    up after itself; deleting the user cascades to any command_enrollments
    rows created against it (ON DELETE CASCADE).
    """

    user_id = uuid.uuid4()

    with SessionLocal.begin() as db:
        db.add(
            User(
                user_id=user_id,
                name="Test Fixture User",
                email=f"test-fixture-{user_id}@example.com",
                password_hash="not-a-real-hash",
                role="STUDENT",
            )
        )

    yield user_id

    with SessionLocal.begin() as db:
        db.query(User).filter(User.user_id == user_id).delete()


# --- API-level fixtures ---------------------------------------------------
#
# The fixtures below back the HTTP tests (test_api_*.py). They create real
# users in the database and sign real JWTs, so the tests exercise the actual
# authentication dependency rather than bypassing it with a mock - the point
# of an access-control test is that the real guard runs.


@dataclass(frozen=True)
class TestUser:
    user_id: uuid.UUID
    email: str
    password: str
    role: str
    token: str

    @property
    def auth(self) -> dict[str, str]:
        """Headers for an authenticated request as this user."""
        return {"Authorization": f"Bearer {self.token}"}


def _make_user(role: str, password: str = "test-password-123") -> TestUser:
    user_id = uuid.uuid4()
    email = f"test-{role.lower()}-{user_id}@example.com"

    with SessionLocal.begin() as db:
        db.add(
            User(
                user_id=user_id,
                name=f"Test {role.title()}",
                email=email,
                password_hash=hash_password(password),
                role=role,
            )
        )

    return TestUser(
        user_id=user_id,
        email=email,
        password=password,
        role=role,
        token=create_access_token(user_id),
    )


def _delete_user(user_id: uuid.UUID) -> None:
    """Deletes a fixture user and anything the API tests created for them.

    Quizzes and submissions use ON DELETE RESTRICT on their owner/student
    foreign keys (deliberately - a real user's quiz history must not
    vanish silently if their account is removed), so unlike
    command_enrollments' ON DELETE CASCADE, those rows have to be cleared
    explicitly before the user row itself can go.
    """

    from app.models.media import MediaFile
    from app.models.quiz import AnswerSubmission, Quiz, QuizSubmission
    from app.models.transcription import Transcript, TranscriptionJob, TranscriptSegment
    from app.services.media_storage_service import resolve_stored_media

    with SessionLocal.begin() as db:
        submission_ids = [
            row[0]
            for row in db.query(QuizSubmission.submission_id).filter(
                QuizSubmission.student_id == user_id
            )
        ]
        if submission_ids:
            db.query(AnswerSubmission).filter(
                AnswerSubmission.submission_id.in_(submission_ids)
            ).delete(synchronize_session=False)
        db.query(QuizSubmission).filter(QuizSubmission.student_id == user_id).delete()
        db.query(Quiz).filter(Quiz.created_by == user_id).delete()

        transcript_ids = [
            row[0]
            for row in db.query(Transcript.transcript_id).filter(
                Transcript.owner_id == user_id
            )
        ]
        if transcript_ids:
            db.query(TranscriptSegment).filter(
                TranscriptSegment.transcript_id.in_(transcript_ids)
            ).delete(synchronize_session=False)
        db.query(Transcript).filter(Transcript.owner_id == user_id).delete()
        db.query(TranscriptionJob).filter(TranscriptionJob.requested_by == user_id).delete()

        media_files = list(db.query(MediaFile).filter(MediaFile.owner_id == user_id))
        for media in media_files:
            # resolve_stored_media() raises if the file is already gone, and a
            # test may have deleted it deliberately - cleanup must still finish.
            try:
                resolve_stored_media(media.storage_path).unlink(missing_ok=True)
            except FileNotFoundError:
                pass
        db.query(MediaFile).filter(MediaFile.owner_id == user_id).delete()

        db.query(User).filter(User.user_id == user_id).delete()


@pytest.fixture
def student():
    user = _make_user("STUDENT")
    yield user
    _delete_user(user.user_id)


@pytest.fixture
def other_student():
    """A second student, for checking one student cannot reach another's work."""
    user = _make_user("STUDENT")
    yield user
    _delete_user(user.user_id)


@pytest.fixture
def teacher():
    user = _make_user("TEACHER")
    yield user
    _delete_user(user.user_id)


@pytest.fixture
def other_teacher():
    """A second teacher, for checking one teacher cannot act on another
    teacher's quizzes or submissions."""

    user = _make_user("TEACHER")
    yield user
    _delete_user(user.user_id)


@pytest.fixture(scope="session")
def client():
    """FastAPI test client.

    `streaming_enabled` is forced off for the lifespan so the app does not
    spend ~70 seconds loading Whisper models on import - these tests only
    exercise the HTTP API, and the streaming path has its own tests in
    test_streaming_commands_route.py.
    """

    from fastapi.testclient import TestClient

    from app.core.config import settings

    original = settings.streaming_enabled
    settings.streaming_enabled = False
    try:
        from app.main import app

        with TestClient(app) as test_client:
            yield test_client
    finally:
        settings.streaming_enabled = original


@pytest.fixture
def empty_job_queue():
    """Skips the test unless the shared dev database has no queued or
    processing transcription jobs.

    `claim_next_job()` / `process_next_job()` operate on the oldest QUEUED
    row across the WHOLE table, and these tests run against the shared dev
    database. If a real, unrelated job were pending, a test calling them
    would claim it - and "complete" someone else's upload with the fake
    transcriber's canned text. Skipping is always safer than forcing.
    """

    from app.models.transcription import TranscriptionJob

    with SessionLocal() as db:
        pending = (
            db.query(TranscriptionJob)
            .filter(TranscriptionJob.status.in_(("QUEUED", "PROCESSING")))
            .count()
        )
    if pending:
        pytest.skip(
            f"{pending} real job(s) already queued/processing in the shared dev "
            "database - skipping to avoid claiming someone else's job"
        )
