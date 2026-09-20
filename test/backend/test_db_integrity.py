"""Data and database integrity tests (Master Test Plan §3.1.1) beyond what
test_voice_enrollment.py already covers: constraint enforcement and
foreign-key delete behaviour, exercised directly against Postgres rather
than through the service layer.
"""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.session import SessionLocal
from app.models.quiz import Quiz, QuizSubmission
from app.models.voice_enrollment import CommandEnrollment


# --- Unique constraints ----------------------------------------------------


def test_duplicate_enrollment_slot_is_rejected_at_the_database_level(student):
    """uq_command_enrollments_slot is (user_id, command_id, language,
    sample_index). This bypasses app.services.voice_enrollment entirely -
    unlike its application-level check, this proves the database itself
    would refuse the duplicate even if a future code path forgot to call
    that service."""

    row = dict(
        user_id=student.user_id,
        command_id="next",
        language="si",
        sample_index=0,
        embedding=(b"\x00" * 4),
        embedding_dim=1,
        model_version="test",
    )

    with SessionLocal.begin() as db:
        db.add(CommandEnrollment(**row))

    with pytest.raises(IntegrityError):
        with SessionLocal.begin() as db:
            db.add(CommandEnrollment(**{**row, "embedding": b"\x01" * 4}))

    with SessionLocal.begin() as db:
        db.query(CommandEnrollment).filter(
            CommandEnrollment.user_id == student.user_id
        ).delete()


def test_same_sample_index_is_allowed_for_a_different_language(student):
    """The uniqueness is scoped per-language on purpose (see
    app.services.voice_enrollment's module docstring) - sample_index 0
    must be usable independently for 'si' and 'en'."""

    base = dict(
        user_id=student.user_id,
        command_id="next",
        sample_index=0,
        embedding=b"\x00" * 4,
        embedding_dim=1,
        model_version="test",
    )

    with SessionLocal.begin() as db:
        db.add(CommandEnrollment(**{**base, "language": "si"}))
        db.add(CommandEnrollment(**{**base, "language": "en"}))

    with SessionLocal() as db:
        count = (
            db.query(CommandEnrollment)
            .filter(CommandEnrollment.user_id == student.user_id)
            .count()
        )
    assert count == 2

    with SessionLocal.begin() as db:
        db.query(CommandEnrollment).filter(
            CommandEnrollment.user_id == student.user_id
        ).delete()


# --- Foreign-key delete behaviour ------------------------------------------


def test_deleting_a_user_cascades_their_voice_enrollments(student):
    """command_enrollments.user_id is ON DELETE CASCADE - deleting the
    user must remove their enrollment rows rather than leaving them
    orphaned or blocking the delete."""

    with SessionLocal.begin() as db:
        db.add(
            CommandEnrollment(
                user_id=student.user_id,
                command_id="next",
                language="si",
                sample_index=0,
                embedding=b"\x00" * 4,
                embedding_dim=1,
                model_version="test",
            )
        )

    from app.models.user import User

    with SessionLocal.begin() as db:
        db.query(User).filter(User.user_id == student.user_id).delete()

    with SessionLocal() as db:
        remaining = (
            db.query(CommandEnrollment)
            .filter(CommandEnrollment.user_id == student.user_id)
            .count()
        )
    assert remaining == 0, "deleting a user left orphaned command_enrollments rows"


def test_deleting_a_teacher_with_quizzes_is_restricted(client, teacher, mcq_quiz_payload):
    """quizzes.created_by is ON DELETE RESTRICT - a teacher's quiz history
    must not silently vanish (or, worse, leave quizzes with a dangling
    owner) if their account row is removed. The API layer never exposes
    user deletion; this proves the database's own constraint would still
    catch it if it ever did."""

    response = client.post("/quizzes", headers=teacher.auth, json=mcq_quiz_payload)
    assert response.status_code == 201
    quiz_id = response.json()["id"]

    from app.models.user import User

    with pytest.raises(IntegrityError):
        with SessionLocal.begin() as db:
            db.query(User).filter(User.user_id == teacher.user_id).delete()

    # Clean up in the correct order so the fixture's own teardown doesn't
    # also hit this constraint.
    with SessionLocal.begin() as db:
        db.query(Quiz).filter(Quiz.quiz_id == uuid.UUID(quiz_id)).delete()


def test_deleting_a_quiz_cascades_its_questions(client, teacher, mcq_quiz_payload):
    """questions.quiz_id is ON DELETE CASCADE - removing a quiz must not
    leave orphaned question/option rows behind."""

    response = client.post("/quizzes", headers=teacher.auth, json=mcq_quiz_payload)
    quiz_id = uuid.UUID(response.json()["id"])

    from app.models.quiz import Question

    with SessionLocal.begin() as db:
        db.query(Quiz).filter(Quiz.quiz_id == quiz_id).delete()

    with SessionLocal() as db:
        remaining_questions = (
            db.query(Question).filter(Question.quiz_id == quiz_id).count()
        )
    assert remaining_questions == 0, "deleting a quiz left orphaned question rows"


@pytest.fixture
def mcq_quiz_payload():
    return {
        "title": "DB integrity check quiz",
        "questions": [
            {
                "text": "placeholder",
                "type": "MCQ",
                "options": [
                    {"text": "a", "isCorrect": True},
                    {"text": "b", "isCorrect": False},
                    {"text": "c", "isCorrect": False},
                    {"text": "d", "isCorrect": False},
                ],
            }
        ],
    }
