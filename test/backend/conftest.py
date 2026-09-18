import uuid

import pytest

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
                email=f"test-fixture-{user_id}@example.invalid",
                password_hash="not-a-real-hash",
                role="STUDENT",
            )
        )

    yield user_id

    with SessionLocal.begin() as db:
        db.query(User).filter(User.user_id == user_id).delete()
