from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.models.user import User


def get_user_by_email(
        db: Session,
        email: str,
) -> User | None:
    normalized_email = email.strip().lower()
    statement = select(User).where(User.email == normalized_email)
    return db.scalar(statement)

def authenticate_user(
        db: Session,
        email: str,
        password: str,
) -> User | None:
    user = get_user_by_email(db, email)
    if user is None or not user.is_active:
        return None

    if not verify_password(password, user.password_hash):
        return None

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    return user
