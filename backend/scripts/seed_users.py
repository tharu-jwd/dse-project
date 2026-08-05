from sqlalchemy import select

from app.core.security import hash_password
from app.db.session import SessionLocal
from app.models.user import User


DEMO_USERS = [
    {
        "name": "Nethum Perera",
        "email": "student@sinhaspeech.lk",
        "password": "demo123",
        "role": "STUDENT",
    },
    {
        "name": "Dr. Kasun Silva",
        "email": "teacher@sinhaspeech.lk",
        "password": "demo123",
        "role": "TEACHER",
    },
]


def seed_users() -> None:
    created_count = 0

    with SessionLocal.begin() as db:
        for user_data in DEMO_USERS:
            email = user_data["email"].strip().lower()

            existing_user = db.scalar(
                select(User).where(User.email == email)
            )

            if existing_user is not None:
                print(f"Skipped existing user: {email}")
                continue

            user = User(
                name=user_data["name"],
                email=email,
                password_hash=hash_password(user_data["password"]),
                role=user_data["role"],
                is_active=True,
            )

            db.add(user)
            created_count += 1
            print(f"Created user: {email}")

    print(f"Seeding completed. Created {created_count} user(s).")


if __name__ == "__main__":
    seed_users()
