from datetime import datetime, timedelta, timezone
from uuid import UUID
import jwt
from jwt.exceptions import InvalidTokenError as JWTInvalidTokenError
from pydantic import ValidationError
from pwdlib import PasswordHash

from app.core.config import settings
from app.schemas.auth import TokenPayload


password_hasher = PasswordHash.recommended()

class TokenValidationError(ValueError):
    """Raised when an access token is missing, invalid or expired"""

def hash_password(plain_password: str) -> str:
    if not plain_password:
        raise ValueError("Password cannot be empty")

    return password_hasher.hash(plain_password)

def verify_password(
        plain_password: str,
        hashed_password: str,
) -> bool:
    if not plain_password or not hashed_password:
        return False

    return password_hasher.verify(
        plain_password,
        hashed_password,
    )

def create_access_token(
        user_id: UUID,
        expires_delta: timedelta | None = None
) -> str:
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )

    payload = {
        "sub": str(user_id),
        "iat": issued_at,
        "exp": expires_at,
        "type": "access",
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

def decode_access_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={
                "require": ["sub", "iat", "exp", "type"]
            },
        )
        return TokenPayload.model_validate(payload)
    except (JWTInvalidTokenError, ValidationError) as error:
        raise TokenValidationError(
            "The access token is invalid or expired"
        ) from error
