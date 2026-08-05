from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, EmailStr, SecretStr, Field, field_validator

from app.schemas.user import UserResponse


class LoginRequest(BaseModel):
    email: EmailStr
    password: SecretStr = Field(
        min_length=1,
        max_length=128,
    )

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class LoginResponse(BaseModel):
    token: str
    user: UserResponse


class TokenPayload(BaseModel):
    sub: UUID
    iat: datetime
    exp: datetime
    type: Literal["access"]
