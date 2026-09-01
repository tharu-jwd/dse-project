from typing import Literal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, EmailStr, Field


UserRole = Literal["STUDENT", "TEACHER"]

class UserResponse(BaseModel):
    id: UUID = Field(
        validation_alias=AliasChoices("user_id", "id"),
    )
    name: str
    email: EmailStr
    role: UserRole
    command_language: Literal["si", "en"] = Field(
        validation_alias=AliasChoices("command_language", "commandLanguage"),
        serialization_alias="commandLanguage",
    )

    model_config= ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )
