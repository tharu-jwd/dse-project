from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


QuestionType = Literal["MCQ", "SPOKEN"]

QuizStatus = Literal["DRAFT", "PUBLISHED", "CLOSED"]

SubmissionStatus = Literal["NOT_STARTED", "IN_PROGRESS", "SUBMITTED", "REVIEWED"]

AnswerStatus = Literal["NOT_STARTED", "PROCESSING", "COMPLETED", "FAILED"]


# --- Options -----------------------------------------------------------

class QuestionOptionCreate(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    isCorrect: bool = False

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class QuestionOptionResponse(BaseModel):
    id: UUID
    order: int = Field(serialization_alias="order")
    text: str

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class QuestionOptionOwnerResponse(QuestionOptionResponse):
    is_correct: bool = Field(serialization_alias="isCorrect")


# --- Questions -----------------------------------------------------------

class QuestionCreate(BaseModel):
    id: str | None = None
    text: str = Field(min_length=1, max_length=10_000)
    type: QuestionType = "SPOKEN"
    required: bool = True
    options: list[QuestionOptionCreate] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @model_validator(mode="after")
    def validate_options(self) -> "QuestionCreate":
        if self.type == "MCQ":
            if len(self.options) != 4:
                raise ValueError("MCQ questions must have exactly 4 options.")
            if sum(1 for o in self.options if o.isCorrect) != 1:
                raise ValueError(
                    "MCQ questions must have exactly one correct option."
                )
            if any(not o.text.strip() for o in self.options):
                raise ValueError("MCQ options cannot be empty.")
        return self


class QuestionResponse(BaseModel):
    id: UUID
    text: str = Field(serialization_alias="text")
    type: QuestionType = Field(serialization_alias="type")
    required: bool = Field(serialization_alias="required")
    options: list[QuestionOptionResponse] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class QuestionOwnerResponse(QuestionResponse):
    options: list[QuestionOptionOwnerResponse] = Field(default_factory=list)


# --- Quiz -----------------------------------------------------------

class QuizCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = None
    dueDate: str | None = None
    questions: list[QuestionCreate] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class QuizUpdate(QuizCreate):
    pass


class QuizListItem(BaseModel):
    id: UUID
    title: str
    description: str | None = None
    status: QuizStatus
    dueDate: str | None = Field(default=None, serialization_alias="dueDate")
    questions: list[QuestionResponse] = Field(default_factory=list)
    submissionStatus: SubmissionStatus | None = Field(
        default=None, serialization_alias="submissionStatus"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class QuizResponse(QuizListItem):
    pass


class QuizOwnerResponse(QuizListItem):
    questions: list[QuestionOwnerResponse] = Field(default_factory=list)


# --- Submissions -----------------------------------------------------------

class AnswerSubmit(BaseModel):
    questionId: UUID
    selectedOptionId: UUID | None = None
    transcriptId: UUID | None = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class QuizSubmitRequest(BaseModel):
    answers: list[AnswerSubmit] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AnswerSubmissionResponse(BaseModel):
    questionId: UUID
    question: str
    type: QuestionType
    options: list[QuestionOptionOwnerResponse] = Field(default_factory=list)
    selectedOptionId: UUID | None = None
    isCorrect: bool | None = None
    transcript: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class QuizSubmissionResponse(BaseModel):
    id: UUID
    quizId: UUID
    quizTitle: str
    studentId: UUID
    studentName: str
    status: SubmissionStatus
    mark: float | None = None
    feedback: str | None = None
    submittedAt: datetime | None = None
    answers: list[AnswerSubmissionResponse] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class QuizSubmissionListItem(BaseModel):
    id: UUID
    quizId: UUID
    quizTitle: str
    studentId: UUID
    studentName: str
    status: SubmissionStatus
    mark: float | None = None
    submittedAt: datetime | None = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SubmissionReviewRequest(BaseModel):
    mark: float | None = Field(default=None, ge=0, le=100)
    feedback: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
