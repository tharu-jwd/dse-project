import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.dependencies import CurrentUserDependency, SessionDependency
from app.models.quiz import (
    AnswerSubmission,
    Question,
    QuestionOption,
    Quiz,
    QuizSubmission,
)
from app.models.transcription import Transcript
from app.schemas.quiz import (
    AnswerSubmissionResponse,
    QuestionOptionOwnerResponse,
    QuestionOptionResponse,
    QuestionOwnerResponse,
    QuestionResponse,
    QuizCreate,
    QuizListItem,
    QuizOwnerResponse,
    QuizSubmissionListItem,
    QuizSubmissionResponse,
    QuizSubmitRequest,
    QuizUpdate,
    SubmissionReviewRequest,
)


router = APIRouter(tags=["Quizzes"])


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def require_teacher(user) -> None:
    if user.role != "TEACHER":
        raise _forbidden("Only teachers may perform this action.")


def require_student(user) -> None:
    if user.role != "STUDENT":
        raise _forbidden("Only students may perform this action.")


def get_owned_quiz(db, quiz_id: uuid.UUID, user) -> Quiz:
    quiz = db.get(
        Quiz,
        quiz_id,
        options=[selectinload(Quiz.questions).selectinload(Question.options)],
    )
    if quiz is None:
        raise _not_found("Quiz was not found.")
    if quiz.created_by != user.user_id:
        raise _forbidden("You do not own this quiz.")
    return quiz


def serialize_question_owner(question: Question) -> QuestionOwnerResponse:
    return QuestionOwnerResponse(
        id=question.question_id,
        text=question.question_text,
        type=question.question_type,
        required=question.is_required,
        options=[
            QuestionOptionOwnerResponse(
                id=o.option_id,
                order=o.option_order,
                text=o.option_text,
                is_correct=o.is_correct,
            )
            for o in question.options
        ],
    )


def serialize_question_student(question: Question) -> QuestionResponse:
    return QuestionResponse(
        id=question.question_id,
        text=question.question_text,
        type=question.question_type,
        required=question.is_required,
        options=[
            QuestionOptionResponse(
                id=o.option_id,
                order=o.option_order,
                text=o.option_text,
            )
            for o in question.options
        ],
    )


def serialize_quiz_owner(quiz: Quiz) -> QuizOwnerResponse:
    return QuizOwnerResponse(
        id=quiz.quiz_id,
        title=quiz.title,
        description=quiz.description,
        status=quiz.status,
        dueDate=quiz.available_until.date().isoformat() if quiz.available_until else None,
        questions=[serialize_question_owner(q) for q in quiz.questions],
    )


def submission_status_for(db, quiz: Quiz, student_id: uuid.UUID) -> str:
    submission = db.execute(
        select(QuizSubmission).where(
            QuizSubmission.quiz_id == quiz.quiz_id,
            QuizSubmission.student_id == student_id,
        )
    ).scalar_one_or_none()
    if submission is None:
        return "NOT_STARTED"
    return submission.status


def serialize_quiz_student(quiz: Quiz, db, student_id: uuid.UUID) -> QuizListItem:
    return QuizListItem(
        id=quiz.quiz_id,
        title=quiz.title,
        description=quiz.description,
        status=quiz.status,
        dueDate=quiz.available_until.date().isoformat() if quiz.available_until else None,
        questions=[serialize_question_student(q) for q in quiz.questions],
        submissionStatus=submission_status_for(db, quiz, student_id),
    )


@router.get("/quizzes")
def get_quizzes(db: SessionDependency, current_user: CurrentUserDependency):
    if current_user.role == "TEACHER":
        quizzes = db.execute(
            select(Quiz)
            .where(Quiz.created_by == current_user.user_id)
            .options(selectinload(Quiz.questions).selectinload(Question.options))
        ).scalars().all()
        return [serialize_quiz_owner(q) for q in quizzes]

    quizzes = db.execute(
        select(Quiz)
        .where(Quiz.status == "PUBLISHED")
        .options(selectinload(Quiz.questions).selectinload(Question.options))
    ).scalars().all()
    return [serialize_quiz_student(q, db, current_user.user_id) for q in quizzes]


@router.get("/quizzes/{quiz_id}")
def get_quiz(quiz_id: uuid.UUID, db: SessionDependency, current_user: CurrentUserDependency):
    quiz = db.get(
        Quiz,
        quiz_id,
        options=[selectinload(Quiz.questions).selectinload(Question.options)],
    )
    if quiz is None:
        raise _not_found("Quiz was not found.")

    if current_user.role == "TEACHER":
        if quiz.created_by != current_user.user_id:
            raise _forbidden("You do not own this quiz.")
        return serialize_quiz_owner(quiz)

    if quiz.status != "PUBLISHED":
        raise _not_found("Quiz was not found.")
    return serialize_quiz_student(quiz, db, current_user.user_id)


def _apply_questions(db, quiz: Quiz, questions_data) -> None:
    quiz.questions.clear()
    db.flush()
    for index, q in enumerate(questions_data, start=1):
        question = Question(
            quiz_id=quiz.quiz_id,
            question_order=index,
            question_text=q.text,
            question_type=q.type,
            is_required=q.required,
        )
        if q.type == "MCQ":
            for opt_index, opt in enumerate(q.options, start=1):
                question.options.append(
                    QuestionOption(
                        option_order=opt_index,
                        option_text=opt.text,
                        is_correct=opt.isCorrect,
                    )
                )
        quiz.questions.append(question)


@router.post("/quizzes", response_model=QuizOwnerResponse, status_code=status.HTTP_201_CREATED)
def create_quiz(payload: QuizCreate, db: SessionDependency, current_user: CurrentUserDependency):
    require_teacher(current_user)
    quiz = Quiz(
        created_by=current_user.user_id,
        title=payload.title,
        description=payload.description,
        status="DRAFT",
        available_until=(
            datetime.fromisoformat(f"{payload.dueDate}T23:59:59+00:00")
            if payload.dueDate
            else None
        ),
    )
    db.add(quiz)
    db.flush()
    _apply_questions(db, quiz, payload.questions)
    db.commit()
    db.refresh(quiz)
    return serialize_quiz_owner(quiz)


@router.patch("/quizzes/{quiz_id}", response_model=QuizOwnerResponse)
def update_quiz(
    quiz_id: uuid.UUID,
    payload: QuizUpdate,
    db: SessionDependency,
    current_user: CurrentUserDependency,
):
    require_teacher(current_user)
    quiz = get_owned_quiz(db, quiz_id, current_user)
    quiz.title = payload.title
    quiz.description = payload.description
    quiz.available_until = (
        datetime.fromisoformat(f"{payload.dueDate}T23:59:59+00:00")
        if payload.dueDate
        else None
    )
    _apply_questions(db, quiz, payload.questions)
    db.commit()
    db.refresh(quiz)
    return serialize_quiz_owner(quiz)


@router.post("/quizzes/{quiz_id}/publish", response_model=QuizOwnerResponse)
def publish_quiz(quiz_id: uuid.UUID, db: SessionDependency, current_user: CurrentUserDependency):
    require_teacher(current_user)
    quiz = get_owned_quiz(db, quiz_id, current_user)
    if not quiz.questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A quiz needs at least one question before it can be published.",
        )
    quiz.status = "PUBLISHED"
    db.commit()
    db.refresh(quiz)
    return serialize_quiz_owner(quiz)


@router.post("/quizzes/{quiz_id}/submit", response_model=QuizSubmissionResponse)
def submit_quiz(
    quiz_id: uuid.UUID,
    payload: QuizSubmitRequest,
    db: SessionDependency,
    current_user: CurrentUserDependency,
):
    require_student(current_user)
    quiz = db.get(
        Quiz,
        quiz_id,
        options=[selectinload(Quiz.questions).selectinload(Question.options)],
    )
    if quiz is None or quiz.status != "PUBLISHED":
        raise _not_found("Quiz was not found.")

    questions_by_id = {q.question_id: q for q in quiz.questions}

    submission = db.execute(
        select(QuizSubmission).where(
            QuizSubmission.quiz_id == quiz_id,
            QuizSubmission.student_id == current_user.user_id,
        )
    ).scalar_one_or_none()
    if submission is None:
        submission = QuizSubmission(
            quiz_id=quiz_id,
            student_id=current_user.user_id,
            status="IN_PROGRESS",
        )
        db.add(submission)
        db.flush()

    for answer in payload.answers:
        question = questions_by_id.get(answer.questionId)
        if question is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Answer references a question that does not belong to this quiz.",
            )

        existing = db.execute(
            select(AnswerSubmission).where(
                AnswerSubmission.submission_id == submission.submission_id,
                AnswerSubmission.question_id == question.question_id,
            )
        ).scalar_one_or_none()

        if question.question_type == "MCQ":
            if answer.selectedOptionId is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="MCQ answers must set selectedOptionId.",
                )
            option_ids = {o.option_id for o in question.options}
            if answer.selectedOptionId not in option_ids:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Selected option does not belong to this question.",
                )
            if existing:
                existing.selected_option_id = answer.selectedOptionId
                existing.status = "COMPLETED"
            else:
                db.add(
                    AnswerSubmission(
                        submission_id=submission.submission_id,
                        question_id=question.question_id,
                        selected_option_id=answer.selectedOptionId,
                        status="COMPLETED",
                    )
                )
        else:
            if answer.transcriptId is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Spoken answers must set transcriptId.",
                )
            transcript = db.get(Transcript, answer.transcriptId)
            if transcript is None or transcript.owner_id != current_user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Transcript does not belong to this student.",
                )
            if existing:
                existing.transcription_id = answer.transcriptId
                existing.status = "COMPLETED"
            else:
                db.add(
                    AnswerSubmission(
                        submission_id=submission.submission_id,
                        question_id=question.question_id,
                        transcription_id=answer.transcriptId,
                        status="COMPLETED",
                    )
                )

    submission.status = "SUBMITTED"
    submission.submitted_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(submission)
    return serialize_submission(db, submission, include_correctness=False)


def serialize_submission(
    db,
    submission: QuizSubmission,
    include_correctness: bool,
) -> QuizSubmissionResponse:
    quiz = db.get(Quiz, submission.quiz_id)
    student = submission.student
    answers = db.execute(
        select(AnswerSubmission).where(
            AnswerSubmission.submission_id == submission.submission_id
        )
    ).scalars().all()

    answer_responses = []
    for answer in answers:
        question = db.get(
            Question,
            answer.question_id,
            options=[selectinload(Question.options)],
        )
        transcript_text = None
        if answer.transcription_id:
            transcript = db.get(Transcript, answer.transcription_id)
            if transcript is not None:
                segments = sorted(transcript.segments, key=lambda s: s.segment_order)
                transcript_text = " ".join(
                    (s.edited_text if s.edited_text is not None else s.generated_text)
                    for s in segments
                )

        is_correct = None
        if include_correctness and question.question_type == "MCQ" and answer.selected_option_id:
            selected = next(
                (o for o in question.options if o.option_id == answer.selected_option_id),
                None,
            )
            is_correct = bool(selected and selected.is_correct)

        answer_responses.append(
            AnswerSubmissionResponse(
                questionId=question.question_id,
                question=question.question_text,
                type=question.question_type,
                options=(
                    [
                        QuestionOptionOwnerResponse(
                            id=o.option_id,
                            order=o.option_order,
                            text=o.option_text,
                            is_correct=o.is_correct,
                        )
                        for o in question.options
                    ]
                    if include_correctness
                    else []
                ),
                selectedOptionId=answer.selected_option_id,
                isCorrect=is_correct,
                transcript=transcript_text,
            )
        )

    return QuizSubmissionResponse(
        id=submission.submission_id,
        quizId=submission.quiz_id,
        quizTitle=quiz.title if quiz else "",
        studentId=submission.student_id,
        studentName=student.name if student else "",
        status=submission.status,
        mark=submission.mark,
        feedback=submission.feedback,
        submittedAt=submission.submitted_at,
        answers=answer_responses,
    )


@router.get("/submissions", response_model=list[QuizSubmissionListItem])
def get_submissions(db: SessionDependency, current_user: CurrentUserDependency):
    require_teacher(current_user)
    submissions = db.execute(
        select(QuizSubmission)
        .join(Quiz, Quiz.quiz_id == QuizSubmission.quiz_id)
        .where(Quiz.created_by == current_user.user_id)
    ).scalars().all()
    result = []
    for submission in submissions:
        quiz = db.get(Quiz, submission.quiz_id)
        student = submission.student
        result.append(
            QuizSubmissionListItem(
                id=submission.submission_id,
                quizId=submission.quiz_id,
                quizTitle=quiz.title if quiz else "",
                studentId=submission.student_id,
                studentName=student.name if student else "",
                status=submission.status,
                mark=submission.mark,
                submittedAt=submission.submitted_at,
            )
        )
    return result


@router.get("/submissions/{submission_id}", response_model=QuizSubmissionResponse)
def get_submission(
    submission_id: uuid.UUID,
    db: SessionDependency,
    current_user: CurrentUserDependency,
):
    submission = db.get(QuizSubmission, submission_id)
    if submission is None:
        raise _not_found("Submission was not found.")

    quiz = db.get(Quiz, submission.quiz_id)
    is_owner_teacher = (
        current_user.role == "TEACHER"
        and quiz is not None
        and quiz.created_by == current_user.user_id
    )
    is_owning_student = (
        current_user.role == "STUDENT" and submission.student_id == current_user.user_id
    )
    if not (is_owner_teacher or is_owning_student):
        raise _forbidden("You do not have access to this submission.")

    return serialize_submission(db, submission, include_correctness=is_owner_teacher)


@router.patch("/submissions/{submission_id}/review", response_model=QuizSubmissionResponse)
def review_submission(
    submission_id: uuid.UUID,
    payload: SubmissionReviewRequest,
    db: SessionDependency,
    current_user: CurrentUserDependency,
):
    require_teacher(current_user)
    submission = db.get(QuizSubmission, submission_id)
    if submission is None:
        raise _not_found("Submission was not found.")

    quiz = db.get(Quiz, submission.quiz_id)
    if quiz is None or quiz.created_by != current_user.user_id:
        raise _forbidden("You do not own the quiz for this submission.")

    submission.mark = payload.mark
    submission.feedback = payload.feedback
    submission.status = "REVIEWED"
    submission.reviewed_by = current_user.user_id
    submission.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(submission)
    return serialize_submission(db, submission, include_correctness=True)
