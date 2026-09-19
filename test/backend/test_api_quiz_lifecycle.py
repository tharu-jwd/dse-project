"""Function tests for the quiz REST API (Master Test Plan §3.1.2).

Drives the real FastAPI app through TestClient with real signed JWTs,
covering the full quiz lifecycle as HTTP requests: create -> publish ->
submit -> review. test_api_access_control.py already covers who is
allowed to call each of these; this file covers whether the calls that
ARE allowed actually do the right thing.

Both answer types are covered: MCQ answers in the main sections, and
SPOKEN answers (which reference a transcript the student already owns) in
the "Spoken answers" section at the bottom.
"""

import pytest


@pytest.fixture
def mcq_quiz_payload():
    """One MCQ question with exactly 4 options and exactly one correct
    one - QuizCreate's validator rejects anything else (schemas/quiz.py)."""

    return {
        "title": "Sinhala vocabulary check",
        "description": "A short quiz",
        "questions": [
            {
                "text": "What does 'ආයුබෝවන්' mean?",
                "type": "MCQ",
                "required": True,
                "options": [
                    {"text": "Hello", "isCorrect": True},
                    {"text": "Goodbye", "isCorrect": False},
                    {"text": "Thank you", "isCorrect": False},
                    {"text": "Please", "isCorrect": False},
                ],
            }
        ],
    }


def _create_quiz(client, teacher, payload):
    response = client.post("/quizzes", headers=teacher.auth, json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _publish(client, teacher, quiz_id):
    response = client.post(f"/quizzes/{quiz_id}/publish", headers=teacher.auth)
    assert response.status_code == 200, response.text
    return response.json()


# --- Create -----------------------------------------------------------


def test_teacher_can_create_a_quiz(client, teacher, mcq_quiz_payload):
    quiz = _create_quiz(client, teacher, mcq_quiz_payload)

    assert quiz["title"] == mcq_quiz_payload["title"]
    assert quiz["status"] == "DRAFT", "a newly created quiz must start as a draft"
    assert len(quiz["questions"]) == 1
    assert len(quiz["questions"][0]["options"]) == 4


def test_created_quiz_starts_unpublished_and_hidden_from_students(
    client, teacher, student, mcq_quiz_payload
):
    """GET /quizzes for a student only returns PUBLISHED quizzes
    (get_quizzes in routes/quiz.py) - a draft must not leak to students before the teacher
    chooses to publish it."""

    quiz = _create_quiz(client, teacher, mcq_quiz_payload)

    response = client.get("/quizzes", headers=student.auth)
    assert response.status_code == 200
    visible_ids = {q["id"] for q in response.json()}
    assert quiz["id"] not in visible_ids


# --- Publish ------------------------------------------------------------


def test_publish_requires_at_least_one_question(client, teacher):
    empty_quiz = _create_quiz(
        client, teacher, {"title": "Empty quiz", "description": None, "questions": []}
    )

    response = client.post(f"/quizzes/{empty_quiz['id']}/publish", headers=teacher.auth)
    assert response.status_code == 400, (
        "publishing a quiz with zero questions should be rejected"
    )


def test_published_quiz_becomes_visible_to_students(client, teacher, student, mcq_quiz_payload):
    quiz = _create_quiz(client, teacher, mcq_quiz_payload)
    _publish(client, teacher, quiz["id"])

    response = client.get("/quizzes", headers=student.auth)
    assert response.status_code == 200
    visible_ids = {q["id"] for q in response.json()}
    assert quiz["id"] in visible_ids


def test_student_view_of_a_published_quiz_hides_correct_answers(
    client, teacher, student, mcq_quiz_payload
):
    """QuestionResponse (student-facing) omits isCorrect; only
    QuestionOwnerResponse (teacher-facing) includes it (quiz.py schemas).
    A student who can see which option is correct could trivially cheat."""

    quiz = _create_quiz(client, teacher, mcq_quiz_payload)
    _publish(client, teacher, quiz["id"])

    response = client.get(f"/quizzes/{quiz['id']}", headers=student.auth)
    assert response.status_code == 200

    for option in response.json()["questions"][0]["options"]:
        assert "isCorrect" not in option, (
            "a student-facing quiz response leaked which MCQ option is correct"
        )


# --- Submit ---------------------------------------------------------------


def test_student_can_submit_an_mcq_answer(client, teacher, student, mcq_quiz_payload):
    quiz = _create_quiz(client, teacher, mcq_quiz_payload)
    quiz = _publish(client, teacher, quiz["id"])
    question = quiz["questions"][0]
    correct_option = next(o for o in question["options"] if o.get("isCorrect"))

    response = client.post(
        f"/quizzes/{quiz['id']}/submit",
        headers=student.auth,
        json={
            "answers": [
                {"questionId": question["id"], "selectedOptionId": correct_option["id"]}
            ]
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["quizId"] == quiz["id"]
    assert body["answers"][0]["selectedOptionId"] == correct_option["id"]


def test_cannot_submit_to_an_unpublished_quiz(client, teacher, student, mcq_quiz_payload):
    quiz = _create_quiz(client, teacher, mcq_quiz_payload)  # never published

    response = client.post(
        f"/quizzes/{quiz['id']}/submit", headers=student.auth, json={"answers": []}
    )
    assert response.status_code == 404


def test_mcq_submission_rejects_an_option_from_a_different_question(
    client, teacher, student
):
    """quiz.py checks the selected option actually belongs to the question
    being answered - submitting a plausible-looking but foreign option id
    must fail rather than silently record a nonsensical answer."""

    def _four_options(correct_text):
        return [
            {"text": correct_text, "isCorrect": True},
            {"text": "distractor 1", "isCorrect": False},
            {"text": "distractor 2", "isCorrect": False},
            {"text": "distractor 3", "isCorrect": False},
        ]

    two_question_payload = {
        "title": "Two questions",
        "questions": [
            {"text": "Question A", "type": "MCQ", "options": _four_options("A-correct")},
            {"text": "Question B", "type": "MCQ", "options": _four_options("B-correct")},
        ],
    }
    quiz = _create_quiz(client, teacher, two_question_payload)
    quiz = _publish(client, teacher, quiz["id"])

    question_a = quiz["questions"][0]
    option_from_b = quiz["questions"][1]["options"][0]

    response = client.post(
        f"/quizzes/{quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question_a["id"], "selectedOptionId": option_from_b["id"]}]},
    )

    assert response.status_code == 400, (
        "an option belonging to a different question was accepted as an answer"
    )


def test_resubmitting_updates_the_same_submission_not_a_duplicate(
    client, teacher, student, mcq_quiz_payload
):
    """A student who submits, then changes their answer and submits again,
    must update their one submission - not create a second one. Otherwise
    a teacher's marking queue would show duplicate entries per student."""

    quiz = _create_quiz(client, teacher, mcq_quiz_payload)
    quiz = _publish(client, teacher, quiz["id"])
    question = quiz["questions"][0]
    option_a, option_b = question["options"][0], question["options"][1]

    first = client.post(
        f"/quizzes/{quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question["id"], "selectedOptionId": option_a["id"]}]},
    )
    second = client.post(
        f"/quizzes/{quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question["id"], "selectedOptionId": option_b["id"]}]},
    )

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["id"] == second.json()["id"], (
        "resubmitting created a new submission instead of updating the existing one"
    )
    assert second.json()["answers"][0]["selectedOptionId"] == option_b["id"]


# --- Review ---------------------------------------------------------------


def test_teacher_can_review_a_submission(client, teacher, student, mcq_quiz_payload):
    quiz = _create_quiz(client, teacher, mcq_quiz_payload)
    quiz = _publish(client, teacher, quiz["id"])
    question = quiz["questions"][0]
    option = question["options"][0]

    submission = client.post(
        f"/quizzes/{quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question["id"], "selectedOptionId": option["id"]}]},
    ).json()

    response = client.patch(
        f"/submissions/{submission['id']}/review",
        headers=teacher.auth,
        json={"mark": 85, "feedback": "Good work"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mark"] == 85
    assert body["feedback"] == "Good work"
    assert body["status"] == "REVIEWED"


def test_teacher_cannot_review_a_submission_for_another_teachers_quiz(
    client, teacher, other_teacher, student, mcq_quiz_payload
):
    """review_submission (routes/quiz.py) checks `quiz.created_by != current_user.user_id` -
    without it, any teacher could mark (and see) any other teacher's
    students' work."""

    quiz = _create_quiz(client, teacher, mcq_quiz_payload)
    quiz = _publish(client, teacher, quiz["id"])
    question = quiz["questions"][0]
    option = question["options"][0]

    submission = client.post(
        f"/quizzes/{quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question["id"], "selectedOptionId": option["id"]}]},
    ).json()

    response = client.patch(
        f"/submissions/{submission['id']}/review",
        headers=other_teacher.auth,
        json={"mark": 100, "feedback": "should not be allowed"},
    )

    assert response.status_code == 403, (
        f"a teacher was allowed to review a submission for a quiz they don't "
        f"own (got {response.status_code})"
    )


# --- Spoken answers (previously the untested half of submit) ----------------
#
# A SPOKEN answer does not carry text - it references a Transcript the
# student already produced (via the upload or live-streaming path). The route
# checks that transcript exists AND belongs to the submitting student.


def _make_answer_transcript(owner_id):
    from uuid import uuid4

    from app.db.session import SessionLocal
    from app.models.transcription import Transcript

    transcript_id = uuid4()
    with SessionLocal.begin() as db:
        db.add(
            Transcript(
                transcript_id=transcript_id,
                owner_id=owner_id,
                title=f"Quiz answer {transcript_id}",
                transcript_type="QUIZ_ANSWER",
                status="DRAFT",
            )
        )
    return transcript_id


@pytest.fixture
def spoken_quiz(client, teacher):
    payload = {
        "title": "Spoken quiz",
        "questions": [{"text": "Describe your day.", "type": "SPOKEN", "required": True}],
    }
    quiz = _create_quiz(client, teacher, payload)
    return _publish(client, teacher, quiz["id"])


def test_student_can_submit_a_spoken_answer(client, student, spoken_quiz):
    transcript_id = _make_answer_transcript(student.user_id)
    question = spoken_quiz["questions"][0]

    response = client.post(
        f"/quizzes/{spoken_quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question["id"], "transcriptId": str(transcript_id)}]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["answers"][0]["type"] == "SPOKEN"


def test_spoken_answer_cannot_reference_another_students_transcript(
    client, student, other_student, spoken_quiz
):
    """Without the ownership check a student could submit someone else's
    recording as their own answer."""

    someone_elses = _make_answer_transcript(other_student.user_id)
    question = spoken_quiz["questions"][0]

    response = client.post(
        f"/quizzes/{spoken_quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question["id"], "transcriptId": str(someone_elses)}]},
    )

    assert response.status_code == 400


def test_spoken_answer_requires_a_transcript_id(client, student, spoken_quiz):
    question = spoken_quiz["questions"][0]

    response = client.post(
        f"/quizzes/{spoken_quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question["id"]}]},
    )

    assert response.status_code == 400


def test_spoken_answer_rejects_a_transcript_that_does_not_exist(client, student, spoken_quiz):
    from uuid import uuid4

    question = spoken_quiz["questions"][0]

    response = client.post(
        f"/quizzes/{spoken_quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question["id"], "transcriptId": str(uuid4())}]},
    )

    assert response.status_code == 400


def test_mcq_answer_requires_a_selected_option(client, teacher, student, mcq_quiz_payload):
    quiz = _create_quiz(client, teacher, mcq_quiz_payload)
    quiz = _publish(client, teacher, quiz["id"])
    question = quiz["questions"][0]

    response = client.post(
        f"/quizzes/{quiz['id']}/submit",
        headers=student.auth,
        json={"answers": [{"questionId": question["id"]}]},
    )

    assert response.status_code == 400
