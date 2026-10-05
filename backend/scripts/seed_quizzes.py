from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.quiz import Question, QuestionOption, Quiz
from app.models.user import User


TEACHER_EMAIL = "teacher@sinhaspeech.lk"

DEMO_QUIZZES = [
    {
        "title": "Sri Lankan History Basics",
        "description": "A quick mixed quiz covering key facts and a short spoken reflection.",
        "due_date": None,
        "questions": [
            {
                "text": "Sri Lanka gained independence from British rule in which year?",
                "type": "MCQ",
                "options": [
                    ("1945", False),
                    ("1948", True),
                    ("1952", False),
                    ("1960", False),
                ],
            },
            {
                "text": "What is the capital of Sri Lanka (administrative capital)?",
                "type": "MCQ",
                "options": [
                    ("Colombo", False),
                    ("Kandy", False),
                    ("Sri Jayawardenepura Kotte", True),
                    ("Galle", False),
                ],
            },
            {
                "text": "In your own words, explain why preserving local languages like Sinhala matters for education.",
                "type": "SPOKEN",
                "options": [],
            },
        ],
    },
    {
        "title": "Sinhala Greetings and Daily Phrases",
        "description": "Pick the right phrase for everyday situations, then answer aloud.",
        "due_date": None,
        "questions": [
            {
                "text": "Which phrase means \"thank you\" in Sinhala?",
                "type": "MCQ",
                "options": [
                    ("ආයුබෝවන්", False),
                    ("ස්තූතියි", True),
                    ("සමාවෙන්න", False),
                    ("සුබ රාත්‍රියක්", False),
                ],
            },
            {
                "text": "Which word means \"water\" in Sinhala?",
                "type": "MCQ",
                "options": [
                    ("වතුර", True),
                    ("ගින්න", False),
                    ("පොත", False),
                    ("මාළු", False),
                ],
            },
            {
                "text": "Introduce yourself in Sinhala: say your name and what you study.",
                "type": "SPOKEN",
                "options": [],
            },
        ],
    },
    {
        "title": "Sri Lanka Geography and Nature",
        "description": "Short questions about the island, followed by a spoken reflection.",
        "due_date": None,
        "questions": [
            {
                "text": "What is the highest mountain in Sri Lanka?",
                "type": "MCQ",
                "options": [
                    ("Adam's Peak", False),
                    ("Pidurutalagala", True),
                    ("Knuckles Peak", False),
                    ("Sigiriya Rock", False),
                ],
            },
            {
                "text": "Which is the longest river in Sri Lanka?",
                "type": "MCQ",
                "options": [
                    ("Kelani River", False),
                    ("Kalu River", False),
                    ("Mahaweli River", True),
                    ("Walawe River", False),
                ],
            },
            {
                "text": "Describe your favourite place in Sri Lanka and why you like it.",
                "type": "SPOKEN",
                "options": [],
            },
        ],
    },
]


def seed_quizzes() -> None:
    with SessionLocal.begin() as db:
        teacher = db.scalar(select(User).where(User.email == TEACHER_EMAIL))
        if teacher is None:
            print(f"Teacher account {TEACHER_EMAIL} not found. Run seed_users first.")
            return

        created_count = 0
        for quiz_data in DEMO_QUIZZES:
            existing = db.scalar(
                select(Quiz).where(
                    Quiz.created_by == teacher.user_id,
                    Quiz.title == quiz_data["title"],
                )
            )
            if existing is not None:
                print(f"Skipped existing quiz: {quiz_data['title']}")
                continue

            quiz = Quiz(
                created_by=teacher.user_id,
                title=quiz_data["title"],
                description=quiz_data["description"],
                status="PUBLISHED",
                available_from=None,
                available_until=None,
            )
            db.add(quiz)
            db.flush()

            for order, question_data in enumerate(quiz_data["questions"], start=1):
                question = Question(
                    quiz_id=quiz.quiz_id,
                    question_order=order,
                    question_text=question_data["text"],
                    question_type=question_data["type"],
                    is_required=True,
                )
                db.add(question)
                db.flush()

                for opt_order, (option_text, is_correct) in enumerate(
                    question_data["options"], start=1
                ):
                    db.add(
                        QuestionOption(
                            question_id=question.question_id,
                            option_order=opt_order,
                            option_text=option_text,
                            is_correct=is_correct,
                        )
                    )

            created_count += 1
            print(f"Created quiz: {quiz_data['title']}")

    print(f"Seeding completed. Created {created_count} quiz(zes).")


if __name__ == "__main__":
    seed_quizzes()
