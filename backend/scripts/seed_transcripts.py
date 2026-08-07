from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.transcription import Transcript, TranscriptSegment
from app.models.user import User


STUDENT_EMAIL = "student@sinhaspeech.lk"
SAMPLE_TITLE = "Sample Sinhala Database Lecture"


def seed_transcript() -> None:
    with SessionLocal.begin() as db:
        student = db.scalar(
            select(User).where(User.email == STUDENT_EMAIL)
        )

        if student is None:
            raise RuntimeError(
                "Demo student does not exist. "
                "Run 'python -m scripts.seed_users' first."
            )

        existing_transcript = db.scalar(
            select(Transcript).where(
                Transcript.owner_id == student.user_id,
                Transcript.title == SAMPLE_TITLE,
            )
        )

        if existing_transcript is not None:
            print(
                "Skipped existing transcript: "
                f"{existing_transcript.transcript_id}"
            )
            return

        transcript = Transcript(
            owner_id=student.user_id,
            media_id=None,
            title=SAMPLE_TITLE,
            transcript_type="LECTURE",
            status="DRAFT",
            segments=[
                TranscriptSegment(
                    segment_order=0,
                    generated_text=(
                        "අද අපි දත්ත සමුදා පද්ධති "
                        "පිළිබඳව ඉගෙන ගනිමු."
                    ),
                    edited_text=None,
                    start_time=0.0,
                    end_time=5.5,
                    confidence=0.94,
                    word_metadata=[
                        {
                            "text": "අද",
                            "confidence": 0.98,
                        },
                        {
                            "text": "අපි",
                            "confidence": 0.97,
                        },
                        {
                            "text": "දත්ත",
                            "confidence": 0.95,
                        },
                        {
                            "text": "සමුදා",
                            "confidence": 0.91,
                        },
                        {
                            "text": "පද්ධති",
                            "confidence": 0.92,
                        },
                        {
                            "text": "පිළිබඳව",
                            "confidence": 0.89,
                        },
                        {
                            "text": "ඉගෙන",
                            "confidence": 0.96,
                        },
                        {
                            "text": "ගනිමු.",
                            "confidence": 0.95,
                        },
                    ],
                ),
                TranscriptSegment(
                    segment_order=1,
                    generated_text=(
                        "ප්‍රාථමික යතුරක් වගුවක එක් එක් "
                        "පේළිය අනන්‍යව හඳුනා ගනී."
                    ),
                    edited_text=None,
                    start_time=5.5,
                    end_time=12.0,
                    confidence=0.88,
                    word_metadata=[
                        {
                            "text": "ප්‍රාථමික",
                            "confidence": 0.89,
                        },
                        {
                            "text": "යතුරක්",
                            "confidence": 0.91,
                        },
                        {
                            "text": "වගුවක",
                            "confidence": 0.87,
                        },
                        {
                            "text": "එක්",
                            "confidence": 0.96,
                        },
                        {
                            "text": "එක්",
                            "confidence": 0.95,
                        },
                        {
                            "text": "පේළිය",
                            "confidence": 0.84,
                        },
                        {
                            "text": "අනන්‍යව",
                            "confidence": 0.82,
                        },
                        {
                            "text": "හඳුනා",
                            "confidence": 0.90,
                        },
                        {
                            "text": "ගනී.",
                            "confidence": 0.93,
                        },
                    ],
                ),
            ],
        )

        db.add(transcript)
        db.flush()

        print(f"Created transcript: {transcript.transcript_id}")


if __name__ == "__main__":
    seed_transcript()
