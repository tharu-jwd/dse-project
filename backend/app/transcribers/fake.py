from pathlib import Path

from app.transcribers.base import (
    TranscriptionResult,
    TranscriptionSegmentResult,
    TranscriptionWord,
)


class FakeTranscriber:
    """Development transcriber used to test the worker pipeline."""

    def transcribe(
        self,
        media_path: Path,
    ) -> TranscriptionResult:
        if not media_path.is_file():
            raise FileNotFoundError(
                "The uploaded media file does not exist."
            )

        first_text = (
            "අද අපි දත්ත සමුදා පද්ධති පිළිබඳව ඉගෙන ගනිමු."
        )
        second_text = (
            "ප්‍රාථමික යතුරක් එක් එක් පේළිය අනන්‍යව හඳුනා ගනී."
        )

        return TranscriptionResult(
            text=f"{first_text} {second_text}",
            segments=[
                TranscriptionSegmentResult(
                    start=0.0,
                    end=5.5,
                    text=first_text,
                    confidence=0.94,
                    words=[
                        TranscriptionWord(
                            text="අද",
                            confidence=0.98,
                        ),
                        TranscriptionWord(
                            text="අපි",
                            confidence=0.97,
                        ),
                        TranscriptionWord(
                            text="දත්ත",
                            confidence=0.95,
                        ),
                        TranscriptionWord(
                            text="සමුදා",
                            confidence=0.91,
                        ),
                    ],
                ),
                TranscriptionSegmentResult(
                    start=5.5,
                    end=11.8,
                    text=second_text,
                    confidence=0.88,
                    words=[
                        TranscriptionWord(
                            text="ප්‍රාථමික",
                            confidence=0.89,
                        ),
                        TranscriptionWord(
                            text="යතුරක්",
                            confidence=0.91,
                        ),
                        TranscriptionWord(
                            text="අනන්‍යව",
                            confidence=0.84,
                        ),
                    ],
                ),
            ],
        )
