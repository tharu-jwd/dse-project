import argparse
import logging
import time

from app.services.transcription_processor import process_next_job
from app.transcribers.factory import create_transcriber


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process queued SinhaSpeech transcription jobs."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one job and exit.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds to wait when the queue is empty.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    transcriber = create_transcriber()

    if arguments.once:
        processed = process_next_job(transcriber)

        if not processed:
            print("No queued transcription jobs.")

        return

    print("Transcription worker started. Press Ctrl+C to stop.")

    try:
        while True:
            processed = process_next_job(transcriber)

            if not processed:
                time.sleep(max(arguments.poll_interval, 0.5))

    except KeyboardInterrupt:
        print("\nTranscription worker stopped.")


if __name__ == "__main__":
    main()
