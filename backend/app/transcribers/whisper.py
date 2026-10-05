import logging
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)
from transformers.models.whisper.tokenization_whisper import _combine_tokens_into_words

from app.transcribers.base import (
    TranscriptionResult,
    TranscriptionSegmentResult,
    TranscriptionWord,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


class WhisperTranscriber:
    """Local transcription using a complete Hugging Face Whisper checkpoint."""

    def __init__(self, model_name: str, language: str = "si") -> None:
        self.model_name = model_name
        self.language = language
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.is_dir():
            raise FileNotFoundError(
                f"The configured Whisper model directory does not exist: {model_path}"
            )

        processor = AutoProcessor.from_pretrained(self.model_name)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        # Checkpoints saved by Transformers 4.x may store a single EOS token
        # as a one-item list. Transformers 5.x beam search requires an integer.
        eos_token_id = model.generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            if len(eos_token_id) != 1:
                raise ValueError(
                    "The Whisper checkpoint defines multiple EOS token IDs, "
                    "which this transcription pipeline does not support."
                )
            model.generation_config.eos_token_id = eos_token_id[0]

        model.eval()

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model.to("cuda")

        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if use_cuda else -1,
        )
        return self._pipeline

    def transcribe(self, media_path: Path) -> TranscriptionResult:
        if not media_path.is_file():
            raise FileNotFoundError("The uploaded media file does not exist.")

        asr_pipeline = self._load_pipeline()
        generate_kwargs = {"task": "transcribe"}
        if self.language:
            generate_kwargs["language"] = self.language

        with self._lock:
            # The wrapping pipeline only surfaces token ids/text - it builds
            # its `out` dict straight from `model.generate()`'s `sequences`
            # and drops everything else, so `output_scores` passed through
            # `generate_kwargs` never reaches us here (see
            # AutomaticSpeechRecognitionPipeline._forward). Getting real
            # per-word confidence means re-decoding each chunk's own audio
            # ourselves with `output_scores=True` - see _score_segment.
            audio = self._decode_audio(media_path)

            # Whisper only sees 30s per forward pass, and the pipeline's own
            # `chunk_length_s` is unreliable with this checkpoint (it drops
            # timestamps and all but the first chunk's text). Without any
            # windowing, recordings longer than 30s were silently cut to
            # their first 30s. So transcribe one window at a time and shift
            # each window's timestamps back onto the full recording.
            chunks: list[dict[str, Any]] = []
            for window_start, window_end in self._windows(audio):
                window = audio[
                    int(window_start * SAMPLE_RATE) : int(window_end * SAMPLE_RATE)
                ]
                output = asr_pipeline(
                    {"raw": window, "sampling_rate": SAMPLE_RATE},
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs,
                )
                for chunk in output.get("chunks", []):
                    if not chunk.get("text", "").strip():
                        continue
                    start, end = chunk.get("timestamp") or (None, None)
                    start = window_start + (start or 0.0)
                    # No closing timestamp token (common for a window's last
                    # chunk): it runs to the end of the window.
                    end = window_start + end if end is not None else window_end
                    chunks.append(
                        {"timestamp": (start, min(end, window_end)), "text": chunk["text"]}
                    )

            output = {"chunks": chunks}
            segments = self._convert_segments(output, asr_pipeline, audio)

        text = " ".join(segment.text for segment in segments).strip()
        if not text or not segments:
            raise ValueError("Whisper did not detect any speech.")

        return TranscriptionResult(text=text, segments=segments)

    @staticmethod
    def _decode_audio(media_path: Path) -> np.ndarray:
        """Decode any media file to 16kHz mono float32 with ffmpeg, reading it
        by path. Piping the bytes through stdin (as transformers'
        `ffmpeg_read` does) fails for .m4a/.mp4 files whose index sits at the
        end of the file, because stdin can't be seeked."""
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(media_path),
                "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
            ],
            capture_output=True,
        )
        if result.returncode != 0 or not result.stdout:
            raise ValueError(
                "ffmpeg could not decode the media file: "
                + result.stderr.decode(errors="replace")[-300:]
            )
        return np.frombuffer(result.stdout, dtype=np.float32)

    @staticmethod
    def _windows(audio) -> list[tuple[float, float]]:
        """Split a recording into windows of at most 30s (Whisper's input
        size). Each cut is moved to the quietest 100ms inside the last 4s of
        the window, so words aren't sliced in half where speech pauses."""
        total = len(audio) / SAMPLE_RATE
        max_len, search, frame = 30.0, 4.0, int(0.1 * SAMPLE_RATE)
        windows: list[tuple[float, float]] = []
        start = 0.0
        while total - start > max_len:
            lo = int((start + max_len - search) * SAMPLE_RATE)
            hi = int((start + max_len) * SAMPLE_RATE)
            region = np.abs(audio[lo:hi])
            usable = (len(region) // frame) * frame
            energy = region[:usable].reshape(-1, frame).mean(axis=1)
            cut = (lo + int(energy.argmin()) * frame + frame // 2) / SAMPLE_RATE
            windows.append((start, cut))
            start = cut
        windows.append((start, total))
        return windows

    def _convert_segments(
        self,
        output: dict[str, Any],
        asr_pipeline: Any,
        audio,
    ) -> list[TranscriptionSegmentResult]:
        segments: list[TranscriptionSegmentResult] = []
        audio_duration = len(audio) / SAMPLE_RATE

        chunks = [
            chunk
            for chunk in output.get("chunks", [])
            if chunk.get("timestamp")
            and chunk["timestamp"][0] is not None
            and chunk.get("text", "").strip()
        ]

        for index, chunk in enumerate(chunks):
            timestamp = chunk["timestamp"]
            chunk_text = chunk["text"].strip()

            start = float(timestamp[0])
            end = timestamp[1]
            if end is None:
                # No closing timestamp token for this chunk (common on the
                # last chunk, or when generation stops early) - falling
                # back to `end = start` would hand _score_segment a
                # zero-length audio slice, which always yields empty
                # confidence/words. Use the next chunk's start, or the end
                # of the audio, so there is real audio left to re-score.
                end = chunks[index + 1]["timestamp"][0] if index + 1 < len(chunks) else audio_duration
            end = max(start, float(end))

            text, confidence, words = chunk_text, 0.0, []
            try:
                scored_text, confidence, words = self._score_segment(
                    asr_pipeline, audio, start, end
                )
                if scored_text:
                    text = scored_text
            except Exception:
                # A scoring failure loses confidence data for this one
                # segment, not the whole job - fall back to the pipeline's
                # own text with no confidence, same as before this existed.
                logger.exception(
                    "Could not score confidence for segment %.2f-%.2f; "
                    "keeping its text without a confidence score.",
                    start,
                    end,
                )

            segments.append(
                TranscriptionSegmentResult(
                    start=start,
                    end=end,
                    text=text,
                    confidence=confidence,
                    words=words,
                )
            )

        return segments

    def _score_segment(
        self,
        asr_pipeline: Any,
        audio,
        start: float,
        end: float,
    ) -> tuple[str, float, list[TranscriptionWord]]:
        """Re-decode one segment's own audio slice with `output_scores=True`
        so its text, its overall confidence, and each word's confidence all
        come from the same generation call - never a teacher-forced re-score
        of text decided elsewhere, which could silently drift out of sync
        with what's displayed. Each slice is a few seconds at most, well
        under Whisper's 30s window, so this needs no long-form chunking of
        its own.
        """

        model = asr_pipeline.model
        feature_extractor = asr_pipeline.feature_extractor
        tokenizer = asr_pipeline.tokenizer

        start_sample = max(0, int(start * SAMPLE_RATE))
        end_sample = min(len(audio), int(end * SAMPLE_RATE))
        clip = audio[start_sample:end_sample]

        if clip.size == 0:
            return "", 0.0, []

        inputs = feature_extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)

        generate_kwargs = {
            "task": "transcribe",
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if self.language:
            generate_kwargs["language"] = self.language

        with torch.no_grad():
            outputs = model.generate(input_features=input_features, **generate_kwargs)

        scores = outputs.scores  # one (1, vocab) tensor per freely-generated step
        if not scores:
            text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
            return text, 0.0, []

        # `sequences` is [forced prompt tokens..., generated tokens...]; the
        # forced prefix (start-of-transcript/language/task/no-timestamps)
        # isn't in `scores` at all, so the tail of length len(scores) is
        # exactly the tokens the model actually chose - this holds
        # regardless of how many forced tokens preceded them.
        generated_ids = outputs.sequences[0][-len(scores) :].tolist()

        # Whisper appends exactly one EOS at the end of free generation;
        # _combine_tokens_into_words decodes with special tokens included
        # (it needs exact byte alignment), so it must be dropped first or
        # it shows up as a spurious trailing "word".
        if generated_ids and generated_ids[-1] == tokenizer.eos_token_id:
            generated_ids = generated_ids[:-1]
            scores = scores[:-1]

        if not generated_ids:
            return "", 0.0, []

        token_probs = [
            float(step_logits[0].log_softmax(dim=-1)[token_id].exp())
            for step_logits, token_id in zip(scores, generated_ids)
        ]

        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            return "", 0.0, []

        words_text, _word_tokens, token_indices = _combine_tokens_into_words(
            tokenizer, generated_ids, language=self.language
        )

        words: list[TranscriptionWord] = []
        for word_text, indices in zip(words_text, token_indices):
            clean = word_text.strip()
            if not clean:
                continue
            word_probs = [token_probs[i] for i in indices if i < len(token_probs)]
            if not word_probs:
                continue
            words.append(
                TranscriptionWord(
                    text=clean,
                    confidence=sum(word_probs) / len(word_probs),
                )
            )

        segment_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

        return text, segment_confidence, words
