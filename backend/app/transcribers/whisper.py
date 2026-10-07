import logging
import subprocess
import threading
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
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

_vad_model = None  # Silero VAD, loaded on first use

# Longest window decoded in one pass - see _windows() for why this is not 30s.
MAX_WINDOW_SECONDS = 20.0


# A segment whose text compresses better than this is a repetition loop
# rather than speech - the same test Whisper itself uses.
MAX_COMPRESSION_RATIO = 2.4


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

        with self._lock:
            audio = self._decode_audio(media_path)

            # Whisper only sees 30s per forward pass, so decode one window at
            # a time and shift each window's timestamps back onto the full
            # recording. Each window is decoded ONCE: the same generate() call
            # returns the text, the timestamps and the token probabilities, so
            # nothing is decoded a second time just to score it.
            segments: list[TranscriptionSegmentResult] = []
            for window_start, window_end in self._windows(audio):
                window = audio[
                    int(window_start * SAMPLE_RATE) : int(window_end * SAMPLE_RATE)
                ]
                segments.extend(
                    self._transcribe_window(asr_pipeline, window, window_start, window_end)
                )

        text = " ".join(segment.text for segment in segments).strip()
        if not text or not segments:
            raise ValueError("Whisper did not detect any speech.")

        return TranscriptionResult(text=text, segments=segments)


    # --- DISABLED: previous transcribe(): pipeline decode, then a second
    # --- decode per segment (_convert_segments / _score_segment) for scores.
    #
#     def transcribe(self, media_path: Path) -> TranscriptionResult:
#         if not media_path.is_file():
#             raise FileNotFoundError("The uploaded media file does not exist.")
#
#         asr_pipeline = self._load_pipeline()
#         generate_kwargs = {"task": "transcribe"}
#         if self.language:
#             generate_kwargs["language"] = self.language
#
#         with self._lock:
#             # The wrapping pipeline only surfaces token ids/text - it builds
#             # its `out` dict straight from `model.generate()`'s `sequences`
#             # and drops everything else, so `output_scores` passed through
#             # `generate_kwargs` never reaches us here (see
#             # AutomaticSpeechRecognitionPipeline._forward). Getting real
#             # per-word confidence means re-decoding each chunk's own audio
#             # ourselves with `output_scores=True` - see _score_segment.
#             audio = self._decode_audio(media_path)
#
#             # Whisper only sees 30s per forward pass, and the pipeline's own
#             # `chunk_length_s` is unreliable with this checkpoint (it drops
#             # timestamps and all but the first chunk's text). Without any
#             # windowing, recordings longer than 30s were silently cut to
#             # their first 30s. So transcribe one window at a time and shift
#             # each window's timestamps back onto the full recording.
#             chunks: list[dict[str, Any]] = []
#             for window_start, window_end in self._windows(audio):
#                 window = audio[
#                     int(window_start * SAMPLE_RATE) : int(window_end * SAMPLE_RATE)
#                 ]
#                 output = asr_pipeline(
#                     {"raw": window, "sampling_rate": SAMPLE_RATE},
#                     return_timestamps=True,
#                     generate_kwargs=generate_kwargs,
#                 )
#                 for chunk in output.get("chunks", []):
#                     if not chunk.get("text", "").strip():
#                         continue
#                     start, end = chunk.get("timestamp") or (None, None)
#                     start = window_start + (start or 0.0)
#                     # No closing timestamp token (common for a window's last
#                     # chunk): it runs to the end of the window.
#                     end = window_start + end if end is not None else window_end
#                     chunks.append(
#                         {"timestamp": (start, min(end, window_end)), "text": chunk["text"]}
#                     )
#
#             output = {"chunks": chunks}
#             segments = self._convert_segments(output, asr_pipeline, audio)
#
#         text = " ".join(segment.text for segment in segments).strip()
#         if not text or not segments:
#             raise ValueError("Whisper did not detect any speech.")
#
#         return TranscriptionResult(text=text, segments=segments)
#

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
    def _speech_pauses(audio) -> list[tuple[float, float]]:
        """(start, end) in seconds of every pause between two stretches of
        speech, found with Silero VAD."""
        global _vad_model
        if _vad_model is None:
            _vad_model = load_silero_vad()

        speech = get_speech_timestamps(
            torch.from_numpy(audio),
            _vad_model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            min_silence_duration_ms=100,
        )
        return [
            (speech[i]["end"], speech[i + 1]["start"])
            for i in range(len(speech) - 1)
        ]

    @staticmethod
    def _quietest_cut(audio, start: float, max_len: float = MAX_WINDOW_SECONDS) -> float:
        """Fallback cut: the quietest 100ms inside the last 4s of the window."""
        search, frame = 4.0, int(0.1 * SAMPLE_RATE)
        lo = int((start + max_len - search) * SAMPLE_RATE)
        hi = int((start + max_len) * SAMPLE_RATE)
        region = np.abs(audio[lo:hi])
        usable = (len(region) // frame) * frame
        energy = region[:usable].reshape(-1, frame).mean(axis=1)
        return (lo + int(energy.argmin()) * frame + frame // 2) / SAMPLE_RATE

    @staticmethod
    def _windows(audio) -> list[tuple[float, float]]:
        """Split a recording into windows of at most MAX_WINDOW_SECONDS,
        cutting only where the speaker actually pauses.

        Each cut goes in the middle of the longest speech pause (>= 0.2s) found
        by VAD in the last 10s of the window. Cutting at the quietest 100ms
        instead often landed inside a sentence, and the words right after
        such a cut were dropped from the transcript. If a stretch has no
        usable pause, it falls back to the quietest 100ms."""
        total = len(audio) / SAMPLE_RATE
        # Whisper's decoder holds at most 448 tokens, and Sinhala needs about 18
        # tokens per second of speech (byte-level tokens), so a full 30s window
        # does not fit: the transcript was cut off mid-word near its end, and
        # that is the text that went missing. 20s windows fit with room to spare.
        max_len, search, min_pause = MAX_WINDOW_SECONDS, 10.0, 0.2

        try:
            pauses = WhisperTranscriber._speech_pauses(audio)
        except Exception:
            logger.exception("VAD failed; cutting windows at the quietest point instead.")
            pauses = []

        windows: list[tuple[float, float]] = []
        start = 0.0
        while total - start > max_len:
            candidates = [
                (pause_end - pause_start, (pause_start + pause_end) / 2)
                for pause_start, pause_end in pauses
                if pause_end - pause_start >= min_pause
                and start + max_len - search <= (pause_start + pause_end) / 2 <= start + max_len
            ]
            cut = (
                max(candidates)[1]
                if candidates
                else WhisperTranscriber._quietest_cut(audio, start, max_len)
            )
            windows.append((start, cut))
            start = cut
        windows.append((start, total))
        return windows

    # --- DISABLED: previous _windows(), which always cut at the quietest 100ms
    # --- in the last 4s - often in the middle of a sentence.
    #
#     def _windows(audio) -> list[tuple[float, float]]:
#         """Split a recording into windows of at most 30s (Whisper's input
#         size). Each cut is moved to the quietest 100ms inside the last 4s of
#         the window, so words aren't sliced in half where speech pauses."""
#         total = len(audio) / SAMPLE_RATE
#         max_len, search, frame = 30.0, 4.0, int(0.1 * SAMPLE_RATE)
#         windows: list[tuple[float, float]] = []
#         start = 0.0
#         while total - start > max_len:
#             lo = int((start + max_len - search) * SAMPLE_RATE)
#             hi = int((start + max_len) * SAMPLE_RATE)
#             region = np.abs(audio[lo:hi])
#             usable = (len(region) // frame) * frame
#             energy = region[:usable].reshape(-1, frame).mean(axis=1)
#             cut = (lo + int(energy.argmin()) * frame + frame // 2) / SAMPLE_RATE
#             windows.append((start, cut))
#             start = cut
#         windows.append((start, total))
#         return windows

    def _transcribe_window(
        self,
        asr_pipeline: Any,
        window,
        window_start: float,
        window_end: float,
    ) -> list[TranscriptionSegmentResult]:
        """Decode one <=30s window in a single generate() call and turn the
        result into scored segments.

        The generated sequence is [<|t0|> text tokens <|t1|>] [<|t1|> text
        tokens <|t2|>] ...: timestamp tokens delimit the segments. Every
        generated token's probability is read from the same call's scores, so
        the text, segment confidence and per-word confidence cannot drift out
        of sync with each other.
        """

        model = asr_pipeline.model
        feature_extractor = asr_pipeline.feature_extractor
        tokenizer = asr_pipeline.tokenizer

        if window.size == 0:
            return []

        inputs = feature_extractor(window, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)

        generate_kwargs = {
            "task": "transcribe",
            "return_timestamps": True,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if self.language:
            generate_kwargs["language"] = self.language

        with torch.no_grad():
            outputs = model.generate(input_features=input_features, **generate_kwargs)

        # With return_timestamps=True, generate() returns a dict whose
        # "segments" hold, per 30s window, the generated token ids and one
        # score tensor (vocab,) per generated token. Older versions returned a
        # plain output object with .sequences / .scores instead; handle both.
        if isinstance(outputs, dict):
            decoded = [
                (hf_segment["tokens"].tolist(), hf_segment["result"]["scores"])
                for hf_segment in outputs["segments"][0]
            ]
        else:
            step_scores = outputs.scores  # one (1, vocab) tensor per step
            decoded = (
                [(outputs.sequences[0][-len(step_scores) :].tolist(), [s[0] for s in step_scores])]
                if step_scores
                else []
            )

        generated_ids: list[int] = []
        token_probs: list[float] = []
        for tokens, step_scores in decoded:
            # scores[i] is the distribution the i-th generated token was
            # chosen from, so its probability is that token's softmax value.
            for step_logits, token_id in zip(step_scores, tokens):
                generated_ids.append(token_id)
                token_probs.append(float(step_logits.log_softmax(dim=-1)[token_id].exp()))

        if not generated_ids:
            return []

        timestamp_begin = tokenizer.convert_tokens_to_ids("<|0.00|>")
        eos_token_id = tokenizer.eos_token_id

        def seconds(token_id: int) -> float:
            return (token_id - timestamp_begin) * 0.02

        # Split on timestamp tokens into (start, end, token indices) spans.
        spans: list[tuple[float, float | None, list[int]]] = []
        current_start = 0.0
        current: list[int] = []
        last_stamp = -1
        for index, token_id in enumerate(generated_ids):
            if token_id == eos_token_id:
                break
            if token_id >= timestamp_begin:
                if token_id < last_stamp:
                    # Timestamps went backwards: the model started over, and
                    # everything after this point is a repetition loop.
                    break
                last_stamp = token_id
                if current:
                    spans.append((current_start, seconds(token_id), current))
                    current = []
                current_start = seconds(token_id)
            else:
                current.append(index)
        if current:
            # No closing timestamp token (common for a window's last segment):
            # it runs to the end of the window.
            spans.append((current_start, None, current))

        results: list[TranscriptionSegmentResult] = []
        for start, end, indices in spans:
            ids = [generated_ids[i] for i in indices]
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            if not text:
                continue

            raw = text.encode("utf-8")
            if len(raw) > 200 and len(raw) / len(zlib.compress(raw)) > MAX_COMPRESSION_RATIO:
                logger.warning("Dropped a repetition-loop segment (%.1fs).", window_start + start)
                continue

            probs = [token_probs[i] for i in indices]
            words_text, _word_tokens, token_indices = _combine_tokens_into_words(
                tokenizer, ids, language=self.language
            )

            words: list[TranscriptionWord] = []
            for word_text, word_indices in zip(words_text, token_indices):
                clean = word_text.strip()
                if not clean:
                    continue
                word_probs = [probs[i] for i in word_indices if i < len(probs)]
                if not word_probs:
                    continue
                words.append(
                    TranscriptionWord(
                        text=clean,
                        confidence=sum(word_probs) / len(word_probs),
                    )
                )

            absolute_start = window_start + start
            absolute_end = window_start + end if end is not None else window_end
            results.append(
                TranscriptionSegmentResult(
                    start=absolute_start,
                    end=max(absolute_start, min(absolute_end, window_end)),
                    text=text,
                    confidence=sum(probs) / len(probs),
                    words=words,
                )
            )

        return results


    # --- DISABLED: previous _convert_segments() and _score_segment(), which
    # --- re-decoded every segment a second time just to get confidence scores.
    # --- Replaced by _transcribe_window() above (one decode per window).
    #
#     def _convert_segments(
#         self,
#         output: dict[str, Any],
#         asr_pipeline: Any,
#         audio,
#     ) -> list[TranscriptionSegmentResult]:
#         segments: list[TranscriptionSegmentResult] = []
#         audio_duration = len(audio) / SAMPLE_RATE
#
#         chunks = [
#             chunk
#             for chunk in output.get("chunks", [])
#             if chunk.get("timestamp")
#             and chunk["timestamp"][0] is not None
#             and chunk.get("text", "").strip()
#         ]
#
#         for index, chunk in enumerate(chunks):
#             timestamp = chunk["timestamp"]
#             chunk_text = chunk["text"].strip()
#
#             start = float(timestamp[0])
#             end = timestamp[1]
#             if end is None:
#                 # No closing timestamp token for this chunk (common on the
#                 # last chunk, or when generation stops early) - falling
#                 # back to `end = start` would hand _score_segment a
#                 # zero-length audio slice, which always yields empty
#                 # confidence/words. Use the next chunk's start, or the end
#                 # of the audio, so there is real audio left to re-score.
#                 end = chunks[index + 1]["timestamp"][0] if index + 1 < len(chunks) else audio_duration
#             end = max(start, float(end))
#
#             text, confidence, words = chunk_text, 0.0, []
#             try:
#                 scored_text, confidence, words = self._score_segment(
#                     asr_pipeline, audio, start, end
#                 )
#                 if scored_text:
#                     text = scored_text
#             except Exception:
#                 # A scoring failure loses confidence data for this one
#                 # segment, not the whole job - fall back to the pipeline's
#                 # own text with no confidence, same as before this existed.
#                 logger.exception(
#                     "Could not score confidence for segment %.2f-%.2f; "
#                     "keeping its text without a confidence score.",
#                     start,
#                     end,
#                 )
#
#             segments.append(
#                 TranscriptionSegmentResult(
#                     start=start,
#                     end=end,
#                     text=text,
#                     confidence=confidence,
#                     words=words,
#                 )
#             )
#
#         return segments
#
#     def _score_segment(
#         self,
#         asr_pipeline: Any,
#         audio,
#         start: float,
#         end: float,
#     ) -> tuple[str, float, list[TranscriptionWord]]:
#         """Re-decode one segment's own audio slice with `output_scores=True`
#         so its text, its overall confidence, and each word's confidence all
#         come from the same generation call - never a teacher-forced re-score
#         of text decided elsewhere, which could silently drift out of sync
#         with what's displayed. Each slice is a few seconds at most, well
#         under Whisper's 30s window, so this needs no long-form chunking of
#         its own.
#         """
#
#         model = asr_pipeline.model
#         feature_extractor = asr_pipeline.feature_extractor
#         tokenizer = asr_pipeline.tokenizer
#
#         start_sample = max(0, int(start * SAMPLE_RATE))
#         end_sample = min(len(audio), int(end * SAMPLE_RATE))
#         clip = audio[start_sample:end_sample]
#
#         if clip.size == 0:
#             return "", 0.0, []
#
#         inputs = feature_extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt")
#         input_features = inputs.input_features.to(model.device)
#
#         generate_kwargs = {
#             "task": "transcribe",
#             "output_scores": True,
#             "return_dict_in_generate": True,
#         }
#         if self.language:
#             generate_kwargs["language"] = self.language
#
#         with torch.no_grad():
#             outputs = model.generate(input_features=input_features, **generate_kwargs)
#
#         scores = outputs.scores  # one (1, vocab) tensor per freely-generated step
#         if not scores:
#             text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
#             return text, 0.0, []
#
#         # `sequences` is [forced prompt tokens..., generated tokens...]; the
#         # forced prefix (start-of-transcript/language/task/no-timestamps)
#         # isn't in `scores` at all, so the tail of length len(scores) is
#         # exactly the tokens the model actually chose - this holds
#         # regardless of how many forced tokens preceded them.
#         generated_ids = outputs.sequences[0][-len(scores) :].tolist()
#
#         # Whisper appends exactly one EOS at the end of free generation;
#         # _combine_tokens_into_words decodes with special tokens included
#         # (it needs exact byte alignment), so it must be dropped first or
#         # it shows up as a spurious trailing "word".
#         if generated_ids and generated_ids[-1] == tokenizer.eos_token_id:
#             generated_ids = generated_ids[:-1]
#             scores = scores[:-1]
#
#         if not generated_ids:
#             return "", 0.0, []
#
#         token_probs = [
#             float(step_logits[0].log_softmax(dim=-1)[token_id].exp())
#             for step_logits, token_id in zip(scores, generated_ids)
#         ]
#
#         text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
#         if not text:
#             return "", 0.0, []
#
#         words_text, _word_tokens, token_indices = _combine_tokens_into_words(
#             tokenizer, generated_ids, language=self.language
#         )
#
#         words: list[TranscriptionWord] = []
#         for word_text, indices in zip(words_text, token_indices):
#             clean = word_text.strip()
#             if not clean:
#                 continue
#             word_probs = [token_probs[i] for i in indices if i < len(token_probs)]
#             if not word_probs:
#                 continue
#             words.append(
#                 TranscriptionWord(
#                     text=clean,
#                     confidence=sum(word_probs) / len(word_probs),
#                 )
#             )
#
#         segment_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
#
#         return text, segment_confidence, words
#

# =============================================================================
# DISABLED: faster-whisper (CTranslate2) implementation of the batch
# transcriber, kept for reference. It was about 20x faster on this machine and
# took each word's confidence from a single decode pass, but accuracy dropped,
# so the transformers implementation above is the active one again. To use it
# again, comment out the class above and uncomment this one (and its imports).
# =============================================================================
#
# import logging
# import math
# import os
# import subprocess
# import threading
# from pathlib import Path
#
# import numpy as np
# from faster_whisper import WhisperModel
#
# from app.transcribers.base import (
#     TranscriptionResult,
#     TranscriptionSegmentResult,
#     TranscriptionWord,
# )
#
# logger = logging.getLogger(__name__)
#
# SAMPLE_RATE = 16_000
#
# # Batch (lecture) transcription runs on faster-whisper. On this machine the
# # GPU is power-limited, and faster-whisper int8 measured the same speed on the
# # CPU and the GPU (about 0.3x real time) - so the CPU is used, which leaves the
# # GPU free for live dictation and voice commands.
# DEVICE = "cpu"
# COMPUTE_TYPE = "int8"
#
#
# class WhisperTranscriber:
#     """Local transcription with faster-whisper (CTranslate2) on a converted
#     copy of the fine-tuned checkpoint.
#
#     Per-word confidence comes straight out of the one decode pass
#     (`word_timestamps=True` returns each word's probability), so nothing is
#     decoded a second time to score it - see the disabled transformers
#     implementation below, which re-decoded every segment for that.
#     """
#
#     def __init__(self, model_name: str, language: str = "si") -> None:
#         self.model_name = model_name
#         self.language = language
#         self._model: WhisperModel | None = None
#         self._lock = threading.Lock()
#
#     def _model_dir(self) -> str:
#         """The CTranslate2 model directory: `model_name` itself when it already
#         is one, otherwise the converted sibling `<model_name>-ct2`."""
#
#         path = Path(self.model_name)
#         if (path / "model.bin").is_file():
#             return str(path)
#
#         converted = path.parent / f"{path.name}-ct2"
#         if (converted / "model.bin").is_file():
#             return str(converted)
#
#         raise FileNotFoundError(
#             "No CTranslate2 model was found for faster-whisper at "
#             f"{path} or {converted}."
#         )
#
#     def _load_model(self) -> WhisperModel:
#         if self._model is None:
#             self._model = WhisperModel(
#                 self._model_dir(),
#                 device=DEVICE,
#                 compute_type=COMPUTE_TYPE,
#                 cpu_threads=os.cpu_count() or 4,
#             )
#         return self._model
#
#     def transcribe(self, media_path: Path) -> TranscriptionResult:
#         if not media_path.is_file():
#             raise FileNotFoundError("The uploaded media file does not exist.")
#
#         model = self._load_model()
#
#         with self._lock:
#             audio = self._decode_audio(media_path)
#
#             # faster-whisper slides its own 30s window over the recording, so
#             # no manual windowing is needed. `word_timestamps=True` is what
#             # makes it return a probability for every word.
#             raw_segments, _info = model.transcribe(
#                 audio,
#                 language=self.language or None,
#                 task="transcribe",
#                 beam_size=1,
#                 word_timestamps=True,
#                 vad_filter=True,
#                 condition_on_previous_text=False,
#             )
#
#             segments: list[TranscriptionSegmentResult] = []
#             for raw in raw_segments:
#                 text = raw.text.strip()
#                 if not text:
#                     continue
#
#                 words = [
#                     TranscriptionWord(
#                         text=word.word.strip(),
#                         confidence=min(1.0, max(0.0, float(word.probability))),
#                     )
#                     for word in (raw.words or [])
#                     if word.word.strip()
#                 ]
#
#                 if words:
#                     confidence = sum(word.confidence for word in words) / len(words)
#                 else:
#                     # No word data for this segment: fall back to the
#                     # segment's own average log-probability.
#                     confidence = math.exp(min(0.0, float(raw.avg_logprob)))
#
#                 start = max(0.0, float(raw.start))
#                 segments.append(
#                     TranscriptionSegmentResult(
#                         start=start,
#                         end=max(start, float(raw.end)),
#                         text=text,
#                         confidence=min(1.0, max(0.0, confidence)),
#                         words=words,
#                     )
#                 )
#
#         text = " ".join(segment.text for segment in segments).strip()
#         if not text or not segments:
#             raise ValueError("Whisper did not detect any speech.")
#
#         return TranscriptionResult(text=text, segments=segments)
#
#     @staticmethod
#     def _decode_audio(media_path: Path) -> np.ndarray:
#         """Decode any media file to 16kHz mono float32 with ffmpeg, reading it
#         by path. Piping the bytes through stdin fails for .m4a/.mp4 files whose
#         index sits at the end of the file, because stdin can't be seeked."""
#         result = subprocess.run(
#             [
#                 "ffmpeg", "-v", "error", "-i", str(media_path),
#                 "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
#             ],
#             capture_output=True,
#         )
#         if result.returncode != 0 or not result.stdout:
#             raise ValueError(
#                 "ffmpeg could not decode the media file: "
#                 + result.stderr.decode(errors="replace")[-300:]
#             )
#         return np.frombuffer(result.stdout, dtype=np.float32)
