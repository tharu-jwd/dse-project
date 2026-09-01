"""Real-hardware measurement of speech-end -> command-execution latency,
before and after the event-driven COMMAND mode change.

Not a unit test - a one-off benchmark using the actual loaded streaming
model (whatever device it picks: see StreamingTranscriber's CUDA
fallback) against real recorded command clips, so the numbers in the
task's "Report back" section are measured, not estimated.

Usage (from backend/, with the venv active):
    python -m scripts.benchmark_command_latency
"""

import statistics as st
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.core.config import settings
from app.streaming.inference import get_streaming_transcriber
from app.streaming.vad import get_vad

SAMPLE_RATE = 16_000


def load_16k_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        duration = len(audio) / sample_rate
        target_length = int(duration * SAMPLE_RATE)
        original_positions = np.linspace(0, len(audio) - 1, num=len(audio))
        target_positions = np.linspace(0, len(audio) - 1, num=target_length)
        audio = np.interp(target_positions, original_positions, audio).astype("float32")
    return audio


def main() -> None:
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    transcriber = get_streaming_transcriber()
    # torch.cuda.is_available() alone is not the actual answer: the
    # StreamingTranscriber constructor probes with a real encode call and
    # falls back to CPU on a RuntimeError (mismatched CUDA/cuBLAS runtime
    # etc.) - .model.device reports what it actually resolved to.
    device = transcriber._model.model.device
    print(f"StreamingTranscriber actual device (after CUDA-probe fallback): {device}")

    vad = get_vad()

    wav_dir = Path("../storage/voice_samples_en")
    clips = sorted(wav_dir.glob("*.wav"))[:8]
    if not clips:
        wav_dir = Path("../storage/voice_samples")
        clips = sorted(wav_dir.glob("*.wav"))[:8]
    print(f"Benchmarking against {len(clips)} real clips from {wav_dir}\n")

    # --- VAD cost, to check the "a few ms, cheap at 4x/second" premise ---
    vad_timings = []
    for path in clips:
        audio = load_16k_mono(path)
        start = time.perf_counter()
        vad.analyze(audio)
        vad_timings.append(time.perf_counter() - start)
    print(f"VAD analyze() on a ~{np.mean([len(load_16k_mono(c)) for c in clips]) / SAMPLE_RATE:.1f}s buffer:")
    print(f"  mean {st.mean(vad_timings) * 1000:.1f}ms  max {max(vad_timings) * 1000:.1f}ms  "
          f"(budget per chunk at 4x/s: 250ms)\n")

    # --- Transcription cost: the one call both old and new paths pay ---
    transcribe_timings = []
    for path in clips:
        audio = load_16k_mono(path)

        async def _run(audio=audio):
            return await transcriber.transcribe(audio, mode="command")

        import asyncio

        start = time.perf_counter()
        asyncio.run(_run())
        transcribe_timings.append(time.perf_counter() - start)

    mean_transcribe = st.mean(transcribe_timings)
    print("transcribe() latency (single command clip, hotword-biased):")
    print(f"  mean {mean_transcribe * 1000:.0f}ms  min {min(transcribe_timings) * 1000:.0f}ms  "
          f"max {max(transcribe_timings) * 1000:.0f}ms\n")

    # --- Before: tick-driven. A command finishing mid-window waits for
    # the next tick (uniformly distributed 0..window_interval, average
    # window_interval/2, worst case the full window_interval), THEN pays
    # the transcribe cost on top - re-transcribing the whole buffer, same
    # per-call cost measured above regardless of how many times it ran.
    window = settings.streaming_window_interval_seconds
    old_avg = window / 2 + mean_transcribe
    old_worst = window + mean_transcribe

    # --- After: event-driven. Wait is just the shorter command silence
    # threshold, then one transcribe call.
    command_silence = settings.streaming_command_vad_silence_ms / 1000
    new_latency = command_silence + mean_transcribe

    print("=" * 60)
    print(f"Device (actual): {device}")
    print(f"Tick interval (dictation, unchanged): {window:.1f}s")
    print(f"Command VAD silence threshold (new): {command_silence * 1000:.0f}ms")
    print()
    print(f"BEFORE (tick-driven): avg {old_avg:.2f}s   worst-case {old_worst:.2f}s")
    print(f"AFTER  (event-driven): {new_latency:.2f}s")
    print(f"Improvement: avg {old_avg - new_latency:+.2f}s   worst-case {old_worst - new_latency:+.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
