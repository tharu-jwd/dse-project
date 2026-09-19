"""Performance profiling tests (Master Test Plan §3.1.4).

These turn one-off measurements into checks that can fail a build. They are
deliberately **loose**: a budget that flakes on a busy CI runner teaches
people to ignore red builds. Each threshold is set an order of magnitude
above what was measured on a development laptop (values noted inline), so
a failure means "something got dramatically slower" - an accidental O(n^2),
a query in a loop, a model call on the wrong path - not "the runner was
busy".

What is and is not measured:
  - API latency here is IN-PROCESS (TestClient): application + database
    time, no network. Network latency to the deployed server (us-east-1) is
    real but is a deployment property, not a code property.
  - Nothing here loads a Whisper model. Model speed is measured separately
    by backend/scripts/benchmark_command_latency.py.
"""

import time
import tracemalloc
from contextlib import contextmanager

import numpy as np
import pytest
from sqlalchemy import event

from app.db.session import engine
from app.streaming.buffer import StreamingBuffer
from app.streaming.command_resolution import resolve_command
from app.streaming.commands import COMMANDS_SI, match_command
from app.streaming.embeddings import best_match, l2_normalize


def _percentile(samples, fraction):
    ordered = sorted(samples)
    return ordered[max(0, int(len(ordered) * fraction) - 1)]


def _time_ms(fn, runs, warmup=3):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


@contextmanager
def count_sql_statements():
    counter = {"n": 0}

    def _count(*_args, **_kwargs):
        counter["n"] += 1

    event.listen(engine, "before_cursor_execute", _count)
    try:
        yield counter
    finally:
        event.remove(engine, "before_cursor_execute", _count)


# --- API response times ------------------------------------------------------

# Measured locally: /health 2ms, /auth/me 8ms, /transcripts 10ms, /quizzes 26ms (p95).
API_ENDPOINTS = ["/health", "/auth/me", "/transcripts", "/quizzes"]


@pytest.mark.parametrize("path", API_ENDPOINTS)
def test_api_response_time_stays_within_budget(client, student, path):
    samples = _time_ms(lambda: client.get(path, headers=student.auth), runs=30)

    p50, p95 = _percentile(samples, 0.5), _percentile(samples, 0.95)
    assert p50 < 100, f"GET {path} median {p50:.1f}ms exceeds the 100ms budget"
    assert p95 < 250, f"GET {path} p95 {p95:.1f}ms exceeds the 250ms budget"


# --- Query counts (N+1 detection) --------------------------------------------


def _publish_quizzes(client, teacher, how_many):
    options = [{"text": f"option {i}", "isCorrect": i == 0} for i in range(4)]
    for n in range(how_many):
        body = {"title": f"perf quiz {n}", "questions": [{"text": "q", "type": "MCQ", "options": options}]}
        quiz_id = client.post("/quizzes", headers=teacher.auth, json=body).json()["id"]
        client.post(f"/quizzes/{quiz_id}/publish", headers=teacher.auth)


def _statements_for_listing(client, user):
    with count_sql_statements() as counter:
        assert client.get("/quizzes", headers=user.auth).status_code == 200
    return counter["n"]


def test_teacher_quiz_list_query_count_does_not_grow_with_quiz_count(client, teacher):
    """The teacher listing eager-loads questions and options in one pass, so
    the number of SQL statements must not depend on how many quizzes exist."""

    _publish_quizzes(client, teacher, 1)
    with_one = _statements_for_listing(client, teacher)

    _publish_quizzes(client, teacher, 11)
    with_twelve = _statements_for_listing(client, teacher)

    assert with_twelve <= with_one + 1, (
        f"teacher quiz list went from {with_one} to {with_twelve} SQL statements "
        "as quizzes grew from 1 to 12 - a query is running per quiz"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT (N+1): GET /quizzes as a student runs one extra query per "
        "published quiz - serialize_quiz_student() calls _own_submission() inside "
        "the loop (routes/quiz.py, called from the list comprehension in get_quizzes). Measured: 9, 13, 18, 28 statements for 1, 5, "
        "10, 20 quizzes (8 + N). Harmless at classroom scale on a local database, "
        "but every query is a network round trip to a managed database. Fix: load "
        "the student's submissions for all listed quizzes in a single query. "
        "strict=True means this test will start FAILING the moment the defect is "
        "fixed, forcing this marker to be removed."
    ),
)
def test_student_quiz_list_query_count_does_not_grow_with_quiz_count(client, teacher, student):
    _publish_quizzes(client, teacher, 1)
    with_one = _statements_for_listing(client, student)

    _publish_quizzes(client, teacher, 11)
    with_twelve = _statements_for_listing(client, student)

    assert with_twelve <= with_one + 1, (
        f"student quiz list went from {with_one} to {with_twelve} SQL statements "
        "as quizzes grew from 1 to 12"
    )


# --- Matching latency (runs on every finalized voice-command utterance) ---------

UTTERANCES = [c.phrase for c in COMMANDS_SI] + ["අද කාලගුණය හොඳයි", "මම උදේට නැගිටිනවා", "", "zimi"]


def _bank(commands=13, samples=5, dim=768):
    rng = np.random.default_rng(0)
    return {
        f"cmd{c}": [l2_normalize(rng.normal(size=dim).astype(np.float32)) for _ in range(samples)]
        for c in range(commands)
    }


def test_fuzzy_matching_is_fast():
    """rapidfuzz over ~12 short phrases. Measured well under 1ms."""

    samples = _time_ms(lambda: [match_command(u) for u in UTTERANCES], runs=100)
    per_call_p95 = _percentile(samples, 0.95) / len(UTTERANCES)
    assert per_call_p95 < 5, f"match_command p95 {per_call_p95:.2f}ms per utterance exceeds 5ms"


def test_embedding_matching_against_a_realistic_bank_is_fast():
    """The real enrolled bank is 13 commands x 5 samples of 768 dims = 65 vectors."""

    bank = _bank()
    query = l2_normalize(np.random.default_rng(1).normal(size=768).astype(np.float32))

    samples = _time_ms(lambda: best_match(query, bank), runs=100)
    p95 = _percentile(samples, 0.95)
    assert p95 < 20, f"best_match over 65 vectors p95 {p95:.2f}ms exceeds 20ms"


def test_full_command_resolution_is_fast():
    bank = _bank()
    query = l2_normalize(np.random.default_rng(2).normal(size=768).astype(np.float32))

    samples = _time_ms(
        lambda: resolve_command("ඊළඟට", avg_logprob=-0.1, embedding=query, bank=bank), runs=100
    )
    p95 = _percentile(samples, 0.95)
    assert p95 < 30, f"resolve_command p95 {p95:.2f}ms exceeds 30ms"


# --- Memory: the streaming buffer must stay bounded -----------------------------


def test_streaming_buffer_stays_bounded_over_a_long_session():
    """Simulates ten minutes of continuous speech that never pauses (so VAD
    never finalizes) under the route's real policy: when the buffer exceeds
    its cap it is force-cut down to the overlap. Without that the buffer -
    and the memory behind it - would grow for as long as the student talks."""

    sample_rate = 16_000
    chunk_seconds = 0.1
    chunk = (np.random.default_rng(0).integers(-3000, 3000, int(sample_rate * chunk_seconds)).astype(np.int16)).tobytes()

    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    largest_seen = 0.0

    tracemalloc.start()
    try:
        for _ in range(int(600 / chunk_seconds)):  # 10 minutes
            buffer.append(chunk)
            largest_seen = max(largest_seen, buffer.duration_seconds)
            if buffer.exceeded_max_buffer():
                buffer.force_cut()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert largest_seen <= 15.0 + chunk_seconds + 1e-6, (
        f"buffer reached {largest_seen:.1f}s - the cap is not being enforced"
    )
    # 15s of int16 mono is ~0.5MB; concatenation copies briefly. 20MB means a leak.
    assert peak_bytes < 20 * 1024 * 1024, f"peak traced memory {peak_bytes / 1e6:.1f}MB - buffer is leaking"
    assert buffer.consumed_seconds == pytest.approx(600 - buffer.duration_seconds, abs=0.2), (
        "audio was lost or double-counted across repeated force-cuts"
    )
