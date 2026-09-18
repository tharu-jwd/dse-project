"""Integration-style tests for the delete/stop voice-command branch inside
the streaming route, without a real model, database, or WebSocket.

`_emit_final` only needs an object with an async `send_json`, plus the two
persistence functions it calls - both are monkeypatched here so this stays
a fast, deterministic unit test of the routing logic itself: does a
command-shaped utterance skip persistence and notify the client, while an
ordinary utterance is still saved as dictated text.
"""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.routes import streaming as streaming_route
from app.streaming.buffer import StreamingBuffer
from app.streaming.commands import COMMANDS
from app.streaming.inference import TranscriptionResult
from app.streaming.vad import VadResult


def _phrase(command_id: str) -> str:
    return next(c for c in COMMANDS if c.id == command_id).phrase


# A session that has already heard the wake word and never times out -
# lets the tests below exercise command routing without the wake gate.
ARMED = float("inf")


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_delete_command_removes_last_segment_instead_of_persisting(monkeypatch):
    deleted_for = []
    persisted = []

    monkeypatch.setattr(
        streaming_route,
        "delete_last_segment",
        lambda transcript_id: deleted_for.append(transcript_id) or 0,
    )
    monkeypatch.setattr(
        streaming_route,
        "add_final_segment",
        lambda *args, **kwargs: persisted.append(args),
    )

    websocket = FakeWebSocket()
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    buffer.append(b"\x00\x00" * 1600)  # silence, only used for timing here
    transcript_id = uuid4()
    state = {"segment_order": 0, "bank": {}, "mode": "NOTE", "language": "si", "armed_until": ARMED}

    await streaming_route._emit_final(
        websocket, buffer, state, transcript_id, _phrase("delete"), None, None
    )

    assert deleted_for == [transcript_id]
    assert persisted == []
    assert websocket.sent == [{"type": "command", "command": "delete"}]
    # A command utterance must not consume a segment-order slot.
    assert state["segment_order"] == 0


@pytest.mark.asyncio
async def test_stop_command_notifies_client_without_persisting(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        streaming_route,
        "add_final_segment",
        lambda *args, **kwargs: persisted.append(args),
    )
    monkeypatch.setattr(
        streaming_route,
        "delete_last_segment",
        lambda transcript_id: (_ for _ in ()).throw(
            AssertionError("stop must not delete a segment")
        ),
    )

    websocket = FakeWebSocket()
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    buffer.append(b"\x00\x00" * 1600)
    state = {"segment_order": 0, "bank": {}, "mode": "NOTE", "language": "si", "armed_until": ARMED}

    await streaming_route._emit_final(
        websocket, buffer, state, uuid4(), _phrase("stop"), None, None
    )

    assert persisted == []
    assert websocket.sent[0]["type"] == "command"
    assert websocket.sent[0]["command"] == "stop"


@pytest.mark.asyncio
async def test_ordinary_dictation_is_still_persisted_normally(monkeypatch):
    persisted = []
    monkeypatch.setattr(
        streaming_route,
        "add_final_segment",
        lambda *args, **kwargs: persisted.append(args),
    )
    monkeypatch.setattr(
        streaming_route,
        "delete_last_segment",
        lambda transcript_id: (_ for _ in ()).throw(
            AssertionError("ordinary dictation must not trigger a delete")
        ),
    )

    websocket = FakeWebSocket()
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    buffer.append(b"\x00\x00" * 1600)
    state = {"segment_order": 0, "bank": {}, "mode": "NOTE", "language": "si", "armed_until": ARMED}

    await streaming_route._emit_final(
        websocket, buffer, state, uuid4(), "අද කාලගුණය හොඳයි", None, None
    )

    assert len(persisted) == 1
    assert websocket.sent[0]["type"] == "final"
    assert state["segment_order"] == 1


@pytest.mark.asyncio
async def test_partial_dictation_never_invokes_embedding_matching(monkeypatch):
    """Embedding matching only ever runs on a *finalized* command-mode
    utterance (inside _emit_final/_resolve_and_dispatch) - never on the
    provisional "partial" hypotheses _process_window sends while the
    student is still mid-utterance."""

    monkeypatch.setattr(streaming_route.settings, "voice_command_embedding_matching_enabled", True)

    embed_calls = []

    class FakeTranscriber:
        async def transcribe(self, audio, **kwargs):
            return TranscriptionResult(text="අද", avg_logprob=-0.1)

        async def embed(self, audio):
            embed_calls.append(1)
            return None

    class FakeVad:
        def analyze(self, audio):
            # has_speech, but not a pause - _process_window takes the
            # "partial" branch, not _emit_final.
            return VadResult(has_speech=True, trailing_silence_seconds=0.0)

    websocket = FakeWebSocket()
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    buffer.append(b"\x00\x00" * 1600)
    state = {"segment_order": 0, "bank": {"stop": []}, "mode": "NOTE", "language": "si", "armed_until": ARMED}  # non-empty bank, feature on

    await streaming_route._process_window(
        websocket, buffer, state, uuid4(), FakeVad(), FakeTranscriber()
    )

    assert websocket.sent == [{"type": "partial", "text": "අද", "segment": 0}]
    assert embed_calls == []


@pytest.mark.asyncio
async def test_unenrolled_student_still_gets_working_fuzzy_commands(monkeypatch):
    """Embedding matching enabled globally, but this student has no
    enrollment bank - resolve_command's fallback must still let the
    fuzzy-only "delete" command execute, and the embedding path (which
    has nothing to match against) must never even be consulted."""

    monkeypatch.setattr(streaming_route.settings, "voice_command_embedding_matching_enabled", True)

    deleted_for = []
    monkeypatch.setattr(
        streaming_route, "delete_last_segment", lambda transcript_id: deleted_for.append(transcript_id)
    )
    monkeypatch.setattr(streaming_route, "add_final_segment", lambda *args, **kwargs: None)

    embed_calls = []

    async def fake_embed(audio):
        embed_calls.append(1)
        return None

    monkeypatch.setattr(
        streaming_route, "get_streaming_transcriber", lambda: SimpleNamespace(embed=fake_embed)
    )

    websocket = FakeWebSocket()
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    buffer.append(b"\x00\x00" * 1600)
    transcript_id = uuid4()
    state = {"segment_order": 0, "bank": {}, "mode": "NOTE", "language": "si", "armed_until": ARMED}  # unenrolled

    await streaming_route._emit_final(
        websocket, buffer, state, transcript_id, _phrase("delete"), -0.1, None
    )

    assert deleted_for == [transcript_id]
    assert websocket.sent == [{"type": "command", "command": "delete"}]
    assert embed_calls == []  # empty bank short-circuits before embed() is ever called


# --- Event-driven COMMAND mode (_process_command_chunk) -------------------
#
# COMMAND mode has no tick loop: _process_command_chunk is called once per
# incoming chunk from the receive loop itself, and decides on every call
# whether to keep waiting, announce "listening", or fire the single
# transcription for the utterance that just ended.


class QueuedFakeVad:
    """Returns a pre-scripted VadResult per call, one per simulated chunk -
    lets a test describe "still speaking" then "just paused" without a
    real VAD model."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def analyze(self, audio):
        self.calls += 1
        return self._results.pop(0)


class CountingFakeTranscriber:
    def __init__(self, text="මකන්න", avg_logprob=-0.1):
        self.text = text
        self.avg_logprob = avg_logprob
        self.calls = 0

    async def transcribe(self, audio, **kwargs):
        self.calls += 1
        return TranscriptionResult(text=self.text, avg_logprob=self.avg_logprob)


@pytest.mark.asyncio
async def test_command_executes_on_speech_end_not_on_a_tick(monkeypatch):
    """Mid-utterance (has_speech, short trailing silence) must not fire
    anything - only the chunk where VAD reports a real pause finalizes
    and dispatches the command. There is no tick to wait for either way:
    _process_command_chunk is the only trigger."""

    monkeypatch.setattr(streaming_route, "add_final_segment", lambda *a, **k: None)

    vad = QueuedFakeVad(
        [
            VadResult(has_speech=True, trailing_silence_seconds=0.0),  # still speaking
            VadResult(has_speech=True, trailing_silence_seconds=0.0),  # still speaking
            VadResult(has_speech=True, trailing_silence_seconds=0.35),  # pause >= 300ms
        ]
    )
    transcriber = CountingFakeTranscriber(text=_phrase("delete"))
    websocket = FakeWebSocket()
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    state = {"segment_order": 0, "bank": {}, "mode": "COMMAND", "language": "si", "listening_sent": False, "last_command": None, "armed_until": ARMED}

    for _ in range(3):
        buffer.append(b"\x00\x00" * 1600)
        await streaming_route._process_command_chunk(
            websocket, buffer, state, uuid4(), vad, transcriber
        )

    assert transcriber.calls == 1
    assert websocket.sent == [
        {"type": "listening"},
        {"type": "command", "command": "delete"},
    ]
    # Finalizing on the pause must clear the buffer, same as dictation's pause path.
    assert buffer.is_empty


@pytest.mark.asyncio
async def test_exactly_one_transcription_per_command_utterance(monkeypatch):
    """Even a long run of mid-utterance chunks before the pause must
    still add up to a single transcribe() call - re-transcribing on
    every chunk is exactly the cost this change removes."""

    monkeypatch.setattr(streaming_route, "add_final_segment", lambda *a, **k: None)

    mid_utterance = [VadResult(has_speech=True, trailing_silence_seconds=0.0) for _ in range(6)]
    vad = QueuedFakeVad(mid_utterance + [VadResult(has_speech=True, trailing_silence_seconds=0.4)])
    transcriber = CountingFakeTranscriber(text=_phrase("stop"))
    websocket = FakeWebSocket()
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    state = {"segment_order": 0, "bank": {}, "mode": "COMMAND", "language": "si", "listening_sent": False, "last_command": None, "armed_until": ARMED}

    for _ in range(7):
        buffer.append(b"\x00\x00" * 1600)
        await streaming_route._process_command_chunk(
            websocket, buffer, state, uuid4(), vad, transcriber
        )

    assert transcriber.calls == 1


@pytest.mark.asyncio
async def test_listening_sent_once_per_utterance_before_any_transcription(monkeypatch):
    """The "listening" ack must go out the moment speech starts - before
    transcription, not after - and only once per utterance even across
    several still-speaking chunks."""

    monkeypatch.setattr(streaming_route, "add_final_segment", lambda *a, **k: None)

    vad = QueuedFakeVad(
        [
            VadResult(has_speech=True, trailing_silence_seconds=0.0),
            VadResult(has_speech=True, trailing_silence_seconds=0.0),
            VadResult(has_speech=True, trailing_silence_seconds=0.5),
        ]
    )
    transcriber = CountingFakeTranscriber(text=_phrase("next"))
    websocket = FakeWebSocket()
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    state = {"segment_order": 0, "bank": {}, "mode": "COMMAND", "language": "si", "listening_sent": False, "last_command": None, "armed_until": ARMED}

    for _ in range(3):
        buffer.append(b"\x00\x00" * 1600)
        await streaming_route._process_command_chunk(
            websocket, buffer, state, uuid4(), vad, transcriber
        )

    # Exactly one "listening" message, sent before the one "command" message.
    assert [m["type"] for m in websocket.sent] == ["listening", "command"]


@pytest.mark.asyncio
async def test_command_mode_never_sends_partial_text(monkeypatch):
    """COMMAND mode has nothing to preview - a mid-utterance chunk must
    produce no message at all, not a "partial" the way dictation does."""

    vad = QueuedFakeVad([VadResult(has_speech=True, trailing_silence_seconds=0.0)])
    transcriber = CountingFakeTranscriber()
    websocket = FakeWebSocket()
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    buffer.append(b"\x00\x00" * 1600)
    state = {"segment_order": 0, "bank": {}, "mode": "COMMAND", "language": "si", "listening_sent": True, "last_command": None}

    await streaming_route._process_command_chunk(
        websocket, buffer, state, uuid4(), vad, transcriber
    )

    assert websocket.sent == []
    assert transcriber.calls == 0


# --- Debounce ---------------------------------------------------------


@pytest.mark.asyncio
async def test_debounce_suppresses_rapid_repeat_of_same_command(monkeypatch):
    """A student unsure they were heard repeats the command - the second
    "next" within the debounce window must not fire twice."""

    clock = {"now": 100.0}
    monkeypatch.setattr(streaming_route.time, "monotonic", lambda: clock["now"])

    websocket = FakeWebSocket()
    state = {"last_command": None}

    await streaming_route._send_command(websocket, state, "next")
    clock["now"] += 0.5  # well inside the 2s debounce window
    await streaming_route._send_command(websocket, state, "next")

    assert websocket.sent == [{"type": "command", "command": "next"}]


@pytest.mark.asyncio
async def test_debounce_allows_repeat_after_the_window_elapses(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(streaming_route.time, "monotonic", lambda: clock["now"])

    websocket = FakeWebSocket()
    state = {"last_command": None}

    await streaming_route._send_command(websocket, state, "next")
    clock["now"] += 2.5  # past the 2s debounce window
    await streaming_route._send_command(websocket, state, "next")

    assert websocket.sent == [
        {"type": "command", "command": "next"},
        {"type": "command", "command": "next"},
    ]


@pytest.mark.asyncio
async def test_debounce_does_not_suppress_a_different_command(monkeypatch):
    """Only a repeat of the *same* command is debounced - "next" then
    "previous" in quick succession are two different, deliberate actions."""

    clock = {"now": 100.0}
    monkeypatch.setattr(streaming_route.time, "monotonic", lambda: clock["now"])

    websocket = FakeWebSocket()
    state = {"last_command": None}

    await streaming_route._send_command(websocket, state, "next")
    clock["now"] += 0.5
    await streaming_route._send_command(websocket, state, "previous")

    assert websocket.sent == [
        {"type": "command", "command": "next"},
        {"type": "command", "command": "previous"},
    ]


# --- Buffer clock correctness -----------------------------------------


def test_buffer_clock_stays_correct_across_mixed_finalize_and_force_cut():
    """The event-driven path still calls finalize()/force_cut() exactly
    like the tick loop did - consumed_seconds must stay an accurate
    running clock across a mix of paused (finalized) and forced (no
    pause, hit the cap) segments, same as it did for dictation."""

    buffer = StreamingBuffer(max_buffer_seconds=2.0, overlap_seconds=0.5)
    chunk = b"\x00\x00" * 8000  # 0.5s of silence per chunk at 16kHz mono 16-bit

    # First "utterance": two chunks (1.0s), then a pause -> finalize.
    buffer.append(chunk)
    buffer.append(chunk)
    assert buffer.duration_seconds == pytest.approx(1.0)
    segment = buffer.finalize("first command")
    assert segment.start == pytest.approx(0.0)
    assert segment.end == pytest.approx(1.0)
    assert buffer.consumed_seconds == pytest.approx(1.0)
    assert buffer.is_empty

    # Second "utterance": runs long with no pause, hits the 2s cap ->
    # force_cut keeps only the trailing 0.5s overlap.
    buffer.append(chunk)
    buffer.append(chunk)
    buffer.append(chunk)
    buffer.append(chunk)  # 2.0s total, >= max_buffer_seconds
    assert buffer.exceeded_max_buffer()
    buffer.force_cut()
    assert buffer.consumed_seconds == pytest.approx(1.0 + 1.5)  # kept 0.5s overlap
    assert buffer.duration_seconds == pytest.approx(0.5)

    # Third "utterance": the retained overlap (0.5s) plus one more chunk
    # (0.5s) = 1.0s, then pauses.
    buffer.append(chunk)
    segment = buffer.finalize("second command")
    assert segment.start == pytest.approx(2.5)
    assert segment.end == pytest.approx(3.5)
    assert buffer.consumed_seconds == pytest.approx(3.5)
    assert buffer.is_empty


# --- Ticker only for NOTE mode ------------------------------------------


class ReceiveQueueWebSocket:
    """Fakes the parts of a Starlette WebSocket _run_session touches:
    receive() replays a scripted message queue, send_json/close record
    what happened, close ends the queue early."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []
        self.closed = False

    async def receive(self):
        # A real yield point, so a task scheduled just before this call
        # (e.g. the ticker task) actually gets to run at least once
        # before _run_session might cancel it - plain `return` here
        # never cedes control back to the event loop at all.
        await asyncio.sleep(0)
        if not self._messages:
            return {"type": "websocket.disconnect"}
        return self._messages.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=None):
        self.closed = True


def _text_message(payload: dict) -> dict:
    return {"type": "websocket.receive", "text": json.dumps(payload)}


def _bytes_message(data: bytes) -> dict:
    return {"type": "websocket.receive", "bytes": data}


@pytest.mark.asyncio
async def test_command_mode_never_starts_the_tick_loop(monkeypatch):
    ticker_started = []

    async def spy_ticker(*args, **kwargs):
        ticker_started.append(True)
        await asyncio.sleep(3600)

    monkeypatch.setattr(streaming_route, "_window_ticker", spy_ticker)
    monkeypatch.setattr(streaming_route, "get_vad", lambda: QueuedFakeVad([]))
    monkeypatch.setattr(
        streaming_route, "get_streaming_transcriber", lambda: CountingFakeTranscriber()
    )
    monkeypatch.setattr(streaming_route.voice_enrollment, "load_bank", lambda user_id, language: {})

    websocket = ReceiveQueueWebSocket(
        [
            _text_message({"type": "start", "mode": "COMMAND"}),
            _text_message({"type": "stop"}),
        ]
    )
    user = SimpleNamespace(user_id=uuid4(), command_language="si")

    await streaming_route._run_session(websocket, user)

    assert ticker_started == []
    assert websocket.sent[-1]["type"] == "session_end"


@pytest.mark.asyncio
async def test_note_mode_still_starts_the_tick_loop(monkeypatch):
    ticker_started = []

    async def spy_ticker(*args, **kwargs):
        ticker_started.append(True)
        await asyncio.sleep(3600)

    monkeypatch.setattr(streaming_route, "_window_ticker", spy_ticker)
    monkeypatch.setattr(streaming_route, "create_live_transcript", lambda *a, **k: uuid4())
    monkeypatch.setattr(streaming_route, "get_vad", lambda: QueuedFakeVad([]))
    monkeypatch.setattr(
        streaming_route, "get_streaming_transcriber", lambda: CountingFakeTranscriber()
    )
    monkeypatch.setattr(streaming_route.voice_enrollment, "load_bank", lambda user_id, language: {})

    websocket = ReceiveQueueWebSocket(
        [
            _text_message({"type": "start", "mode": "NOTE", "title": "My note"}),
            _text_message({"type": "stop"}),
        ]
    )
    user = SimpleNamespace(user_id=uuid4(), command_language="si")

    await streaming_route._run_session(websocket, user)

    assert ticker_started == [True]
    assert websocket.sent[-1]["type"] == "session_end"


# --- Wake word gate ------------------------------------------------------


def _session(mode="COMMAND", bank=None, wake_bank=None, armed_until=None):
    return {
        "segment_order": 0,
        "bank": bank or {},
        "wake_bank": wake_bank or [],
        "mode": mode,
        "language": "si",
        "listening_sent": False,
        "last_command": None,
        "armed_until": armed_until,
    }


async def _say(websocket, state, text, transcript_id=None):
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    buffer.append(b"\x00\x00" * 1600)
    await streaming_route._emit_final(
        websocket, buffer, state, transcript_id or uuid4(), text, -0.1, None
    )


def _vec(*values):
    import numpy as np

    from app.streaming.embeddings import l2_normalize

    return l2_normalize(np.array(values, dtype=np.float32))


def _fake_embedding(monkeypatch, vector):
    async def embed(audio):
        return vector

    monkeypatch.setattr(
        streaming_route, "get_streaming_transcriber", lambda: SimpleNamespace(embed=embed)
    )


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr(streaming_route.time, "monotonic", lambda: now["t"])
    return now


@pytest.fixture
def storage(monkeypatch):
    calls = SimpleNamespace(persisted=[], deleted=[])
    monkeypatch.setattr(
        streaming_route, "add_final_segment", lambda *a, **k: calls.persisted.append(a)
    )
    monkeypatch.setattr(
        streaming_route, "delete_last_segment", lambda tid: calls.deleted.append(tid)
    )
    return calls


WINDOW = streaming_route.settings.voice_wake_window_seconds


@pytest.mark.asyncio
async def test_zimi_alone_arms_without_running_anything(clock, storage):
    websocket, state = FakeWebSocket(), _session()

    await _say(websocket, state, "zimi")

    assert websocket.sent == [{"type": "armed", "seconds": WINDOW}]
    assert state["armed_until"] == pytest.approx(100.0 + WINDOW)


@pytest.mark.asyncio
async def test_command_without_wake_word_is_ignored_in_command_mode(clock, storage):
    websocket, state = FakeWebSocket(), _session()

    await _say(websocket, state, _phrase("delete"))

    assert websocket.sent == []


@pytest.mark.asyncio
async def test_command_without_wake_word_stays_dictation_in_note_mode(clock, storage):
    websocket, state = FakeWebSocket(), _session(mode="NOTE")

    await _say(websocket, state, _phrase("delete"))

    assert storage.deleted == []
    assert len(storage.persisted) == 1
    assert websocket.sent[0]["type"] == "final"


@pytest.mark.asyncio
async def test_wake_then_command_within_window_executes(clock, storage):
    """The normal two-clip path, for a student with no recordings at all -
    the text fallback alone detects "zimi"."""

    websocket, state = FakeWebSocket(), _session()

    await _say(websocket, state, "zimi")
    clock["t"] += WINDOW - 0.5
    await _say(websocket, state, _phrase("next"))

    assert websocket.sent[-1] == {"type": "command", "command": "next"}
    assert state["armed_until"] is None


@pytest.mark.asyncio
async def test_wake_window_expires(clock, storage):
    websocket, state = FakeWebSocket(), _session()

    await _say(websocket, state, "zimi")
    clock["t"] += WINDOW + 0.5
    await _say(websocket, state, _phrase("next"))

    assert [m["type"] for m in websocket.sent] == ["armed"]


@pytest.mark.asyncio
async def test_one_wake_word_unlocks_exactly_one_command(clock, storage):
    websocket, state = FakeWebSocket(), _session()

    await _say(websocket, state, "zimi")
    await _say(websocket, state, _phrase("next"))
    await _say(websocket, state, _phrase("previous"))

    assert [m for m in websocket.sent if m["type"] == "command"] == [
        {"type": "command", "command": "next"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wake_spelling", ["zimi", "zini", "සිමි"])
async def test_wake_and_command_in_one_clip_execute_immediately(clock, storage, wake_spelling):
    websocket, state = FakeWebSocket(), _session(mode="NOTE")
    transcript_id = uuid4()

    await _say(websocket, state, f"{wake_spelling} {_phrase('delete')}", transcript_id)

    assert storage.deleted == [transcript_id]
    assert storage.persisted == []
    assert websocket.sent == [{"type": "command", "command": "delete"}]


@pytest.mark.asyncio
async def test_wake_clip_is_never_saved_as_note_text(clock, storage):
    websocket, state = FakeWebSocket(), _session(mode="NOTE")

    await _say(websocket, state, "zimi")
    await _say(websocket, state, "zimi ඉදි")  # wake + unrecognisable remainder

    assert storage.persisted == []
    assert [m["type"] for m in websocket.sent] == ["armed", "armed"]


@pytest.mark.asyncio
async def test_non_actionable_command_after_wake_is_not_saved_as_note_text(clock, storage):
    websocket, state = FakeWebSocket(), _session(mode="NOTE")

    await _say(websocket, state, f"zimi {_phrase('next')}")

    assert storage.persisted == []


@pytest.mark.asyncio
async def test_wake_detected_by_voice_when_whisper_mangles_the_word(monkeypatch, clock, storage):
    wake_voice = _vec(1.0, 0.0, 0.0)
    _fake_embedding(monkeypatch, wake_voice)
    websocket = FakeWebSocket()
    state = _session(bank={"next": [_vec(0.0, 1.0, 0.0)]}, wake_bank=[wake_voice])

    await _say(websocket, state, "ඒක හැමෝදා")

    assert websocket.sent == [{"type": "armed", "seconds": WINDOW}]
    assert state["armed_until"] is not None


@pytest.mark.asyncio
async def test_voice_closer_to_a_command_than_to_wake_does_not_arm(monkeypatch, clock, storage):
    next_voice = _vec(0.0, 1.0, 0.0)
    _fake_embedding(monkeypatch, next_voice)
    websocket = FakeWebSocket()
    state = _session(bank={"next": [next_voice]}, wake_bank=[_vec(1.0, 0.0, 0.0)])

    await _say(websocket, state, _phrase("next"))

    assert websocket.sent == []  # recognised as "next", but blocked: no wake word
    assert state["armed_until"] is None


@pytest.mark.asyncio
async def test_wake_requirement_can_be_switched_off(monkeypatch, clock, storage):
    monkeypatch.setattr(streaming_route.settings, "voice_command_wake_required", False)
    websocket, state = FakeWebSocket(), _session()

    await _say(websocket, state, _phrase("delete"))

    assert websocket.sent == [{"type": "command", "command": "delete"}]


@pytest.mark.asyncio
async def test_session_keeps_wake_samples_out_of_the_command_bank(monkeypatch):
    captured = {}

    async def spy_process(websocket, buffer, state, *args):
        captured.update(state)

    monkeypatch.setattr(streaming_route, "get_vad", lambda: QueuedFakeVad([]))
    monkeypatch.setattr(
        streaming_route, "get_streaming_transcriber", lambda: CountingFakeTranscriber()
    )
    monkeypatch.setattr(
        streaming_route.voice_enrollment,
        "load_bank",
        lambda user_id, language: {"next": ["n"], "wake": ["w"]},
    )
    monkeypatch.setattr(streaming_route, "_process_command_chunk", spy_process)

    websocket = ReceiveQueueWebSocket(
        [
            _text_message({"type": "start", "mode": "COMMAND"}),
            _bytes_message(b"\x00\x00" * 160),
            _text_message({"type": "stop"}),
        ]
    )
    await streaming_route._run_session(websocket, SimpleNamespace(user_id=uuid4(), command_language="si"))

    assert captured["bank"] == {"next": ["n"]}
    assert captured["wake_bank"] == ["w"]
