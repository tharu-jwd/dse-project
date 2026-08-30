"""Integration-style tests for the delete/stop voice-command branch inside
the streaming route, without a real model, database, or WebSocket.

`_emit_final` only needs an object with an async `send_json`, plus the two
persistence functions it calls - both are monkeypatched here so this stays
a fast, deterministic unit test of the routing logic itself: does a
command-shaped utterance skip persistence and notify the client, while an
ordinary utterance is still saved as dictated text.
"""

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
    state = {"segment_order": 0, "bank": {}}

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
    state = {"segment_order": 0, "bank": {}}

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
    state = {"segment_order": 0, "bank": {}}

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
    state = {"segment_order": 0, "bank": {"stop": []}}  # non-empty bank, feature on

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
    state = {"segment_order": 0, "bank": {}}  # unenrolled

    await streaming_route._emit_final(
        websocket, buffer, state, transcript_id, _phrase("delete"), -0.1, None
    )

    assert deleted_for == [transcript_id]
    assert websocket.sent == [{"type": "command", "command": "delete"}]
    assert embed_calls == []  # empty bank short-circuits before embed() is ever called
