"""The streaming route's "best effort" sends must survive an abrupt client drop.

Every notification the route sends after an utterance is finalized is written to
be best-effort: the client may already be gone, and losing a notification must
never break the session's cleanup. Each of those sends is wrapped in
`try/except RuntimeError`, which covers a *graceful* close - Starlette raises
`RuntimeError("Cannot call send once a close message has been sent")` there.

It does not cover the far more common case. When the peer vanishes without a
close handshake - a dropped wifi connection, a closed tab, or the route change
that `useVoiceCommands.js` says unmounts a listening session constantly -
Starlette raises `WebSocketDisconnect(1006)` instead, which is not a
`RuntimeError`. Found while load/failover testing: the escaping exception
aborted `_finalize_remaining_buffer` and surfaced as an unhandled ASGI error.

These tests pin the intent (a dead client is never fatal) against both ways a
client can go away.
"""

import asyncio
from uuid import uuid4

import pytest
from starlette.websockets import WebSocketDisconnect

from app.api.routes import streaming as streaming_route


class DeadWebSocket:
    """A peer that is already gone. `error` is what Starlette would raise."""

    def __init__(self, error: BaseException):
        self._error = error
        self.attempts = 0

    async def send_json(self, payload):
        self.attempts += 1
        raise self._error


def _dead_peers():
    """Every way a send reports that the client is gone.

    `ClientDisconnected` is uvicorn's, raised straight out of `WebSocket.send`,
    and is an OSError rather than a RuntimeError - it was the second half of
    this bug, seen under load once the WebSocketDisconnect half was fixed.
    """

    from uvicorn.protocols.utils import ClientDisconnected

    return [
        pytest.param(WebSocketDisconnect(code=1006), id="abrupt-drop"),
        pytest.param(
            RuntimeError('Cannot call "send" once a close message has been sent.'),
            id="graceful-close",
        ),
        pytest.param(ClientDisconnected(), id="uvicorn-client-disconnected"),
        pytest.param(ConnectionResetError("peer reset"), id="connection-reset"),
    ]


@pytest.mark.parametrize("error", _dead_peers())
@pytest.mark.asyncio
async def test_send_command_survives_a_dead_client(error):
    websocket = DeadWebSocket(error)
    state = {"last_command": None}

    await streaming_route._send_command(websocket, state, "next")

    assert websocket.attempts == 1, "the send should be attempted, just not fatal"


@pytest.mark.parametrize("error", _dead_peers())
@pytest.mark.asyncio
async def test_send_command_maybe_survives_a_dead_client(error):
    decision = type("Decision", (), {"fuzzy_command_id": "save", "embedding_command_id": None})()

    await streaming_route._send_command_maybe(DeadWebSocket(error), decision)


@pytest.mark.parametrize("error", _dead_peers())
@pytest.mark.asyncio
async def test_send_armed_survives_a_dead_client(error):
    await streaming_route._send_armed(DeadWebSocket(error))


@pytest.mark.parametrize("error", _dead_peers())
@pytest.mark.asyncio
async def test_final_segment_is_persisted_even_though_the_client_is_gone(error, monkeypatch):
    """The text must reach the database whether or not anyone is listening."""

    persisted = []
    segment_id = uuid4()
    monkeypatch.setattr(
        streaming_route,
        "add_final_segment",
        lambda *args: persisted.append(args) or segment_id,
    )

    websocket = DeadWebSocket(error)
    state = {"segment_order": 0}
    segment = type("Segment", (), {"text": "hello", "start": 0.0, "end": 1.0})()

    await streaming_route._persist_and_send_final(websocket, state, uuid4(), segment)

    assert len(persisted) == 1, "the dictated text must be saved regardless"
    assert state["segment_order"] == 1


@pytest.mark.asyncio
async def test_a_dead_client_does_not_leak_the_per_user_session_slot():
    """A leaked slot would lock the student out after `streaming_max_sessions_per_user`
    abrupt drops - they could never start another session without a server restart."""

    user_id = uuid4()
    assert await streaming_route._acquire_session_slot(user_id)
    try:
        raise WebSocketDisconnect(code=1006)
    except WebSocketDisconnect:
        pass
    finally:
        await streaming_route._release_session_slot(user_id)

    assert user_id not in streaming_route._active_sessions
