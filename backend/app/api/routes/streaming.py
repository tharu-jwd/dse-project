import asyncio
import json
import logging
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.api.dependencies import get_current_user_ws
from app.core.config import settings
from app.models.user import User
from app.services import voice_enrollment
from app.services.streaming_persistence import (
    add_final_segment,
    create_live_transcript,
    delete_last_segment,
)
from app.streaming.buffer import StreamingBuffer
from app.streaming.command_resolution import resolve_command
from app.streaming.embeddings import ClipTooShortError
from app.streaming.inference import get_streaming_transcriber
from app.streaming.vad import get_vad

# Voice commands that are handled while taking a live note. Anything else
# in the command vocabulary (next/previous/save/submit) has no defined
# action in this view yet, so a fuzzy match on those is left as ordinary
# dictated text rather than silently swallowed.
_ACTIONABLE_NOTE_COMMANDS = {"delete", "stop"}


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/streaming",
    tags=["Streaming"],
)

# Per-user concurrent session count. In-memory, single-process only -
# adequate while streaming runs on one worker; a multi-worker deployment
# would need a shared store (e.g. Redis) instead.
_active_sessions: dict[UUID, int] = {}
_active_sessions_lock = asyncio.Lock()

TRANSCRIPT_TYPE_BY_MODE = {
    "NOTE": "NOTE",
    "EXAM": "QUIZ_ANSWER",
}


async def _acquire_session_slot(user_id: UUID) -> bool:
    async with _active_sessions_lock:
        current = _active_sessions.get(user_id, 0)

        if current >= settings.streaming_max_sessions_per_user:
            return False

        _active_sessions[user_id] = current + 1
        return True


async def _release_session_slot(user_id: UUID) -> None:
    async with _active_sessions_lock:
        current = _active_sessions.get(user_id, 0)

        if current <= 1:
            _active_sessions.pop(user_id, None)
        else:
            _active_sessions[user_id] = current - 1


@router.websocket("/ws")
async def stream_transcription(websocket: WebSocket) -> None:
    """Live transcription WebSocket - NOTE mode.

    Protocol:
      client -> {"type": "start", "title": str, "mode": "NOTE"}
      client -> binary frames of 16kHz mono 16-bit PCM
      client -> {"type": "stop"}
      server -> {"type": "partial", "text": str, "segment": int}
      server -> {"type": "final", "text": str, "start": float, "end": float, "segment": int}
      server -> {"type": "command", "command": str}
      server -> {"type": "command_maybe", "fuzzy_command": str|null, "embedding_command": str|null}
      server -> {"type": "session_end", "transcript_id": str}
      server -> {"type": "error", "message": str}

    Every finalized utterance is checked against the voice-command
    vocabulary via app.streaming.command_resolution, which combines the
    fuzzy-text match (app.streaming.commands) with, for an enrolled
    student, a speaker-embedding match (app.streaming.embeddings) - see
    that module for the exact combination rule. Three outcomes:
      - "execute": treated as a command instead of dictated text -
        nothing is added to the note; a "command" message is sent so the
        client can act on it (today: delete the last line / stop the
        session - other recognized commands have no defined action in
        this view yet).
      - "confirm": ambiguous (the two signals disagree, or only a weak
        candidate exists on either side). Nothing is guessed - the
        utterance is persisted as ordinary note text as a safe default,
        but a "command_maybe" message is also sent so the client can
        surface it to the student.
      - "none": ordinary dictation, persisted as normal note text.

    A student with no enrollment bank (or with embedding matching
    switched off globally) gets exactly the fuzzy-only "execute"/"none"
    behaviour that existed before enrollment did - see
    resolve_command's fallback.
    """

    if not settings.streaming_enabled:
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user = get_current_user_ws(websocket)

    if user is None:
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    if not await _acquire_session_slot(user.user_id):
        await websocket.send_json(
            {
                "type": "error",
                "message": "Too many concurrent streaming sessions for this account.",
            }
        )
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    try:
        await _run_session(websocket, user)
    finally:
        await _release_session_slot(user.user_id)


async def _run_session(websocket: WebSocket, user: User) -> None:
    start_message = await _receive_start_message(websocket)

    if start_message is None:
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    title = start_message.get("title") or "Untitled note"
    mode = start_message.get("mode")

    if mode != "NOTE":
        await websocket.send_json(
            {"type": "error", "message": "Only NOTE mode is currently supported."}
        )
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    transcript_id = await asyncio.to_thread(
        create_live_transcript, user, title, TRANSCRIPT_TYPE_BY_MODE[mode]
    )

    # Loaded once per session, not per utterance - small (a handful of
    # commands x a handful of samples x one float vector each) and never
    # changes mid-session. Empty for a student who hasn't enrolled, which
    # is exactly what makes resolve_command's fallback kick in below.
    bank = (
        await asyncio.to_thread(voice_enrollment.load_bank, user.user_id)
        if settings.voice_command_embedding_matching_enabled
        else {}
    )

    buffer = StreamingBuffer(
        max_buffer_seconds=settings.streaming_max_buffer_seconds,
        overlap_seconds=settings.streaming_overlap_seconds,
        memory_ceiling_seconds=settings.streaming_memory_ceiling_seconds,
    )
    state = {"segment_order": 0, "bank": bank}

    ticker = asyncio.create_task(
        _window_ticker(websocket, buffer, state, transcript_id)
    )

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                buffer.append(data)
                continue

            text = message.get("text")
            if text is not None:
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue

                if payload.get("type") == "stop":
                    break

    except WebSocketDisconnect:
        pass
    finally:
        ticker.cancel()
        try:
            await ticker
        except asyncio.CancelledError:
            pass

        await _finalize_remaining_buffer(websocket, buffer, state, transcript_id)
        # The transcript is left in DRAFT status so the user can review, edit
        # and correct low-confidence words before finalizing it themselves.

        try:
            await websocket.send_json(
                {"type": "session_end", "transcript_id": str(transcript_id)}
            )
        except RuntimeError:
            # Socket already closed (client disconnected first) - nothing to notify.
            pass


async def _receive_start_message(websocket: WebSocket) -> dict | None:
    try:
        message = await websocket.receive()
    except WebSocketDisconnect:
        return None

    text = message.get("text")

    if text is None:
        return None

    try:
        payload = json.loads(text)
    except ValueError:
        return None

    if payload.get("type") != "start":
        return None

    return payload


async def _window_ticker(
    websocket: WebSocket,
    buffer: StreamingBuffer,
    state: dict,
    transcript_id: UUID,
) -> None:
    vad = get_vad()
    transcriber = get_streaming_transcriber()

    while True:
        await asyncio.sleep(settings.streaming_window_interval_seconds)

        if buffer.is_empty:
            continue

        try:
            await _process_window(websocket, buffer, state, transcript_id, vad, transcriber)
        except Exception:
            logger.exception("Streaming inference failed; continuing session.")
            try:
                await websocket.send_json(
                    {"type": "error", "message": "Transcription failed for this window."}
                )
            except RuntimeError:
                return


async def _process_window(
    websocket: WebSocket,
    buffer: StreamingBuffer,
    state: dict,
    transcript_id: UUID,
    vad,
    transcriber,
) -> None:
    audio = buffer.as_float32()
    vad_result = await asyncio.to_thread(vad.analyze, audio)

    if not vad_result.has_speech:
        # Whisper hallucinates on silence - skip inference entirely.
        if buffer.exceeded_max_buffer():
            buffer.force_cut()
        return

    result = await transcriber.transcribe(audio)
    text = result.text

    is_pause = (
        vad_result.trailing_silence_seconds * 1000
        >= settings.streaming_vad_silence_ms
    )
    should_force = buffer.exceeded_max_buffer()

    if is_pause or should_force:
        await _emit_final(
            websocket, buffer, state, transcript_id, text, result.avg_logprob, audio
        )
    else:
        await websocket.send_json(
            {"type": "partial", "text": text, "segment": state["segment_order"]}
        )

    if buffer.exceeded_memory_ceiling():
        logger.warning(
            "Streaming buffer exceeded memory ceiling for transcript %s; force-finalizing.",
            transcript_id,
        )


async def _emit_final(
    websocket: WebSocket,
    buffer: StreamingBuffer,
    state: dict,
    transcript_id: UUID,
    text: str,
    avg_logprob: float | None,
    audio,
) -> None:
    segment = buffer.finalize(text) if text else None

    if segment is None or not segment.text.strip():
        buffer.force_cut()
        return

    await _resolve_and_dispatch(websocket, state, transcript_id, segment, avg_logprob, audio)


async def _resolve_and_dispatch(
    websocket: WebSocket,
    state: dict,
    transcript_id: UUID,
    segment,
    avg_logprob: float | None,
    audio,
) -> None:
    bank = state["bank"]
    embedding = None

    if settings.voice_command_embedding_matching_enabled and bank:
        try:
            embedding = await get_streaming_transcriber().embed(audio)
        except ClipTooShortError:
            embedding = None

    decision = resolve_command(
        segment.text, avg_logprob=avg_logprob, embedding=embedding, bank=bank
    )

    if decision.outcome == "execute" and decision.command_id in _ACTIONABLE_NOTE_COMMANDS:
        await _handle_command(websocket, transcript_id, decision.command_id)
        return

    if decision.outcome == "confirm":
        try:
            await websocket.send_json(
                {
                    "type": "command_maybe",
                    "fuzzy_command": decision.fuzzy_command_id,
                    "embedding_command": decision.embedding_command_id,
                }
            )
        except RuntimeError:
            pass
        # Nothing was guessed - fall through and keep the words the
        # student actually said, same as ordinary dictation.

    await _persist_and_send_final(websocket, state, transcript_id, segment)


async def _handle_command(websocket: WebSocket, transcript_id: UUID, command_id: str) -> None:
    """Act on a recognized voice command instead of persisting it as text."""

    if command_id == "delete":
        await asyncio.to_thread(delete_last_segment, transcript_id)

    try:
        await websocket.send_json({"type": "command", "command": command_id})
    except RuntimeError:
        pass


async def _persist_and_send_final(
    websocket: WebSocket,
    state: dict,
    transcript_id: UUID,
    segment,
) -> None:
    order = state["segment_order"]
    state["segment_order"] = order + 1

    await asyncio.to_thread(
        add_final_segment,
        transcript_id,
        order,
        segment.text,
        segment.start,
        segment.end,
    )

    try:
        await websocket.send_json(
            {
                "type": "final",
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "segment": order,
            }
        )
    except RuntimeError:
        pass


async def _finalize_remaining_buffer(
    websocket: WebSocket,
    buffer: StreamingBuffer,
    state: dict,
    transcript_id: UUID,
) -> None:
    if buffer.is_empty:
        return

    vad = get_vad()
    audio = buffer.as_float32()

    try:
        vad_result = await asyncio.to_thread(vad.analyze, audio)

        if not vad_result.has_speech:
            return

        transcriber = get_streaming_transcriber()
        result = await transcriber.transcribe(audio)
    except Exception:
        logger.exception(
            "Failed to finalize trailing buffer for transcript %s.", transcript_id
        )
        return

    if not result.text.strip():
        return

    segment = buffer.finalize(result.text)
    await _resolve_and_dispatch(
        websocket, state, transcript_id, segment, result.avg_logprob, audio
    )
