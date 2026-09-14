import asyncio
import json
import logging
import time
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
from app.services.transcript_service import DuplicateTranscriptTitleError
from app.streaming.buffer import StreamingBuffer
from app.streaming.command_resolution import resolve_command
from app.streaming.embeddings import ClipTooShortError
from app.streaming.inference import get_streaming_transcriber
from app.streaming.vad import get_vad

# Voice commands with a defined action *inside NOTE dictation* - anything
# else in the vocabulary (next/previous/save/submit) has no meaning while
# taking a note, so a fuzzy match on those is left as ordinary dictated
# text rather than silently swallowed. This restriction does not apply to
# COMMAND mode (see _run_session) - there, every recognized command is
# forwarded and the *client* decides what it means for the page it's on
# (the transcript review page maps save/submit to its own buttons, the
# quiz page maps next/previous/submit to its own navigation).
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
    """Live transcription WebSocket - NOTE (dictation) and COMMAND
    (voice-commands-only, no transcript) modes.

    Protocol:
      client -> {"type": "start", "mode": "NOTE", "title": str}
      client -> {"type": "start", "mode": "COMMAND"}
      client -> binary frames of 16kHz mono 16-bit PCM
      client -> {"type": "stop"}
      server -> {"type": "partial", "text": str, "segment": int}
      server -> {"type": "final", "text": str, "start": float, "end": float, "segment": int,
                 "segment_id": str, "transcript_id": str}
        (NOTE mode only - COMMAND mode never persists or reports dictated text.
        segment_id/transcript_id are the real database ids, included so the
        client can PATCH /transcripts/{transcript_id} to edit this exact
        line without waiting for session_end.)
      server -> {"type": "listening"}
        (COMMAND mode only - sent the instant VAD detects speech starting,
        before any transcription, so the client can show feedback with no
        perceptible delay)
      server -> {"type": "command", "command": str}
      server -> {"type": "command_maybe", "fuzzy_command": str|null, "embedding_command": str|null}
      server -> {"type": "session_end", "transcript_id": str|null}
      server -> {"type": "error", "message": str}

    Every finalized utterance is checked against the voice-command
    vocabulary via app.streaming.command_resolution, which combines the
    fuzzy-text match (app.streaming.commands) with, for an enrolled
    student, a speaker-embedding match (app.streaming.embeddings) - see
    that module for the exact combination rule.

    In NOTE mode, only "delete" and "stop" have a server-side effect
    (delete the last line / stop the session); anything else recognized
    is left as ordinary dictated text, since most commands have no
    meaning mid-note. In COMMAND mode there is no dictation to protect -
    every recognized command is forwarded via a "command" message with
    no server-side effect at all; the client (transcript review page,
    quiz page, ...) decides what each command means there. Neither mode
    ever guesses: a "confirm" outcome only ever sends "command_maybe",
    never "command".

    A student with no enrollment bank (or with embedding matching
    switched off globally) gets exactly the fuzzy-only behaviour in both
    modes - see resolve_command's fallback.

    NOTE mode stays tick-driven: the whole buffer is re-transcribed
    every `streaming_window_interval_seconds` so partial previews can be
    shown mid-utterance, which is the point of dictation. COMMAND mode
    has no preview to show, so it is event-driven instead - VAD runs on
    every incoming chunk (see `_process_command_chunk`) and a single
    transcription fires the moment speech ends, rather than waiting for
    the next tick and re-transcribing 2-3x along the way. Any executed
    command (either mode) is debounced - see `_send_command` - so a
    student repeating themselves doesn't trigger the action twice.
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

    mode = start_message.get("mode")

    if mode not in TRANSCRIPT_TYPE_BY_MODE and mode != "COMMAND":
        await websocket.send_json(
            {"type": "error", "message": "Only NOTE, EXAM and COMMAND modes are currently supported."}
        )
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    transcript_id: UUID | None = None
    if mode in TRANSCRIPT_TYPE_BY_MODE:
        title = start_message.get("title") or "Untitled note"
        try:
            transcript_id = await asyncio.to_thread(
                create_live_transcript, user, title, TRANSCRIPT_TYPE_BY_MODE[mode]
            )
        except DuplicateTranscriptTitleError as error:
            await websocket.send_json({"type": "error", "message": str(error)})
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

    # Which phrase set this student's commands are matched against - a
    # database setting (see User.command_language), never a code change.
    # Loaded once per session, same as the bank below.
    language = user.command_language

    # Loaded once per session, not per utterance - small (a handful of
    # commands x a handful of samples x one float vector each) and never
    # changes mid-session. Empty for a student who hasn't enrolled, which
    # is exactly what makes resolve_command's fallback kick in below.
    bank = (
        await asyncio.to_thread(voice_enrollment.load_bank, user.user_id, language)
        if settings.voice_command_embedding_matching_enabled
        else {}
    )

    buffer = StreamingBuffer(
        max_buffer_seconds=settings.streaming_max_buffer_seconds,
        overlap_seconds=settings.streaming_overlap_seconds,
        memory_ceiling_seconds=settings.streaming_memory_ceiling_seconds,
    )
    state = {
        "segment_order": 0,
        "bank": bank,
        "mode": mode,
        "language": language,
        # COMMAND mode only: has a "listening" ack already been sent for
        # the utterance currently in the buffer (reset once it's
        # finalized), and when + which command was last executed, for
        # debouncing a rapid repeat - see _process_command_chunk and
        # _send_command.
        "listening_sent": False,
        "last_command": None,
    }

    # NOTE mode keeps the tick loop exactly as before, for partial
    # previews. COMMAND mode has nothing to preview, so it runs
    # event-driven instead, straight off the receive loop below - no
    # ticker task for it at all.
    ticker = (
        asyncio.create_task(_window_ticker(websocket, buffer, state, transcript_id))
        if mode != "COMMAND"
        else None
    )
    vad = get_vad()
    transcriber = get_streaming_transcriber()

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                buffer.append(data)

                if mode == "COMMAND":
                    try:
                        await _process_command_chunk(
                            websocket, buffer, state, transcript_id, vad, transcriber
                        )
                    except Exception:
                        logger.exception(
                            "Command-mode inference failed; continuing session."
                        )
                        try:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": "Transcription failed for this window.",
                                }
                            )
                        except RuntimeError:
                            break

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
        if ticker is not None:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass

        await _finalize_remaining_buffer(websocket, buffer, state, transcript_id)
        # A NOTE transcript is left in DRAFT status so the user can
        # review, edit and correct low-confidence words before finalizing
        # it themselves. COMMAND mode never created a transcript.

        try:
            await websocket.send_json(
                {
                    "type": "session_end",
                    "transcript_id": str(transcript_id) if transcript_id else None,
                }
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
    transcript_id: UUID | None,
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
    transcript_id: UUID | None,
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
    elif state["mode"] != "COMMAND":
        # Partial previews only matter for the dictation flow (NOTE and
        # EXAM) - COMMAND mode has no note text to show the client
        # mid-utterance.
        await websocket.send_json(
            {"type": "partial", "text": text, "segment": state["segment_order"]}
        )

    if buffer.exceeded_memory_ceiling():
        logger.warning(
            "Streaming buffer exceeded memory ceiling for transcript %s; force-finalizing.",
            transcript_id,
        )


async def _process_command_chunk(
    websocket: WebSocket,
    buffer: StreamingBuffer,
    state: dict,
    transcript_id: UUID | None,
    vad,
    transcriber,
) -> None:
    """COMMAND mode's event-driven trigger: called once per incoming
    chunk instead of on a fixed-interval tick.

    VAD is cheap (a few ms) so it runs on every chunk. Speech starting
    sends an instant "listening" ack, before any transcription - this is
    what actually removes the perceived latency. Speech ending (a pause
    of `streaming_command_vad_silence_ms`) triggers exactly one
    transcription of the whole utterance, then reuses the same
    finalize/resolve/dispatch path dictation's tick loop uses (see
    `_emit_final`) - resolve_command, match_command and best_match are
    untouched by this change, only how often and when they get called.
    """

    if buffer.is_empty:
        return

    audio = buffer.as_float32()
    vad_result = await asyncio.to_thread(vad.analyze, audio)

    if not vad_result.has_speech:
        # Whisper hallucinates on silence - skip inference entirely, but
        # the 15s cap is still a hard safety net even with no speech in
        # the buffer (e.g. a stuck-open mic picking up only noise).
        if buffer.exceeded_max_buffer():
            buffer.force_cut()
        return

    if not state["listening_sent"]:
        state["listening_sent"] = True
        try:
            await websocket.send_json({"type": "listening"})
        except RuntimeError:
            return

    is_pause = (
        vad_result.trailing_silence_seconds * 1000
        >= settings.streaming_command_vad_silence_ms
    )
    should_force = buffer.exceeded_max_buffer()

    if not (is_pause or should_force):
        return

    state["listening_sent"] = False
    result = await transcriber.transcribe(
        audio, mode="command", command_language=state["language"]
    )
    await _emit_final(
        websocket, buffer, state, transcript_id, result.text, result.avg_logprob, audio
    )


async def _emit_final(
    websocket: WebSocket,
    buffer: StreamingBuffer,
    state: dict,
    transcript_id: UUID | None,
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
    transcript_id: UUID | None,
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
        segment.text,
        avg_logprob=avg_logprob,
        embedding=embedding,
        bank=bank,
        language=state["language"],
    )

    if state["mode"] == "COMMAND":
        # No note to protect and nothing persisted either way - every
        # recognized command is just forwarded, the client owns what it
        # means on whatever page it's listening from.
        if decision.outcome == "execute":
            await _send_command(websocket, state, decision.command_id)
        elif decision.outcome == "confirm":
            await _send_command_maybe(websocket, decision)
        return

    if decision.outcome == "execute" and decision.command_id in _ACTIONABLE_NOTE_COMMANDS:
        await _handle_note_command(websocket, state, transcript_id, decision.command_id)
        return

    if decision.outcome == "confirm":
        await _send_command_maybe(websocket, decision)
        # Nothing was guessed - fall through and keep the words the
        # student actually said, same as ordinary dictation.

    await _persist_and_send_final(websocket, state, transcript_id, segment)


async def _handle_note_command(
    websocket: WebSocket, state: dict, transcript_id: UUID | None, command_id: str
) -> None:
    """Act on a recognized voice command instead of persisting it as note text."""

    if command_id == "delete" and transcript_id is not None:
        await asyncio.to_thread(delete_last_segment, transcript_id)

    await _send_command(websocket, state, command_id)


async def _send_command(websocket: WebSocket, state: dict, command_id: str) -> None:
    """Send an executed command to the client - debounced so a student
    repeating themselves within `voice_command_debounce_seconds` (they
    aren't sure they were heard) doesn't execute the action twice, e.g.
    advancing two quiz questions off one spoken "next"."""

    now = time.monotonic()
    last = state.get("last_command")

    if (
        last is not None
        and last[0] == command_id
        and (now - last[1]) < settings.voice_command_debounce_seconds
    ):
        logger.debug(
            "Debounced repeat of command %r within %.1fs.",
            command_id,
            settings.voice_command_debounce_seconds,
        )
        return

    state["last_command"] = (command_id, now)

    try:
        await websocket.send_json({"type": "command", "command": command_id})
    except RuntimeError:
        pass


async def _send_command_maybe(websocket: WebSocket, decision) -> None:
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


async def _persist_and_send_final(
    websocket: WebSocket,
    state: dict,
    transcript_id: UUID | None,
    segment,
) -> None:
    order = state["segment_order"]
    state["segment_order"] = order + 1

    segment_id = await asyncio.to_thread(
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
                "segment_id": str(segment_id),
                "transcript_id": str(transcript_id),
            }
        )
    except RuntimeError:
        pass


async def _finalize_remaining_buffer(
    websocket: WebSocket,
    buffer: StreamingBuffer,
    state: dict,
    transcript_id: UUID | None,
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
        result = await transcriber.transcribe(
            audio,
            mode="command" if state["mode"] == "COMMAND" else "dictation",
            command_language=state["language"],
        )
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
