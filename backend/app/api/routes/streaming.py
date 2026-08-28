import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.api.dependencies import get_current_user_ws
from app.core.config import settings


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/streaming",
    tags=["Streaming"],
)


@router.websocket("/ws")
async def stream_transcription(websocket: WebSocket) -> None:
    """Live transcription WebSocket.

    Build step 1: authenticate, accept, echo the byte count of each
    binary frame received. No model, no VAD, no persistence yet.
    """

    if not settings.streaming_enabled:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user = get_current_user_ws(websocket)

    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                await websocket.send_json(
                    {"type": "echo", "bytes": len(data)}
                )
                continue

            text = message.get("text")
            if text is not None:
                await websocket.send_json(
                    {"type": "echo_text", "length": len(text)}
                )

    except WebSocketDisconnect:
        logger.info(
            "Streaming echo connection closed for user %s",
            user.user_id,
        )
