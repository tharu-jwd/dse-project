from fastapi import APIRouter

from app.api.routes.auth import router as auth_router
from app.api.routes.transcripts import router as transcripts_router
from app.api.routes.transcriptions import router as transcriptions_router
from app.api.routes.media import router as media_router
from app.api.routes.streaming import router as streaming_router


api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(transcripts_router)
api_router.include_router(transcriptions_router)
api_router.include_router(media_router)
api_router.include_router(streaming_router)

