import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.session import engine
from app.api.router import api_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.streaming_enabled:
        # Load once at process startup, never per WebSocket session.
        from app.streaming.inference import get_streaming_transcriber
        from app.streaming.vad import get_vad

        logger.info("Streaming enabled: loading faster-whisper and VAD models...")
        get_streaming_transcriber()
        get_vad()
        logger.info("Streaming models loaded.")

    yield


app = FastAPI(
    title="SinhaSpeech API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

app.include_router(api_router)

@app.exception_handler(HTTPException)
async def http_exception_handler(
    _request: Request,
    exception: HTTPException,
) -> JSONResponse:
    message = (
        exception.detail
        if isinstance(exception.detail, str)
        else "The request could not be completed"
    )

    return JSONResponse(
        status_code=exception.status_code,
        content={"message" : message},
        headers=exception.headers,
    )


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {
        "name": "SinhaSpeech API",
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/health/database", tags=["System"])
def database_health_check() -> dict[str, str]:
    try:
        with engine.connect() as connection:
            connection.execute(text("select 1"))
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable"
        ) from error

    return {
        "status" : "healthy",
        "database" : "connected",
    }
