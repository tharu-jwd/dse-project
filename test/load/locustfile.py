"""Load test for the API - HTTP request path and live-streaming WebSocket.

NOT part of the default test run (`pytest.ini` only collects test/backend) and
NOT run in CI. Point it at a scratch environment, never the live deployment
during a demo: it writes real rows and burns the same 2 vCPUs the demo needs.

    pip install locust websocket-client

    # web UI on http://localhost:8089
    locust -f test/load/locustfile.py --host http://localhost:8000

    # headless: 5 users, 1/s spawn, 2 minutes
    locust -f test/load/locustfile.py --host http://localhost:8000 \
        --headless -u 5 -r 1 -t 2m

Environment:
    LOAD_EMAIL / LOAD_PASSWORD   account to log in as (default: seeded demo student)
    LOAD_WAV_DIR                 clips streamed to the WS scenario
                                 (default: storage/voice_samples)

Two things to know before reading the numbers:

* The server caps concurrent streaming sessions per user
  (`streaming_max_sessions_per_user`, default 3). All simulated users share one
  account here, so past 3 StreamingUser instances you are measuring that cap,
  not CPU. Those rejections are reported as a distinct "ws rejected" request
  rather than hidden. Raise the cap on the scratch server to measure CPU.
* Production requires the "zimi" wake word, so the bare command clips used here
  would be silently ignored and never answer. The scratch stack
  (docker-compose.scratch.yml) sets VOICE_COMMAND_WAKE_REQUIRED=false; against
  any server that keeps the gate on, "ws speech->verdict" will time out.
* This drives the request path and real audio through Whisper; it says nothing
  about transcription *quality* under load.
"""

import json
import os
import random
import time
import wave
from pathlib import Path

import websocket
from locust import HttpUser, between, events, task

EMAIL = os.environ.get("LOAD_EMAIL", "student@sinhaspeech.lk")
PASSWORD = os.environ.get("LOAD_PASSWORD", "demo123")
WAV_DIR = Path(
    os.environ.get(
        "LOAD_WAV_DIR", Path(__file__).resolve().parents[2] / "storage" / "voice_samples"
    )
)

CHUNK_SAMPLES = 4000  # 250 ms at 16 kHz, matching the browser client


def _load_clips() -> list[bytes]:
    """16 kHz mono 16-bit PCM clips; anything else is skipped, not converted."""

    clips = []
    for path in sorted(WAV_DIR.rglob("*.wav")):
        try:
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2):
                    clips.append(wav.readframes(wav.getnframes()))
        except (wave.Error, EOFError):
            continue
    return clips


CLIPS = _load_clips()


@events.test_start.add_listener
def _warn_if_no_clips(environment, **_):
    if not CLIPS:
        print(f"[load] no usable 16kHz mono wav clips in {WAV_DIR}; StreamingUser will idle")


def _fire(environment, name, started, exception=None, length=0):
    environment.events.request.fire(
        request_type="WS",
        name=name,
        response_time=(time.time() - started) * 1000,
        response_length=length,
        exception=exception,
    )


class ApiUser(HttpUser):
    """Ordinary authenticated browsing: the cheap request path."""

    wait_time = between(1, 3)
    weight = 3

    def on_start(self):
        with self.client.post(
            "/auth/login", json={"email": EMAIL, "password": PASSWORD}, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"login failed: {response.status_code}")
                self.stop()
                return
            self.headers = {"Authorization": f"Bearer {response.json()['token']}"}

    @task(3)
    def me(self):
        self.client.get("/auth/me", headers=self.headers, name="/auth/me")

    @task(2)
    def transcripts(self):
        self.client.get("/transcripts", headers=self.headers, name="/transcripts")

    @task(1)
    def health(self):
        self.client.get("/health")


class StreamingUser(HttpUser):
    """One COMMAND-mode streaming session: the CPU-heavy path (VAD + Whisper)."""

    wait_time = between(2, 5)
    weight = 1

    def on_start(self):
        response = self.client.post(
            "/auth/login", json={"email": EMAIL, "password": PASSWORD}, name="/auth/login"
        )
        self.token = response.json()["token"] if response.ok else None
        if not self.token:
            self.stop()

    @task
    def stream_one_clip(self):
        if not CLIPS:
            time.sleep(5)
            return

        url = self.host.replace("http", "ws", 1) + f"/streaming/ws?token={self.token}"
        started = time.time()
        try:
            ws = websocket.create_connection(url, timeout=15)
        except Exception as error:
            _fire(self.environment, "ws connect", started, error)
            return
        _fire(self.environment, "ws connect", started)

        clip = random.choice(CLIPS)
        try:
            ws.send(json.dumps({"type": "start", "mode": "COMMAND"}))
            ws.settimeout(0.01)
            sent_at = time.time()
            answered = None

            # Real-time pacing: one 250 ms chunk every 250 ms, like a mic.
            for offset in range(0, len(clip), CHUNK_SAMPLES * 2):
                ws.send_binary(clip[offset : offset + CHUNK_SAMPLES * 2])
                deadline = time.time() + 0.25
                while time.time() < deadline:
                    try:
                        message = json.loads(ws.recv())
                    except websocket.WebSocketTimeoutException:
                        continue
                    if message["type"] == "error":
                        _fire(self.environment, "ws rejected", started, Exception(message["message"]))
                        return
                    if message["type"] in ("command", "command_maybe") and answered is None:
                        answered = time.time()

            # Trailing silence so VAD closes the utterance, then wait for the verdict.
            silence = b"\x00\x00" * CHUNK_SAMPLES
            wait_until = time.time() + 15
            while answered is None and time.time() < wait_until:
                ws.send_binary(silence)
                time.sleep(0.25)
                try:
                    message = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                if message["type"] in ("command", "command_maybe"):
                    answered = time.time()

            if answered is None:
                _fire(self.environment, "ws speech->verdict", sent_at, Exception("no verdict in 15s"))
            else:
                # Latency from when the clip's audio finished to the verdict.
                clip_seconds = len(clip) / 2 / 16000
                latency_start = sent_at + clip_seconds
                _fire(self.environment, "ws speech->verdict", latency_start)
            ws.send(json.dumps({"type": "stop"}))
        except Exception as error:
            _fire(self.environment, "ws session", started, error)
        finally:
            ws.close()
