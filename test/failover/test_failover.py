"""Failover and recovery tests (TESTING_REPORT.md section 3.1.7, A.3).

These are DESTRUCTIVE: they stop and kill containers. They run only against the
disposable scratch stack (docker-compose.scratch.yml, compose project
`dse-scratch`, api on :8001) and refuse to run against anything else. Never point
them at dev or production.

Not part of the default run (`pytest.ini` collects test/backend only) and not in CI.

    docker compose -f docker-compose.scratch.yml up -d
    docker compose -f docker-compose.scratch.yml exec backend alembic upgrade head
    docker compose -f docker-compose.scratch.yml exec backend python -m scripts.seed_users
    pytest test/failover -v
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "docker-compose.scratch.yml"
PROJECT = "dse-scratch"
API = os.environ.get("SCRATCH_API_URL", "http://localhost:8001").rstrip("/")
EMAIL, PASSWORD = "student@sinhaspeech.lk", "demo123"


def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", "-p", PROJECT, "-f", str(COMPOSE_FILE), *args],
        capture_output=True, text=True, check=check, cwd=ROOT,
    )


def request(method: str, path: str, body=None, token=None, timeout=10):
    """Returns (status, parsed_json_or_text); status 0 means no connection."""

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw, status = resp.read().decode(), resp.status
    except urllib.error.HTTPError as error:
        raw, status = error.read().decode(), error.code
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return 0, ""
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def wait_for(predicate, timeout=180, interval=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def healthy() -> bool:
    return request("GET", "/health")[0] == 200


def login() -> str:
    status, body = request("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})
    assert status == 200, f"login failed: {status} {body}"
    return body["token"]


def restart_count() -> int:
    out = subprocess.run(
        ["docker", "inspect", f"{PROJECT}-backend-1", "--format", "{{.RestartCount}}"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


@pytest.fixture(scope="module", autouse=True)
def scratch_stack():
    """Refuse to run unless the target is the scratch stack, then leave it healthy."""

    if not COMPOSE_FILE.exists() or "dse-scratch" not in COMPOSE_FILE.read_text():
        pytest.exit("docker-compose.scratch.yml missing or not the scratch stack", 2)
    if not healthy():
        pytest.skip(f"scratch stack not reachable at {API}; see the module docstring")
    yield
    compose("start", "database", "backend", check=False)
    wait_for(healthy)


# --- backups ---------------------------------------------------------------

def _psql(db: str, sql: str) -> str:
    return compose("exec", "-T", "database", "psql", "-U", "scratch", "-d", db, "-At", "-c", sql).stdout


def _table_counts(db: str) -> dict[str, int]:
    tables = _psql(db, "select tablename from pg_tables where schemaname='public' order by 1").split()
    return {table: int(_psql(db, f'select count(*) from "{table}"').strip()) for table in tables}


def test_pg_dump_backup_restores_into_a_separate_database():
    """A backup that has never been restored is a hypothesis, not a backup."""

    dump = compose("exec", "-T", "database", "pg_dump", "-U", "scratch", "-d", "scratch").stdout
    assert "CREATE TABLE" in dump

    compose("exec", "-T", "database", "dropdb", "-U", "scratch", "--if-exists", "scratch_restore")
    compose("exec", "-T", "database", "createdb", "-U", "scratch", "scratch_restore")
    try:
        restore = subprocess.run(
            ["docker", "compose", "-p", PROJECT, "-f", str(COMPOSE_FILE), "exec", "-T",
             "database", "psql", "-U", "scratch", "-d", "scratch_restore", "-v", "ON_ERROR_STOP=1"],
            input=dump, capture_output=True, text=True, cwd=ROOT,
        )
        assert restore.returncode == 0, restore.stderr[-500:]

        original, restored = _table_counts("scratch"), _table_counts("scratch_restore")
        assert original == restored
        assert original.get("users", 0) >= 2, "seed users missing; nothing meaningful compared"
    finally:
        compose("exec", "-T", "database", "dropdb", "-U", "scratch", "scratch_restore", check=False)


# --- database outage -------------------------------------------------------

def test_api_recovers_after_database_outage_without_a_restart():
    """`pool_pre_ping` is configured; this proves it end to end."""

    token = login()
    assert request("GET", "/auth/me", token=token)[0] == 200
    restarts_before = restart_count()

    compose("stop", "database")
    try:
        status, _ = request("GET", "/auth/me", token=token)
        assert status != 200, "request cannot succeed while the database is down"
        assert request("GET", "/health")[0] == 200, "liveness must not depend on the database"
    finally:
        compose("start", "database")

    assert wait_for(lambda: request("GET", "/auth/me", token=token)[0] == 200, timeout=90), \
        "API never recovered after the database came back"
    assert restart_count() == restarts_before, "recovery should not need a backend restart"


# --- process / host loss ---------------------------------------------------

def test_killed_backend_comes_back_by_itself():
    """`restart: unless-stopped` must revive an exited backend with no human involved.

    The process exits from inside the container (SIGTERM to PID 1; a container's PID 1
    ignores SIGKILL sent from within). `docker kill` / `docker stop` would NOT do: Docker
    treats those as a deliberate manual stop and, correctly, skips the restart policy, so
    they would fail this test for the wrong reason.
    """

    subprocess.run(
        ["docker", "exec", f"{PROJECT}-backend-1", "sh", "-c", "kill -TERM 1"],
        check=True, capture_output=True,
    )
    assert wait_for(lambda: not healthy(), timeout=30, interval=0.5), "kill did not take effect"
    assert wait_for(healthy, timeout=240), "backend did not restart on its own"
    assert restart_count() >= 1, "recovered, but not via the restart policy"
    assert request("GET", "/auth/me", token=login())[0] == 200, "restarted backend not fully usable"


# --- dropped WebSocket -----------------------------------------------------

def _log_line_count() -> int:
    return len(compose("logs", "backend").stdout.splitlines())


def _log_since(line_count: int) -> str:
    return "\n".join(compose("logs", "backend").stdout.splitlines()[line_count:])


def _stream_then_drop_abruptly(ws_module, token: str, mode: str, title: str | None = None):
    """Speak into a session, then vanish - no {"type":"stop"}, no close handshake.

    This is what a closed tab, lost wifi or client-side route change looks like
    to the server, and it is the common way a session ends, not the rare one.
    """

    clips = sorted((ROOT / "storage" / "voice_samples").glob("*.wav"))
    with wave.open(str(clips[0]), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())

    start = {"type": "start", "mode": mode}
    if title:
        start["title"] = title

    ws = ws_module.create_connection(
        API.replace("http", "ws", 1) + f"/streaming/ws?token={token}", timeout=10
    )
    ws.send(json.dumps(start))
    for offset in range(0, len(pcm), 8000):  # 8000 bytes = 250ms, like the browser
        ws.send_binary(pcm[offset:offset + 8000])
        time.sleep(0.25)
    ws.sock.close()


@pytest.mark.parametrize("mode", ["NOTE", "COMMAND"])
def test_an_abruptly_dropped_websocket_is_handled_cleanly(mode):
    """A vanished client must not produce an unhandled server error.

    Regression cover for the bug this suite found: the route's post-utterance
    notifications are all best-effort, but only caught `RuntimeError` (a
    *graceful* close). An abrupt drop raises `WebSocketDisconnect`, and under
    load uvicorn's `ClientDisconnected`, either of which escaped and aborted
    `_finalize_remaining_buffer` mid-cleanup. `test/backend/
    test_streaming_disconnect.py` pins the same behaviour at unit speed; this
    proves it through a real socket against a real server.

    Verified to fail against the unfixed route, via the COMMAND case - that one
    reproduces on every run. The NOTE case depends on what the model makes of
    the clip (the only samples available are command words, which NOTE mode
    routes differently), so treat it as a guard on the NOTE path rather than as
    the reproducer.
    """

    ws_module = pytest.importorskip("websocket")
    if not sorted((ROOT / "storage" / "voice_samples").glob("*.wav")):
        pytest.skip("no voice samples to stream")

    token = login()
    before = _log_line_count()

    _stream_then_drop_abruptly(
        ws_module, token, mode, title=f"failover-{mode}-{int(time.time())}" if mode == "NOTE" else None
    )
    time.sleep(15)  # let the session finish its own cleanup

    new_logs = _log_since(before)
    assert "Exception in ASGI application" not in new_logs, (
        f"abrupt {mode} disconnect produced an unhandled server error:\n{new_logs[-2000:]}"
    )
    assert healthy(), "server must stay up after a client vanishes"


def test_abrupt_drops_do_not_leak_the_per_user_session_slot():
    """A leaked slot is unrecoverable without a restart: after
    `streaming_max_sessions_per_user` drops the student can never start again."""

    ws_module = pytest.importorskip("websocket")
    if not sorted((ROOT / "storage" / "voice_samples").glob("*.wav")):
        pytest.skip("no voice samples to stream")

    token = login()
    cap_probe = 5  # more than the production cap of 3, so a leak cannot hide

    for _ in range(cap_probe):
        _stream_then_drop_abruptly(ws_module, token, "COMMAND")
        time.sleep(3)

    # If any slot leaked, this session is refused with a "Too many concurrent
    # streaming sessions" error instead of accepting audio.
    ws = ws_module.create_connection(
        API.replace("http", "ws", 1) + f"/streaming/ws?token={token}", timeout=10
    )
    try:
        ws.send(json.dumps({"type": "start", "mode": "COMMAND"}))
        ws.settimeout(5)
        try:
            message = json.loads(ws.recv())
            assert message.get("type") != "error", f"session slot leaked: {message.get('message')}"
        except ws_module.WebSocketTimeoutException:
            pass  # silence is correct: COMMAND mode says nothing until it hears speech
    finally:
        ws.close()
