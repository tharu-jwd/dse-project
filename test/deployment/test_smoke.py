"""Smoke tests against a LIVE deployment.

These are not unit tests - they make real network calls to a running
frontend and backend, and they fail if the deployment is down, however
correct the code is. That is the point: they answer "is the thing I just
deployed actually working?", which the unit suite cannot.

They are deliberately kept out of the default test run. `pytest.ini` sets
`testpaths = test/backend`, so plain `pytest` never picks these up and CI
never depends on the server being reachable.

Run them by hand after a deploy:

    pytest test/deployment/ -v

Point them at somewhere else by overriding the URLs:

    SMOKE_API_URL=https://staging.example.com \
    SMOKE_APP_URL=https://staging.vercel.app \
    pytest test/deployment/ -v

Every check here corresponds to something that actually broke, or could
silently break, during the real deployment - see DEPLOYMENT.md.
"""

import asyncio
import json
import os
import urllib.error
import urllib.request

import pytest


API_URL = os.environ.get("SMOKE_API_URL", "https://sinhaspeech.duckdns.org").rstrip("/")
APP_URL = os.environ.get("SMOKE_APP_URL", "https://sinhaspeech.vercel.app").rstrip("/")
EMAIL = os.environ.get("SMOKE_EMAIL", "student@sinhaspeech.lk")
PASSWORD = os.environ.get("SMOKE_PASSWORD", "demo123")

TIMEOUT = 30


def _get(url: str, headers: dict | None = None):
    """Returns (status, body_text). Does not raise on 4xx/5xx."""

    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def _post_json(url: str, payload: dict, headers: dict | None = None):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


@pytest.fixture(scope="session")
def token() -> str:
    """A real access token, so the authenticated checks below exercise the
    same path the browser does."""

    status, body = _post_json(
        f"{API_URL}/auth/login", {"email": EMAIL, "password": PASSWORD}
    )
    assert status == 200, f"login failed ({status}): {body[:200]}"

    data = json.loads(body)
    value = data.get("token") or data.get("access_token")
    assert value, f"login succeeded but returned no token: {sorted(data)}"
    return value


# --- Backend reachability -------------------------------------------------


def test_api_is_reachable_over_https():
    """Also proves the TLS certificate is valid - urllib verifies it by
    default and raises on an expired or mismatched certificate, which is
    how a lapsed Let's Encrypt renewal would surface."""

    status, body = _get(f"{API_URL}/health")
    assert status == 200, f"expected 200, got {status}: {body[:200]}"
    assert json.loads(body)["status"] == "healthy"


def test_api_can_reach_its_database():
    """Separate from /health: the API process can be up while Postgres is
    unreachable, which looks fine until the first real request."""

    status, body = _get(f"{API_URL}/health/database")
    assert status == 200, f"expected 200, got {status}: {body[:200]}"
    assert json.loads(body)["database"] == "connected"


def test_interactive_docs_are_served():
    status, _ = _get(f"{API_URL}/docs")
    assert status == 200


# --- Authentication -------------------------------------------------------


def test_login_returns_a_working_token(token):
    assert isinstance(token, str) and len(token) > 20


def test_authenticated_request_succeeds(token):
    status, body = _get(
        f"{API_URL}/transcripts", headers={"Authorization": f"Bearer {token}"}
    )
    assert status == 200, f"expected 200, got {status}: {body[:200]}"


def test_unauthenticated_request_is_rejected():
    """A misconfigured deployment that accepts anonymous requests would be
    a far worse failure than one that is simply down."""

    status, _ = _get(f"{API_URL}/transcripts")
    assert status in (401, 403), f"expected 401/403, got {status}"


# --- CORS -----------------------------------------------------------------


def test_cors_allows_the_deployed_frontend():
    """The most common "deployed but nothing works" cause: the browser
    blocks every call because the backend does not list the frontend's
    origin. curl and these helpers ignore CORS, so it must be asserted
    explicitly rather than inferred from the API working."""

    request = urllib.request.Request(
        f"{API_URL}/auth/login",
        method="OPTIONS",
        headers={
            "Origin": APP_URL,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        allowed = response.headers.get("access-control-allow-origin")

    assert allowed == APP_URL, (
        f"backend does not allow {APP_URL}; it returned {allowed!r}. "
        "Check CORS_ORIGINS in the server's .env, then restart the backend."
    )


# --- Frontend -------------------------------------------------------------


def test_frontend_is_served():
    status, body = _get(APP_URL)
    assert status == 200
    assert "<div id=\"root\"" in body or "<script" in body


def test_frontend_was_built_against_this_api():
    """Vite bakes VITE_API_BASE_URL into the bundle at build time, so a
    frontend built with the wrong value looks perfectly healthy while
    failing every request. Verify the artifact, not the config."""

    import re

    status, html = _get(APP_URL)
    assert status == 200

    match = re.search(r'/assets/index-[A-Za-z0-9_-]+\.js', html)
    assert match, "could not find the JS bundle in the served HTML"

    status, bundle = _get(f"{APP_URL}{match.group(0)}")
    assert status == 200

    assert API_URL in bundle, f"bundle does not reference {API_URL}"
    assert "localhost:8000" not in bundle, (
        "bundle still points at localhost - it was built without "
        "VITE_API_BASE_URL set. Rebuild and redeploy the frontend."
    )


# --- Live streaming -------------------------------------------------------


@pytest.mark.asyncio
async def test_websocket_accepts_an_authenticated_session(token):
    """Live captioning and voice commands both run over this socket. It is
    a separate code path from the HTTP API and can fail independently -
    e.g. if a proxy in front of the backend does not forward upgrades."""

    websockets = pytest.importorskip("websockets")

    url = f"{API_URL.replace('https://', 'wss://').replace('http://', 'ws://')}"
    url = f"{url}/streaming/ws?token={token}"

    async with websockets.connect(url, open_timeout=TIMEOUT) as socket:
        await socket.send(json.dumps({"type": "start", "mode": "COMMAND"}))
        # No audio is sent, so the server has nothing to reply with; simply
        # reaching this point proves the upgrade, the auth and the session
        # start all worked.
        await asyncio.sleep(0.5)
        assert socket.state.name == "OPEN"


@pytest.mark.asyncio
async def test_websocket_rejects_a_bad_token():
    websockets = pytest.importorskip("websockets")

    url = f"{API_URL.replace('https://', 'wss://')}/streaming/ws?token=not-a-real-token"

    with pytest.raises(Exception):
        async with websockets.connect(url, open_timeout=TIMEOUT) as socket:
            await socket.recv()
