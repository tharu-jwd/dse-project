"""Direct tests for get_current_user_ws (Appendix A.6: api/dependencies.py
was 58% covered - the WebSocket auth branches were only ever exercised
indirectly, through a real socket connection in test_streaming_commands_route.py
and test/failover/). get_current_user_ws takes a plain object with a
.query_params.get() method, so it's callable directly with a stub - no
real WebSocket connection needed.

get_current_user's own non-bearer-scheme check (line 33,
`credentials.scheme.lower() != "bearer"`) turns out to be dead code:
FastAPI's `HTTPBearer(auto_error=False)` already returns `None` for any
non-"Bearer" Authorization scheme before get_current_user ever runs, so
`credentials` there is always either a real Bearer token or None - the
explicit scheme check can never actually execute. Confirmed directly
against HTTPBearer.__call__ before writing this note, not assumed.
Left as-is (removing it changes no behaviour and isn't this sprint's
job) but documented here so the "why is this still uncovered" question
has an answer next time someone looks at Appendix A.6.
"""

from types import SimpleNamespace

from app.api.dependencies import get_current_user_ws
from app.core.security import create_access_token


class _FakeWebSocket:
    """Just enough of Starlette's WebSocket for get_current_user_ws:
    it only ever reads websocket.query_params.get("token")."""

    def __init__(self, token: str | None = None):
        self.query_params = SimpleNamespace(get=lambda key: token if key == "token" else None)


def test_returns_none_when_no_token_is_present():
    assert get_current_user_ws(_FakeWebSocket(token=None)) is None


def test_returns_none_for_a_garbage_token():
    assert get_current_user_ws(_FakeWebSocket(token="not-a-real-jwt")) is None


def test_returns_none_for_a_token_signed_with_the_wrong_key(student):
    import jwt

    from app.core.config import settings

    wrong_key_token = jwt.encode(
        {"sub": str(student.user_id), "type": "access"},
        "a-completely-different-secret",
        algorithm=settings.jwt_algorithm,
    )
    assert get_current_user_ws(_FakeWebSocket(token=wrong_key_token)) is None


def test_returns_the_user_for_a_valid_token(student):
    token = create_access_token(student.user_id)
    user = get_current_user_ws(_FakeWebSocket(token=token))

    assert user is not None
    assert user.user_id == student.user_id


def test_returns_none_for_a_token_belonging_to_a_since_deleted_user():
    import uuid

    token = create_access_token(uuid.uuid4())
    assert get_current_user_ws(_FakeWebSocket(token=token)) is None


def test_http_auth_rejects_a_non_bearer_scheme(client, student):
    """Confirms the 401 end-to-end. Doesn't exercise get_current_user's own
    scheme check (see module docstring) - HTTPBearer rejects it first."""

    response = client.get("/auth/me", headers={"Authorization": f"Basic {student.token}"})
    assert response.status_code == 401
