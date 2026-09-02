from typing import Annotated, Callable

from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import TokenValidationError, decode_access_token
from app.db.session import SessionLocal, get_db
from app.models.user import User


bearer_scheme = HTTPBearer(auto_error=False)
SessionDependency = Annotated[Session, Depends(get_db)]

def authenticaton_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication credentials are missing or invalid.",
        headers={"WWW-Authenticate": "Bearer"}
    )

def get_current_user(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(bearer_scheme)
        ],
        db: SessionDependency,
) -> User:
    if credentials is None:
        raise authenticaton_error()

    if credentials.scheme.lower() != "bearer":
        raise authenticaton_error()

    try:
        payload = decode_access_token(credentials.credentials)
    except TokenValidationError as error:
        raise authenticaton_error() from error

    user = db.get(User, payload.sub)

    if user is None or not user.is_active:
        raise authenticaton_error()

    return user

CurrentUserDependency = Annotated[
    User,
    Depends(get_current_user),
]

def get_current_user_ws(
        websocket: WebSocket,
) -> User | None:
    """Authenticate a WebSocket connection.

    Browsers cannot set an Authorization header on a WebSocket handshake,
    so the access token is passed as a `token` query parameter instead.
    Returns None (rather than raising) so the caller can close the
    connection with code 1008 per the streaming protocol.
    """

    token = websocket.query_params.get("token")

    if not token:
        return None

    try:
        payload = decode_access_token(token)
    except TokenValidationError:
        return None

    with SessionLocal() as db:
        user = db.get(User, payload.sub)

        if user is None or not user.is_active:
            return None

        db.expunge(user)
        return user


def require_roles(
        *allowed_roles: str,
)  -> Callable[..., User]:
    def role_dependency(
            current_user: CurrentUserDependency,
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action."
            )
        return current_user

    return role_dependency
