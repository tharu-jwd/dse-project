from fastapi import APIRouter, HTTPException, Response, status

from app.api.dependencies import (
    CurrentUserDependency,
    SessionDependency,
)
from app.core.security import create_access_token
from app.schemas.auth import LoginRequest, LoginResponse
from app.schemas.user import UserResponse
from app.services.auth_service import authenticate_user


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/login",
    response_model=LoginResponse,
)
def login(
    login_data: LoginRequest,
    db: SessionDependency,
) -> LoginResponse:
    user = authenticate_user(
        db=db,
        email=str(login_data.email),
        password=login_data.password.get_secret_value(),
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The email or password is incorrect.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user.user_id)

    return LoginResponse(
        token = token,
        user=UserResponse.model_validate(user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_authenticated_user(
    current_user: CurrentUserDependency,
) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def logout(
    _current_user: CurrentUserDependency,
) -> Response:
    #JWT authentication is stateless. frontend removes the token.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


