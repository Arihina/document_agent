from uuid import UUID

from fastapi import HTTPException, Request


async def get_user_id(request: Request) -> UUID:
    # user_id кладёт InternalAuthMiddleware в request.state ПОСЛЕ проверки
    # X-Internal-Token.
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        raise HTTPException(401, "Требуется внутренний токен master")
    return user_id
