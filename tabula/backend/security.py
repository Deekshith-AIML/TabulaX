"""Authentication helpers: password hashing, JWT issuing/validation, and the
FastAPI dependency that resolves the current user from a bearer token.

Passwords are hashed with bcrypt.  JWTs are signed with the configured
``SECRET_KEY`` and carry an expiry.  User lookups go through the async MongoDB
layer and always convert ``_id`` to a proper ``ObjectId``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from bson import ObjectId

from config import settings
from database import get_user_by_username, get_password_hash as hash_password, verify_password

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Password hashing (bcrypt) is implemented in database.py and re-exported here.
__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "get_current_user",
    "get_current_active_user",
    "as_object_id",
]


# --------------------------------------------------------------------------- #
# JWT
# --------------------------------------------------------------------------- #
def create_access_token(
    subject: str, expires_delta: Optional[timedelta] = None
) -> str:
    """Create a signed JWT whose ``sub`` claim is the username."""
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload = {"sub": subject, "exp": expire, "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Optional[str]:
    """Validate a JWT and return the username (``sub``) or ``None`` if invalid."""
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
        sub = payload.get("sub")
        return str(sub) if sub else None
    except JWTError:
        return None


# --------------------------------------------------------------------------- #
# FastAPI dependencies
# --------------------------------------------------------------------------- #
def _credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Resolve the authenticated ``AppUser`` from the bearer token."""
    username = decode_access_token(token)
    if not username:
        raise _credentials_exception()

    user_doc = await get_user_by_username(username)
    if not user_doc:
        raise _credentials_exception()

    from app import AppUser  # local import to avoid circular dependency at load

    return AppUser(
        id=str(user_doc["_id"]),
        username=user_doc["username"],
        email=user_doc.get("email"),
        full_name=user_doc.get("full_name"),
        created_at=user_doc.get("created_at", datetime.now(timezone.utc)),
        last_login=user_doc.get("last_login"),
        is_active=user_doc.get("is_active", True),
    )


async def get_current_active_user(current_user=Depends(get_current_user)):
    """Reject deactivated accounts."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def as_object_id(value: str) -> ObjectId:
    """Convert a string to an ``ObjectId`` or raise a 400."""
    if not ObjectId.is_valid(value):
        raise HTTPException(status_code=400, detail="Invalid identifier")
    return ObjectId(value)
