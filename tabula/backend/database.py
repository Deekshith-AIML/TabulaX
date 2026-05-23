"""MongoDB access layer for user accounts and session/dataset storage.

Password hashing lives here (bcrypt).  ``update_last_login`` correctly converts
the user id to an ``ObjectId`` before querying, fixing the original bug where a
raw string id never matched a document.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import bcrypt
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from config import settings

logger = logging.getLogger(__name__)

client: Optional[AsyncIOMotorClient] = None
db = None
users = None
sessions = None
delete_backups = None


# --------------------------------------------------------------------------- #
# Connection lifecycle
# --------------------------------------------------------------------------- #
async def connect_to_mongo() -> None:
    """Connect to MongoDB and ensure indexes."""
    global client, db, users, sessions, delete_backups
    logger.info("Connecting to MongoDB at %s (db=%s)", _safe_url(settings.mongodb_url), settings.database_name)
    client = AsyncIOMotorClient(
        settings.mongodb_url,
        serverSelectionTimeoutMS=5000,
        uuidRepresentation="standard",
    )
    # Verify connectivity early so failures are obvious at startup.
    await client.server_info()
    db = client[settings.database_name]
    users = db.users
    sessions = db.sessions
    delete_backups = db.delete_backups

    await users.create_index("username", unique=True)
    await users.create_index("email", unique=True)
    # Sessions expire automatically via a TTL index on expires_at.
    await sessions.create_index("expires_at", expireAfterSeconds=0)
    # Owner lookup + isolation.
    await sessions.create_index([("owner_id", 1), ("expires_at", -1)])
    logger.info("MongoDB connected; indexes ensured.")


async def close_mongo_connection() -> None:
    """Close the MongoDB connection."""
    global client
    if client is not None:
        client.close()
        client = None
        logger.info("MongoDB connection closed.")


def _safe_url(url: str) -> str:
    """Return the URL with any credentials redacted for logging."""
    if "@" in url and "//" in url:
        scheme, _, rest = url.partition("//")
        creds, _, host = rest.partition("@")
        if creds:
            return f"{scheme}//***:***@{host}"
    return url


def _require_db():
    if users is None:
        raise RuntimeError("Database not initialised; connect_to_mongo() not called.")
    return users


# --------------------------------------------------------------------------- #
# Password hashing (bcrypt)
# --------------------------------------------------------------------------- #
def get_password_hash(password: str) -> str:
    """Hash a password with a per-password salt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time-ish password check; never raises."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# User queries
# --------------------------------------------------------------------------- #
async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Return the user document for ``username`` or ``None``."""
    return await _require_db().find_one({"username": username})


async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Return the user document for ``email`` or ``None``."""
    return await _require_db().find_one({"email": email})


async def get_user_by_id(user_id: Any) -> Optional[Dict[str, Any]]:
    """Return a user document by ``_id`` (accepts str or ObjectId)."""
    oid = user_id if isinstance(user_id, ObjectId) else ObjectId(str(user_id))
    return await _require_db().find_one({"_id": oid})


async def create_user(user_data: Dict[str, Any]) -> ObjectId:
    """Insert a new user.  ``password`` is hashed and never stored raw."""
    doc = dict(user_data)
    raw_password = doc.pop("password", None)
    if not raw_password:
        raise ValueError("A password is required.")
    doc["hashed_password"] = get_password_hash(raw_password)
    doc["created_at"] = datetime.now(timezone.utc)
    doc["last_login"] = None
    doc["is_active"] = True
    result = await _require_db().insert_one(doc)
    logger.info("Created user %s", doc.get("username"))
    return result.inserted_id


async def update_last_login(user_id: Any) -> None:
    """Update ``last_login`` for a user, converting the id to an ObjectId.

    The original implementation passed a raw string id which never matched a
    document, so this field silently failed to update.
    """
    oid = user_id if isinstance(user_id, ObjectId) else ObjectId(str(user_id))
    await _require_db().update_one(
        {"_id": oid}, {"$set": {"last_login": datetime.now(timezone.utc)}}
    )
