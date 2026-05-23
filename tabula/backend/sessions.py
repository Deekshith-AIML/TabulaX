"""Dataset/session storage.

Replaces the original process-local ``df_storage`` dict, which lost data on
restart, leaked memory, could not be shared across workers, and had weak
ownership guarantees.

Design
------
* Each uploaded/loaded dataset becomes an **owned session** with a random
  ``session_id`` and an ``owner_id`` (the authenticated user's ``_id``).
* DataFrames are serialised to gzipped CSV and stored in **GridFS**, avoiding
  the 16 MB document limit and keeping large blobs out of the metadata
  collection.  Metadata (columns, source, timestamps) lives in the ``sessions``
  collection.
* Ownership is enforced on **every** read/write by filtering on ``owner_id``;
  a session id alone is never sufficient.
* Sessions carry an ``expires_at`` timestamp backed by a MongoDB TTL index, so
  abandoned datasets are removed automatically.  A configurable size cap
  prevents unbounded growth.
* Because state lives in MongoDB, any number of backend workers share it and a
  container restart does not lose a user's data (subject to the TTL).
"""

from __future__ import annotations

import gzip
import io
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId

from config import settings

logger = logging.getLogger(__name__)

# Lazily-created GridFS buckets keyed by database handle id.
_gridfs = None
_gridfs_db = None


class SessionError(Exception):
    """Base class for session storage errors."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class SessionNotFound(SessionError):
    def __init__(self, message: str = "Session not found."):
        super().__init__(message, 404)


class SessionTooLarge(SessionError):
    def __init__(self, message: str = "Dataset exceeds the allowed size limit."):
        super().__init__(message, 413)


def _gridfs_bucket():
    """Return (and cache) the GridFS bucket, importing lazily to avoid cycles."""
    global _gridfs, _gridfs_db
    import database

    if database.db is None:
        raise SessionError("Database not initialised.", 503)
    if _gridfs is None or _gridfs_db is not database.db:
        from motor.motor_asyncio import AsyncIOMotorGridFSBucket

        _gridfs = AsyncIOMotorGridFSBucket(database.db, bucket_name="datasets")
        _gridfs_db = database.db
    return _gridfs


def _sessions():
    import database

    if database.sessions is None:
        raise SessionError("Database not initialised.", 503)
    return database.sessions


def _as_oid(value: Any) -> ObjectId:
    try:
        return value if isinstance(value, ObjectId) else ObjectId(str(value))
    except (InvalidId, TypeError):
        raise SessionNotFound("Session not found.")


def _as_utc(value: Any) -> Any:
    """Naive Mongo datetime -> aware UTC (browsers misparse offset-less ISO)."""
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to gzipped UTF-8 CSV."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return gzip.compress(buf.getvalue().encode("utf-8"))


def _bytes_to_df(data: bytes) -> pd.DataFrame:
    """Deserialise gzipped CSV back into a DataFrame."""
    return pd.read_csv(io.BytesIO(gzip.decompress(data)))


def _check_size(raw_bytes: bytes) -> None:
    if len(raw_bytes) > settings.max_upload_bytes:
        raise SessionTooLarge(
            f"Dataset is {len(raw_bytes) // (1024 * 1024)} MB which exceeds the "
            f"{settings.max_upload_bytes // (1024 * 1024)} MB limit."
        )


# --------------------------------------------------------------------------- #
# Create / update
# --------------------------------------------------------------------------- #
async def create_session(
    owner_id: Any,
    df: pd.DataFrame,
    *,
    source: str = "file",
    source_name: Optional[str] = None,
    file_type: Optional[str] = None,
    source_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Store ``df`` as a new owned session and return its ``session_id``."""
    if len(df) > settings.max_rows_per_dataset:
        raise SessionTooLarge(
            f"Dataset has {len(df)} rows which exceeds the "
            f"{settings.max_rows_per_dataset} row limit."
        )

    payload = _df_to_bytes(df)
    _check_size(payload)

    owner = _as_oid(owner_id)
    session_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    fs = _gridfs_bucket()

    file_id = await fs.upload_from_stream(
        f"{session_id}-original", payload, metadata={"session_id": session_id, "kind": "original"}
    )

    doc = {
        "session_id": session_id,
        "owner_id": owner,
        "created_at": now,
        "expires_at": now + timedelta(minutes=settings.session_ttl_minutes),
        "source": source,
        "source_name": source_name,
        "file_type": file_type,
        # Source-specific identity metadata (e.g. MongoDB _id type) used to
        # safely map working-copy rows back to real records on permanent delete.
        "source_meta": source_meta,
        "columns": list(df.columns),
        # Column order at import time.  ``columns`` tracks the CURRENT working
        # dataset (it is updated on every edit); ``original_columns`` stays
        # fixed so deleted columns can be restored to their original position.
        "original_columns": list(df.columns),
        "row_count": int(len(df)),
        "original_file_id": file_id,
        "transformed_file_id": None,
        "transformed_columns": [],
    }
    await _sessions().insert_one(doc)
    logger.info(
        "Created session %s for user %s (%d rows, %d cols, %s)",
        session_id, owner, len(df), df.shape[1], source,
    )
    return session_id


async def save_transformed(owner_id: Any, session_id: str, df: pd.DataFrame) -> None:
    """Persist the transformed DataFrame for an existing owned session."""
    doc = await _get_owned_doc(owner_id, session_id)
    payload = _df_to_bytes(df)
    _check_size(payload)

    fs = _gridfs_bucket()
    # Replace any previous transformed blob.
    if doc.get("transformed_file_id"):
        try:
            await fs.delete(doc["transformed_file_id"])
        except Exception:  # pragma: no cover - best effort
            pass
    file_id = await fs.upload_from_stream(
        f"{session_id}-transformed", payload,
        metadata={"session_id": session_id, "kind": "transformed"},
    )
    await _sessions().update_one(
        {"session_id": session_id, "owner_id": doc["owner_id"]},
        {
            "$set": {
                "transformed_file_id": file_id,
                "transformed_columns": list(df.columns),
                # Keep session metadata in sync with the CURRENT working
                # dataset so listings, previews, and column restores never
                # see import-time row counts or stale column names.  The
                # internal row-id column is not part of the user-visible
                # column list (this module cannot import app for the
                # constant — circular import).
                "columns": [c for c in df.columns if c != "_tabula_row_id"],
                "row_count": int(len(df)),
                "transformed_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc)
                + timedelta(minutes=settings.session_ttl_minutes),
            }
        },
    )


async def set_source_meta(
    owner_id: Any, session_id: str, meta: Dict[str, Any]
) -> None:
    """Merge ``meta`` into a session's ``source_meta`` (e.g. restore markers)."""
    doc = await _get_owned_doc(owner_id, session_id)
    merged = dict(doc.get("source_meta") or {})
    merged.update(meta)
    await _sessions().update_one(
        {"session_id": session_id, "owner_id": doc["owner_id"]},
        {"$set": {"source_meta": merged}},
    )


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
async def _get_owned_doc(owner_id: Any, session_id: str) -> Dict[str, Any]:
    """Fetch a session doc, enforcing ownership.  Raises SessionNotFound."""
    owner = _as_oid(owner_id)
    doc = await _sessions().find_one({"session_id": session_id, "owner_id": owner})
    if doc is None:
        # Do not reveal whether the id exists for another user.
        raise SessionNotFound("Session not found or access denied.")
    return doc


async def get_original_df(owner_id: Any, session_id: str) -> pd.DataFrame:
    doc = await _get_owned_doc(owner_id, session_id)
    if not doc.get("original_file_id"):
        raise SessionNotFound("Session has no original data.")
    return await _read_gridfs(doc["original_file_id"])


async def get_transformed_df(owner_id: Any, session_id: str) -> Optional[pd.DataFrame]:
    doc = await _get_owned_doc(owner_id, session_id)
    if not doc.get("transformed_file_id"):
        return None
    return await _read_gridfs(doc["transformed_file_id"])


async def _read_gridfs(file_id) -> pd.DataFrame:
    fs = _gridfs_bucket()
    try:
        stream = await fs.open_download_stream(file_id)
        data = await stream.read()
    except Exception as exc:
        logger.error("Failed to read dataset blob: %s", exc)
        raise SessionNotFound("Stored dataset could not be read (it may have expired).")
    return _bytes_to_df(data)


async def get_session_meta(owner_id: Any, session_id: str) -> Dict[str, Any]:
    """Return metadata (no data blob) for an owned session."""
    doc = await _get_owned_doc(owner_id, session_id)
    return {
        "session_id": doc["session_id"],
        "source": doc.get("source"),
        "source_name": doc.get("source_name"),
        "file_type": doc.get("file_type"),
        "source_meta": doc.get("source_meta") or {},
        "columns": doc.get("columns", []),
        "original_columns": doc.get("original_columns")
        or doc.get("columns", []),
        "row_count": doc.get("row_count", 0),
        "has_transformed": bool(doc.get("transformed_file_id")),
        # Aware UTC: naive Mongo values serialise without an offset and the
        # browser would render them as local time (hours off).
        "created_at": _as_utc(doc.get("created_at")),
        "expires_at": _as_utc(doc.get("expires_at")),
    }


async def list_sessions(owner_id: Any) -> List[Dict[str, Any]]:
    """List metadata for all of a user's live sessions."""
    owner = _as_oid(owner_id)
    cursor = _sessions().find({"owner_id": owner}).sort("created_at", -1)
    out = []
    async for doc in cursor:
        out.append(
            {
                "session_id": doc["session_id"],
                "source": doc.get("source"),
                "source_name": doc.get("source_name"),
                "row_count": doc.get("row_count", 0),
                "created_at": _as_utc(doc.get("created_at")),
                "expires_at": _as_utc(doc.get("expires_at")),
            }
        )
    return out


async def delete_session(owner_id: Any, session_id: str) -> None:
    """Delete a session and its data blobs.  Enforces ownership."""
    doc = await _get_owned_doc(owner_id, session_id)
    fs = _gridfs_bucket()
    for key in ("original_file_id", "transformed_file_id"):
        if doc.get(key):
            try:
                await fs.delete(doc[key])
            except Exception:  # pragma: no cover - best effort
                pass
    await _sessions().delete_one({"session_id": session_id, "owner_id": doc["owner_id"]})


async def extend_expiry(owner_id: Any, session_id: str) -> None:
    """Slide the expiry forward on activity (best effort)."""
    try:
        doc = await _get_owned_doc(owner_id, session_id)
    except SessionError:
        return
    await _sessions().update_one(
        {"session_id": doc["session_id"], "owner_id": doc["owner_id"]},
        {"$set": {"expires_at": datetime.now(timezone.utc)
                  + timedelta(minutes=settings.session_ttl_minutes)}},
    )


# --------------------------------------------------------------------------- #
# Transformation registry
# --------------------------------------------------------------------------- #
# Generated transformation code is stored server-side and referenced by an
# opaque id.  Clients never send Python source to the apply endpoints; they
# send a transformation_id that is looked up and ownership-checked here.  This
# is the core control that removes arbitrary client-supplied code execution.
_TRANSFORMATION_TTL_HOURS = 24


def _transformations():
    import database

    if database.db is None:
        raise SessionError("Database not initialised.", 503)
    return database.db.transformations


async def ensure_transformation_indexes() -> None:
    """Idempotently create indexes for the transformations collection."""
    import database

    if database.db is None:
        return
    coll = database.db.transformations
    await coll.create_index("expires_at", expireAfterSeconds=0)
    await coll.create_index([("owner_id", 1), ("transformation_id", 1)])


async def save_transformation(
    owner_id: Any,
    code: str,
    *,
    classification: str,
    description: str = "",
    template: Optional[str] = None,
    used_llm: bool = False,
    column: Optional[str] = None,
    spec: Optional[Dict[str, Any]] = None,
    examples: Optional[List[List[str]]] = None,
) -> str:
    """Store generated code server-side and return its ``transformation_id``.

    ``spec`` is the planner's explicit transformation specification (strategy,
    kind, labels, assumptions); ``examples`` are the pairs the transformation
    was inferred from.  Model-backed transformations store no code but always
    store both, so apply-time execution can demonstrate the rule to the model.
    """
    import database

    if database.db is None:
        raise SessionError("Database not initialised.", 503)
    owner = _as_oid(owner_id)
    transformation_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    doc: Dict[str, Any] = {
        "transformation_id": transformation_id,
        "owner_id": owner,
        "code": code,
        "classification": classification,
        "description": description,
        "template": template,
        "used_llm": used_llm,
        "column": column,
        "created_at": now,
        "expires_at": now + timedelta(hours=_TRANSFORMATION_TTL_HOURS),
    }
    if spec is not None:
        doc["spec"] = spec
    if examples is not None:
        doc["examples"] = [[str(a), str(b)] for a, b in examples]
    await _transformations().insert_one(doc)
    return transformation_id


async def get_transformation(owner_id: Any, transformation_id: str) -> Dict[str, Any]:
    """Fetch a stored transformation, enforcing ownership.

    Raises :class:`SessionNotFound` when missing or not owned by ``owner_id``.
    """
    owner = _as_oid(owner_id)
    doc = await _transformations().find_one(
        {"transformation_id": transformation_id, "owner_id": owner}
    )
    if doc is None:
        raise SessionNotFound(
            "Transformation not found or expired. Please regenerate it."
        )
    return doc
