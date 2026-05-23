"""Recoverable backups of source data taken before permanent deletions.

Backups live in the project's existing MongoDB (GridFS bucket ``backups`` +
``delete_backups`` metadata) so no paid service is introduced.  Every backup
is restricted to the owning user, expires after ``settings.backup_ttl_days``
(TTL index), and is sized within ``settings.max_upload_bytes``.

Formats (per ``meta["format"]``):

* ``csv``      – gzipped CSV of MySQL rows / column values (header included).
* ``jsonl``    – gzipped Extended JSON lines of MongoDB documents (type-safe:
                 ``$oid``/``$date`` round-trip through ``mongoimport``).

Restoration steps are documented in README.md and returned by the API
(``/backups/`` and ``/backups/{id}/download/`` include a ``restoration`` hint).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId

from config import settings

logger = logging.getLogger(__name__)

_fs = None
_fs_db = None

# Human-facing restore instructions per backup kind.
RESTORATION: Dict[str, str] = {
    "mysql_rows": (
        "Download the .csv.gz backup, gunzip it, then re-insert the rows "
        "(e.g. LOAD DATA INFILE or an INSERT script) using the included "
        "primary-key columns to identify the missing records."
    ),
    "mysql_columns": (
        "Download the .csv.gz backup (column values; schema is stored in the "
        "backup metadata), re-add the column with ALTER TABLE ... ADD COLUMN "
        "using the recorded definition, then reload the values keyed by the "
        "primary key."
    ),
    "mongo_documents": (
        "Download the .jsonl.gz backup (BSON Extended JSON), gunzip it, then "
        "run: mongoimport --uri <uri> --db <db> --collection <coll> "
        "--file backup.jsonl"
    ),
    "mongo_field_values": (
        "Download the .jsonl.gz backup (BSON Extended JSON of {_id, <field>}), "
        "gunzip it, then re-set the field per document with a mongo script or "
        "mongoimport of the values."
    ),
}


class BackupError(Exception):
    """Backup could not be created / found.  ``status_code`` guides HTTP."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _require_db():
    import database

    if database.db is None:
        raise BackupError("Database not initialised.", 503)
    return database.db


def _collection():
    return _require_db().delete_backups


def _bucket():
    """GridFS bucket for backup blobs (cached per database handle)."""
    global _fs, _fs_db
    db = _require_db()
    if _fs is None or _fs_db is not db:
        from motor.motor_asyncio import AsyncIOMotorGridFSBucket

        _fs = AsyncIOMotorGridFSBucket(db, bucket_name="backups")
        _fs_db = db
    return _fs


async def ensure_backup_indexes() -> None:
    """TTL on expiry (auto-purge) + owner lookup index."""
    await _collection().create_index("expires_at", expireAfterSeconds=0)
    await _collection().create_index([("owner_id", 1), ("created_at", -1)])


def _as_oid(value: Any) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(str(value))


async def create_backup(
    owner_id: Any,
    *,
    kind: str,
    source_name: str,
    payload: bytes,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store ``payload`` (already compressed) as an owned, expiring backup.

    Raises :class:`BackupError` (never partially succeeds silently) so callers
    can abort a destructive operation when a mandatory backup fails.
    """
    if not payload:
        raise BackupError("Backup payload is empty; refusing to continue.")
    if len(payload) > settings.max_upload_bytes:
        raise BackupError(
            f"Backup size ({len(payload) // (1024 * 1024)} MB) exceeds the "
            f"{settings.max_upload_bytes // (1024 * 1024)} MB backup limit.",
            413,
        )

    owner = _as_oid(owner_id)
    backup_id = uuid.uuid4().hex
    fs = _bucket()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=settings.backup_ttl_days)

    file_id = None
    try:
        file_id = await fs.upload_from_stream(
            backup_id,
            payload,
            metadata={"backup_id": backup_id, "kind": kind, "owner_id": owner},
        )
        doc = {
            "backup_id": backup_id,
            "owner_id": owner,
            "kind": kind,
            "source_name": source_name,
            "created_at": now,
            "expires_at": expires,
            "size_bytes": len(payload),
            "file_id": file_id,
            "format": (meta or {}).get("format", "csv"),
            "encoding": "gzip",
            "meta": dict(meta or {}),
            "restoration": RESTORATION.get(kind, "Contact your administrator."),
        }
        await _collection().insert_one(doc)
    except BackupError:
        raise
    except Exception as exc:
        if file_id is not None:  # pragma: no cover - best effort cleanup
            try:
                await fs.delete(file_id)
            except Exception:
                pass
        logger.error("Backup creation failed: %s", exc, exc_info=True)
        raise BackupError(
            "The backup could not be stored; no data was modified."
        )

    logger.info(
        "Backup %s created (%s, %d bytes, owner=%s)",
        backup_id, kind, len(payload), owner,
    )
    return {
        "backup_id": backup_id,
        "kind": kind,
        "source_name": source_name,
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "expires_in_days": settings.backup_ttl_days,
        "size_bytes": len(payload),
        "format": doc["format"],
        "restoration": doc["restoration"],
    }


async def verify_backup(owner_id: Any, backup_id: str) -> Dict[str, Any]:
    """Read the backup back and confirm it matches what was written.

    Called after :func:`create_backup` when the user requested a mandatory
    backup — deletion must not proceed unless the bytes are really stored.
    """
    data, meta = await read_backup(owner_id, backup_id)
    expected = int(meta.get("size_bytes") or -1)
    if len(data) != expected:
        raise BackupError(
            "Backup verification failed (stored size mismatch); "
            "no data was modified."
        )
    return meta


async def list_backups(owner_id: Any) -> List[Dict[str, Any]]:
    """Metadata for all of a user's live backups (newest first)."""
    owner = _as_oid(owner_id)
    out: List[Dict[str, Any]] = []
    cursor = _collection().find({"owner_id": owner}).sort("created_at", -1)
    async for doc in cursor:
        out.append(_public_meta(doc))
    return out


async def get_backup_meta(owner_id: Any, backup_id: str) -> Dict[str, Any]:
    owner = _as_oid(owner_id)
    doc = await _collection().find_one(
        {"backup_id": backup_id, "owner_id": owner}
    )
    if doc is None:
        # Never reveal whether the id exists for another user.
        raise BackupError("Backup not found or access denied.", 404)
    return _public_meta(doc)


async def read_backup(owner_id: Any, backup_id: str) -> Tuple[bytes, Dict[str, Any]]:
    """Return ``(payload, meta)`` for an owned backup (404 when not owned)."""
    owner = _as_oid(owner_id)
    doc = await _collection().find_one(
        {"backup_id": backup_id, "owner_id": owner}
    )
    if doc is None:
        raise BackupError("Backup not found or access denied.", 404)
    try:
        stream = await _bucket().open_download_stream(doc["file_id"])
        data = await stream.read()
    except Exception as exc:
        logger.error("Backup blob unreadable: %s", exc)
        raise BackupError("Backup data could not be read.", 500)
    return data, _public_meta(doc)


async def delete_backup(owner_id: Any, backup_id: str) -> str:
    """Delete an owned backup (metadata + GridFS blob).

    Idempotent: a missing/already-deleted backup returns ``"missing"`` rather
    than raising, so cleanup retries are safe.  Raises :class:`BackupError`
    only when the blob cannot be removed and still exists (the metadata doc is
    then left in place so a retry can find it).
    """
    owner = _as_oid(owner_id)
    doc = await _collection().find_one(
        {"backup_id": backup_id, "owner_id": owner}
    )
    if doc is None:
        return "missing"
    file_id = doc.get("file_id")
    if file_id is not None:
        try:
            await _bucket().delete(file_id)
        except Exception as exc:
            # gridfs.NoFile means the blob is already gone -> idempotent OK.
            # Anything else: verify whether the blob really remains.
            try:
                still_there = await _require_db().backups.files.find_one(
                    {"_id": file_id}, {"_id": 1}
                )
            except Exception:
                still_there = {"_id": file_id}
            if still_there:
                logger.error("Backup blob delete failed for %s: %s", backup_id, exc)
                raise BackupError(
                    "The recovery snapshot blob could not be deleted; the "
                    "backup entry was kept so cleanup can be retried.",
                    500,
                )
    await _collection().delete_one({"backup_id": backup_id, "owner_id": owner})
    logger.info("Backup %s deleted (owner=%s)", backup_id, owner)
    return "deleted"


def _as_utc(value):
    """Naive Mongo datetime -> aware UTC (browsers misparse offset-less ISO)."""
    if value is None:
        return None
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _public_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(doc.get("meta") or {})
    return {
        "backup_id": doc.get("backup_id"),
        "kind": doc.get("kind"),
        "source_name": doc.get("source_name"),
        "created_at": _as_utc(doc.get("created_at")),
        "expires_at": _as_utc(doc.get("expires_at")),
        "size_bytes": doc.get("size_bytes", 0),
        "format": doc.get("format", "csv"),
        "encoding": doc.get("encoding", "gzip"),
        "meta": meta,
        "restoration": doc.get("restoration", ""),
    }
