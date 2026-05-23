"""Transformation history and working-copy version snapshots.

Every data-changing operation (import, transformation apply, working-copy
edits, permanent source deletions, write-backs, restorations) records an
immutable **history entry**, and working-copy mutations additionally persist
an after-state **version snapshot** (gzipped CSV in the shared ``datasets``
GridFS bucket) so the working dataset can be restored within the seven-day
recovery window.

Design
------
* History *metadata* is kept beyond the recovery window so users can always
  audit what happened, but a metadata entry never implies restorability:
  ``restorable`` is computed from the associated version or backup expiry and
  downgraded to ``False`` once the window closes.
* Version blobs carry ``expires_at`` (``settings.backup_ttl_days``) backed by a
  MongoDB TTL index, keeping storage bounded.
* Ownership is enforced on every read/write by filtering on ``owner_id``.
* Entries never store credentials, API keys, or raw connection details --
  only a sanitized ``source_meta`` (database/table or db/collection).
"""

from __future__ import annotations

import gzip
import io
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId

from config import settings

logger = logging.getLogger(__name__)

# Operations that appear in the history panel.
OPERATIONS = (
    "import",
    "transform_apply",
    "working_copy_delete_rows",
    "working_copy_delete_columns",
    "working_copy_restore_rows",
    "working_copy_restore_columns",
    "working_copy_revert",
    "source_delete_mysql_rows",
    "source_delete_mysql_columns",
    "source_delete_mongo_documents",
    "source_delete_mongo_fields",
    "write_back_mysql",
    "write_back_mongodb",
    "restore_working_copy",
    "restore_source",
    "open_version",
)

_gridfs = None
_gridfs_db = None


class HistoryError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class VersionNotFound(HistoryError):
    def __init__(self, message: str = "Version not found."):
        super().__init__(message, 404)


def _gridfs_bucket():
    """Reuse the sessions GridFS bucket (same database, same blobs layout)."""
    global _gridfs, _gridfs_db
    import database

    if database.db is None:
        raise HistoryError("Database not initialised.", 503)
    if _gridfs is None or _gridfs_db is not database.db:
        from motor.motor_asyncio import AsyncIOMotorGridFSBucket

        _gridfs = AsyncIOMotorGridFSBucket(database.db, bucket_name="datasets")
        _gridfs_db = database.db
    return _gridfs


def _history_coll():
    import database

    if database.db is None:
        raise HistoryError("Database not initialised.", 503)
    return database.db.history


def _versions_coll():
    import database

    if database.db is None:
        raise HistoryError("Database not initialised.", 503)
    return database.db.session_versions


def _as_oid(value: Any) -> ObjectId:
    try:
        return value if isinstance(value, ObjectId) else ObjectId(str(value))
    except (InvalidId, TypeError):
        raise HistoryError("Invalid owner.", 400)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Mongo may return naive datetimes; normalise to aware UTC for compare."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


async def ensure_history_indexes() -> None:
    """Idempotently create indexes (TTL keeps version storage bounded)."""
    import database

    if database.db is None:
        return
    await database.db.session_versions.create_index("expires_at", expireAfterSeconds=0)
    await database.db.session_versions.create_index([("owner_id", 1), ("session_id", 1)])
    await database.db.session_versions.create_index(
        [("owner_id", 1), ("session_id", 1), ("version_number", -1)]
    )
    await database.db.history.create_index(
        [("owner_id", 1), ("session_id", 1), ("timestamp", -1)]
    )
    await database.db.history.create_index([("owner_id", 1), ("history_id", 1)])


# --------------------------------------------------------------------------- #
# Version snapshots (working-copy rollback artifacts)
# --------------------------------------------------------------------------- #
def _df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return gzip.compress(buf.getvalue().encode("utf-8"))


def _bytes_to_df(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(gzip.decompress(data)))


def _check_size(raw: bytes) -> None:
    if len(raw) > settings.max_upload_bytes:
        raise HistoryError(
            f"Version snapshot is {len(raw) // (1024 * 1024)} MB which exceeds "
            f"the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
            413,
        )


async def create_version(
    owner_id: Any,
    session_id: str,
    df: pd.DataFrame,
    *,
    history_id: Optional[str] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist the working-copy state as a new numbered version snapshot.

    Returns ``{version_id, version_number, created_at, expires_at,
    row_count, columns, size_bytes}``.  Raises :class:`HistoryError` when the
    snapshot cannot be stored -- callers that advertise rollbackability must
    treat that as a hard failure.
    """
    payload = _df_to_bytes(df)
    _check_size(payload)
    owner = _as_oid(owner_id)
    version_id = uuid.uuid4().hex
    now = _now()
    expires = now + timedelta(days=settings.backup_ttl_days)

    last = await _versions_coll().find_one(
        {"owner_id": owner, "session_id": session_id},
        sort=[("version_number", -1)],
    )
    version_number = int(last["version_number"]) + 1 if last else 1

    fs = _gridfs_bucket()
    file_id = await fs.upload_from_stream(
        f"{session_id}-v{version_number}-{version_id}",
        payload,
        metadata={
            "session_id": session_id,
            "kind": "version",
            "owner_id": owner,
            "version_id": version_id,
        },
    )
    doc = {
        "version_id": version_id,
        "owner_id": owner,
        "session_id": session_id,
        "version_number": version_number,
        "history_id": history_id,
        "label": label,
        "created_at": now,
        "expires_at": expires,
        "file_id": file_id,
        "size_bytes": len(payload),
        "row_count": int(len(df)),
        "columns": [str(c) for c in df.columns],
    }
    try:
        await _versions_coll().insert_one(doc)
    except Exception as exc:
        try:
            await fs.delete(file_id)
        except Exception:  # pragma: no cover - best effort
            pass
        raise HistoryError(f"Could not store version snapshot: {exc}", 500)
    logger.info(
        "Stored version v%d (%s) for session %s (%d rows)",
        version_number, version_id, session_id, len(df),
    )
    return {
        "version_id": version_id,
        "version_number": version_number,
        "created_at": now,
        "expires_at": expires,
        "row_count": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "size_bytes": len(payload),
    }


async def list_versions(owner_id: Any, session_id: str) -> List[Dict[str, Any]]:
    """Newest-first version snapshots for an owned session (unexpired only)."""
    owner = _as_oid(owner_id)
    now = _now()
    cursor = (
        _versions_coll()
        .find({"owner_id": owner, "session_id": session_id, "expires_at": {"$gt": now}})
        .sort("version_number", -1)
    )
    docs = [doc async for doc in cursor]
    presence = await _version_file_presence(docs)
    return [_public_version(d, snapshot_ok=presence.get(d["version_id"], False)) for d in docs]


async def _version_file_presence(docs: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Map version_id -> whether its GridFS blob actually exists.

    A version is only advertised as restorable when both the metadata and the
    stored snapshot data are present; a missing blob must never be claimed as
    a recoverable snapshot.
    """
    import database

    if not docs:
        return {}
    file_ids = [d.get("file_id") for d in docs if d.get("file_id") is not None]
    present_ids = set()
    if file_ids and database.db is not None:
        cursor = database.db.datasets.files.find({"_id": {"$in": file_ids}}, {"_id": 1})
        async for f in cursor:
            present_ids.add(f["_id"])
    return {
        d["version_id"]: d.get("file_id") in present_ids
        for d in docs
        if d.get("version_id")
    }


async def get_version(owner_id: Any, session_id: str, version_id: str) -> Dict[str, Any]:
    owner = _as_oid(owner_id)
    doc = await _versions_coll().find_one(
        {"owner_id": owner, "session_id": session_id, "version_id": version_id}
    )
    if doc is None:
        raise VersionNotFound("Version not found or access denied.")
    expires = _as_utc(doc.get("expires_at"))
    if expires and expires <= _now():
        raise VersionNotFound(
            "This version snapshot has expired (seven-day recovery window)."
        )
    return doc


async def get_version_by_id(owner_id: Any, version_id: str) -> Dict[str, Any]:
    """Fetch a version by its unique id (ownership enforced, expiry checked)."""
    owner = _as_oid(owner_id)
    doc = await _versions_coll().find_one(
        {"owner_id": owner, "version_id": version_id}
    )
    if doc is None:
        raise VersionNotFound("Version not found or access denied.")
    expires = _as_utc(doc.get("expires_at"))
    if expires and expires <= _now():
        raise VersionNotFound(
            "This version snapshot has expired (seven-day recovery window)."
        )
    return doc


async def read_version_df(owner_id: Any, session_id: str, version_id: str) -> pd.DataFrame:
    doc = await get_version(owner_id, session_id, version_id)
    fs = _gridfs_bucket()
    try:
        stream = await fs.open_download_stream(doc["file_id"])
        data = await stream.read()
    except Exception as exc:
        logger.error("Failed to read version blob: %s", exc)
        raise VersionNotFound("Version snapshot could not be read.")
    return _bytes_to_df(data)


async def read_version_df_by_id(owner_id: Any, version_id: str) -> pd.DataFrame:
    doc = await get_version_by_id(owner_id, version_id)
    fs = _gridfs_bucket()
    try:
        stream = await fs.open_download_stream(doc["file_id"])
        data = await stream.read()
    except Exception as exc:
        logger.error("Failed to read version blob: %s", exc)
        raise VersionNotFound("Version snapshot could not be read.")
    return _bytes_to_df(data)


def _public_version(
    doc: Dict[str, Any], snapshot_ok: Optional[bool] = None
) -> Dict[str, Any]:
    expires = _as_utc(doc.get("expires_at"))
    within_window = bool(expires and expires > _now())
    # ``snapshot_ok=False`` downgrades restorable when the GridFS blob is
    # missing; ``None`` keeps the time-only verdict (blob not verified).
    restorable = within_window and (snapshot_ok is not False)
    out = {
        "version_id": doc["version_id"],
        "version_number": doc["version_number"],
        "session_id": doc["session_id"],
        "history_id": doc.get("history_id"),
        "label": doc.get("label"),
        # Normalise BOTH timestamps to aware UTC. FastAPI serialises naive
        # datetimes WITHOUT an offset, which browsers parse as *local* time
        # (hours early in non-UTC zones), while aware values convert to the
        # correct wall-clock time. Returning created_at naive and expires_at
        # aware made the recovery window appear to start 5½ hours (UTC+5:30)
        # after the version's own creation time.
        "created_at": _as_utc(doc.get("created_at")),
        "expires_at": expires,
        "restorable": restorable,
        "row_count": doc.get("row_count", 0),
        "columns": doc.get("columns", []),
        "size_bytes": doc.get("size_bytes", 0),
    }
    if snapshot_ok is not None:
        out["snapshot_available"] = bool(snapshot_ok)
    return out


# --------------------------------------------------------------------------- #
# History entries
# --------------------------------------------------------------------------- #
# Source categories used for section-scoped views (Files / MySQL / MongoDB).
SOURCE_TYPES: Tuple[str, str, str] = ("file", "mysql", "mongodb")
# Historical spellings that map unambiguously onto a canonical category.
_SOURCE_TYPE_ALIASES = {"database": "mysql", "mongodb_stored": "mongodb"}
_SOURCE_TYPE_LABELS = {"file": "file", "mysql": "MySQL", "mongodb": "MongoDB"}


def normalize_source_type(value: Optional[str]) -> Optional[str]:
    """Validate a requested source category (``None``/empty passes through).

    Raises :class:`HistoryError` (400) for anything outside the supported
    categories so an unknown filter can never silently return everything.
    """
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    canonical = _SOURCE_TYPE_ALIASES.get(raw, raw)
    if canonical not in SOURCE_TYPES:
        raise HistoryError(
            f"Invalid source_type '{value}'. Supported: file, mysql, mongodb.",
            400,
        )
    return canonical


def classify_source_type(doc: Dict[str, Any]) -> Optional[str]:
    """Best-effort category for a raw history document (legacy entries).

    Returns ``None`` when the category cannot be proven from the document's
    own authoritative fields — ambiguous legacy entries must never be guessed
    into a source-specific view.
    """
    raw = str(doc.get("source_type") or "").strip().lower()
    if raw in SOURCE_TYPES:
        return raw
    if raw in _SOURCE_TYPE_ALIASES:
        return _SOURCE_TYPE_ALIASES[raw]
    meta = doc.get("source_meta") or {}
    # Legacy entries recorded before ``source_type`` existed still carry the
    # session's source metadata, which is authoritative for DB-backed ops.
    if meta.get("database") and meta.get("collection"):
        return "mongodb"
    if meta.get("database") and meta.get("table"):
        return "mysql"
    return None


def _source_type_filter(source: str) -> Dict[str, Any]:
    """Mongo query fragment matching one canonical source category.

    Matches current entries (canonical ``source_type``), documented legacy
    aliases, and pre-``source_type`` documents whose own ``source_meta``
    proves the category (e.g. ``database``+``table``).  Entries with no
    reliable marker are excluded from every category on purpose.
    """
    aliases = {
        "file": ["file"],
        "mysql": ["mysql", "database"],
        "mongodb": ["mongodb", "mongodb_stored"],
    }[source]
    branches: List[Dict[str, Any]] = [{"source_type": {"$in": aliases}}]
    legacy: Dict[str, Any] = {"source_type": {"$in": [None, ""]}}
    if source == "mongodb":
        legacy["source_meta.database"] = {"$exists": True}
        legacy["source_meta.collection"] = {"$exists": True}
        branches.append({"$and": [legacy]})
    elif source == "mysql":
        legacy["source_meta.database"] = {"$exists": True}
        legacy["source_meta.table"] = {"$exists": True}
        branches.append({"$and": [legacy]})
    # Legacy file entries have no distinguishing metadata: they are only
    # matched via a stored ``source_type`` (or alias) — never guessed.
    return {"$or": branches}


def source_type_label(source: Optional[str]) -> str:
    """Human label for a category (used in confirmation messages)."""
    return _SOURCE_TYPE_LABELS.get(source or "", "history")


async def record(
    owner_id: Any,
    *,
    session_id: Optional[str],
    operation: str,
    status: str = "success",
    source_type: Optional[str] = None,
    source_name: Optional[str] = None,
    source_meta: Optional[Dict[str, Any]] = None,
    before: Optional[Dict[str, Any]] = None,
    after: Optional[Dict[str, Any]] = None,
    summary: str = "",
    column: Optional[str] = None,
    transformation: Optional[Dict[str, Any]] = None,
    backup_id: Optional[str] = None,
    restorable: bool = False,
    rollback_expires_at: Optional[datetime] = None,
    version_id: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append one immutable history entry; returns the public entry.

    ``restorable`` must only be set when a verified recovery artifact
    (version snapshot or backup) actually exists.  ``rollback_expires_at`` is
    the artifact's expiry -- after that instant the entry is presented as
    non-restorable even if the metadata remains.
    """
    owner = _as_oid(owner_id)
    history_id = uuid.uuid4().hex
    now = _now()
    doc = {
        "history_id": history_id,
        "owner_id": owner,
        "session_id": session_id,
        "operation": operation,
        "status": status,
        "timestamp": now,
        "source_type": source_type,
        "source_name": source_name,
        "source_meta": source_meta or {},
        "before": before or {},
        "after": after or {},
        "summary": summary,
        "column": column,
        "transformation": transformation,
        "backup_id": backup_id,
        "restorable": bool(restorable),
        "rollback_expires_at": rollback_expires_at,
        "version_id": version_id,
        "warnings": list(warnings or []),
        "extra": extra or {},
    }
    try:
        await _history_coll().insert_one(doc)
    except Exception as exc:  # pragma: no cover - never break the main flow
        logger.error("Failed to record history entry: %s", exc)
        return {"history_id": history_id, "operation": operation, "status": status}
    logger.info(
        "History: %s (%s) session=%s status=%s",
        operation, history_id, session_id, status,
    )
    return _public_entry(doc)


def _public_entry(
    doc: Dict[str, Any], version: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    expires = _as_utc(doc.get("rollback_expires_at"))
    restorable = bool(doc.get("restorable")) and bool(expires) and expires > _now()
    return {
        "history_id": doc.get("history_id"),
        "session_id": doc.get("session_id"),
        "operation": doc.get("operation"),
        "status": doc.get("status"),
        # Aware UTC (see _public_version): naive values render hours early
        # in the browser while expiry fields render correctly.
        "timestamp": _as_utc(doc.get("timestamp")),
        "source_type": doc.get("source_type"),
        "source_name": doc.get("source_name"),
        "source_meta": doc.get("source_meta") or {},
        "before": doc.get("before") or {},
        "after": doc.get("after") or {},
        "summary": doc.get("summary", ""),
        "column": doc.get("column"),
        "transformation": doc.get("transformation"),
        "backup_id": doc.get("backup_id"),
        "restorable": restorable,
        "rollback_expires_at": expires,
        "version_id": doc.get("version_id"),
        # Nested snapshot metadata when the entry has a linked version; the
        # ``version.restorable`` flag already accounts for expiry AND whether
        # the GridFS snapshot blob is actually present.
        "version": version,
        "warnings": doc.get("warnings") or [],
        "extra": doc.get("extra") or {},
    }


async def _entries_with_versions(
    owner: ObjectId, entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Attach verified snapshot metadata to raw history docs."""
    version_ids = [e["version_id"] for e in entries if e.get("version_id")]
    if not version_ids:
        return [_public_entry(e) for e in entries]
    cursor = _versions_coll().find(
        {"owner_id": owner, "version_id": {"$in": version_ids}}
    )
    vdocs = {d["version_id"]: d async for d in cursor}
    presence = await _version_file_presence(list(vdocs.values()))
    out: List[Dict[str, Any]] = []
    for e in entries:
        vdoc = vdocs.get(e.get("version_id") or "")
        version = (
            _public_version(vdoc, snapshot_ok=presence.get(vdoc["version_id"], False))
            if vdoc
            else None
        )
        out.append(_public_entry(e, version=version))
    return out


async def list_history(
    owner_id: Any,
    session_id: Optional[str] = None,
    limit: int = 50,
    source_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Newest-first history for the owner.

    Optionally scoped to one session and/or one source category
    (``file`` / ``mysql`` / ``mongodb``).  The category filter is applied
    inside the database query and is always combined with the owner scope;
    entries whose category cannot be proven are excluded from category views
    (but still visible in the unfiltered audit trail).
    """
    owner = _as_oid(owner_id)
    query: Dict[str, Any] = {"owner_id": owner}
    if session_id:
        query["session_id"] = session_id
    source = normalize_source_type(source_type)
    if source:
        query["$and"] = [_source_type_filter(source)]
    cursor = _history_coll().find(query).sort("timestamp", -1).limit(limit)
    docs = [doc async for doc in cursor]
    return await _entries_with_versions(owner, docs)


async def get_entry(owner_id: Any, history_id: str) -> Dict[str, Any]:
    owner = _as_oid(owner_id)
    doc = await _history_coll().find_one(
        {"owner_id": owner, "history_id": history_id}
    )
    if doc is None:
        raise HistoryError("History entry not found or access denied.", 404)
    enriched = await _entries_with_versions(owner, [doc])
    return enriched[0]


# --------------------------------------------------------------------------- #
# Selective / full history clearing (audit cleanup)
# --------------------------------------------------------------------------- #
async def _shared_reference_ids(
    owner: ObjectId,
    field: str,
    ids: Set[str],
    exclude_history_ids: Set[str],
) -> Set[str]:
    """Ids of ``field`` (version_id / backup_id) still referenced by *other*
    surviving history entries owned by ``owner``.

    A snapshot that another entry still points at must NOT be deleted — it may
    be the only recovery artifact for that entry's operation.
    """
    if not ids:
        return set()
    cursor = _history_coll().find(
        {
            "owner_id": owner,
            field: {"$in": sorted(ids)},
            "history_id": {"$nin": sorted(exclude_history_ids)},
        },
        {field: 1},
    )
    out: Set[str] = set()
    async for doc in cursor:
        value = doc.get(field)
        if value:
            out.add(value)
    return out


async def delete_version_artifact(owner_id: Any, version_id: str) -> str:
    """Delete a version snapshot (metadata doc + GridFS blob).

    Idempotent: returns ``"missing"`` when the version doc is already gone.
    Raises :class:`HistoryError` (500) only when the blob cannot be removed
    and still exists — the metadata doc is then left so a retry can find it.
    """
    owner = _as_oid(owner_id)
    doc = await _versions_coll().find_one(
        {"owner_id": owner, "version_id": version_id}
    )
    if doc is None:
        return "missing"
    file_id = doc.get("file_id")
    if file_id is not None:
        try:
            await _gridfs_bucket().delete(file_id)
        except Exception as exc:
            import database

            try:
                still_there = await database.db.datasets.files.find_one(
                    {"_id": file_id}, {"_id": 1}
                )
            except Exception:
                still_there = {"_id": file_id}
            if still_there:
                logger.error("Version blob delete failed for %s: %s", version_id, exc)
                raise HistoryError(
                    "The version snapshot blob could not be deleted; the "
                    "version entry was kept so cleanup can be retried.",
                    500,
                )
    await _versions_coll().delete_one({"owner_id": owner, "version_id": version_id})
    logger.info("Version snapshot %s deleted (owner=%s)", version_id, owner)
    return "deleted"


def _empty_clear_result(requested: int) -> Dict[str, Any]:
    return {
        "requested": requested,
        "entries_deleted": 0,
        "entries_missing": [],
        "snapshots_deleted": 0,
        "snapshots_retained": 0,
        "backups_deleted": 0,
        "backups_retained": 0,
        "failures": [],
    }


async def clear_history_entries(
    owner_id: Any, history_ids: Sequence[str]
) -> Dict[str, Any]:
    """Delete owned history entries and safely clean up their snapshots.

    Safety rules (enforced server-side, never from client state):

    * Only entries actually owned by ``owner_id`` are touched; unknown or
      cross-user ids are reported in ``entries_missing`` and never deleted.
    * A version snapshot or recovery backup is deleted ONLY when no other
      surviving history entry of the same owner still references it.
    * Missing / already-deleted artifacts are handled idempotently.
    * A failure to remove one artifact never aborts the rest; the structured
      result reports exactly what was removed, retained, and failed so the
      caller can retry safely.  Nothing reports success it did not achieve.
    * The active working dataset (sessions + their GridFS blobs) and any live
      source database are NEVER touched.
    """
    import backups as backup_store

    owner = _as_oid(owner_id)
    ids = list(dict.fromkeys(str(h).strip() for h in history_ids if str(h).strip()))
    result = _empty_clear_result(len(ids))
    if not ids:
        return result

    cursor = _history_coll().find({"owner_id": owner, "history_id": {"$in": ids}})
    owned = {d["history_id"]: d async for d in cursor}
    result["entries_missing"] = [i for i in ids if i not in owned]
    if not owned:
        return result

    owned_ids = set(owned.keys())
    version_ids = {d["version_id"] for d in owned.values() if d.get("version_id")}
    backup_ids = {d["backup_id"] for d in owned.values() if d.get("backup_id")}

    shared_versions = await _shared_reference_ids(
        owner, "version_id", version_ids, owned_ids
    )
    shared_backups = await _shared_reference_ids(
        owner, "backup_id", backup_ids, owned_ids
    )

    for vid in sorted(version_ids):
        if vid in shared_versions:
            # Another surviving entry still needs this snapshot.
            result["snapshots_retained"] += 1
            continue
        try:
            outcome = await delete_version_artifact(owner, vid)
            if outcome == "deleted":
                result["snapshots_deleted"] += 1
        except Exception as exc:
            result["failures"].append(
                {"target": "version", "id": vid, "error": str(exc)}
            )

    for bid in sorted(backup_ids):
        if bid in shared_backups:
            result["backups_retained"] += 1
            continue
        try:
            outcome = await backup_store.delete_backup(owner, bid)
            if outcome == "deleted":
                result["backups_deleted"] += 1
        except Exception as exc:
            result["failures"].append(
                {"target": "backup", "id": bid, "error": str(exc)}
            )

    for hid in sorted(owned_ids):
        try:
            res = await _history_coll().delete_one(
                {"owner_id": owner, "history_id": hid}
            )
            if res.deleted_count:
                result["entries_deleted"] += 1
        except Exception as exc:
            result["failures"].append(
                {"target": "history", "id": hid, "error": str(exc)}
            )

    logger.info(
        "History clear: owner=%s requested=%d deleted=%d snapshots=%d "
        "backups=%d retained_v=%d retained_b=%d failures=%d",
        owner, result["requested"], result["entries_deleted"],
        result["snapshots_deleted"], result["backups_deleted"],
        result["snapshots_retained"], result["backups_retained"],
        len(result["failures"]),
    )
    return result


async def clear_all_history(
    owner_id: Any, source_type: Optional[str] = None
) -> Dict[str, Any]:
    """Delete every history entry (and eligible snapshots) of one owner.

    When ``source_type`` is given, ONLY that source category is cleared —
    other categories' entries, snapshots and backups are untouched.  The
    category filter and the owner scope are both enforced server-side.
    """
    owner = _as_oid(owner_id)
    query: Dict[str, Any] = {"owner_id": owner}
    source = normalize_source_type(source_type)
    if source:
        query["$and"] = [_source_type_filter(source)]
    cursor = _history_coll().find(query, {"history_id": 1})
    ids = [d["history_id"] async for d in cursor]
    return await clear_history_entries(owner_id, ids)


async def history_stats(
    owner_id: Any, source_type: Optional[str] = None
) -> Dict[str, Any]:
    """Counts used by the clear-all confirmation dialog (owner-scoped).

    ``source_type`` scopes the counts to one source category, mirroring the
    category-scoped clear so the dialog never over- or under-states impact.
    """
    owner = _as_oid(owner_id)
    query: Dict[str, Any] = {"owner_id": owner}
    source = normalize_source_type(source_type)
    if source:
        query["$and"] = [_source_type_filter(source)]
    total = await _history_coll().count_documents(query)
    with_version = await _history_coll().count_documents(
        {**query, "version_id": {"$ne": None}}
    )
    with_backup = await _history_coll().count_documents(
        {**query, "backup_id": {"$ne": None}}
    )
    out: Dict[str, Any] = {
        "total_entries": total,
        "entries_with_snapshot": with_version,
        "entries_with_backup": with_backup,
    }
    if source:
        out["source_type"] = source
    return out
