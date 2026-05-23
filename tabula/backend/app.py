"""TabulaX FastAPI application.

Security model (important):
    Transformation code is **generated server-side** and stored against an
    opaque ``transformation_id`` that is scoped to the authenticated user.  The
    ``/apply/*`` endpoints accept only a ``transformation_id`` — never raw
    Python from the client.  Generated code is validated by the AST sandbox in
    :mod:`sandbox` and executed in an isolated subprocess with time/memory
    limits.  This removes the arbitrary client-code execution path present in
    the original implementation.

Run with:  ``uvicorn app:app --reload``
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import hmac
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, field_validator

from config import settings
from database import (
    close_mongo_connection,
    connect_to_mongo,
    create_user,
    get_user_by_email,
    get_user_by_username,
    update_last_login,
    verify_password,
)
from security import (
    create_access_token,
    get_current_active_user,
    get_current_user,
)
import sessions as session_store
import history as history_store
import quality as quality_module
from transform import (
    GenerationError,
    classify,
    generate,
    parse_examples_payload,
)
from transform.apply import ApplyError, apply_to_series
from transform.model_exec import ModelExecutionError, apply_model_backed
from transform import engine as general_engine
from transform.spec import STRATEGY_MODEL, TransformationSpec, family_label
import llm

try:  # MySQL driver may be absent in some environments
    import db_mysql
except Exception:  # pragma: no cover
    db_mysql = None  # type: ignore

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tabulax")


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting TabulaX (env=%s, llm_enabled=%s)", settings.environment, settings.llm_enabled)
    await connect_to_mongo()
    await session_store.ensure_transformation_indexes()
    import backups as backup_store

    await backup_store.ensure_backup_indexes()
    await history_store.ensure_history_indexes()
    yield
    await close_mongo_connection()
    logger.info("TabulaX shutdown complete.")


app = FastAPI(
    title="TabulaX API",
    description="Example-based data transformation platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Optional[Dict[str, Any]] = None


class AppUser(BaseModel):
    id: str
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=100)

    @field_validator("username")
    @classmethod
    def _username_chars(cls, v: str) -> str:
        v = v.strip()
        if not v.isalnum() and not all(c.isalnum() or c in "._-" for c in v):
            raise ValueError("Username may only contain letters, digits, dot, underscore or hyphen.")
        return v


class MySQLConnectionInfo(BaseModel):
    host: str = Field(min_length=1, max_length=255)
    user: str = Field(min_length=1, max_length=128)
    password: str = Field(default="", max_length=512)
    port: int = Field(default=3306, ge=1, le=65535)
    database: Optional[str] = None

    @field_validator("host")
    @classmethod
    def _host_not_url(cls, v: str) -> str:
        v = v.strip()
        if any(c in v for c in "/\\ "):
            raise ValueError("Invalid host.")
        return v


class MongoConnectionInfo(BaseModel):
    uri: str = Field(min_length=1, max_length=2048)

    @field_validator("uri")
    @classmethod
    def _valid_scheme(cls, v: str) -> str:
        v = v.strip()
        if not (v.startswith("mongodb://") or v.startswith("mongodb+srv://")):
            raise ValueError("URI must start with mongodb:// or mongodb+srv://")
        if " " in v:
            raise ValueError("URI must not contain spaces.")
        return v


class DBInfo(BaseModel):
    host: Optional[str] = None
    port: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    table: Optional[str] = None
    uri: Optional[str] = None


class LLMConfigPayload(BaseModel):
    """Transient, per-request LLM settings (e.g. a personal API key).

    The key is used for exactly this request and is never persisted to
    MongoDB/GridFS/history/logs.  ``provider`` defaults to OpenAI.
    """

    provider: Optional[str] = Field(default=None, max_length=32)
    api_key: Optional[str] = Field(default=None, max_length=256)
    model: Optional[str] = Field(default=None, max_length=128)


class ApplyPayload(BaseModel):
    """Client request to apply a *stored* transformation.

    NOTE: ``code`` is intentionally NOT accepted.  Clients reference a
    ``transformation_id`` produced by ``/generate_function/``.  Model-backed
    transformations may optionally carry a transient ``llm_config`` so rows
    are processed with the user's own key.
    """

    transformation_id: str = Field(min_length=8, max_length=64)
    selected_column: str = Field(min_length=1, max_length=255)
    source_type: Optional[str] = None
    session_id: Optional[str] = None
    db_info: Optional[DBInfo] = None
    llm_config: Optional[LLMConfigPayload] = None


class ApplyToSQLPayload(BaseModel):
    connection_details: MySQLConnectionInfo
    table_name: str = Field(min_length=1, max_length=128)
    column_to_transform: str = Field(min_length=1, max_length=255)
    transformation_id: str = Field(min_length=8, max_length=64)
    confirm: bool = False


class ExampleItem(BaseModel):
    """One structured example: ``{"input": "...", "output": "..."}``."""

    input: str = Field(max_length=4096)
    output: str = Field(max_length=4096)


class ClassifyPayload(BaseModel):
    column: str = Field(min_length=1, max_length=255)
    examples: List[ExampleItem] = Field(min_length=1, max_length=1000)
    llm_config: Optional[LLMConfigPayload] = None


class GenerateFunctionPayload(ClassifyPayload):
    classification: str = Field(min_length=1, max_length=64)
    # Optional planner inputs (additive; older clients simply omit them).
    # treat_as_mapping: the user asserts the examples ARE the complete rule.
    treat_as_mapping: bool = False
    # Task/labels decided by the earlier /classify/ call, reused here so the
    # classify -> generate pair costs at most one model call.
    task: Optional[str] = Field(default=None, max_length=32)
    labels: List[str] = Field(default_factory=list, max_length=64)


class SessionIdPayload(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)


class DeleteRowsPayload(SessionIdPayload):
    row_ids: List[int] = Field(min_length=1, max_length=100_000)


class RestoreRowsPayload(SessionIdPayload):
    # Rows previously returned by /delete_rows/ (records incl. the row id).
    rows: List[Dict[str, Any]] = Field(min_length=1, max_length=100_000)


class DeleteColumnsPayload(SessionIdPayload):
    columns: List[str] = Field(min_length=1, max_length=1000)


class RestoredColumn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    position: int = Field(default=0, ge=0)
    # [{row id, value}, ...] as returned by /delete_columns/.
    values: List[Dict[str, Any]] = Field(default_factory=list, max_length=500_000)


class RestoreColumnsPayload(SessionIdPayload):
    columns: List[RestoredColumn] = Field(min_length=1, max_length=1000)


# --------------------------------------------------------------------------- #
# Permanent deletion from the SOURCE database (destructive endpoints).
# A preview request first issues an HMAC-bound ``preview_token``; the matching
# delete request must carry that token plus ``confirm: true`` and an exact
# typed ``"DELETE"`` confirmation.  Nothing is ever deleted from the source
# without a verified preview + explicit intent + (by default) a stored backup.
# --------------------------------------------------------------------------- #
SOURCE_DELETE_TARGETS = (
    "mysql_rows",
    "mysql_columns",
    "mongo_documents",
    "mongo_fields",
)


class SourceDeletePreviewPayload(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    target: str
    row_ids: List[int] = Field(default_factory=list, max_length=100_000)
    columns: List[str] = Field(default_factory=list, max_length=1000)
    fields: List[str] = Field(default_factory=list, max_length=1000)
    mysql_connection: Optional[MySQLConnectionInfo] = None
    mongo_connection: Optional[MongoConnectionInfo] = None

    @field_validator("target")
    @classmethod
    def _known_target(cls, v: str) -> str:
        if v not in SOURCE_DELETE_TARGETS:
            raise ValueError(f"Unknown target '{v}'.")
        return v


class _SourceDeleteBase(SessionIdPayload):
    """Shared intent fields every destructive request must supply."""

    target: str
    preview_token: str = Field(min_length=8, max_length=256)
    confirm: bool = False
    typed_confirmation: str = Field(default="", max_length=32)
    require_backup: bool = True


class SourceDeleteMySQLRowsPayload(_SourceDeleteBase):
    row_ids: List[int] = Field(min_length=1, max_length=100_000)
    connection_details: MySQLConnectionInfo
    allow_partial: bool = False


class SourceDeleteMySQLColumnsPayload(_SourceDeleteBase):
    columns: List[str] = Field(min_length=1, max_length=1000)
    connection_details: MySQLConnectionInfo


class SourceDeleteMongoDocumentsPayload(_SourceDeleteBase):
    row_ids: List[int] = Field(min_length=1, max_length=100_000)
    mongo_connection: MongoConnectionInfo
    allow_partial: bool = False


class SourceDeleteMongoFieldsPayload(_SourceDeleteBase):
    fields: List[str] = Field(min_length=1, max_length=1000)
    mongo_connection: MongoConnectionInfo


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _obj_to_records(df: pd.DataFrame, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convert a DataFrame head to JSON-safe records."""
    view = df.head(limit) if limit else df
    records = view.to_dict(orient="records")
    return _json_safe(records)


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and NaN to JSON-safe values."""
    import numpy as _np

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if obj is None:
        return None
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        f = float(obj)
        return None if _np.isnan(f) else f
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, float) and pd.isna(obj):
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _pairs_from_payload_or_400(items: List[ExampleItem]):
    """Validate a structured examples payload, mapping errors to 400."""
    try:
        return parse_examples_payload([(i.input, i.output) for i in items])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Internal, stable per-row identifier used for selection/deletion.  It is
# stored with the working copy of a session but is never surfaced as a real
# dataset column (excluded from ``columns`` responses and CSV exports).
ROW_ID_COL = "_tabula_row_id"


def _visible_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != ROW_ID_COL]


def _ensure_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return the frame with a usable ``ROW_ID_COL`` (assigning it if needed)."""
    if ROW_ID_COL in df.columns:
        ids = df[ROW_ID_COL]
        # An empty working copy (every row deleted) roundtrips as a non-integer
        # dtype; there are no ids to validate in that case.
        usable = len(df) == 0 or (
            pd.api.types.is_integer_dtype(ids) and bool(ids.is_unique)
        )
        if not usable:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Cannot edit: the dataset contains a column named "
                    f"'{ROW_ID_COL}' which is reserved for row tracking and "
                    "does not hold unique integer ids. Rename that column."
                ),
            )
        return df
    out = df.copy()
    out[ROW_ID_COL] = pd.Series(range(len(out)), index=df.index, dtype="int64")
    return out


def _edit_state(df: pd.DataFrame, hide: Any = ()) -> Dict[str, Any]:
    """Row/column state after an edit (``ROW_ID_COL`` included in records).

    ``hide`` lists internal columns that must never reach the client (e.g.
    ``_id`` for MongoDB sessions: it stays in the working copy so row
    identities survive, but it is not a user-visible column).
    """
    hidden = set(hide)
    columns = [c for c in _visible_columns(df) if c not in hidden]
    records = _obj_to_records(df, settings.preview_rows)
    if hidden:
        records = [
            {k: v for k, v in rec.items() if k not in hidden} for rec in records
        ]
    return {
        "columns": columns,
        "preview": records,
        "row_count": int(len(df)),
    }


def _mongo_hidden(meta: Optional[Dict[str, Any]]) -> List[str]:
    """Client-hidden columns for a session's source type."""
    if not meta:
        return []
    if meta.get("source") == "mongodb" or meta.get("file_type") == "mongodb":
        return ["_id"]
    return []


async def _session_meta_or_empty(owner_id: Any, session_id: str) -> Dict[str, Any]:
    """Session metadata, or ``{}`` when it cannot be read (never raises)."""
    try:
        return await session_store.get_session_meta(owner_id, session_id)
    except session_store.SessionError:
        return {}


async def _get_working_df(owner_id: Any, session_id: str) -> pd.DataFrame:
    """The canonical current dataset: the transformed copy if one exists,
    otherwise the original upload."""
    df = await session_store.get_transformed_df(owner_id, session_id)
    if df is None:
        df = await session_store.get_original_df(owner_id, session_id)
    return df


async def _load_working_or_400(owner_id: Any, session_id: str) -> pd.DataFrame:
    try:
        return await _get_working_df(owner_id, session_id)
    except session_store.SessionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


def _looks_binary(data: bytes) -> bool:
    """True if the sample is mostly non-text control bytes.

    Guards the lenient text parsers so binary uploads fail fast with a clear
    message instead of decoding (latin1 never fails) into a garbage table.
    """
    sample = data[:8192]
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    bad = sum(1 for b in sample if b < 9 or (13 < b < 32) or b == 127)
    return (bad / len(sample)) > 0.10


def _read_csv_tolerant(content: bytes) -> pd.DataFrame:
    """Parse CSV bytes robustly.

    Real-world exports are messy: ragged rows (extra/missing commas), BOMs,
    non-UTF-8 bytes, and non-comma delimiters.  We try a strict parse first and
    only fall back to lenient options (skipping malformed rows, sniffing the
    delimiter), logging so the user knows data was dropped.
    """
    attempts = (
        ("utf-8-sig", {}, None),
        ("utf-8-sig", {"on_bad_lines": "skip"}, "utf8_skip"),
        ("latin1", {"on_bad_lines": "skip"}, "latin1_skip"),
        ("utf-8-sig", {"sep": None, "engine": "python", "on_bad_lines": "skip"}, "sniff"),
    )
    last_exc: Exception = ValueError("unknown CSV parse failure")
    for encoding, kwargs, label in attempts:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            last_exc = exc
            continue
        if label:
            logger.warning(
                "CSV parsed leniently (%s): file contained malformed rows or a "
                "non-default delimiter; bad rows were skipped.",
                label,
            )
        if len(df.columns) == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(
                io.BytesIO(content), encoding=encoding, sep=";", on_bad_lines="skip"
            )
        return df
    raise last_exc


def _read_upload(file: UploadFile) -> pd.DataFrame:
    """Parse an uploaded file into a DataFrame with size/row limits."""
    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_bytes // (1024*1024)} MB limit.",
        )

    name = (file.filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""

    if ext in ("csv", "json") and _looks_binary(content):
        raise HTTPException(
            status_code=400,
            detail="This file appears to be binary data, not CSV/JSON text. "
            "Please export it as UTF-8 text and try again.",
        )

    try:
        if ext == "csv":
            df = _read_csv_tolerant(content)
        elif ext == "json":
            try:
                df = pd.read_json(io.BytesIO(content))
            except ValueError:
                df = pd.read_json(io.BytesIO(content), lines=True, encoding="utf-8")
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload a CSV, JSON, or XLSX file.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", file.filename, type(exc).__name__)
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    if len(df) > settings.max_rows_per_dataset:
        raise HTTPException(
            status_code=413,
            detail=f"File has {len(df)} rows which exceeds the "
            f"{settings.max_rows_per_dataset} row limit.",
        )
    return df


async def _get_owned_transformation(owner_id: str, transformation_id: str) -> Dict[str, Any]:
    try:
        return await session_store.get_transformation(owner_id, transformation_id)
    except session_store.SessionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


async def _run_transformation(code: str, series: pd.Series) -> Any:
    try:
        return apply_to_series(code, series)
    except ApplyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logger.error("Transformation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Transformation failed to run.")


async def _execute_transformation(
    trans_doc: Dict[str, Any],
    series: pd.Series,
    current_user: AppUser,
    llm_override: Optional[dict] = None,
) -> Any:
    """Execute a stored transformation on one column.

    Routes on the planner's persisted spec through the General Transformation
    Engine: deterministic transformations run their sandbox code;
    model-backed transformations (semantic classification, extraction,
    normalization, rewriting) are executed by the configured model through
    :mod:`transform.model_exec`, which batches by stable row id, validates and
    row-aligns the responses.  Legacy transformations (no spec) keep the code
    path.  The report is validated (row count, allowed labels) before it is
    returned so a mis-aligned result never reaches preview.
    """
    spec = TransformationSpec.from_dict((trans_doc or {}).get("spec"))
    if spec.strategy != STRATEGY_MODEL:
        return await _run_transformation((trans_doc or {}).get("code") or "", series)

    examples = [tuple(p) for p in ((trans_doc or {}).get("examples") or [])]
    budget = {"remaining": settings.llm_max_calls_per_request}
    try:
        report = await asyncio.to_thread(
            general_engine.execute,
            series,
            spec,
            examples,
            llm_override=llm_override,
            budget=budget,
        )
        return general_engine.validate_report(report, spec)
    except general_engine.EngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except llm.LLMUnavailableError as exc:
        # No usable provider for the rows that still need processing.
        raise HTTPException(status_code=400, detail=str(exc))
    except ModelExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logger.error("Model-backed transformation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Transformation failed to run.")


# --------------------------------------------------------------------------- #
# History / version helpers
#
# Recording an audit entry or a working-copy snapshot must NEVER break the
# operation that already succeeded; failures are logged and the entry simply
# reports restorable=false.  Source mutations are different: their recovery
# artifact (a verified backup) is created and checked BEFORE the change and a
# failure there aborts the mutation.
# --------------------------------------------------------------------------- #
async def _record_history(owner_id: Any, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        return await history_store.record(owner_id, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("History recording failed: %s", exc)
        return None


async def _snapshot_version(
    owner_id: Any, session_id: str, df: pd.DataFrame, *, label: str
) -> Optional[Dict[str, Any]]:
    try:
        return await history_store.create_version(owner_id, session_id, df, label=label)
    except Exception as exc:
        logger.error("Version snapshot failed for session %s: %s", session_id, exc)
        return None


async def _record_and_snapshot(
    owner_id: Any,
    session_id: str,
    df: pd.DataFrame,
    *,
    operation: str,
    label: str,
    summary: str,
    before: Optional[Dict[str, Any]] = None,
    after: Optional[Dict[str, Any]] = None,
    source_type: Optional[str] = None,
    source_name: Optional[str] = None,
    source_meta: Optional[Dict[str, Any]] = None,
    column: Optional[str] = None,
    transformation: Optional[Dict[str, Any]] = None,
    status: str = "success",
    warnings: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Snapshot the after-state then record one history entry."""
    version = await _snapshot_version(owner_id, session_id, df, label=label)
    await _record_history(
        owner_id,
        session_id=session_id,
        operation=operation,
        status=status,
        source_type=source_type,
        source_name=source_name,
        source_meta=source_meta,
        before=before,
        after=after,
        summary=summary,
        column=column,
        transformation=transformation,
        version_id=(version or {}).get("version_id"),
        restorable=bool(version),
        rollback_expires_at=(version or {}).get("expires_at"),
        warnings=warnings,
        extra=extra,
    )


def _counts(df: pd.DataFrame, hide: Any = ()) -> Dict[str, Any]:
    hidden = set(hide or ())
    return {
        "row_count": int(len(df)),
        "column_count": len([c for c in _visible_columns(df) if c not in hidden]),
    }


# --------------------------------------------------------------------------- #
# Health / readiness
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["health"])
async def health() -> Dict[str, Any]:
    """Liveness probe.  Cheap; never touches external services or secrets."""
    return {"status": "ok", "service": settings.app_name, "time": datetime.now(timezone.utc)}


@app.get("/ready", tags=["health"])
async def ready() -> Dict[str, Any]:
    """Readiness probe.  Verifies MongoDB connectivity without leaking details."""
    from database import client as mongo_client

    mongo_ok = False
    try:
        if mongo_client is not None:
            await mongo_client.admin.command("ping")
            mongo_ok = True
    except Exception:
        mongo_ok = False

    payload = {
        "status": "ready" if mongo_ok else "degraded",
        "mongo": mongo_ok,
        "llm_configured": settings.llm_enabled,
    }
    if not mongo_ok:
        return JSONResponse(status_code=503, content=payload)
    return payload


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
@app.post("/token", response_model=Token, tags=["auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_doc = await get_user_by_username(form_data.username)
    if not user_doc or not verify_password(form_data.password, user_doc.get("hashed_password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user_doc.get("is_active", True):
        raise HTTPException(status_code=400, detail="Inactive user")

    await update_last_login(user_doc["_id"])
    token = create_access_token(subject=user_doc["username"])
    return Token(
        access_token=token,
        token_type="bearer",
        user={
            "id": str(user_doc["_id"]),
            "username": user_doc["username"],
            "email": user_doc.get("email"),
            "full_name": user_doc.get("full_name"),
        },
    )


@app.get("/users/me/", response_model=AppUser, tags=["auth"])
async def read_me(current_user: AppUser = Depends(get_current_active_user)):
    return current_user


@app.post("/register/", response_model=AppUser, status_code=201, tags=["auth"])
async def register(user_data: UserRegister):
    if len(user_data.password) < settings.min_password_length:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {settings.min_password_length} characters.",
        )
    if await get_user_by_username(user_data.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    if await get_user_by_email(user_data.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    try:
        await create_user(user_data.model_dump())
    except Exception as exc:
        logger.error("Registration failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Could not create the account.")

    created = await get_user_by_username(user_data.username)
    if not created:
        raise HTTPException(status_code=500, detail="Could not create the account.")
    return _to_app_user(created)


def _to_app_user(doc: Dict[str, Any]) -> AppUser:
    def _utc(value):
        # Mongo returns naive UTC datetimes; make them aware so the API always
        # serialises with an offset (offset-less ISO is parsed as LOCAL time
        # by browsers, shifting the wall clock).
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    return AppUser(
        id=str(doc["_id"]),
        username=doc["username"],
        email=doc.get("email"),
        full_name=doc.get("full_name"),
        created_at=_utc(doc.get("created_at")) or datetime.now(timezone.utc),
        last_login=_utc(doc.get("last_login")),
        is_active=doc.get("is_active", True),
    )


# --------------------------------------------------------------------------- #
# File upload / download
# --------------------------------------------------------------------------- #
@app.post("/upload_file/", tags=["data"])
async def upload_file(
    file: UploadFile = File(...),
    current_user: AppUser = Depends(get_current_active_user),
):
    name = (file.filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""

    if ext == "sql":
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large.")
        try:
            return {
                "sql_content": content.decode("utf-8"),
                "preview": None,
                "columns": None,
                "file_type": "sql",
            }
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Invalid SQL file encoding.")

    df = _read_upload(file)
    session_id = await session_store.create_session(
        current_user.id,
        df,
        source="file",
        source_name=file.filename,
        file_type=ext,
    )
    await _record_and_snapshot(
        current_user.id, session_id, df,
        operation="import", label=f"Imported {file.filename}",
        summary=f"Imported '{file.filename}' ({len(df)} rows, {df.shape[1]} columns).",
        source_type="file", source_name=file.filename,
        after=_counts(df),
    )
    return {
        "columns": list(df.columns),
        "preview": _obj_to_records(df, settings.preview_rows),
        "file_type": ext,
        "session_id": session_id,
        "row_count": int(len(df)),
    }


@app.get("/download/", tags=["data"])
async def download(
    session_id: Optional[str] = Query(None),
    current_user: AppUser = Depends(get_current_active_user),
):
    """Stream the transformed (or original) dataset as CSV.

    Ownership is enforced via the session; a caller cannot download another
    user's data by guessing a session id.
    """
    # Find the requested session, or the caller's most recent one.
    try:
        if session_id:
            df = None
            try:
                df = await session_store.get_transformed_df(current_user.id, session_id)
            except session_store.SessionError:
                df = None
            if df is None:
                df = await session_store.get_original_df(current_user.id, session_id)
        else:
            recent = await session_store.list_sessions(current_user.id)
            if not recent:
                raise HTTPException(status_code=400, detail="No data available to download.")
            sid = recent[0]["session_id"]
            df = await session_store.get_transformed_df(current_user.id, sid)
            if df is None:
                df = await session_store.get_original_df(current_user.id, sid)
    except session_store.SessionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    # Internal row ids are session bookkeeping, never part of the export.
    if ROW_ID_COL in df.columns:
        df = df.drop(columns=[ROW_ID_COL])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    data = buf.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=transformed_data.csv"},
    )


# --------------------------------------------------------------------------- #
# Classification / generation
# --------------------------------------------------------------------------- #
def _llm_override_from(llm_config: Optional[LLMConfigPayload]) -> Optional[dict]:
    """Build a transient override dict from the request, or None.

    The key never leaves this request; it is not logged and not stored.
    """
    if llm_config is None:
        return None
    key = (llm_config.api_key or "").strip()
    if not key:
        return None
    override = {"api_key": key}
    provider = (llm_config.provider or "").strip().lower()
    if provider:
        override["provider"] = provider
    model = (llm_config.model or "").strip()
    if model:
        override["model"] = model
    return override


def _llm_available_with(override: Optional[dict]) -> bool:
    """True when this request can use an LLM (server config or personal key)."""
    return settings.llm_enabled or bool((override or {}).get("api_key"))


@app.get("/llm/status/", tags=["llm"])
async def llm_status(current_user: AppUser = Depends(get_current_active_user)):
    """Non-sensitive provider status; never returns any API key."""
    return llm.provider_status()


class OllamaPullPayload(BaseModel):
    model: Optional[str] = Field(default=None, max_length=128)
    confirm: bool = False


@app.post("/llm/ollama/pull/", tags=["llm"])
async def llm_ollama_pull(
    payload: OllamaPullPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Download a model into the local Ollama server (explicit user action)."""
    if not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model downloads use significant disk and time. Set "
                "confirm=true to proceed."
            ),
        )
    try:
        return await asyncio.to_thread(
            llm.pull_ollama_model, payload.model or None
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/classify/", tags=["transform"])
async def classify_endpoint(
    payload: ClassifyPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    pairs = _pairs_from_payload_or_400(payload.examples)
    override = _llm_override_from(payload.llm_config)
    allow = _llm_available_with(override)
    result = classify(pairs, allow_llm=allow, llm_override=override)
    return {
        "column": payload.column,
        "classification": result.classification,
        "evidence": result.evidence,
        "template": result.template,
        "used_llm": result.used_llm,
        "llm_available": allow,
        # Task the examples demonstrate (mapping vs rule inference) and the
        # closed label set when the task is semantic classification.
        "task": result.task,
        "labels": result.labels,
        # Set when a configured model was consulted but could not answer
        # (rate limit / quota / network); None otherwise.  Lets the UI show
        # the real reason instead of a generic classification fallback.
        "llm_error": result.llm_error,
    }


@app.post("/generate_function/", tags=["transform"])
async def generate_function(
    payload: GenerateFunctionPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    pairs = _pairs_from_payload_or_400(payload.examples)
    override = _llm_override_from(payload.llm_config)
    allow = _llm_available_with(override)

    # Re-derive the template deterministically so Algorithmic code is exact.
    template = None
    if "algorithmic" in payload.classification.lower():
        from transform.classify import _match_template

        template = _match_template(pairs)

    try:
        generated = generate(
            pairs,
            payload.classification,
            template=template,
            allow_llm=allow,
            llm_override=override,
            treat_as_mapping=payload.treat_as_mapping,
            task_hint=payload.task,
            labels_hint=payload.labels,
            source_column=payload.column,
        )
    except GenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Code generation failed.")

    spec_dict = generated.spec.to_dict() if generated.spec else None
    transformation_id = await session_store.save_transformation(
        current_user.id,
        generated.code,
        classification=generated.classification,
        description=generated.description,
        template=generated.template,
        used_llm=generated.used_llm,
        column=payload.column,
        spec=spec_dict,
        examples=[list(p) for p in generated.examples],
    )

    return {
        "code": generated.code,
        "transformation_id": transformation_id,
        "classification": generated.classification,
        "description": generated.description,
        "used_llm": generated.used_llm,
        "template": generated.template,
        "candidates": generated.candidates,
        "llm_available": allow,
        # Planner output (additive): how this transformation will execute and
        # anything ambiguous the user should resolve.
        "strategy": generated.strategy,
        "spec": spec_dict,
        "ambiguities": generated.ambiguities,
        # Transformation family for the UI (spec §15).
        "family": (generated.spec.family if generated.spec else generated.classification),
    }


# --------------------------------------------------------------------------- #
# Apply transformation
# --------------------------------------------------------------------------- #
@app.post("/apply/", tags=["transform"])
async def apply_transformation(
    payload: ApplyPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    trans_doc = await _get_owned_transformation(current_user.id, payload.transformation_id)
    source = payload.source_type or "file"
    llm_override = _llm_override_from(payload.llm_config)

    if source == "file":
        if not payload.session_id:
            raise HTTPException(status_code=400, detail="session_id is required for file data.")
        df = await _load_working_or_400(current_user.id, payload.session_id)
        return await _apply_to_df(
            df, payload, current_user,
            persist_session=payload.session_id, trans_doc=trans_doc,
            llm_override=llm_override,
        )

    if source == "mongodb_stored":
        if not payload.session_id:
            raise HTTPException(status_code=400, detail="session_id is required for MongoDB data.")
        if payload.selected_column == "_id":
            raise HTTPException(
                status_code=400,
                detail="The MongoDB '_id' field is the document identity and cannot be transformed.",
            )
        df = await _load_working_or_400(current_user.id, payload.session_id)
        # `_id` STAYS in the working copy (dropping it previously destroyed
        # row identity, so permanent deletion could never map rows back to
        # real documents); it is only hidden from the response.
        return await _apply_to_df(
            df, payload, current_user,
            persist_session=payload.session_id,
            hide_columns={"_id"},
            trans_doc=trans_doc,
            llm_override=llm_override,
        )

    if source == "database":
        # Prefer the stored session snapshot (it reflects row/column edits);
        # fall back to a live read when no session handle is available.
        if payload.session_id:
            df = await _load_working_or_400(current_user.id, payload.session_id)
            return await _apply_to_df(
                df, payload, current_user,
                persist_session=payload.session_id, trans_doc=trans_doc,
                llm_override=llm_override,
            )
        df = await _load_mysql_df(payload.db_info)
        return await _apply_to_df(
            df, payload, current_user, persist_session=None, trans_doc=trans_doc,
            llm_override=llm_override,
        )

    raise HTTPException(
        status_code=400,
        detail=f"Invalid source_type: '{source}'. Must be file, mongodb_stored, or database.",
    )


async def _apply_to_df(
    df: pd.DataFrame,
    payload: ApplyPayload,
    current_user: AppUser,
    persist_session: Optional[str],
    hide_columns: Optional[Any] = None,
    trans_doc: Optional[Dict[str, Any]] = None,
    llm_override: Optional[dict] = None,
) -> Dict[str, Any]:
    col = payload.selected_column
    failed = 0
    errors: List[str] = []
    success_rate = 1.0
    processed = 0
    review = 0
    family = ""
    provider = ""
    warnings: List[str] = []

    if df.empty:
        preview = _obj_to_records(df, settings.preview_rows)
        transformed_columns: List[str] = _visible_columns(df)
        row_count = int(len(df))
    else:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' not found. Available: {_visible_columns(df)}",
            )
        before_counts = _counts(df, hide_columns)
        report = await _execute_transformation(
            trans_doc or {}, df[col], current_user, llm_override=llm_override
        )
        failed = report.failed
        errors = report.errors
        success_rate = report.success_rate
        processed = report.total
        review = report.review
        warnings = list(report.warnings)
        # Which family / provider served these rows.  For model-backed runs the
        # report carries them; for deterministic runs we derive the family from
        # the spec and state plainly that no model was used.
        spec = TransformationSpec.from_dict((trans_doc or {}).get("spec"))
        family = report.family or spec.family or (
            (trans_doc or {}).get("classification") or ""
        )
        provider = report.provider or (
            "No model (deterministic rule)" if spec.strategy != STRATEGY_MODEL else ""
        )
        if report.failed and report.failed == report.total:
            raise HTTPException(
                status_code=400,
                detail="Every row failed to transform: " + "; ".join(report.errors[:3]),
            )
        # Reconstruct through the engine so row identity, order and every
        # unselected column are preserved and a mis-aligned result is refused.
        try:
            df_out = general_engine.reconstruct(df, col, report)
        except general_engine.EngineError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if persist_session:
            await session_store.save_transformed(current_user.id, persist_session, df_out)
            meta = await _session_meta_or_empty(current_user.id, persist_session)
            await _record_and_snapshot(
                current_user.id, persist_session, df_out,
                operation="transform_apply",
                label=f"Applied rule to '{col}'",
                summary=(
                    f"Applied transformation to column '{col}' "
                    f"({report.total - report.failed}/{report.total} rows succeeded"
                    + (f", {review} need review" if review else "")
                    + ")."
                ),
                source_type=meta.get("source"), source_name=meta.get("source_name"),
                column=col,
                transformation={
                    "classification": (trans_doc or {}).get("classification"),
                    "description": (trans_doc or {}).get("description"),
                    "template": (trans_doc or {}).get("template"),
                    "used_llm": bool((trans_doc or {}).get("used_llm")),
                    "family": family,
                    "provider": provider,
                } if trans_doc else None,
                before=before_counts,
                after=_counts(df_out, hide_columns),
                status="partial" if (report.failed or review) else "success",
                warnings=(warnings or None),
            )
        preview = _obj_to_records(df_out, settings.preview_rows)
        transformed_columns = _visible_columns(df_out)
        row_count = int(len(df_out))

    hidden = set(hide_columns or ())
    if hidden:
        preview = [{k: v for k, v in rec.items() if k not in hidden} for rec in preview]
        transformed_columns = [c for c in transformed_columns if c not in hidden]

    # Overall status so the UI never shows a generic green when rows failed or
    # need review: "success" (all clean), "review" (accepted, some flagged), or
    # "partial" (some rows failed).
    if failed:
        status = "partial"
    elif review:
        status = "review"
    else:
        status = "success"

    return {
        "message": "Transformation applied successfully",
        "preview": preview,
        "transformed_columns": transformed_columns,
        "failed_rows": failed,
        # Full-dataset accounting: the rule always runs over EVERY row of the
        # authoritative dataset, never just the preview sample.
        "processed_rows": processed,
        "transformed_rows": processed - failed,
        "review_rows": review,
        "row_errors": errors,
        "success_rate": success_rate,
        "session_id": persist_session,
        "row_count": row_count,
        # Enriched reporting (spec §15): which family ran, which provider
        # served it, and any validation warnings, so the user can tell whether
        # the transformation actually succeeded.
        "family": family,
        "provider": provider,
        "status": status,
        "warnings": warnings,
    }


async def _load_mysql_df(db_info: Optional[DBInfo]) -> pd.DataFrame:
    if db_mysql is None:
        raise HTTPException(status_code=503, detail="MySQL support is unavailable.")
    if not db_info or not db_info.host or not db_info.database or not db_info.table:
        raise HTTPException(status_code=400, detail="Missing database connection details.")
    port = 3306
    if db_info.port:
        try:
            port = int(db_info.port)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid port.")
    try:
        with db_mysql.connection(
            host=db_info.host,
            user=db_info.user or "",
            password=db_info.password or "",
            port=port,
            database=db_info.database,
        ) as conn:
            rows = db_mysql.fetch_all(conn, db_info.database, db_info.table)
    except db_mysql.DatabaseError as exc:
        _raise_mysql_error(exc)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# MySQL connectivity
# --------------------------------------------------------------------------- #
def _mysql_conn_info_check():
    if db_mysql is None:
        raise HTTPException(status_code=503, detail="MySQL support is unavailable.")


def _raise_mysql_error(exc) -> None:
    """Translate a driver error into an HTTP response.

    ``db_mysql`` reports *database* authentication failures as 401, but in
    this API HTTP 401 is reserved for the caller's own session (the frontend
    logs the user out on any 401).  Bad database credentials are a bad
    request — surface them as 400 with the driver's clear message instead of
    redirecting the user to the login screen.
    """
    code = 400 if exc.status_code == 401 else exc.status_code
    raise HTTPException(status_code=code, detail=str(exc))


@app.post("/connect_database/", tags=["mysql"])
async def connect_database(
    info: MySQLConnectionInfo,
    current_user: AppUser = Depends(get_current_active_user),
):
    _mysql_conn_info_check()
    try:
        with db_mysql.connection(
            host=info.host, user=info.user, password=info.password, port=info.port
        ) as conn:
            databases = db_mysql.list_databases(conn)
    except db_mysql.DatabaseError as exc:
        _raise_mysql_error(exc)
    return {"databases": databases}


@app.post("/get_tables/", tags=["mysql"])
async def get_tables(
    info: MySQLConnectionInfo,
    current_user: AppUser = Depends(get_current_active_user),
):
    _mysql_conn_info_check()
    if not info.database:
        raise HTTPException(status_code=400, detail="Database name must be provided.")
    try:
        with db_mysql.connection(
            host=info.host, user=info.user, password=info.password,
            port=info.port, database=info.database,
        ) as conn:
            tables = db_mysql.list_tables(conn, info.database)
    except db_mysql.DatabaseError as exc:
        _raise_mysql_error(exc)
    return {"tables": tables}


@app.post("/get_columns/", tags=["mysql"])
async def get_columns(
    info: MySQLConnectionInfo,
    table_name: str = Query(...),
    current_user: AppUser = Depends(get_current_active_user),
):
    """Snapshot a MySQL table into a user-owned session and preview it.

    Mirrors ``/get_mongo_data/``: the full table (capped at the dataset row
    limit) is stored server-side so row/column edits, undo, apply and CSV
    export all operate on the same working copy.  Live write-back remains a
    separate explicit step (``/apply_to_database/``).
    """
    _mysql_conn_info_check()
    if not info.database:
        raise HTTPException(status_code=400, detail="Database name must be provided.")
    try:
        with db_mysql.connection(
            host=info.host, user=info.user, password=info.password,
            port=info.port, database=info.database,
        ) as conn:
            columns = db_mysql.describe_table(conn, info.database, table_name)
            rows = db_mysql.fetch_all(
                conn, info.database, table_name,
                limit=settings.max_rows_per_dataset + 1,
            )
    except db_mysql.DatabaseError as exc:
        _raise_mysql_error(exc)

    if len(rows) > settings.max_rows_per_dataset:
        raise HTTPException(
            status_code=413,
            detail=f"Table '{table_name}' has more than "
            f"{settings.max_rows_per_dataset} rows.",
        )
    if not rows:
        return {"columns": columns, "preview": [], "session_id": None, "row_count": 0}

    df = pd.DataFrame(rows, columns=columns)
    try:
        session_id = await session_store.create_session(
            current_user.id,
            df,
            source="mysql",
            source_name=f"{info.database}.{table_name}",
            file_type="mysql",
            # Server-recorded namespace: source deletion resolves the real
            # table from here (never from client input).
            source_meta={"database": info.database, "table": table_name},
        )
    except session_store.SessionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    await _record_and_snapshot(
        current_user.id, session_id, df,
        operation="import", label=f"Connected {info.database}.{table_name}",
        summary=(
            f"Connected MySQL table '{info.database}.{table_name}' "
            f"({len(df)} rows, {df.shape[1]} columns)."
        ),
        source_type="mysql", source_name=f"{info.database}.{table_name}",
        source_meta={"database": info.database, "table": table_name},
        after=_counts(df),
    )
    return {
        "columns": columns,
        "preview": _obj_to_records(df, settings.preview_rows),
        "session_id": session_id,
        "row_count": int(len(df)),
    }


@app.post("/apply_to_database/", tags=["mysql"])
async def apply_to_database(
    payload: ApplyToSQLPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Write a stored transformation back to a MySQL table.

    Requires ``confirm: true``.  Identifiers are validated, values are
    parameterised, and the swap is reversible on failure (see db_mysql).
    """
    _mysql_conn_info_check()
    if not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to modify the table.",
        )
    if not payload.connection_details.database:
        raise HTTPException(status_code=400, detail="Database name missing.")

    trans_doc = await _get_owned_transformation(current_user.id, payload.transformation_id)

    try:
        with db_mysql.connection(
            host=payload.connection_details.host,
            user=payload.connection_details.user,
            password=payload.connection_details.password,
            port=payload.connection_details.port,
            database=payload.connection_details.database,
        ) as conn:
            rows = db_mysql.fetch_all(
                conn, payload.connection_details.database, payload.table_name
            )
            if not rows:
                return {"message": "Table has no rows; nothing to update."}
            df = pd.DataFrame(rows)
            col = payload.column_to_transform
            if col not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{col}' not found in table. Available: {list(df.columns)}",
                )
            report = await _execute_transformation(trans_doc, df[col], current_user)
            if report.failed and report.failed == report.total:
                raise HTTPException(
                    status_code=400,
                    detail="Every row failed to transform: " + "; ".join(report.errors[:3]),
                )
            df[col] = report.series
            columns = list(df.columns)
            records = df.to_dict(orient="records")
            records = _json_safe(records)

            # Recovery artifact BEFORE mutating the live table: the previous
            # values of the transformed column (keyed by primary key).  A
            # write-back is a permanent source mutation; without a verified
            # backup it is not committed under the recoverable workflow.
            pk = db_mysql.get_primary_key(conn, payload.connection_details.database, payload.table_name)
            if not pk:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Table '{payload.connection_details.database}."
                        f"{payload.table_name}' has no PRIMARY KEY; refusing a "
                        "write-back that could not be restored. Add a primary "
                        "key or edit a working copy instead."
                    ),
                )
            pre_rows = db_mysql.select_columns(
                conn, payload.connection_details.database, payload.table_name,
                list(dict.fromkeys(list(pk) + [col])),
            )
            pre_backup = await _make_backup(
                current_user.id,
                kind="mysql_column_values",
                source_name=f"{payload.connection_details.database}.{payload.table_name}",
                payload=_gzip_csv(pd.DataFrame(pre_rows)),
                meta={
                    "format": "csv",
                    "database": payload.connection_details.database,
                    "table": payload.table_name,
                    "primary_key": list(pk),
                    "column": col,
                    "columns": list(pre_rows[0].keys()) if pre_rows else [],
                    "row_count": len(pre_rows),
                },
                require_backup=True,
                warnings=[],
            )
            summary = db_mysql.write_back_transformed(
                conn,
                payload.connection_details.database,
                payload.table_name,
                columns,
                records,
            )
    except db_mysql.DatabaseError as exc:
        _raise_mysql_error(exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("apply_to_database failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Database update failed.")

    await _record_history(
        current_user.id,
        session_id=payload.session_id,
        operation="write_back_mysql",
        source_type="mysql",
        source_name=f"{payload.connection_details.database}.{payload.table_name}",
        source_meta={
            "database": payload.connection_details.database,
            "table": payload.table_name,
        },
        column=payload.column_to_transform,
        summary=(
            f"Wrote transformation of '{payload.column_to_transform}' back to "
            f"'{payload.connection_details.database}.{payload.table_name}' "
            f"({summary['rows_written']} rows)."
        ),
        after={"rows_written": summary["rows_written"]},
        backup_id=pre_backup.get("backup_id"),
        restorable=bool(pre_backup.get("backup_id")),
        rollback_expires_at=pre_backup.get("expires_at"),
        transformation={
            "classification": trans_doc.get("classification"),
            "description": trans_doc.get("description"),
            "used_llm": bool(trans_doc.get("used_llm")),
        },
        warnings=(["Some rows failed to transform."] if report.failed else None),
    )
    return {
        "message": f"Updated {summary['rows_written']} rows in {payload.table_name}.",
        "rows_written": summary["rows_written"],
        "failed_rows": report.failed,
        "row_errors": report.errors,
        "backup": pre_backup,
    }


# --------------------------------------------------------------------------- #
# MongoDB data connectivity
# --------------------------------------------------------------------------- #
def _map_mongo_uri(uri: str) -> str:
    """Rewrite loopback hosts in a user URI to host.docker.internal (Docker).

    A user typing mongodb://localhost:27017 means *their machine*, but inside
    a container localhost is the container itself.  Non-loopback hosts
    (``mongo``, Atlas URIs, ...) are returned unchanged.
    """
    import re as _re

    if not settings.in_docker or not uri:
        return uri
    m = _re.match(r"^(mongodb(?:\+srv)?://)([^/@]*@)?([^/?#]+)(.*)$", uri)
    if not m:
        return uri
    scheme, creds, authority, rest = m.groups()
    new_hosts, changed = [], False
    for h in (authority or "").split(","):
        hm = _re.match(r"^(.*?)(:\d+)?$", h)
        name, port = hm.group(1), hm.group(2) or ""
        if name.strip("[]").lower() in ("localhost", "127.0.0.1"):
            new_hosts.append("host.docker.internal" + port)
            changed = True
        else:
            new_hosts.append(h)
    if not changed:
        return uri
    logger.info("Docker detected: MongoDB localhost mapped to 'host.docker.internal'.")
    return f"{scheme}{creds or ''}{','.join(new_hosts)}{rest or ''}"


def _is_mongo_auth_error(exc: Exception) -> bool:
    """True when a driver error means *bad credentials in the URI*."""
    if getattr(exc, "code", None) in (18, 13):  # AuthenticationFailed / Unauthorized
        return True
    msg = str(exc).lower()
    return "authentication failed" in msg or "auth failed" in msg


def _mongo_auth_http_error() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail="Incorrect MongoDB username or password (authentication failed).",
    )


async def _mongo_client(conn: MongoConnectionInfo):
    import motor.motor_asyncio

    target_uri = _map_mongo_uri(conn.uri)
    client = motor.motor_asyncio.AsyncIOMotorClient(
        target_uri, serverSelectionTimeoutMS=8000, connectTimeoutMS=8000
    )
    try:
        await client.server_info()
    except Exception as exc:
        client.close()
        logger.warning("MongoDB connection failed: %s", type(exc).__name__)
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        detail = "Could not connect to MongoDB."
        if target_uri != conn.uri:
            detail += (
                " 'localhost' was mapped to your host machine (Docker): check "
                "the server is running there and the port is open, or use your "
                "Atlas URI."
            )
        elif "localhost" in conn.uri or "127.0.0.1" in conn.uri:
            detail += " Is the server running on this machine?"
        raise HTTPException(status_code=400, detail=detail)
    return client


def _parse_mongo_conn(raw: str) -> MongoConnectionInfo:
    """Build a validated ``MongoConnectionInfo`` from a client-supplied blob.

    Accepts either a JSON object string (``{"uri": "mongodb://..."}``) or a raw
    URI.  Any additional JSON keys are ignored so the response always matches
    the ``MongoConnectionInfo`` schema (single ``uri`` field).
    """
    import json as _json

    raw = (raw or "").strip()
    if raw.startswith("{"):
        try:
            data = _json.loads(raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid connection info (bad JSON)."
            ) from exc
        if isinstance(data, dict):
            uri = str(data.get("uri", "")).strip()
        else:
            uri = str(data).strip()
    else:
        uri = raw
    return MongoConnectionInfo(uri=uri)


@app.post("/connect_mongo/", tags=["mongodb"])
async def connect_mongo(
    conn: MongoConnectionInfo,
    current_user: AppUser = Depends(get_current_active_user),
):
    client = await _mongo_client(conn)
    try:
        dbs = await client.list_database_names()
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.warning("list_database_names failed: %s", exc)
        raise HTTPException(
            status_code=400, detail="Failed to list MongoDB databases."
        )
    finally:
        client.close()
    return {"databases": dbs}


@app.post("/get_mongo_collections/", tags=["mongodb"])
async def get_mongo_collections(
    conn: MongoConnectionInfo,
    database_name: str = Query(...),
    current_user: AppUser = Depends(get_current_active_user),
):
    client = await _mongo_client(conn)
    try:
        collections = await client[database_name].list_collection_names()
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.warning("list_collection_names failed: %s", exc)
        raise HTTPException(
            status_code=400, detail="Failed to list MongoDB collections."
        )
    finally:
        client.close()
    return {"collections": collections}


def _mongo_id_kind(docs: List[Dict[str, Any]]) -> str:
    """Single BSON type shared by every ``_id`` in ``docs`` (else ``mixed``)."""
    from bson import ObjectId as _OID

    kinds = set()
    for d in docs:
        v = d.get("_id")
        if isinstance(v, bool):
            kinds.add("other")
        elif isinstance(v, _OID):
            kinds.add("objectid")
        elif isinstance(v, int):
            kinds.add("int")
        elif isinstance(v, str):
            kinds.add("string")
        else:
            kinds.add("other")
    if len(kinds) != 1:
        return "mixed"
    return kinds.pop()


def _mongo_ids_roundtrip_ok(series: pd.Series) -> bool:
    """True when ``series`` survives the exact session CSV round-trip.

    Sessions are stored as gzipped CSV with pandas type inference, so string
    ids that look numeric (``"00123"``, ``"1e5"``) would be re-parsed as
    numbers and silently corrupt identity.  When this check fails, permanent
    source deletion is disabled for the session (working-copy edits keep
    working) rather than ever risking a delete against the wrong document.
    """
    try:
        data = session_store._df_to_bytes(series.to_frame())
        back = session_store._bytes_to_df(data)
        col = back.columns[0]
        return series.astype(str).tolist() == back[col].astype(str).tolist()
    except Exception:  # pragma: no cover - conservative fallback
        return False


async def _mongo_session_from_live(
    owner_id: Any,
    client,
    database_name: str,
    collection_name: str,
) -> Dict[str, Any]:
    """Fetch live documents → a new owned session with identity metadata.

    Returns ``{"session_id", "df", "columns", "preview", "row_count",
    "source_meta"}`` (``session_id`` is ``None`` for an empty collection).
    ``_id`` is kept inside the stored frame — it is the row identity used to
    map working-copy selections back to real documents for permanent
    deletion — but never surfaced in ``columns``/``preview``.
    """
    from bson import ObjectId as _OID

    coll = client[database_name][collection_name]
    docs = await coll.find().limit(settings.max_rows_per_dataset).to_list(
        length=settings.max_rows_per_dataset
    )
    base_meta = {"database": database_name, "collection": collection_name}
    if not docs:
        return {
            "session_id": None,
            "df": pd.DataFrame(),
            "columns": [],
            "preview": [],
            "row_count": 0,
            "source_meta": base_meta,
        }

    cleaned = []
    for d in docs:
        row = {}
        for k, v in d.items():
            row[k] = str(v) if isinstance(v, _OID) else v
        cleaned.append(row)
    df = pd.DataFrame(cleaned)

    id_kind = _mongo_id_kind(docs)
    identity_ok = "_id" in df.columns and _mongo_ids_roundtrip_ok(df["_id"])
    source_meta = {
        **base_meta,
        "id_kind": id_kind,
        "identity_ok": bool(identity_ok),
    }
    session_id = await session_store.create_session(
        owner_id,
        df,
        source="mongodb",
        source_name=f"{database_name}.{collection_name}",
        file_type="mongodb",
        source_meta=source_meta,
    )
    hidden = {"_id"}
    return {
        "session_id": session_id,
        "df": df,
        "columns": [c for c in df.columns if c not in hidden],
        "preview": [
            {k: v for k, v in rec.items() if k not in hidden}
            for rec in _obj_to_records(df, settings.preview_rows)
        ],
        "row_count": int(len(df)),
        "source_meta": source_meta,
    }


@app.post("/get_mongo_data/", tags=["mongodb"])
async def get_mongo_data(
    conn: MongoConnectionInfo,
    database_name: str = Query(...),
    collection_name: str = Query(...),
    current_user: AppUser = Depends(get_current_active_user),
):
    client = await _mongo_client(conn)
    try:
        info = await _mongo_session_from_live(
            current_user.id, client, database_name, collection_name
        )
        if not info["session_id"]:
            return {"columns": [], "preview": [], "file_type": "mongodb", "session_id": None}
        await _record_and_snapshot(
            current_user.id, info["session_id"], info["df"],
            operation="import", label=f"Connected {database_name}.{collection_name}",
            summary=(
                f"Connected MongoDB collection '{database_name}.{collection_name}' "
                f"({info['row_count']} documents)."
            ),
            source_type="mongodb", source_name=f"{database_name}.{collection_name}",
            source_meta=info["source_meta"],
            after=_counts(info["df"], hide={"_id"}),
        )
        return {
            "columns": info["columns"],
            "preview": info["preview"],
            "file_type": "mongodb",
            "session_id": info["session_id"],
            "row_count": info["row_count"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.error("get_mongo_data failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch MongoDB data.")
    finally:
        client.close()


class ApplyToMongoPayload(BaseModel):
    conn_info_json: str
    database_name: str
    collection_name: str
    column_to_transform: str
    transformation_id: str
    confirm: bool = False


@app.post("/apply_to_mongodb/", tags=["mongodb"])
async def apply_to_mongodb(
    payload: ApplyToMongoPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    from bson import ObjectId as _OID
    from pymongo import UpdateOne

    if not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to modify the collection.",
        )

    trans_doc = await _get_owned_transformation(current_user.id, payload.transformation_id)

    conn = _parse_mongo_conn(payload.conn_info_json)
    client = await _mongo_client(conn)
    try:
        coll = client[payload.database_name][payload.collection_name]
        docs = await coll.find().to_list(length=None)
        if not docs:
            return {"message": "No documents to transform.", "modified": 0}

        cleaned = []
        ids = []
        for d in docs:
            ids.append(d["_id"])
            row = {}
            for k, v in d.items():
                row[k] = str(v) if isinstance(v, _OID) else v
            cleaned.append(row)
        df = pd.DataFrame(cleaned)

        col = payload.column_to_transform
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' not found. Available: {list(df.columns)}",
            )
        report = await _execute_transformation(trans_doc, df[col], current_user)
        if report.failed and report.failed == report.total:
            raise HTTPException(status_code=400, detail="All rows failed: " + "; ".join(report.errors[:3]))

        # Recovery artifact BEFORE mutating the live collection: the current
        # values of the transformed field (keyed by _id).
        pre_docs = await coll.find({}, {col: 1}).to_list(settings.max_rows_per_dataset)
        pre_backup = await _make_backup(
            current_user.id,
            kind="mongo_field_values",
            source_name=f"{payload.database_name}.{payload.collection_name}",
            payload=_gzip_jsonl(pre_docs),
            meta={
                "format": "jsonl",
                "database": payload.database_name,
                "collection": payload.collection_name,
                "fields": [col],
                "document_count": len(pre_docs),
            },
            require_backup=True,
            warnings=[],
        )

        ops = [
            UpdateOne({"_id": ids[i]}, {"$set": {col: _json_safe(df[col].iloc[i])}})
            for i in range(len(ids))
        ]
        result = await coll.bulk_write(ops)
        logger.info(
            "MongoDB update db=%s coll=%s matched=%d modified=%d",
            payload.database_name, payload.collection_name,
            result.matched_count, result.modified_count,
        )
        await _record_history(
            current_user.id,
            session_id=None,
            operation="write_back_mongodb",
            source_type="mongodb",
            source_name=f"{payload.database_name}.{payload.collection_name}",
            source_meta={
                "database": payload.database_name,
                "collection": payload.collection_name,
            },
            column=payload.column_to_transform,
            summary=(
                f"Wrote transformation of '{payload.column_to_transform}' back to "
                f"'{payload.database_name}.{payload.collection_name}' "
                f"({result.modified_count} documents)."
            ),
            after={"modified": result.modified_count},
            backup_id=pre_backup.get("backup_id"),
            restorable=bool(pre_backup.get("backup_id")),
            rollback_expires_at=pre_backup.get("expires_at"),
            transformation={
                "classification": trans_doc.get("classification"),
                "description": trans_doc.get("description"),
                "used_llm": bool(trans_doc.get("used_llm")),
            },
            warnings=(["Some rows failed to transform."] if report.failed else None),
        )
        return {
            "message": f"Updated {result.modified_count} documents.",
            "modified": result.modified_count,
            "failed_rows": report.failed,
            "row_errors": report.errors,
            "backup": pre_backup,
        }
    except HTTPException:
        raise
    except session_store.SessionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.error("apply_to_mongodb failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update MongoDB.")
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# Dataset edits: row/column deletion with undo, persisted to the session's
# canonical working copy (originals are never modified; live databases are
# only written through the explicit confirm=true write-back endpoints).
# --------------------------------------------------------------------------- #
@app.post("/delete_rows/", tags=["data"])
async def delete_rows(
    payload: DeleteRowsPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    df = await _load_working_or_400(current_user.id, payload.session_id)
    df = _ensure_row_ids(df)
    ids = list(dict.fromkeys(payload.row_ids))
    present = {int(v) for v in df[ROW_ID_COL].tolist()}
    missing = [i for i in ids if i not in present]
    if missing:
        shown = ", ".join(str(i) for i in missing[:10])
        more = ", ..." if len(missing) > 10 else ""
        raise HTTPException(
            status_code=400,
            detail=(
                f"Stale row selection: {len(missing)} row(s) are no longer in "
                f"the dataset ({shown}{more}). Refresh the preview and try again."
            ),
        )
    mask = df[ROW_ID_COL].isin(ids)
    deleted = df[mask]
    df_new = df[~mask]
    await session_store.save_transformed(current_user.id, payload.session_id, df_new)
    meta = await _session_meta_or_empty(current_user.id, payload.session_id)
    hide = _mongo_hidden(meta)
    await _record_and_snapshot(
        current_user.id, payload.session_id, df_new,
        operation="working_copy_delete_rows",
        label=f"Deleted {len(deleted)} row(s)",
        summary=f"Deleted {len(deleted)} row(s) from the working copy (source unchanged).",
        source_type=meta.get("source"), source_name=meta.get("source_name"),
        before=_counts(df, hide), after=_counts(df_new, hide),
        extra={"deleted_row_ids": [int(v) for v in deleted[ROW_ID_COL].tolist()]},
    )
    logger.info(
        "Deleted %d row(s) from session %s by user %s",
        len(deleted), payload.session_id, current_user.id,
    )
    # ``deleted_rows`` keeps every stored column (including ``_id``) so the
    # undo path can restore identities; only the visible state is hidden.
    return {"removed": int(len(deleted)), "deleted_rows": _obj_to_records(deleted),
            **_edit_state(df_new, hide)}


@app.post("/restore_rows/", tags=["data"])
async def restore_rows(
    payload: RestoreRowsPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    df = await _load_working_or_400(current_user.id, payload.session_id)
    df = _ensure_row_ids(df)
    new = pd.DataFrame(payload.rows)
    if ROW_ID_COL not in new.columns:
        raise HTTPException(status_code=400, detail="Row payload is missing internal row ids.")
    try:
        new[ROW_ID_COL] = pd.to_numeric(new[ROW_ID_COL], errors="raise").round().astype("int64")
    except Exception:
        raise HTTPException(status_code=400, detail="Row payload contains invalid row ids.")
    if bool(new[ROW_ID_COL].duplicated().any()):
        raise HTTPException(status_code=400, detail="Row payload contains duplicate row ids.")
    present = {int(v) for v in df[ROW_ID_COL].tolist()}
    overlap = [int(v) for v in new[ROW_ID_COL].tolist() if int(v) in present]
    if overlap:
        shown = ", ".join(str(i) for i in overlap[:10])
        raise HTTPException(
            status_code=400,
            detail=f"Rows are already present in the dataset ({shown}). "
            "Refresh the preview and try again.",
        )
    new = new.reindex(columns=df.columns)
    df_new = pd.concat([df, new], ignore_index=True)
    df_new = df_new.sort_values(ROW_ID_COL, kind="stable").reset_index(drop=True)
    await session_store.save_transformed(current_user.id, payload.session_id, df_new)
    meta = await _session_meta_or_empty(current_user.id, payload.session_id)
    hide = _mongo_hidden(meta)
    await _record_and_snapshot(
        current_user.id, payload.session_id, df_new,
        operation="working_copy_restore_rows",
        label=f"Restored {len(new)} row(s)",
        summary=f"Restored {len(new)} previously deleted row(s) to the working copy.",
        source_type=meta.get("source"), source_name=meta.get("source_name"),
        before=_counts(df, hide), after=_counts(df_new, hide),
    )
    logger.info("Restored %d row(s) in session %s", len(new), payload.session_id)
    return {"restored": int(len(new)), **_edit_state(df_new, hide)}


@app.post("/delete_columns/", tags=["data"])
async def delete_columns(
    payload: DeleteColumnsPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    df = await _load_working_or_400(current_user.id, payload.session_id)
    df = _ensure_row_ids(df)
    meta = await _session_meta_or_empty(current_user.id, payload.session_id)
    hide = _mongo_hidden(meta)
    visible = [c for c in _visible_columns(df) if c not in hide]
    requested = list(dict.fromkeys(payload.columns))
    if ROW_ID_COL in requested:
        raise HTTPException(
            status_code=400,
            detail=f"'{ROW_ID_COL}' is an internal column and cannot be deleted.",
        )
    if "_id" in requested and hide:
        raise HTTPException(
            status_code=400,
            detail="'_id' is the MongoDB document identity and cannot be deleted "
                   "from the working copy (row selection depends on it).",
        )
    missing = [c for c in requested if c not in visible]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Column(s) not found: {missing}. Available: {visible}",
        )
    if len(requested) >= len(visible):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete every column; at least one column must remain.",
        )
    deleted_meta = []
    for name in requested:
        deleted_meta.append(
            {
                "name": name,
                "position": visible.index(name),
                "values": _obj_to_records(df[[ROW_ID_COL, name]]),
            }
        )
    df_new = df.drop(columns=requested)
    await session_store.save_transformed(current_user.id, payload.session_id, df_new)
    await _record_and_snapshot(
        current_user.id, payload.session_id, df_new,
        operation="working_copy_delete_columns",
        label=f"Deleted {len(requested)} column(s)",
        summary=(
            f"Deleted column(s) {requested} from the working copy "
            "(source unchanged)."
        ),
        source_type=meta.get("source"), source_name=meta.get("source_name"),
        before=_counts(df, hide), after=_counts(df_new, hide),
        extra={"deleted_columns": requested},
    )
    logger.info(
        "Deleted column(s) %s from session %s", requested, payload.session_id
    )
    return {"deleted_columns": deleted_meta, **_edit_state(df_new, hide)}


@app.post("/restore_columns/", tags=["data"])
async def restore_columns(
    payload: RestoreColumnsPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    df = await _load_working_or_400(current_user.id, payload.session_id)
    df = _ensure_row_ids(df)

    # Prefer the original column order (from the session metadata) when
    # deciding where a restored column belongs; fall back to the recorded
    # position for names the metadata does not know.
    try:
        meta = await session_store.get_session_meta(current_user.id, payload.session_id)
        # ``original_columns`` is the import-time order; sessions created
        # before it existed fall back to ``columns`` (which then still held
        # the import-time order when the session was written).
        orig_order = list(meta.get("original_columns") or meta.get("columns") or [])
    except session_store.SessionError:
        meta = {}
        orig_order = []
    hide = _mongo_hidden(meta)
    orig_index = {name: i for i, name in enumerate(orig_order)}
    big = 10**9

    items = sorted(
        payload.columns, key=lambda it: orig_index.get(it.name, big + it.position)
    )
    restored: List[str] = []
    for item in items:
        if item.name == ROW_ID_COL or item.name in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{item.name}' already exists in the dataset.",
            )
        values: Dict[Any, Any] = {}
        for record in item.values:
            if ROW_ID_COL in record:
                try:
                    values[int(record[ROW_ID_COL])] = record.get(item.name)
                except (TypeError, ValueError):
                    continue
        if not values:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{item.name}' payload contains no row values.",
            )
        series = df[ROW_ID_COL].map(values)
        series.name = item.name

        visible_now = _visible_columns(df)
        if item.name in orig_index:
            before = sum(
                1
                for c in visible_now
                if orig_index.get(c, big) < orig_index[item.name]
            )
        else:
            before = min(item.position, len(visible_now))
        if before < len(visible_now):
            anchor = visible_now[before]
            insert_at = int(df.columns.get_loc(anchor))
        else:
            insert_at = len(df.columns)
        df.insert(insert_at, item.name, series)
        restored.append(item.name)

    await session_store.save_transformed(current_user.id, payload.session_id, df)
    await _record_and_snapshot(
        current_user.id, payload.session_id, df,
        operation="working_copy_restore_columns",
        label=f"Restored {len(restored)} column(s)",
        summary=f"Restored column(s) {restored} to the working copy.",
        source_type=meta.get("source"), source_name=meta.get("source_name"),
        extra={"restored_columns": restored},
    )
    logger.info("Restored column(s) %s in session %s", restored, payload.session_id)
    return {"restored_columns": restored, **_edit_state(df, hide)}


@app.get("/session_preview/", tags=["data"])
async def session_preview(
    session_id: str = Query(...),
    current_user: AppUser = Depends(get_current_active_user),
):
    """Current (edited/transformed) + original preview for a saved session.

    Used to restore the UI after a page refresh so it never shows stale data.
    """
    try:
        meta = await session_store.get_session_meta(current_user.id, session_id)
        original = await session_store.get_original_df(current_user.id, session_id)
        working = await _get_working_df(current_user.id, session_id)
    except session_store.SessionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    hide = _mongo_hidden(meta)
    state = _edit_state(working, hide)
    original_preview = _obj_to_records(original, settings.preview_rows)
    if hide:
        original_preview = [
            {k: v for k, v in rec.items() if k not in set(hide)}
            for rec in original_preview
        ]
    return {
        "session_id": session_id,
        "columns": state["columns"],
        "preview": state["preview"],
        "original_preview": original_preview,
        "row_count": int(len(working)),
        "source": meta.get("source"),
        "source_name": meta.get("source_name"),
        "file_type": meta.get("file_type"),
        "has_transformed": meta.get("has_transformed"),
        # Sanitized namespace + history marker (e.g. "opened from v3"); never
        # contains credentials or connection strings.
        "source_meta": meta.get("source_meta") or {},
    }


@app.post("/revert_session/", tags=["data"])
async def revert_session(
    payload: SessionIdPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Reset the working copy back to the original upload (undo everything)."""
    try:
        original = await session_store.get_original_df(current_user.id, payload.session_id)
    except session_store.SessionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    working = _ensure_row_ids(original)
    await session_store.save_transformed(current_user.id, payload.session_id, working)
    meta = await _session_meta_or_empty(current_user.id, payload.session_id)
    hide = _mongo_hidden(meta)
    await _record_and_snapshot(
        current_user.id, payload.session_id, working,
        operation="working_copy_revert",
        label="Reverted to original",
        summary="Reverted the working copy to the original upload (source unchanged).",
        source_type=meta.get("source"), source_name=meta.get("source_name"),
        after=_counts(working, hide),
    )
    logger.info("Reverted session %s to the original dataset", payload.session_id)
    return {
        "message": "Reverted to the original dataset.",
        "transformed_columns": [],
        **_edit_state(working, hide),
    }


# --------------------------------------------------------------------------- #
# Session introspection (used by the UI / debugging)
# --------------------------------------------------------------------------- #
@app.get("/sessions/", tags=["data"])
async def list_my_sessions(current_user: AppUser = Depends(get_current_active_user)):
    return {"sessions": await session_store.list_sessions(current_user.id)}


@app.delete("/sessions/{session_id}", tags=["data"])
async def delete_my_session(
    session_id: str, current_user: AppUser = Depends(get_current_active_user)
):
    try:
        await session_store.delete_session(current_user.id, session_id)
    except session_store.SessionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return {"deleted": True}


# --------------------------------------------------------------------------- #
# Permanent deletion from the SOURCE database (destructive, explicit)
#
# Contract for every delete request:
#   1. a fresh POST /source_delete/preview/ must have issued a preview_token
#      (HMAC-bound to session, target, connection fingerprint, namespace and
#      the exact selection; valid 15 minutes);
#   2. confirm=true AND typed_confirmation="DELETE" (exact, case-sensitive);
#   3. a backup of the affected rows/columns/documents is stored first and
#      verified — if it cannot be stored, nothing is deleted (unless the user
#      explicitly set require_backup=false);
#   4. the working-copy session is then re-synced from the live source (a new
#      session replaces the stale one) so the UI never shows deleted data.
#
# The namespace (database/table or db/collection) is ALWAYS resolved from the
# server-side session record — never from client input.
# --------------------------------------------------------------------------- #
_PREVIEW_TOKEN_TTL_SECONDS = 15 * 60


def _require_destructive_intent(confirm: bool, typed_confirmation: str) -> None:
    """Both gates must pass before any source data can be touched."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required: set confirm=true. Permanent deletion "
                   "cannot be undone.",
        )
    if typed_confirmation != "DELETE":
        raise HTTPException(
            status_code=400,
            detail='Type the word DELETE exactly (uppercase) to confirm permanent deletion.',
        )


def _source_namespace_or_400(meta: Dict[str, Any]):
    """Resolve (source, database, table|collection) from the session record.

    Uploaded-file sessions are rejected with an explicit message: their
    original file is never modified, so "permanent deletion" does not apply.
    """
    source = meta.get("source")
    if source not in ("mysql", "mongodb"):
        raise HTTPException(
            status_code=400,
            detail="This session is backed by an uploaded file — the original "
                   "file is never modified, so there is nothing to delete "
                   "permanently. Use working-copy delete for local edits.",
        )
    sm = meta.get("source_meta") or {}

    if source == "mysql":
        db = str(sm.get("database") or "")
        table = str(sm.get("table") or "")
        if not db or not table:
            db2, sep, table2 = (meta.get("source_name") or "").partition(".")
            if not sep:
                raise HTTPException(
                    status_code=400,
                    detail="This session does not record which table it came "
                           "from; reconnect the table to enable source deletion.",
                )
            db, table = db2, table2
        _mysql_conn_info_check()
        try:
            db = db_mysql.validate_identifier(db, "database name")
            table = db_mysql.validate_identifier(table, "table name")
        except db_mysql.DatabaseError as exc:
            _raise_mysql_error(exc)
        return source, db, table

    db = str(sm.get("database") or "")
    coll = str(sm.get("collection") or "")
    if not db or not coll:
        db2, sep, coll2 = (meta.get("source_name") or "").partition(".")
        if not sep:
            raise HTTPException(
                status_code=400,
                detail="This session does not record which collection it came "
                       "from; reconnect the collection to enable source deletion.",
            )
        db, coll = db2, coll2
    forbidden = set('/\\"$*<>:|?') | {chr(i) for i in range(32)}
    if not db or any(c in db for c in forbidden) or "." in db:
        raise HTTPException(status_code=400, detail="Invalid MongoDB database name in session metadata.")
    if not coll or coll.startswith("system.") or coll.startswith("$") or any(c in coll for c in forbidden):
        raise HTTPException(status_code=400, detail="Invalid MongoDB collection name in session metadata.")
    return source, db, coll


async def _delete_session_context(owner_id: Any, session_id: str, target: str):
    """Load + validate session context for a source-delete request."""
    meta = await _session_meta_or_empty(owner_id, session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found or access denied.")
    source, db, name = _source_namespace_or_400(meta)
    wants_mysql = target.startswith("mysql")
    if (source == "mysql") != wants_mysql:
        raise HTTPException(
            status_code=400,
            detail=f"Target '{target}' does not match this session's source ({source}).",
        )
    return meta, source, db, name


def _mysql_conn_fingerprint(info: MySQLConnectionInfo) -> str:
    """One-way identity of the *target server + account* (never credentials)."""
    raw = f"mysql://{info.host.lower()}:{int(info.port)}/{info.user}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _mongo_conn_fingerprint(uri: str) -> str:
    """Server part of the URI, hashed: password changes keep the same target."""
    scheme, _, rest = uri.partition("://")
    hosts = rest.split("@", 1)[1] if "@" in rest else rest
    raw = f"mongo://{scheme}://{hosts}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _keys_digest(items: Any) -> str:
    canon = json.dumps(sorted(str(i) for i in items))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _token_body(session_id: str, target: str, conn_fp: str, scope: str,
                keys_digest: str, iat: int) -> str:
    return "|".join([session_id, target, conn_fp, scope, keys_digest, str(iat)])


def _sign(body: str) -> str:
    return hmac.new(
        settings.secret_key.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _make_preview_token(*, session_id: str, target: str, conn_fp: str,
                        scope: str, keys_digest: str) -> str:
    iat = int(time.time())
    return f"{iat}.{_sign(_token_body(session_id, target, conn_fp, scope, keys_digest, iat))}"


def _verify_preview_token(token: str, *, session_id: str, target: str,
                          conn_fp: str, scope: str, keys_digest: str) -> None:
    """Reject any delete whose selection/target drifted since the preview."""
    parts = (token or "").split(".", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=409,
            detail="Invalid preview token. Run the preview again.",
        )
    try:
        iat = int(parts[0])
    except ValueError:
        raise HTTPException(
            status_code=409, detail="Invalid preview token. Run the preview again."
        )
    age = time.time() - iat
    if age > _PREVIEW_TOKEN_TTL_SECONDS or age < -60:
        raise HTTPException(
            status_code=409,
            detail="The preview expired (valid for 15 minutes). Run the preview again.",
        )
    expected = _sign(_token_body(session_id, target, conn_fp, scope, keys_digest, iat))
    if not hmac.compare_digest(parts[1], expected):
        raise HTTPException(
            status_code=409,
            detail="The preview no longer matches this request (the selection, "
                   "table/collection or connection changed). Run the preview again.",
        )


def _select_working_rows(df: pd.DataFrame, row_ids: List[int]) -> pd.DataFrame:
    """Validate the row-id selection against the working copy (400 on stale)."""
    df = _ensure_row_ids(df)
    ids = list(dict.fromkeys(int(i) for i in row_ids))
    present = {int(v) for v in df[ROW_ID_COL].tolist()}
    stale = [i for i in ids if i not in present]
    if stale:
        shown = ", ".join(str(i) for i in stale[:10])
        more = ", ..." if len(stale) > 10 else ""
        raise HTTPException(
            status_code=400,
            detail=(
                f"Stale row selection: {len(stale)} row(s) are no longer in "
                f"the dataset ({shown}{more}). Refresh the preview and try again."
            ),
        )
    return df[df[ROW_ID_COL].isin(ids)]


def _pk_value(v: Any, column: str) -> Any:
    """Convert a working-copy PK value to a parameterisable Python value."""
    import numpy as _np

    null = v is None
    if not null:
        try:
            null = bool(pd.isna(v))
        except (TypeError, ValueError):
            null = False
    if null:
        raise HTTPException(
            status_code=400,
            detail=f"Primary-key column '{column}' has a missing value for a "
                   "selected row; rows without a key value cannot be matched "
                   "in the source table.",
        )
    if isinstance(v, _np.integer):
        return int(v)
    if isinstance(v, _np.floating):
        return float(v)
    if isinstance(v, _np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _map_row_ids_to_pk(df: pd.DataFrame, row_ids: List[int], pk_cols: Sequence) -> List[tuple]:
    """Map working-copy row ids to primary-key tuples for the source table."""
    sub = _select_working_rows(df, row_ids)
    missing = [c for c in pk_cols if c not in sub.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The working copy no longer contains primary-key column(s) "
                f"{missing}; rows cannot be matched to the source table. "
                "Reconnect the table and retry."
            ),
        )
    keys: List[tuple] = []
    for _, row in sub.iterrows():
        keys.append(tuple(_pk_value(row[c], c) for c in pk_cols))
    return keys


def _require_mongo_identity(meta: Dict[str, Any]) -> str:
    """Validate that this session can map rows to real ``_id`` values."""
    sm = meta.get("source_meta") or {}
    kind = sm.get("id_kind")
    if not kind:
        raise HTTPException(
            status_code=400,
            detail="This session was created before source identity tracking and "
                   "cannot be mapped to real documents. Reconnect the collection "
                   "to enable permanent deletion.",
        )
    if not sm.get("identity_ok"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Row identities in this session do not round-trip safely "
                "(string ids that look numeric are re-typed by storage), so "
                "permanent deletion is disabled to avoid matching the wrong "
                "document. Working-copy edits remain available."
            ),
        )
    if kind not in ("objectid", "string", "int"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Document _id types in this collection ({kind}) cannot be "
                "mapped unambiguously; permanent deletion is disabled. "
                "Working-copy edits remain available."
            ),
        )
    return kind


def _decode_mongo_row_ids(df: pd.DataFrame, row_ids: List[int], meta: Dict[str, Any]) -> List[Any]:
    """Working-copy row ids → real ``_id`` BSON values (validated)."""
    kind = _require_mongo_identity(meta)
    sub = _select_working_rows(df, row_ids)
    if "_id" not in sub.columns:
        raise HTTPException(
            status_code=400,
            detail="The working copy no longer contains '_id'; reconnect the "
                   "collection to enable permanent deletion.",
        )
    values = sub["_id"].tolist()

    if kind == "objectid":
        from bson import ObjectId as _OID
        from bson.errors import InvalidId

        out: List[Any] = []
        for v in values:
            try:
                out.append(_OID(str(v)))
            except (InvalidId, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Row identity '{v}' is not a valid ObjectId; the "
                           "working copy appears corrupted. Reconnect the collection.",
                )
        return out
    if kind == "string":
        return [str(v) for v in values]

    # int
    out_int: List[Any] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (
            isinstance(v, float) and not float(v).is_integer()
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Row identity '{v}' is not an integer; reconnect the collection.",
            )
        out_int.append(int(v))
    return out_int


def _validate_mongo_fields(fields: List[str]) -> List[str]:
    """Only plain top-level, non-identity field names may be unset."""
    out: List[str] = []
    for f in dict.fromkeys(fields):
        if not f or "\0" in f:
            raise HTTPException(status_code=400, detail="Invalid field name.")
        if f == "_id":
            raise HTTPException(
                status_code=400,
                detail="'_id' is the document identity and cannot be removed.",
            )
        if f == ROW_ID_COL:
            raise HTTPException(
                status_code=400,
                detail=f"'{ROW_ID_COL}' is an internal column and cannot be deleted.",
            )
        if "." in f:
            raise HTTPException(
                status_code=400,
                detail=f"Field '{f}': only top-level fields can be removed "
                       "(dotted paths are not supported).",
            )
        if f.startswith("$"):
            raise HTTPException(
                status_code=400,
                detail=f"Field '{f}': names must not start with '$'.",
            )
        out.append(f)
    return out


def _gzip_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return gzip.compress(buf.getvalue().encode("utf-8"))


def _gzip_jsonl(docs: List[Dict[str, Any]]) -> bytes:
    from bson import json_util

    text = "\n".join(json_util.dumps(d) for d in docs)
    return gzip.compress(text.encode("utf-8"))


async def _make_backup(
    owner_id: Any,
    *,
    kind: str,
    source_name: str,
    payload: bytes,
    meta: Dict[str, Any],
    require_backup: bool,
    warnings: List[str],
) -> Dict[str, Any]:
    """Create + verify a backup; abort the deletion when one is required."""
    import backups as backup_store

    try:
        created = await backup_store.create_backup(
            owner_id, kind=kind, source_name=source_name, payload=payload, meta=meta
        )
        return await backup_store.verify_backup(owner_id, created["backup_id"])
    except backup_store.BackupError as exc:
        if require_backup:
            raise HTTPException(
                status_code=exc.status_code,
                detail=f"Backup failed: {exc} No source data was modified.",
            )
        warnings.append(f"Backup failed ({exc}); continuing because require_backup=false.")
        return {"skipped": True, "reason": str(exc)}


async def _retire_session(owner_id: Any, session_id: Optional[str]) -> None:
    """Drop a stale session after its source changed (best effort)."""
    if not session_id:
        return
    try:
        await session_store.delete_session(owner_id, session_id)
    except session_store.SessionError:  # pragma: no cover - non-fatal
        logger.warning("Could not retire session %s after source delete", session_id)


def _sample_records(rows: List[Dict[str, Any]], hide: Any = ()) -> List[Dict[str, Any]]:
    if not rows:
        return []
    records = _obj_to_records(pd.DataFrame(rows))
    hidden = set(hide)
    if hidden:
        records = [{k: v for k, v in r.items() if k not in hidden} for r in records]
    return records


@app.post("/source_delete/preview/", tags=["destructive"])
async def source_delete_preview(
    payload: SourceDeletePreviewPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Inspect exactly what a permanent delete would touch + issue its token."""
    meta, source, db, name = await _delete_session_context(
        current_user.id, payload.session_id, payload.target
    )

    if payload.target == "mysql_rows":
        if not payload.row_ids:
            raise HTTPException(status_code=400, detail="row_ids are required for this target.")
        if not payload.mysql_connection:
            raise HTTPException(status_code=400, detail="MySQL connection details are required.")
        return await _preview_mysql_rows(current_user, payload, db, name, meta)

    if payload.target == "mysql_columns":
        if not payload.columns:
            raise HTTPException(status_code=400, detail="columns are required for this target.")
        if not payload.mysql_connection:
            raise HTTPException(status_code=400, detail="MySQL connection details are required.")
        return await _preview_mysql_columns(current_user, payload, db, name, meta)

    if payload.target == "mongo_documents":
        if not payload.row_ids:
            raise HTTPException(status_code=400, detail="row_ids are required for this target.")
        if not payload.mongo_connection:
            raise HTTPException(status_code=400, detail="MongoDB connection details are required.")
        return await _preview_mongo_documents(current_user, payload, db, name, meta)

    # mongo_fields
    if not payload.fields:
        raise HTTPException(status_code=400, detail="fields are required for this target.")
    if not payload.mongo_connection:
        raise HTTPException(status_code=400, detail="MongoDB connection details are required.")
    return await _preview_mongo_fields(current_user, payload, db, name, meta)


def _check_connection_database(info_database: Optional[str], db: str) -> None:
    if info_database and info_database != db:
        raise HTTPException(
            status_code=400,
            detail=f"Connection database '{info_database}' does not match the "
                   f"session's source database '{db}'.",
        )


async def _preview_mysql_rows(user, payload, db: str, table: str, meta) -> Dict[str, Any]:
    info = payload.mysql_connection
    _check_connection_database(info.database, db)
    _mysql_conn_info_check()
    df = await _load_working_or_400(user.id, payload.session_id)
    import backups as backup_store

    try:
        with db_mysql.connection(
            host=info.host, user=info.user, password=info.password,
            port=info.port, database=db,
        ) as conn:
            pk = db_mysql.get_primary_key(conn, db, table)
            if not pk:
                raise HTTPException(
                    status_code=409,
                    detail=f"Table '{db}.{table}' has no PRIMARY KEY, so rows "
                           "cannot be identified safely for deletion. Add a "
                           "primary key (or delete rows from the table view) "
                           "and reconnect.",
                )
            keys = _map_row_ids_to_pk(df, payload.row_ids, pk)
            matched = db_mysql.count_by_pk(conn, db, table, pk, keys)
            sample = db_mysql.select_by_pk(conn, db, table, pk, keys, limit=20)
    except db_mysql.DatabaseError as exc:
        _raise_mysql_error(exc)

    token = _make_preview_token(
        session_id=payload.session_id, target="mysql_rows",
        conn_fp=_mysql_conn_fingerprint(info), scope=f"{db}.{table}",
        keys_digest=_keys_digest("|".join(str(v) for v in k) for k in keys),
    )
    warnings = []
    if matched < len(keys):
        warnings.append(
            f"{len(keys) - matched} of {len(keys)} selected rows no longer exist "
            "in the source table; only existing rows can be deleted."
        )
    return {
        "preview_token": token,
        "target": "mysql_rows",
        "session_id": payload.session_id,
        "source": "mysql",
        "database": db,
        "table": table,
        "primary_key": pk,
        "requested_count": len(keys),
        "matched_count": int(matched),
        "sample": _sample_records(sample),
        "warnings": warnings,
        "irreversible": True,
        "backup": {
            "will_create": True,
            "kind": "mysql_rows",
            "format": "csv",
            "encoding": "gzip",
            "expires_in_days": settings.backup_ttl_days,
            "restoration": backup_store.RESTORATION["mysql_rows"],
        },
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }


async def _preview_mysql_columns(user, payload, db: str, table: str, meta) -> Dict[str, Any]:
    info = payload.mysql_connection
    _check_connection_database(info.database, db)
    _mysql_conn_info_check()
    requested = list(dict.fromkeys(payload.columns))
    if ROW_ID_COL in requested:
        raise HTTPException(
            status_code=400,
            detail=f"'{ROW_ID_COL}' is an internal column and cannot be deleted.",
        )
    import backups as backup_store

    try:
        with db_mysql.connection(
            host=info.host, user=info.user, password=info.password,
            port=info.port, database=db,
        ) as conn:
            schema = db_mysql.describe_table_full(conn, db, table)
            live_cols = [r["Field"] for r in schema]
            missing = [c for c in requested if c not in live_cols]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column(s) not found in '{db}.{table}': {missing}. "
                           "Refresh the table connection and try again.",
                )
            if len(requested) >= len(live_cols):
                raise HTTPException(
                    status_code=400,
                    detail="Cannot drop every column of the table; at least one "
                           "column must remain.",
                )
            pk = db_mysql.get_primary_key(conn, db, table)
            risks: Dict[str, Any] = {}
            hard: List[str] = []
            warnings: List[str] = []
            for c in requested:
                try:
                    r = db_mysql.column_drop_risks(conn, db, table, c)
                except db_mysql.DatabaseError as exc:
                    _raise_mysql_error(exc)
                risks[c] = r
                for fk in r["foreign_keys_out"]:
                    hard.append(f"'{c}' participates in foreign key '{fk['constraint']}' → {fk['references']}")
                for fk in r["foreign_keys_in"]:
                    hard.append(f"'{c}' is referenced by foreign key '{fk['constraint']}' on table '{fk['table']}'")
                if r["is_primary"]:
                    hard.append(f"'{c}' is part of the PRIMARY KEY")
                if r["is_generated"]:
                    hard.append(f"'{c}' is a generated column")
                for g in r["referenced_by_generated"]:
                    hard.append(f"'{c}' is used by generated column '{g}'")
                for idx in r["indexes"]:
                    if not idx["primary"]:
                        warnings.append(
                            f"Index '{idx['name']}' will be dropped along with column '{c}'."
                        )
            if hard:
                raise HTTPException(
                    status_code=409,
                    detail="Unsafe to drop: " + "; ".join(hard[:6]) + ". "
                           "Resolve these dependencies first (e.g. drop the "
                           "foreign key); no data was modified.",
                )
            if not pk:
                warnings.append(
                    "The table has no PRIMARY KEY; the backup cannot be "
                    "re-keyed automatically."
                )
            warnings.append(
                "ALTER TABLE commits immediately (MySQL DDL auto-commit): a "
                "dropped column cannot be rolled back by a transaction — use "
                "the backup to restore it."
            )
            backup_cols = list(dict.fromkeys(list(pk) + requested))
            sample = db_mysql.select_columns(conn, db, table, backup_cols, limit=20)
            schema_meta = {
                r["Field"]: {
                    "type": r["Type"], "null": r["Null"], "default": r["Default"],
                    "extra": r["Extra"],
                }
                for r in schema if r["Field"] in requested
            }
    except db_mysql.DatabaseError as exc:
        _raise_mysql_error(exc)

    token = _make_preview_token(
        session_id=payload.session_id, target="mysql_columns",
        conn_fp=_mysql_conn_fingerprint(info), scope=f"{db}.{table}",
        keys_digest=_keys_digest(requested),
    )
    return {
        "preview_token": token,
        "target": "mysql_columns",
        "session_id": payload.session_id,
        "source": "mysql",
        "database": db,
        "table": table,
        "primary_key": pk,
        "columns": requested,
        "schema": schema_meta,
        "matched_count": len(requested),
        "requested_count": len(requested),
        "sample": _sample_records(sample),
        "warnings": warnings,
        "irreversible": True,
        "backup": {
            "will_create": True,
            "kind": "mysql_columns",
            "format": "csv",
            "encoding": "gzip",
            "expires_in_days": settings.backup_ttl_days,
            "restoration": backup_store.RESTORATION["mysql_columns"],
        },
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }


async def _preview_mongo_documents(user, payload, db: str, coll_name: str, meta) -> Dict[str, Any]:
    info = payload.mongo_connection
    df = await _load_working_or_400(user.id, payload.session_id)
    import backups as backup_store

    client = await _mongo_client(info)
    try:
        coll = client[db][coll_name]
        ids = _decode_mongo_row_ids(df, payload.row_ids, meta)
        matched = await coll.count_documents({"_id": {"$in": ids}})
        sample = await coll.find({"_id": {"$in": ids}}).limit(20).to_list(20)
    except HTTPException:
        raise
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.error("mongo documents preview failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to inspect the collection.")
    finally:
        client.close()

    token = _make_preview_token(
        session_id=payload.session_id, target="mongo_documents",
        conn_fp=_mongo_conn_fingerprint(info.uri), scope=f"{db}.{coll_name}",
        keys_digest=_keys_digest(str(i) for i in ids),
    )
    warnings = []
    if matched < len(ids):
        warnings.append(
            f"{len(ids) - matched} of {len(ids)} selected documents no longer "
            "exist in the collection; only existing documents can be deleted."
        )
    return {
        "preview_token": token,
        "target": "mongo_documents",
        "session_id": payload.session_id,
        "source": "mongodb",
        "database": db,
        "collection": coll_name,
        "primary_key": ["_id"],
        "requested_count": len(ids),
        "matched_count": int(matched),
        "sample": _sample_records(sample, hide={"_id"}),
        "warnings": warnings,
        "irreversible": True,
        "backup": {
            "will_create": True,
            "kind": "mongo_documents",
            "format": "jsonl",
            "encoding": "gzip",
            "expires_in_days": settings.backup_ttl_days,
            "restoration": backup_store.RESTORATION["mongo_documents"],
        },
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }


async def _preview_mongo_fields(user, payload, db: str, coll_name: str, meta) -> Dict[str, Any]:
    info = payload.mongo_connection
    fields = _validate_mongo_fields(payload.fields)
    import backups as backup_store

    client = await _mongo_client(info)
    try:
        coll = client[db][coll_name]
        counts = {f: int(await coll.count_documents({f: {"$exists": True}})) for f in fields}
        projection = {**{f: 1 for f in fields}, "_id": 1}
        sample = await coll.find({}, projection).limit(20).to_list(20)
    except HTTPException:
        raise
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.error("mongo fields preview failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to inspect the collection.")
    finally:
        client.close()

    token = _make_preview_token(
        session_id=payload.session_id, target="mongo_fields",
        conn_fp=_mongo_conn_fingerprint(info.uri), scope=f"{db}.{coll_name}",
        keys_digest=_keys_digest(fields),
    )
    warnings = []
    missing = [f for f, n in counts.items() if n == 0]
    if missing:
        warnings.append(
            f"Field(s) {missing} do not exist in any document and will be skipped."
        )
    return {
        "preview_token": token,
        "target": "mongo_fields",
        "session_id": payload.session_id,
        "source": "mongodb",
        "database": db,
        "collection": coll_name,
        "fields": fields,
        "field_counts": counts,
        "matched_count": int(max(counts.values()) if counts else 0),
        "requested_count": len(fields),
        "sample": _sample_records(sample, hide={"_id"}),
        "warnings": warnings,
        "irreversible": True,
        "backup": {
            "will_create": True,
            "kind": "mongo_field_values",
            "format": "jsonl",
            "encoding": "gzip",
            "expires_in_days": settings.backup_ttl_days,
            "restoration": backup_store.RESTORATION["mongo_field_values"],
        },
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/source_delete/mysql_rows/", tags=["destructive"])
async def source_delete_mysql_rows(
    payload: SourceDeleteMySQLRowsPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Permanently DELETE rows from the live MySQL table (by primary key)."""
    _require_destructive_intent(payload.confirm, payload.typed_confirmation)
    if payload.target != "mysql_rows":
        raise HTTPException(status_code=400, detail="Target does not match this endpoint.")
    meta, _, db, table = await _delete_session_context(
        current_user.id, payload.session_id, payload.target
    )
    info = payload.connection_details
    _check_connection_database(info.database, db)
    _mysql_conn_info_check()
    df = await _load_working_or_400(current_user.id, payload.session_id)
    warnings: List[str] = []
    import backups as backup_store

    new_session_id: Optional[str] = None
    state: Dict[str, Any] = {"columns": [], "preview": [], "row_count": 0}
    refresh_error: Optional[str] = None

    try:
        with db_mysql.connection(
            host=info.host, user=info.user, password=info.password,
            port=info.port, database=db,
        ) as conn:
            pk = db_mysql.get_primary_key(conn, db, table)
            if not pk:
                raise HTTPException(
                    status_code=409,
                    detail=f"Table '{db}.{table}' has no PRIMARY KEY; refusing "
                           "to delete rows that cannot be identified safely.",
                )
            keys = _map_row_ids_to_pk(df, payload.row_ids, pk)
            _verify_preview_token(
                payload.preview_token, session_id=payload.session_id,
                target="mysql_rows", conn_fp=_mysql_conn_fingerprint(info),
                scope=f"{db}.{table}",
                keys_digest=_keys_digest("|".join(str(v) for v in k) for k in keys),
            )
            matched = db_mysql.count_by_pk(conn, db, table, pk, keys)
            if matched == 0:
                raise HTTPException(
                    status_code=409,
                    detail=f"None of the {len(keys)} selected row(s) exist in "
                           f"'{db}.{table}' anymore. Nothing was deleted — run "
                           "the preview again.",
                )
            if matched < len(keys) and not payload.allow_partial:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{matched} of {len(keys)} selected rows exist in the "
                        "source table; the table changed since your selection. "
                        "Run the preview again, or resubmit with "
                        "allow_partial=true to delete only the existing rows."
                    ),
                )

            # Backup BEFORE the delete (abort on failure when required).
            backup_rows = db_mysql.select_by_pk(conn, db, table, pk, keys)
            backup = await _make_backup(
                current_user.id,
                kind="mysql_rows",
                source_name=f"{db}.{table}",
                payload=_gzip_csv(pd.DataFrame(backup_rows)),
                meta={
                    "format": "csv", "database": db, "table": table,
                    "primary_key": list(pk),
                    "columns": list(backup_rows[0].keys()) if backup_rows else [],
                    "row_count": len(backup_rows),
                },
                require_backup=payload.require_backup,
                warnings=warnings,
            )

            affected = db_mysql.delete_by_pk(conn, db, table, pk, keys)

            # Re-read the table so the UI shows the post-delete truth.
            try:
                columns = db_mysql.describe_table(conn, db, table)
                rows = db_mysql.fetch_all(
                    conn, db, table, limit=settings.max_rows_per_dataset + 1
                )
            except Exception as exc:  # pragma: no cover - keep honest
                logger.error("Post-delete refresh failed: %s", exc, exc_info=True)
                rows = None  # type: ignore
                refresh_error = str(exc)
    except db_mysql.DatabaseError as exc:
        _raise_mysql_error(exc)

    if rows is None:
        warnings.append(
            "The deletion succeeded, but re-reading the table failed "
            f"({refresh_error}); the view still shows pre-delete data. "
            "Reconnect the table to refresh."
        )
    else:
        if len(rows) > settings.max_rows_per_dataset:
            rows = rows[: settings.max_rows_per_dataset]
            warnings.append("Table truncated to the dataset row limit for the preview.")
        new_df = (
            pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
        )
        try:
            new_session_id = await session_store.create_session(
                current_user.id,
                new_df,
                source="mysql",
                source_name=f"{db}.{table}",
                file_type="mysql",
                source_meta={"database": db, "table": table},
            )
            await _retire_session(current_user.id, payload.session_id)
        except session_store.SessionError as exc:  # pragma: no cover
            warnings.append(f"Deleted, but the session could not be refreshed: {exc}")
            new_session_id = None
        state = _edit_state(new_df) if new_session_id else state

    if affected < matched:
        warnings.append(
            f"The table engine reported {affected} row(s) affected while "
            f"{matched} matched (some engines/tables do not report deletes or "
            "do not use transactions); verify the table."
        )
    logger.warning(
        "PERMANENT delete: %d row(s) from %s.%s by user %s (matched=%d)",
        affected, db, table, current_user.id, matched,
    )
    await _record_history(
        current_user.id,
        session_id=payload.session_id,
        operation="source_delete_mysql_rows",
        source_type="mysql", source_name=f"{db}.{table}",
        source_meta={"database": db, "table": table},
        summary=f"Permanently deleted {affected} row(s) from '{db}.{table}'.",
        before={"row_count": int(matched)},
        backup_id=backup.get("backup_id"),
        restorable=bool(backup.get("backup_id")),
        rollback_expires_at=backup.get("expires_at"),
        extra={"affected": int(affected), "matched": int(matched)},
        warnings=warnings,
    )
    return {
        "message": f"Permanently deleted {affected} row(s) from '{db}.{table}'.",
        "target": "mysql_rows",
        "source": "mysql",
        "database": db,
        "table": table,
        "requested_count": len(keys),
        "matched_count": int(matched),
        "affected": int(affected),
        "backup": backup,
        "warnings": warnings,
        "session_id": new_session_id,
        **state,
    }


@app.post("/source_delete/mysql_columns/", tags=["destructive"])
async def source_delete_mysql_columns(
    payload: SourceDeleteMySQLColumnsPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Permanently ALTER TABLE ... DROP COLUMN on the live MySQL table."""
    _require_destructive_intent(payload.confirm, payload.typed_confirmation)
    if payload.target != "mysql_columns":
        raise HTTPException(status_code=400, detail="Target does not match this endpoint.")
    meta, _, db, table = await _delete_session_context(
        current_user.id, payload.session_id, payload.target
    )
    info = payload.connection_details
    _check_connection_database(info.database, db)
    _mysql_conn_info_check()
    requested = list(dict.fromkeys(payload.columns))
    warnings: List[str] = []
    import backups as backup_store

    new_session_id: Optional[str] = None
    state: Dict[str, Any] = {"columns": [], "preview": [], "row_count": 0}
    refresh_error: Optional[str] = None
    dropped: List[str] = []
    failure: Optional[Dict[str, str]] = None

    try:
        with db_mysql.connection(
            host=info.host, user=info.user, password=info.password,
            port=info.port, database=db,
        ) as conn:
            schema = db_mysql.describe_table_full(conn, db, table)
            live_cols = [r["Field"] for r in schema]
            missing = [c for c in requested if c not in live_cols]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column(s) not found in '{db}.{table}': {missing}. "
                           "Refresh the table connection and try again.",
                )
            if len(requested) >= len(live_cols):
                raise HTTPException(
                    status_code=400,
                    detail="Cannot drop every column of the table; at least one "
                           "column must remain.",
                )
            pk = db_mysql.get_primary_key(conn, db, table)
            hard: List[str] = []
            for c in requested:
                try:
                    r = db_mysql.column_drop_risks(conn, db, table, c)
                except db_mysql.DatabaseError as exc:
                    _raise_mysql_error(exc)
                if r["foreign_keys_out"] or r["foreign_keys_in"] or r["is_primary"] \
                        or r["is_generated"] or r["referenced_by_generated"]:
                    hard.append(c)
            if hard:
                raise HTTPException(
                    status_code=409,
                    detail=f"Unsafe to drop {hard} (foreign key, primary key or "
                           "generated-column dependency). Resolve those first; "
                           "no data was modified.",
                )
            _verify_preview_token(
                payload.preview_token, session_id=payload.session_id,
                target="mysql_columns", conn_fp=_mysql_conn_fingerprint(info),
                scope=f"{db}.{table}", keys_digest=_keys_digest(requested),
            )

            # Backup the column values (+ keys) BEFORE any DDL.
            backup_cols = list(dict.fromkeys(list(pk) + requested))
            backup_rows = db_mysql.select_columns(conn, db, table, backup_cols)
            schema_meta = {
                r["Field"]: {
                    "type": r["Type"], "null": r["Null"], "default": r["Default"],
                    "extra": r["Extra"],
                }
                for r in schema if r["Field"] in requested
            }
            backup = await _make_backup(
                current_user.id,
                kind="mysql_columns",
                source_name=f"{db}.{table}",
                payload=_gzip_csv(pd.DataFrame(backup_rows)),
                meta={
                    "format": "csv", "database": db, "table": table,
                    "primary_key": list(pk),
                    "columns": schema_meta,
                    "row_count": len(backup_rows),
                },
                require_backup=payload.require_backup,
                warnings=warnings,
            )

            for c in requested:
                try:
                    db_mysql.drop_column(conn, db, table, c)
                    dropped.append(c)
                except db_mysql.DatabaseError as exc:
                    failure = {"column": c, "error": str(exc)}
                    break
            if not failure:
                warnings.append(
                    "MySQL DDL auto-commits: the dropped column(s) cannot be "
                    "rolled back — use the backup to restore them."
                )

            try:
                columns = db_mysql.describe_table(conn, db, table)
                rows = db_mysql.fetch_all(
                    conn, db, table, limit=settings.max_rows_per_dataset + 1
                )
            except Exception as exc:  # pragma: no cover
                logger.error("Post-drop refresh failed: %s", exc, exc_info=True)
                rows = None  # type: ignore
                refresh_error = str(exc)
    except db_mysql.DatabaseError as exc:
        _raise_mysql_error(exc)

    if rows is None:
        warnings.append(
            f"The schema change was applied, but re-reading the table failed "
            f"({refresh_error}); reconnect the table to refresh."
        )
    else:
        if len(rows) > settings.max_rows_per_dataset:
            rows = rows[: settings.max_rows_per_dataset]
        new_df = (
            pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
        )
        try:
            new_session_id = await session_store.create_session(
                current_user.id,
                new_df,
                source="mysql",
                source_name=f"{db}.{table}",
                file_type="mysql",
                source_meta={"database": db, "table": table},
            )
            await _retire_session(current_user.id, payload.session_id)
        except session_store.SessionError as exc:  # pragma: no cover
            warnings.append(f"Schema changed, but the session could not be refreshed: {exc}")
            new_session_id = None
        state = _edit_state(new_df) if new_session_id else state

    if failure:
        logger.warning(
            "PARTIAL column drop: %s by user %s (dropped=%s, failed on %s)",
            requested, current_user.id, dropped, failure["column"],
        )
        message = (
            f"Dropped {len(dropped)} of {len(requested)} column(s) "
            f"({dropped}) before '{failure['column']}' failed: "
            f"{failure['error']}. The table WAS modified; the backup "
            f"(id {backup.get('backup_id')}) preserves the dropped values."
        )
    else:
        logger.warning(
            "PERMANENT delete: column(s) %s from %s.%s by user %s",
            dropped, db, table, current_user.id,
        )
        message = (
            f"Permanently dropped column(s) {dropped} from '{db}.{table}'."
        )
    await _record_history(
        current_user.id,
        session_id=payload.session_id,
        operation="source_delete_mysql_columns",
        status="partial" if failure else "success",
        source_type="mysql", source_name=f"{db}.{table}",
        source_meta={"database": db, "table": table},
        summary=message,
        extra={"dropped_columns": dropped, "requested": requested},
        backup_id=backup.get("backup_id"),
        restorable=bool(backup.get("backup_id")),
        rollback_expires_at=backup.get("expires_at"),
        warnings=warnings,
    )
    return {
        "message": message,
        "target": "mysql_columns",
        "source": "mysql",
        "database": db,
        "table": table,
        "requested_count": len(requested),
        "matched_count": len(dropped),
        "affected": len(dropped),
        "dropped_columns": dropped,
        "partial_failure": failure,
        "backup": backup,
        "warnings": warnings,
        "session_id": new_session_id,
        **state,
    }


@app.post("/source_delete/mongo_documents/", tags=["destructive"])
async def source_delete_mongo_documents(
    payload: SourceDeleteMongoDocumentsPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Permanently delete documents from the live MongoDB collection."""
    _require_destructive_intent(payload.confirm, payload.typed_confirmation)
    if payload.target != "mongo_documents":
        raise HTTPException(status_code=400, detail="Target does not match this endpoint.")
    meta, _, db, coll_name = await _delete_session_context(
        current_user.id, payload.session_id, payload.target
    )
    df = await _load_working_or_400(current_user.id, payload.session_id)
    warnings: List[str] = []
    import backups as backup_store

    client = await _mongo_client(payload.mongo_connection)
    try:
        coll = client[db][coll_name]
        ids = _decode_mongo_row_ids(df, payload.row_ids, meta)
        _verify_preview_token(
            payload.preview_token, session_id=payload.session_id,
            target="mongo_documents",
            conn_fp=_mongo_conn_fingerprint(payload.mongo_connection.uri),
            scope=f"{db}.{coll_name}",
            keys_digest=_keys_digest(str(i) for i in ids),
        )
        matched = int(await coll.count_documents({"_id": {"$in": ids}}))
        if matched == 0:
            raise HTTPException(
                status_code=409,
                detail=f"None of the {len(ids)} selected document(s) exist in "
                       f"'{db}.{coll_name}' anymore. Nothing was deleted — run "
                       "the preview again.",
            )
        if matched < len(ids) and not payload.allow_partial:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{matched} of {len(ids)} selected documents exist in the "
                    "collection; it changed since your selection. Run the "
                    "preview again, or resubmit with allow_partial=true to "
                    "delete only the existing documents."
                ),
            )

        matched_docs = await coll.find({"_id": {"$in": ids}}).to_list(matched)
        backup = await _make_backup(
            current_user.id,
            kind="mongo_documents",
            source_name=f"{db}.{coll_name}",
            payload=_gzip_jsonl(matched_docs),
            meta={
                "format": "jsonl", "database": db, "collection": coll_name,
                "primary_key": ["_id"], "document_count": len(matched_docs),
            },
            require_backup=payload.require_backup,
            warnings=warnings,
        )

        result = await coll.delete_many({"_id": {"$in": ids}})
        affected = int(result.deleted_count)

        try:
            refreshed = await _mongo_session_from_live(
                current_user.id, client, db, coll_name
            )
        except Exception as exc:  # pragma: no cover - honest degradation
            logger.error("Post-delete refresh failed: %s", exc, exc_info=True)
            refreshed = None
            warnings.append(
                f"The deletion succeeded, but re-reading the collection failed "
                f"({exc}); reconnect the collection to refresh."
            )
    except HTTPException:
        raise
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.error("mongo documents delete failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to delete documents from the collection (see server logs).",
        )
    finally:
        client.close()

    new_session_id: Optional[str] = None
    state: Dict[str, Any] = {"columns": [], "preview": [], "row_count": 0}
    if refreshed is not None:
        new_session_id = refreshed["session_id"]  # None when now empty
        state = {
            "columns": refreshed["columns"],
            "preview": refreshed["preview"],
            "row_count": refreshed["row_count"],
        }
        await _retire_session(current_user.id, payload.session_id)

    if affected < matched:
        warnings.append(
            f"The collection reported {affected} deleted while {matched} "
            "matched; verify the collection."
        )
    logger.warning(
        "PERMANENT delete: %d document(s) from %s.%s by user %s (matched=%d)",
        affected, db, coll_name, current_user.id, matched,
    )
    await _record_history(
        current_user.id,
        session_id=payload.session_id,
        operation="source_delete_mongo_documents",
        source_type="mongodb", source_name=f"{db}.{coll_name}",
        source_meta={"database": db, "collection": coll_name},
        summary=f"Permanently deleted {affected} document(s) from '{db}.{coll_name}'.",
        before={"row_count": int(matched)},
        backup_id=backup.get("backup_id"),
        restorable=bool(backup.get("backup_id")),
        rollback_expires_at=backup.get("expires_at"),
        extra={"affected": int(affected), "matched": int(matched)},
        warnings=warnings,
    )
    return {
        "message": f"Permanently deleted {affected} document(s) from '{db}.{coll_name}'.",
        "target": "mongo_documents",
        "source": "mongodb",
        "database": db,
        "collection": coll_name,
        "requested_count": len(ids),
        "matched_count": matched,
        "affected": affected,
        "backup": backup,
        "warnings": warnings,
        "session_id": new_session_id,
        **state,
    }


@app.post("/source_delete/mongo_fields/", tags=["destructive"])
async def source_delete_mongo_fields(
    payload: SourceDeleteMongoFieldsPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Permanently $unset top-level fields from live documents."""
    _require_destructive_intent(payload.confirm, payload.typed_confirmation)
    if payload.target != "mongo_fields":
        raise HTTPException(status_code=400, detail="Target does not match this endpoint.")
    meta, _, db, coll_name = await _delete_session_context(
        current_user.id, payload.session_id, payload.target
    )
    fields = _validate_mongo_fields(payload.fields)
    warnings: List[str] = []
    import backups as backup_store

    client = await _mongo_client(payload.mongo_connection)
    try:
        coll = client[db][coll_name]
        _verify_preview_token(
            payload.preview_token, session_id=payload.session_id,
            target="mongo_fields",
            conn_fp=_mongo_conn_fingerprint(payload.mongo_connection.uri),
            scope=f"{db}.{coll_name}", keys_digest=_keys_digest(fields),
        )
        counts = {f: int(await coll.count_documents({f: {"$exists": True}})) for f in fields}
        total = sum(counts.values())
        if total == 0:
            raise HTTPException(
                status_code=409,
                detail=f"None of the requested fields exist in any document of "
                       f"'{db}.{coll_name}'. Nothing was deleted — run the "
                       "preview again.",
            )
        for f, n in counts.items():
            if n == 0:
                warnings.append(f"Field '{f}' does not exist in any document (skipped).")

        live_fields = [f for f in fields if counts[f] > 0]
        projection = {**{f: 1 for f in live_fields}, "_id": 1}
        backup_docs = await coll.find(
            {"$or": [{f: {"$exists": True}} for f in live_fields]}, projection
        ).to_list(settings.max_rows_per_dataset)
        backup = await _make_backup(
            current_user.id,
            kind="mongo_field_values",
            source_name=f"{db}.{coll_name}",
            payload=_gzip_jsonl(backup_docs),
            meta={
                "format": "jsonl", "database": db, "collection": coll_name,
                "fields": [f for f in fields if counts[f] > 0],
                "document_count": len(backup_docs),
            },
            require_backup=payload.require_backup,
            warnings=warnings,
        )

        results: Dict[str, Any] = {}
        for f in fields:
            if counts[f] == 0:
                results[f] = {"matched": 0, "modified": 0}
                continue
            res = await coll.update_many({f: {"$exists": True}}, {"$unset": {f: ""}})
            results[f] = {"matched": int(res.matched_count), "modified": int(res.modified_count)}

        try:
            refreshed = await _mongo_session_from_live(
                current_user.id, client, db, coll_name
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Post-unset refresh failed: %s", exc, exc_info=True)
            refreshed = None
            warnings.append(
                f"The fields were removed, but re-reading the collection failed "
                f"({exc}); reconnect the collection to refresh."
            )
    except HTTPException:
        raise
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.error("mongo fields delete failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to remove fields from the collection (see server logs).",
        )
    finally:
        client.close()

    new_session_id: Optional[str] = None
    state: Dict[str, Any] = {"columns": [], "preview": [], "row_count": 0}
    if refreshed is not None:
        new_session_id = refreshed["session_id"]
        state = {
            "columns": refreshed["columns"],
            "preview": refreshed["preview"],
            "row_count": refreshed["row_count"],
        }
        await _retire_session(current_user.id, payload.session_id)

    affected_fields = [f for f in fields if results[f]["modified"] > 0]
    logger.warning(
        "PERMANENT delete: field(s) %s from %s.%s by user %s",
        affected_fields, db, coll_name, current_user.id,
    )
    await _record_history(
        current_user.id,
        session_id=payload.session_id,
        operation="source_delete_mongo_fields",
        source_type="mongodb", source_name=f"{db}.{coll_name}",
        source_meta={"database": db, "collection": coll_name},
        summary=(
            f"Permanently removed field(s) {affected_fields} from "
            f"{sum(r['modified'] for r in results.values())} document(s)."
        ),
        extra={"fields": fields, "field_results": results},
        backup_id=backup.get("backup_id"),
        restorable=bool(backup.get("backup_id")),
        rollback_expires_at=backup.get("expires_at"),
        warnings=warnings,
    )
    return {
        "message": (
            f"Permanently removed field(s) {affected_fields} from "
            f"{sum(r['modified'] for r in results.values())} document(s) in "
            f"'{db}.{coll_name}'."
        ),
        "target": "mongo_fields",
        "source": "mongodb",
        "database": db,
        "collection": coll_name,
        "requested_count": len(fields),
        "matched_count": total,
        "affected": sum(r["modified"] for r in results.values()),
        "field_results": results,
        "backup": backup,
        "warnings": warnings,
        "session_id": new_session_id,
        **state,
    }


@app.get("/backups/", tags=["destructive"])
async def list_delete_backups(current_user: AppUser = Depends(get_current_active_user)):
    """All of the caller's pre-delete backups (ownership-scoped)."""
    import backups as backup_store

    try:
        return {"backups": await backup_store.list_backups(current_user.id)}
    except backup_store.BackupError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


@app.get("/backups/{backup_id}/download/", tags=["destructive"])
async def download_delete_backup(
    backup_id: str, current_user: AppUser = Depends(get_current_active_user)
):
    """Download one owned backup (gzip CSV or gzip Extended JSON lines)."""
    import backups as backup_store

    try:
        data, meta = await backup_store.read_backup(current_user.id, backup_id)
    except backup_store.BackupError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    ext = "jsonl.gz" if meta.get("format") == "jsonl" else "csv.gz"
    filename = f"tabula-backup-{backup_id[:12]}.{ext}"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------- #
# Transformation history + seven-day rollback
# --------------------------------------------------------------------------- #
class RestoreWorkingPayload(BaseModel):
    session_id: str
    version_id: str
    confirm: bool = False


class OpenVersionPayload(BaseModel):
    version_id: str
    history_id: Optional[str] = None


async def _resolve_version_source(
    current_user: AppUser,
    version: Dict[str, Any],
    history_id: Optional[str],
) -> Dict[str, Any]:
    """Resolve the source identity behind a version snapshot.

    Prefers the linked (owned) history entry; falls back to the originating
    session's metadata when it still exists.  Never queries a live database.
    """
    source = "file"
    source_name = None
    file_type = None
    source_meta: Dict[str, Any] = {}
    entry: Optional[Dict[str, Any]] = None
    if history_id:
        entry = await history_store.get_entry(current_user.id, history_id)
        if entry.get("version_id") and entry["version_id"] != version["version_id"]:
            raise HTTPException(
                status_code=400,
                detail="The history entry does not reference this version snapshot.",
            )
        source = entry.get("source_type") or "file"
        source_name = entry.get("source_name")
        source_meta = dict(entry.get("source_meta") or {})
        file_type = source_meta.get("file_type")
    if entry is None:
        try:
            meta = await session_store.get_session_meta(
                current_user.id, version["session_id"]
            )
            source = meta.get("source") or "file"
            source_name = meta.get("source_name")
            file_type = meta.get("file_type")
            source_meta = dict(meta.get("source_meta") or {})
        except session_store.SessionError:
            pass
    if not file_type:
        file_type = source if source in ("mysql", "mongodb") else None
    return {
        "entry": entry,
        "source": source,
        "source_name": source_name,
        "file_type": file_type,
        "source_meta": source_meta,
    }


async def _create_working_session_from_version(
    current_user: AppUser,
    version: Dict[str, Any],
    df: pd.DataFrame,
    *,
    resolved: Dict[str, Any],
    operation: str,
    label: str,
    summary: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Isolate a version snapshot into a brand-new working session.

    The historical snapshot itself is never mutated: the new session owns its
    own GridFS blobs, so later edits/transforms cannot touch the saved version.
    """
    source = resolved.get("source") or "file"
    hide = _mongo_hidden({"source": source, "file_type": resolved.get("file_type")})
    history_marker: Dict[str, Any] = {
        "version_id": version["version_id"],
        "version_number": version["version_number"],
        "operation": operation,
        "opened_at": datetime.now(timezone.utc),
    }
    if resolved.get("entry"):
        history_marker["history_id"] = resolved["entry"].get("history_id")
    source_meta = {**(resolved.get("source_meta") or {}), "history": history_marker}
    session_id = await session_store.create_session(
        current_user.id,
        df,
        source=source,
        source_name=resolved.get("source_name"),
        file_type=resolved.get("file_type"),
        source_meta=source_meta,
    )
    await _record_and_snapshot(
        current_user.id, session_id, df,
        operation=operation,
        label=label,
        summary=summary,
        source_type=source,
        source_name=resolved.get("source_name"),
        after=_counts(df, hide),
        extra={
            **(extra or {}),
            "opened_from_version_id": version["version_id"],
            "opened_from_version_number": version["version_number"],
        },
    )
    state = _edit_state(df, hide)
    return {
        "session_id": session_id,
        "source": source,
        "source_name": resolved.get("source_name"),
        "file_type": resolved.get("file_type"),
        **state,
    }


@app.get("/history/", tags=["history"])
async def list_history(
    session_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    source_type: Optional[str] = Query(
        None,
        description="Optional source category filter: file, mysql or mongodb. "
                    "Entries whose category cannot be proven are excluded.",
    ),
    current_user: AppUser = Depends(get_current_active_user),
):
    """Newest-first audit trail for the caller (optionally one session).

    Each entry carries nested ``version`` snapshot metadata (row/column counts,
    expiry, ``restorable``) so the UI can offer open/download/restore actions
    without trusting client-side state.

    ``source_type`` scopes the listing to one data-source category (validated
    server-side; always combined with the caller's identity).  Without it the
    full account-wide audit trail is returned.
    """
    try:
        entries = await history_store.list_history(
            current_user.id,
            session_id=session_id,
            limit=limit,
            source_type=source_type,
        )
    except history_store.HistoryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return {"history": entries, "recovery_window_days": settings.backup_ttl_days}


@app.get("/history/versions/", tags=["history"])
async def list_history_versions(
    session_id: str = Query(...),
    current_user: AppUser = Depends(get_current_active_user),
):
    """Version snapshots of a session's working copy (unexpired only)."""
    try:
        versions = await history_store.list_versions(current_user.id, session_id)
    except history_store.HistoryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return {"versions": versions, "recovery_window_days": settings.backup_ttl_days}


@app.get("/history/versions/{version_id}/download/", tags=["history"])
async def download_history_version(
    version_id: str,
    format: str = Query("csv", pattern="^(csv|json|xlsx)$"),
    current_user: AppUser = Depends(get_current_active_user),
):
    """Download the EXACT saved dataset of a version snapshot.

    Ownership and expiry are re-verified server-side; the current working copy
    and any live database are never touched.  The internal row-id column is
    stripped; Mongo ``_id`` is preserved (needed to restore document identity).
    """
    try:
        version = await history_store.get_version_by_id(current_user.id, version_id)
        df = await history_store.read_version_df_by_id(current_user.id, version_id)
    except history_store.HistoryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    if ROW_ID_COL in df.columns:
        df = df.drop(columns=[ROW_ID_COL])

    base = "dataset"
    source_fmt = "csv"
    try:
        meta = await session_store.get_session_meta(
            current_user.id, version["session_id"]
        )
        base = meta.get("source_name") or base
        source_fmt = (meta.get("file_type") or "csv").lower()
    except session_store.SessionError:
        if version.get("history_id"):
            try:
                entry = await history_store.get_entry(
                    current_user.id, version["history_id"]
                )
                base = entry.get("source_name") or base
            except history_store.HistoryError:
                pass
    stem = str(base).rsplit(".", 1)[0] if "." in str(base) else str(base)
    stem = stem.replace("/", "_").replace("\\", "_") or "dataset"
    fmt = format.lower()
    filename = f"{stem}_v{version['version_number']}.{fmt}"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Snapshot-Version": str(version["version_number"]),
        "X-Format-Converted": "true" if fmt != source_fmt else "false",
    }
    logger.info(
        "Version download: v%s (%s) by user %s",
        version["version_number"], fmt, current_user.id,
    )
    if fmt == "json":
        payload = df.to_json(orient="records", force_ascii=False, indent=2)
        return StreamingResponse(
            iter([payload]), media_type="application/json", headers=headers
        )
    if fmt == "xlsx":
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return StreamingResponse(
            iter([out.getvalue()]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@app.post("/history/open_version/", tags=["history"])
async def open_history_version(
    payload: OpenVersionPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Open a saved version as a NEW isolated working session.

    Loads the exact historical snapshot (never the live database or a stale
    in-memory frame) into a fresh session so the workspace can transform,
    preview, profile, and download it.  The historical snapshot itself is
    preserved read-only and is never modified by subsequent edits.
    """
    try:
        version = await history_store.get_version_by_id(
            current_user.id, payload.version_id
        )
        df = await history_store.read_version_df_by_id(
            current_user.id, payload.version_id
        )
        resolved = await _resolve_version_source(
            current_user, version, payload.history_id
        )
    except history_store.HistoryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    try:
        out = await _create_working_session_from_version(
            current_user, version, df,
            resolved=resolved,
            operation="open_version",
            label=f"Opened v{version['version_number']} as a new working copy",
            summary=(
                f"Opened version v{version['version_number']} "
                f"({version.get('row_count', len(df))} rows) as a new isolated "
                "working copy. The historical snapshot was not modified."
            ),
            extra={"source_session_id": version["session_id"]},
        )
    except session_store.SessionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    logger.info(
        "Opened version v%s (session %s) as session %s by user %s",
        version["version_number"], version["session_id"],
        out["session_id"], current_user.id,
    )
    return {
        "message": (
            f"Opened version v{version['version_number']} as a new working "
            "copy. The historical snapshot was preserved."
        ),
        "version_id": version["version_id"],
        "from_version_number": version["version_number"],
        "from_operation": (resolved.get("entry") or {}).get("operation"),
        **out,
    }


@app.post("/history/restore_working/", tags=["history"])
async def restore_working_copy(
    payload: RestoreWorkingPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Restore the WORKING COPY to a stored version snapshot.

    This never touches a connected live database.  If the originating session
    has expired or been deleted, a fresh isolated session is created from the
    snapshot instead of failing.  The restore itself is recorded (and
    snapshotted) so it appears in history and can be rolled forward again.
    """
    if not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required: set confirm=true to restore the "
                   "working copy (the live source database is not modified).",
        )
    try:
        version = await history_store.get_version(
            current_user.id, payload.session_id, payload.version_id
        )
        df = await history_store.read_version_df(
            current_user.id, payload.session_id, payload.version_id
        )
    except history_store.HistoryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    session_id = payload.session_id
    try:
        df = _ensure_row_ids(df)
        await session_store.save_transformed(current_user.id, session_id, df)
        meta = await _session_meta_or_empty(current_user.id, session_id)
    except session_store.SessionError as exc:
        if exc.status_code != 404:
            raise HTTPException(status_code=exc.status_code, detail=str(exc))
        # The originating session expired or was deleted: isolate the
        # snapshot into a brand-new working session instead of failing.
        try:
            resolved = await _resolve_version_source(current_user, version, None)
            out = await _create_working_session_from_version(
                current_user, version, df,
                resolved=resolved,
                operation="restore_working_copy",
                label=f"Restored working copy to v{version['version_number']}",
                summary=(
                    f"Restored version v{version['version_number']} into a new "
                    f"session ({version.get('row_count', len(df))} rows); the "
                    "original session had expired. The live source database "
                    "was not modified."
                ),
                extra={"restored_version_number": version["version_number"]},
            )
        except session_store.SessionError as exc2:
            raise HTTPException(status_code=exc2.status_code, detail=str(exc2))
        await session_store.set_source_meta(
            current_user.id, out["session_id"],
            {
                "history": {
                    "version_id": version["version_id"],
                    "version_number": version["version_number"],
                    "operation": "restore_working_copy",
                    "restored_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        logger.warning(
            "Working-copy restore (new session %s): snapshot v%s by user %s",
            out["session_id"], version["version_number"], current_user.id,
        )
        return {
            "message": (
                f"Working copy restored to version v{version['version_number']} "
                f"as new session {out['session_id']} (the original session had "
                "expired). The source database was not modified."
            ),
            "restored_version_number": version["version_number"],
            "transformed_columns": [],
            "session_id": out["session_id"],
            "new_session": True,
            "source": out.get("source"),
            "source_name": out.get("source_name"),
            "file_type": out.get("file_type"),
            **{k: out[k] for k in ("columns", "preview", "row_count")},
        }

    hide = _mongo_hidden(meta)
    await _record_and_snapshot(
        current_user.id, session_id, df,
        operation="restore_working_copy",
        label=f"Restored working copy to v{version['version_number']}",
        summary=(
            f"Restored the working copy to version "
            f"v{version['version_number']} "
            f"({version.get('row_count', len(df))} rows). The live source "
            "database was not modified."
        ),
        source_type=meta.get("source"), source_name=meta.get("source_name"),
        after=_counts(df, hide),
        extra={"restored_version_number": version["version_number"]},
    )
    logger.warning(
        "Working-copy restore: session %s -> v%s by user %s",
        payload.session_id, version["version_number"], current_user.id,
    )
    # Stamp the session with the restored-version marker so the UI can show
    # "restored to vN" after a refresh without trusting client-side state.
    await session_store.set_source_meta(
        current_user.id, session_id,
        {
            "history": {
                "version_id": version["version_id"],
                "version_number": version["version_number"],
                "operation": "restore_working_copy",
                "restored_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    )
    state = _edit_state(df, hide)
    return {
        "message": (
            f"Working copy restored to version v{version['version_number']}. "
            "The source database was not modified."
        ),
        "restored_version_number": version["version_number"],
        "transformed_columns": [],
        "session_id": session_id,
        "source": meta.get("source"),
        "source_name": meta.get("source_name"),
        "file_type": meta.get("file_type"),
        **state,
    }


# --------------------------------------------------------------------------- #
# Selective / full history clearing
# --------------------------------------------------------------------------- #
class ClearHistoryPayload(BaseModel):
    history_ids: List[str] = Field(min_length=1, max_length=500)
    confirm: bool = False


class ClearAllHistoryPayload(BaseModel):
    confirm: bool = False
    typed_confirmation: str = ""
    # Optional source category ("file" / "mysql" / "mongodb") scoping the
    # clear to one section; omitted keeps the pre-existing global meaning.
    source_type: Optional[str] = None


@app.get("/history/stats/", tags=["history"])
async def history_stats(
    source_type: Optional[str] = Query(
        None,
        description="Optional source category filter: file, mysql or mongodb.",
    ),
    current_user: AppUser = Depends(get_current_active_user),
):
    """Owner-scoped counts for the clear-history confirmation dialogs.

    ``source_type`` scopes the counts to one category, matching the
    category-scoped clear-all action.
    """
    try:
        return await history_store.history_stats(
            current_user.id, source_type=source_type
        )
    except history_store.HistoryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


def _clear_result_response(
    result: Dict[str, Any], scope_label: str = ""
) -> Dict[str, Any]:
    """Shape a clear result into an HTTP response with an honest message.

    ``scope_label`` (e.g. "file", "MySQL") names the source category when the
    clear was scoped to one section, so the message cannot be misread as a
    global wipe.
    """
    deleted = result.get("entries_deleted", 0)
    requested = result.get("requested", 0)
    missing = len(result.get("entries_missing") or [])
    failures = result.get("failures") or []
    scope = f"{scope_label} " if scope_label else ""
    parts = [
        f"Cleared {deleted} {scope}history entr{'y' if deleted == 1 else 'ies'}"
    ]
    snap_d = result.get("snapshots_deleted", 0)
    bak_d = result.get("backups_deleted", 0)
    artifacts = snap_d + bak_d
    if artifacts:
        parts.append(f"removed {artifacts} recovery snapshot(s)")
    retained = result.get("snapshots_retained", 0) + result.get("backups_retained", 0)
    if retained:
        parts.append(f"retained {retained} still referenced by other entries")
    if missing:
        parts.append(f"{missing} id(s) not found or not owned")
    if failures:
        parts.append(f"{len(failures)} cleanup failure(s) — see `failures`")
    message = "; ".join(parts) + "."
    return {
        "message": message,
        "partial": bool(failures),
        **result,
    }


@app.post("/history/clear/", tags=["history"])
async def clear_selected_history(
    payload: ClearHistoryPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Delete selected history entries (and eligible snapshots) of the caller.

    Ownership, reference-sharing, and idempotency are revalidated server-side;
    the active working dataset and live source databases are never touched.
    """
    if not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required: set confirm=true to clear the "
                   "selected history entries and their recovery snapshots.",
        )
    ids = [i for i in dict.fromkeys(x.strip() for x in payload.history_ids) if i]
    if not ids:
        raise HTTPException(
            status_code=400, detail="No valid history ids supplied."
        )
    if len(ids) > 500:
        raise HTTPException(
            status_code=400,
            detail="Too many entries in one request (max 500); "
                   "use Clear All History for a full wipe.",
        )
    try:
        result = await history_store.clear_history_entries(current_user.id, ids)
    except history_store.HistoryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return _clear_result_response(result)


@app.post("/history/clear_all/", tags=["history"])
async def clear_all_history(
    payload: ClearAllHistoryPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Delete the caller's history entries (and eligible snapshots).

    Requires an explicit typed confirmation.  When ``source_type`` is given
    (file / mysql / mongodb) only that source category is cleared — other
    categories, the active working dataset, and any live database are never
    modified.  Without ``source_type`` the historical global behaviour is
    preserved.  Only the authenticated user's records are ever touched.
    """
    if not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required: set confirm=true to clear all "
                   "history (the live source database is not modified).",
        )
    if payload.typed_confirmation != "CLEAR HISTORY":
        raise HTTPException(
            status_code=400,
            detail="Type CLEAR HISTORY exactly (uppercase, with a space) to "
                   "confirm permanently clearing all history entries.",
        )
    try:
        # Validates the category (400 on anything unknown) via the store.
        source = history_store.normalize_source_type(payload.source_type)
        result = await history_store.clear_all_history(
            current_user.id, source_type=source
        )
    except history_store.HistoryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    logger.warning(
        "Clear-all history: user %s removed %d entries (source=%s)",
        current_user.id, result.get("entries_deleted", 0), source or "all",
    )
    scope = history_store.source_type_label(source) if source else ""
    return _clear_result_response(result, scope_label=scope)


@app.delete("/history/{history_id}", tags=["history"])
async def delete_history_entry(
    history_id: str,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Delete ONE owned history entry from persistent storage.

    The authoritative ``history`` document is removed only when it is owned by
    the authenticated user; otherwise the response is 404 so neither existence
    nor ownership leaks.  Associated recovery snapshots/backups are cleaned up
    only when no other surviving entry still references them (retained counts
    are reported).  The active working dataset and live source databases are
    never touched.  Repeating the request is safe: the second call reports 404
    for the already-removed entry and changes nothing else.
    """
    hid = (history_id or "").strip()
    if not hid or len(hid) > 64 or not all(c.isalnum() or c in "-_" for c in hid):
        # Malformed ids can never match a real entry; fail before touching Mongo.
        raise HTTPException(status_code=400, detail="Invalid history id.")
    try:
        result = await history_store.clear_history_entries(current_user.id, [hid])
    except history_store.HistoryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    if hid in (result.get("entries_missing") or []):
        # Unknown id, another user's entry, or an already-deleted one.
        raise HTTPException(
            status_code=404, detail="History entry not found or access denied."
        )
    if result.get("entries_deleted") == 1:
        logger.info(
            "Delete history entry %s by user %s (snapshots=%d backups=%d "
            "retained_v=%d retained_b=%d failures=%d)",
            hid, current_user.id, result.get("snapshots_deleted", 0),
            result.get("backups_deleted", 0), result.get("snapshots_retained", 0),
            result.get("backups_retained", 0), len(result.get("failures") or []),
        )
        # Honest result: deleted plus exactly what cleanup achieved/retained.
        return {**_clear_result_response(result), "deleted": True}
    # The document was owned but its removal failed: never claim success.
    raise HTTPException(
        status_code=500,
        detail="The history entry could not be deleted; please retry. "
               "Nothing outside history was modified.",
    )


# --------------------------------------------------------------------------- #
# Data quality & transformation preflight validation
# --------------------------------------------------------------------------- #
class QualityProfilePayload(BaseModel):
    session_id: str


class QualityValidatePayload(BaseModel):
    session_id: str
    column: str
    examples: List[ExampleItem]
    transformation_id: Optional[str] = None


@app.post("/quality/profile/", tags=["quality"])
async def quality_profile(
    payload: QualityProfilePayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Read-only quality profile of the session's CURRENT working copy."""
    df = await _load_working_or_400(current_user.id, payload.session_id)
    meta = await _session_meta_or_empty(current_user.id, payload.session_id)
    hide = set(_mongo_hidden(meta))
    view = df[[c for c in df.columns if c not in hide]] if hide else df
    profile = quality_module.profile_dataframe(
        view,
        source=meta.get("source"),
        source_name=meta.get("source_name"),
    )
    return {"session_id": payload.session_id, **profile}


@app.post("/quality/validate/", tags=["quality"])
async def quality_validate(
    payload: QualityValidatePayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Preflight a transformation before applying it to the whole column.

    Blocking errors (malformed examples, missing column, a rule that cannot
    reproduce the examples or fails every sampled value) always prevent
    application; warnings are reviewable.
    """
    df = await _load_working_or_400(current_user.id, payload.session_id)
    pairs = _pairs_from_payload_or_400(payload.examples)

    code = None
    strategy = ""  # no transformation_id referenced -> no strategy note
    if payload.transformation_id:
        trans_doc = await _get_owned_transformation(
            current_user.id, payload.transformation_id
        )
        spec = TransformationSpec.from_dict(trans_doc.get("spec"))
        strategy = spec.strategy
        if strategy != STRATEGY_MODEL:
            code = trans_doc.get("code")

    result = quality_module.validate_examples_for_column(
        df, payload.column, pairs, code=code, apply_fn=apply_to_series
    )
    if strategy == STRATEGY_MODEL:
        # Model-backed transformations have no sandbox code to preflight;
        # every row is validated by the execution path at apply time.
        info = list(result.get("info") or [])
        info.append(
            "This transformation is model-backed: rows are processed and "
            "validated at apply time rather than by a preflight code check."
        )
        result["info"] = info
    return {"session_id": payload.session_id, **result}


# --------------------------------------------------------------------------- #
# Source-database RESTORE from a verified recovery artifact
#
# Distinct from working-copy restore: this writes to the live MySQL table or
# MongoDB collection.  Contract:
#   1. the backup must exist (ownership-scoped) and be unexpired;
#   2. the current source state is backed up and verified FIRST — if that
#      fails, nothing is restored;
#   3. conflicts (source rows/fields that changed since the snapshot) are
#      detected and require overwrite_conflicts=true;
#   4. confirm=true AND typed_confirmation="RESTORE" (exact);
#   5. actual operation counts are reported; partial failure is honest.
# --------------------------------------------------------------------------- #
RESTOREABLE_KINDS = (
    "mysql_rows",
    "mysql_columns",
    "mysql_column_values",
    "mongo_documents",
    "mongo_field_values",
)


class SourceRestorePreviewPayload(BaseModel):
    backup_id: str
    mysql_connection: Optional[MySQLConnectionInfo] = None
    mongo_connection: Optional[MongoConnectionInfo] = None


class SourceRestorePayload(BaseModel):
    backup_id: str
    confirm: bool = False
    typed_confirmation: str = ""
    overwrite_conflicts: bool = False
    mysql_connection: Optional[MySQLConnectionInfo] = None
    mongo_connection: Optional[MongoConnectionInfo] = None
    # The caller's active session.  After the source is restored the stale
    # session is retired and a refreshed one (live data) is returned so the
    # UI can adopt it without a manual reconnect.
    session_id: Optional[str] = None


async def _load_owned_backup(owner_id: Any, backup_id: str):
    """Ownership + expiry enforced; returns (meta, payload_bytes)."""
    import backups as backup_store

    try:
        meta = await backup_store.get_backup_meta(owner_id, backup_id)
        data, meta = await backup_store.read_backup(owner_id, backup_id)
    except backup_store.BackupError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    kind = meta.get("kind")
    if kind not in RESTOREABLE_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Backup kind '{kind}' cannot be restored automatically. "
                   "Download it from the backups list and restore manually.",
        )
    expires = meta.get("expires_at")
    if expires is not None and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if not expires or expires <= datetime.now(timezone.utc):
        raise HTTPException(
            status_code=409,
            detail="This backup has expired (seven-day recovery window) and "
                   "can no longer be restored. The metadata is kept for audit.",
        )
    return meta, data


def _parse_backup_rows(data: bytes, fmt: str) -> pd.DataFrame:
    if fmt == "jsonl":
        from bson import json_util

        lines = [ln for ln in gzip.decompress(data).decode("utf-8").splitlines() if ln]
        return pd.DataFrame([json_util.loads(ln) for ln in lines])
    return pd.read_csv(io.BytesIO(gzip.decompress(data)))


def _same_value(a: Any, b: Any) -> bool:
    """Compare a restored value against the live one across type drift.

    ``None``/NaN, numeric strings, and ``1`` vs ``1.0`` must not be treated
    as verification failures; genuinely different values must.
    """
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    if a is None or b is None:
        return a is None and b is None
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _verify_mysql_restore(
    conn: Any, db: str, table: str, kind: str,
    bmeta: Dict[str, Any], df: pd.DataFrame, pk: List[str],
) -> Optional[Dict[str, Any]]:
    """Re-read the live table after the restore writes and confirm the
    snapshot landed.  Returns ``None`` when verification cannot run —
    never raises, so a completed write is never turned into a 5xx."""
    try:
        if kind == "mysql_columns":
            schema = bmeta.get("columns") or {}
            expected_cols = list(schema.keys())
            live_cols = set(db_mysql.describe_table(conn, db, table))
            missing = [c for c in expected_cols if c not in live_cols]
            return {
                "kind": kind,
                "expected": len(expected_cols),
                "found": len(expected_cols) - len(missing),
                "mismatches": len(missing),
                "verified": bool(expected_cols) and not missing,
                "missing_columns": missing,
            }

        col = str(bmeta.get("column") or "") if kind == "mysql_column_values" else ""
        keys = _sqlite_pk_keys(df, pk) if pk else []
        expected = len(keys)
        if not keys:
            return {
                "kind": kind, "expected": expected, "found": 0,
                "mismatches": 0, "verified": expected == 0,
                "note": "no rows or primary key to verify",
            }

        found = db_mysql.count_by_pk(conn, db, table, pk, keys)
        mismatches = 0
        # Value spot-check is capped so huge backups stay cheap; presence is
        # always checked for EVERY restored row.
        sample_keys = keys[:500]
        live_rows = db_mysql.select_by_pk(conn, db, table, pk, sample_keys)
        live_by_key = {tuple(r.get(k) for k in pk): r for r in live_rows}
        df_by_key = {tuple(row.get(k) for k in pk): row for _, row in df.iterrows()}
        for key in sample_keys:
            live = live_by_key.get(key)
            src = df_by_key.get(key)
            if live is None or src is None:
                continue
            cols = [col] if col else [c for c in df.columns if c not in pk]
            if any(c in live and not _same_value(live.get(c), src.get(c)) for c in cols):
                mismatches += 1
        return {
            "kind": kind, "expected": expected, "found": int(found),
            "mismatches": mismatches,
            "verified": int(found) == expected and mismatches == 0,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Post-restore MySQL verification could not run: %s", exc)
        return None


async def _verify_mongo_restore(
    coll: Any, kind: str, bmeta: Dict[str, Any], docs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Re-read the collection after the restore writes (see MySQL variant)."""
    try:
        ids = [d["_id"] for d in docs if "_id" in d]
        expected = len(ids)
        if not expected:
            return {
                "kind": kind, "expected": 0, "found": 0,
                "mismatches": 0, "verified": True,
            }
        found_docs = await coll.find({"_id": {"$in": ids}}).to_list(len(ids))
        by_id = {d["_id"]: d for d in found_docs}
        mismatches = 0
        fields = list(bmeta.get("fields") or []) if kind == "mongo_field_values" else None
        for d in docs[:500 if fields is None else len(docs)]:
            live = by_id.get(d.get("_id"))
            if live is None:
                continue
            check = fields if fields is not None else list(d.keys())
            if any(not _same_value(live.get(k), d.get(k)) for k in check if k != "_id"):
                mismatches += 1
        return {
            "kind": kind, "expected": expected, "found": len(by_id),
            "mismatches": mismatches,
            "verified": len(by_id) == expected and mismatches == 0,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Post-restore Mongo verification could not run: %s", exc)
        return None


def _honest_restore_message(
    verification: Optional[Dict[str, Any]], warnings: List[str], message: str,
) -> str:
    """Never present a partial/unverified restore as a clean success."""
    if verification is None:
        warnings.append(
            "Post-restore verification could not run; the live source was "
            "not re-checked."
        )
        return message + " The live source could not be verified."
    if not verification.get("verified"):
        warnings.append(
            "Post-restore verification reported discrepancies: "
            f"{verification.get('found')} of {verification.get('expected')} "
            f"rows/documents confirmed ({verification.get('mismatches', 0)} "
            "mismatched)."
        )
        return message + " The live source was verified WITH DISCREPANCIES — review the warnings."
    return message


def _parse_backup_docs(data: bytes) -> List[Dict[str, Any]]:
    from bson import json_util

    lines = [ln for ln in gzip.decompress(data).decode("utf-8").splitlines() if ln]
    return [json_util.loads(ln) for ln in lines]


def _require_restore_intent(confirm: bool, typed: str) -> None:
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required: set confirm=true. Restoring overwrites "
                   "live source data.",
        )
    if typed != "RESTORE":
        raise HTTPException(
            status_code=400,
            detail='Type the word RESTORE exactly (uppercase) to confirm.',
        )


def _sqlite_pk_keys(df: pd.DataFrame, pk: Sequence[str]) -> List[tuple]:
    """Backup rows -> PK tuples (null keys rejected)."""
    missing = [c for c in pk if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"The backup is missing primary-key column(s) {missing}; "
                   "it cannot be matched to the table. Restore manually from "
                   "the downloaded file.",
        )
    keys: List[tuple] = []
    for _, row in df.iterrows():
        keys.append(tuple(_pk_value(row[c], c) for c in pk))
    return keys


@app.post("/restore_source/preview/", tags=["destructive"])
async def restore_source_preview(
    payload: SourceRestorePreviewPayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Inspect what a source restore would do (counts, conflicts, expiry)."""
    meta, data = await _load_owned_backup(current_user.id, payload.backup_id)
    kind = meta["kind"]
    bmeta = meta.get("meta") or {}
    expires = meta.get("expires_at")
    if expires is not None and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    created = meta.get("created_at")
    if isinstance(created, datetime) and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    preview: Dict[str, Any] = {
        "backup_id": payload.backup_id,
        "kind": kind,
        "source_name": meta.get("source_name"),
        "created_at": created,
        "expires_at": expires,
        "restorable": bool(expires and expires > datetime.now(timezone.utc)),
        "requires_typed_confirmation": "RESTORE",
        "snapshot": {},
        "conflicts": None,
        "live": None,
        "warnings": [],
    }

    if kind in ("mysql_rows", "mysql_columns", "mysql_column_values"):
        if payload.mysql_connection is None:
            preview["warnings"].append(
                "Provide MySQL connection details to detect conflicts against "
                "the live table."
            )
            if kind == "mysql_rows":
                df = _parse_backup_rows(data, bmeta.get("format", "csv"))
                preview["snapshot"] = {"row_count": int(len(df)), "columns": list(df.columns)}
            elif kind == "mysql_columns":
                preview["snapshot"] = {
                    "columns": list((bmeta.get("columns") or {}).keys()),
                    "row_count": bmeta.get("row_count"),
                }
            else:
                preview["snapshot"] = {
                    "column": bmeta.get("column"),
                    "row_count": bmeta.get("row_count"),
                }
            return preview

        info = payload.mysql_connection
        db = str(bmeta.get("database") or "")
        table = str(bmeta.get("table") or "")
        _mysql_conn_info_check()
        _check_connection_database(info.database, db)
        try:
            with db_mysql.connection(
                host=info.host, user=info.user, password=info.password,
                port=info.port, database=db,
            ) as conn:
                if kind == "mysql_rows":
                    df = _parse_backup_rows(data, bmeta.get("format", "csv"))
                    pk = list(bmeta.get("primary_key") or [])
                    keys = _sqlite_pk_keys(df, pk)
                    existing = db_mysql.count_by_pk(conn, db, table, pk, keys)
                    preview["snapshot"] = {"row_count": int(len(df)), "columns": list(df.columns)}
                    preview["conflicts"] = {
                        "existing_rows": int(existing),
                        "explanation": (
                            "Rows whose primary key already exists would be "
                            "overwritten (requires overwrite_conflicts)."
                        ),
                    }
                    preview["live"] = {"rows_to_insert": int(len(keys) - existing)}
                elif kind == "mysql_columns":
                    df = _parse_backup_rows(data, bmeta.get("format", "csv"))
                    pk = list(bmeta.get("primary_key") or [])
                    cols = list((bmeta.get("columns") or {}).keys())
                    live_cols = db_mysql.describe_table(conn, db, table)
                    missing_cols = [c for c in cols if c not in live_cols]
                    preview["snapshot"] = {"columns": cols, "row_count": int(len(df))}
                    if missing_cols:
                        preview["conflicts"] = {
                            "missing_columns": missing_cols,
                            "explanation": "These columns will be re-added (ALTER TABLE).",
                        }
                    else:
                        preview["conflicts"] = {
                            "existing_columns": cols,
                            "explanation": "Columns already exist; their values would be updated from the snapshot.",
                        }
                    preview["live"] = {"table_columns": live_cols}
                else:  # mysql_column_values
                    df = _parse_backup_rows(data, bmeta.get("format", "csv"))
                    pk = list(bmeta.get("primary_key") or [])
                    col = str(bmeta.get("column") or "")
                    live_cols = db_mysql.describe_table(conn, db, table)
                    if col not in live_cols:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Column '{col}' no longer exists in "
                                   f"'{db}.{table}'. Restore it via its "
                                   "drop-column backup instead.",
                        )
                    keys = _sqlite_pk_keys(df, pk)
                    current_rows = db_mysql.select_by_pk(conn, db, table, pk, keys)
                    current_by_key = {
                        tuple(r.get(k) for k in pk): r for r in current_rows
                    }
                    differing = 0
                    for _, row in df.iterrows():
                        key = tuple(row.get(k) for k in pk)
                        cur = current_by_key.get(key)
                        if cur is not None and str(cur.get(col)) != str(row.get(col)):
                            differing += 1
                    preview["snapshot"] = {"column": col, "row_count": int(len(df))}
                    preview["conflicts"] = {
                        "differing_values": differing,
                        "missing_rows": int(len(keys) - len(current_rows)),
                        "explanation": (
                            "Rows whose current value differs from the snapshot "
                            "require overwrite_conflicts to be updated."
                        ),
                    }
        except db_mysql.DatabaseError as exc:
            _raise_mysql_error(exc)
        return preview

    # MongoDB kinds
    if payload.mongo_connection is None:
        preview["warnings"].append(
            "Provide the MongoDB connection to detect conflicts against the live collection."
        )
        docs = _parse_backup_docs(data)
        preview["snapshot"] = {"document_count": len(docs)}
        return preview

    db = str(bmeta.get("database") or "")
    coll_name = str(bmeta.get("collection") or "")
    client = await _mongo_client(payload.mongo_connection)
    try:
        coll = client[db][coll_name]
        docs = _parse_backup_docs(data)
        preview["snapshot"] = {
            "document_count": len(docs),
            "fields": bmeta.get("fields"),
        }
        if kind == "mongo_documents":
            ids = [d["_id"] for d in docs if "_id" in d]
            existing = int(await coll.count_documents({"_id": {"$in": ids}}))
            preview["conflicts"] = {
                "existing_documents": existing,
                "explanation": (
                    "Documents whose _id already exists would be replaced "
                    "(requires overwrite_conflicts); missing ones are inserted."
                ),
            }
            preview["live"] = {"documents_to_insert": int(len(ids) - existing)}
        else:
            fields = list(bmeta.get("fields") or [])
            differing = 0
            absent = 0
            missing_docs = 0
            for d in docs:
                cur = await coll.find_one({"_id": d["_id"]}) if "_id" in d else None
                if cur is None:
                    missing_docs += 1
                    continue
                for f in fields:
                    if f not in cur:
                        absent += 1
                    elif cur.get(f) != d.get(f):
                        differing += 1
            preview["conflicts"] = {
                "differing_values": differing,
                "absent_fields": absent,
                "missing_documents": missing_docs,
                "explanation": (
                    "Differing field values require overwrite_conflicts; absent "
                    "fields are re-set; documents that no longer exist are "
                    "reported and skipped unless overwrite_conflicts inserts them."
                ),
            }
    except HTTPException:
        raise
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.error("restore preview failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Restore preview failed.")
    finally:
        client.close()
    return preview


@app.post("/restore_source/", tags=["destructive"])
async def restore_source(
    payload: SourceRestorePayload,
    current_user: AppUser = Depends(get_current_active_user),
):
    """Restore live source data from a verified, unexpired backup."""
    _require_restore_intent(payload.confirm, payload.typed_confirmation)
    meta, data = await _load_owned_backup(current_user.id, payload.backup_id)
    kind = meta["kind"]
    bmeta = meta.get("meta") or {}
    warnings: List[str] = []
    results: Dict[str, Any] = {}
    db = str(bmeta.get("database") or "")
    target_name = str(bmeta.get("table") or bmeta.get("collection") or "")

    if kind in ("mysql_rows", "mysql_columns", "mysql_column_values"):
        if payload.mysql_connection is None:
            raise HTTPException(status_code=400, detail="MySQL connection details are required.")
        info = payload.mysql_connection
        _mysql_conn_info_check()
        _check_connection_database(info.database, db)
        df = _parse_backup_rows(data, bmeta.get("format", "csv"))
        pk = list(bmeta.get("primary_key") or [])
        pre_backup: Optional[Dict[str, Any]] = None
        verification: Optional[Dict[str, Any]] = None
        refreshed: Optional[pd.DataFrame] = None

        try:
            with db_mysql.connection(
                host=info.host, user=info.user, password=info.password,
                port=info.port, database=db,
            ) as conn:
                table = str(bmeta.get("table") or "")
                live_cols = db_mysql.describe_table(conn, db, table)

                # 1. Back up the CURRENT state first (abort on failure).
                if kind == "mysql_rows":
                    keys = _sqlite_pk_keys(df, pk)
                    current = db_mysql.select_by_pk(conn, db, table, pk, keys)
                    pre_backup = await _make_backup(
                        current_user.id, kind="mysql_rows",
                        source_name=f"{db}.{table}",
                        payload=_gzip_csv(pd.DataFrame(current)) if current
                        else gzip.compress(b""),
                        meta={
                            "format": "csv", "database": db, "table": table,
                            "primary_key": pk,
                            "columns": list(current[0].keys()) if current else [],
                            "row_count": len(current),
                        },
                        require_backup=True, warnings=warnings,
                    )
                    existing_keys = {
                        tuple(r.get(k) for k in pk) for r in current
                    }
                    to_insert = [
                        tuple(row.get(c) for c in df.columns if c in live_cols)
                        for _, row in df.iterrows()
                        if tuple(row.get(k) for k in pk) not in existing_keys
                    ]
                    conflicts = len(df) - len(to_insert)
                    if conflicts and not payload.overwrite_conflicts:
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                f"{conflicts} of {len(df)} row(s) already exist in "
                                f"'{db}.{table}' (primary key collision). Nothing "
                                "was restored — resubmit with "
                                "overwrite_conflicts=true to overwrite them."
                            ),
                        )
                    inserted = db_mysql.insert_rows(
                        conn, db, table,
                        [c for c in df.columns if c in live_cols], to_insert,
                    )
                    updated = 0
                    if conflicts:
                        shared = [c for c in df.columns if c in live_cols and c not in pk]
                        # Overwrite every shared column of the conflicting rows.
                        cursor_updates = 0
                        col_sql = ", ".join(
                            f"{db_mysql.quote_identifier(c, 'column name')}=%s"
                            for c in shared
                        ) if shared else None
                        if col_sql:
                            pk_sql = " AND ".join(
                                f"{db_mysql.quote_identifier(k, 'column name')}=%s" for k in pk
                            )
                            sql = (
                                f"UPDATE {db_mysql.quote_identifier(db, 'database name')}."
                                f"{db_mysql.quote_identifier(table, 'table name')} "
                                f"SET {col_sql} WHERE {pk_sql}"
                            )
                            cur = conn.cursor()
                            try:
                                for _, row in df.iterrows():
                                    key = tuple(row.get(k) for k in pk)
                                    if key in existing_keys:
                                        cur.execute(
                                            sql,
                                            tuple(row.get(c) for c in shared) + key,
                                        )
                                        cursor_updates += 1
                                conn.commit()
                            except Exception as exc:
                                try:
                                    conn.rollback()
                                except Exception:  # pragma: no cover
                                    pass
                                raise db_mysql.DatabaseError(
                                    f"Overwriting existing rows failed: {exc}", 400
                                )
                            finally:
                                cur.close()
                        updated = cursor_updates
                    results = {
                        "inserted": inserted,
                        "updated": updated,
                        "conflicts_overwritten": updated if conflicts else 0,
                    }

                elif kind == "mysql_columns":
                    cols_schema = bmeta.get("columns") or {}
                    pk = list(bmeta.get("primary_key") or pk)
                    missing_cols = [c for c in cols_schema if c not in live_cols]
                    added = []
                    for c in missing_cols:
                        info_c = cols_schema[c] or {}
                        db_mysql.add_column(
                            conn, db, table, c,
                            str(info_c.get("type") or "TEXT"),
                            nullable=str(info_c.get("null", "YES")).upper() != "NO",
                        )
                        added.append(c)
                    restored_cols = list(cols_schema.keys())
                    # Populate values for every restored column (by PK).
                    per_col = {}
                    for c in restored_cols:
                        if c not in df.columns or not pk:
                            per_col[c] = {"updated": 0, "missing": 0}
                            continue
                        summary_c = db_mysql.update_column_by_pk(
                            conn, db, table, c, pk,
                            df.to_dict(orient="records"),
                        )
                        per_col[c] = summary_c
                    results = {
                        "columns_added": added,
                        "columns_updated": restored_cols,
                        "per_column": per_col,
                    }
                    if not pk:
                        warnings.append(
                            "The backup recorded no primary key; column values "
                            "could not be written back — only the schema was restored."
                        )

                else:  # mysql_column_values
                    col = str(bmeta.get("column") or "")
                    if col not in live_cols:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Column '{col}' does not exist in '{db}.{table}'. "
                                   "Restore it from its drop-column backup instead.",
                        )
                    keys = _sqlite_pk_keys(df, pk)
                    current = db_mysql.select_by_pk(conn, db, table, pk, keys)
                    pre_backup = await _make_backup(
                        current_user.id, kind="mysql_column_values",
                        source_name=f"{db}.{table}",
                        payload=_gzip_csv(pd.DataFrame(current)) if current
                        else gzip.compress(b""),
                        meta={
                            "format": "csv", "database": db, "table": table,
                            "primary_key": pk, "column": col,
                            "columns": list(current[0].keys()) if current else [],
                            "row_count": len(current),
                        },
                        require_backup=True, warnings=warnings,
                    )
                    current_by_key = {
                        tuple(r.get(k) for k in pk): r for r in current
                    }
                    differing_rows = [
                        row for _, row in df.iterrows()
                        if tuple(row.get(k) for k in pk) in current_by_key
                        and str(current_by_key[tuple(row.get(k) for k in pk)].get(col))
                        != str(row.get(col))
                    ]
                    missing_rows = [
                        row for _, row in df.iterrows()
                        if tuple(row.get(k) for k in pk) not in current_by_key
                    ]
                    if (differing_rows or missing_rows) and not payload.overwrite_conflicts:
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                f"{len(differing_rows)} row(s) now hold different "
                                f"values and {len(missing_rows)} row(s) no longer "
                                "exist. Nothing was restored — resubmit with "
                                "overwrite_conflicts=true to proceed."
                            ),
                        )
                    upd = db_mysql.update_column_by_pk(
                        conn, db, table, col, pk, df.to_dict(orient="records"),
                    )
                    results = {
                        "column": col,
                        "updated": upd["updated"],
                        "missing_rows": upd["missing"],
                    }

                # While the connection is still open: (a) re-read the live
                # table to verify the writes actually landed, (b) capture a
                # fresh working copy so the caller can adopt it immediately.
                verification = _verify_mysql_restore(conn, db, table, kind, bmeta, df, pk)
                try:
                    live_cols2 = db_mysql.describe_table(conn, db, table)
                    live_rows = db_mysql.fetch_all(
                        conn, db, table,
                        limit=settings.max_rows_per_dataset + 1,
                    )
                    refreshed = (
                        pd.DataFrame(live_rows, columns=live_cols2)
                        if live_rows else pd.DataFrame(columns=live_cols2)
                    )
                except Exception as exc:  # pragma: no cover - honest degradation
                    logger.warning("Post-restore refresh failed: %s", exc)
                    refreshed = None
                    warnings.append(
                        f"The restore was applied, but re-reading "
                        f"'{db}.{table}' failed ({exc}); reconnect the table "
                        "to refresh the working copy."
                    )
        except db_mysql.DatabaseError as exc:
            _raise_mysql_error(exc)

        new_session_id: Optional[str] = None
        state: Dict[str, Any] = {"columns": [], "preview": [], "row_count": 0}
        if refreshed is not None:
            if len(refreshed):
                try:
                    new_session_id = await session_store.create_session(
                        current_user.id, refreshed,
                        source="mysql",
                        source_name=f"{db}.{target_name}",
                        file_type="mysql",
                        source_meta={"database": db, "table": target_name},
                    )
                    state = _edit_state(refreshed, ())
                except session_store.SessionError as exc:
                    warnings.append(
                        f"The restore was applied, but the refreshed working "
                        f"copy could not be created ({exc}); reconnect the "
                        "table to refresh."
                    )
            else:
                state = {"columns": list(refreshed.columns), "preview": [], "row_count": 0}
            # The previous session mirrors pre-restore data — retire it so
            # nothing stale stays selectable.
            await _retire_session(current_user.id, payload.session_id)

        message = _honest_restore_message(
            verification, warnings,
            f"Restored {kind} backup into '{db}.{target_name}'.",
        )
        logger.warning(
            "SOURCE RESTORE: %s of %s from backup %s by user %s (%s)",
            kind, f"{db}.{target_name}", payload.backup_id, current_user.id, results,
        )
        await _record_history(
            current_user.id,
            session_id=new_session_id,
            operation="restore_source",
            source_type="mysql", source_name=f"{db}.{target_name}",
            source_meta={"database": db, "table": target_name},
            summary=(
                f"Restored {kind} backup into '{db}.{target_name}': {results}."
            ),
            extra={"kind": kind, "results": results, "verification": verification},
            backup_id=(pre_backup or {}).get("backup_id"),
            restorable=bool((pre_backup or {}).get("backup_id")),
            rollback_expires_at=(pre_backup or {}).get("expires_at"),
            warnings=warnings,
        )
        return {
            "message": message,
            "kind": kind, "source": "mysql", "database": db, "table": target_name,
            "results": results, "pre_restore_backup": pre_backup,
            "warnings": warnings, "restored_backup_id": payload.backup_id,
            # Refreshed working copy the UI adopts without reconnecting
            # (session_id is None when the restored table is empty or the
            # refresh failed — the warnings say so honestly).
            "session_id": new_session_id,
            "source_name": f"{db}.{target_name}",
            "file_type": "mysql",
            "columns": state["columns"],
            "preview": state["preview"],
            "row_count": state["row_count"],
            "verification": verification,
        }

    # ---------------- MongoDB ---------------- #
    if payload.mongo_connection is None:
        raise HTTPException(status_code=400, detail="MongoDB connection details are required.")
    coll_name = str(bmeta.get("collection") or "")
    docs = _parse_backup_docs(data)
    client = await _mongo_client(payload.mongo_connection)
    pre_backup = None
    verification: Optional[Dict[str, Any]] = None
    refreshed: Optional[Dict[str, Any]] = None
    try:
        coll = client[db][coll_name]
        if kind == "mongo_documents":
            ids = [d["_id"] for d in docs if "_id" in d]
            current = await coll.find({"_id": {"$in": ids}}).to_list(len(ids))
            pre_backup = await _make_backup(
                current_user.id, kind="mongo_documents",
                source_name=f"{db}.{coll_name}",
                payload=_gzip_jsonl(current) if current else gzip.compress(b""),
                meta={
                    "format": "jsonl", "database": db, "collection": coll_name,
                    "primary_key": ["_id"], "document_count": len(current),
                },
                require_backup=True, warnings=warnings,
            )
            existing_ids = {d["_id"] for d in current}
            to_insert = [d for d in docs if d.get("_id") not in existing_ids]
            conflicts = [d for d in docs if d.get("_id") in existing_ids]
            if conflicts and not payload.overwrite_conflicts:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{len(conflicts)} of {len(docs)} document(s) already "
                        f"exist in '{db}.{coll_name}' (_id collision). Nothing "
                        "was restored — resubmit with overwrite_conflicts=true "
                        "to replace them."
                    ),
                )
            inserted = 0
            if to_insert:
                res = await coll.insert_many(to_insert, ordered=False)
                inserted = len(res.inserted_ids)
            replaced = 0
            if conflicts:
                from pymongo import ReplaceOne

                res = await coll.bulk_write(
                    [ReplaceOne({"_id": d["_id"]}, d) for d in conflicts],
                    ordered=False,
                )
                replaced = int(res.modified_count)
            results = {
                "inserted": inserted,
                "replaced": replaced,
                "conflicts_overwritten": replaced,
            }
        else:  # mongo_field_values
            fields = list(bmeta.get("fields") or [])
            differing = absent = restored = 0
            missing_docs = 0
            replaced_docs = 0
            from pymongo import ReplaceOne, UpdateOne

            current_ids = [d["_id"] for d in docs if "_id" in d]
            current = await coll.find({"_id": {"$in": current_ids}}).to_list(len(current_ids))
            pre_backup = await _make_backup(
                current_user.id, kind="mongo_field_values",
                source_name=f"{db}.{coll_name}",
                payload=_gzip_jsonl(current) if current else gzip.compress(b""),
                meta={
                    "format": "jsonl", "database": db, "collection": coll_name,
                    "fields": fields, "document_count": len(current),
                },
                require_backup=True, warnings=warnings,
            )
            current_by_id = {d["_id"]: d for d in current}
            updates = []
            reinsert = []
            for d in docs:
                _id = d.get("_id")
                cur = current_by_id.get(_id)
                if cur is None:
                    missing_docs += 1
                    if payload.overwrite_conflicts:
                        reinsert.append(d)
                    continue
                set_vals = {}
                for f in fields:
                    if f not in cur:
                        absent += 1
                        set_vals[f] = d.get(f)
                    elif cur.get(f) != d.get(f):
                        differing += 1
                        if payload.overwrite_conflicts:
                            set_vals[f] = d.get(f)
                if set_vals:
                    updates.append(UpdateOne({"_id": _id}, {"$set": set_vals}))
            blocked = (differing and not payload.overwrite_conflicts)
            if blocked:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{differing} field value(s) changed since the snapshot. "
                        "Nothing was restored — resubmit with "
                        "overwrite_conflicts=true to overwrite them."
                    ),
                )
            if updates:
                res = await coll.bulk_write(updates, ordered=False)
                restored = int(res.modified_count)
            if reinsert:
                res = await coll.insert_many(reinsert, ordered=False)
                replaced_docs = len(res.inserted_ids)
            results = {
                "documents_updated": restored,
                "fields_absent_restored": absent,
                "differing_overwritten": differing if payload.overwrite_conflicts else 0,
                "documents_reinserted": replaced_docs,
                "missing_documents_skipped": missing_docs - replaced_docs,
            }

        # While the client is still open: (a) re-read the collection to
        # verify the writes actually landed, (b) build a refreshed working
        # session from the LIVE data so the caller can adopt it immediately.
        verification = await _verify_mongo_restore(coll, kind, bmeta, docs)
        try:
            refreshed = await _mongo_session_from_live(
                current_user.id, client, db, coll_name
            )
        except Exception as exc:  # pragma: no cover - honest degradation
            logger.error("Post-restore refresh failed: %s", exc, exc_info=True)
            refreshed = None
            warnings.append(
                f"The restore was applied, but re-reading the collection "
                f"failed ({exc}); reconnect the collection to refresh the "
                "working copy."
            )
    except HTTPException:
        raise
    except Exception as exc:
        if _is_mongo_auth_error(exc):
            raise _mongo_auth_http_error()
        logger.error("mongo restore failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Restore failed; the collection was not fully restored "
                   "(see server logs and the pre-restore backup).",
        )
    finally:
        client.close()

    new_session_id: Optional[str] = None
    state: Dict[str, Any] = {"columns": [], "preview": [], "row_count": 0}
    if refreshed is not None:
        new_session_id = refreshed["session_id"]  # None when now empty
        state = {
            "columns": refreshed["columns"],
            "preview": refreshed["preview"],
            "row_count": refreshed["row_count"],
        }
        # The previous session mirrors pre-restore data — retire it so
        # nothing stale stays selectable.
        await _retire_session(current_user.id, payload.session_id)

    message = _honest_restore_message(
        verification, warnings,
        f"Restored {kind} backup into '{db}.{coll_name}'.",
    )
    logger.warning(
        "SOURCE RESTORE: %s of %s from backup %s by user %s (%s)",
        kind, f"{db}.{coll_name}", payload.backup_id, current_user.id, results,
    )
    await _record_history(
        current_user.id,
        session_id=new_session_id,
        operation="restore_source",
        source_type="mongodb", source_name=f"{db}.{coll_name}",
        source_meta={"database": db, "collection": coll_name},
        summary=f"Restored {kind} backup into '{db}.{coll_name}': {results}.",
        extra={"kind": kind, "results": results, "verification": verification},
        backup_id=(pre_backup or {}).get("backup_id"),
        restorable=bool((pre_backup or {}).get("backup_id")),
        rollback_expires_at=(pre_backup or {}).get("expires_at"),
        warnings=warnings,
    )
    return {
        "message": message,
        "kind": kind, "source": "mongodb", "database": db, "collection": coll_name,
        "results": results, "pre_restore_backup": pre_backup,
        "warnings": warnings, "restored_backup_id": payload.backup_id,
        # Refreshed working copy the UI adopts without reconnecting
        # (session_id is None when the collection is now empty or the
        # refresh failed — the warnings say so honestly).
        "session_id": new_session_id,
        "source_name": f"{db}.{coll_name}",
        "file_type": "mongodb",
        "columns": state["columns"],
        "preview": state["preview"],
        "row_count": state["row_count"],
        "verification": verification,
    }


# --------------------------------------------------------------------------- #
# Error handlers (avoid leaking internals)
# --------------------------------------------------------------------------- #
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred."})
