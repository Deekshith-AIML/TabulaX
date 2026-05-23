"""Safe MySQL connectivity helpers.

Provides:

* strict **identifier validation** for database/table/column names (so user
  input cannot inject SQL through an identifier),
* **parameterised** value queries,
* connection handling with timeouts and clear, credential-free errors,
* a recoverable **write-back** routine: the original table is *renamed* (not
  dropped) before the replacement is swapped in, and a backup is retained on
  failure so data is never silently destroyed.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from config import settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when the driver is present
    import mysql.connector
    from mysql.connector import Error as MySQLError
except Exception:  # pragma: no cover
    mysql = None  # type: ignore

    class MySQLError(Exception):  # type: ignore
        pass


class DatabaseError(Exception):
    """A database operation failed.  ``status_code`` guides the HTTP mapping."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# MySQL identifiers: letters, digits, underscore, dollar; must not start with a
# digit.  We additionally allow a single leading/trailing pair of backticks that
# the caller may have included.  Anything else is rejected outright.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,63}$")


def validate_identifier(name: str, kind: str = "identifier") -> str:
    """Return ``name`` if it is a safe MySQL identifier, else raise.

    This is the primary defence against SQL injection through database,
    table or column names.  We never interpolate unvalidated identifiers.
    """
    if not isinstance(name, str):
        raise DatabaseError(f"Invalid {kind}: must be a string.")
    stripped = name.strip().strip("`")
    if not stripped:
        raise DatabaseError(f"Invalid {kind}: empty.")
    if not _IDENT_RE.match(stripped):
        raise DatabaseError(
            f"Invalid {kind} '{name}'. Only letters, digits and underscores are "
            "allowed and it must not start with a digit."
        )
    return stripped


def quote_identifier(name: str, kind: str = "identifier") -> str:
    """Validate then backtick-quote an identifier."""
    return f"`{validate_identifier(name, kind)}`"


def _redact(msg: str) -> str:
    """Best-effort removal of credentials from driver error text."""
    return re.sub(r"(?i)(password|pwd|passwd)\s*[:=]\s*\S+", r"\1=***", msg)


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
# Loopback hostnames a user might type meaning "this machine".
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def map_docker_host(host: str) -> str:
    """Rewrite loopback to ``host.docker.internal`` when running in Docker.

    A user typing "localhost" in the UI means *their machine*, but inside a
    container localhost is the container itself.  Mapping keeps the common
    case working without the user knowing Docker networking.  Hosts outside
    Docker (and non-loopback hosts, e.g. ``db.example.com`` or ``mongo``)
    are returned unchanged.
    """
    h = (host or "").strip()
    if settings.in_docker and h.lower() in _LOCAL_HOSTS:
        logger.info(
            "Docker detected: MySQL host %r mapped to 'host.docker.internal'.", host
        )
        return "host.docker.internal"
    return h


def connect(
    host: str,
    user: str,
    password: str,
    port: int = 3306,
    database: Optional[str] = None,
):
    """Open a MySQL connection with timeouts and mapped, credential-free errors.

    Raises :class:`DatabaseError` with a safe message on failure.
    """
    if mysql is None:  # pragma: no cover
        raise DatabaseError("MySQL driver is not available on this server.", 503)

    host = (host or "").strip()
    if not host:
        raise DatabaseError("Database host is required.")
    host = map_docker_host(host)
    if not (1 <= int(port) <= 65535):
        raise DatabaseError("Database port must be between 1 and 65535.")

    db_name = None
    if database:
        db_name = validate_identifier(database, "database name")

    try:
        conn = mysql.connector.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=db_name or "",
            connection_timeout=10,
            auth_plugin="mysql_native_password",
        )
        if not conn.is_connected():
            raise DatabaseError("Could not establish an active database connection.", 503)
        return conn
    except MySQLError as exc:
        msg = str(exc)
        logger.error("MySQL connect error for %s:%s - %s", host, port, _redact(msg))
        if "Access denied" in msg:
            raise DatabaseError("Access denied. Check your username and password.", 401)
        if "Unknown database" in msg:
            raise DatabaseError(f"Unknown database '{db_name}'.", 400)
        if "Can't connect" in msg or "Connection refused" in msg or "timed out" in msg:
            if host == "host.docker.internal":
                detail = (
                    f"Cannot connect to MySQL server at {host}:{port}. "
                    "'localhost' was mapped to your host machine (Docker): "
                    "check MySQL is running there and the port is open."
                )
            elif host.lower() in _LOCAL_HOSTS:
                detail = (
                    f"Cannot connect to MySQL server at {host}:{port}. "
                    "Is the server running on this machine?"
                )
            else:
                detail = f"Cannot connect to MySQL server at {host}:{port}."
            raise DatabaseError(detail, 503)
        raise DatabaseError(f"Database connection error: {_redact(msg)}", 400)


@contextmanager
def connection(**kwargs):
    """Context manager yielding a connected MySQL connection."""
    conn = connect(**kwargs)
    try:
        yield conn
    finally:
        try:
            if conn.is_connected():
                conn.close()
        except Exception:  # pragma: no cover
            pass


# --------------------------------------------------------------------------- #
# Read helpers
# --------------------------------------------------------------------------- #
def list_databases(conn) -> List[str]:
    """List non-system databases."""
    cursor = conn.cursor()
    try:
        cursor.execute("SHOW DATABASES")
        system = {"information_schema", "mysql", "performance_schema", "sys", "tmp"}
        return [r[0] for r in cursor.fetchall() if r[0].lower() not in system]
    finally:
        cursor.close()


def list_tables(conn, database: str) -> List[str]:
    """List tables in ``database`` (validated identifier)."""
    db = validate_identifier(database, "database name")
    cursor = conn.cursor()
    try:
        cursor.execute(f"SHOW TABLES FROM {quote_identifier(db)}")
        return [r[0] for r in cursor.fetchall()]
    finally:
        cursor.close()


def describe_table(conn, database: str, table: str) -> List[str]:
    """Return column names for ``database.table``."""
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(f"DESCRIBE {db}.{tb}")
        rows = cursor.fetchall()
        return [r["Field"] for r in rows]
    finally:
        cursor.close()


def fetch_preview(conn, database: str, table: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch up to ``limit`` rows using a parameterised LIMIT."""
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    limit = max(1, min(int(limit), 1000))
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(f"SELECT * FROM {db}.{tb} LIMIT %s", (limit,))
        return list(cursor.fetchall())
    finally:
        cursor.close()


def fetch_all(
    conn, database: str, table: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetch rows from a validated table name (all rows, or up to ``limit``)."""
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    cursor = conn.cursor(dictionary=True)
    try:
        if limit is None:
            cursor.execute(f"SELECT * FROM {db}.{tb}")
        else:
            cursor.execute(f"SELECT * FROM {db}.{tb} LIMIT %s", (int(limit),))
        return list(cursor.fetchall())
    finally:
        cursor.close()


def count_rows(conn, database: str, table: str) -> int:
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {db}.{tb}")
        row = cursor.fetchone()
        return int(row[0]) if row else 0
    finally:
        cursor.close()


# --------------------------------------------------------------------------- #
# Stable row identity + permanent deletion helpers
# --------------------------------------------------------------------------- #
def get_primary_key(conn, database: str, table: str) -> List[str]:
    """Return the ordered PRIMARY KEY column names (empty list when none)."""
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(f"SHOW KEYS FROM {db}.{tb} WHERE Key_name = 'PRIMARY'")
        rows = cursor.fetchall()
        return [
            r["Column_name"]
            for r in sorted(rows, key=lambda r: int(r.get("Seq_in_index") or 0))
        ]
    finally:
        cursor.close()


def describe_table_full(conn, database: str, table: str) -> List[Dict[str, Any]]:
    """Full DESCRIBE rows (Field/Type/Null/Key/Default/Extra) for ``db.table``."""
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(f"DESCRIBE {db}.{tb}")
        return list(cursor.fetchall())
    finally:
        cursor.close()


def _pk_where_sql(pk_cols: Sequence[str], n_keys: int) -> str:
    """``(a = %s AND b = %s) OR (...)`` for ``n_keys`` composite key tuples."""
    cols = [quote_identifier(c, "column name") for c in pk_cols]
    single = "(" + " AND ".join(f"{c} = %s" for c in cols) + ")"
    return "(" + " OR ".join([single] * max(1, int(n_keys))) + ")"


def _flatten_keys(keys: Sequence[Sequence[Any]]) -> List[Any]:
    return [v for key in keys for v in key]


def select_by_pk(
    conn,
    database: str,
    table: str,
    pk_cols: Sequence[str],
    keys: Sequence[Sequence[Any]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch full rows matching ``keys`` (composite PK tuples) — parameterised."""
    if not keys:
        return []
    if len(pk_cols) == 0:  # pragma: no cover - guarded by callers
        raise DatabaseError("Table has no primary key.")
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    where = _pk_where_sql(pk_cols, len(keys))
    params = _flatten_keys(keys)
    sql = f"SELECT * FROM {db}.{tb} WHERE {where}"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(sql, params)
        return list(cursor.fetchall())
    finally:
        cursor.close()


def count_by_pk(
    conn,
    database: str,
    table: str,
    pk_cols: Sequence[str],
    keys: Sequence[Sequence[Any]],
) -> int:
    """Count how many of ``keys`` currently exist in the source table."""
    if not keys:
        return 0
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    where = _pk_where_sql(pk_cols, len(keys))
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"SELECT COUNT(*) FROM {db}.{tb} WHERE {where}", _flatten_keys(keys)
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0
    finally:
        cursor.close()


def delete_by_pk(
    conn,
    database: str,
    table: str,
    pk_cols: Sequence[str],
    keys: Sequence[Sequence[Any]],
) -> int:
    """Delete rows matching ``keys`` by primary key; returns affected count.

    * values are fully parameterised (never interpolated);
    * identifiers go through ``quote_identifier``;
    * executed in chunks and committed once — a failure rolls back where the
      table engine supports transactions.
    """
    if not keys:
        return 0
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    cursor = conn.cursor()
    total = 0
    chunk_size = 500
    try:
        for start in range(0, len(keys), chunk_size):
            part = list(keys[start : start + chunk_size])
            where = _pk_where_sql(pk_cols, len(part))
            cursor.execute(
                f"DELETE FROM {db}.{tb} WHERE {where}", _flatten_keys(part)
            )
            total += int(cursor.rowcount or 0)
        conn.commit()
        logger.info(
            "delete_by_pk: db=%s table=%s keys=%d affected=%d",
            database, table, len(keys), total,
        )
        return total
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:  # pragma: no cover
            pass
        raise DatabaseError(
            f"Row deletion failed: {_redact(str(exc))}. The transaction was "
            "rolled back where supported; on tables without transaction "
            "support some rows may already have been removed.",
            500,
        )
    finally:
        cursor.close()


def select_columns(
    conn,
    database: str,
    table: str,
    columns: Sequence[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch the given (validated) columns for backup purposes."""
    if not columns:
        return []
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    cols_sql = ", ".join(quote_identifier(c, "column name") for c in columns)
    sql = f"SELECT {cols_sql} FROM {db}.{tb}"
    params: List[Any] = []
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(sql, params)
        return list(cursor.fetchall())
    finally:
        cursor.close()


def column_drop_risks(
    conn, database: str, table: str, column: str
) -> Dict[str, Any]:
    """Inspect schema dependencies that make ``DROP COLUMN`` unsafe.

    Returns::

        {
          "foreign_keys_out": [{"constraint", "references"}],   # this col in an FK
          "foreign_keys_in":  [{"constraint", "table"}],        # others FK to this col
          "indexes":          [{"name", "primary", "unique"}],  # indexes using this col
          "is_primary":       bool,
          "is_generated":     bool,
          "referenced_by_generated": [column names],
        }

    Callers must refuse hard risks (FK involvement, primary-key membership,
    generated-column dependencies) instead of silently proceeding.
    """
    db = validate_identifier(database, "database name")
    tb = validate_identifier(table, "table name")
    col = validate_identifier(column, "column name")
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT CONSTRAINT_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME "
            "FROM information_schema.KEY_COLUMN_USAGE "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s "
            "AND REFERENCED_TABLE_NAME IS NOT NULL",
            (db, tb, col),
        )
        foreign_keys_out = [
            {
                "constraint": r["CONSTRAINT_NAME"],
                "references": f"{r['REFERENCED_TABLE_NAME']}.{r['REFERENCED_COLUMN_NAME']}",
            }
            for r in cursor.fetchall()
        ]

        cursor.execute(
            "SELECT CONSTRAINT_NAME, TABLE_NAME "
            "FROM information_schema.KEY_COLUMN_USAGE "
            "WHERE REFERENCED_TABLE_SCHEMA = %s AND REFERENCED_TABLE_NAME = %s "
            "AND REFERENCED_COLUMN_NAME = %s",
            (db, tb, col),
        )
        foreign_keys_in = [
            {"constraint": r["CONSTRAINT_NAME"], "table": r["TABLE_NAME"]}
            for r in cursor.fetchall()
        ]

        cursor.execute(
            "SELECT INDEX_NAME, NON_UNIQUE "
            "FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s",
            (db, tb, col),
        )
        seen: Dict[str, Dict[str, Any]] = {}
        for r in cursor.fetchall():
            name = r["INDEX_NAME"]
            seen[name] = {
                "name": name,
                "primary": name == "PRIMARY",
                "unique": not bool(r["NON_UNIQUE"]),
            }
        indexes = list(seen.values())

        cursor.execute(
            "SELECT COLUMN_NAME, EXTRA, GENERATION_EXPRESSION "
            "FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s",
            (db, tb, col),
        )
        col_row = cursor.fetchone() or {}
        extra = str(col_row.get("EXTRA") or "")
        is_generated = "GENERATED" in extra.upper()

        cursor.execute(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s "
            "AND GENERATION_EXPRESSION LIKE %s AND COLUMN_NAME <> %s",
            (db, tb, f"%`{col}`%", col),
        )
        referenced_by_generated = [r["COLUMN_NAME"] for r in cursor.fetchall()]

        return {
            "foreign_keys_out": foreign_keys_out,
            "foreign_keys_in": foreign_keys_in,
            "indexes": indexes,
            "is_primary": any(i["primary"] for i in indexes),
            "is_generated": is_generated,
            "referenced_by_generated": referenced_by_generated,
        }
    finally:
        cursor.close()


def drop_column(conn, database: str, table: str, column: str) -> None:
    """Execute ``ALTER TABLE ... DROP COLUMN`` (identifiers validated).

    MySQL DDL auto-commits: once this returns successfully the change cannot
    be rolled back by a transaction — callers must take a backup first.
    """
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    col = quote_identifier(column, "column name")
    cursor = conn.cursor()
    try:
        cursor.execute(f"ALTER TABLE {db}.{tb} DROP COLUMN {col}")
        conn.commit()
        logger.info("drop_column: %s.%s.%s", database, table, column)
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:  # pragma: no cover
            pass
        raise DatabaseError(
            f"Dropping column failed: {_redact(str(exc))}", 400
        )
    finally:
        cursor.close()


# --------------------------------------------------------------------------- #
# Safe write-back
# --------------------------------------------------------------------------- #
def write_back_transformed(
    conn,
    database: str,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Atomically replace ``database.table`` with ``rows``.

    Safety properties:

    * every identifier is validated before interpolation;
    * all **values** are inserted via a parameterised ``executemany``;
    * the original table is **renamed** to a backup (not dropped), the new
      table is created from the original's schema (``CREATE TABLE ... LIKE``),
      populated, then swapped in.  On any error the backup is renamed back, so
      the original data survives a failure.

    Returns a summary dict (no credentials, no row contents).
    """
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    col_sql = ", ".join(quote_identifier(c, "column name") for c in columns)
    col_list = [quote_identifier(c, "column name") for c in columns]
    placeholders = ", ".join(["%s"] * len(col_list))
    insert_sql = f"INSERT INTO {db}.`__tabulax_new` ({col_sql}) VALUES ({placeholders})"

    rows = [tuple(r.get(c) for c in columns) for r in rows]
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    backup = f"__tabulax_bak_{stamp}"
    tmp = "__tabulax_new"

    cursor = conn.cursor()
    started = False
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {db}.`{tmp}`")
        cursor.execute(f"DROP TABLE IF EXISTS {db}.`{backup}`")
        cursor.execute(f"CREATE TABLE {db}.`{tmp}` LIKE {db}.{tb}")
        if rows:
            cursor.executemany(insert_sql, rows)

        # Swap: original -> backup, new -> original, drop backup.
        cursor.execute(f"RENAME TABLE {db}.{tb} TO {db}.`{backup}`")
        started = True
        cursor.execute(f"RENAME TABLE {db}.`{tmp}` TO {db}.{tb}")
        cursor.execute(f"DROP TABLE IF EXISTS {db}.`{backup}`")
        conn.commit()
        logger.info(
            "write_back complete: db=%s table=%s rows=%d", database, table, len(rows)
        )
        return {"table": table, "rows_written": len(rows)}
    except Exception as exc:
        # Roll back to a consistent state: restore the original if we renamed it.
        logger.error("write_back failed for %s.%s: %s", database, table, _redact(str(exc)), exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
        if started:
            try:
                # Restore original name if the swap was interrupted.
                cursor.execute(f"DROP TABLE IF EXISTS {db}.{tb}")
                cursor.execute(f"RENAME TABLE {db}.`{backup}` TO {db}.{tb}")
                conn.commit()
                logger.info("Restored original table %s.%s after failure", database, table)
            except Exception as restore_exc:  # pragma: no cover
                logger.error(
                    "CRITICAL: could not restore %s.%s from backup %s: %s",
                    database, table, backup, restore_exc,
                )
                raise DatabaseError(
                    f"Write failed and automatic restore also failed. The original "
                    f"data is preserved in table `{backup}` in database `{database}`.",
                    500,
                )
        raise DatabaseError(f"Database update failed: {_redact(str(exc))}", 500)
    finally:
        try:
            # Clean up any dangling temp table on success paths.
            if not started:
                cursor.execute(f"DROP TABLE IF EXISTS {db}.`{tmp}`")
                conn.commit()
        except Exception:
            pass
        cursor.close()


# --------------------------------------------------------------------------- #
# Source restoration helpers
# --------------------------------------------------------------------------- #
# Only these type tokens may appear in a restored column definition.  The
# values come from our own DESCRIBE snapshot, but they are re-validated so a
# tampered backup can never inject SQL.
_COL_TYPE_RE = re.compile(
    r"^(?:(?:var)?char|tinyint|smallint|mediumint|int|integer|bigint|decimal"
    r"|float|double|real|bit|datetime|timestamp|date|time|year|binary|varbinary"
    r"|tinyblob|blob|mediumblob|longblob|tinytext|text|mediumtext|longtext"
    r"|enum|set)(?:\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\))?"
    r"(?:\s+(?:unsigned|zerofill))*(?:\s+collate\s+\w+)?$",
    re.IGNORECASE,
)


def validate_column_type(col_type: str) -> str:
    """Validate a DESCRIBE ``Type`` string before using it in DDL."""
    cleaned = " ".join(str(col_type or "").split())
    if not cleaned or not _COL_TYPE_RE.match(cleaned):
        raise DatabaseError(f"Unsupported or unsafe column type: {col_type!r}", 400)
    return cleaned


def column_exists(conn, database: str, table: str, column: str) -> bool:
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    cursor = conn.cursor()
    try:
        cursor.execute(f"DESCRIBE {db}.{tb}")
        return any(row[0] == column for row in cursor.fetchall())
    finally:
        cursor.close()


def add_column(
    conn, database: str, table: str, column: str, col_type: str, nullable: bool = True
) -> None:
    """``ALTER TABLE ... ADD COLUMN`` with a validated type (DDL auto-commits)."""
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    col = quote_identifier(column, "column name")
    safe_type = validate_column_type(col_type)
    null_sql = "NULL" if nullable else "NOT NULL"
    cursor = conn.cursor()
    try:
        cursor.execute(f"ALTER TABLE {db}.{tb} ADD COLUMN {col} {safe_type} {null_sql}")
        conn.commit()
        logger.info("add_column: %s.%s.%s %s", database, table, column, safe_type)
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:  # pragma: no cover
            pass
        raise DatabaseError(f"Adding column failed: {_redact(str(exc))}", 400)
    finally:
        cursor.close()


def insert_rows(
    conn, database: str, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]
) -> int:
    """Parameterised multi-row INSERT for restored rows.

    Only ``columns`` that already exist on the table are written; callers pass
    the intersection so a schema drift cannot abort the whole restore.
    """
    if not columns:
        return 0
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    col_sql = ", ".join(quote_identifier(c, "column name") for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO {db}.{tb} ({col_sql}) VALUES ({placeholders})"
    materialised = [tuple(r) for r in rows]
    cursor = conn.cursor()
    try:
        cursor.executemany(sql, materialised)
        conn.commit()
        logger.info("insert_rows: %s.%s rows=%d", database, table, len(materialised))
        return len(materialised)
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:  # pragma: no cover
            pass
        raise DatabaseError(f"Restoring rows failed: {_redact(str(exc))}", 400)
    finally:
        cursor.close()


def update_column_by_pk(
    conn,
    database: str,
    table: str,
    column: str,
    pk: Sequence[str],
    rows: Iterable[Dict[str, Any]],
) -> Dict[str, int]:
    """Update ``column`` from backup rows keyed by the primary key.

    Returns ``{"matched": n, "updated": n, "missing": m}`` where ``missing``
    counts backup keys that no longer exist in the table.
    """
    db = quote_identifier(database, "database name")
    tb = quote_identifier(table, "table name")
    col = quote_identifier(column, "column name")
    pk_sql = " AND ".join(f"{quote_identifier(k, 'column name')}=%s" for k in pk)
    sql = f"UPDATE {db}.{tb} SET {col}=%s WHERE {pk_sql}"

    matched = 0
    updated = 0
    missing = 0
    cursor = conn.cursor()
    try:
        for row in rows:
            key = tuple(row.get(k) for k in pk)
            if any(v is None for v in key):
                missing += 1
                continue
            cursor.execute(sql, (row.get(column), *key))
            if cursor.rowcount:
                matched += 1
                updated += 1
            else:
                missing += 1
        conn.commit()
        logger.info(
            "update_column_by_pk: %s.%s.%s matched=%d missing=%d",
            database, table, column, matched, missing,
        )
        return {"matched": matched, "updated": updated, "missing": missing}
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:  # pragma: no cover
            pass
        raise DatabaseError(
            f"Restoring column values failed: {_redact(str(exc))}", 400
        )
    finally:
        cursor.close()
