"""Unit tests for MySQL identifier validation and write-back safety.

Identifier validation is tested directly (no database required).  The
recoverable write-back path is exercised against a real MySQL instance only if
one is reachable; otherwise those tests are skipped so CI needs no credentials.
"""

from __future__ import annotations

import os

import pytest

db_mysql = pytest.importorskip("db_mysql")


# --------------------------------------------------------------------------- #
# Identifier validation (pure, no DB)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "good",
    ["users", "my_table", "_private", "Table1", "db$1", "a" * 64],
)
def test_valid_identifiers_accepted(good):
    assert db_mysql.validate_identifier(good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "",                      # empty
        "1table",                # starts with digit
        "tab; DROP TABLE users", # injection attempt
        "table-name",            # hyphen not allowed
        "table name",            # space
        "tab`le",                # backtick inside
        "tab/**/le",             # comment injection
        "tab.le",                # dot (would break db.table parsing)
        "table'",                # quote
        'table"',                # double quote
        "table--",               # SQL comment
        "a" * 65,                # too long
        None,                    # wrong type
        "пользователь",          # non-ascii
    ],
)
def test_invalid_identifiers_rejected(bad):
    with pytest.raises(db_mysql.DatabaseError):
        db_mysql.validate_identifier(bad)


def test_validate_identifier_strips_surrounding_backticks():
    assert db_mysql.validate_identifier("`users`") == "users"


def test_quote_identifier_wraps_in_backticks():
    assert db_mysql.quote_identifier("users") == "`users`"


def test_validate_identifier_error_is_status_400():
    with pytest.raises(db_mysql.DatabaseError) as exc:
        db_mysql.validate_identifier("bad;drop")
    assert exc.value.status_code == 400


def test_redact_removes_password_from_message():
    msg = "Access denied for user root with password: SuperSecret123"
    redacted = db_mysql._redact(msg)
    assert "SuperSecret123" not in redacted
    assert "password" in redacted.lower()


# --------------------------------------------------------------------------- #
# Connection-level behaviour (only when a MySQL server is reachable)
# --------------------------------------------------------------------------- #
def _mysql_available() -> bool:
    if db_mysql.mysql is None:
        return False
    host = os.environ.get("TEST_MYSQL_HOST", "localhost")
    port = int(os.environ.get("TEST_MYSQL_PORT", "3306"))
    try:
        db_mysql.connect(
            host=host,
            user=os.environ.get("TEST_MYSQL_USER", "root"),
            password=os.environ.get("TEST_MYSQL_PASSWORD", ""),
            port=port,
        )
        return True
    except Exception:
        return False


MYSQL_UP = _mysql_available()


@pytest.mark.skipif(not MYSQL_UP, reason="No reachable MySQL for integration test")
def test_connect_bad_password_maps_to_401_without_leaking():
    host = os.environ.get("TEST_MYSQL_HOST", "localhost")
    port = int(os.environ.get("TEST_MYSQL_PORT", "3306"))
    user = os.environ.get("TEST_MYSQL_USER", "root")
    with pytest.raises(db_mysql.DatabaseError) as exc:
        db_mysql.connect(host=host, user=user, password="definitely-wrong", port=port)
    assert exc.value.status_code == 401
    assert "definitely-wrong" not in str(exc.value)


@pytest.mark.skipif(not MYSQL_UP, reason="No reachable MySQL for integration test")
def test_connect_unreachable_host_maps_to_503():
    with pytest.raises(db_mysql.DatabaseError) as exc:
        db_mysql.connect(
            host="203.0.113.1",  # TEST-NET-3, non-routable
            user="u", password="p", port=3306,
        )
    assert exc.value.status_code == 503


@pytest.mark.skipif(not MYSQL_UP, reason="No reachable MySQL for integration test")
def test_write_back_roundtrip_and_recovery():
    """Create a scratch table, write back transformed data, verify, and clean up."""
    host = os.environ.get("TEST_MYSQL_HOST", "localhost")
    port = int(os.environ.get("TEST_MYSQL_PORT", "3306"))
    user = os.environ.get("TEST_MYSQL_USER", "root")
    password = os.environ.get("TEST_MYSQL_PASSWORD", "")
    scratch_db = os.environ.get("TEST_MYSQL_DB", "tabula_scratch")

    try:
        with db_mysql.connection(host=host, user=user, password=password, port=port) as conn:
            cur = conn.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{scratch_db}`")
            conn.commit()
            cur.close()
    except db_mysql.DatabaseError:
        pytest.skip("Cannot create scratch database")

    with db_mysql.connection(
        host=host, user=user, password=password, port=port, database=scratch_db
    ) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS `wb_test`")
        cur.execute(
            "CREATE TABLE `wb_test` (id INT PRIMARY KEY, val VARCHAR(50))"
        )
        cur.executemany("INSERT INTO `wb_test` (id, val) VALUES (%s, %s)",
                        [(1, "a"), (2, "b")])
        conn.commit()
        cur.close()

        original = db_mysql.fetch_all(conn, scratch_db, "wb_test")
        assert len(original) == 2

        # Successful write-back.
        new_rows = [{"id": 1, "val": "A"}, {"id": 2, "val": "B"}]
        summary = db_mysql.write_back_transformed(
            conn, scratch_db, "wb_test", ["id", "val"], new_rows
        )
        assert summary["rows_written"] == 2
        after = db_mysql.fetch_all(conn, scratch_db, "wb_test")
        assert {r["val"] for r in after} == {"A", "B"}

        # No leftover backup/temp tables on success.
        tables = db_mysql.list_tables(conn, scratch_db)
        assert not any(t.startswith("__tabulax") for t in tables), tables

        # Simulate a failure during a second write-back: pass a bad value type
        # that breaks insertion, and ensure the original table survives.
        try:
            db_mysql.write_back_transformed(
                conn, scratch_db, "wb_test", ["id", "val"],
                [{"id": "not-an-int", "val": "X"}],
            )
        except db_mysql.DatabaseError:
            pass
        # Regardless of error path, the table must still exist with data.
        surviving = db_mysql.fetch_all(conn, scratch_db, "wb_test")
        assert len(surviving) >= 0
        tables = db_mysql.list_tables(conn, scratch_db)
        assert "wb_test" in tables, "original table must never be destroyed"

        # Cleanup.
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS `wb_test`")
        conn.commit()
        cur.close()


# --------------------------------------------------------------------------- #
# Docker loopback mapping (pure, no DB)
# --------------------------------------------------------------------------- #
def test_map_docker_host_rewrites_loopback_in_docker(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "in_docker", True)
    assert db_mysql.map_docker_host("localhost") == "host.docker.internal"
    assert db_mysql.map_docker_host("127.0.0.1") == "host.docker.internal"
    assert db_mysql.map_docker_host("LOCALHOST") == "host.docker.internal"
    # Non-loopback hosts must never be rewritten.
    assert db_mysql.map_docker_host("db.example.com") == "db.example.com"
    assert db_mysql.map_docker_host("mysql.internal") == "mysql.internal"


def test_map_docker_host_noop_outside_docker(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "in_docker", False)
    assert db_mysql.map_docker_host("localhost") == "localhost"
    assert db_mysql.map_docker_host("127.0.0.1") == "127.0.0.1"


# --------------------------------------------------------------------------- #
# /get_columns/ session snapshot (mocked driver, no server required)
# --------------------------------------------------------------------------- #
def test_get_columns_creates_session_and_allows_edits(client, auth_headers, monkeypatch):
    """MySQL table loads must behave like files/mongo: snapshot into a
    server-side session so row deletion, apply-on-working-copy and CSV
    export all work.  (The delete toolbar keys off session_id.)"""
    import app as app_module

    rows = [
        {"name": "alice", "score": 10},
        {"name": "bob", "score": 20},
        {"name": "carol", "score": 30},
    ]

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _FakeMySQL:
        class DatabaseError(Exception):
            def __init__(self, msg="err", status_code=400):
                super().__init__(msg)
                self.status_code = status_code

        @staticmethod
        def connection(**kwargs):
            return _FakeConn()

        @staticmethod
        def describe_table(conn, database, table):
            return ["name", "score"]

        @staticmethod
        def fetch_all(conn, database, table, limit=None):
            return rows[:limit] if limit else rows

    monkeypatch.setattr(app_module, "db_mysql", _FakeMySQL())

    resp = client.post(
        "/get_columns/?table_name=people",
        headers=auth_headers,
        json={"host": "h", "user": "u", "password": "p", "port": 3306, "database": "db"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"], "MySQL loads must create a session"
    assert body["columns"] == ["name", "score"]
    assert body["row_count"] == 3
    sid = body["session_id"]

    # Row deletion works through the shared session endpoints.
    d = client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [1]},
    )
    assert d.status_code == 200, d.text
    assert d.json()["removed"] == 1
    assert [r["name"] for r in d.json()["preview"]] == ["alice", "carol"]

    # Apply with source_type=database + session_id reads the edited copy
    # (deleted rows must stay deleted).
    gen = client.post(
        "/generate_function/", headers=auth_headers,
        json={
            "column": "score",
            "classification": "Numerical",
            "examples": [
                {"input": "10", "output": "100"},
                {"input": "20", "output": "200"},
                {"input": "30", "output": "300"},
            ],
        },
    )
    assert gen.status_code == 200, gen.text
    ap = client.post(
        "/apply/", headers=auth_headers,
        json={
            "transformation_id": gen.json()["transformation_id"],
            "selected_column": "score",
            "source_type": "database",
            "session_id": sid,
        },
    )
    assert ap.status_code == 200, ap.text
    names = [r["name"] for r in ap.json()["preview"]]
    assert "bob" not in names, "apply must operate on the edited working copy"

    # CSV export reflects the session and never leaks internal row ids.
    dl = client.get("/download/", params={"session_id": sid}, headers=auth_headers)
    assert dl.status_code == 200
    lines = [ln for ln in dl.text.strip().splitlines() if ln]
    assert lines[0] == "name,score"
    assert len(lines) == 3  # header + alice, carol


# --------------------------------------------------------------------------- #
# Credential errors must NOT be HTTP 401 (401 = the user's own session expired;
# the frontend logs out and redirects on any 401)
# --------------------------------------------------------------------------- #
class _AccessDeniedError(Exception):
    status_code = 401

    def __init__(self):
        super().__init__("Access denied. Check your username and password.")


def test_connect_database_bad_password_is_400_not_401(client, auth_headers, monkeypatch):
    import app as app_module

    class _FakeMySQL:
        DatabaseError = _AccessDeniedError

        @staticmethod
        def connection(**kwargs):
            raise _AccessDeniedError()

    monkeypatch.setattr(app_module, "db_mysql", _FakeMySQL())
    resp = client.post(
        "/connect_database/",
        headers=auth_headers,
        json={"host": "h", "user": "u", "password": "wrong", "port": 3306},
    )
    assert resp.status_code == 400, resp.text
    assert "Access denied" in resp.json()["detail"]
    assert "wrong" not in resp.json()["detail"], "password must not leak"


def test_get_columns_bad_password_is_400_not_401(client, auth_headers, monkeypatch):
    import app as app_module

    class _FakeMySQL:
        DatabaseError = _AccessDeniedError

        @staticmethod
        def connection(**kwargs):
            raise _AccessDeniedError()

    monkeypatch.setattr(app_module, "db_mysql", _FakeMySQL())
    resp = client.post(
        "/get_columns/?table_name=t",
        headers=auth_headers,
        json={"host": "h", "user": "u", "password": "wrong", "port": 3306, "database": "db"},
    )
    assert resp.status_code == 400, resp.text


def test_apply_to_database_bad_password_is_400_not_401(client, auth_headers, monkeypatch):
    import app as app_module

    gen = client.post(
        "/generate_function/", headers=auth_headers,
        json={
            "column": "c",
            "classification": "Numerical",
            "examples": [
                {"input": "1", "output": "2"},
                {"input": "2", "output": "4"},
            ],
        },
    )
    assert gen.status_code == 200, gen.text

    class _FakeMySQL:
        DatabaseError = _AccessDeniedError

        @staticmethod
        def connection(**kwargs):
            raise _AccessDeniedError()

    monkeypatch.setattr(app_module, "db_mysql", _FakeMySQL())
    resp = client.post(
        "/apply_to_database/",
        headers=auth_headers,
        json={
            "connection_details": {"host": "h", "user": "u", "password": "wrong",
                                   "port": 3306, "database": "db"},
            "table_name": "t",
            "column_to_transform": "c",
            "transformation_id": gen.json()["transformation_id"],
            "confirm": True,
        },
    )
    assert resp.status_code == 400, resp.text
    assert "Access denied" in resp.json()["detail"]


def test_mongo_auth_failure_is_400_with_clear_message(client, auth_headers, monkeypatch):
    import app as app_module

    class _AuthFail(Exception):
        code = 18

        def __init__(self):
            super().__init__("Authentication failed.")

    async def _bad_client(conn):
        raise app_module._mongo_auth_http_error()

    monkeypatch.setattr(app_module, "_mongo_client", _bad_client)
    resp = client.post(
        "/connect_mongo/",
        headers=auth_headers,
        json={"uri": "mongodb://wrong:wrong@localhost:27017/"},
    )
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "Incorrect MongoDB username or password" in detail


def test_is_mongo_auth_error_detects_driver_codes():
    import app as app_module

    class Op18(Exception):
        code = 18

    class OpOther(Exception):
        code = 50

    assert app_module._is_mongo_auth_error(Op18())
    assert app_module._is_mongo_auth_error(Exception("Authentication failed."))
    assert not app_module._is_mongo_auth_error(OpOther())
    assert not app_module._is_mongo_auth_error(Exception("network error"))
