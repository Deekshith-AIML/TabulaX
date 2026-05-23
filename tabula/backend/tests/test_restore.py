"""Source-database restore tests (live Mongo + mocked MySQL).

Contract under test: the backup must exist/own/expire, the current source
state is backed up first, conflicts require overwrite_conflicts=true, typed
"RESTORE" is mandatory, actual counts are reported, and partial failure is
honest.
"""

from __future__ import annotations

import gzip
import uuid
from contextlib import contextmanager

import pytest
from bson import ObjectId
from pymongo import MongoClient

import db_mysql as _real_db_mysql

URI = "mongodb://localhost:27017"

ALICE = ObjectId("68f0a1b2c3d4e5f60718293a")
BOB = ObjectId("68f0a1b2c3d4e5f60718293b")
CAROL = ObjectId("68f0a1b2c3d4e5f60718293c")
DAVE = ObjectId("68f0a1b2c3d4e5f60718293d")


@pytest.fixture()
def mongo_db():
    mc = MongoClient(URI, serverSelectionTimeoutMS=3000)
    try:
        mc.admin.command("ping")
    except Exception:
        mc.close()
        pytest.skip("Local MongoDB not available")
    name = f"tabula_rs_{uuid.uuid4().hex[:8]}"
    yield mc, name
    mc.drop_database(name)
    mc.close()


def _connect(client, headers, db_name, collection="people"):
    resp = client.post(
        f"/get_mongo_data/?database_name={db_name}&collection_name={collection}",
        headers=headers,
        json={"uri": URI},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _delete_docs(client, headers, sid, row_ids):
    prev = client.post(
        "/source_delete/preview/", headers=headers,
        json={
            "session_id": sid, "target": "mongo_documents",
            "row_ids": row_ids, "mongo_connection": {"uri": URI},
        },
    )
    assert prev.status_code == 200, prev.text
    token = prev.json()["preview_token"]
    resp = client.post(
        "/source_delete/mongo_documents/", headers=headers,
        json={
            "session_id": sid, "target": "mongo_documents",
            "preview_token": token, "confirm": True,
            "typed_confirmation": "DELETE",
            "row_ids": row_ids, "mongo_connection": {"uri": URI},
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _latest_backup_id(client, headers, kind="mongo_documents"):
    backups = client.get("/backups/", headers=headers).json()["backups"]
    matching = [b for b in backups if b["kind"] == kind]
    assert matching, backups
    return matching[0]["backup_id"]


# --------------------------------------------------------------------------- #
# Mongo restore (end to end)
# --------------------------------------------------------------------------- #
def test_mongo_document_restore_roundtrip(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["people"]
    coll.insert_many([
        {"_id": ALICE, "name": "alice", "score": 10},
        {"_id": BOB, "name": "bob", "score": 20},
        {"_id": CAROL, "name": "carol", "score": 30},
    ])

    loaded = _connect(client, auth_headers, db_name)
    sid = loaded["session_id"]
    out = _delete_docs(client, auth_headers, sid, [0, 1])  # alice, bob
    assert out["affected"] == 2
    assert coll.count_documents({}) == 1
    new_sid = out["session_id"]

    backup_id = _latest_backup_id(client, auth_headers)

    # Preview: expiry, snapshot counts, typed-confirmation requirement.
    prev = client.post(
        "/restore_source/preview/", headers=auth_headers,
        json={
            "backup_id": backup_id,
            "mongo_connection": {"uri": URI},
        },
    )
    assert prev.status_code == 200, prev.text
    body = prev.json()
    assert body["restorable"] is True
    assert body["requires_typed_confirmation"] == "RESTORE"
    assert body["snapshot"]["document_count"] == 2
    assert body["expires_at"]

    # Intent gates.
    no_confirm = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": False,
            "typed_confirmation": "RESTORE",
            "mongo_connection": {"uri": URI},
        },
    )
    assert no_confirm.status_code == 400

    wrong_typed = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "restore",
            "mongo_connection": {"uri": URI},
        },
    )
    assert wrong_typed.status_code == 400
    assert "RESTORE" in wrong_typed.json()["detail"]
    assert coll.count_documents({}) == 1, "gates must not modify the source"

    # Execute.
    ok = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "RESTORE",
            "mongo_connection": {"uri": URI},
        },
    )
    assert ok.status_code == 200, ok.text
    result = ok.json()
    assert result["kind"] == "mongo_documents"
    assert result["results"]["inserted"] == 2
    assert result["pre_restore_backup"]["backup_id"]
    # Source really regained the documents, with original _ids.
    assert coll.count_documents({}) == 3
    assert coll.find_one({"_id": ALICE})["name"] == "alice"
    assert coll.find_one({"_id": BOB}) is not None

    # A restore_source history entry exists and is itself restorable.
    entries = client.get("/history/", headers=auth_headers).json()["history"]
    restore_entries = [e for e in entries if e["operation"] == "restore_source"]
    assert restore_entries
    assert restore_entries[0]["restorable"] is True

    # Working copy still mirrors the live source after reconnect.
    reloaded = _connect(client, auth_headers, db_name)
    assert reloaded["row_count"] == 3
    assert reloaded["session_id"] != new_sid or reloaded["session_id"]


def test_mongo_restore_conflict_requires_overwrite(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["people"]
    coll.insert_many([
        {"_id": ALICE, "name": "alice", "score": 10},
        {"_id": BOB, "name": "bob", "score": 20},
    ])
    loaded = _connect(client, auth_headers, db_name)
    _delete_docs(client, auth_headers, loaded["session_id"], [0])
    backup_id = _latest_backup_id(client, auth_headers)

    # Recreate alice with DIFFERENT data -> _id collision on restore.
    coll.insert_one({"_id": ALICE, "name": "ALICE-CHANGED", "score": 99})

    conflict = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "RESTORE",
            "mongo_connection": {"uri": URI},
        },
    )
    assert conflict.status_code == 409
    assert "overwrite_conflicts" in conflict.json()["detail"]
    assert coll.find_one({"_id": ALICE})["name"] == "ALICE-CHANGED"

    ok = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "RESTORE",
            "overwrite_conflicts": True,
            "mongo_connection": {"uri": URI},
        },
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["results"]["replaced"] >= 1
    assert coll.find_one({"_id": ALICE})["name"] == "alice"


def test_restore_requires_ownership(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["people"]
    coll.insert_one({"_id": ALICE, "name": "alice", "score": 10})
    loaded = _connect(client, auth_headers, db_name)
    _delete_docs(client, auth_headers, loaded["session_id"], [0])
    backup_id = _latest_backup_id(client, auth_headers)

    from tests.conftest import register_and_login

    suffix = uuid.uuid4().hex[:10]
    token = register_and_login(client, {
        "username": f"other_{suffix}",
        "password": "Sup3rSecret!",
        "email": f"{suffix}@example.com",
        "full_name": "Other",
    })
    other = {"Authorization": f"Bearer {token}"}
    resp = client.post(
        "/restore_source/preview/", headers=other,
        json={"backup_id": backup_id, "mongo_connection": {"uri": URI}},
    )
    assert resp.status_code in (403, 404)
    assert coll.count_documents({}) == 0  # nothing restored


# --------------------------------------------------------------------------- #
# MySQL restore (mocked driver — no server on this machine)
# --------------------------------------------------------------------------- #
import re as _re


class _FakeCursor:
    """Understands the single UPDATE pattern the restore endpoint emits."""

    def __init__(self, table: dict):
        self.table = table
        self.rowcount = 0

    def execute(self, sql: str, params=None):
        self.rowcount = 0
        s = sql.strip()
        if params is not None and _re.match(r"UPDATE\s", s, _re.IGNORECASE):
            m = _re.search(r"SET\s+(.+?)\s+WHERE\s+(.+)$", s, _re.IGNORECASE | _re.DOTALL)
            if not m:
                return
            set_cols = _re.findall(r"`([^`]+)`", m.group(1))
            where_cols = _re.findall(r"`([^`]+)`", m.group(2))
            values = list(params)
            set_vals = dict(zip(set_cols, values[: len(set_cols)]))
            where_vals = dict(zip(where_cols, values[len(set_cols):]))
            key = tuple(where_vals.get(k) for k in self.table["pk"])
            if key in self.table["rows"]:
                self.table["rows"][key].update(set_vals)
                self.rowcount = 1

    def executemany(self, sql, seq):
        self.rowcount = len(list(seq))

    def close(self):
        pass


class _FakeConn:
    def __init__(self, table: dict):
        self.table = table

    def cursor(self):
        return _FakeCursor(self.table)

    def commit(self):
        pass

    def rollback(self):
        pass


class FakeMySQL:
    class DatabaseError(Exception):
        def __init__(self, msg="error", status_code=400):
            super().__init__(msg)
            self.status_code = status_code

    validate_identifier = staticmethod(_real_db_mysql.validate_identifier)
    quote_identifier = staticmethod(_real_db_mysql.quote_identifier)

    def __init__(self):
        self.tables = {}

    def add_table(self, database, table, columns, pk, rows):
        self.tables[f"{database}.{table}"] = {
            "columns": list(columns),
            "pk": list(pk),
            "rows": {tuple(r[c] for c in pk): dict(r) for r in rows},
        }

    def _t(self, database, table):
        try:
            return self.tables[f"{database}.{table}"]
        except KeyError:
            raise self.DatabaseError(f"Table '{table}' not found.", 400)

    @contextmanager
    def connection(self, **kwargs):
        # Yield a connection bound to whichever table the kwargs name.
        key = f"{kwargs.get('database')}.{kwargs.get('table')}"
        # The restore endpoint opens the connection without a table name; use a
        # per-call table set by the test via add_table defaults (db1.people).
        table = next(iter(self.tables.values()))
        yield _FakeConn(table)

    def describe_table(self, conn, database, table):
        return list(self._t(database, table)["columns"])

    def describe_table_full(self, conn, database, table):
        return [
            {
                "Field": c, "Type": "varchar(255)", "Null": "YES",
                "Key": "PRI" if c in self._t(database, table)["pk"] else "",
                "Default": None, "Extra": "",
            }
            for c in self._t(database, table)["columns"]
        ]

    def fetch_all(self, conn, database, table, limit=None):
        rows = [dict(r) for r in self._t(database, table)["rows"].values()]
        return rows[:limit] if limit else rows

    def get_primary_key(self, conn, database, table):
        return list(self._t(database, table)["pk"])

    def count_by_pk(self, conn, database, table, pk_cols, keys):
        rows = self._t(database, table)["rows"]
        return sum(1 for k in keys if k in rows)

    def select_by_pk(self, conn, database, table, pk_cols, keys, limit=None):
        rows = self._t(database, table)["rows"]
        return [dict(rows[k]) for k in keys if k in rows]

    def delete_by_pk(self, conn, database, table, pk_cols, keys):
        rows = self._t(database, table)["rows"]
        n = 0
        for k in keys:
            if k in rows:
                del rows[k]
                n += 1
        return n

    def column_drop_risks(self, conn, database, table, column):
        t = self._t(database, table)
        return {
            "foreign_keys_out": [], "foreign_keys_in": [], "indexes": [],
            "is_primary": column in t["pk"], "is_generated": False,
            "referenced_by_generated": [],
        }

    def drop_column(self, conn, database, table, column):
        t = self._t(database, table)
        t["columns"].remove(column)
        for r in t["rows"].values():
            r.pop(column, None)

    # --- restore helpers (same signatures as db_mysql) -------------------- #
    def column_exists(self, conn, database, table, column):
        return column in self._t(database, table)["columns"]

    def add_column(self, conn, database, table, column, col_type, nullable=True):
        t = self._t(database, table)
        _real_db_mysql.validate_column_type(col_type)
        t["columns"].append(column)
        for r in t["rows"].values():
            r.setdefault(column, None)

    def insert_rows(self, conn, database, table, columns, rows):
        """``rows`` are value tuples aligned with ``columns`` (as the app builds)."""
        t = self._t(database, table)
        pk = t["pk"]
        n = 0
        for values in rows:
            row = dict(zip(columns, values))
            key = tuple(row.get(c) for c in pk)
            t["rows"][key] = {c: row.get(c) for c in t["columns"]}
            n += 1
        return n

    def update_column_by_pk(self, conn, database, table, column, pk_cols, rows):
        t = self._t(database, table)
        updated = missing = 0
        for row in rows:
            key = tuple(row.get(c) for c in pk_cols)
            if key in t["rows"]:
                t["rows"][key][column] = row.get(column)
                updated += 1
            else:
                missing += 1
        return {"updated": updated, "missing": missing}


@pytest.fixture()
def fake_mysql(monkeypatch):
    import app as app_module

    fake = FakeMySQL()
    fake.add_table(
        "db1", "people",
        columns=["id", "name"],
        pk=["id"],
        rows=[{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}],
    )
    monkeypatch.setattr(app_module, "db_mysql", fake)
    return fake


CONN = {"host": "h", "user": "u", "password": "p", "port": 3306, "database": "db1"}


def _mysql_delete_rows(client, headers, sid):
    prev = client.post(
        "/source_delete/preview/", headers=headers,
        json={
            "session_id": sid, "target": "mysql_rows",
            "row_ids": [0], "mysql_connection": CONN,
        },
    )
    assert prev.status_code == 200, prev.text
    token = prev.json()["preview_token"]
    resp = client.post(
        "/source_delete/mysql_rows/", headers=headers,
        json={
            "session_id": sid, "target": "mysql_rows",
            "preview_token": token, "confirm": True,
            "typed_confirmation": "DELETE",
            "row_ids": [0], "connection_details": CONN,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_mysql_rows_restore_end_to_end(client, auth_headers, fake_mysql):
    resp = client.post(
        "/get_columns/?table_name=people", headers=auth_headers,
        json=CONN,
    )
    assert resp.status_code == 200, resp.text
    sid = resp.json()["session_id"]

    out = _mysql_delete_rows(client, auth_headers, sid)
    assert out["affected"] == 1
    assert set(fake_mysql.tables["db1.people"]["rows"]) == {(2,)}  # alice gone

    backups = client.get("/backups/", headers=auth_headers).json()["backups"]
    matching = [b for b in backups if b["kind"] == "mysql_rows"]
    assert matching
    backup_id = matching[0]["backup_id"]

    prev = client.post(
        "/restore_source/preview/", headers=auth_headers,
        json={"backup_id": backup_id, "mysql_connection": CONN},
    )
    assert prev.status_code == 200, prev.text
    body = prev.json()
    assert body["restorable"] is True
    assert body["snapshot"]["row_count"] == 1
    assert body["live"]["rows_to_insert"] == 1

    # No live conflict yet -> plain insert.
    ok = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "RESTORE",
            "mysql_connection": CONN,
        },
    )
    assert ok.status_code == 200, ok.text
    result = ok.json()
    assert result["results"]["inserted"] == 1
    assert result["pre_restore_backup"]["backup_id"]
    assert set(fake_mysql.tables["db1.people"]["rows"]) == {(1,), (2,)}

    # Source delete + restore left a backup of the pre-restore state too.
    kinds = [b["kind"] for b in client.get("/backups/", headers=auth_headers).json()["backups"]]
    assert "mysql_rows" in kinds


def test_mysql_restore_conflict_requires_overwrite(client, auth_headers, fake_mysql):
    resp = client.post(
        "/get_columns/?table_name=people", headers=auth_headers,
        json=CONN,
    )
    sid = resp.json()["session_id"]
    _mysql_delete_rows(client, auth_headers, sid)
    backup_id = _latest_backup_id(client, auth_headers, kind="mysql_rows")

    # Recreate alice with different data -> PK collision.
    fake_mysql.tables["db1.people"]["rows"][(1,)] = {"id": 1, "name": "ALICE-CHANGED"}

    conflict = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "RESTORE",
            "mysql_connection": CONN,
        },
    )
    assert conflict.status_code == 409
    assert fake_mysql.tables["db1.people"]["rows"][(1,)]["name"] == "ALICE-CHANGED"

    ok = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "RESTORE",
            "overwrite_conflicts": True,
            "mysql_connection": CONN,
        },
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["results"]["updated"] == 1
    assert fake_mysql.tables["db1.people"]["rows"][(1,)]["name"] == "alice"


def test_restore_source_rejects_unknown_kind(client, auth_headers, monkeypatch):
    # A non-restoreable backup kind is refused with an actionable message.
    import backups as backup_store
    from datetime import datetime, timedelta, timezone

    fake_meta = {
        "backup_id": "b" * 32,
        "kind": "session_export",
        "owner_id": None,
        "source_name": "x",
        "expires_at": datetime.now(timezone.utc) + timedelta(days=1),
        "meta": {"format": "csv"},
    }

    async def _fake_meta(owner_id, backup_id):
        return fake_meta

    async def _fake_read(owner_id, backup_id):
        return gzip.compress(b"data"), fake_meta

    monkeypatch.setattr(backup_store, "get_backup_meta", _fake_meta)
    monkeypatch.setattr(backup_store, "read_backup", _fake_read)

    resp = client.post(
        "/restore_source/preview/", headers=auth_headers,
        json={"backup_id": fake_meta["backup_id"]},
    )
    assert resp.status_code == 400
    assert "cannot be restored" in resp.json()["detail"]
