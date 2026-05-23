"""Source deletion tests for MySQL (mocked driver, no server required).

Covers the full destructive-deletion contract:
preview token -> typed intent -> verified backup -> parameterised delete ->
session re-sync, plus every refusal path (no PK, stale token/selection,
zero match, partial without consent, unsafe column drops, backup failure,
wrong owner, file sessions).
"""

from __future__ import annotations

import gzip
from contextlib import contextmanager

import pytest

import db_mysql as _real_db_mysql


# --------------------------------------------------------------------------- #
# In-memory fake covering exactly the db_mysql surface the endpoints use.
# Identifier validation stays REAL (injection defences must keep running).
# --------------------------------------------------------------------------- #
class FakeMySQL:
    class DatabaseError(Exception):
        def __init__(self, msg="error", status_code=400):
            super().__init__(msg)
            self.status_code = status_code

    validate_identifier = staticmethod(_real_db_mysql.validate_identifier)
    quote_identifier = staticmethod(_real_db_mysql.quote_identifier)

    def __init__(self):
        # "db.table" -> {"columns", "pk", "rows": {pk_tuple: row}, extras}
        self.tables = {}
        self.deleted_keys = []
        self.dropped = []
        self.fail_drop_on = None

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
        yield object()

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
        out = [dict(rows[k]) for k in keys if k in rows]
        return out[:limit] if limit is not None else out

    def delete_by_pk(self, conn, database, table, pk_cols, keys):
        rows = self._t(database, table)["rows"]
        n = 0
        for k in keys:
            if k in rows:
                del rows[k]
                n += 1
        self.deleted_keys.append((f"{database}.{table}", [tuple(k) for k in keys]))
        return n

    def select_columns(self, conn, database, table, columns, limit=None):
        rows = [
            {c: r.get(c) for c in columns}
            for r in self._t(database, table)["rows"].values()
        ]
        return rows[:limit] if limit is not None else rows

    def column_drop_risks(self, conn, database, table, column):
        t = self._t(database, table)
        return {
            "foreign_keys_out": t.get("fk_out", {}).get(column, []),
            "foreign_keys_in": t.get("fk_in", {}).get(column, []),
            "indexes": t.get("indexes", {}).get(column, []),
            "is_primary": column in t["pk"],
            "is_generated": column in t.get("generated", set()),
            "referenced_by_generated": t.get("gen_refs", {}).get(column, []),
        }

    def drop_column(self, conn, database, table, column):
        if self.fail_drop_on == column:
            raise self.DatabaseError(f"denied dropping {column}", 400)
        t = self._t(database, table)
        t["columns"].remove(column)
        for r in t["rows"].values():
            r.pop(column, None)
        self.dropped.append(f"{database}.{table}.{column}")


@pytest.fixture()
def fake_mysql(monkeypatch):
    import app as app_module

    fake = FakeMySQL()
    fake.add_table(
        "db1", "people",
        columns=["id", "name", "score"],
        pk=["id"],
        rows=[
            {"id": 1, "name": "alice", "score": 10},
            {"id": 2, "name": "bob", "score": 20},
            {"id": 3, "name": "carol", "score": 30},
        ],
    )
    monkeypatch.setattr(app_module, "db_mysql", fake)
    return fake


CONN = {"host": "h", "user": "u", "password": "p", "port": 3306, "database": "db1"}


def _make_session(client, headers, table="people", db="db1"):
    resp = client.post(
        f"/get_columns/?table_name={table}",
        headers=headers,
        json={"host": "h", "user": "u", "password": "p", "port": 3306, "database": db},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["session_id"], resp.text
    return resp.json()["session_id"]


def _preview(client, headers, sid, target, **extra):
    return client.post(
        "/source_delete/preview/",
        headers=headers,
        json={"session_id": sid, "target": target, **extra},
    )


def _delete(client, headers, sid, target, token, **extra):
    payload = {
        "session_id": sid,
        "target": target,
        "preview_token": token,
        "confirm": True,
        "typed_confirmation": "DELETE",
        **extra,
    }
    return client.post(f"/source_delete/{target}/", headers=headers, json=payload)


# --------------------------------------------------------------------------- #
# Rows: happy path
# --------------------------------------------------------------------------- #
def test_preview_and_delete_rows_end_to_end(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)

    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0, 2], mysql_connection=CONN,
    )
    assert prev.status_code == 200, prev.text
    body = prev.json()
    assert body["preview_token"]
    assert body["requested_count"] == 2
    assert body["matched_count"] == 2
    assert body["irreversible"] is True
    assert body["primary_key"] == ["id"]
    assert len(body["sample"]) == 2
    assert body["backup"]["kind"] == "mysql_rows"
    assert body["backup"]["format"] == "csv"

    # Intent gates.
    no_confirm = client.post(
        "/source_delete/mysql_rows/", headers=auth_headers,
        json={
            "session_id": sid, "target": "mysql_rows",
            "preview_token": body["preview_token"], "confirm": False,
            "typed_confirmation": "DELETE",
            "row_ids": [0, 2], "connection_details": CONN,
        },
    )
    assert no_confirm.status_code == 400
    assert "confirm" in no_confirm.json()["detail"].lower()

    wrong_typed = client.post(
        "/source_delete/mysql_rows/", headers=auth_headers,
        json={
            "session_id": sid, "target": "mysql_rows",
            "preview_token": body["preview_token"], "confirm": True,
            "typed_confirmation": "delete",
            "row_ids": [0, 2], "connection_details": CONN,
        },
    )
    assert wrong_typed.status_code == 400
    assert "DELETE" in wrong_typed.json()["detail"]

    bad_token = _delete(
        client, auth_headers, sid, "mysql_rows", "1.deadbeef",
        row_ids=[0, 2], connection_details=CONN,
    )
    assert bad_token.status_code == 409

    ok = _delete(
        client, auth_headers, sid, "mysql_rows", body["preview_token"],
        row_ids=[0, 2], connection_details=CONN,
    )
    assert ok.status_code == 200, ok.text
    out = ok.json()
    assert out["affected"] == 2
    assert "Permanently deleted 2 row(s)" in out["message"]
    table, keys = fake_mysql.deleted_keys[-1]
    assert table == "db1.people"
    assert keys == [(1,), (3,)]
    # Source really lost those rows.
    assert set(fake_mysql.tables["db1.people"]["rows"]) == {(2,)}

    # Working copy re-synced: a NEW session replaces the stale one.
    assert out["session_id"] and out["session_id"] != sid
    assert out["row_count"] == 1
    assert [r["name"] for r in out["preview"]] == ["bob"]

    old = client.get(f"/session_preview/?session_id={sid}", headers=auth_headers)
    assert old.status_code == 404, "stale session must be retired"
    new_prev = client.get(
        f"/session_preview/?session_id={out['session_id']}", headers=auth_headers
    )
    assert new_prev.status_code == 200
    assert new_prev.json()["row_count"] == 1

    # Backup exists, is owned, and downloads as gzipped CSV.
    lst = client.get("/backups/", headers=auth_headers)
    assert lst.status_code == 200
    backups = lst.json()["backups"]
    assert any(b["kind"] == "mysql_rows" for b in backups)
    bid = [b for b in backups if b["kind"] == "mysql_rows"][0]["backup_id"]
    dl = client.get(f"/backups/{bid}/download/", headers=auth_headers)
    assert dl.status_code == 200
    assert "attachment" in dl.headers["content-disposition"]
    csv_text = gzip.decompress(dl.content).decode("utf-8")
    lines = [ln for ln in csv_text.strip().splitlines() if ln]
    assert lines[0] == "id,name,score"
    assert len(lines) == 3  # header + alice + carol


def test_backup_failure_aborts_delete(client, auth_headers, fake_mysql, monkeypatch):
    import backups as backup_store
    from backups import BackupError

    async def _boom(*args, **kwargs):
        raise BackupError("GridFS unavailable")

    monkeypatch.setattr(backup_store, "create_backup", _boom)

    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0, 1], mysql_connection=CONN,
    )
    assert prev.status_code == 200, prev.text

    resp = _delete(
        client, auth_headers, sid, "mysql_rows", prev.json()["preview_token"],
        row_ids=[0, 1], connection_details=CONN,
    )
    assert resp.status_code == 400, resp.text
    assert "No source data was modified" in resp.json()["detail"]
    assert fake_mysql.deleted_keys == [], "delete must not run when backup fails"
    assert len(fake_mysql.tables["db1.people"]["rows"]) == 3


def test_require_backup_false_still_deletes_with_warning(
    client, auth_headers, fake_mysql, monkeypatch
):
    import backups as backup_store
    from backups import BackupError

    async def _boom(*args, **kwargs):
        raise BackupError("GridFS unavailable")

    monkeypatch.setattr(backup_store, "create_backup", _boom)

    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection=CONN,
    )
    resp = _delete(
        client, auth_headers, sid, "mysql_rows", prev.json()["preview_token"],
        row_ids=[0], connection_details=CONN, require_backup=False,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["affected"] == 1
    assert any("Backup failed" in w for w in resp.json()["warnings"])


# --------------------------------------------------------------------------- #
# Rows: refusal paths
# --------------------------------------------------------------------------- #
def test_table_without_primary_key_is_rejected(client, auth_headers, fake_mysql):
    fake_mysql.add_table(
        "db1", "nopk", columns=["a", "b"], pk=[], rows=[{"a": 1, "b": 2}]
    )
    sid = _make_session(client, auth_headers, table="nopk")
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection=CONN,
    )
    assert prev.status_code == 409, prev.text
    assert "PRIMARY KEY" in prev.json()["detail"]


def test_stale_preview_token_rejected(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection=CONN,
    )
    assert prev.status_code == 200
    # Same token but a DIFFERENT selection must not be deletable.
    resp = _delete(
        client, auth_headers, sid, "mysql_rows", prev.json()["preview_token"],
        row_ids=[1], connection_details=CONN,
    )
    assert resp.status_code == 409, resp.text
    assert "no longer matches" in resp.json()["detail"]
    assert fake_mysql.deleted_keys == []


def test_zero_match_409_and_allow_partial(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0, 1, 2], mysql_connection=CONN,
    )
    assert prev.json()["matched_count"] == 3

    # Concurrent change: row id=3 disappears from the SOURCE before delete.
    del fake_mysql.tables["db1.people"]["rows"][(3,)]

    resp = _delete(
        client, auth_headers, sid, "mysql_rows", prev.json()["preview_token"],
        row_ids=[0, 1, 2], connection_details=CONN,
    )
    assert resp.status_code == 409, resp.text
    assert "3 of 3" not in resp.json()["detail"]  # sanity: it reports 2 of 3
    assert "2 of 3" in resp.json()["detail"]
    assert fake_mysql.deleted_keys == []

    resp2 = _delete(
        client, auth_headers, sid, "mysql_rows", prev.json()["preview_token"],
        row_ids=[0, 1, 2], connection_details=CONN, allow_partial=True,
    )
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["matched_count"] == 2
    assert resp2.json()["affected"] == 2


def test_zero_match_when_all_rows_vanished(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection=CONN,
    )
    fake_mysql.tables["db1.people"]["rows"].clear()
    resp = _delete(
        client, auth_headers, sid, "mysql_rows", prev.json()["preview_token"],
        row_ids=[0], connection_details=CONN,
    )
    assert resp.status_code == 409
    assert "Nothing was deleted" in resp.json()["detail"]


def test_stale_row_selection_is_400(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    # Working-copy delete first, then a preview for a removed row.
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection=CONN,
    )
    assert prev.status_code == 400, prev.text
    assert "Stale row selection" in prev.json()["detail"]


def test_file_session_cannot_delete_from_source(client, auth_headers):
    up = client.post(
        "/upload_file/",
        headers=auth_headers,
        files={"file": ("t.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert up.status_code == 200, up.text
    sid = up.json()["session_id"]
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection=CONN,
    )
    assert prev.status_code == 400
    assert "original file is never modified" in prev.json()["detail"]


def test_preview_requires_auth(client, fake_mysql):
    resp = client.post(
        "/source_delete/preview/",
        json={"session_id": "a" * 16, "target": "mysql_rows", "row_ids": [0]},
    )
    assert resp.status_code == 401


def test_other_users_session_is_404(client, auth_headers, unique_user, fake_mysql):
    from tests.conftest import register_and_login

    sid = _make_session(client, auth_headers)
    other = {
        "username": "other_" + unique_user["username"][-6:],
        "password": "Sup3rSecret!",
        "email": "other_" + unique_user["email"],
        "full_name": "Other",
    }
    other_headers = {"Authorization": f"Bearer {register_and_login(client, other)}"}
    prev = _preview(
        client, other_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection=CONN,
    )
    assert prev.status_code == 404


def test_connection_database_mismatch_is_400(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection={**CONN, "database": "otherdb"},
    )
    assert prev.status_code == 400
    assert "does not match" in prev.json()["detail"]


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
def test_column_drop_end_to_end(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_columns",
        columns=["score"], mysql_connection=CONN,
    )
    assert prev.status_code == 200, prev.text
    body = prev.json()
    assert body["columns"] == ["score"]
    assert "score" in body["schema"]
    assert body["backup"]["kind"] == "mysql_columns"
    assert any("auto-commit" in w for w in body["warnings"])

    ok = _delete(
        client, auth_headers, sid, "mysql_columns", body["preview_token"],
        columns=["score"], connection_details=CONN,
    )
    assert ok.status_code == 200, ok.text
    out = ok.json()
    assert out["dropped_columns"] == ["score"]
    assert out["partial_failure"] is None
    assert "score" not in out["columns"], "refreshed state must not show dropped column"
    assert "db1.people.score" in fake_mysql.dropped
    assert "score" not in fake_mysql.tables["db1.people"]["columns"]

    # Backup holds the column values keyed by pk.
    backups = client.get("/backups/", headers=auth_headers).json()["backups"]
    col_bak = [b for b in backups if b["kind"] == "mysql_columns"][0]
    dl = client.get(f"/backups/{col_bak['backup_id']}/download/", headers=auth_headers)
    csv_text = gzip.decompress(dl.content).decode("utf-8")
    lines = [ln for ln in csv_text.strip().splitlines() if ln]
    assert lines[0] == "id,score"
    assert len(lines) == 4


def test_column_drop_blocked_by_foreign_key(client, auth_headers, fake_mysql):
    fake_mysql.tables["db1.people"]["fk_in"] = {
        "score": [{"constraint": "fk_score", "table": "orders"}]
    }
    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_columns",
        columns=["score"], mysql_connection=CONN,
    )
    assert prev.status_code == 409, prev.text
    assert "Unsafe to drop" in prev.json()["detail"]
    assert "fk_score" in prev.json()["detail"]

    # And the delete path re-checks live schema even with a token.
    good_prev = _preview(
        client, auth_headers, sid, "mysql_columns",
        columns=["name"], mysql_connection=CONN,
    )
    resp = _delete(
        client, auth_headers, sid, "mysql_columns", good_prev.json()["preview_token"],
        columns=["score"], connection_details=CONN,  # token was for "name"
    )
    assert resp.status_code in (400, 409), resp.text
    assert fake_mysql.dropped == []


def test_cannot_drop_every_column_or_primary_key(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    all_cols = _preview(
        client, auth_headers, sid, "mysql_columns",
        columns=["id", "name", "score"], mysql_connection=CONN,
    )
    assert all_cols.status_code == 400
    assert "every column" in all_cols.json()["detail"]

    pk = _preview(
        client, auth_headers, sid, "mysql_columns",
        columns=["id"], mysql_connection=CONN,
    )
    assert pk.status_code == 409
    assert "PRIMARY KEY" in pk.json()["detail"]


def test_column_drop_partial_failure_is_honest(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_columns",
        columns=["name", "score"], mysql_connection=CONN,
    )
    assert prev.status_code == 200, prev.text

    fake_mysql.fail_drop_on = "score"
    resp = _delete(
        client, auth_headers, sid, "mysql_columns", prev.json()["preview_token"],
        columns=["name", "score"], connection_details=CONN,
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["dropped_columns"] == ["name"]
    assert out["partial_failure"]["column"] == "score"
    assert "before 'score' failed" in out["message"]
    assert out["backup"]["backup_id"], "backup id must be reported"
    # Source really changed for the successful part.
    assert "name" not in fake_mysql.tables["db1.people"]["columns"]
    assert "score" in fake_mysql.tables["db1.people"]["columns"]


def test_internal_row_id_column_cannot_be_dropped(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_columns",
        columns=["_tabula_row_id"], mysql_connection=CONN,
    )
    assert prev.status_code == 400
    assert "internal column" in prev.json()["detail"]


# --------------------------------------------------------------------------- #
# Target / endpoint mismatches
# --------------------------------------------------------------------------- #
def test_wrong_target_on_endpoint_is_400(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection=CONN,
    )
    resp = client.post(
        "/source_delete/mongo_documents/", headers=auth_headers,
        json={
            "session_id": sid, "target": "mongo_documents",
            "preview_token": prev.json()["preview_token"], "confirm": True,
            "typed_confirmation": "DELETE", "row_ids": [0],
            "mongo_connection": {"uri": "mongodb://localhost:27017"},
        },
    )
    # Session source (mysql) vs target (mongo) mismatch is caught first.
    assert resp.status_code == 400
    assert "does not match" in resp.json()["detail"]


def test_unknown_preview_target_is_rejected(client, auth_headers, fake_mysql):
    sid = _make_session(client, auth_headers)
    resp = _preview(client, auth_headers, sid, "drop_everything")
    assert resp.status_code == 422  # pydantic validation


# --------------------------------------------------------------------------- #
# Backup ownership isolation
# --------------------------------------------------------------------------- #
def test_backup_download_denied_for_other_user(client, auth_headers, unique_user, fake_mysql):
    from tests.conftest import register_and_login

    sid = _make_session(client, auth_headers)
    prev = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0], mysql_connection=CONN,
    )
    ok = _delete(
        client, auth_headers, sid, "mysql_rows", prev.json()["preview_token"],
        row_ids=[0], connection_details=CONN,
    )
    assert ok.status_code == 200, ok.text
    bid = ok.json()["backup"]["backup_id"]

    other = {
        "username": "bkr_" + unique_user["username"][-8:],
        "password": "Sup3rSecret!",
        "email": "bk_" + unique_user["email"],
        "full_name": "Other",
    }
    other_headers = {"Authorization": f"Bearer {register_and_login(client, other)}"}
    denied = client.get(f"/backups/{bid}/download/", headers=other_headers)
    assert denied.status_code == 404
    lst = client.get("/backups/", headers=other_headers)
    assert lst.json()["backups"] == []
