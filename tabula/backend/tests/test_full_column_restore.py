"""Regression tests: full-column transformation + restore/preview synchronization.

Locks in three behaviors:

1. ``/apply/`` transforms the ENTIRE underlying column — never just the rows
   visible in the preview sample — reports processed/failed counts, and the
   download exports the complete transformed dataset with unrelated columns
   untouched.
2. "Restore as working copy" makes the exact saved snapshot the active working
   dataset (preview, download, subsequent transforms) even when the original
   session was deleted, keeps the historical snapshot unchanged, and stamps
   the session with the restored-version marker.
3. "Restore source database to this version" verifies the live source after
   the write and returns a refreshed session (session_id + preview + counts)
   so the UI never needs a manual reconnect; partial outcomes are reported
   honestly instead of as clean successes.
"""

from __future__ import annotations

import io
import uuid

import pandas as pd
import pytest
from bson import ObjectId
from pymongo import MongoClient

PREVIEW_LIMIT = 50
ROWS = 1000
URI = "mongodb://localhost:27017"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _csv_bytes(n: int = ROWS) -> bytes:
    lines = ["id,weight,label"] + [f"{i},{i},row{i}" for i in range(1, n + 1)]
    return ("\n".join(lines) + "\n").encode()


def _upload(client, headers, name: str = "big.csv", payload: bytes | None = None, n: int = ROWS):
    resp = client.post(
        "/upload_file/",
        headers=headers,
        files={"file": (name, payload if payload is not None else _csv_bytes(n), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _gen(client, headers, column: str, examples, classification: str | None = None) -> str:
    items = [{"input": a, "output": b} for a, b in examples]
    if classification is None:
        r = client.post(
            "/classify/", headers=headers,
            json={"column": column, "examples": items},
        )
        assert r.status_code == 200, r.text
        classification = r.json()["classification"]
    r = client.post(
        "/generate_function/", headers=headers,
        json={"column": column, "examples": items, "classification": classification},
    )
    assert r.status_code == 200, r.text
    return r.json()["transformation_id"]


def _apply(client, headers, sid: str, column: str, tid: str, source: str = "file"):
    r = client.post(
        "/apply/", headers=headers,
        json={
            "transformation_id": tid,
            "selected_column": column,
            "source_type": source,
            "session_id": sid,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _download_csv(client, headers, sid: str) -> pd.DataFrame:
    r = client.get(f"/download/?session_id={sid}", headers=headers)
    assert r.status_code == 200, r.text
    return pd.read_csv(io.BytesIO(r.content))


def _session_preview(client, headers, sid: str) -> dict:
    r = client.get(f"/session_preview/?session_id={sid}", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def _history(client, headers, **params) -> list:
    from urllib.parse import urlencode

    qs = urlencode(params)
    r = client.get(f"/history/{'?' + qs if qs else ''}", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["history"]


def _user_oid(client, headers) -> ObjectId:
    me = client.get("/users/me/", headers=headers)
    assert me.status_code == 200, me.text
    return ObjectId(me.json()["id"])


def _insert_transformation(owner: ObjectId, code: str, column: str) -> str:
    """Store a transformation directly (server-side code registry shape)."""
    from datetime import datetime, timedelta, timezone

    import database

    tid = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    database.db.transformations.insert_one(
        {
            "transformation_id": tid,
            "owner_id": owner,
            "code": code,
            "classification": "General",
            "description": "test",
            "template": None,
            "used_llm": False,
            "column": column,
            "created_at": now,
            "expires_at": now + timedelta(hours=24),
        }
    )
    return tid


# --------------------------------------------------------------------------- #
# 1. Full-column transformation (preview is only a sample)
# --------------------------------------------------------------------------- #
def test_apply_transforms_entire_column_beyond_preview(client, auth_headers, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "preview_rows", PREVIEW_LIMIT)

    up = _upload(client, auth_headers, n=ROWS)
    sid = up["session_id"]
    assert up["row_count"] == ROWS
    assert len(up["preview"]) == PREVIEW_LIMIT, "upload preview must be sampled"

    tid = _gen(client, auth_headers, "weight", [("7", "70"), ("8", "80")])
    out = _apply(client, auth_headers, sid, "weight", tid)

    # Response reports on the full dataset, and the preview stays a sample.
    assert out["row_count"] == ROWS
    assert out["processed_rows"] == ROWS
    assert out["failed_rows"] == 0
    assert out["transformed_rows"] == ROWS
    assert len(out["preview"]) == PREVIEW_LIMIT

    # Every row of the authoritative dataset is transformed, including rows
    # 51..1000 that never appeared in the preview.
    df = _download_csv(client, auth_headers, sid)
    assert len(df) == ROWS
    assert (df["weight"] == df["id"] * 10).all()
    beyond = df.iloc[PREVIEW_LIMIT:]
    assert (beyond["weight"] == beyond["id"] * 10).all()

    # Unrelated columns keep their values; preview was regenerated from the
    # updated full dataset (not a preview-only patch).
    assert (df["label"] == df["id"].map(lambda i: f"row{i}")).all()
    assert all(rec["weight"] == rec["id"] * 10 for rec in out["preview"])

    # Session metadata tracks the working dataset.
    sessions = client.get("/sessions/", headers=auth_headers).json()["sessions"]
    meta = next(s for s in sessions if s["session_id"] == sid)
    assert meta["row_count"] == ROWS


def test_full_column_transform_handles_nulls(client, auth_headers, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "preview_rows", 10)
    n = 300
    lines = ["id,weight,label"]
    for i in range(1, n + 1):
        w = "" if i % 100 == 0 else str(i)  # nulls at 100, 200, 300
        lines.append(f"{i},{w},row{i}")
    payload = ("\n".join(lines) + "\n").encode()

    up = _upload(client, auth_headers, name="nulls.csv", payload=payload, n=n)
    sid = up["session_id"]
    tid = _gen(client, auth_headers, "weight", [("7", "70"), ("8", "80")])
    out = _apply(client, auth_headers, sid, "weight", tid)

    assert out["processed_rows"] == n
    assert out["failed_rows"] == 0

    df = _download_csv(client, auth_headers, sid)
    assert len(df) == n
    # The weight column is mixed ("MISSING" + numbers), so compare numerically.
    ok = df[df["id"] % 100 != 0]
    assert (pd.to_numeric(ok["weight"], errors="coerce") == ok["id"] * 10).all()
    nulls = df[df["id"] % 100 == 0]
    assert set(nulls["weight"].astype(str)) == {"MISSING"}
    assert (df["label"] == df["id"].map(lambda i: f"row{i}")).all()


def test_partial_failures_are_reported_not_silent(client, auth_headers):
    up = _upload(client, auth_headers, n=300)
    sid = up["session_id"]
    owner = _user_oid(client, auth_headers)
    code = (
        "def transform(x):\n"
        "    v = int(x)\n"
        "    if v > 250:\n"
        "        raise ValueError('value above 250 not supported')\n"
        "    return v * 10\n"
    )
    tid = _insert_transformation(owner, code, "weight")
    out = _apply(client, auth_headers, sid, "weight", tid)

    # All 300 rows were processed; 50 failed and are reported (nulled), the
    # remaining 250 succeeded.  Nothing is silently half-written.
    assert out["processed_rows"] == 300
    assert out["failed_rows"] == 50
    assert out["transformed_rows"] == 250
    assert out["row_count"] == 300
    assert out["row_errors"], "failure reasons must be surfaced"

    df = _download_csv(client, auth_headers, sid)
    assert len(df) == 300
    assert (df[df["id"] <= 250]["weight"] == df[df["id"] <= 250]["id"] * 10).all()
    assert df[df["id"] > 250]["weight"].isna().all()

    entries = _history(client, auth_headers, session_id=sid)
    partial = [e for e in entries if e["operation"] == "transform_apply" and e["status"] == "partial"]
    assert partial, "a partial apply must be recorded as such in history"


# --------------------------------------------------------------------------- #
# 2. Restore as working copy (files) — exact version becomes active
# --------------------------------------------------------------------------- #
def test_restore_working_copy_activates_exact_version(client, auth_headers):
    up = _upload(client, auth_headers, n=20)
    sid = up["session_id"]
    tid = _gen(client, auth_headers, "weight", [("7", "70"), ("8", "80")])
    _apply(client, auth_headers, sid, "weight", tid)  # after-state v2: weight x10

    # Further changes that must be abandoned by the restore.
    deleted = client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0, 1, 2, 3, 4]},
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["row_count"] == 15

    entry = next(e for e in _history(client, auth_headers, session_id=sid)
                 if e["operation"] == "transform_apply")
    vid = entry["version_id"]
    assert vid
    snap_before = client.get(
        f"/history/versions/{vid}/download/", headers=auth_headers
    )
    assert snap_before.status_code == 200, snap_before.text

    r = client.post(
        "/history/restore_working/", headers=auth_headers,
        json={"session_id": sid, "version_id": vid, "confirm": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == sid
    assert body["row_count"] == 20, "restore must bring back all 20 rows"
    assert body["source"] == "file"
    assert body["restored_version_number"] == entry["version"]["version_number"]
    assert all(rec["weight"] == rec["id"] * 10 for rec in body["preview"])

    # Authoritative session state now matches the restored version, and the
    # session is stamped with the restored-version marker.
    sp = _session_preview(client, auth_headers, sid)
    assert sp["row_count"] == 20
    assert all(rec["weight"] == rec["id"] * 10 for rec in sp["preview"])
    marker = sp["source_meta"]["history"]
    assert marker["operation"] == "restore_working_copy"
    assert marker["version_number"] == entry["version"]["version_number"]

    # Download exports the restored dataset.
    df = _download_csv(client, auth_headers, sid)
    assert len(df) == 20
    assert (df["weight"] == df["id"] * 10).all()
    assert (df["label"] == df["id"].map(lambda i: f"row{i}")).all()

    # Subsequent transformations run on the RESTORED dataset (20 rows), not
    # the abandoned 15-row state.
    tid2 = _gen(client, auth_headers, "weight", [("70", "700"), ("80", "800")])
    out2 = _apply(client, auth_headers, sid, "weight", tid2)
    assert out2["row_count"] == 20
    df2 = _download_csv(client, auth_headers, sid)
    assert len(df2) == 20
    assert (df2["weight"] == df2["id"] * 100).all()

    # The historical snapshot was not mutated, and the restore itself was
    # recorded in history.
    snap_after = client.get(f"/history/versions/{vid}/download/", headers=auth_headers)
    assert snap_after.status_code == 200
    assert snap_after.content == snap_before.content
    ops = [e["operation"] for e in _history(client, auth_headers, session_id=sid)]
    assert "restore_working_copy" in ops


def test_restore_working_copy_after_session_deleted(client, auth_headers):
    up = _upload(client, auth_headers, n=10)
    sid = up["session_id"]
    tid = _gen(client, auth_headers, "weight", [("7", "70"), ("8", "80")])
    _apply(client, auth_headers, sid, "weight", tid)
    entry = next(e for e in _history(client, auth_headers, session_id=sid)
                 if e["operation"] == "transform_apply")
    vid = entry["version_id"]

    # The original session disappears (expired / deleted).
    gone = client.delete(f"/sessions/{sid}", headers=auth_headers)
    assert gone.status_code < 400, gone.text

    r = client.post(
        "/history/restore_working/", headers=auth_headers,
        json={"session_id": sid, "version_id": vid, "confirm": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("new_session") is True
    new_sid = body["session_id"]
    assert new_sid and new_sid != sid
    assert body["row_count"] == 10
    assert all(rec["weight"] == rec["id"] * 10 for rec in body["preview"])

    # The replacement session is fully usable: preview + download agree.
    sp = _session_preview(client, auth_headers, new_sid)
    assert sp["row_count"] == 10
    df = _download_csv(client, auth_headers, new_sid)
    assert len(df) == 10
    assert (df["weight"] == df["id"] * 10).all()


def test_restore_working_copy_for_json_and_xlsx(client, auth_headers):
    base = pd.DataFrame({
        "id": range(1, 9),
        "weight": [i * 2 for i in range(1, 9)],
        "label": [f"r{i}" for i in range(1, 9)],
    })

    # JSON
    payload = base.to_json(orient="records").encode()
    up = _upload(client, auth_headers, name="data.json", payload=payload)
    assert up["file_type"] == "json"
    sid = up["session_id"]
    tid = _gen(client, auth_headers, "weight", [("2", "20"), ("4", "40")])
    _apply(client, auth_headers, sid, "weight", tid)
    entries = _history(client, auth_headers, session_id=sid)
    import_entry = next(e for e in entries if e["operation"] == "import")
    r = client.post(
        "/history/restore_working/", headers=auth_headers,
        json={"session_id": sid, "version_id": import_entry["version_id"], "confirm": True},
    )
    assert r.status_code == 200, r.text
    df = _download_csv(client, auth_headers, sid)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), base.reset_index(drop=True), check_dtype=False
    )

    # XLSX (requires openpyxl, same as upload support)
    try:
        import openpyxl  # noqa: F401
    except ImportError:  # pragma: no cover - environment-dependent
        pytest.skip("openpyxl not installed")
    buf = io.BytesIO()
    base.to_excel(buf, index=False)
    up2 = _upload(client, auth_headers, name="data.xlsx", payload=buf.getvalue())
    assert up2["file_type"] == "xlsx"
    sid2 = up2["session_id"]
    tid2 = _gen(client, auth_headers, "weight", [("2", "20"), ("4", "40")])
    _apply(client, auth_headers, sid2, "weight", tid2)
    import2 = next(e for e in _history(client, auth_headers, session_id=sid2)
                   if e["operation"] == "import")
    r2 = client.post(
        "/history/restore_working/", headers=auth_headers,
        json={"session_id": sid2, "version_id": import2["version_id"], "confirm": True},
    )
    assert r2.status_code == 200, r2.text
    df2 = _download_csv(client, auth_headers, sid2)
    pd.testing.assert_frame_equal(
        df2.reset_index(drop=True), base.reset_index(drop=True), check_dtype=False
    )


def test_column_metadata_and_order_survive_edits(client, auth_headers):
    lines = ["id,weight,label"] + [f"{i},{i},row{i}" for i in range(1, 11)]
    payload = ("\n".join(lines) + "\n").encode()
    up = _upload(client, auth_headers, name="cols.csv", payload=payload)
    sid = up["session_id"]

    deleted = client.post(
        "/delete_columns/", headers=auth_headers,
        json={"session_id": sid, "columns": ["weight"]},
    )
    assert deleted.status_code == 200, deleted.text
    del_body = deleted.json()
    assert del_body["row_count"] == 10

    # Session metadata reflects the CURRENT working dataset (row count
    # stays correct after edits).
    sessions = client.get("/sessions/", headers=auth_headers).json()["sessions"]
    meta = next(s for s in sessions if s["session_id"] == sid)
    assert meta["row_count"] == 10

    # Restoring the deleted column re-inserts it at its ORIGINAL position,
    # regardless of the position the client echoes back.
    del_col = del_body["deleted_columns"][0]
    assert del_col["name"] == "weight"
    echo = {"name": del_col["name"], "position": 99, "values": del_col["values"]}
    restored = client.post(
        "/restore_columns/", headers=auth_headers,
        json={"session_id": sid, "columns": [echo]},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["columns"] == ["id", "weight", "label"]


# --------------------------------------------------------------------------- #
# 3. Restore source database — verify the live source + refresh the session
# --------------------------------------------------------------------------- #
def _mongo_db_fixture():
    mc = MongoClient(URI, serverSelectionTimeoutMS=3000)
    try:
        mc.admin.command("ping")
    except Exception:
        mc.close()
        pytest.skip("Local MongoDB not available")
    name = f"tabula_rs_{uuid.uuid4().hex[:8]}"
    return mc, name


@pytest.fixture()
def mongo_db():
    mc, name = _mongo_db_fixture()
    yield mc, name
    mc.drop_database(name)
    mc.close()


def _mongo_connect(client, headers, db_name, collection="people"):
    resp = client.post(
        f"/get_mongo_data/?database_name={db_name}&collection_name={collection}",
        headers=headers, json={"uri": URI},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _mongo_delete_docs(client, headers, sid, row_ids):
    prev = client.post(
        "/source_delete/preview/", headers=headers,
        json={
            "session_id": sid, "target": "mongo_documents",
            "row_ids": row_ids, "mongo_connection": {"uri": URI},
        },
    )
    assert prev.status_code == 200, prev.text
    resp = client.post(
        "/source_delete/mongo_documents/", headers=headers,
        json={
            "session_id": sid, "target": "mongo_documents",
            "preview_token": prev.json()["preview_token"], "confirm": True,
            "typed_confirmation": "DELETE",
            "row_ids": row_ids, "mongo_connection": {"uri": URI},
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _latest_backup(client, headers, kind):
    backups = client.get("/backups/", headers=headers).json()["backups"]
    matching = [b for b in backups if b["kind"] == kind]
    assert matching, backups
    return matching[0]["backup_id"]


def test_mongo_restore_returns_fresh_session_without_reconnect(
    client, auth_headers, mongo_db
):
    from bson import ObjectId as OID

    mc, db_name = mongo_db
    ids = [OID("68f0a1b2c3d4e5f607182941"), OID("68f0a1b2c3d4e5f607182942"),
           OID("68f0a1b2c3d4e5f607182943")]
    coll = mc[db_name]["people"]
    coll.insert_many([
        {"_id": ids[0], "name": "alice", "score": 10},
        {"_id": ids[1], "name": "bob", "score": 20},
        {"_id": ids[2], "name": "carol", "score": 30},
    ])

    loaded = _mongo_connect(client, auth_headers, db_name)
    out = _mongo_delete_docs(client, auth_headers, loaded["session_id"], [0, 1])
    assert coll.count_documents({}) == 1
    backup_id = _latest_backup(client, auth_headers, "mongo_documents")

    ok = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "RESTORE",
            "session_id": out["session_id"],
            "mongo_connection": {"uri": URI},
        },
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()

    # Actual source really restored...
    assert coll.count_documents({}) == 3

    # ...verified server-side (the backup covers the 2 deleted documents)...
    assert body["verification"]["verified"] is True
    assert body["verification"]["found"] == 2
    assert body["verification"]["expected"] == 2

    # ...and the response carries a refreshed session the UI can adopt
    # WITHOUT a manual reconnect.
    assert body["session_id"], "restore must return a refreshed session"
    assert body["source"] == "mongodb"
    assert body["source_name"] == f"{db_name}.people"
    assert body["row_count"] == 3
    assert isinstance(body["preview"], list) and body["preview"]
    assert all("score" in rec for rec in body["preview"])

    sp = _session_preview(client, auth_headers, body["session_id"])
    assert sp["row_count"] == 3
    assert sp["source"] == "mongodb"
    assert sp["source_name"] == f"{db_name}.people"

    # History entry is linked to the refreshed session.
    entries = _history(client, auth_headers, source_type="mongodb")
    restore_entries = [e for e in entries if e["operation"] == "restore_source"]
    assert restore_entries and restore_entries[0]["session_id"] == body["session_id"]


def test_mongo_restore_reports_missing_documents_as_partial(
    client, auth_headers, mongo_db
):
    from bson import ObjectId as OID

    mc, db_name = mongo_db
    a = OID("68f0a1b2c3d4e5f607182951")
    b = OID("68f0a1b2c3d4e5f607182952")
    coll = mc[db_name]["people"]
    coll.insert_many([
        {"_id": a, "name": "alice", "score": 10},
        {"_id": b, "name": "bob", "score": 20},
    ])
    loaded = _mongo_connect(client, auth_headers, db_name)
    sid = loaded["session_id"]

    # 1) Unset 'score' on both docs -> a mongo_field_values backup is taken.
    prev_fields = client.post(
        "/source_delete/preview/", headers=auth_headers,
        json={
            "session_id": sid, "target": "mongo_fields",
            "row_ids": [0, 1], "fields": ["score"],
            "mongo_connection": {"uri": URI},
        },
    )
    assert prev_fields.status_code == 200, prev_fields.text
    del_fields = client.post(
        "/source_delete/mongo_fields/", headers=auth_headers,
        json={
            "session_id": sid, "target": "mongo_fields",
            "preview_token": prev_fields.json()["preview_token"],
            "confirm": True, "typed_confirmation": "DELETE",
            "row_ids": [0, 1], "fields": ["score"],
            "mongo_connection": {"uri": URI},
        },
    )
    assert del_fields.status_code == 200, del_fields.text
    assert "score" not in (coll.find_one({"_id": a}) or {})

    field_backup = _latest_backup(client, auth_headers, "mongo_field_values")

    # 2) bob's document vanishes completely (external change).
    coll.delete_one({"_id": b})

    # 3) Restore the field values: alice gets 'score' back, bob is missing.
    ok = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": field_backup, "confirm": True,
            "typed_confirmation": "RESTORE",
            "session_id": del_fields.json()["session_id"],
            "mongo_connection": {"uri": URI},
        },
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()

    # The outcome is partial and must NOT be presented as a clean success.
    assert body["verification"]["verified"] is False
    assert body["verification"]["found"] == 1
    assert body["verification"]["expected"] == 2
    joined = " ".join([body["message"], *body.get("warnings", [])]).lower()
    assert "discrepanc" in joined or "could not be verified" in joined or "partial" in joined

    # The refreshed preview reflects the ACTUAL source (1 doc), not the claim.
    assert body["session_id"]
    assert body["row_count"] == 1
    sp = _session_preview(client, auth_headers, body["session_id"])
    assert sp["row_count"] == 1
    assert coll.count_documents({}) == 1


# --------------------------------------------------------------------------- #
# MySQL restore (mocked driver — no live server on this machine)
# --------------------------------------------------------------------------- #
from tests.test_restore import FakeMySQL  # noqa: E402


class SneakyMySQL(FakeMySQL):
    """Reports success while silently dropping rows — verification must catch it."""

    def insert_rows(self, conn, database, table, columns, rows):
        super().insert_rows(conn, database, table, columns, rows[:0])
        return len(rows)


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
    resp = client.post(
        "/source_delete/mysql_rows/", headers=headers,
        json={
            "session_id": sid, "target": "mysql_rows",
            "preview_token": prev.json()["preview_token"], "confirm": True,
            "typed_confirmation": "DELETE",
            "row_ids": [0], "connection_details": CONN,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture()
def fake_mysql(monkeypatch):
    import app as app_module

    fake = FakeMySQL()
    fake.add_table(
        "db1", "people",
        columns=["id", "name"], pk=["id"],
        rows=[{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}],
    )
    monkeypatch.setattr(app_module, "db_mysql", fake)
    return fake


@pytest.fixture()
def sneaky_mysql(monkeypatch):
    import app as app_module

    fake = SneakyMySQL()
    fake.add_table(
        "db1", "people",
        columns=["id", "name"], pk=["id"],
        rows=[{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}],
    )
    monkeypatch.setattr(app_module, "db_mysql", fake)
    return fake


def test_mysql_restore_returns_fresh_session(client, auth_headers, fake_mysql):
    resp = client.post(
        "/get_columns/?table_name=people", headers=auth_headers, json=CONN,
    )
    assert resp.status_code == 200, resp.text
    sid = resp.json()["session_id"]

    out = _mysql_delete_rows(client, auth_headers, sid)
    assert out["affected"] == 1
    backup_id = _latest_backup(client, auth_headers, "mysql_rows")

    ok = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "RESTORE",
            "session_id": sid,
            "mysql_connection": CONN,
        },
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()

    assert set(fake_mysql.tables["db1.people"]["rows"]) == {(1,), (2,)}
    assert body["verification"]["verified"] is True
    assert body["session_id"], "restore must return a refreshed session"
    assert body["source"] == "mysql"
    assert body["source_name"] == "db1.people"
    assert body["row_count"] == 2

    sp = _session_preview(client, auth_headers, body["session_id"])
    assert sp["row_count"] == 2
    assert sp["source"] == "mysql"

    entries = _history(client, auth_headers, source_type="mysql")
    restore_entries = [e for e in entries if e["operation"] == "restore_source"]
    assert restore_entries and restore_entries[0]["session_id"] == body["session_id"]


def test_mysql_restore_partial_write_reported_honestly(
    client, auth_headers, sneaky_mysql
):
    resp = client.post(
        "/get_columns/?table_name=people", headers=auth_headers, json=CONN,
    )
    sid = resp.json()["session_id"]
    _mysql_delete_rows(client, auth_headers, sid)
    backup_id = _latest_backup(client, auth_headers, "mysql_rows")

    ok = client.post(
        "/restore_source/", headers=auth_headers,
        json={
            "backup_id": backup_id, "confirm": True,
            "typed_confirmation": "RESTORE",
            "session_id": sid,
            "mysql_connection": CONN,
        },
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()

    # The write lied (returned 1 inserted, actually wrote 0) — verification
    # re-reads the source and must refuse to claim a clean restore.  The
    # backup covers the single at-risk row (alice): expected 1, confirmed 0.
    assert body["verification"]["verified"] is False
    assert body["verification"]["found"] == 0
    assert body["verification"]["expected"] == 1
    joined = " ".join([body["message"], *body.get("warnings", [])]).lower()
    assert "discrepanc" in joined or "partial" in joined or "could not" in joined

    # The refreshed preview shows the ACTUAL table (bob only).
    assert body["row_count"] == 1
    sp = _session_preview(client, auth_headers, body["session_id"])
    assert sp["row_count"] == 1
