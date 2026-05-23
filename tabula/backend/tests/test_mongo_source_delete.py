"""Source deletion tests for MongoDB against the real local server.

Covers: document deletion by real ``_id`` (including after apply, which must
preserve identities), field ``$unset``, backups in Extended JSON, refresh into
a new session, stale tokens, zero matches, ambiguous identity refusal, and
working-copy isolation (edits never touch the source).
"""

from __future__ import annotations

import gzip
import json
import uuid

import pytest
from bson import ObjectId
from pymongo import MongoClient

URI = "mongodb://localhost:27017"

ALICE = ObjectId("68f0a1b2c3d4e5f60718293a")
BOB = ObjectId("68f0a1b2c3d4e5f60718293b")
CAROL = ObjectId("68f0a1b2c3d4e5f60718293c")


@pytest.fixture()
def mongo_db():
    mc = MongoClient(URI, serverSelectionTimeoutMS=3000)
    try:
        mc.admin.command("ping")
    except Exception:
        mc.close()
        pytest.skip("Local MongoDB not available")
    name = f"tabula_sd_{uuid.uuid4().hex[:8]}"
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


def _preview(client, headers, sid, target, **extra):
    return client.post(
        "/source_delete/preview/",
        headers=headers,
        json={"session_id": sid, "target": target, **extra},
    )


def _delete(client, headers, sid, target, token, **extra):
    return client.post(
        f"/source_delete/{target}/",
        headers=headers,
        json={
            "session_id": sid,
            "target": target,
            "preview_token": token,
            "confirm": True,
            "typed_confirmation": "DELETE",
            **extra,
        },
    )


def _seed(coll):
    coll.insert_many([
        {"_id": ALICE, "name": "alice", "score": 10, "extra": "x"},
        {"_id": BOB, "name": "bob", "score": 20, "extra": "y"},
        {"_id": CAROL, "name": "carol", "score": 30},
    ])


# --------------------------------------------------------------------------- #
# Document deletion (end to end)
# --------------------------------------------------------------------------- #
def test_mongo_document_delete_end_to_end(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["people"]
    _seed(coll)

    loaded = _connect(client, auth_headers, db_name)
    sid = loaded["session_id"]
    assert sid
    assert "_id" not in loaded["columns"], "_id must never be surfaced"
    assert all("_id" not in r for r in loaded["preview"])

    prev = _preview(
        client, auth_headers, sid, "mongo_documents",
        row_ids=[0, 1], mongo_connection={"uri": URI},
    )
    assert prev.status_code == 200, prev.text
    body = prev.json()
    assert body["matched_count"] == 2
    assert body["primary_key"] == ["_id"]
    assert body["backup"]["format"] == "jsonl"
    assert all("_id" not in r for r in body["sample"])

    # Typed-confirmation gate.
    wrong = client.post(
        "/source_delete/mongo_documents/", headers=auth_headers,
        json={
            "session_id": sid, "target": "mongo_documents",
            "preview_token": body["preview_token"], "confirm": True,
            "typed_confirmation": "delete",
            "row_ids": [0, 1], "mongo_connection": {"uri": URI},
        },
    )
    assert wrong.status_code == 400
    assert "DELETE" in wrong.json()["detail"]

    ok = _delete(
        client, auth_headers, sid, "mongo_documents", body["preview_token"],
        row_ids=[0, 1], mongo_connection={"uri": URI},
    )
    assert ok.status_code == 200, ok.text
    out = ok.json()
    assert out["affected"] == 2
    # Source really lost the documents.
    assert coll.count_documents({}) == 1
    assert coll.find_one({"_id": ALICE}) is None
    assert coll.find_one({"_id": CAROL}) is not None

    # New session mirrors the live source; old one is retired.
    assert out["session_id"] and out["session_id"] != sid
    assert out["row_count"] == 1
    assert [r["name"] for r in out["preview"]] == ["carol"]
    assert "_id" not in out["columns"]
    old = client.get(f"/session_preview/?session_id={sid}", headers=auth_headers)
    assert old.status_code == 404

    # Backup: Extended JSON lines that mongoimport can restore.
    backups = client.get("/backups/", headers=auth_headers).json()["backups"]
    bak = [b for b in backups if b["kind"] == "mongo_documents"][0]
    dl = client.get(f"/backups/{bak['backup_id']}/download/", headers=auth_headers)
    assert dl.status_code == 200
    lines = [
        ln for ln in gzip.decompress(dl.content).decode("utf-8").splitlines() if ln
    ]
    assert len(lines) == 2
    parsed = [json.loads(ln) for ln in lines]
    assert {"$oid": str(ALICE)} in [d["_id"] for d in parsed]
    assert all("name" in d for d in parsed)


def test_working_copy_delete_leaves_source_untouched(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["people"]
    _seed(coll)

    sid = _connect(client, auth_headers, db_name)["session_id"]
    resp = client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["removed"] == 1
    assert "_id" not in resp.json()["columns"], "_id must stay hidden in edit responses"
    # The live collection is NOT modified by working-copy edits.
    assert coll.count_documents({}) == 3


def test_apply_preserves_identities_and_hides_id(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["people"]
    _seed(coll)

    sid = _connect(client, auth_headers, db_name)["session_id"]

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
            "source_type": "mongodb_stored",
            "session_id": sid,
        },
    )
    assert ap.status_code == 200, ap.text
    assert all("_id" not in r for r in ap.json()["preview"]), "hide _id in response"
    assert "_id" not in ap.json()["transformed_columns"]
    assert [r["score"] for r in ap.json()["preview"]] == [100, 200, 300]

    # The persisted working copy still has identities: a later source delete
    # maps rows to REAL documents (this failed before _id was preserved).
    prev = _preview(
        client, auth_headers, sid, "mongo_documents",
        row_ids=[0], mongo_connection={"uri": URI},
    )
    assert prev.status_code == 200, prev.text
    assert prev.json()["matched_count"] == 1
    ok = _delete(
        client, auth_headers, sid, "mongo_documents", prev.json()["preview_token"],
        row_ids=[0], mongo_connection={"uri": URI},
    )
    assert ok.status_code == 200, ok.text
    assert coll.find_one({"_id": ALICE}) is None
    assert coll.count_documents({}) == 2


def test_stale_token_rejected_after_selection_change(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["people"]
    _seed(coll)

    sid = _connect(client, auth_headers, db_name)["session_id"]
    prev = _preview(
        client, auth_headers, sid, "mongo_documents",
        row_ids=[0], mongo_connection={"uri": URI},
    )
    assert prev.status_code == 200, prev.text

    resp = _delete(
        client, auth_headers, sid, "mongo_documents", prev.json()["preview_token"],
        row_ids=[1], mongo_connection={"uri": URI},  # different selection
    )
    assert resp.status_code == 409, resp.text
    assert "no longer matches" in resp.json()["detail"]
    assert coll.count_documents({}) == 3, "nothing may be deleted on a stale token"


def test_zero_match_after_concurrent_delete_409(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["people"]
    _seed(coll)

    sid = _connect(client, auth_headers, db_name)["session_id"]
    prev = _preview(
        client, auth_headers, sid, "mongo_documents",
        row_ids=[0], mongo_connection={"uri": URI},
    )
    coll.delete_one({"_id": ALICE})  # concurrent change in the source

    resp = _delete(
        client, auth_headers, sid, "mongo_documents", prev.json()["preview_token"],
        row_ids=[0], mongo_connection={"uri": URI},
    )
    assert resp.status_code == 409
    assert "Nothing was deleted" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Identity safety
# --------------------------------------------------------------------------- #
def test_mixed_id_types_refuse_source_deletion(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["mixed"]
    coll.insert_many([{"_id": ALICE, "n": 1}, {"_id": "string-id", "n": 2}])

    loaded = _connect(client, auth_headers, db_name, collection="mixed")
    sid = loaded["session_id"]
    prev = _preview(
        client, auth_headers, sid, "mongo_documents",
        row_ids=[0], mongo_connection={"uri": URI},
    )
    assert prev.status_code == 400, prev.text
    assert "unambiguously" in prev.json()["detail"]


def test_numeric_string_ids_disable_source_delete(client, auth_headers, mongo_db):
    """CSV round-trip would strip leading zeros -> refuse, never guess."""
    mc, db_name = mongo_db
    coll = mc[db_name]["numeric_ids"]
    coll.insert_many([{"_id": "00123", "n": 1}, {"_id": "00456", "n": 2}])

    loaded = _connect(client, auth_headers, db_name, collection="numeric_ids")
    sid = loaded["session_id"]
    prev = _preview(
        client, auth_headers, sid, "mongo_documents",
        row_ids=[0], mongo_connection={"uri": URI},
    )
    assert prev.status_code == 400, prev.text
    assert "round-trip" in prev.json()["detail"]
    # Working-copy edits stay available.
    edit = client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    assert edit.status_code == 200


# --------------------------------------------------------------------------- #
# Field removal ($unset)
# --------------------------------------------------------------------------- #
def test_mongo_field_removal_end_to_end(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    coll = mc[db_name]["people"]
    _seed(coll)

    loaded = _connect(client, auth_headers, db_name)
    sid = loaded["session_id"]
    assert "extra" in loaded["columns"]

    prev = _preview(
        client, auth_headers, sid, "mongo_fields",
        fields=["extra"], mongo_connection={"uri": URI},
    )
    assert prev.status_code == 200, prev.text
    body = prev.json()
    assert body["fields"] == ["extra"]
    assert body["field_counts"] == {"extra": 2}
    assert body["backup"]["kind"] == "mongo_field_values"

    ok = _delete(
        client, auth_headers, sid, "mongo_fields", body["preview_token"],
        fields=["extra"], mongo_connection={"uri": URI},
    )
    assert ok.status_code == 200, ok.text
    out = ok.json()
    assert out["affected"] == 2
    assert out["field_results"]["extra"]["modified"] == 2

    # Source truth: field gone where it existed, docs otherwise untouched.
    assert coll.count_documents({"extra": {"$exists": True}}) == 0
    assert coll.count_documents({}) == 3
    assert coll.find_one({"_id": CAROL, "name": "carol"}) is not None

    # Refreshed session no longer shows the column.
    assert "extra" not in out["columns"]
    assert out["row_count"] == 3

    # Backup holds the removed values.
    backups = client.get("/backups/", headers=auth_headers).json()["backups"]
    bak = [b for b in backups if b["kind"] == "mongo_field_values"][0]
    dl = client.get(f"/backups/{bak['backup_id']}/download/", headers=auth_headers)
    lines = [
        ln for ln in gzip.decompress(dl.content).decode("utf-8").splitlines() if ln
    ]
    assert len(lines) == 2
    assert all("extra" in json.loads(ln) for ln in lines)


def test_invalid_field_names_rejected(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    _seed(mc[db_name]["people"])
    sid = _connect(client, auth_headers, db_name)["session_id"]

    for bad in (["_id"], ["a.b"], ["$bad"], [""]):
        resp = _preview(
            client, auth_headers, sid, "mongo_fields",
            fields=bad, mongo_connection={"uri": URI},
        )
        assert resp.status_code == 400, (bad, resp.text)


def test_field_zero_match_409(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    _seed(mc[db_name]["people"])
    sid = _connect(client, auth_headers, db_name)["session_id"]

    prev = _preview(
        client, auth_headers, sid, "mongo_fields",
        fields=["nonexistent"], mongo_connection={"uri": URI},
    )
    assert prev.status_code == 200
    assert any("do not exist" in w for w in prev.json()["warnings"])

    resp = _delete(
        client, auth_headers, sid, "mongo_fields", prev.json()["preview_token"],
        fields=["nonexistent"], mongo_connection={"uri": URI},
    )
    assert resp.status_code == 409
    assert "Nothing was deleted" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Working-copy guards and target mismatches
# --------------------------------------------------------------------------- #
def test_working_copy_cannot_delete_id_column(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    _seed(mc[db_name]["people"])
    sid = _connect(client, auth_headers, db_name)["session_id"]

    resp = client.post(
        "/delete_columns/", headers=auth_headers,
        json={"session_id": sid, "columns": ["_id"]},
    )
    assert resp.status_code == 400
    assert "document identity" in resp.json()["detail"]


def test_mongo_session_cannot_use_mysql_target(client, auth_headers, mongo_db):
    mc, db_name = mongo_db
    _seed(mc[db_name]["people"])
    sid = _connect(client, auth_headers, db_name)["session_id"]

    resp = _preview(
        client, auth_headers, sid, "mysql_rows",
        row_ids=[0],
        mysql_connection={"host": "h", "user": "u", "password": "p",
                          "port": 3306, "database": "db"},
    )
    assert resp.status_code == 400
    assert "does not match" in resp.json()["detail"]
