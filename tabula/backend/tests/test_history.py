"""Transformation history + working-copy version rollback tests.

Covers: history entries recorded for import/edit/apply, version snapshots with
the seven-day window, working-copy restore to a snapshot (and that the source
DB is never touched), ownership enforcement, expired-version refusal, the
selectable-history workflow (download/open/restore-fallback), selective /
full history clearing (shared-snapshot retention, idempotency, typed
confirmation, partial-failure reporting), single-entry deletion
(``DELETE /history/{id}``: persistence, validation, 404 semantics, backup
cleanup, live-data safety), and source-type section filtering
(``?source_type=file|mysql|mongodb`` listing, validation/aliases, legacy
classification, owner scoping, category-scoped clear-all/stats, and source
preservation when opening a snapshot).
"""

from __future__ import annotations

import uuid

from tests.conftest import register_and_login


def _upload(client, headers, csv: str, name: str = "data.csv") -> str:
    resp = client.post(
        "/upload_file/",
        headers=headers,
        files={"file": (name, csv.encode(), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def _other_headers(client) -> dict:
    suffix = uuid.uuid4().hex[:10]
    token = register_and_login(client, {
        "username": f"other_{suffix}",
        "password": "Sup3rSecret!",
        "email": f"{suffix}@example.com",
        "full_name": "Other User",
    })
    return {"Authorization": f"Bearer {token}"}


def test_import_and_edit_record_history_and_versions(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)

    # Import recorded + v1 snapshot.
    hist = client.get(f"/history/?session_id={sid}", headers=auth_headers)
    assert hist.status_code == 200, hist.text
    entries = hist.json()["history"]
    assert any(e["operation"] == "import" for e in entries)
    assert hist.json()["recovery_window_days"] >= 1

    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    assert versions and versions[0]["version_number"] == 1
    assert versions[0]["restorable"] is True
    assert versions[0]["row_count"] == 3
    assert "name" in versions[0]["columns"]

    # A working-copy edit records an operation + a new version.
    resp = client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    assert resp.status_code == 200, resp.text

    entries = client.get(
        f"/history/?session_id={sid}", headers=auth_headers
    ).json()["history"]
    ops = [e["operation"] for e in entries]
    assert "working_copy_delete_rows" in ops
    delete_entry = next(e for e in entries if e["operation"] == "working_copy_delete_rows")
    assert delete_entry["restorable"] is True
    assert delete_entry["rollback_expires_at"]

    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    assert versions[0]["version_number"] == 2
    assert versions[0]["row_count"] == 2


def test_restore_working_copy_to_version(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0, 1]},
    )
    preview = client.get(
        f"/session_preview/?session_id={sid}", headers=auth_headers
    ).json()
    assert preview["row_count"] == 1

    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    v1 = [v for v in versions if v["version_number"] == 1][0]

    # Confirm gate.
    no_confirm = client.post(
        "/history/restore_working/", headers=auth_headers,
        json={"session_id": sid, "version_id": v1["version_id"], "confirm": False},
    )
    assert no_confirm.status_code == 400

    ok = client.post(
        "/history/restore_working/", headers=auth_headers,
        json={"session_id": sid, "version_id": v1["version_id"], "confirm": True},
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["row_count"] == 3
    assert body["restored_version_number"] == 1
    names = [r["name"] for r in body["preview"]]
    assert names == ["alice", "bob", "carol"]

    # The restore itself appears in history (roll-forward possible).
    entries = client.get(
        f"/history/?session_id={sid}", headers=auth_headers
    ).json()["history"]
    assert any(e["operation"] == "restore_working_copy" for e in entries)


def test_history_and_versions_require_ownership(client, auth_headers):
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    other = _other_headers(client)

    # Another user sees no history for this session and cannot restore.
    hist = client.get(f"/history/?session_id={sid}", headers=other)
    assert hist.status_code == 200
    assert hist.json()["history"] == []

    # Versions are ownership-filtered: another user gets an empty list, and
    # restoring a version id they cannot see is refused.
    own_versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=other
    )
    assert versions.status_code == 200
    assert versions.json()["versions"] == []
    if own_versions:
        forbidden = client.post(
            "/history/restore_working/", headers=other,
            json={
                "session_id": sid,
                "version_id": own_versions[0]["version_id"],
                "confirm": True,
            },
        )
        assert forbidden.status_code == 404

    # Unauthenticated requests are rejected.
    assert client.get("/history/").status_code == 401
    assert client.get(f"/history/versions/?session_id={sid}").status_code == 401


def test_expired_version_is_not_restorable(client, auth_headers):
    from pymongo import MongoClient

    import history as history_store

    sid = _upload(client, auth_headers, "a,b\n1,2\n3,4\n")
    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    target = versions[0]

    # Expire the snapshot directly (pymongo: motor is bound to the app loop).
    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        mc["tabula_test"]["session_versions"].update_one(
            {"version_id": target["version_id"]},
            {"$set": {"expires_at": history_store._now()}},
        )
    finally:
        mc.close()

    # Listed versions only include unexpired snapshots.
    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    assert all(v["version_id"] != target["version_id"] for v in versions)

    # Restoring the expired version is refused.
    resp = client.post(
        "/history/restore_working/", headers=auth_headers,
        json={"session_id": sid, "version_id": target["version_id"], "confirm": True},
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"].lower()
    assert "expired" in detail or "not found" in detail


# --------------------------------------------------------------------------- #
# Selectable history: version metadata, download, open, restore fallback
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.fixture()
def mongo_db():
    """Isolated local MongoDB database (skipped when the server is down)."""
    from pymongo import MongoClient

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        mc.admin.command("ping")
    except Exception:
        mc.close()
        pytest.skip("Local MongoDB not available")
    name = f"tabula_hist_{uuid.uuid4().hex[:8]}"
    yield mc, name
    mc.drop_database(name)
    mc.close()


def _expire_version(version_id: str) -> None:
    from pymongo import MongoClient

    import history as history_store

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        mc["tabula_test"]["session_versions"].update_one(
            {"version_id": version_id},
            {"$set": {"expires_at": history_store._now()}},
        )
    finally:
        mc.close()


def _delete_version_blob(version_id: str) -> None:
    """Simulate a lost snapshot artifact (metadata without data)."""
    from pymongo import MongoClient

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        doc = mc["tabula_test"]["session_versions"].find_one(
            {"version_id": version_id}
        )
        assert doc, "version doc missing"
        file_id = doc["file_id"]
        mc["tabula_test"]["datasets.files"].delete_one({"_id": file_id})
        mc["tabula_test"]["datasets.chunks"].delete_many({"files_id": file_id})
    finally:
        mc.close()


def _history_with_versions(client, headers, sid):
    resp = client.get(f"/history/?session_id={sid}", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["history"]


def test_history_entries_include_verified_version_metadata(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )

    entries = _history_with_versions(client, auth_headers, sid)
    import_entry = next(e for e in entries if e["operation"] == "import")
    delete_entry = next(
        e for e in entries if e["operation"] == "working_copy_delete_rows"
    )

    # Each entry links its snapshot with verified availability.
    assert import_entry["version"] is not None
    assert import_entry["version"]["version_number"] == 1
    assert import_entry["version"]["row_count"] == 3
    assert import_entry["version"]["restorable"] is True
    assert import_entry["version"]["snapshot_available"] is True
    assert delete_entry["version"]["version_number"] == 2
    assert delete_entry["version"]["row_count"] == 2

    # A lost blob downgrades restorable: metadata alone is never enough.
    _delete_version_blob(import_entry["version"]["version_id"])
    entries = _history_with_versions(client, auth_headers, sid)
    import_entry = next(e for e in entries if e["operation"] == "import")
    assert import_entry["version"]["snapshot_available"] is False
    assert import_entry["version"]["restorable"] is False


def test_download_version_returns_exact_snapshot(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    v1 = next(v for v in versions if v["version_number"] == 1)
    v2 = next(v for v in versions if v["version_number"] == 2)

    # CSV download of v1 is the ORIGINAL 3-row snapshot, not the 2-row
    # current dataset.
    r = client.get(
        f"/history/versions/{v1['version_id']}/download/", headers=auth_headers
    )
    assert r.status_code == 200, r.text
    lines = [ln for ln in r.text.strip().splitlines() if ln]
    assert len(lines) == 4  # header + 3 rows
    assert "alice" in r.text and "carol" in r.text
    assert "_v1.csv" in r.headers["Content-Disposition"]
    assert r.headers["X-Format-Converted"] == "false"

    # v2 is the post-delete 2-row snapshot.
    r2 = client.get(
        f"/history/versions/{v2['version_id']}/download/", headers=auth_headers
    )
    assert "alice" not in r2.text
    assert "bob" in r2.text and "carol" in r2.text

    # JSON export of the exact snapshot.
    r3 = client.get(
        f"/history/versions/{v1['version_id']}/download/?format=json",
        headers=auth_headers,
    )
    assert r3.status_code == 200, r3.text
    data = r3.json()
    assert len(data) == 3
    assert [d["name"] for d in data] == ["alice", "bob", "carol"]

    # XLSX export of the exact snapshot.
    r4 = client.get(
        f"/history/versions/{v1['version_id']}/download/?format=xlsx",
        headers=auth_headers,
    )
    assert r4.status_code == 200, r4.text
    assert r4.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument"
    )

    # Downloading never mutated the current working copy.
    preview = client.get(
        f"/session_preview/?session_id={sid}", headers=auth_headers
    ).json()
    assert preview["row_count"] == 2

    # Invalid format is rejected.
    bad = client.get(
        f"/history/versions/{v1['version_id']}/download/?format=pdf",
        headers=auth_headers,
    )
    assert bad.status_code == 422


def test_download_and_open_version_ownership_and_expiry(client, auth_headers):
    sid = _upload(client, auth_headers, "a,b\n1,2\n3,4\n")
    other = _other_headers(client)
    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    target = versions[0]
    vid = target["version_id"]

    # Another user cannot download or open this snapshot.
    assert (
        client.get(f"/history/versions/{vid}/download/", headers=other).status_code
        == 404
    )
    open_other = client.post(
        "/history/open_version/", headers=other, json={"version_id": vid}
    )
    assert open_other.status_code == 404

    # Unauthenticated requests are rejected.
    assert client.get(f"/history/versions/{vid}/download/").status_code == 401
    assert (
        client.post("/history/open_version/", json={"version_id": vid}).status_code
        == 401
    )

    # Expired snapshots are refused for download and open (server-side).
    _expire_version(vid)
    expired_dl = client.get(
        f"/history/versions/{vid}/download/", headers=auth_headers
    )
    assert expired_dl.status_code == 404
    assert "expired" in expired_dl.json()["detail"].lower()
    expired_open = client.post(
        "/history/open_version/", headers=auth_headers, json={"version_id": vid}
    )
    assert expired_open.status_code == 404
    assert "expired" in expired_open.json()["detail"].lower()


def test_open_version_creates_isolated_session(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    entries = _history_with_versions(client, auth_headers, sid)
    import_entry = next(e for e in entries if e["operation"] == "import")
    vid = import_entry["version"]["version_id"]

    resp = client.post(
        "/history/open_version/", headers=auth_headers,
        json={"version_id": vid, "history_id": import_entry["history_id"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    new_sid = body["session_id"]

    # Exact historical dataset (3 rows), not the current 2-row working copy.
    assert new_sid != sid
    assert body["row_count"] == 3
    assert [r["name"] for r in body["preview"]] == ["alice", "bob", "carol"]
    assert body["from_version_number"] == 1
    assert body["source"] == "file"

    # The original session is unchanged.
    old_preview = client.get(
        f"/session_preview/?session_id={sid}", headers=auth_headers
    ).json()
    assert old_preview["row_count"] == 2

    # The source snapshot remains available (historical reference intact).
    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    assert any(
        v["version_id"] == vid and v["restorable"] for v in versions
    ), "opening must not consume or modify the historical snapshot"

    # The open is recorded and the new session previews/downloads correctly.
    new_entries = _history_with_versions(client, auth_headers, new_sid)
    assert any(e["operation"] == "open_version" for e in new_entries)
    prev = client.get(
        f"/session_preview/?session_id={new_sid}", headers=auth_headers
    ).json()
    assert prev["row_count"] == 3
    assert prev["source_meta"]["history"]["version_number"] == 1

    # The new session's working copy can be edited independently.
    edit = client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": new_sid, "row_ids": [0]},
    )
    assert edit.status_code == 200, edit.text
    # ... and the historical snapshot is still the full 3 rows.
    dl = client.get(
        f"/history/versions/{vid}/download/", headers=auth_headers
    )
    assert len([ln for ln in dl.text.strip().splitlines() if ln]) == 4


def test_restore_working_when_session_gone_creates_new_session(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    v1 = next(v for v in versions if v["version_number"] == 1)

    # Delete the session (simulates expiry/cleanup) then restore from history.
    deleted = client.delete(f"/sessions/{sid}", headers=auth_headers)
    assert deleted.status_code == 200, deleted.text

    resp = client.post(
        "/history/restore_working/", headers=auth_headers,
        json={"session_id": sid, "version_id": v1["version_id"], "confirm": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["new_session"] is True
    assert body["session_id"] != sid
    assert body["row_count"] == 3
    assert [r["name"] for r in body["preview"]] == ["alice", "bob", "carol"]

    # The restored dataset is live in the new session.
    prev = client.get(
        f"/session_preview/?session_id={body['session_id']}", headers=auth_headers
    ).json()
    assert prev["row_count"] == 3
    entries = _history_with_versions(client, auth_headers, body["session_id"])
    assert any(e["operation"] == "restore_working_copy" for e in entries)


def test_mongo_version_download_json_preserves_ids(client, auth_headers, mongo_db):
    from bson import ObjectId

    URI = "mongodb://localhost:27017"
    mc, db_name = mongo_db
    alice = ObjectId()
    bob = ObjectId()
    mc[db_name]["people"].insert_many([
        {"_id": alice, "name": "alice", "score": 10},
        {"_id": bob, "name": "bob", "score": 20},
    ])

    loaded = client.post(
        f"/get_mongo_data/?database_name={db_name}&collection_name=people",
        headers=auth_headers,
        json={"uri": URI},
    )
    assert loaded.status_code == 200, loaded.text
    sid = loaded.json()["session_id"]

    # The snapshot hides _id from the interactive preview...
    assert all("_id" not in r for r in loaded.json()["preview"])

    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    assert versions, "mongo session should have a v1 snapshot"
    vid = versions[0]["version_id"]

    # ...but the JSON recovery export preserves each document's _id so the
    # snapshot can be used to restore document identity.
    r = client.get(
        f"/history/versions/{vid}/download/?format=json", headers=auth_headers
    )
    assert r.status_code == 200, r.text
    docs = r.json()
    assert len(docs) == 2
    ids = {d.get("_id") for d in docs}
    assert ids == {str(alice), str(bob)}
    assert all("name" in d and "score" in d for d in docs)


# --------------------------------------------------------------------------- #
# Selective / full history clearing
# --------------------------------------------------------------------------- #
def _owner_oid(client, headers, sid: str):
    """Read the ObjectId owner of a session's history entries via pymongo."""
    from bson import ObjectId
    from pymongo import MongoClient

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        doc = mc["tabula_test"]["history"].find_one({"session_id": sid})
        assert doc, "expected at least one history entry for the session"
        return doc["owner_id"]
    finally:
        mc.close()


def _version_doc(version_id: str):
    from pymongo import MongoClient

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        return mc["tabula_test"]["session_versions"].find_one(
            {"version_id": version_id}
        )
    finally:
        mc.close()


def _blob_exists(file_id) -> bool:
    from pymongo import MongoClient

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        return mc["tabula_test"]["datasets.files"].find_one(
            {"_id": file_id}
        ) is not None
    finally:
        mc.close()


def test_clear_selected_history_removes_entry_and_snapshot(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    entries = _history_with_versions(client, auth_headers, sid)
    import_entry = next(e for e in entries if e["operation"] == "import")
    delete_entry = next(
        e for e in entries if e["operation"] == "working_copy_delete_rows"
    )
    vid = import_entry["version"]["version_id"]
    vdoc = _version_doc(vid)
    assert vdoc and _blob_exists(vdoc["file_id"])

    # Confirmation gate.
    no_confirm = client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [import_entry["history_id"]], "confirm": False},
    )
    assert no_confirm.status_code == 400

    resp = client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [import_entry["history_id"]], "confirm": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_deleted"] == 1
    assert body["snapshots_deleted"] == 1
    assert body["entries_missing"] == []
    assert body["failures"] == []
    assert body["partial"] is False
    assert body["message"]

    # The entry is gone from history; the other entry is untouched.
    remaining = _history_with_versions(client, auth_headers, sid)
    ids = [e["history_id"] for e in remaining]
    assert import_entry["history_id"] not in ids
    assert delete_entry["history_id"] in ids

    # The snapshot (version doc + GridFS blob) was really deleted.
    assert _version_doc(vid) is None
    assert not _blob_exists(vdoc["file_id"])

    # The surviving entry's snapshot still downloads.
    dl = client.get(
        f"/history/versions/{delete_entry['version']['version_id']}/download/",
        headers=auth_headers,
    )
    assert dl.status_code == 200, dl.text

    # The active working dataset is untouched.
    preview = client.get(
        f"/session_preview/?session_id={sid}", headers=auth_headers
    ).json()
    assert preview["row_count"] == 2


def test_clear_selected_history_multiple_unknown_and_duplicate_ids(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\n"
    sid = _upload(client, auth_headers, csv)
    entries = _history_with_versions(client, auth_headers, sid)
    real_id = entries[0]["history_id"]

    resp = client.post(
        "/history/clear/", headers=auth_headers,
        json={
            "history_ids": [real_id, real_id, "does-not-exist"],
            "confirm": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Duplicates are deduplicated; unknown ids are reported, never deleted.
    assert body["requested"] == 2
    assert body["entries_deleted"] == 1
    assert body["entries_missing"] == ["does-not-exist"]
    assert body["failures"] == []

    # Idempotent: clearing the same (already deleted) id reports missing.
    again = client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [real_id], "confirm": True},
    )
    assert again.status_code == 200
    assert again.json()["entries_deleted"] == 0
    assert again.json()["entries_missing"] == [real_id]


def test_clear_selected_history_retains_shared_snapshot(client, auth_headers):
    """A snapshot still referenced by another surviving entry is retained."""
    from pymongo import MongoClient

    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    entries = _history_with_versions(client, auth_headers, sid)
    import_entry = next(e for e in entries if e["operation"] == "import")
    vid = import_entry["version"]["version_id"]
    vdoc = _version_doc(vid)
    assert vdoc and _blob_exists(vdoc["file_id"])

    # Make a second entry reference the same version snapshot (schema allows
    # shared references; the clear path must not delete under it).
    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        other_id = "shared-ref-" + import_entry["history_id"]
        mc["tabula_test"]["history"].insert_one({
            "history_id": other_id,
            "owner_id": _owner_oid(client, auth_headers, sid),
            "session_id": sid,
            "operation": "import",
            "status": "success",
            "timestamp": import_entry["timestamp"],
            "summary": "second entry sharing the same snapshot",
            "version_id": vid,
            "restorable": True,
            "warnings": [],
            "extra": {},
        })
    finally:
        mc.close()

    resp = client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [import_entry["history_id"]], "confirm": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_deleted"] == 1
    assert body["snapshots_deleted"] == 0
    assert body["snapshots_retained"] == 1

    # Snapshot is still there because the other entry references it.
    assert _version_doc(vid) is not None
    assert _blob_exists(vdoc["file_id"])

    # Clearing the remaining sharer deletes the snapshot for real.
    resp2 = client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [other_id], "confirm": True},
    )
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["snapshots_deleted"] == 1
    assert _version_doc(vid) is None
    assert not _blob_exists(vdoc["file_id"])


def test_clear_selected_history_missing_snapshot_is_idempotent(client, auth_headers):
    """A version whose blob was already lost is cleaned up without failure."""
    csv = "name,score\nalice,10\nbob,20\n"
    sid = _upload(client, auth_headers, csv)
    entries = _history_with_versions(client, auth_headers, sid)
    entry = entries[0]
    vid = entry["version"]["version_id"]
    vdoc = _version_doc(vid)
    assert vdoc
    # Simulate a lost blob (metadata remains).
    _delete_version_blob(vid)
    assert not _blob_exists(vdoc["file_id"])

    resp = client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [entry["history_id"]], "confirm": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_deleted"] == 1
    # The version doc is removed even though the blob was already gone.
    assert body["snapshots_deleted"] == 1
    assert body["failures"] == []
    assert _version_doc(vid) is None


def test_clear_selected_history_requires_ownership(client, auth_headers):
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    entries = _history_with_versions(client, auth_headers, sid)
    victim_id = entries[0]["history_id"]
    other = _other_headers(client)

    # Another user cannot delete someone else's history (reported missing).
    resp = client.post(
        "/history/clear/", headers=other,
        json={"history_ids": [victim_id], "confirm": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["entries_deleted"] == 0
    assert body["entries_missing"] == [victim_id]

    # The victim's entry is still there.
    remaining = _history_with_versions(client, auth_headers, sid)
    assert any(e["history_id"] == victim_id for e in remaining)

    # Unauthenticated requests are rejected.
    assert (
        client.post(
            "/history/clear/", json={"history_ids": [victim_id], "confirm": True}
        ).status_code
        == 401
    )
    assert client.get("/history/stats/").status_code == 401


def test_clear_all_history_requires_typed_confirmation(client, auth_headers):
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )

    # Missing confirm flag.
    r1 = client.post(
        "/history/clear_all/", headers=auth_headers,
        json={"confirm": False, "typed_confirmation": "CLEAR HISTORY"},
    )
    assert r1.status_code == 400

    # Wrong typed phrase (exact match required, case sensitive).
    r2 = client.post(
        "/history/clear_all/", headers=auth_headers,
        json={"confirm": True, "typed_confirmation": "clear history"},
    )
    assert r2.status_code == 400
    assert "CLEAR HISTORY" in r2.json()["detail"]

    # History is untouched while confirmations fail.
    entries = _history_with_versions(client, auth_headers, sid)
    assert len(entries) >= 2

    # Correct typed confirmation clears everything for this owner.
    r3 = client.post(
        "/history/clear_all/", headers=auth_headers,
        json={"confirm": True, "typed_confirmation": "CLEAR HISTORY"},
    )
    assert r3.status_code == 200, r3.text
    body = r3.json()
    assert body["entries_deleted"] >= 2
    assert body["failures"] == []

    # All entries and their snapshots are gone.
    assert _history_with_versions(client, auth_headers, sid) == []
    versions = client.get(
        f"/history/versions/?session_id={sid}", headers=auth_headers
    ).json()["versions"]
    assert versions == []

    # The active working dataset and the live session are untouched.
    preview = client.get(
        f"/session_preview/?session_id={sid}", headers=auth_headers
    ).json()
    # The only row was deleted pre-clear; the session itself still exists.
    assert preview["row_count"] == 0

    # Stats now report zero.
    stats = client.get("/history/stats/", headers=auth_headers)
    assert stats.status_code == 200
    assert stats.json()["total_entries"] == 0

    # Other users' history is unaffected.
    other = _other_headers(client)
    other_sid = _upload(client, other, "x,y\n9,8\n")
    client.post(
        "/history/clear_all/", headers=auth_headers,
        json={"confirm": True, "typed_confirmation": "CLEAR HISTORY"},
    )
    other_hist = client.get(f"/history/?session_id={other_sid}", headers=other)
    assert len(other_hist.json()["history"]) == 1

    # Unauthenticated requests are rejected.
    assert (
        client.post(
            "/history/clear_all/",
            json={"confirm": True, "typed_confirmation": "CLEAR HISTORY"},
        ).status_code
        == 401
    )


def test_clear_all_history_preserves_live_database_and_backups(client, auth_headers):
    """Clear-all never touches live DBs; unshared backups are cleaned, shared
    ones retained."""
    from datetime import datetime, timezone

    from pymongo import MongoClient

    csv = "name,score\nalice,10\nbob,20\n"
    sid = _upload(client, auth_headers, csv)
    owner = _owner_oid(client, auth_headers, sid)

    # Simulate an entry with a live-source backup link (like a DB delete op).
    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        db = mc["tabula_test"]
        # A real GridFS backup blob + metadata doc, like backups.create_backup.
        from gridfs import GridFS

        fs = GridFS(db, collection="backups")
        payload = b"fake,backup\n1,2\n"
        file_id = fs.put(payload, metadata={"owner_id": owner, "kind": "mysql_rows"})
        backup_id = "bk-" + sid[:8]
        db["delete_backups"].insert_one({
            "backup_id": backup_id,
            "owner_id": owner,
            "kind": "mysql_rows",
            "source_name": "customers",
            "size_bytes": len(payload),
            "file_id": file_id,
            "format": "csv",
            "encoding": "gzip",
            "meta": {},
            "restoration": "manual",
        })
        # A history entry that owns this backup (plus a second entry that
        # SHARES the same backup id â€” it must be retained).
        backup_entry_id = "hist-bk-" + sid[:8]
        shared_entry_id = "hist-bk-shared-" + sid[:8]
        now = datetime.now(timezone.utc)
        expires = datetime(2099, 1, 1, tzinfo=timezone.utc)
        base = {
            "owner_id": owner,
            "session_id": sid,
            "operation": "mysql_delete_rows",
            "status": "success",
            "timestamp": now,
            "summary": "live db op with backup",
            "backup_id": backup_id,
            "restorable": True,
            "rollback_expires_at": expires,
            "warnings": [],
            "extra": {},
        }
        db["history"].insert_one({**base, "history_id": backup_entry_id})
        db["history"].insert_one({**base, "history_id": shared_entry_id})
    finally:
        mc.close()

    # Clear only the first entry: the shared backup is retained.
    resp = client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [backup_entry_id], "confirm": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["backups_deleted"] == 0
    assert body["backups_retained"] == 1

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        assert mc["tabula_test"]["delete_backups"].find_one(
            {"backup_id": backup_id}
        ) is not None
    finally:
        mc.close()

    # Clear the sharer too: the backup is now deleted (doc + blob).
    resp2 = client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [shared_entry_id], "confirm": True},
    )
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["backups_deleted"] == 1

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        db = mc["tabula_test"]
        assert db["delete_backups"].find_one({"backup_id": backup_id}) is None
        assert db["backups.files"].find_one({"_id": file_id}) is None
        assert db["backups.chunks"].find_one({"files_id": file_id}) is None
    finally:
        mc.close()

    # Clear-all leaves other users alone and never touches the live session.
    other = _other_headers(client)
    other_sid = _upload(client, other, "x,y\n9,8\n")
    client.post(
        "/history/clear_all/", headers=auth_headers,
        json={"confirm": True, "typed_confirmation": "CLEAR HISTORY"},
    )
    # The caller's own working dataset still previews fine.
    preview = client.get(
        f"/session_preview/?session_id={sid}", headers=auth_headers
    ).json()
    assert preview["row_count"] == 2
    # The other user's history is intact.
    other_hist = client.get(f"/history/?session_id={other_sid}", headers=other)
    assert len(other_hist.json()["history"]) == 1


def test_clear_all_history_partial_failure_is_reported(client, auth_headers, monkeypatch):
    """A snapshot that cannot be removed is reported, never silent success."""
    csv = "name,score\nalice,10\nbob,20\n"
    sid = _upload(client, auth_headers, csv)
    entries = _history_with_versions(client, auth_headers, sid)
    vid = entries[0]["version"]["version_id"]

    import history as history_store

    original = history_store.delete_version_artifact

    async def flaky(owner_id, version_id):
        if version_id == vid:
            raise history_store.HistoryError("simulated blob failure", 500)
        return await original(owner_id, version_id)

    monkeypatch.setattr(history_store, "delete_version_artifact", flaky)

    resp = client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [e["history_id"] for e in entries], "confirm": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The failing snapshot is reported; the entry itself is still cleared.
    assert any(
        f["target"] == "version" and f["id"] == vid for f in body["failures"]
    )
    assert body["partial"] is True
    assert body["entries_deleted"] == len(entries)
    # The snapshot that failed to delete is still present (not falsely claimed
    # as deleted).
    assert _version_doc(vid) is not None


def test_history_after_clear_still_supports_open_download_restore(client, auth_headers):
    """After clearing some entries, the surviving ones still open/download/
    restore normally (clear never breaks the remaining workflow)."""
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    entries = _history_with_versions(client, auth_headers, sid)
    import_entry = next(e for e in entries if e["operation"] == "import")
    delete_entry = next(
        e for e in entries if e["operation"] == "working_copy_delete_rows"
    )

    # Clear the import entry (and its v1 snapshot).
    client.post(
        "/history/clear/", headers=auth_headers,
        json={"history_ids": [import_entry["history_id"]], "confirm": True},
    )

    # The surviving delete entry can still be opened, downloaded, restored.
    vid = delete_entry["version"]["version_id"]
    dl = client.get(
        f"/history/versions/{vid}/download/", headers=auth_headers
    )
    assert dl.status_code == 200, dl.text
    opened = client.post(
        "/history/open_version/", headers=auth_headers,
        json={"version_id": vid, "history_id": delete_entry["history_id"]},
    )
    assert opened.status_code == 200, opened.text
    restored = client.post(
        "/history/restore_working/", headers=auth_headers,
        json={"session_id": sid, "version_id": vid, "confirm": True},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["row_count"] == 2


# ---------------------------------------------------------------------------
# Single-entry deletion (DELETE /history/{history_id})
# ---------------------------------------------------------------------------

def _history_ids_pymongo(owner, sid=None):
    """Owner's history ids straight from MongoDB (not the API)."""
    from pymongo import MongoClient

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        q = {"owner_id": owner}
        if sid:
            q["session_id"] = sid
        return [
            d["history_id"]
            for d in mc["tabula_test"]["history"].find(q, {"history_id": 1})
        ]
    finally:
        mc.close()


def test_delete_single_entry_removes_entry_and_snapshot_persistently(
    client, auth_headers
):
    """DELETE removes the entry + its snapshot; it stays gone after refresh."""
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    entries = _history_with_versions(client, auth_headers, sid)
    import_entry = next(e for e in entries if e["operation"] == "import")
    keep_entry = next(
        e for e in entries if e["operation"] == "working_copy_delete_rows"
    )
    hid = import_entry["history_id"]
    vid = import_entry["version"]["version_id"]
    vdoc = _version_doc(vid)
    assert vdoc and _blob_exists(vdoc["file_id"])
    owner = _owner_oid(client, auth_headers, sid)

    resp = client.delete(f"/history/{hid}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deleted"] is True
    assert body["entries_deleted"] == 1
    assert body["entries_missing"] == []
    assert body["snapshots_deleted"] == 1
    assert body["backups_deleted"] == 0
    assert body["failures"] == []
    assert body["partial"] is False
    assert body["message"]

    # Authoritative: gone from the API list AND from MongoDB itself.
    remaining = _history_with_versions(client, auth_headers, sid)
    ids = [e["history_id"] for e in remaining]
    assert hid not in ids
    assert keep_entry["history_id"] in ids
    assert hid not in _history_ids_pymongo(owner, sid)

    # The snapshot (version doc + GridFS blob) was really deleted.
    assert _version_doc(vid) is None
    assert not _blob_exists(vdoc["file_id"])

    # Simulates the UI's post-delete refresh: still absent (no recreation).
    refreshed = client.get(f"/history/?session_id={sid}", headers=auth_headers)
    assert hid not in [e["history_id"] for e in refreshed.json()["history"]]

    # Repeating the delete reports 404 and changes nothing else.
    again = client.delete(f"/history/{hid}", headers=auth_headers)
    assert again.status_code == 404
    assert keep_entry["history_id"] in _history_ids_pymongo(owner, sid)

    # The surviving entry's snapshot still downloads.
    dl = client.get(
        f"/history/versions/{keep_entry['version']['version_id']}/download/",
        headers=auth_headers,
    )
    assert dl.status_code == 200, dl.text

    # The active working dataset is untouched (row deleted BEFORE the clear).
    preview = client.get(
        f"/session_preview/?session_id={sid}", headers=auth_headers
    ).json()
    assert preview["row_count"] == 2


def test_delete_single_entry_invalid_ids_rejected(client, auth_headers):
    """Malformed ids never reach Mongo: they fail fast with 400."""
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    entries = _history_with_versions(client, auth_headers, sid)
    hid = entries[0]["history_id"]

    bad_ids = ["   ", "has space", "bang!", "x" * 65, "dot.name"]
    for bad in bad_ids:
        resp = client.delete(f"/history/{bad}", headers=auth_headers)
        assert resp.status_code == 400, (bad, resp.status_code, resp.text)

    # The real entry was untouched by any malformed attempt.
    assert hid in _history_ids_pymongo(
        _owner_oid(client, auth_headers, sid), sid
    )


def test_delete_single_entry_unknown_missing_and_ownership(
    client, auth_headers
):
    """Unknown/cross-user/repeat deletes are 404 and never leak or mutate."""
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    entries = _history_with_versions(client, auth_headers, sid)
    victim_id = entries[0]["history_id"]
    other = _other_headers(client)

    # A random well-formed id that does not exist -> 404.
    ghost = "deadbeef" * 4
    assert client.delete(f"/history/{ghost}", headers=auth_headers).status_code == 404

    # Another user cannot delete someone else's entry -> 404, victim intact.
    resp = client.delete(f"/history/{victim_id}", headers=other)
    assert resp.status_code == 404
    assert victim_id in _history_ids_pymongo(
        _owner_oid(client, auth_headers, sid), sid
    )

    # Unauthenticated requests are rejected.
    assert client.delete(f"/history/{victim_id}").status_code == 401

    # A foreign user repeating their own (already failed) delete still 404s.
    assert client.delete(f"/history/{victim_id}", headers=other).status_code == 404

    # The owner's delete succeeds exactly once, then 404 on repeat.
    first = client.delete(f"/history/{victim_id}", headers=auth_headers)
    assert first.status_code == 200, first.text
    assert first.json()["deleted"] is True
    assert client.delete(f"/history/{victim_id}", headers=auth_headers).status_code == 404


def test_delete_single_entry_retains_shared_snapshot_then_frees_it(
    client, auth_headers
):
    """Shared snapshots survive the first delete; freed on the last one."""
    from datetime import datetime, timezone

    from pymongo import MongoClient

    csv = "name,score\nalice,10\nbob,20\n"
    sid = _upload(client, auth_headers, csv)
    entries = _history_with_versions(client, auth_headers, sid)
    import_entry = next(e for e in entries if e["operation"] == "import")
    vid = import_entry["version"]["version_id"]
    vdoc = _version_doc(vid)
    assert vdoc and _blob_exists(vdoc["file_id"])
    owner = _owner_oid(client, auth_headers, sid)

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        other_id = "shared-single-" + import_entry["history_id"]
        mc["tabula_test"]["history"].insert_one({
            "history_id": other_id,
            "owner_id": owner,
            "session_id": sid,
            "operation": "import",
            "status": "success",
            "timestamp": import_entry["timestamp"],
            "summary": "second entry sharing the same snapshot",
            "version_id": vid,
            "restorable": True,
            "warnings": [],
            "extra": {},
        })
    finally:
        mc.close()

    resp = client.delete(f"/history/{import_entry['history_id']}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_deleted"] == 1
    assert body["snapshots_deleted"] == 0
    assert body["snapshots_retained"] == 1

    # Snapshot still there because the surviving sharer references it.
    assert _version_doc(vid) is not None
    assert _blob_exists(vdoc["file_id"])

    # Deleting the last sharer removes the snapshot for real.
    resp2 = client.delete(f"/history/{other_id}", headers=auth_headers)
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["snapshots_deleted"] == 1
    assert _version_doc(vid) is None
    assert not _blob_exists(vdoc["file_id"])


def _insert_backup_history_entry(sid, owner, backup_id, hid):
    """Insert a history entry linked to a real GridFS backup (test fixture)."""
    from datetime import datetime, timezone

    from gridfs import GridFS
    from pymongo import MongoClient

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        db = mc["tabula_test"]
        fs = GridFS(db, collection="backups")
        payload = b"fake,backup\n1,2\n"
        file_id = fs.put(payload, metadata={"owner_id": owner, "kind": "mysql_rows"})
        db["delete_backups"].insert_one({
            "backup_id": backup_id,
            "owner_id": owner,
            "kind": "mysql_rows",
            "source_name": "customers",
            "size_bytes": len(payload),
            "file_id": file_id,
            "format": "csv",
            "encoding": "gzip",
            "meta": {},
            "restoration": "manual",
        })
        db["history"].insert_one({
            "history_id": hid,
            "owner_id": owner,
            "session_id": sid,
            "operation": "mysql_delete_rows",
            "status": "success",
            "timestamp": datetime.now(timezone.utc),
            "summary": "live db op with backup",
            "backup_id": backup_id,
            "restorable": True,
            "rollback_expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc),
            "warnings": [],
            "extra": {},
        })
        return file_id
    finally:
        mc.close()


def test_delete_single_entry_cleans_unreferenced_backup(client, auth_headers):
    """An unshared source-restore backup (doc + GridFS) is removed too."""
    from pymongo import MongoClient

    sid = _upload(client, auth_headers, "name,score\nalice,10\n")
    owner = _owner_oid(client, auth_headers, sid)
    hid = "hist-bk-single-" + sid[:8]
    backup_id = "bk-single-" + sid[:8]
    file_id = _insert_backup_history_entry(sid, owner, backup_id, hid)

    resp = client.delete(f"/history/{hid}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deleted"] is True
    assert body["backups_deleted"] == 1

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        db = mc["tabula_test"]
        assert db["history"].find_one({"history_id": hid}) is None
        assert db["delete_backups"].find_one({"backup_id": backup_id}) is None
        assert db["backups.files"].find_one({"_id": file_id}) is None
        assert db["backups.chunks"].find_one({"files_id": file_id}) is None
    finally:
        mc.close()


def test_delete_single_entry_retains_shared_backup(client, auth_headers):
    """A backup referenced by another surviving entry is retained."""
    from datetime import datetime, timezone

    from pymongo import MongoClient

    sid = _upload(client, auth_headers, "name,score\nalice,10\n")
    owner = _owner_oid(client, auth_headers, sid)
    hid_a = "hist-bk-a-" + sid[:8]
    hid_b = "hist-bk-b-" + sid[:8]
    backup_id = "bk-shared-" + sid[:8]
    _insert_backup_history_entry(sid, owner, backup_id, hid_a)

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        mc["tabula_test"]["history"].insert_one({
            "history_id": hid_b,
            "owner_id": owner,
            "session_id": sid,
            "operation": "mysql_delete_rows",
            "status": "success",
            "timestamp": datetime.now(timezone.utc),
            "summary": "shares the same backup",
            "backup_id": backup_id,
            "restorable": True,
            "rollback_expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc),
            "warnings": [],
            "extra": {},
        })
    finally:
        mc.close()

    resp = client.delete(f"/history/{hid_a}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["backups_deleted"] == 0
    assert body["backups_retained"] == 1

    from pymongo import MongoClient as _MC

    mc2 = _MC("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        assert mc2["tabula_test"]["delete_backups"].find_one(
            {"backup_id": backup_id}
        ) is not None
        assert mc2["tabula_test"]["history"].find_one({"history_id": hid_b})
    finally:
        mc2.close()


def test_delete_single_entry_partial_backup_failure_is_reported(
    client, auth_headers, monkeypatch
):
    """A backup that cannot be removed is reported; never silent success."""
    from pymongo import MongoClient

    sid = _upload(client, auth_headers, "name,score\nalice,10\n")
    owner = _owner_oid(client, auth_headers, sid)
    hid = "hist-bk-flaky-" + sid[:8]
    backup_id = "bk-flaky-" + sid[:8]
    file_id = _insert_backup_history_entry(sid, owner, backup_id, hid)

    import backups as backup_store

    async def flaky(owner_id, bid):
        if bid == backup_id:
            raise backup_store.BackupError("simulated backup failure", 500)
        return await backup_store.delete_backup(owner_id, bid)

    monkeypatch.setattr(backup_store, "delete_backup", flaky)

    resp = client.delete(f"/history/{hid}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deleted"] is True
    assert body["entries_deleted"] == 1
    assert body["backups_deleted"] == 0
    assert body["partial"] is True
    assert any(
        f["target"] == "backup" and f["id"] == backup_id
        for f in body["failures"]
    )

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        db = mc["tabula_test"]
        # The history entry is gone (its removal is not held hostage by the
        # artifact failure) but the unrecoverable backup is honestly still
        # there and reported for retry.
        assert db["history"].find_one({"history_id": hid}) is None
        assert db["delete_backups"].find_one({"backup_id": backup_id}) is not None
        assert db["backups.files"].find_one({"_id": file_id}) is not None
    finally:
        mc.close()


def test_delete_single_entry_never_touches_live_data(client, auth_headers):
    """Deleting history never modifies sessions, working data, or sources."""
    from pymongo import MongoClient

    csv = "name,score\nalice,10\nbob,20\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    entries = _history_with_versions(client, auth_headers, sid)
    owner = _owner_oid(client, auth_headers, sid)

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        db = mc["tabula_test"]
        session_doc = db["sessions"].find_one({"session_id": sid})
        assert session_doc
        datasets_before = db["datasets.files"].count_documents(
            {
                "metadata.session_id": sid,
                "metadata.kind": {"$in": ["original", "transformed"]},
            }
        )
        assert datasets_before >= 2  # original + transformed blobs
    finally:
        mc.close()

    for e in entries:
        resp = client.delete(f"/history/{e['history_id']}", headers=auth_headers)
        assert resp.status_code == 200, resp.text

    assert _history_ids_pymongo(owner) == []

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        db = mc["tabula_test"]
        # The live session doc and its working GridFS blobs are untouched.
        still = db["sessions"].find_one({"session_id": sid})
        assert still is not None
        assert still["row_count"] == session_doc["row_count"]
        assert still["original_file_id"] == session_doc["original_file_id"]
        assert still["transformed_file_id"] == session_doc["transformed_file_id"]
        assert db["datasets.files"].count_documents(
            {
                "metadata.session_id": sid,
                "metadata.kind": {"$in": ["original", "transformed"]},
            }
        ) == datasets_before
    finally:
        mc.close()

    # And the app still previews the same working copy.
    preview = client.get(
        f"/session_preview/?session_id={sid}", headers=auth_headers
    ).json()
    assert preview["row_count"] == 1


# ---------------------------------------------------------------------------
# Source-type filtering (Files / MySQL / MongoDB sections)
# ---------------------------------------------------------------------------

def _insert_typed_entry(
    owner,
    sid,
    hid,
    *,
    source_type=None,
    source_meta=None,
    operation="op",
    version_id=None,
    restorable=False,
    summary="fixture entry",
):
    """Insert one raw history doc via pymongo (category fixtures)."""
    from datetime import datetime, timezone

    from pymongo import MongoClient

    doc = {
        "history_id": hid,
        "owner_id": owner,
        "session_id": sid,
        "operation": operation,
        "status": "success",
        "timestamp": datetime.now(timezone.utc),
        "summary": summary,
        "restorable": bool(restorable),
        "warnings": [],
        "extra": {},
    }
    if source_type is not None:
        doc["source_type"] = source_type
    if source_meta is not None:
        doc["source_meta"] = source_meta
    if version_id is not None:
        doc["version_id"] = version_id
        doc["rollback_expires_at"] = datetime(2099, 1, 1, tzinfo=timezone.utc)
    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        mc["tabula_test"]["history"].insert_one(doc)
    finally:
        mc.close()


def _insert_version_artifact(sid, owner, hid, vid="vfx1"):
    """Insert a real session_versions doc + gzipped GridFS blob (fixture)."""
    import gzip as _gzip
    from datetime import datetime, timezone

    from gridfs import GridFS
    from pymongo import MongoClient

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        db = mc["tabula_test"]
        fs = GridFS(db, collection="datasets")
        payload = _gzip.compress(b"id,val\n1,2\n")
        file_id = fs.put(
            payload,
            metadata={
                "session_id": sid,
                "kind": "version",
                "owner_id": owner,
                "version_id": vid,
            },
        )
        now = datetime.now(timezone.utc)
        db["session_versions"].insert_one({
            "version_id": vid,
            "owner_id": owner,
            "session_id": sid,
            "version_number": 1,
            "history_id": hid,
            "label": None,
            "created_at": now,
            "expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc),
            "file_id": file_id,
            "size_bytes": len(payload),
            "row_count": 2,
            "columns": ["id", "val"],
        })
        return file_id
    finally:
        mc.close()


def test_history_filter_returns_only_active_source_category(client, auth_headers):
    """Files / MySQL / MongoDB listings each return only their category."""
    from pymongo import MongoClient

    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    owner = _owner_oid(client, auth_headers, sid)
    mysql_id = "fx-mysql-" + sid[:8]
    mongo_id = "fx-mongo-" + sid[:8]
    _insert_typed_entry(
        owner, sid, mysql_id, source_type="mysql",
        source_meta={"database": "sales", "table": "customers"},
        operation="write_back_mysql",
    )
    _insert_typed_entry(
        owner, sid, mongo_id, source_type="mongodb",
        source_meta={"database": "sales", "collection": "customers"},
        operation="write_back_mongodb",
    )

    # Unfiltered listing keeps every entry (audit trail unchanged).
    all_entries = client.get("/history/", headers=auth_headers).json()["history"]
    all_ids = {e["history_id"] for e in all_entries}
    assert {mysql_id, mongo_id} <= all_ids

    def ids_for(source):
        resp = client.get(
            f"/history/?source_type={source}", headers=auth_headers
        )
        assert resp.status_code == 200, resp.text
        return {e["history_id"] for e in resp.json()["history"]}

    file_ids = ids_for("file")
    # The two real file operations are file entries; DB entries never leak in.
    real_file_ids = all_ids - {mysql_id, mongo_id}
    assert real_file_ids and real_file_ids <= file_ids
    assert mysql_id not in file_ids
    assert mongo_id not in file_ids

    assert ids_for("mysql") == {mysql_id}
    assert ids_for("mongodb") == {mongo_id}

    # Filtering composes with the session scope used by the UI.
    resp = client.get(
        f"/history/?session_id={sid}&source_type=file", headers=auth_headers
    )
    assert resp.status_code == 200
    scoped_ids = {e["history_id"] for e in resp.json()["history"]}
    assert mysql_id not in scoped_ids and mongo_id not in scoped_ids

    # Open/download still work for an entry fetched under its filter.
    entry = next(e for e in all_entries if e["history_id"] in real_file_ids)
    assert entry.get("version"), "file entries keep version metadata when filtered"
    dl = client.get(
        f"/history/versions/{entry['version']['version_id']}/download/",
        headers=auth_headers,
    )
    assert dl.status_code == 200, dl.text

    # Mongo working data / live sources untouched by listing.
    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        assert mc["tabula_test"]["history"].count_documents(
            {"owner_id": owner}
        ) == len(all_ids)
    finally:
        mc.close()


def test_history_filter_source_type_validation_and_aliases(client, auth_headers):
    """Unknown categories are rejected; documented aliases normalize."""
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    owner = _owner_oid(client, auth_headers, sid)
    mysql_id = "fx-alias-" + sid[:8]
    _insert_typed_entry(
        owner, sid, mysql_id, source_type="mysql",
        source_meta={"database": "d", "table": "t"},
    )

    bad = client.get("/history/?source_type=postgres", headers=auth_headers)
    assert bad.status_code == 400
    assert "source_type" in bad.json()["detail"]
    assert client.get("/history/?source_type=postgres").status_code == 401

    # Legacy alias spellings normalize onto the canonical categories.
    alias = client.get("/history/?source_type=database", headers=auth_headers)
    assert alias.status_code == 200, alias.text
    assert {e["history_id"] for e in alias.json()["history"]} == {mysql_id}

    legacy_alias = client.get(
        "/history/?source_type=mongodb_stored", headers=auth_headers
    )
    assert legacy_alias.status_code == 200
    assert legacy_alias.json()["history"] == []

    # An empty category returns an empty list, never "everything".
    empty = client.get("/history/?source_type=mongodb", headers=auth_headers)
    assert empty.status_code == 200
    assert empty.json()["history"] == []


def test_history_filter_source_type_owner_scoped(client, auth_headers):
    """A filtered listing never returns another user's entries."""
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    owner = _owner_oid(client, auth_headers, sid)
    _insert_typed_entry(
        owner, sid, "fx-own-" + sid[:8], source_type="mysql",
        source_meta={"database": "d", "table": "t"},
    )

    other = _other_headers(client)
    other_sid = _upload(client, other, "x,y\n9,8\n")
    other_owner = _owner_oid(client, other, other_sid)
    _insert_typed_entry(
        other_owner, other_sid, "fx-other-" + other_sid[:8], source_type="mysql",
        source_meta={"database": "d", "table": "t"},
    )

    mine = client.get("/history/?source_type=mysql", headers=auth_headers)
    assert mine.status_code == 200
    ids = {e["history_id"] for e in mine.json()["history"]}
    assert "fx-own-" + sid[:8] in ids
    assert "fx-other-" + other_sid[:8] not in ids


def test_history_filter_classifies_legacy_entries_safely(client, auth_headers):
    """Legacy docs are classified only from reliable metadata."""
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    owner = _owner_oid(client, auth_headers, sid)

    legacy_mongo = "fx-lg-mongo-" + sid[:8]
    legacy_mysql = "fx-lg-mysql-" + sid[:8]
    legacy_mysql_empty = "fx-lg-mysql0-" + sid[:8]
    legacy_ambiguous = "fx-lg-none-" + sid[:8]
    _insert_typed_entry(
        owner, sid, legacy_mongo,  # no source_type; meta proves mongodb
        source_meta={"database": "d", "collection": "c"},
    )
    _insert_typed_entry(
        owner, sid, legacy_mysql,  # no source_type; meta proves mysql
        source_meta={"database": "d", "table": "t"},
    )
    _insert_typed_entry(
        owner, sid, legacy_mysql_empty,  # empty-string source_type + table meta
        source_type="", source_meta={"database": "d", "table": "t"},
    )
    _insert_typed_entry(
        owner, sid, legacy_ambiguous,  # nothing to prove a category
    )

    def ids_for(source):
        resp = client.get(
            f"/history/?source_type={source}", headers=auth_headers
        )
        assert resp.status_code == 200, resp.text
        return {e["history_id"] for e in resp.json()["history"]}

    mongo_ids = ids_for("mongodb")
    mysql_ids = ids_for("mysql")
    file_ids = ids_for("file")

    assert legacy_mongo in mongo_ids
    assert legacy_mysql not in mongo_ids
    assert legacy_mysql in mysql_ids
    assert legacy_mysql_empty in mysql_ids
    assert legacy_mongo not in mysql_ids
    # Ambiguous legacy entries never appear in a source-specific view...
    assert legacy_ambiguous not in mongo_ids
    assert legacy_ambiguous not in mysql_ids
    assert legacy_ambiguous not in file_ids
    # ...but remain in the unfiltered audit trail.
    audit = client.get("/history/", headers=auth_headers).json()["history"]
    assert legacy_ambiguous in {e["history_id"] for e in audit}


def test_clear_all_history_scoped_by_source_type(client, auth_headers):
    """Category-scoped clear-all removes one section, never the others."""
    from datetime import datetime, timezone

    from pymongo import MongoClient

    csv = "name,score\nalice,10\nbob,20\n"
    sid = _upload(client, auth_headers, csv)
    client.post(
        "/delete_rows/", headers=auth_headers,
        json={"session_id": sid, "row_ids": [0]},
    )
    owner = _owner_oid(client, auth_headers, sid)

    mysql_a = "fx-clr-mysql-a-" + sid[:8]
    mysql_b = "fx-clr-mysql-b-" + sid[:8]
    mongo_id = "fx-clr-mongo-" + sid[:8]
    mysql_vid = "fx-clr-mv-" + sid[:8]
    mysql_file_id = _insert_version_artifact(sid, owner, mysql_a, mysql_vid)
    _insert_typed_entry(
        owner, sid, mysql_a, source_type="mysql",
        source_meta={"database": "d", "table": "t"},
        operation="write_back_mysql", version_id=mysql_vid, restorable=True,
    )
    _insert_typed_entry(
        owner, sid, mysql_b, source_type="mysql",
        source_meta={"database": "d", "table": "t"},
        operation="source_delete_mysql_rows",
    )
    _insert_typed_entry(
        owner, sid, mongo_id, source_type="mongodb",
        source_meta={"database": "d", "collection": "c"},
        operation="write_back_mongodb",
    )

    def clear(source=None):
        payload = {"confirm": True, "typed_confirmation": "CLEAR HISTORY"}
        if source:
            payload["source_type"] = source
        return client.post(
            "/history/clear_all/", headers=auth_headers, json=payload
        )

    def count_by(source):
        mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
        try:
            if source == "file":
                return mc["tabula_test"]["history"].count_documents(
                    {"owner_id": owner, "source_type": "file"}
                )
            if source == "mysql":
                return mc["tabula_test"]["history"].count_documents(
                    {"owner_id": owner, "source_type": "mysql"}
                )
            return mc["tabula_test"]["history"].count_documents(
                {"owner_id": owner, "source_type": "mongodb"}
            )
        finally:
            mc.close()

    # Wrong typed phrase still rejected under a category scope.
    bad = client.post(
        "/history/clear_all/", headers=auth_headers,
        json={"confirm": True, "typed_confirmation": "nope", "source_type": "file"},
    )
    assert bad.status_code == 400
    # Unknown category is a 400, never a silent global wipe.
    unknown = client.post(
        "/history/clear_all/", headers=auth_headers,
        json={"confirm": True, "typed_confirmation": "CLEAR HISTORY",
              "source_type": "oracle"},
    )
    assert unknown.status_code == 400

    file_before = count_by("file")
    assert file_before == 2  # import + delete_rows

    # --- clear FILE only ---
    resp = clear("file")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_deleted"] == file_before
    assert "file history" in body["message"]
    assert count_by("file") == 0
    assert count_by("mysql") == 2
    assert count_by("mongodb") == 1
    # The mysql snapshot blob was not touched by the file clear.
    assert _blob_exists(mysql_file_id)

    # --- clear MYSQL only ---
    resp = clear("mysql")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entries_deleted"] == 2
    assert "MySQL history" in body["message"]
    assert count_by("mysql") == 0
    assert count_by("mongodb") == 1
    # The unshared mysql snapshot was cleaned up; the mongo entry remains.
    assert _version_doc(mysql_vid) is None
    assert not _blob_exists(mysql_file_id)

    # --- clear MONGODB (last category) ---
    resp = clear("mongodb")
    assert resp.status_code == 200, resp.text
    assert "MongoDB history" in resp.json()["message"]
    assert count_by("mongodb") == 0

    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    try:
        assert mc["tabula_test"]["history"].count_documents(
            {"owner_id": owner}
        ) == 0
    finally:
        mc.close()

    # Working dataset untouched by all three scoped clears.
    preview = client.get(
        f"/session_preview/?session_id={sid}", headers=auth_headers
    ).json()
    assert preview["row_count"] == 1

    # Scoped stats reflect the (now empty) categories.
    stats = client.get("/history/stats/?source_type=file", headers=auth_headers)
    assert stats.status_code == 200
    assert stats.json()["total_entries"] == 0
    assert stats.json()["source_type"] == "file"


def test_history_stats_scoped_by_source_type(client, auth_headers):
    """Scoped stats count only the requested category."""
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    owner = _owner_oid(client, auth_headers, sid)
    _insert_typed_entry(
        owner, sid, "fx-st-mysql-" + sid[:8], source_type="mysql",
        source_meta={"database": "d", "table": "t"},
    )
    _insert_typed_entry(
        owner, sid, "fx-st-mongo-" + sid[:8], source_type="mongodb",
        source_meta={"database": "d", "collection": "c"},
    )

    total = client.get("/history/stats/", headers=auth_headers).json()
    assert total["total_entries"] == 3  # 1 file + 2 fixtures

    file_stats = client.get(
        "/history/stats/?source_type=file", headers=auth_headers
    ).json()
    assert file_stats["total_entries"] == 1
    assert file_stats["source_type"] == "file"

    mysql_stats = client.get(
        "/history/stats/?source_type=mysql", headers=auth_headers
    ).json()
    assert mysql_stats["total_entries"] == 1
    assert mysql_stats["entries_with_backup"] == 0

    assert client.get(
        "/history/stats/?source_type=oracle", headers=auth_headers
    ).status_code == 400


def test_open_version_preserves_source_category(client, auth_headers):
    """Opening a MySQL snapshot keeps it MySQL (not silently a file entry)."""
    csv = "name,score\nalice,10\nbob,20\n"
    sid = _upload(client, auth_headers, csv)
    owner = _owner_oid(client, auth_headers, sid)

    mysql_entry = "fx-open-mysql-" + sid[:8]
    mysql_vid = "fx-open-v-" + sid[:8]
    _insert_version_artifact(sid, owner, mysql_entry, mysql_vid)
    _insert_typed_entry(
        owner, sid, mysql_entry, source_type="mysql",
        source_meta={"database": "d", "table": "t"},
        operation="write_back_mysql", version_id=mysql_vid, restorable=True,
    )

    # The entry is visible under the MySQL filter.
    listed = client.get("/history/?source_type=mysql", headers=auth_headers)
    assert mysql_entry in {e["history_id"] for e in listed.json()["history"]}

    opened = client.post(
        "/history/open_version/", headers=auth_headers,
        json={"version_id": mysql_vid, "history_id": mysql_entry},
    )
    assert opened.status_code == 200, opened.text
    out = opened.json()
    assert out["source"] == "mysql"

    # The NEW history entry recorded by the open is also MySQL, so it shows
    # up in the MySQL section — not the Files section.
    resp = client.get("/history/?source_type=mysql", headers=auth_headers)
    new_ids = {e["history_id"] for e in resp.json()["history"]}
    assert out["session_id"] and any(
        e["session_id"] == out["session_id"] for e in resp.json()["history"]
    )
    assert mysql_entry in new_ids
    file_list = client.get("/history/?source_type=file", headers=auth_headers)
    file_ids = {e["history_id"] for e in file_list.json()["history"]}
    assert not any(
        e["session_id"] == out["session_id"] for e in file_list.json()["history"]
    )
    assert mysql_entry not in file_ids

    # The new session previews fine and the snapshot is intact.
    preview = client.get(
        f"/session_preview/?session_id={out['session_id']}", headers=auth_headers
    )
    assert preview.status_code == 200
    assert _version_doc(mysql_vid) is not None
