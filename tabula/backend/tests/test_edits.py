"""Row/column deletion with undo and working-copy synchronisation.

Covers: stable internal row ids, preview/count updates, CSV export excluding
internal ids and deleted data, undo (restore), ownership enforcement, stale
selections, apply operating on the edited working dataset, and the
/session_preview/ + /revert_session/ endpoints used after a page refresh.
"""

from __future__ import annotations

import uuid

from tests.conftest import register_and_login

ROW_ID = "_tabula_row_id"


def _upload(client, headers, csv: str, name: str = "data.csv") -> str:
    resp = client.post(
        "/upload_file/",
        headers=headers,
        files={"file": (name, csv.encode(), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def _delete_rows(client, headers, sid, row_ids):
    return client.post(
        "/delete_rows/", headers=headers,
        json={"session_id": sid, "row_ids": row_ids},
    )


def _delete_columns(client, headers, sid, columns):
    return client.post(
        "/delete_columns/", headers=headers,
        json={"session_id": sid, "columns": columns},
    )


def _other_headers(client) -> dict:
    suffix = uuid.uuid4().hex[:10]
    token = register_and_login(client, {
        "username": f"other_{suffix}",
        "password": "Sup3rSecret!",
        "email": f"{suffix}@example.com",
        "full_name": "Other User",
    })
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# Row deletion
# --------------------------------------------------------------------------- #
def test_delete_rows_updates_preview_count_and_export(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\ncarol,30\ndave,40\n"
    sid = _upload(client, auth_headers, csv)

    resp = _delete_rows(client, auth_headers, sid, [1, 2])  # bob, carol
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["removed"] == 2
    assert body["row_count"] == 2
    assert body["columns"] == ["name", "score"]  # no internal id column
    assert ROW_ID not in body["columns"]
    names = [r["name"] for r in body["preview"]]
    assert names == ["alice", "dave"]
    # Records keep the internal id so the UI can keep selecting rows.
    assert all(ROW_ID in r for r in body["preview"])
    # Undo payload carries the full deleted rows with their ids.
    deleted = body["deleted_rows"]
    assert [r["name"] for r in deleted] == ["bob", "carol"]
    assert [r[ROW_ID] for r in deleted] == [1, 2]

    dl = client.get("/download/", params={"session_id": sid}, headers=auth_headers)
    assert dl.status_code == 200
    lines = [ln for ln in dl.text.strip().splitlines() if ln]
    assert lines[0] == "name,score"
    assert ROW_ID not in lines[0]
    assert len(lines) == 3  # header + 2 rows
    assert "bob" not in dl.text and "carol" not in dl.text


def test_delete_rows_then_restore_roundtrip(client, auth_headers):
    csv = "v\na\nb\nc\n"
    sid = _upload(client, auth_headers, csv)

    resp = _delete_rows(client, auth_headers, sid, [0, 2])
    assert resp.status_code == 200, resp.text
    deleted = resp.json()["deleted_rows"]
    assert resp.json()["row_count"] == 1

    restore = client.post(
        "/restore_rows/", headers=auth_headers,
        json={"session_id": sid, "rows": deleted},
    )
    assert restore.status_code == 200, restore.text
    body = restore.json()
    assert body["row_count"] == 3
    assert [r["v"] for r in body["preview"]] == ["a", "b", "c"]  # order kept

    dl = client.get("/download/", params={"session_id": sid}, headers=auth_headers)
    lines = [ln for ln in dl.text.strip().splitlines() if ln]
    assert lines == ["v", "a", "b", "c"]


def test_delete_all_rows_then_undo(client, auth_headers):
    csv = "v\nx\ny\n"
    sid = _upload(client, auth_headers, csv)
    resp = _delete_rows(client, auth_headers, sid, [0, 1])
    assert resp.status_code == 200, resp.text
    assert resp.json()["row_count"] == 0
    assert resp.json()["preview"] == []

    restore = client.post(
        "/restore_rows/", headers=auth_headers,
        json={"session_id": sid, "rows": resp.json()["deleted_rows"]},
    )
    assert restore.status_code == 200, restore.text
    assert restore.json()["row_count"] == 2


def test_delete_rows_rejects_stale_ids(client, auth_headers):
    csv = "v\na\nb\n"
    sid = _upload(client, auth_headers, csv)
    resp = _delete_rows(client, auth_headers, sid, [0, 99])
    assert resp.status_code == 400
    assert "stale" in resp.json()["detail"].lower()

    # Deleting id 0 succeeded only if the whole request failed atomically.
    resp2 = client.get("/session_preview/", params={"session_id": sid},
                       headers=auth_headers)
    assert resp2.json()["row_count"] == 2  # unchanged (nothing deleted)


def test_delete_rows_requires_auth(client):
    resp = client.post("/delete_rows/", json={"session_id": "a" * 32, "row_ids": [0]})
    assert resp.status_code == 401


def test_delete_rows_other_users_session_is_404(client, auth_headers):
    sid = _upload(client, auth_headers, "v\na\nb\n")
    other = _other_headers(client)
    resp = _delete_rows(client, other, sid, [0])
    assert resp.status_code == 404


def test_restore_rows_rejects_duplicates_and_missing_ids(client, auth_headers):
    sid = _upload(client, auth_headers, "v\na\nb\n")
    resp = client.post(
        "/restore_rows/", headers=auth_headers,
        json={"session_id": sid, "rows": [{"v": "x"}]},
    )
    assert resp.status_code == 400
    assert "missing internal row ids" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Column deletion
# --------------------------------------------------------------------------- #
def test_delete_column_with_undo(client, auth_headers):
    csv = "a,b,c\n1,2,3\n4,5,6\n"
    sid = _upload(client, auth_headers, csv)

    resp = _delete_columns(client, auth_headers, sid, ["b"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["columns"] == ["a", "c"]
    assert body["row_count"] == 2
    assert "b" not in body["preview"][0]
    deleted_meta = body["deleted_columns"]
    assert deleted_meta[0]["name"] == "b"
    assert deleted_meta[0]["position"] == 1
    assert [v["b"] for v in deleted_meta[0]["values"]] == [2, 5]

    dl = client.get("/download/", params={"session_id": sid}, headers=auth_headers)
    assert dl.text.splitlines()[0] == "a,c"
    assert ",b," not in dl.text

    restore = client.post(
        "/restore_columns/", headers=auth_headers,
        json={"session_id": sid, "columns": deleted_meta},
    )
    assert restore.status_code == 200, restore.text
    body = restore.json()
    assert body["columns"] == ["a", "b", "c"]  # original order restored
    assert body["preview"][0]["b"] == 2
    assert body["preview"][1]["a"] == 4

    dl = client.get("/download/", params={"session_id": sid}, headers=auth_headers)
    assert dl.text.splitlines()[0] == "a,b,c"


def test_delete_multiple_columns_preserves_remaining_order(client, auth_headers):
    csv = "a,b,c,d\n1,2,3,4\n"
    sid = _upload(client, auth_headers, csv)
    resp = _delete_columns(client, auth_headers, sid, ["b", "d"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["columns"] == ["a", "c"]

    restore = client.post(
        "/restore_columns/", headers=auth_headers,
        json={"session_id": sid, "columns": resp.json()["deleted_columns"]},
    )
    assert [c for c in restore.json()["columns"]] == ["a", "b", "c", "d"]


def test_delete_columns_missing_or_all_rejected(client, auth_headers):
    csv = "a,b,c\n1,2,3\n"
    sid = _upload(client, auth_headers, csv)
    resp = _delete_columns(client, auth_headers, sid, ["nope"])
    assert resp.status_code == 400
    resp = _delete_columns(client, auth_headers, sid, ["a", "b", "c"])
    assert resp.status_code == 400
    assert "at least one column" in resp.json()["detail"]


def test_delete_columns_other_users_session_is_404(client, auth_headers):
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    other = _other_headers(client)
    resp = _delete_columns(client, other, sid, ["a"])
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Sync: apply runs on the edited dataset; preview/revert restore after refresh
# --------------------------------------------------------------------------- #
def test_apply_uses_edited_working_dataset(client, auth_headers):
    csv = "word\napple\nbanana\ncherry\n"
    sid = _upload(client, auth_headers, csv)

    # Delete "banana" first; the transformation must apply to the edited copy.
    resp = _delete_rows(client, auth_headers, sid, [1])
    assert resp.status_code == 200, resp.text

    cls = client.post("/classify/", headers=auth_headers, json={
        "column": "word",
        "examples": [{"input": "apple", "output": "APPLE"},
                     {"input": "cherry", "output": "CHERRY"}],
    })
    assert cls.status_code == 200, cls.text
    gen = client.post("/generate_function/", headers=auth_headers, json={
        "column": "word",
        "examples": [{"input": "apple", "output": "APPLE"},
                     {"input": "cherry", "output": "CHERRY"}],
        "classification": cls.json()["classification"],
    })
    assert gen.status_code == 200, gen.text

    ap = client.post("/apply/", headers=auth_headers, json={
        "transformation_id": gen.json()["transformation_id"],
        "selected_column": "word",
        "source_type": "file",
        "session_id": sid,
    })
    assert ap.status_code == 200, ap.text
    assert [r["word"] for r in ap.json()["preview"]] == ["APPLE", "CHERRY"]
    assert ap.json()["row_count"] == 2
    assert ap.json()["transformed_columns"] == ["word"]

    dl = client.get("/download/", params={"session_id": sid}, headers=auth_headers)
    assert "banana" not in dl.text
    assert "APPLE" in dl.text and "CHERRY" in dl.text


def test_transform_then_delete_rows_then_export(client, auth_headers):
    """Row deletions made after a transform apply to the transformed copy."""
    csv = "word\napple\nbanana\ncherry\n"
    sid = _upload(client, auth_headers, csv)
    cls = client.post("/classify/", headers=auth_headers, json={
        "column": "word",
        "examples": [{"input": "apple", "output": "APPLE"},
                     {"input": "banana", "output": "BANANA"}],
    })
    gen = client.post("/generate_function/", headers=auth_headers, json={
        "column": "word",
        "examples": [{"input": "apple", "output": "APPLE"},
                     {"input": "banana", "output": "BANANA"}],
        "classification": cls.json()["classification"],
    })
    ap = client.post("/apply/", headers=auth_headers, json={
        "transformation_id": gen.json()["transformation_id"],
        "selected_column": "word",
        "source_type": "file",
        "session_id": sid,
    })
    assert ap.status_code == 200, ap.text

    resp = _delete_rows(client, auth_headers, sid, [0])  # delete APPLE row
    assert resp.status_code == 200, resp.text
    # The uppercase rule applies to every surviving row of the edited copy.
    assert [r["word"] for r in resp.json()["preview"]] == ["BANANA", "CHERRY"]

    dl = client.get("/download/", params={"session_id": sid}, headers=auth_headers)
    lines = dl.text.splitlines()
    assert ROW_ID not in lines[0]
    assert not any("APPLE" in ln for ln in lines[1:])
    assert any("BANANA" in ln for ln in lines[1:])


def test_session_preview_returns_edited_and_original(client, auth_headers):
    csv = "v\n1\n2\n3\n"
    sid = _upload(client, auth_headers, csv)
    resp = _delete_rows(client, auth_headers, sid, [0])
    assert resp.status_code == 200, resp.text

    prev = client.get("/session_preview/", params={"session_id": sid},
                      headers=auth_headers)
    assert prev.status_code == 200, prev.text
    body = prev.json()
    assert body["columns"] == ["v"]
    assert body["row_count"] == 2
    assert [r["v"] for r in body["preview"]] == [2, 3]
    # The original (unedited) preview still has every row.
    assert [r["v"] for r in body["original_preview"]] == [1, 2, 3]
    assert body["source_name"] == "data.csv"


def test_session_preview_requires_ownership(client, auth_headers):
    sid = _upload(client, auth_headers, "v\n1\n")
    other = _other_headers(client)
    resp = client.get("/session_preview/", params={"session_id": sid},
                      headers=other)
    assert resp.status_code == 404


def test_revert_session_restores_original(client, auth_headers):
    csv = "v\na\nb\nc\n"
    sid = _upload(client, auth_headers, csv)
    resp = _delete_rows(client, auth_headers, sid, [0, 1])
    assert resp.status_code == 200, resp.text
    assert resp.json()["row_count"] == 1

    rev = client.post("/revert_session/", headers=auth_headers,
                      json={"session_id": sid})
    assert rev.status_code == 200, rev.text
    body = rev.json()
    assert body["row_count"] == 3
    assert [r["v"] for r in body["preview"]] == ["a", "b", "c"]

    dl = client.get("/download/", params={"session_id": sid}, headers=auth_headers)
    assert dl.text.strip().splitlines() == ["v", "a", "b", "c"]
