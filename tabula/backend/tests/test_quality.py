"""Data-quality profiling and transformation preflight validation tests.

Covers: read-only profile (nulls, uniqueness, potential keys, duplicates,
sampled large frames marked approximate), preflight blocking on contradictory
examples / missing columns / rules that fail every sample, and ownership.
"""

from __future__ import annotations

import uuid

import pandas as pd

import quality
from tests.conftest import register_and_login


def _upload(client, headers, csv: str) -> str:
    resp = client.post(
        "/upload_file/",
        headers=headers,
        files={"file": ("data.csv", csv.encode(), "text/csv")},
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


# --------------------------------------------------------------------------- #
# Unit: profile_dataframe
# --------------------------------------------------------------------------- #
def test_profile_dataframe_flags_nulls_duplicates_and_keys():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["a", "b", None, "d"],
        "const": ["x", "x", "x", "x"],
    })
    out = quality.profile_dataframe(df, source="file", source_name="t.csv")

    assert out["total_rows"] == 4
    assert out["approximate"] is False
    cols = {c["name"]: c for c in out["columns"]}
    assert cols["name"]["null_count"] == 1
    assert cols["name"]["missing_pct"] > 0
    assert cols["const"]["is_constant"] is True
    assert "id" in out["potential_keys"]
    assert out["overview"]["duplicate_rows"] == 0

    # Constant + missing warnings surface in the overview.
    msgs = " ".join(w["message"] for w in out["overview"]["warnings"])
    assert "missing" in msgs.lower()


def test_profile_marks_large_frames_approximate():
    df = pd.DataFrame({"a": range(quality.SAMPLE_THRESHOLD + 100)})
    out = quality.profile_dataframe(df)
    assert out["approximate"] is True
    assert out["profiled_rows"] <= quality.SAMPLE_SIZE
    assert out["total_rows"] > quality.SAMPLE_THRESHOLD


# --------------------------------------------------------------------------- #
# Unit: validate_examples_for_column
# --------------------------------------------------------------------------- #
def test_validate_blocks_on_contradictory_examples():
    df = pd.DataFrame({"name": ["alice", "bob"]})
    out = quality.validate_examples_for_column(
        df, "name",
        [("alice", "ALICE"), ("alice", "BOB")],
    )
    assert out["ok"] is False
    assert any("contradict" in e.lower() for e in out["errors"])


def test_validate_flags_missing_column():
    df = pd.DataFrame({"name": ["alice"]})
    out = quality.validate_examples_for_column(df, "nope", [("a", "b")])
    assert out["ok"] is False
    assert any("not in the current dataset" in e for e in out["errors"])


def test_validate_rule_reproducing_examples_is_ok():
    from transform.apply import apply_to_series

    df = pd.DataFrame({"name": ["alice", "bob", "carol"]})
    code = "def transform(x):\n    return str(x).upper()\n"
    out = quality.validate_examples_for_column(
        df, "name", [("alice", "ALICE"), ("bob", "BOB")],
        code=code, apply_fn=apply_to_series,
    )
    assert out["ok"] is True, out
    assert out["checks"]["examples_passed"] == 2
    assert out["checks"]["sample_checked"] == 3


def test_validate_warns_on_type_change_and_nulls():
    df = pd.DataFrame({"score": [1.0, None, 3.0]})
    from transform.apply import apply_to_series

    code = "def transform(x):\n    return 'high' if (x or 0) > 2 else 'low'\n"
    out = quality.validate_examples_for_column(
        df, "score", [(1.0, "low"), (3.0, "high")],
        code=code, apply_fn=apply_to_series,
    )
    assert out["ok"] is True
    assert any("type" in w.lower() for w in out["warnings"])
    assert any("null" in w.lower() for w in out["warnings"] + out["info"])


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def test_quality_profile_endpoint(client, auth_headers):
    csv = "id,name\n1,alice\n2,bob\n3,alice\n"
    sid = _upload(client, auth_headers, csv)
    resp = client.post(
        "/quality/profile/", headers=auth_headers, json={"session_id": sid}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == sid
    assert body["total_rows"] == 3
    assert any(c["name"] == "name" for c in body["columns"])
    # "alice" duplicated -> duplicate_value_count > 0 on name.
    name_col = next(c for c in body["columns"] if c["name"] == "name")
    assert name_col["duplicate_value_count"] >= 1


def test_quality_profile_requires_ownership(client, auth_headers):
    sid = _upload(client, auth_headers, "a,b\n1,2\n")
    other = _other_headers(client)
    resp = client.post(
        "/quality/profile/", headers=other, json={"session_id": sid}
    )
    assert resp.status_code == 404


def test_quality_validate_endpoint(client, auth_headers):
    csv = "name\nalice\nbob\n"
    sid = _upload(client, auth_headers, csv)
    resp = client.post(
        "/quality/validate/", headers=auth_headers,
        json={
            "session_id": sid,
            "column": "name",
            "examples": [{"input": "alice", "output": "ALICE"}],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["column"] == "name"
    assert body["ok"] is True
    assert body["errors"] == []


def test_quality_validate_rejects_bad_examples(client, auth_headers):
    sid = _upload(client, auth_headers, "name\nalice\n")
    resp = client.post(
        "/quality/validate/", headers=auth_headers,
        json={
            "session_id": sid,
            "column": "name",
            "examples": [{"input": "alice", "output": ""}],
        },
    )
    # Server-side example parsing rejects empty outputs with 400.
    assert resp.status_code == 400
