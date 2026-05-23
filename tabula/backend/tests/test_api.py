"""API integration tests: health, upload, classify, generate, apply, download.

Covers the core user flow and verifies that no OpenAI key is needed for
supported transformations.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_reports_llm_and_mongo_without_leaking_secrets(client):
    resp = client.get("/ready")
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert "mongo" in body
    assert body["llm_configured"] is False  # conftest disables the LLM
    # Never expose credentials in a health payload.
    text = resp.text.lower()
    assert "mongodb://" not in text
    assert "secret" not in text
    assert "password" not in text
    assert "api_key" not in text


def test_upload_csv_returns_columns_preview_and_session(client, auth_headers):
    csv = "name,score\nalice,10\nbob,20\ncarol,30\n"
    resp = client.post(
        "/upload_file/",
        headers=auth_headers,
        files={"file": ("data.csv", csv.encode(), "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["columns"] == ["name", "score"]
    assert body["session_id"]
    assert len(body["preview"]) == 3


def test_upload_rejects_unsupported_type(client, auth_headers):
    resp = client.post(
        "/upload_file/",
        headers=auth_headers,
        files={"file": ("bad.exe", b"MZ....", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_file(client, auth_headers):
    resp = client.post(
        "/upload_file/",
        headers=auth_headers,
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert resp.status_code == 400


def test_upload_malformed_csv_rejected(client, auth_headers):
    # Bytes that are not valid UTF-8 cannot be decoded into a table.
    resp = client.post(
        "/upload_file/",
        headers=auth_headers,
        files={"file": ("broken.csv", bytes(range(256)), "text/csv")},
    )
    assert resp.status_code == 400


def test_upload_ragged_csv_skips_bad_rows(client, auth_headers):
    # Real-world exports often contain a row with extra commas; it must be
    # skipped rather than failing the whole upload with a ParserError.
    csv = "id,name\n1,alice\n2,bob,extra,commas\n3,carol\n"
    resp = client.post(
        "/upload_file/",
        headers=auth_headers,
        files={"file": ("ragged.csv", csv.encode(), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["columns"] == ["id", "name"]
    assert len(body["preview"]) == 2  # the malformed row was dropped


def test_upload_semicolon_delimited_csv(client, auth_headers):
    csv = "id;name;city\n1;alice;berlin\n2;bob;rome\n"
    resp = client.post(
        "/upload_file/",
        headers=auth_headers,
        files={"file": ("euro.csv", csv.encode(), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["columns"] == ["id", "name", "city"]


def test_sql_file_returns_sql_content(client, auth_headers):
    sql = b"CREATE TABLE t (id INT);"
    resp = client.post(
        "/upload_file/",
        headers=auth_headers,
        files={"file": ("schema.sql", sql, "application/sql")},
    )
    assert resp.status_code == 200
    assert resp.json()["file_type"] == "sql"
    assert "CREATE TABLE" in resp.json()["sql_content"]


def _pairs(*pairs):
    """Structured examples payload: [{"input": ..., "output": ...}, ...]."""
    return [{"input": s, "output": t} for s, t in pairs]


def _split_examples(examples: str):
    """Test helper: legacy text lines -> structured pairs (split at first arrow)."""
    out = []
    for line in examples.splitlines():
        line = line.strip()
        if not line:
            continue
        positions = [i for i in (line.find("->"), line.find("\u2192")) if i >= 0]
        if positions:
            i = min(positions)
            if line.startswith("->", i):
                s, t = line[:i], line[i + 2:]
            else:
                s, t = line[:i], line[i + 1:]
        else:
            s, _, t = line.partition(",")
        out.append((s.strip(), t.strip()))
    return out


def _flow(client, auth_headers, csv, examples, expected_first_col_val=None):
    """Run upload -> classify -> generate -> apply -> download."""
    up = client.post(
        "/upload_file/",
        headers=auth_headers,
        files={"file": ("data.csv", csv.encode(), "text/csv")},
    )
    assert up.status_code == 200, up.text
    session_id = up.json()["session_id"]
    columns = up.json()["columns"]
    target = columns[0]
    pairs = _pairs(*_split_examples(examples))

    cls = client.post(
        "/classify/",
        headers=auth_headers,
        json={"column": target, "examples": pairs},
    )
    assert cls.status_code == 200, cls.text
    classification = cls.json()["classification"]

    gen = client.post(
        "/generate_function/",
        headers=auth_headers,
        json={
            "column": target,
            "examples": pairs,
            "classification": classification,
        },
    )
    assert gen.status_code == 200, gen.text
    gen_body = gen.json()
    assert gen_body["code"]
    assert gen_body["transformation_id"]
    # Deterministic families must not require the LLM.
    assert gen_body["llm_available"] is False

    ap = client.post(
        "/apply/",
        headers=auth_headers,
        json={
            "transformation_id": gen_body["transformation_id"],
            "selected_column": target,
            "source_type": "file",
            "session_id": session_id,
        },
    )
    assert ap.status_code == 200, ap.text
    ap_body = ap.json()
    assert ap_body["preview"], "apply must return a preview"
    if expected_first_col_val is not None:
        assert str(ap_body["preview"][0][target]) == expected_first_col_val

    dl = client.get(
        "/download/",
        params={"session_id": session_id},
        headers=auth_headers,
    )
    assert dl.status_code == 200
    assert "text/csv" in dl.headers["content-type"]
    return ap_body, gen_body


def test_full_flow_uppercase(client, auth_headers):
    csv = "word\napple\nbanana\n"
    _flow(client, auth_headers, csv, "apple -> APPLE\nbanana -> BANANA", expected_first_col_val="APPLE")


def test_full_flow_numeric(client, auth_headers):
    csv = "val\n1\n2\n3\n"
    ap_body, gen = _flow(client, auth_headers, csv, "1 -> 3\n2 -> 5\n3 -> 7")
    assert gen["classification"] == "Numerical"


def test_full_flow_template_algorithmic(client, auth_headers):
    csv = "num\n10\n5\n"
    ap_body, gen = _flow(client, auth_headers, csv, "10 -> 1010\n5 -> 101")
    assert gen["classification"] == "Algorithmic"
    assert gen["template"] == "decimal_to_binary"


def test_apply_requires_server_side_transformation_id(client, auth_headers):
    """A client-supplied Python 'code' field must be IGNORED/rejected."""
    csv = "val\n1\n"
    up = client.post(
        "/upload_file/", headers=auth_headers,
        files={"file": ("d.csv", csv.encode(), "text/csv")},
    )
    sid = up.json()["session_id"]
    # Send raw code and no transformation_id -> must be rejected by schema.
    resp = client.post(
        "/apply/",
        headers=auth_headers,
        json={
            "code": "import os\ndef transform(x):\n    return os.system('id')",
            "selected_column": "val",
            "source_type": "file",
            "session_id": sid,
        },
    )
    assert resp.status_code == 422  # transformation_id required, code ignored


def test_apply_with_invalid_transformation_id_rejected(client, auth_headers):
    csv = "val\n1\n"
    up = client.post(
        "/upload_file/", headers=auth_headers,
        files={"file": ("d.csv", csv.encode(), "text/csv")},
    )
    sid = up.json()["session_id"]
    resp = client.post(
        "/apply/",
        headers=auth_headers,
        json={
            "transformation_id": "a" * 32,
            "selected_column": "val",
            "source_type": "file",
            "session_id": sid,
        },
    )
    assert resp.status_code == 404


def test_apply_wrong_column_reports_available(client, auth_headers):
    csv = "val\n1\n2\n"
    up = client.post(
        "/upload_file/", headers=auth_headers,
        files={"file": ("d.csv", csv.encode(), "text/csv")},
    )
    sid = up.json()["session_id"]
    cls = client.post("/classify/", headers=auth_headers,
                      json={"column": "val", "examples": _pairs(*_split_examples("1 -> 3\n2 -> 5"))})
    classification = cls.json()["classification"]
    gen = client.post("/generate_function/", headers=auth_headers,
                      json={"column": "val",
                            "examples": _pairs(*_split_examples("1 -> 3\n2 -> 5")),
                            "classification": classification})
    tid = gen.json()["transformation_id"]
    resp = client.post("/apply/", headers=auth_headers, json={
        "transformation_id": tid, "selected_column": "does_not_exist",
        "source_type": "file", "session_id": sid,
    })
    assert resp.status_code == 400
    assert "does_not_exist" in resp.json()["detail"]


def test_classify_rejects_malformed_examples(client, auth_headers):
    # Contradictory duplicates in a structured payload -> 400 with detail.
    resp = client.post("/classify/", headers=auth_headers,
                       json={"column": "x",
                             "examples": _pairs(("a", "b"), ("a", "c"))})
    assert resp.status_code == 400
    assert "contradictory" in resp.json()["detail"].lower()
    # Missing output -> 400.
    resp = client.post("/classify/", headers=auth_headers,
                       json={"column": "x", "examples": [{"input": "a", "output": ""}]})
    assert resp.status_code == 400
    assert "missing output" in resp.json()["detail"].lower()
    # Missing input -> 400.
    resp = client.post("/classify/", headers=auth_headers,
                       json={"column": "x", "examples": [{"input": "  ", "output": "y"}]})
    assert resp.status_code == 400
    assert "missing input" in resp.json()["detail"].lower()


def test_generation_for_unknown_pattern_without_llm_is_actionable(client, auth_headers):
    resp = client.post("/generate_function/", headers=auth_headers, json={
        "column": "x",
        "examples": _pairs(("foo", "Qq1foo"), ("bar", "Qq1bar")),
        "classification": "String-based",
    })
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "OPENAI_API_KEY" in detail or "LLM" in detail


def test_download_without_session_uses_latest(client, auth_headers):
    csv = "a\n1\n2\n"
    client.post("/upload_file/", headers=auth_headers,
                files={"file": ("d.csv", csv.encode(), "text/csv")})
    resp = client.get("/download/", headers=auth_headers)
    assert resp.status_code == 200
    assert "a" in resp.text


def test_download_with_no_data_is_400(client, auth_headers):
    # auth_headers user is fresh; they have no sessions yet.
    resp = client.get("/download/", headers=auth_headers)
    assert resp.status_code == 400


def test_session_expiry_metadata_present(client, auth_headers):
    csv = "a\n1\n"
    up = client.post("/upload_file/", headers=auth_headers,
                     files={"file": ("d.csv", csv.encode(), "text/csv")})
    sid = up.json()["session_id"]
    sessions = client.get("/sessions/", headers=auth_headers).json()["sessions"]
    match = [s for s in sessions if s["session_id"] == sid]
    assert match and match[0].get("expires_at")


def test_delete_session_then_download_404(client, auth_headers):
    csv = "a\n1\n"
    up = client.post("/upload_file/", headers=auth_headers,
                     files={"file": ("d.csv", csv.encode(), "text/csv")})
    sid = up.json()["session_id"]
    assert client.delete(f"/sessions/{sid}", headers=auth_headers).status_code == 200
    resp = client.get("/download/", params={"session_id": sid}, headers=auth_headers)
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Docker loopback mapping for user-supplied Mongo URIs
# --------------------------------------------------------------------------- #
def test_map_mongo_uri_rewrites_loopback_in_docker(monkeypatch):
    from app import _map_mongo_uri
    from config import settings

    monkeypatch.setattr(settings, "in_docker", True)
    assert (
        _map_mongo_uri("mongodb://localhost:27017/db")
        == "mongodb://host.docker.internal:27017/db"
    )
    assert (
        _map_mongo_uri("mongodb://user:pass@127.0.0.1:27017/x")
        == "mongodb://user:pass@host.docker.internal:27017/x"
    )
    assert (
        _map_mongo_uri("mongodb://db1:27017,localhost:27018/y")
        == "mongodb://db1:27017,host.docker.internal:27018/y"
    )
    # Non-loopback hosts (docker service names, Atlas) stay untouched.
    assert (
        _map_mongo_uri("mongodb://mongo:27017/tabula")
        == "mongodb://mongo:27017/tabula"
    )
    assert (
        _map_mongo_uri("mongodb+srv://cluster.abc.mongodb.net/db")
        == "mongodb+srv://cluster.abc.mongodb.net/db"
    )


def test_map_mongo_uri_noop_outside_docker(monkeypatch):
    from app import _map_mongo_uri
    from config import settings

    monkeypatch.setattr(settings, "in_docker", False)
    assert _map_mongo_uri("mongodb://localhost:27017/db") == "mongodb://localhost:27017/db"
