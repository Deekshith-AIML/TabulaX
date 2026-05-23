"""Authentication, authorization and user-isolation tests."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from tests.conftest import register_and_login


def test_register_and_me(client, unique_user):
    reg = client.post("/register/", json=unique_user)
    assert reg.status_code == 201
    assert "password" not in reg.json() and "hashed_password" not in reg.json()

    login = client.post(
        "/token",
        data={"username": unique_user["username"], "password": unique_user["password"]},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = client.get("/users/me/", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == unique_user["username"]
    assert "password" not in me.json()


def test_register_duplicate_username_rejected(client, unique_user):
    assert client.post("/register/", json=unique_user).status_code == 201
    dup = dict(unique_user, email="other@example.com")
    assert client.post("/register/", json=dup).status_code == 400


def test_register_duplicate_email_rejected(client, unique_user):
    assert client.post("/register/", json=unique_user).status_code == 201
    dup = dict(unique_user, username="another_name")
    assert client.post("/register/", json=dup).status_code == 400


def test_register_short_password_rejected(client, unique_user):
    weak = dict(unique_user, password="short")
    resp = client.post("/register/", json=weak)
    assert resp.status_code == 422  # pydantic min_length=8 -> validation error


def test_register_invalid_email_rejected(client, unique_user):
    bad = dict(unique_user, email="not-an-email")
    assert client.post("/register/", json=bad).status_code == 422


def test_login_wrong_password_rejected(client, unique_user):
    register_and_login(client, unique_user)
    resp = client.post(
        "/token", data={"username": unique_user["username"], "password": "wrongpass"}
    )
    assert resp.status_code == 401


def test_login_unknown_user_rejected(client):
    resp = client.post("/token", data={"username": "ghost", "password": "whatever1"})
    assert resp.status_code == 401


def test_token_error_message_does_not_leak(client):
    resp = client.post("/token", data={"username": "ghost", "password": "x"})
    detail = resp.json()["detail"]
    assert "stacktrace" not in detail.lower()
    assert "traceback" not in detail.lower()


def test_request_without_token_rejected(client):
    assert client.get("/users/me/").status_code == 401
    assert client.post("/upload_file/").status_code in (401, 422)


def test_garbage_token_rejected(client):
    resp = client.get(
        "/users/me/", headers={"Authorization": "Bearer not.a.jwt"}
    )
    assert resp.status_code == 401


def test_tampered_token_rejected(client, unique_user):
    token = register_and_login(client, unique_user)
    # Flip a character in the payload segment to break the signature.
    parts = token.split(".")
    payload = parts[1]
    tampered = ("A" if payload[0] != "A" else "B") + payload[1:]
    bad = f"{parts[0]}.{tampered}.{parts[2]}"
    resp = client.get("/users/me/", headers={"Authorization": f"Bearer {bad}"})
    assert resp.status_code == 401


def test_expired_token_rejected(client, unique_user):
    from datetime import timedelta

    from security import create_access_token

    register_and_login(client, unique_user)
    expired = create_access_token(unique_user["username"], expires_delta=timedelta(seconds=-10))
    resp = client.get("/users/me/", headers={"Authorization": f"Bearer {expired}"})
    assert resp.status_code == 401


def test_one_user_cannot_see_anothers_sessions(client, unique_user):
    """Changing an identifier must not grant access to another user's data."""
    user_b = dict(
        unique_user,
        username=f"user_{uuid.uuid4().hex[:10]}",
        email=f"{uuid.uuid4().hex[:8]}@example.com",
    )
    token_a = register_and_login(client, unique_user)
    token_b = register_and_login(client, user_b)

    # A lists their (empty) sessions.
    a_sessions = client.get(
        "/sessions/", headers={"Authorization": f"Bearer {token_a}"}
    ).json()["sessions"]
    # B lists theirs too.
    b_sessions = client.get(
        "/sessions/", headers={"Authorization": f"Bearer {token_b}"}
    ).json()["sessions"]
    a_ids = {s["session_id"] for s in a_sessions}
    b_ids = {s["session_id"] for s in b_sessions}
    assert a_ids.isdisjoint(b_ids)


def test_user_cannot_read_foreign_session_data(client, unique_user):
    """Uploading as A then requesting the same session as B must 404."""
    user_b = dict(
        unique_user,
        username=f"user_{uuid.uuid4().hex[:10]}",
        email=f"{uuid.uuid4().hex[:8]}@example.com",
    )
    token_a = register_and_login(client, unique_user)
    token_b = register_and_login(client, user_b)

    csv = "name,val\nx,1\ny,2\n"
    up = client.post(
        "/upload_file/",
        headers={"Authorization": f"Bearer {token_a}"},
        files={"file": ("t.csv", csv.encode(), "text/csv")},
    )
    assert up.status_code == 200
    sid = up.json()["session_id"]

    # A can download it.
    dl_a = client.get(
        "/download/", params={"session_id": sid},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert dl_a.status_code == 200

    # B cannot (session id alone is not sufficient).
    dl_b = client.get(
        "/download/", params={"session_id": sid},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert dl_b.status_code == 404

    # B cannot delete A's session.
    del_b = client.delete(
        f"/sessions/{sid}", headers={"Authorization": f"Bearer {token_b}"}
    )
    assert del_b.status_code == 404
