"""Pytest configuration and shared fixtures.

Tests run against an **isolated** database (``DATABASE_NAME=tabula_test``) so
they never touch development data, and always with the LLM disabled so results
are deterministic and no API key is required.
"""

from __future__ import annotations

import os
import uuid

# Configure the environment BEFORE importing config/app so settings are built
# correctly for the test run.
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_NAME"] = "tabula_test"
os.environ["OPENAI_API_KEY"] = ""  # force LLM off for deterministic tests
os.environ["LLM_PROVIDER"] = "openai"  # pin provider (no key -> disabled)
# Pinned OFF so no test can reach a real local Ollama server; fallback
# behaviour is tested with a monkeypatched _invoke_ollama instead.
os.environ["ENABLE_LOCAL_FALLBACK"] = "false"
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")

import pytest
from fastapi.testclient import TestClient

import config

config.get_settings.cache_clear()
config.settings = config.Settings()

# Reset any lazily-cached singletons that captured old settings.
import llm

llm.reset_client()


def pytest_collection_modifyitems(config, items):
    """Deselect ``live_llm`` tests unless RUN_LLM_LIVE=1.

    The default automated run must never consume real model quota or require
    a running Ollama server; those integration tests are opt-in only.
    """
    import os

    if os.environ.get("RUN_LLM_LIVE", "").strip().lower() in ("1", "true", "yes"):
        return
    skip_live = pytest.mark.skip(
        reason="live model test; set RUN_LLM_LIVE=1 to run against a real provider"
    )
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def client():
    """A TestClient bound to the app, connecting to the test database."""
    from app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def unique_user():
    """Generate a unique set of registration fields to avoid collisions."""
    suffix = uuid.uuid4().hex[:10]
    return {
        "username": f"user_{suffix}",
        "password": "Sup3rSecret!",
        "email": f"{suffix}@example.com",
        "full_name": "Test User",
    }


def register_and_login(client: TestClient, user: dict) -> str:
    """Register ``user`` and return an access token."""
    resp = client.post("/register/", json=user)
    assert resp.status_code == 201, f"register failed: {resp.status_code} {resp.text}"
    resp = client.post(
        "/token",
        data={"username": user["username"], "password": user["password"]},
    )
    assert resp.status_code == 200, f"login failed: {resp.status_code} {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture()
def auth_headers(client, unique_user):
    """Return ``{"Authorization": "Bearer ..."}`` for a fresh test user."""
    token = register_and_login(client, unique_user)
    return {"Authorization": f"Bearer {token}"}
