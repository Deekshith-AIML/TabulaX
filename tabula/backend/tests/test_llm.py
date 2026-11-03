"""LLM configuration tests: personal-key handling, budget, status, Ollama.

The personal API key must never be persisted; the per-request budget must
enforce ``llm_max_calls_per_request``; ``/llm/status/`` must never leak keys;
Ollama pull requires explicit confirmation and validates the model name.
"""

from __future__ import annotations

import pytest

import llm
from config import settings


class _FakeMessage:
    content = "String-based"


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def invoke(self, prompt, config=None):
        return _FakeMessage()


# --------------------------------------------------------------------------- #
# invoke() with a transient override
# --------------------------------------------------------------------------- #
def test_override_builds_transient_client_and_returns(monkeypatch):
    created = {}

    def _fake_client(api_key, model=None):
        created["key"] = api_key
        created["model"] = model
        return _FakeClient()

    monkeypatch.setattr(llm, "_transient_openai_client", _fake_client)
    out = llm.invoke(
        "hi", override={"provider": "openai", "api_key": "sk-test-123", "model": "gpt-x"}
    )
    assert out == "String-based"
    assert created["key"] == "sk-test-123"
    assert created["model"] == "gpt-x"


def test_override_key_never_logged_or_stored(monkeypatch):
    """The key must not reach the module-level cache or logger."""
    seen_logs = []

    class _Handler:
        def emit(self, record):
            seen_logs.append(record.getMessage())

    import logging

    handler = _Handler()
    llm.logger.addHandler(handler)
    try:
        monkeypatch.setattr(llm, "_transient_openai_client", lambda k, model=None: _FakeClient())
        llm.invoke("hi", override={"api_key": "sk-super-secret-xyz"})
    finally:
        llm.logger.removeHandler(handler)

    assert not any("sk-super-secret" in m for m in seen_logs)
    # The shared cached client is untouched (still None with no server key).
    assert llm._client is None


def test_override_with_unsupported_provider_is_refused():
    with pytest.raises(llm.LLMUnavailableError) as exc:
        llm.invoke("hi", override={"provider": "anthropic", "api_key": "sk-x"})
    assert "not supported" in str(exc.value)


def test_budget_enforced_across_calls(monkeypatch):
    monkeypatch.setattr(llm, "_transient_openai_client", lambda k, model=None: _FakeClient())
    budget = {"remaining": 2}
    # First two calls pass the budget check (invoke succeeds via fake client).
    for _ in range(2):
        assert llm.invoke(
            "hi", override={"api_key": "sk-x"}, budget=budget
        ) == "String-based"
    assert budget["remaining"] == 0
    with pytest.raises(llm.LLMUnavailableError) as exc:
        llm.invoke("hi", override={"api_key": "sk-x"}, budget=budget)
    assert "limit" in str(exc.value).lower()


def test_budget_defaults_to_settings():
    budget = {"remaining": settings.llm_max_calls_per_request}
    for _ in range(settings.llm_max_calls_per_request):
        llm._spend_budget(budget)
    assert budget["remaining"] == 0
    with pytest.raises(llm.LLMUnavailableError):
        llm._spend_budget(budget)


# --------------------------------------------------------------------------- #
# Provider status must never leak secrets
# --------------------------------------------------------------------------- #
def test_provider_status_has_no_secrets(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-super-secret-server")
    status = llm.provider_status()
    text = str(status)
    assert "sk-super-secret-server" not in text
    assert status["provider"] == "openai"
    assert status["personal_key_supported"] is True
    assert status["deterministic_always_available"] is True


def test_ollama_status_reports_unreachable(monkeypatch):
    import httpx

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    status = llm.ollama_status()
    assert status["reachable"] is False
    assert status["model_installed"] is False
    assert status["error"]


def test_pull_ollama_model_validates_name():
    with pytest.raises(RuntimeError, match="[Ii]nvalid model name"):
        llm.pull_ollama_model("bad name/with spaces")


def test_pull_requires_auto_download_flag_or_explicit(monkeypatch):
    import httpx

    calls = []

    class _Resp:
        status_code = 200
        text = "{}"

        def json(self):
            return {}

    def _post(url, json=None, timeout=None):
        calls.append((url, json))
        return _Resp()

    monkeypatch.setattr(httpx, "post", _post)
    out = llm.pull_ollama_model("llama3.2", timeout=5)
    assert out["downloaded"] is True
    assert calls and calls[0][1]["name"] == "llama3.2"
    assert len(calls) == 1


def test_llm_status_endpoint(client, auth_headers):
    resp = client.get("/llm/status/", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] in ("openai", "ollama", "none")
    assert body["personal_key_supported"] is True
    assert "api_key" not in resp.text


def test_llm_status_requires_auth(client):
    assert client.get("/llm/status/").status_code == 401


def test_ollama_pull_requires_confirm(client, auth_headers):
    resp = client.post(
        "/llm/ollama/pull/", headers=auth_headers, json={"confirm": False}
    )
    assert resp.status_code == 400, resp.text
    assert "confirm" in resp.json()["detail"].lower()


def test_ollama_pull_surfaces_server_error(client, auth_headers, monkeypatch):
    def _boom(model=None, timeout=900.0):
        raise RuntimeError("Could not reach the Ollama server at http://x")

    monkeypatch.setattr(llm, "pull_ollama_model", _boom)
    resp = client.post(
        "/llm/ollama/pull/", headers=auth_headers, json={"confirm": True}
    )
    assert resp.status_code == 502, resp.text
    assert "Ollama" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Per-request personal keys: forwarded once, never stored or echoed
# --------------------------------------------------------------------------- #
def test_classify_sends_override_and_stores_nothing(client, auth_headers, monkeypatch):
    captured = {}

    def _fake_invoke(prompt, max_tokens=400, override=None, budget=None):
        captured["override"] = override
        return "String-based"

    monkeypatch.setattr(llm, "invoke", _fake_invoke)
    resp = client.post(
        "/classify/",
        headers=auth_headers,
        json={
            "column": "x",
            "examples": [
                {"input": "foo", "output": "bar"},
                {"input": "baz", "output": "qux"},
            ],
            "llm_config": {"provider": "openai", "api_key": "sk-ephemeral-1"},
        },
    )
    assert resp.status_code == 200, resp.text
    # The key must never be echoed back to the client.
    assert "sk-ephemeral-1" not in resp.text
    # ...but it WAS forwarded to the provider for this single request.
    assert captured.get("override")
    assert captured["override"]["api_key"] == "sk-ephemeral-1"


def test_generate_with_personal_key_no_persistence(client, auth_headers, monkeypatch):
    resp = client.post(
        "/generate_function/",
        headers=auth_headers,
        json={
            "column": "x",
            "examples": [{"input": "foo", "output": "FOO"}],
            "classification": "String-based",
            "llm_config": {"api_key": "sk-ephemeral-2"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The key must never be echoed in the response.
    assert "sk-ephemeral-2" not in resp.text
    assert "def transform" in body["code"]
    assert body["used_llm"] is False


# --------------------------------------------------------------------------- #
# Provider failure classification: exhausted quota is NOT "retry shortly"
# --------------------------------------------------------------------------- #
_QUOTA_BODY = (
    "Error code: 429 - {'error': {'message': 'You have no credits remaining. "
    "Add credits to continue using the API at "
    "https://platform.openai.com/settings/organization/billing/.', "
    "'type': 'insufficient_quota', 'param': None, "
    "'code': 'credit_balance_exhausted'}}"
)


def test_failure_description_flags_exhausted_quota_as_permanent():
    desc = llm.failure_description(RuntimeError(_QUOTA_BODY))
    assert "no credits remaining" in desc
    assert "platform.openai.com" in desc
    assert "retrying will not help" in desc
    assert "usually transient" not in desc


def test_failure_description_treats_rate_limit_as_transient():
    desc = llm.failure_description(RuntimeError("Error code: 429 - rate limit reached"))
    assert "429" in desc
    assert "usually transient" in desc


def test_failure_description_generic_for_unknown_errors():
    desc = llm.failure_description(RuntimeError("connection reset by peer"))
    assert "provider request failed" in desc


def test_invoke_surfaces_quota_error_from_openai(monkeypatch):
    """The provider's real reason (exhausted credits) must reach the caller."""

    class _QuotaClient(_FakeClient):
        def invoke(self, prompt, config=None):
            raise RuntimeError(_QUOTA_BODY)

    monkeypatch.setattr(
        llm, "_transient_openai_client", lambda k, model=None: _QuotaClient()
    )
    with pytest.raises(RuntimeError) as exc:
        llm.invoke("hi", override={"api_key": "sk-x"})
    assert "no credits remaining" in str(exc.value)
    assert "platform.openai.com" in str(exc.value)


# --------------------------------------------------------------------------- #
# _invoke_ollama transient-failure retry (Ollama crashed/reloading -> 5xx)
# --------------------------------------------------------------------------- #
class _OllamaResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = (
            payload if payload is not None else {"message": {"content": "ok"}}
        )
        self.text = text or "{}"

    def json(self):
        return self._payload


def _no_sleep(monkeypatch):
    """Replace time.sleep with a recorder so retry tests run instantly."""
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return sleeps


def _post_script(monkeypatch, script):
    """Serve scripted responses/exceptions from httpx.post, in order."""
    import httpx

    calls = []

    def _post(url, json=None, timeout=None):
        calls.append(url)
        item = script[min(len(calls) - 1, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(httpx, "post", _post)
    return calls


def _attempts():
    return max(1, int(settings.llm_max_retries) + 1)


def test_ollama_retries_5xx_then_succeeds(monkeypatch):
    """A transient 500 (crashed llama-server) is retried, not fatal."""
    sleeps = _no_sleep(monkeypatch)
    calls = _post_script(
        monkeypatch,
        [
            _OllamaResp(status_code=500, text="llama-server has terminated"),
            _OllamaResp(status_code=200),
        ],
    )
    out = llm._invoke_ollama("hi", max_tokens=10)
    assert out == "ok"
    assert len(calls) == 2
    assert sleeps == [2.0]


def test_ollama_5xx_exhausts_retry_budget(monkeypatch):
    """When the 5xx persists, the bounded budget stops and the error surfaces."""
    sleeps = _no_sleep(monkeypatch)
    calls = _post_script(
        monkeypatch, [_OllamaResp(status_code=500, text="boom")]
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        llm._invoke_ollama("hi", max_tokens=10)
    assert len(calls) == _attempts()
    # Backoff grows: 2s, 4s, ... (capped), never zero, never unbounded.
    assert len(sleeps) == _attempts() - 1
    assert sleeps[0] == 2.0
    assert all(s > 0 for s in sleeps)


def test_ollama_404_missing_model_is_not_retried(monkeypatch):
    """404 = model not pulled; retrying cannot fix it, so fail immediately."""
    sleeps = _no_sleep(monkeypatch)
    calls = _post_script(monkeypatch, [_OllamaResp(status_code=404)])
    with pytest.raises(RuntimeError, match="not available"):
        llm._invoke_ollama("hi", max_tokens=10)
    assert len(calls) == 1
    assert sleeps == []


def test_ollama_client_4xx_is_not_retried(monkeypatch):
    """A 4xx request error is the caller's fault, not a transient blip."""
    sleeps = _no_sleep(monkeypatch)
    calls = _post_script(monkeypatch, [_OllamaResp(status_code=400, text="bad")])
    with pytest.raises(RuntimeError, match="HTTP 400"):
        llm._invoke_ollama("hi", max_tokens=10)
    assert len(calls) == 1
    assert sleeps == []


def test_ollama_connection_error_retried_then_succeeds(monkeypatch):
    """Server restarting (connection refused) is retried like a 5xx."""
    import httpx

    sleeps = _no_sleep(monkeypatch)
    calls = _post_script(
        monkeypatch,
        [
            httpx.ConnectError("connection refused"),
            _OllamaResp(status_code=200),
        ],
    )
    out = llm._invoke_ollama("hi", max_tokens=10)
    assert out == "ok"
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_ollama_connection_error_exhausts_budget(monkeypatch):
    """Unreachable server keeps the actionable 'Could not reach' message."""
    import httpx

    sleeps = _no_sleep(monkeypatch)
    calls = _post_script(
        monkeypatch, [httpx.ConnectError("connection refused")]
    )
    with pytest.raises(RuntimeError, match="Could not reach the Ollama server"):
        llm._invoke_ollama("hi", max_tokens=10)
    assert len(calls) == _attempts()
    assert len(sleeps) == _attempts() - 1
