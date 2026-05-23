"""Unified provider layer: cloud-primary with local Ollama fallback (MOCKED).

Verifies spec §9 without any real network call: ``_invoke_ollama`` and the
OpenAI client are monkeypatched.  The rules under test:

* cloud works            -> cloud serves, never falls back;
* cloud quota exhausted  -> falls back to Ollama (when ENABLE_LOCAL_FALLBACK),
                            and records Ollama as the serving provider;
* cloud 5xx / network    -> falls back to Ollama;
* invalid credentials    -> NO fallback (a config problem must be surfaced);
* Ollama also down       -> an informative combined error;
* fallback disabled      -> cloud error surfaces as-is;
* exhausted quota        -> cloud enters a cooldown so the next call skips it.
"""

from __future__ import annotations

import pytest

import llm
from config import settings


class _Msg:
    def __init__(self, content):
        self.content = content


class _CloudClient:
    """Configurable stand-in for the ChatOpenAI client."""

    def __init__(self, exc=None, content="cloud-says-hi"):
        self.exc = exc
        self.content = content
        self.calls = 0

    def invoke(self, prompt, config=None):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return _Msg(self.content)


QUOTA_EXC = RuntimeError(
    "Error code: 429 - {'error': {'message': 'You have no credits remaining. "
    "Add credits to continue using the API at https://platform.openai.com/"
    "settings/organization/billing/.', 'type': 'insufficient_quota', "
    "'code': 'credit_balance_exhausted'}}"
)
RATE_LIMIT_EXC = RuntimeError("Error code: 429 - rate limit reached")
SERVICE_DOWN_EXC = RuntimeError("503 Service Unavailable")
NETWORK_EXC = RuntimeError("connection timed out")
AUTH_EXC = RuntimeError("401 Incorrect API key provided")


@pytest.fixture()
def cloud(monkeypatch):
    """Configure a working OpenAI provider; return a handle to swap behaviour."""
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-server")
    monkeypatch.setattr(settings, "enable_local_fallback", True)
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434")
    llm._reset_cloud_cooldown()

    handle = {"client": _CloudClient(), "ollama": ["local-says-hi"], "ollama_calls": 0}

    monkeypatch.setattr(llm, "_get_client", lambda: handle["client"])

    def _fake_ollama(prompt, max_tokens):
        handle["ollama_calls"] += 1
        if isinstance(handle["ollama"], Exception):
            raise handle["ollama"]
        return handle["ollama"][-1]

    monkeypatch.setattr(llm, "_invoke_ollama", _fake_ollama)
    return handle


@pytest.fixture()
def fallback_off(cloud, monkeypatch):
    monkeypatch.setattr(settings, "enable_local_fallback", False)
    return cloud


class TestCloudPrimary:
    def test_cloud_success_never_falls_back(self, cloud):
        result = llm.invoke_detailed("hi")
        assert result.text == "cloud-says-hi"
        assert result.provider == "openai"
        assert cloud["ollama_calls"] == 0
        assert llm.last_provider() == "openai"


class TestQuotaFallback:
    def test_exhausted_quota_falls_back_to_ollama(self, cloud):
        cloud["client"] = _CloudClient(exc=QUOTA_EXC)
        result = llm.invoke_detailed("hi")
        assert result.text == "local-says-hi"
        assert result.provider == "ollama"
        assert cloud["ollama_calls"] == 1
        assert llm.last_provider() == "ollama"

    def test_quota_fallback_marks_cloud_degraded(self, cloud):
        cloud["client"] = _CloudClient(exc=QUOTA_EXC)
        llm.invoke_detailed("hi")
        assert llm._cloud_in_cooldown() is True

    def test_cooldown_skips_cloud_on_next_call(self, cloud):
        failing = _CloudClient(exc=QUOTA_EXC)
        cloud["client"] = failing
        llm.invoke_detailed("hi")                 # falls back, marks degraded
        calls_after_first = failing.calls
        result = llm.invoke_detailed("hi again")  # cooldown -> straight to Ollama
        assert result.provider == "ollama"
        assert failing.calls == calls_after_first  # cloud was NOT retried


class TestTransientFallback:
    @pytest.mark.parametrize("exc", [RATE_LIMIT_EXC, SERVICE_DOWN_EXC, NETWORK_EXC])
    def test_transient_cloud_errors_fall_back(self, cloud, exc):
        cloud["client"] = _CloudClient(exc=exc)
        result = llm.invoke_detailed("hi")
        assert result.provider == "ollama"
        assert result.text == "local-says-hi"


class TestNoFallbackOnAuth:
    def test_invalid_credentials_do_not_fall_back(self, cloud):
        cloud["client"] = _CloudClient(exc=AUTH_EXC)
        with pytest.raises(RuntimeError) as exc:
            llm.invoke_detailed("hi")
        assert cloud["ollama_calls"] == 0          # never switched
        assert "credits" in str(exc.value) or "provider" in str(exc.value)

    def test_fallback_disabled_surfaces_cloud_error(self, fallback_off):
        fallback_off["client"] = _CloudClient(exc=QUOTA_EXC)
        with pytest.raises(RuntimeError) as exc:
            llm.invoke_detailed("hi")
        assert fallback_off["ollama_calls"] == 0
        assert "no credits remaining" in str(exc.value)


class TestOllamaDown:
    def test_both_down_yields_informative_error(self, cloud):
        cloud["client"] = _CloudClient(exc=QUOTA_EXC)
        cloud["ollama"] = RuntimeError("Could not reach the Ollama server at x")
        with pytest.raises(RuntimeError) as exc:
            llm.invoke_detailed("hi")
        msg = str(exc.value)
        assert "no credits remaining" in msg        # cloud reason kept
        assert "Ollama fallback also failed" in msg  # and the fallback's


class TestProviderStatus:
    def test_status_exposes_fallback_without_secrets(self, cloud, monkeypatch):
        monkeypatch.setattr(settings, "openai_api_key", "sk-super-secret")
        status = llm.provider_status()
        assert status["local_fallback_enabled"] is True
        assert "sk-super-secret" not in str(status)

    def test_invoke_compat_wrapper_returns_text(self, cloud):
        assert llm.invoke("hi") == "cloud-says-hi"
