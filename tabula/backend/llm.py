"""Optional LLM client.

The application must start and serve deterministic transformations with **no**
LLM configured.  This module therefore:

* lazily constructs the client only when a provider is enabled,
* supports two providers: hosted OpenAI (``llm_provider="openai"``) and a
  local Ollama server (``llm_provider="ollama"``),
* exposes :func:`llm_available` so callers can give actionable errors,
* enforces a timeout and a small retry budget (OpenAI path),
* never logs keys or full prompts containing user data.

OpenAI API calls can incur usage costs on your account.  Ollama runs locally
and is free; see the README for how to enable it via docker compose profiles.
"""

from __future__ import annotations

import contextvars
import logging
import time
from dataclasses import dataclass
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_client = None
_client_initialised = False

# Which provider actually served the most recent invoke() in this task.
# contextvars keeps concurrent requests from mis-attributing each other, and
# asyncio.to_thread (used for model-backed apply) copies the context.
_provider_used: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "llm_provider_used", default=""
)

# When the cloud provider reports an EXHAUSTED QUOTA, retrying cannot help
# until credits are added, so the cloud provider is skipped in favour of the
# local fallback for this many seconds (monotonic clock).
_cloud_degraded_until: float = 0.0


class LLMUnavailableError(RuntimeError):
    """Raised when a transformation needs the LLM but none is configured."""


def llm_available() -> bool:
    """True if a usable LLM provider is configured."""
    return settings.llm_enabled


def _provider() -> str:
    return (settings.llm_provider or "openai").strip().lower()


def failure_description(exc: Exception) -> str:
    """Actionable, human-readable description of a provider failure.

    Distinguishes problems that **retrying cannot fix** (an exhausted
    credit balance / quota) from transient ones (per-minute rate limits,
    network blips).  Suggesting "retry shortly" to someone whose account
    is out of credits wastes their time, so that case says so explicitly.
    Never includes keys; provider error bodies may include the org id,
    which is not sensitive.
    """
    text = str(exc).lower()
    if any(
        marker in text
        for marker in (
            "insufficient_quota",
            "credit_balance_exhausted",
            "no credits remaining",
            "exceeded your current quota",
            "out of credits",
        )
    ):
        return (
            "the provider account has no credits remaining "
            "(insufficient_quota); add credits at "
            "https://platform.openai.com/settings/organization/billing/, "
            "use a personal API key in AI settings, or configure Ollama "
            "(LLM_PROVIDER=ollama) — retrying will not help until the "
            "account is topped up"
        )
    if "rate limit" in text or "429" in text or "too many requests" in text:
        return (
            "the provider rate-limited the request (HTTP 429); this is "
            "usually transient — wait a moment and retry, or reduce how "
            "often you trigger model-backed transformations"
        )
    return (
        "the provider request failed (network, authentication, or an "
        "unusable response); check the provider configuration (API key, "
        "credits, or Ollama server) and retry"
    )


# --------------------------------------------------------------------------- #
# Failure classification for the cloud -> local fallback decision
# --------------------------------------------------------------------------- #
_QUOTA_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "no credits remaining",
    "exceeded your current quota",
    "out of credits",
)

# Errors a different provider could plausibly avoid (so the local fallback
# is worth attempting).  Auth/credential problems are deliberately absent:
# serving a different model would HIDE a misconfiguration.
_RETRYABLE_MARKERS = (
    "rate limit",
    "429",
    "too many requests",
    "overloaded",
    "500",
    "502",
    "503",
    "timeout",
    "timed out",
    "connection",
    "unreachable",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
)

_AUTH_MARKERS = (
    "invalid api key",
    "incorrect api key",
    "invalid_api_key",
    "invalid authentication",
    "authentication_error",
    "unauthorized",
    "401",
    "403",
    "permission_denied",
)


def _is_quota_exhausted(exc: BaseException) -> bool:
    """True when the provider says the account has no credits left."""
    text = str(exc).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def _is_auth_failure(exc: BaseException) -> bool:
    """True for bad credentials -- never a fallback candidate."""
    text = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    return any(marker in text for marker in _AUTH_MARKERS)


def _is_retryable_cloud_failure(exc: BaseException) -> bool:
    """True when falling back to another provider could actually help."""
    if _is_auth_failure(exc):
        return False
    text = str(exc).lower()
    return _is_quota_exhausted(exc) or any(m in text for m in _RETRYABLE_MARKERS)


def _mark_cloud_degraded() -> None:
    global _cloud_degraded_until
    _cloud_degraded_until = time.monotonic() + max(
        0.0, settings.cloud_cooldown_seconds
    )
    logger.warning(
        "Cloud provider reported exhausted quota; skipping it for %.0f s "
        "and using the local fallback (if enabled) instead.",
        settings.cloud_cooldown_seconds,
    )


def _cloud_in_cooldown() -> bool:
    return time.monotonic() < _cloud_degraded_until


def _reset_cloud_cooldown() -> None:
    """Test/admin hook: allow the cloud provider to be tried again now."""
    global _cloud_degraded_until
    _cloud_degraded_until = 0.0


def _ollama_configured() -> bool:
    return bool(settings.ollama_base_url.strip())


def last_provider() -> str:
    """Provider ("openai" / "ollama" / "personal-openai") of the last invoke.

    Empty string when no invoke happened in this task yet.  Callers read it
    right after :func:`invoke` / :func:`invoke_detailed` to report which
    provider actually served the request (spec: never switch silently).
    """
    return _provider_used.get()


@dataclass(frozen=True)
class LLMResult:
    """Text response plus which provider produced it."""

    text: str
    provider: str


def _get_client():
    """Lazily build the ChatOpenAI client (OpenAI provider only).

    Returns None when unconfigured or when the provider is not OpenAI
    (Ollama is called via plain HTTP in :func:`_invoke_ollama`).
    """
    global _client, _client_initialised
    if _client_initialised:
        return _client
    _client_initialised = True

    if _provider() != "openai" or not settings.openai_api_key.strip():
        logger.info(
            "OpenAI client disabled (provider=%s, key set=%s).",
            _provider(),
            bool(settings.openai_api_key.strip()),
        )
        _client = None
        return _client

    try:
        from langchain_openai import ChatOpenAI

        _client = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=0,
        )
        logger.info("OpenAI LLM client initialised (model=%s).", settings.llm_model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to initialise OpenAI LLM client: %s", exc)
        _client = None
    return _client


def _invoke_ollama(prompt: str, max_tokens: int) -> str:
    """Call a local Ollama server via its REST /api/chat endpoint.

    Transient failures are retried with a short exponential backoff using the
    same budget as the OpenAI path (``settings.llm_max_retries``): connection
    errors (server restarting) and 5xx responses (a crashing/reloading
    ``llama-server`` returns 500 while it respawns).  A 404 (model not pulled)
    and other 4xx client errors are NOT retried -- they cannot fix themselves.

    Raises :class:`RuntimeError` with an actionable message when the server is
    unreachable, returns an error status, or the model is missing (not pulled).
    """
    import httpx

    base = settings.ollama_base_url.rstrip("/")
    url = f"{base}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens},
    }
    attempts = max(1, int(settings.llm_max_retries) + 1)
    last_error: Optional[RuntimeError] = None
    last_cause: Optional[BaseException] = None

    for attempt in range(attempts):
        try:
            response = httpx.post(
                url,
                json=payload,
                timeout=settings.llm_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            logger.error(
                "Ollama request failed to reach %s: %s", base, type(exc).__name__
            )
            last_error = RuntimeError(
                f"Could not reach the Ollama server at {base}. "
                "Start it (e.g. `docker compose --profile local-ai up`) and ensure "
                f"the model `{settings.ollama_model}` is pulled, then retry."
            )
            last_cause = exc
        else:
            if response.status_code == 404:
                # Ollama returns 404 when the model is not present locally.
                raise RuntimeError(
                    f"Ollama model `{settings.ollama_model}` is not available. "
                    f"Pull it with `ollama pull {settings.ollama_model}` "
                    "(or into the ollama container), then retry."
                )
            if response.status_code >= 500:
                detail = response.text[:200]
                logger.error(
                    "Ollama returned HTTP %s: %s", response.status_code, detail
                )
                last_error = RuntimeError(
                    f"The Ollama request failed with HTTP {response.status_code}. "
                    "Check the Ollama server logs, then retry."
                )
                last_cause = None
            elif response.status_code >= 400:
                detail = response.text[:200]
                logger.error(
                    "Ollama returned HTTP %s: %s", response.status_code, detail
                )
                raise RuntimeError(
                    f"The Ollama request failed with HTTP {response.status_code}. "
                    "Check the Ollama server logs, then retry."
                )
            else:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        "The Ollama server returned a non-JSON response."
                    ) from exc
                message = (data.get("message") or {}).get("content")
                if message is None:
                    raise RuntimeError(
                        "The Ollama response did not contain a message payload."
                    )
                return str(message)

        # Transient failure: back off (2s, 4s, ... capped) and retry while the
        # budget lasts.  Ollama respawns a crashed llama-server on the next
        # request, so a short wait is usually enough.
        if attempt < attempts - 1:
            delay = min(2.0 * (2 ** attempt), 8.0)
            logger.warning(
                "Ollama transient failure; retrying in %.1fs (attempt %d/%d).",
                delay,
                attempt + 2,
                attempts,
            )
            time.sleep(delay)

    if last_error is None:  # pragma: no cover - loop always sets it on exit
        raise RuntimeError("The Ollama request failed.")
    raise last_error from last_cause


# --------------------------------------------------------------------------- #
# Per-request personal provider (never persisted)
# --------------------------------------------------------------------------- #
def _transient_openai_client(api_key: str, model: Optional[str] = None):
    """Build a short-lived OpenAI client for ONE request.

    The key lives only in this stack frame's client object; the client is
    never cached, never logged, and discarded when the request ends.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=(model or settings.llm_model).strip() or settings.llm_model,
        api_key=api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,  # personal keys: fail fast, no silent retry spend
        temperature=0,
    )


def _budget_exhausted() -> LLMUnavailableError:
    return LLMUnavailableError(
        f"This request reached the LLM call limit "
        f"({settings.llm_max_calls_per_request}). Provide more/better examples "
        "so a deterministic rule can be used, or retry as a new request."
    )


def _spend_budget(budget: Optional[dict]) -> None:
    if budget is None:
        return
    remaining = int(budget.get("remaining", settings.llm_max_calls_per_request))
    if remaining <= 0:
        raise _budget_exhausted()
    budget["remaining"] = remaining - 1


def _try_local_fallback(prompt: str, max_tokens: int, *, reason: str) -> str:
    """One attempt on local Ollama; records the provider and logs the switch.

    Never silent: the log line and the ``last_provider()`` value both say the
    local model answered instead of the configured cloud provider.
    """
    text = _invoke_ollama(prompt, max_tokens)
    _provider_used.set("ollama")
    logger.warning(
        "LLM request served by the LOCAL OLLAMA FALLBACK after the cloud "
        "provider failed (%s).",
        reason,
    )
    return text


def invoke_detailed(
    prompt: str,
    max_tokens: int = 400,
    override: Optional[dict] = None,
    budget: Optional[dict] = None,
) -> LLMResult:
    """Call an LLM and return ``(text, provider)``.

    Provider selection lives ONLY here (spec §9):

    * the configured provider is tried first (cloud OpenAI or local Ollama);
    * when ``ENABLE_LOCAL_FALLBACK`` is on and the cloud provider fails with
      a RETRYABLE error (exhausted quota, rate limit, 5xx, network), the
      request is retried once on local Ollama;
    * invalid credentials never fall back -- that is a configuration problem
      that must be reported, not hidden by silently switching models;
    * an exhausted-quota cloud provider is then skipped for
      ``CLOUD_COOLDOWN_SECONDS`` (retrying cannot help until credits are
      added), so later calls go straight to the local model;
    * every switch is logged and visible through :func:`last_provider`.

    Raises :class:`LLMUnavailableError` when no provider is configured, and
    re-raises a clear RuntimeError on provider failures.
    """
    _spend_budget(budget)
    override = override or {}
    override_key = str(override.get("api_key") or "").strip()
    override_provider = str(override.get("provider") or "").strip().lower()
    fallback_ok = bool(settings.enable_local_fallback) and _ollama_configured()

    if override_key:
        if override_provider and override_provider != "openai":
            raise LLMUnavailableError(
                f"Personal provider '{override_provider}' is not supported; "
                "only a personal OpenAI API key is supported."
            )
        # HTTPS is enforced by the OpenAI SDK (api.openai.com only; no custom
        # base URL is accepted for personal keys).
        client = _transient_openai_client(override_key, override.get("model"))
        try:
            message = client.invoke(prompt, config={"max_tokens": max_tokens})
            _provider_used.set("personal-openai")
            return LLMResult(
                str(getattr(message, "content", message)), "personal-openai"
            )
        except LLMUnavailableError:
            raise
        except Exception as exc:
            detail = failure_description(exc)
            logger.error(
                "Personal-key LLM request failed (%s): %s",
                type(exc).__name__,
                str(exc)[:300],
            )
            if fallback_ok and _is_retryable_cloud_failure(exc):
                try:
                    return LLMResult(
                        _try_local_fallback(
                            prompt, max_tokens,
                            reason=f"personal key: {type(exc).__name__}",
                        ),
                        "ollama",
                    )
                except Exception as ollama_exc:
                    raise RuntimeError(
                        detail
                        + " The local Ollama fallback also failed: "
                        + str(ollama_exc)
                    ) from exc
            raise RuntimeError(detail) from exc

    if not settings.llm_enabled:
        raise LLMUnavailableError(
            "This transformation needs an LLM but no provider is configured. "
            "Set OPENAI_API_KEY (OpenAI), configure your own personal API key "
            "in the AI settings, or configure Ollama "
            "(LLM_PROVIDER=ollama with a reachable OLLAMA_BASE_URL), "
            "or use a deterministic (numerical or template-based) "
            "transformation instead."
        )

    provider = _provider()
    if provider == "ollama":
        text = _invoke_ollama(prompt, max_tokens)
        _provider_used.set("ollama")
        return LLMResult(text, "ollama")

    # ---- Cloud (OpenAI) primary, with the optional local fallback --------- #
    if not (fallback_ok and _cloud_in_cooldown()):
        client = _get_client()
        if client is None:
            raise LLMUnavailableError(
                "This transformation needs an LLM but no OPENAI_API_KEY is "
                "configured. Set OPENAI_API_KEY, add your own personal API "
                "key in the AI settings, or use a deterministic (numerical "
                "or template-based) transformation instead."
            )
        try:
            message = client.invoke(prompt, config={"max_tokens": max_tokens})
            # ChatOpenAI returns an AIMessage; .content holds the text.
            content = getattr(message, "content", message)
            _provider_used.set("openai")
            return LLMResult(str(content), "openai")
        except LLMUnavailableError:
            raise
        except Exception as exc:
            detail = failure_description(exc)
            logger.error(
                "LLM request failed (%s): %s", type(exc).__name__, str(exc)[:300]
            )
            if _is_quota_exhausted(exc):
                _mark_cloud_degraded()
            if fallback_ok and _is_retryable_cloud_failure(exc):
                try:
                    return LLMResult(
                        _try_local_fallback(
                            prompt, max_tokens, reason=type(exc).__name__
                        ),
                        "ollama",
                    )
                except Exception as ollama_exc:
                    raise RuntimeError(
                        detail
                        + " The local Ollama fallback also failed: "
                        + str(ollama_exc)
                    ) from exc
            raise RuntimeError(detail) from exc

    # The cloud provider is inside the exhausted-quota cooldown: skip it and
    # go straight to the local model (this is the whole point of the
    # cooldown -- do not pay the failed cloud attempt on every request).
    logger.warning(
        "Cloud provider skipped (exhausted-quota cooldown); using local Ollama."
    )
    try:
        text = _try_local_fallback(prompt, max_tokens, reason="cloud cooldown")
    except Exception as exc:
        raise RuntimeError(
            "The cloud provider has no credits remaining (cooldown active) "
            "and the local Ollama fallback failed: " + str(exc)
        ) from exc
    return LLMResult(text, "ollama")


def invoke(
    prompt: str,
    max_tokens: int = 400,
    override: Optional[dict] = None,
    budget: Optional[dict] = None,
) -> str:
    """Call an LLM and return the text response.

    ``override`` is a transient, per-request provider configuration::

        {"provider": "openai", "api_key": "sk-...", "model": "gpt-4o-mini"}

    It is used for exactly one request and is never stored anywhere.
    ``budget`` is a mutable ``{"remaining": n}`` dict enforcing
    ``settings.llm_max_calls_per_request`` per user request.

    See :func:`invoke_detailed` for provider selection and fallback rules;
    use it (or :func:`last_provider`) when the caller needs to know which
    provider actually served the request.
    """
    return invoke_detailed(
        prompt, max_tokens=max_tokens, override=override, budget=budget
    ).text


# --------------------------------------------------------------------------- #
# Provider status / local model management
# --------------------------------------------------------------------------- #
def provider_status() -> dict:
    """Non-sensitive status of the configured provider (never leaks keys)."""
    provider = _provider()
    configured = bool(settings.llm_enabled)
    status = {
        "provider": provider,
        "configured": configured,
        "model": settings.llm_model if provider == "openai" else settings.ollama_model,
        "personal_key_supported": True,
        "deterministic_always_available": True,
        "max_calls_per_request": settings.llm_max_calls_per_request,
        # Cloud -> local fallback visibility (spec: never switch silently).
        "local_fallback_enabled": bool(settings.enable_local_fallback),
        "cloud_in_cooldown": _cloud_in_cooldown(),
    }
    if provider == "ollama":
        status["ollama"] = ollama_status()
    elif settings.enable_local_fallback and _ollama_configured():
        # Primary is cloud; report the local fallback's reachability too so
        # the user can see whether the safety net would actually work.
        status["local_fallback"] = ollama_status()
    return status


def ollama_status() -> dict:
    """Reachability + model-installed check for the local Ollama server."""
    import httpx

    base = settings.ollama_base_url.rstrip("/")
    out = {
        "base_url": base,
        "model": settings.ollama_model,
        "reachable": False,
        "model_installed": False,
        "auto_download": bool(settings.ollama_auto_download),
        "error": None,
    }
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=5.0)
        resp.raise_for_status()
        out["reachable"] = True
        names = [m.get("name", "") for m in (resp.json().get("models") or [])]
        out["model_installed"] = any(
            n == settings.ollama_model or n.startswith(f"{settings.ollama_model}:")
            for n in names
        )
        out["installed_models"] = names
    except httpx.HTTPError as exc:
        out["error"] = (
            f"Could not reach the Ollama server at {base} "
            f"({type(exc).__name__}). Start it with "
            "`docker compose --profile local-ai up`."
        )
    except ValueError:
        out["error"] = "The Ollama server returned a non-JSON response."
    return out


def pull_ollama_model(model: Optional[str] = None, timeout: float = 900.0) -> dict:
    """Download a model through the Ollama API (blocking, long timeout).

    The caller must have informed the user about size/hardware requirements
    first — this function does not start silently.
    """
    import httpx

    if not settings.ollama_auto_download and not model:
        # Explicit model name (from a confirmed request) is the normal path;
        # auto-download flag only affects unrequested pulls.
        pass
    target = (model or settings.ollama_model).strip()
    if not target or any(c in target for c in " \t\n/\\"):
        raise RuntimeError("Invalid model name.")
    base = settings.ollama_base_url.rstrip("/")
    try:
        resp = httpx.post(
            f"{base}/api/pull", json={"name": target, "stream": False}, timeout=timeout
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Could not reach the Ollama server at {base} to download "
            f"`{target}` ({type(exc).__name__})."
        ) from exc
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Model download failed (HTTP {resp.status_code}): "
            f"{resp.text[:200]}"
        )
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if data.get("error"):
        raise RuntimeError(f"Model download failed: {data['error']}")
    return {
        "model": target,
        "downloaded": True,
        "detail": data.get("status") or "completed",
    }


def reset_client() -> None:
    """Testing helper: forget the cached client so settings changes take effect."""
    global _client, _client_initialised
    _client = None
    _client_initialised = False
