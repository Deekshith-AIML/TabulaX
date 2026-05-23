"""Application configuration loaded from environment variables.

All secrets come from the environment (or a local ``.env`` file that is never
committed).  Nothing in this module hardcodes a credential or an API key.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _detect_docker() -> bool:
    """True when running inside a Linux container (auto-detected).

    Docker writes ``/.dockerenv``; Podman/other engines set ``container``.
    Explicitly set ``IN_DOCKER`` in the environment overrides this default.
    """
    return os.path.exists("/.dockerenv") or os.environ.get("container", "") in (
        "docker",
        "podman",
    )


class Settings(BaseSettings):
    """Validated application settings.

    Every field is populated from an environment variable of the same name
    (case-insensitive).  ``.env`` is loaded if present so local development is
    convenient, but production deployments should provide real env vars.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Core ---
    app_name: str = "TabulaX"
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # --- Security ---
    # If SECRET_KEY is unset we generate an ephemeral one so the app can still
    # boot in development.  A random key means existing tokens are invalidated
    # on restart, which is acceptable for a dev default and never used in
    # production (where SECRET_KEY is mandatory).
    secret_key: str = Field(default="")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    # Minimum password length enforced at registration.
    min_password_length: int = 8

    # --- MongoDB (user accounts, sessions, datasets) ---
    mongodb_url: str = Field(default="mongodb://localhost:27017")
    database_name: str = Field(default="tabula")

    # --- CORS ---
    # Comma-separated list of allowed origins, e.g.
    # "http://localhost:5173,https://tabulax.vercel.app"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- LLM (optional) ---
    # Provider selection: "openai" (default), "ollama" (local), or "none"
    # (force-disable even when keys/endpoints are present).
    llm_provider: str = Field(default="openai")
    # OpenAI (hosted). Leave OPENAI_API_KEY empty to disable OpenAI-backed
    # transformations when provider is "openai".
    openai_api_key: str = Field(default="")
    # Ollama (local). Used when llm_provider == "ollama". No API key needed;
    # the service must be reachable at ollama_base_url with ollama_model pulled.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    # When True, /llm/ollama/pull/ may run without an extra confirmation
    # (still user-triggered; models are large — see README hardware notes).
    ollama_auto_download: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 2
    # Upper bound on LLM calls per request to avoid runaway spend.
    llm_max_calls_per_request: int = 3
    # Cloud-primary with optional local fallback (see llm.py).  When the
    # configured cloud provider fails with a RETRYABLE error (exhausted
    # quota, rate limit, 5xx, network) the request is retried once on the
    # local Ollama server.  Invalid credentials never fall back -- that is a
    # configuration problem the user must fix, and silently serving a
    # different model would hide it.
    enable_local_fallback: bool = False
    # After an exhausted-quota response the cloud provider is skipped for
    # this many seconds: retrying cannot help until credits are added, so we
    # go straight to the local fallback instead of burning latency each call.
    cloud_cooldown_seconds: float = 900.0
    # Rows per model-backed batch request (structured JSON responses).
    llm_batch_size: int = 25

    # --- Data / session limits ---
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_rows_per_dataset: int = 200_000
    session_ttl_minutes: int = 60 * 12  # datasets expire after 12 hours
    preview_rows: int = 100
    # Backups of source rows/columns/documents taken before a permanent
    # delete (see backups.py).  Stored in the same MongoDB (free) with a TTL
    # index; expiration and restore steps are documented in README.md.
    backup_ttl_days: int = 7

    # --- Sandbox ---
    sandbox_timeout_seconds: float = 15.0
    sandbox_memory_limit_mb: int = 512

    # --- Runtime ---
    # True when the backend runs in a container.  User-supplied "localhost"
    # then refers to the container itself, so loopback hosts supplied through
    # the UI are mapped to host.docker.internal (see db_mysql / app).
    in_docker: bool = Field(default_factory=_detect_docker)

    # --- Derived ---
    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        """True when an LLM provider is usable.

        * ``openai``  – enabled iff ``OPENAI_API_KEY`` is set.
        * ``ollama``  – enabled iff ``ollama_base_url`` is set (always is by
          default); the service may still be unreachable at call time.
        * ``none``    – always disabled.
        """
        provider = (self.llm_provider or "").strip().lower()
        if provider == "none":
            return False
        if provider == "ollama":
            return bool(self.ollama_base_url.strip())
        return bool(self.openai_api_key.strip())

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("llm_provider")
    @classmethod
    def _check_provider(cls, v: str) -> str:
        allowed = {"openai", "ollama", "none"}
        if v not in allowed:
            raise ValueError(
                f"LLM_PROVIDER must be one of {sorted(allowed)}, got {v!r}."
            )
        return v

    @field_validator("secret_key", mode="before")
    @classmethod
    def _default_secret(cls, v: object) -> object:
        # An explicitly empty secret_key falls back to an ephemeral random key.
        if v is None or (isinstance(v, str) and not v.strip()):
            return secrets.token_urlsafe(32)
        return v


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


settings = get_settings()
